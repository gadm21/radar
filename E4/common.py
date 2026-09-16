"""Shared utilities for the E4 combined activity+position pipeline.

E4 provides a single inference function `predict_capture(npz_path)` that
maps a `capture.npz` recording to a combined label of the form
"empty" | "sleep(left)" | "sleep(right)" | "present(left)" | "present(right)".

Two independently trained models are composed:
  * Activity model (3-class empty/sleep/present): trained on t1+t2
    (`minutes/`), tested on t (`test_minutes/`). Uses E2's cached
    range-Doppler + range-azimuth maps (50-frame windows, 24x24).
  * Position model (binary left/right): trained and evaluated on t only
    (`test_minutes/` is the only source with left/right labels). Uses
    E3's cached maps (30-frame windows, 32x32).

This module holds the shared constants, label parsing, and the raw
capture.npz -> radar-frame decoder used by `predict.py`.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MINUTES_DIR = ROOT / "minutes"
TEST_MINUTES_DIR = ROOT / "test_minutes"
E2_CACHE = ROOT / "E2" / "cache"
E3_CACHE = ROOT / "E3" / "cache"
E4_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = E4_DIR / "outputs"
MODELS_DIR = E4_DIR / "saved_models"

FRAME_HEADER_BYTES = 12
NUM_CHIRPS = 64
NUM_SAMPLES = 128
NUM_ANTENNAS = 3
EXPECTED_PAYLOAD = NUM_CHIRPS * NUM_SAMPLES * NUM_ANTENNAS * 3 // 2

ACTIVITY_NAMES = ["empty", "sleep", "present"]
POSITION_NAMES = ["left", "right"]

# E2 preprocessing spec (activity model input)
E2_WINDOW_FRAMES = 50
E2_MAP_SIZE = 24
E2_AZ_BINS = 16
E2_RANGE_BINS = 64

# E3 preprocessing spec (position model input)
E3_WINDOW_FRAMES = 30
E3_MAP_SIZE = 32
E3_AZ_BINS = 32
E3_RANGE_BINS = 64


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------
_PLACEMENTS = ("t1", "t2", "t")
_ACTIVITIES = {"empty": 0, "sleep": 1, "present": 2}
_POSITIONS = {"left": 0, "right": 1}


def parse_labels(labels):
    """Parse a manifest `labels` list -> (placement, activity, position).

    Handles both `minutes/` style labels ("t1_empty" -> placement=t1,
    activity=0) and `test_minutes/` style labels ("t_sleep" + "left" ->
    placement=t, activity=1, position=0). Returns None for any field not
    present in the labels.
    """
    placement = activity = position = None
    for l in labels or []:
        if l in _POSITIONS:
            position = _POSITIONS[l]
            continue
        for p in _PLACEMENTS:
            if l.startswith(p + "_"):
                placement = p
                act = l[len(p) + 1:]
                if act in _ACTIVITIES:
                    activity = _ACTIVITIES[act]
                break
    return placement, activity, position


# ---------------------------------------------------------------------------
# Manifest parsing (tolerant of malformed JSON, same as E2/E3)
# ---------------------------------------------------------------------------
def load_manifest(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text), "json"
    except Exception:
        pass
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        return obj, "json_partial"
    except Exception:
        return {}, "unparsable"


def scan_all_labels():
    """Scan every manifest in minutes/ and test_minutes/; return
    {rec_id: {"placement", "activity", "position", "source", "folder"}}."""
    result = {}
    for source, root in (("minutes", MINUTES_DIR), ("test_minutes", TEST_MINUTES_DIR)):
        if not root.exists():
            continue
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            mpath = folder / "manifest.json"
            if not mpath.exists():
                continue
            man, _ = load_manifest(mpath)
            labels = man.get("labels", []) or []
            placement, activity, position = parse_labels(labels)
            rec_id = f"{source}/{folder.name}"
            result[rec_id] = {
                "placement": placement,
                "activity": activity,
                "position": position,
                "source": source,
                "folder": str(folder),
            }
    return result


# ---------------------------------------------------------------------------
# capture.npz -> radar frames (for the predict function)
# ---------------------------------------------------------------------------
def decode_npz_frames(npz_path):
    """Decode a capture.npz file -> (payloads, unix_ts).

    Frame ORDER comes from `radar_sample_sequence` (the hardware sequence
    number): the per-frame `radar_sample_unix_ns` timestamps are
    burst-flushed and unreliable, so sorting by them scrambles the true
    frame order and corrupts every temporal statistic downstream.

    Timestamps are reconstructed from the reliable per-second buckets
    (`second_start_unix_ns` + `radar_sample_second_index`), distributing
    each second's frames uniformly across the measured span — the same
    rule E2's `recording_radar_frames` used to build the training cache,
    so fresh decodes and cached windows see identical frame sequences.

    Returns raw ADC byte strings (header stripped, each exactly
    EXPECTED_PAYLOAD bytes) and their reconstructed Unix timestamps.
    """
    npz = np.load(npz_path, allow_pickle=True)
    required = {"radar_sample_bytes", "radar_sample_offsets",
                "radar_sample_sequence", "radar_sample_second_index",
                "second_start_unix_ns"}
    if not required.issubset(set(npz.files)):
        raise ValueError(f"{npz_path}: missing required arrays {required - set(npz.files)}")
    raw_bytes = npz["radar_sample_bytes"]
    offsets = npz["radar_sample_offsets"]
    seq = npz["radar_sample_sequence"].astype(np.int64)
    sec_idx = npz["radar_sample_second_index"].astype(np.int64)
    sec_start = npz["second_start_unix_ns"].astype(np.float64) / 1e9
    n = len(offsets) - 1

    # hardware sequence number gives the true frame order
    order = np.argsort(seq[:n], kind="stable")

    # per-frame times from the per-second buckets (uniform within second)
    ts = np.empty(n, dtype=np.float64)
    for s in np.unique(sec_idx):
        members = np.where(sec_idx == s)[0]
        members = members[np.argsort(seq[members], kind="stable")]
        t0 = sec_start[s]
        nxt = sec_start[s + 1] if s + 1 < len(sec_start) else t0 + 1.0
        span = nxt - t0
        if not np.isfinite(span) or span <= 0 or span > 5.0:
            span = 1.0
        for rank, fi in enumerate(members):
            ts[fi] = t0 + rank * (span / len(members))

    payloads, valid_ts = [], []
    for i in order:
        blob = raw_bytes[offsets[i]: offsets[i + 1]]
        if blob.shape[0] <= FRAME_HEADER_BYTES:
            continue
        header = blob[:FRAME_HEADER_BYTES]
        data_len = int(np.frombuffer(header[8:12].tobytes(), dtype="<u4")[0])
        payload = blob[FRAME_HEADER_BYTES:]
        if payload.shape[0] != data_len or data_len != EXPECTED_PAYLOAD:
            continue
        payloads.append(bytes(payload))
        valid_ts.append(ts[i])
    return payloads, np.asarray(valid_ts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Radar FFT maps (parameterized so one function serves both E2 and E3 specs)
# ---------------------------------------------------------------------------
def read_uint12_batch(blob):
    d = np.frombuffer(blob, dtype=np.uint8)
    trip = d.reshape(-1, 3).astype(np.uint16)
    a, b, c = trip[:, 0], trip[:, 1], trip[:, 2]
    x = np.empty(a.shape[0] * 2, dtype=np.float32)
    x[0::2] = (a << 4) + (b >> 4)
    x[1::2] = ((b % 16) << 8) + c
    return x


_HANN_R = np.hanning(NUM_SAMPLES).astype(np.float32)
_HANN_D = np.hanning(NUM_CHIRPS).astype(np.float32)


def frames_to_maps(payloads, map_size, az_bins, range_bins, az_fftshift=False):
    """payloads -> float16 (N, 2, map_size, map_size) log1p-compressed
    range-Doppler + range-azimuth maps.

    `az_fftshift` must match the cache the consuming model was trained
    on: E2's preprocess does NOT shift the azimuth FFT (az_fftshift=False
    for the E2 spec used by stages 1-2); E3's does (az_fftshift=True for
    the E3 spec used by stage 3). A mismatch circularly shifts the
    azimuth channel — invisible to spatial-pooling encoders (occupancy)
    but corrupting for CNN encoders that use spatial structure.
    """
    from scipy import fft as sfft
    from scipy.ndimage import zoom
    n = len(payloads)
    adc = read_uint12_batch(b"".join(payloads)).reshape(
        n, NUM_CHIRPS, NUM_SAMPLES, NUM_ANTENNAS
    )
    adc *= _HANN_R[None, None, :, None] * _HANN_D[None, :, None, None]
    R = sfft.fft(adc, axis=2, workers=-1)
    RD = sfft.fftshift(sfft.fft(R, axis=1, workers=-1), axes=1)
    rd = np.abs(RD[:, :, :range_bins, :]).mean(axis=3)
    RA = sfft.fft(R[:, :, :range_bins, :], n=az_bins, axis=3, workers=-1)
    if az_fftshift:
        RA = sfft.fftshift(RA, axes=3)
    ra = np.abs(RA).mean(axis=1).transpose(0, 2, 1)
    s = map_size
    rd = zoom(np.log1p(rd), (1, s / rd.shape[1], s / rd.shape[2]), order=1)
    ra = zoom(np.log1p(ra), (1, s / ra.shape[1], s / ra.shape[2]), order=1)
    return np.stack([rd, ra], axis=1).astype(np.float16)


def build_windows(maps, window_frames):
    """Non-overlapping windows of `window_frames` consecutive frames."""
    n = maps.shape[0]
    n_win = n // window_frames
    if n_win == 0:
        return np.empty((0, window_frames) + maps.shape[1:], dtype=maps.dtype)
    return maps[: n_win * window_frames].reshape(
        n_win, window_frames, *maps.shape[1:]
    )
