"""Shared dataset utilities for the E3 left/right localization pipeline.

Only `test_minutes/` carries a left/right ground-truth label (see
`E2/common.py` for the occupancy-label conventions used by `minutes/`).
Within `test_minutes/`, radar frames are stored in one of two on-disk
formats (verified by direct inspection):

  * `capture.npz`     -- one synchronized container, real per-frame Unix
                         timestamps in `radar_sample_unix_ns` (majority of
                         recordings).
  * `radar_*.bin`      -- a few chunked raw files referenced from
                         `manifest.json -> outputs.radar.chunks[*]`; real
                         per-frame timestamps are recovered from each
                         chunk's `frame_monotonic_ns` plus the minute's
                         `capture_started` / `capture_started_monotonic_ns`
                         anchor.

Both formats decode to the same raw MMW-HAT frame blob: a 12-byte header
(version, sequence, payload length) followed by a uint12-packed ADC
payload for 64 chirps x 128 samples x 3 RX antennas.

Timing caveat (documented, does not affect window construction): for the
`.bin` chunked format, most chunks' `frame_monotonic_ns` entries are
evenly spaced (~100 ms, matching the ~10 Hz frame rate), but some chunks
have all 10 frames' monotonic timestamps compressed into a ~1 ms burst
(a recorder-side artifact, analogous to the burst-flushed per-frame
timestamps documented for `capture.npz` in `E2/common.py`). Timestamps
remain non-decreasing both within and across chunks, so frame ORDER
(what `preprocess.py` actually uses to build windows) is unaffected;
only the reported per-frame duration/FPS diagnostics are noisy for these
chunks.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEST_MINUTES_DIR = ROOT / "test_minutes"
E3_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = E3_DIR / "outputs"
CACHE_DIR = E3_DIR / "cache"

FRAME_HEADER_BYTES = 12
NUM_CHIRPS = 64
NUM_SAMPLES = 128
NUM_ANTENNAS = 3

POSITION_LABELS = ("left", "right")
CLASS_NAMES = ["left", "right"]


# ---------------------------------------------------------------------------
# Manifest parsing (tolerant of the occasional malformed/duplicated-key JSON
# seen in this dataset, same issue documented in E2/common.py)
# ---------------------------------------------------------------------------
def load_manifest(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text), "json"
    except Exception:
        pass
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text)
        return obj, "json_partial"
    except Exception:
        return {}, "unparsable"


def parse_iso(ts):
    if ts is None:
        return float("nan")
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except Exception:
        return float("nan")


def folder_start_ts(name):
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M").timestamp()
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Recording discovery
# ---------------------------------------------------------------------------
@dataclass
class Recording:
    rec_id: str
    folder: Path
    label: int  # 0 = left, 1 = right
    orig_labels: list
    start_ts: float
    end_ts: float
    parse_mode: str
    storage_format: str  # 'npz' | 'bin'
    status: str = "ok"
    notes: list = field(default_factory=list)


def discover_recordings():
    """Scan test_minutes/ and return (recordings, problems); only minutes
    with exactly one of {left, right} in their manifest `labels` are kept
    as usable."""
    recordings = []
    problems = []
    if not TEST_MINUTES_DIR.exists():
        problems.append(f"missing root: {TEST_MINUTES_DIR}")
        return recordings, problems

    for folder in sorted(TEST_MINUTES_DIR.iterdir()):
        if not folder.is_dir():
            continue
        mpath = folder / "manifest.json"
        if not mpath.exists():
            problems.append(f"{folder.name}: no manifest.json")
            continue
        man, mode = load_manifest(mpath)
        labels = man.get("labels", []) or []
        present = [l for l in labels if l in POSITION_LABELS]
        if not present:
            continue  # not a left/right minute; ignore silently
        if len(set(present)) > 1:
            problems.append(f"{folder.name}: ambiguous position labels {present}")
            continue
        label_name = present[0]
        y = POSITION_LABELS.index(label_name)

        has_npz = (folder / "capture.npz").exists()
        has_bin = any(folder.glob("radar_*.bin"))
        if has_npz:
            storage_format = "npz"
        elif has_bin:
            storage_format = "bin"
        else:
            problems.append(f"{folder.name}: no capture.npz or radar_*.bin found")
            continue

        start = parse_iso(man.get("capture_started") or man.get("scheduled_start"))
        if not np.isfinite(start):
            start = folder_start_ts(folder.name)
        end = parse_iso(man.get("capture_finished"))
        if not np.isfinite(end) or end <= start:
            end = start + float(man.get("duration_seconds") or 60.0)

        rec = Recording(
            rec_id=f"test_minutes/{folder.name}",
            folder=folder,
            label=y,
            orig_labels=list(labels),
            start_ts=float(start),
            end_ts=float(end),
            parse_mode=mode,
            storage_format=storage_format,
        )
        if mode != "json":
            rec.notes.append("malformed manifest JSON; partial-decode fallback used")
            problems.append(f"{rec.rec_id}: manifest parse_mode={mode}")
        recordings.append(rec)
    return recordings, problems


# ---------------------------------------------------------------------------
# Radar frame access -- unifies the capture.npz and radar_*.bin formats
# ---------------------------------------------------------------------------
def _iter_bin_file_blobs(bin_path):
    with open(bin_path, "rb") as fh:
        while True:
            header = fh.read(FRAME_HEADER_BYTES)
            if not header or len(header) != FRAME_HEADER_BYTES:
                return
            data_len = int.from_bytes(header[8:12], byteorder="little", signed=False)
            payload = fh.read(data_len)
            if len(payload) != data_len:
                return
            yield payload


def _recording_frames_from_npz(rec):
    npz = np.load(rec.folder / "capture.npz", allow_pickle=True)
    required = {"radar_sample_bytes", "radar_sample_offsets", "radar_sample_unix_ns"}
    if not required.issubset(set(npz.files)):
        raise ValueError(f"{rec.rec_id}: capture.npz missing required arrays")
    raw_bytes = npz["radar_sample_bytes"]
    offsets = npz["radar_sample_offsets"]
    timestamps = npz["radar_sample_unix_ns"].astype(np.float64) / 1e9
    n = len(offsets) - 1
    payloads = []
    valid_ts = []
    expected_payload_len = NUM_CHIRPS * NUM_SAMPLES * NUM_ANTENNAS * 3 // 2
    for i in range(n):
        blob = raw_bytes[offsets[i]: offsets[i + 1]]
        if blob.shape[0] <= FRAME_HEADER_BYTES:
            continue
        header = blob[:FRAME_HEADER_BYTES]
        data_len = int(np.frombuffer(header[8:12].tobytes(), dtype="<u4")[0])
        payload = blob[FRAME_HEADER_BYTES:]
        if payload.shape[0] != data_len or data_len != expected_payload_len:
            continue
        payloads.append(bytes(payload))
        valid_ts.append(timestamps[i])
    order = np.argsort(valid_ts, kind="stable")
    payloads = [payloads[i] for i in order]
    ts = np.asarray(valid_ts, dtype=np.float64)[order]
    return payloads, ts


def _recording_frames_from_bin(rec):
    man, _ = load_manifest(rec.folder / "manifest.json")
    chunks = man.get("outputs", {}).get("radar", {}).get("chunks", [])
    capture_started_iso = man.get("capture_started")
    capture_started_mono_ns = man.get("capture_started_monotonic_ns")
    capture_started_unix_ns = parse_iso(capture_started_iso) * 1e9

    expected_payload_len = NUM_CHIRPS * NUM_SAMPLES * NUM_ANTENNAS * 3 // 2

    payloads = []
    timestamps = []
    for chunk in sorted(chunks, key=lambda c: c.get("chunk_index", 0)):
        bin_name = Path(chunk.get("bin_path", "")).name
        if not bin_name:
            continue
        bin_path = rec.folder / bin_name
        if not bin_path.exists():
            continue
        mono_list = chunk.get("frame_monotonic_ns", []) or []
        blobs = list(_iter_bin_file_blobs(bin_path))
        n = min(len(blobs), len(mono_list)) if mono_list else len(blobs)
        for i in range(n):
            payload = blobs[i]
            if len(payload) != expected_payload_len:
                continue
            if mono_list and np.isfinite(capture_started_unix_ns) and capture_started_mono_ns is not None:
                unix_ns = capture_started_unix_ns + (int(mono_list[i]) - int(capture_started_mono_ns))
            else:
                unix_ns = capture_started_unix_ns
            payloads.append(payload)
            timestamps.append(unix_ns / 1e9)

    ts = np.asarray(timestamps, dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    payloads = [payloads[i] for i in order]
    ts = ts[order]
    return payloads, ts


def recording_radar_frames(rec):
    """Return (payloads, unix_ts) for all valid radar frames of a
    recording, sorted by real timestamp. `payloads` are raw ADC byte
    strings (uint12-packed, header already stripped), each exactly
    NUM_CHIRPS*NUM_SAMPLES*NUM_ANTENNAS*3//2 bytes."""
    if rec.storage_format == "npz":
        return _recording_frames_from_npz(rec)
    return _recording_frames_from_bin(rec)
