"""Shared dataset-inspection utilities for the E1 data-audit pipeline.

E1 is the exploratory/audit stage: it scans every minute folder in
`train_minutes/`, `train2_minutes/`, `validation_minutes/` and
`test_minutes/`, checks file completeness and integrity, parses manifests
tolerantly (some manifests are malformed JSON — the same issue
documented in `E2/common.py`), tabulates labels, and catalogs every
missing/corrupt instance.

On-disk formats (verified by direct inspection):

  train_minutes/   one folder per captured minute, named YYYYMMDD_HHMM.
                   Radar is stored as chunked `radar_NNN_YYYYMMDD_HHMMSS_mmm.bin`
                   files (each nominally one second / 10 frames of MMW-HAT
                   frames: 12-byte header + uint12-packed ADC payload of
                   64 chirps x 128 samples x 3 RX = 36864 payload bytes).
                   CSI is stored as `wifi_csi.csv` / `wifi_csi_XX.csv` text
                   files (one per ESP32 receiver; typically one is empty).
                   `manifest.json`, `xy-tracking.json` and
                   `.home_assistant_status.json` complete the folder.

  train2_minutes/,   same minute-folder convention, but radar/CSI/sense/
  validation_minutes/,  camera samples are usually packed into a single
  test_minutes/      synchronized `capture.npz` container. A minority of
                   "eventful" minutes fall back to chunked `radar_*.bin`
                   files instead. `sense_hat.error.json` records
                   Sense-HAT failures.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRAIN_MINUTES_DIR = ROOT / "train_minutes"
TRAIN2_MINUTES_DIR = ROOT / "train2_minutes"          # placement t, Sept 6-8
VALIDATION_MINUTES_DIR = ROOT / "validation_minutes"  # placement t, Sept 15
TEST_MINUTES_DIR = ROOT / "test_minutes"              # Pi night, Sept 16-17
E1_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = E1_DIR / "outputs"
FIGS_DIR = OUTPUT_DIR / "figs"

SOURCES = ("train_minutes", "train2_minutes", "validation_minutes",
           "test_minutes")
SOURCE_ROOTS = {"train_minutes": TRAIN_MINUTES_DIR,
                "train2_minutes": TRAIN2_MINUTES_DIR,
                "validation_minutes": VALIDATION_MINUTES_DIR,
                "test_minutes": TEST_MINUTES_DIR}
# sources whose radar/CSI arrive as capture.npz (not chunked .bin/csv)
NPZ_SOURCES = ("train2_minutes", "validation_minutes", "test_minutes")

FRAME_HEADER_BYTES = 12
NUM_CHIRPS = 64
NUM_SAMPLES = 128
NUM_ANTENNAS = 3
# uint12 packing: 2 samples per 3 bytes
EXPECTED_PAYLOAD_BYTES = NUM_CHIRPS * NUM_SAMPLES * NUM_ANTENNAS * 3 // 2  # 36864
EXPECTED_FRAME_BYTES = FRAME_HEADER_BYTES + EXPECTED_PAYLOAD_BYTES          # 36876

# Label taxonomy (documented in E2):
#   train_minutes/      : t1_empty/t1_sleep/t1_present (placement t1),
#                         t2_empty/t2_sleep/t2_present (placement t2)
#   train2_minutes/     : t_empty/t_sleep/t_present    (placement t)
#   validation_minutes/ : empty/present                (placement t, new
#                         naming — no t_ prefix, no sleep)
#   test_minutes/       : empty/present — Pi captures from the night of
#                         Sept 16-17, labeled by E2/label_test_minutes.py
#                         from the 1:10 AM boundary (present before
#                         20260917_0110, empty at/after); the boundary is
#                         the fallback for unlabeled folders
#   position            : left / right                 (train2_minutes only)
#   flags               : radar-missing
#   auxiliary           : absent, occupied (auto-labels from the recorder's
#                         own minute_summary; stripped by E2/clean_minutes.py)
PLACEMENT_LABELS = {
    "train_minutes": {"t1_empty", "t1_sleep", "t1_present",
                      "t2_empty", "t2_sleep", "t2_present"},
    "train2_minutes": {"t_empty", "t_sleep", "t_present"},
    "validation_minutes": {"empty", "present"},
    "test_minutes": {"empty", "present"},
}

# First empty minute on the Pi test night (folder-name timestamp).
TEST_BOUNDARY_FOLDER = "20260917_0110"


def boundary_label(folder_name):
    """test_minutes ground truth: 1 occupied / 0 empty by folder name."""
    return 0 if folder_name >= TEST_BOUNDARY_FOLDER else 1
ACTIVITY_OF = {
    "t1_empty": "empty", "t2_empty": "empty", "t_empty": "empty",
    "empty": "empty",
    "t1_sleep": "sleep", "t2_sleep": "sleep", "t_sleep": "sleep",
    "t1_present": "present", "t2_present": "present",
    "t_present": "present", "present": "present",
}
PLACEMENT_OF = {
    "t1_empty": "t1", "t1_sleep": "t1", "t1_present": "t1",
    "t2_empty": "t2", "t2_sleep": "t2", "t2_present": "t2",
    "t_empty": "t", "t_sleep": "t", "t_present": "t",
    "empty": "t", "present": "t",
}
POSITION_LABELS = ("left", "right")
FLAG_LABELS = ("radar-missing",)
AUX_LABELS = ("absent", "occupied")

NPZ_REQUIRED_ARRAYS = (
    "radar_sample_bytes", "radar_sample_offsets", "radar_sample_sequence",
    "radar_sample_second_index", "second_start_unix_ns",
)


# ---------------------------------------------------------------------------
# Tolerant manifest parsing
# ---------------------------------------------------------------------------
def load_manifest(path):
    """Return (manifest_dict, parse_mode).

    parse_mode is 'json' (clean), 'json_partial' (leading valid JSON
    recovered via raw_decode — duplicated/truncated manifests), 'regex'
    (only labels/timestamps recovered), or 'unparsable'.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text), "json"
    except Exception:
        pass
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text)
        if isinstance(obj, dict):
            return obj, "json_partial"
    except Exception:
        pass
    m = re.search(r'"labels"\s*:\s*\[([^\]]*)\]', text)
    labels = re.findall(r'"([^"]+)"', m.group(1)) if m else []
    if labels or '"labels"' in text:
        return {"labels": labels}, "regex"
    return {}, "unparsable"


def parse_iso(ts):
    """ISO-8601 timestamp -> unix seconds (float). NaN on failure."""
    if ts is None:
        return float("nan")
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except Exception:
        return float("nan")


def folder_start_ts(name):
    """Parse folder name YYYYMMDD_HHMM as a fallback timestamp."""
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M").timestamp()
    except Exception:
        return float("nan")


def radar_fname_ts(name):
    """Parse radar_000_20260817_183201_192.bin -> unix seconds."""
    m = re.search(r"(\d{8})_(\d{6})_(\d{3})", name)
    if not m:
        return float("nan")
    d, t, ms = m.groups()
    try:
        base = datetime.strptime(d + t, "%Y%m%d%H%M%S").timestamp()
        return base + int(ms) / 1000.0
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
def classify_labels(labels, source, folder_name=""):
    """Map a manifest label list to task fields.

    Returns dict with placement, activity, position, flags, aux, and a
    status: 'ok' | 'unlabeled' | 'ambiguous' (conflicting task labels).
    test_minutes has no task labels — activity/placement come from the
    1:10 AM boundary rule instead.
    """
    labels = list(labels or [])
    task = [l for l in labels if l in PLACEMENT_LABELS.get(source, ())]
    positions = [l for l in labels if l in POSITION_LABELS]
    flags = [l for l in labels if l in FLAG_LABELS]
    aux = [l for l in labels if l in AUX_LABELS and l not in task]
    other = [l for l in labels
             if l not in PLACEMENT_LABELS.get(source, ())
             and l not in POSITION_LABELS
             and l not in FLAG_LABELS
             and l not in AUX_LABELS]

    placements = {PLACEMENT_OF[l] for l in task}
    activities = {ACTIVITY_OF[l] for l in task}
    status = "ok"
    if not task:
        status = "unlabeled"
    elif len(placements) > 1 or len(activities) > 1:
        status = "ambiguous"
    if len(set(positions)) > 1:
        status = "ambiguous"
    if source == "test_minutes":
        # manifests carry empty/present written by label_test_minutes.py;
        # fall back to the 1:10 AM boundary for unlabeled folders
        if task:
            act = sorted(activities)[0] if len(activities) == 1 else ""
            st = "ok" if act else "ambiguous"
        else:
            act = "empty" if boundary_label(folder_name) == 0 else "present"
            st = "ok" if folder_name else "unlabeled"
        return {
            "placement": "pi",
            "activity": act,
            "position": "",
            "flags": flags,
            "aux": aux,
            "other": other,
            "task_labels": task,
            "status": st,
        }
    return {
        "placement": sorted(placements)[0] if len(placements) == 1 else "",
        "activity": sorted(activities)[0] if len(activities) == 1 else "",
        "position": positions[0] if len(set(positions)) == 1 else "",
        "flags": flags,
        "aux": aux,
        "other": other,
        "task_labels": task,
        "status": status,
    }


# ---------------------------------------------------------------------------
# File-level integrity checks
# ---------------------------------------------------------------------------
def check_bin_file(path, frame_bytes=EXPECTED_FRAME_BYTES):
    """Cheap integrity check for a chunked radar .bin file.

    Returns (n_frames, ok, note). A file is OK when its size is an exact
    multiple of the frame size (12-byte header + fixed payload). Files
    smaller than one frame or with a trailing partial frame are flagged.
    """
    size = path.stat().st_size
    if size == 0:
        return 0, False, "empty file"
    if size < frame_bytes:
        return 0, False, f"smaller than one frame ({size} B)"
    if size % frame_bytes != 0:
        return size // frame_bytes, False, f"trailing partial frame ({size % frame_bytes} B)"
    return size // frame_bytes, True, ""


def check_npz(path):
    """Inspect a capture.npz container without decompressing payloads.

    Returns (info_dict, ok, note). np.load is lazy: reading array shapes
    only parses the zip directory + array headers, so this is fast.
    """
    info = {}
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as exc:
        return info, False, f"npz open failed: {exc}"
    try:
        files = set(d.files)
        info["npz_keys"] = sorted(files)
        missing = [k for k in NPZ_REQUIRED_ARRAYS if k not in files]
        info["npz_missing_keys"] = missing
        if "radar_sample_offsets" in files:
            info["npz_radar_frames"] = int(len(d["radar_sample_offsets"]) - 1)
        if "csi_sample_offsets" in files:
            info["npz_csi_samples"] = int(len(d["csi_sample_offsets"]) - 1)
        if "csi_sample_receiver_index" in files:
            rx = d["csi_sample_receiver_index"]
            info["npz_csi_receivers"] = sorted(int(v) for v in np.unique(rx))
        if "sense_sample_offsets" in files:
            info["npz_sense_samples"] = int(len(d["sense_sample_offsets"]) - 1)
        if "camera_jpeg_offsets" in files:
            info["npz_camera_frames"] = int(len(d["camera_jpeg_offsets"]) - 1)
        if "camera_present" in files:
            info["npz_camera_present"] = int(np.asarray(d["camera_present"]).sum())
        ok = not missing
        return info, ok, ("missing arrays: " + ",".join(missing)) if missing else ""
    except Exception as exc:
        return info, False, f"npz read failed: {exc}"


def count_csv_lines(path, chunk_size=1 << 20):
    """Count newlines in a file via buffered binary reads (fast)."""
    n = 0
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            n += buf.count(b"\n")
    return n


def check_json_file(path, full_parse=False):
    """Cheap structural check: non-empty, starts with '{', ends with '}'.

    Optionally json.loads the whole file (used on a sample only — the
    multi-MB xy-tracking files are expensive to parse exhaustively).
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"stat failed: {exc}"
    if size == 0:
        return False, "empty file"
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096).lstrip()
            fh.seek(max(0, size - 4096))
            tail = fh.read(4096).rstrip()
    except OSError as exc:
        return False, f"read failed: {exc}"
    if not head.startswith(b"{"):
        return False, "does not start with '{'"
    if not tail.endswith(b"}"):
        return False, "truncated (no closing '}')"
    if full_parse:
        try:
            json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            return False, f"json parse failed: {exc}"
    return True, ""
