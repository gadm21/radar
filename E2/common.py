"""Shared dataset utilities for the E2 occupancy-detection pipeline.

Handles tolerant manifest parsing (some manifests are malformed JSON),
unified discovery of `minutes/` and `test_minutes/` recordings, label
mapping to the binary occupancy task, and session inference.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MINUTES_DIR = ROOT / "minutes"
TEST_MINUTES_DIR = ROOT / "test_minutes"
E2_DIR = ROOT / "E2"
OUTPUT_DIR = E2_DIR / "outputs"
CACHE_DIR = E2_DIR / "cache"

# ---------------------------------------------------------------------------
# Label handling
# ---------------------------------------------------------------------------
# minutes/ : t1_empty, t1_sleep, t1_present, t2_empty, t2_sleep, t2_present
# test_minutes/ : t_empty, t_sleep, t_present   (left/right are ignored)
MINUTES_LABELS = {
    "t1_empty": ("t1", 0),
    "t1_sleep": ("t1", 1),
    "t1_present": ("t1", 1),
    "t2_empty": ("t2", 0),
    "t2_sleep": ("t2", 1),
    "t2_present": ("t2", 1),
}
TEST_LABELS = {
    "t_empty": ("t", 0),
    "t_sleep": ("t", 1),
    "t_present": ("t", 1),
}
CLASS_NAMES = ["empty", "occupied"]

SESSION_GAP_SECONDS = 600.0  # gap larger than this starts a new session


def map_labels(labels, source):
    """Map a manifest label list to (placement, binary_label).

    Returns (placement, y, status) where status is 'ok', 'ignored' or
    'ambiguous'. Only the documented labels are used; everything else is
    ignored. Multiple usable labels on one recording are ambiguous.
    """
    table = MINUTES_LABELS if source == "minutes" else TEST_LABELS
    usable = [lab for lab in labels if lab in table]
    if not usable:
        return None, None, "ignored"
    uniq = {(table[l][0], table[l][1]) for l in usable}
    if len(uniq) > 1:
        return None, None, "ambiguous"
    placement, y = next(iter(uniq))
    return placement, y, "ok"


# ---------------------------------------------------------------------------
# Tolerant manifest parsing
# ---------------------------------------------------------------------------
def _regex_str(text, key):
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"([^"]*)"', text)
    return m.group(1) if m else None


def _regex_num(text, key):
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*(-?[0-9.eE+]+)', text)
    return float(m.group(1)) if m else None


def _regex_labels(text):
    m = re.search(r'"labels"\s*:\s*\[([^\]]*)\]', text)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


def load_manifest(path):
    """Return (manifest_dict, parse_mode). parse_mode is 'json' or 'regex'.

    For malformed JSON we recover the fields the pipeline needs via regex:
    labels, scheduled_start, capture_finished, and per-chunk
    (started, finished_capture, chunk_seconds, chunk_frames, bin_path).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text), "json"
    except Exception:
        pass
    man = {
        "labels": _regex_labels(text),
        "scheduled_start": _regex_str(text, "scheduled_start"),
        "capture_finished": _regex_str(text, "capture_finished"),
        "status": _regex_str(text, "status"),
    }
    starts = re.findall(r'"started"\s*:\s*"([^"]+)"', text)
    fins = re.findall(r'"finished_capture"\s*:\s*"([^"]+)"', text)
    secs = re.findall(r'"chunk_seconds"\s*:\s*(-?[0-9.eE+]+)', text)
    nfr = re.findall(r'"chunk_frames"\s*:\s*(\d+)', text)
    bins = re.findall(r'"bin_path"\s*:\s*"([^"]+)"', text)
    chunks = []
    for i, b in enumerate(bins):
        chunks.append({
            "chunk_index": i,
            "started": starts[i] if i < len(starts) else None,
            "finished_capture": fins[i] if i < len(fins) else None,
            "chunk_seconds": float(secs[i]) if i < len(secs) else None,
            "chunk_frames": int(nfr[i]) if i < len(nfr) else None,
            "bin_path": b,
        })
    man["outputs"] = {"radar": {"chunks": chunks}}
    return man, "regex"


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
# Recording discovery -> unified internal representation
# ---------------------------------------------------------------------------
@dataclass
class Recording:
    """One minute-folder recording in a unified representation."""
    rec_id: str            # e.g. 'minutes/20260817_1832'
    source: str            # 'minutes' | 'test_minutes'
    folder: Path
    placement: str         # 't1' | 't2' | 't'
    label: int             # 0 empty, 1 occupied
    orig_labels: list
    start_ts: float        # unix seconds
    end_ts: float
    parse_mode: str        # 'json' | 'regex'
    status: str = "ok"
    notes: list = field(default_factory=list)
    session_id: str = ""   # filled by assign_sessions


def discover_recordings():
    """Scan both dataset roots and return (recordings, problems)."""
    recordings = []
    problems = []
    for source, root in (("minutes", MINUTES_DIR), ("test_minutes", TEST_MINUTES_DIR)):
        if not root.exists():
            problems.append(f"missing root: {root}")
            continue
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            mpath = folder / "manifest.json"
            if not mpath.exists():
                problems.append(f"{source}/{folder.name}: no manifest.json")
                continue
            man, mode = load_manifest(mpath)
            labels = man.get("labels", []) or []
            placement, y, status = map_labels(labels, source)
            start = parse_iso(man.get("scheduled_start"))
            if not np.isfinite(start):
                start = folder_start_ts(folder.name)
            end = parse_iso(man.get("capture_finished"))
            if not np.isfinite(end) or end <= start:
                end = start + float(man.get("duration_seconds") or 60.0)
            rec = Recording(
                rec_id=f"{source}/{folder.name}",
                source=source,
                folder=folder,
                placement=placement or "?",
                label=-1 if y is None else y,
                orig_labels=list(labels),
                start_ts=float(start),
                end_ts=float(end),
                parse_mode=mode,
                status=status,
            )
            if status != "ok":
                rec.notes.append(f"label status: {status}")
                problems.append(f"{rec.rec_id}: {status} labels={labels}")
            if mode == "regex":
                rec.notes.append("malformed manifest JSON; regex fallback used")
                problems.append(f"{rec.rec_id}: malformed manifest (regex fallback)")
            recordings.append(rec)
    return recordings, problems


def assign_sessions(recordings):
    """Infer collection sessions.

    Rule (documented): sort recordings of one placement by start time;
    a new session begins when (a) the gap from the previous recording's
    start exceeds SESSION_GAP_SECONDS, (b) the binary label changes, or
    (c) the placement changes. Session ids: '<placement>_s<NN>'.
    """
    usable = [r for r in recordings if r.status == "ok"]
    usable.sort(key=lambda r: (r.placement, r.start_ts))
    counters = {}
    prev = {}
    for r in usable:
        p = r.placement
        counters.setdefault(p, 0)
        last = prev.get(p)
        new = (
            last is None
            or (r.start_ts - last.start_ts) > SESSION_GAP_SECONDS
            or r.label != last.label
        )
        if new:
            counters[p] += 1
        r.session_id = f"{p}_s{counters[p]:02d}"
        prev[p] = r
    return recordings


# ---------------------------------------------------------------------------
# Radar frame access (shared by .bin chunks and capture.npz)
# ---------------------------------------------------------------------------
def iter_bin_frames(bin_path):
    """Yield (seq, payload_bytes) for each MMW-HAT frame in a .bin file."""
    with open(bin_path, "rb") as fh:
        while True:
            header = fh.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"truncated frame header in {bin_path}")
            dlen = int.from_bytes(header[8:12], "little")
            payload = fh.read(dlen)
            if len(payload) != dlen:
                raise ValueError(f"truncated frame payload in {bin_path}")
            seq = int.from_bytes(header[4:8], "little")
            yield seq, payload


def recording_radar_frames(rec, manifest=None):
    """Return (payloads, timestamps) for all radar frames of a recording.

    test_minutes: real per-frame unix-ns timestamps from capture.npz.
    minutes: per-chunk timing from the manifest; frames inside a chunk are
    spaced uniformly between the chunk start and the next chunk's start
    (or finished_capture / nominal 100 ms spacing as fallback). The
    filename timestamp is used when manifest timing is missing.
    """
    if rec.source == "test_minutes":
        npz = rec.folder / "capture.npz"
        d = np.load(npz)
        off = d["radar_sample_offsets"]
        raw = d["radar_sample_bytes"]
        n = len(off) - 1
        payloads = [bytes(raw[off[i]:off[i + 1]])[12:] for i in range(n)]
        # Per-frame unix/monotonic timestamps are burst-flushed and
        # unreliable in most captures. The per-second bucket boundaries
        # (second_start_unix_ns + radar_sample_second_index) are reliable:
        # ~10 frames per second. Distribute each second's frames uniformly
        # across the measured second span (documented rule).
        sec_idx = d["radar_sample_second_index"].astype(np.int64)
        sec_start = d["second_start_unix_ns"].astype(np.float64) / 1e9
        seq = d["radar_sample_sequence"].astype(np.int64)
        order = np.argsort(seq, kind="stable")
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
        return [payloads[i] for i in order], ts[order]

    # minutes/: each radar_*.bin file is named by its capture timestamp and
    # holds the frames of one second (nominally 10). Frame times are the
    # filename timestamp + j/n seconds (documented rule).
    entries = []  # (start_ts, bin_path)
    for bp in rec.folder.glob("radar_*.bin"):
        st = radar_fname_ts(bp.name)
        entries.append([st, bp])
    entries.sort(key=lambda e: (e[0] if np.isfinite(e[0]) else 1e18,
                                e[1].name))
    payloads, times = [], []
    for st, bp in entries:
        frames = [p for _, p in iter_bin_frames(bp)]
        n = len(frames)
        if n == 0:
            continue
        if not np.isfinite(st):
            st = times[-1] + 1.0 if times else rec.start_ts
        dt = 1.0 / n  # one bin file spans one second
        for j, p in enumerate(frames):
            payloads.append(p)
            times.append(st + j * dt)
    times = np.asarray(times, dtype=np.float64)
    order = np.argsort(times, kind="stable")
    return [payloads[i] for i in order], times[order]


# ---------------------------------------------------------------------------
# CSI access
# ---------------------------------------------------------------------------
_CSI_RE = re.compile(r"\[([^\]]+)\]")
CSI_SUBCARRIER_MASK = np.array(
    [False] * 6 + [True] * 26 + [False] + [True] * 26 + [False] * 5,
    dtype=bool,
)


def parse_csi_line(line):
    """Parse one CSI_DATA line -> complex64 vector of 52 subcarriers."""
    m = _CSI_RE.search(line)
    if not m:
        return None
    toks = [t for t in m.group(1).split(",") if t.strip()]
    if len(toks) != 128:
        return None
    try:
        v = np.asarray(toks, dtype=np.float64)
    except ValueError:
        return None
    imag = v[0::2][CSI_SUBCARRIER_MASK]
    real = v[1::2][CSI_SUBCARRIER_MASK]
    return (real + 1j * imag).astype(np.complex64)


def _parse_iso_fast(ts):
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return float("nan")


def _csi_csv_coverage(path, t0, t1):
    """Count rows whose host_timestamp falls inside [t0, t1] (cheap pass)."""
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            fh.readline()  # header
            for line in fh:
                comma = line.find(",")
                if comma <= 0:
                    continue
                t = _parse_iso_fast(line[:comma].strip().strip('"'))
                if np.isfinite(t) and t0 <= t <= t1:
                    n += 1
    except OSError:
        return 0
    return n


def _parse_csi_csv(path):
    csi, ts = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        fh.readline()  # header
        for line in fh:
            comma = line.find(",")
            if comma <= 0:
                continue
            t = _parse_iso_fast(line[:comma].strip().strip('"'))
            if not np.isfinite(t):
                continue
            v = parse_csi_line(line)
            if v is not None:
                csi.append(v)
                ts.append(t)
    return csi, ts


def recording_csi(rec, receiver_index=None):
    """Return (complex_csi[N,52], unix_ts[N]) for the best single receiver.

    minutes/: several wifi_csi_XX.csv files may exist; typically one is
    empty. The file with the most samples inside the recording's
    [start_ts, end_ts] window is used (documented rule); ties -> first.
    test_minutes/: csi_sample_* arrays in capture.npz; the receiver index
    with the most samples is used.
    """
    if rec.source == "test_minutes":
        d = np.load(rec.folder / "capture.npz")
        rx = d["csi_sample_receiver_index"]
        off = d["csi_sample_offsets"]
        raw = d["csi_sample_bytes"]
        ts_all = d["csi_sample_unix_ns"].astype(np.float64) / 1e9
        if receiver_index is None:
            vals, cnts = np.unique(rx, return_counts=True)
            receiver_index = int(vals[np.argmax(cnts)])
        csi, ts = [], []
        for i in range(len(off) - 1):
            if int(rx[i]) != receiver_index:
                continue
            line = bytes(raw[off[i]:off[i + 1]]).decode("utf-8", "replace")
            v = parse_csi_line(line)
            if v is not None:
                csi.append(v)
                ts.append(ts_all[i])
        if not csi:
            return None, None
        ts = np.asarray(ts)
        order = np.argsort(ts, kind="stable")
        return np.asarray(csi)[order], ts[order]

    # minutes/: pick the CSV covering the minute best
    files = sorted(rec.folder.glob("wifi_csi_*.csv"))
    if not files:
        return None, None
    if receiver_index is not None:
        if receiver_index >= len(files):
            return None, None
        path = files[receiver_index]
    else:
        t0, t1 = rec.start_ts - 5.0, rec.end_ts + 5.0
        path = max(files, key=lambda f: _csi_csv_coverage(f, t0, t1))
    csi, ts = _parse_csi_csv(path)
    if not csi:
        return None, None
    ts = np.asarray(ts)
    order = np.argsort(ts, kind="stable")
    return np.asarray(csi)[order], ts[order]
