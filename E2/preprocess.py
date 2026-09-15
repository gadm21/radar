"""Preprocessing + window construction for the E2 occupancy pipeline.

Radar: raw 12-bit ADC frames (64 chirps x 128 samples x 3 RX) are turned
into two maps per frame via FFT:
  * range-Doppler  (64 doppler x 64 range)  -- channel 0
  * range-azimuth  (16 azimuth x 64 range)  -- channel 1
Both are log1p-compressed and bilinearly resized to MAP_SIZE x MAP_SIZE.

CSI: complex CSI (52 usable subcarriers) -> amplitude. For each radar
window [t0, t4] the CSI samples whose timestamps fall inside the window
are linearly interpolated onto a fixed CSI_STEPS-point uniform grid
(documented resampling strategy; the true sample count is stored and a
window with <2 samples is flagged csi_valid=0, never fabricated).

Windows: non-overlapping groups of WINDOW_FRAMES=50 consecutive radar
frames inside one recording (~5-7 s — long enough to capture breathing
rate, which 0.5 s windows cannot). Windows never cross recording / session /
placement / label boundaries because they are built per recording.

Each recording is cached to E2/cache/<source>/<name>.npz (incremental;
existing files are skipped).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

WINDOW_FRAMES = 50
MAP_SIZE = 24
CSI_STEPS = 128
N_SUBCARRIERS = 52
RANGE_BINS = 64      # keep first 64 of 128 range FFT bins
AZ_BINS = 16         # zero-padded angle FFT over 3 RX


# ---------------------------------------------------------------------------
# Radar FFT maps
# ---------------------------------------------------------------------------
def read_uint12_batch(blob):
    """Vectorized uint12 decode: bytes -> float32 array of samples."""
    d = np.frombuffer(blob, dtype=np.uint8)
    trip = d.reshape(-1, 3).astype(np.uint16)
    a, b, c = trip[:, 0], trip[:, 1], trip[:, 2]
    x = np.empty(a.shape[0] * 2, dtype=np.float32)
    x[0::2] = (a << 4) + (b >> 4)
    x[1::2] = ((b % 16) << 8) + c
    return x


_HANN_R = np.hanning(128).astype(np.float32)
_HANN_D = np.hanning(64).astype(np.float32)


def frames_to_maps(payloads):
    """payloads: list of 36864-byte frame payloads.
    Returns float16 array (N, 2, MAP_SIZE, MAP_SIZE), log1p compressed."""
    from scipy import fft as sfft
    n = len(payloads)
    adc = read_uint12_batch(b"".join(payloads)).reshape(n, 64, 128, 3)
    adc *= _HANN_R[None, None, :, None] * _HANN_D[None, :, None, None]
    R = sfft.fft(adc, axis=2, workers=-1)                  # range
    RD = sfft.fftshift(sfft.fft(R, axis=1, workers=-1), axes=1)  # doppler
    rd = np.abs(RD[:, :, :RANGE_BINS, :]).mean(axis=3)     # (N,64,64)
    RA = sfft.fft(R[:, :, :RANGE_BINS, :], n=AZ_BINS, axis=3, workers=-1)
    ra = np.abs(RA).mean(axis=1).transpose(0, 2, 1)        # (N,16,64)
    s = MAP_SIZE
    rd = zoom(np.log1p(rd), (1, s / rd.shape[1], s / rd.shape[2]), order=1)
    ra = zoom(np.log1p(ra), (1, s / ra.shape[1], s / ra.shape[2]), order=1)
    return np.stack([rd, ra], axis=1).astype(np.float16)  # (N,2,s,s)


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------
def build_windows(rec):
    """Build all non-overlapping 50-radar-frame windows for one recording.

    Returns dict with radar (nW,50,2,S,S) float16, csi (nW,CSI_STEPS,52)
    float32, csi_valid (nW,), csi_count (nW,), win_dur (nW,), win_t0 (nW,).
    """
    payloads, rts = C.recording_radar_frames(rec)
    n = len(payloads)
    if n < WINDOW_FRAMES:
        return None
    csi, cts = C.recording_csi(rec)
    csi_amp = np.abs(csi).astype(np.float32) if csi is not None else None

    maps = frames_to_maps(payloads)  # (n,2,S,S)
    n_win = n // WINDOW_FRAMES
    radar_w = maps[: n_win * WINDOW_FRAMES].reshape(
        n_win, WINDOW_FRAMES, 2, MAP_SIZE, MAP_SIZE)

    win_t0 = rts[: n_win * WINDOW_FRAMES:WINDOW_FRAMES].copy()
    win_dur = np.empty(n_win, dtype=np.float32)
    csi_w = np.zeros((n_win, CSI_STEPS, N_SUBCARRIERS), dtype=np.float32)
    csi_cnt = np.zeros(n_win, dtype=np.int32)
    for k in range(n_win):
        t0 = rts[k * WINDOW_FRAMES]
        t1 = rts[k * WINDOW_FRAMES + WINDOW_FRAMES - 1]
        win_dur[k] = t1 - t0
        if csi_amp is None or t1 <= t0:
            continue
        m = (cts >= t0) & (cts <= t1)
        cnt = int(m.sum())
        csi_cnt[k] = cnt
        if cnt == 0:
            continue
        grid = np.linspace(t0, t1, CSI_STEPS)
        if cnt == 1:
            csi_w[k] = np.repeat(csi_amp[m], CSI_STEPS, axis=0)
        else:
            ts_sel = cts[m]
            for j in range(N_SUBCARRIERS):
                csi_w[k, :, j] = np.interp(grid, ts_sel, csi_amp[m, j])
    return {
        "radar": radar_w,
        "csi": csi_w,
        "csi_valid": (csi_cnt >= 2).astype(np.int8),
        "csi_count": csi_cnt,
        "win_dur": win_dur,
        "win_t0": win_t0,
        "n_radar_frames": n,
    }


def cache_path(rec):
    d = C.CACHE_DIR / rec.source
    return d / f"{rec.folder.name}.npz"


def process_recording(rec, force=False):
    out = cache_path(rec)
    if out.exists() and not force:
        return "cached"
    try:
        w = build_windows(rec)
    except Exception as e:
        return f"error: {e}"
    if w is None or w["radar"].shape[0] == 0:
        return "no-windows"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        radar=w["radar"], csi=w["csi"], csi_valid=w["csi_valid"],
        csi_count=w["csi_count"], win_dur=w["win_dur"], win_t0=w["win_t0"],
        meta=json.dumps({
            "rec_id": rec.rec_id, "placement": rec.placement,
            "label": rec.label, "session_id": rec.session_id,
            "n_radar_frames": int(w["n_radar_frames"]),
        }),
    )
    return "ok"


def _worker(task):
    rec, force = task
    try:
        return rec.rec_id, process_recording(rec, force=force)
    except Exception as e:
        return rec.rec_id, f"error: {e}"


def run_all(limit=None, force=False, workers=1):
    recs, problems = C.discover_recordings()
    recs = C.assign_sessions(recs)
    recs = [r for r in recs if r.status == "ok"]
    if limit:
        recs = recs[:limit]
    todo = [r for r in recs if force or not cache_path(r).exists()]
    print(f"{len(recs)} usable recordings, {len(todo)} to process", flush=True)
    stats = {"ok": 0, "cached": len(recs) - len(todo), "no-windows": 0}
    errs = []
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, (rid, res) in enumerate(pool.imap_unordered(
                    _worker, [(r, force) for r in todo], chunksize=4)):
                if res.startswith("error"):
                    errs.append(f"{rid}: {res}")
                else:
                    stats[res] = stats.get(res, 0) + 1
                if (i + 1) % 100 == 0:
                    print(f"[{i+1}/{len(todo)}] {time.time()-t0:.0f}s "
                          f"stats={stats}", flush=True)
    else:
        for i, r in enumerate(todo):
            res = process_recording(r, force=force)
            if res.startswith("error"):
                errs.append(f"{r.rec_id}: {res}")
            else:
                stats[res] = stats.get(res, 0) + 1
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(todo)}] {time.time()-t0:.0f}s "
                      f"stats={stats}", flush=True)
    print("DONE", stats, f"errors={len(errs)}", flush=True)
    for e in errs[:30]:
        print("  ", e)
    (C.OUTPUT_DIR).mkdir(exist_ok=True)
    with open(C.OUTPUT_DIR / "preprocess_errors.json", "w") as fh:
        json.dump(errs, fh, indent=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    run_all(limit=a.limit, force=a.force, workers=a.workers)
