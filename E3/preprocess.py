"""Preprocessing + window construction for the E3 left/right pipeline.

Radar: raw 12-bit ADC frames (64 chirps x 128 samples x 3 RX) are turned
into two maps per frame via FFT, matching the convention used by
`E2/preprocess.py` (same map definitions, so this is a proven approach on
this dataset/hardware):

  * range-Doppler  (64 doppler x 64 range)  -- channel 0
  * range-azimuth  (AZ_BINS azimuth x 64 range) -- channel 1, zero-padded
    angle FFT across the 3 RX antennas (RX1, RX2, RX3 in ADC-channel
    order). AZ_BINS is larger than E2's (32 vs 16) because azimuth is the
    PRIMARY discriminative axis for this left/right task (E2 only needed
    occupancy, where angle resolution doesn't matter).

Both maps are log1p-compressed and bilinearly resized to MAP_SIZE x
MAP_SIZE. Unlike E2 (which only needs temporal statistics of the maps,
since occupancy must transfer across an unseen room), here the raw
per-frame maps are cached directly -- absolute azimuth position is exactly
the signal we want the model to learn, and there is only one room/one
radar placement in this dataset.

Windows: non-overlapping groups of WINDOW_FRAMES consecutive radar frames
inside one recording. Windows never cross recording boundaries.

Each recording is cached to E3/cache/<name>.npz (incremental; existing
files are skipped).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

WINDOW_FRAMES = 30  # ~3 s at the nominal ~10 Hz frame rate
MAP_SIZE = 32
RANGE_BINS = 64      # keep first 64 of 128 range FFT bins
AZ_BINS = 32         # zero-padded angle FFT over the 3 RX antennas


def read_uint12_batch(blob):
    """Vectorized uint12 decode: bytes -> float32 array of samples."""
    d = np.frombuffer(blob, dtype=np.uint8)
    trip = d.reshape(-1, 3).astype(np.uint16)
    a, b, c = trip[:, 0], trip[:, 1], trip[:, 2]
    x = np.empty(a.shape[0] * 2, dtype=np.float32)
    x[0::2] = (a << 4) + (b >> 4)
    x[1::2] = ((b % 16) << 8) + c
    return x


_HANN_R = np.hanning(C.NUM_SAMPLES).astype(np.float32)
_HANN_D = np.hanning(C.NUM_CHIRPS).astype(np.float32)


def frames_to_maps(payloads):
    """payloads: list of raw ADC byte strings (NUM_CHIRPS*NUM_SAMPLES*
    NUM_ANTENNAS*3//2 bytes each, header already stripped).
    Returns float16 array (N, 2, MAP_SIZE, MAP_SIZE), log1p compressed."""
    from scipy import fft as sfft
    n = len(payloads)
    adc = read_uint12_batch(b"".join(payloads)).reshape(
        n, C.NUM_CHIRPS, C.NUM_SAMPLES, C.NUM_ANTENNAS
    )
    adc *= _HANN_R[None, None, :, None] * _HANN_D[None, :, None, None]
    R = sfft.fft(adc, axis=2, workers=-1)                        # range
    RD = sfft.fftshift(sfft.fft(R, axis=1, workers=-1), axes=1)  # doppler
    rd = np.abs(RD[:, :, :RANGE_BINS, :]).mean(axis=3)            # (N,64,64)
    RA = sfft.fft(R[:, :, :RANGE_BINS, :], n=AZ_BINS, axis=3, workers=-1)
    RA = sfft.fftshift(RA, axes=3)
    ra = np.abs(RA).mean(axis=1).transpose(0, 2, 1)               # (N,AZ,64)
    s = MAP_SIZE
    rd = zoom(np.log1p(rd), (1, s / rd.shape[1], s / rd.shape[2]), order=1)
    ra = zoom(np.log1p(ra), (1, s / ra.shape[1], s / ra.shape[2]), order=1)
    return np.stack([rd, ra], axis=1).astype(np.float16)  # (N,2,s,s)


def build_windows(rec):
    """Build all non-overlapping WINDOW_FRAMES-radar-frame windows for one
    recording. Returns dict with radar (nW,WINDOW_FRAMES,2,S,S) float16,
    win_t0 (nW,), win_dur (nW,), n_radar_frames."""
    payloads, ts = C.recording_radar_frames(rec)
    n = len(payloads)
    if n < WINDOW_FRAMES:
        return None
    maps = frames_to_maps(payloads)  # (n,2,S,S)
    n_win = n // WINDOW_FRAMES
    radar_w = maps[: n_win * WINDOW_FRAMES].reshape(
        n_win, WINDOW_FRAMES, 2, MAP_SIZE, MAP_SIZE
    )
    win_t0 = ts[: n_win * WINDOW_FRAMES: WINDOW_FRAMES].copy()
    win_dur = np.empty(n_win, dtype=np.float32)
    for k in range(n_win):
        win_dur[k] = ts[k * WINDOW_FRAMES + WINDOW_FRAMES - 1] - ts[k * WINDOW_FRAMES]
    return {
        "radar": radar_w,
        "win_t0": win_t0,
        "win_dur": win_dur,
        "n_radar_frames": n,
    }


def cache_path(rec):
    return C.CACHE_DIR / f"{rec.folder.name}.npz"


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
        radar=w["radar"],
        win_t0=w["win_t0"],
        win_dur=w["win_dur"],
        meta=json.dumps({
            "rec_id": rec.rec_id,
            "label": rec.label,
            "storage_format": rec.storage_format,
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
                    _worker, [(r, force) for r in todo], chunksize=2)):
                if res.startswith("error"):
                    errs.append(f"{rid}: {res}")
                else:
                    stats[res] = stats.get(res, 0) + 1
                if (i + 1) % 10 == 0:
                    print(f"[{i+1}/{len(todo)}] {time.time()-t0:.0f}s stats={stats}", flush=True)
    else:
        for i, r in enumerate(todo):
            res = process_recording(r, force=force)
            if res.startswith("error"):
                errs.append(f"{r.rec_id}: {res}")
            else:
                stats[res] = stats.get(res, 0) + 1
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(todo)}] {time.time()-t0:.0f}s stats={stats}", flush=True)
    print("DONE", stats, f"errors={len(errs)}", flush=True)
    for e in errs[:30]:
        print("  ", e)
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
