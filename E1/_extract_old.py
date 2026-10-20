"""E1 pass-A extraction for the legacy train/test minutes.

Reuses E2's per-minute products (amp60/sec_valid/dots + maps/snr_f/
rd_sec/sec_has_radar) but caches under E1/cache so the E1 analysis
stage is self-contained. calibration_minutes is intentionally skipped
(1394 folders; its PCA-fit statistics are already captured by
E2/outputs/occupancy/pcas.joblib).

    python E1/_extract_old.py --workers 6
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "E2"))
import common as C            # noqa: E402  E2/common.py
import occupancy_pipeline as OP  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache" / "win"
SOURCES = ("train", "test_minutes")


def extract_minute(rec):
    path = CACHE / rec.source / f"{rec.folder.name}.npz"
    if path.exists():
        return rec.rec_id, "cached"
    try:
        csid = OP._csi_minute(rec)
        radd = OP._radar_minute(rec)
    except Exception as e:
        return rec.rec_id, f"error: {e}"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        amp60=csid["amp60"], sec_valid=csid["sec_valid"],
        dots=csid["dots"], csi_hz=np.float32(csid["csi_hz"]),
        maps=radd["maps"], rd12=radd["rd12"], snr_f=radd["snr_f"],
        rts=radd["rts"], rd_sec=radd["rd_sec"],
        sec_has_radar=radd["sec_has_radar"],
        radar_hz=np.float32(radd["radar_hz"]),
        meta=json.dumps({"rec_id": rec.rec_id, "label": rec.label,
                         "placement": rec.placement,
                         "session_id": rec.session_id}))
    return rec.rec_id, "ok"


def _w(rec):
    try:
        return extract_minute(rec)
    except Exception as e:
        return rec.rec_id, f"error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    recs, probs = C.discover_recordings()
    recs = C.assign_sessions(recs)
    recs = [r for r in recs if r.source in SOURCES and r.status == "ok"]
    if args.limit:
        recs = recs[: args.limit]
    todo = [r for r in recs
            if not (CACHE / r.source / f"{r.folder.name}.npz").exists()]
    print(f"{len(recs)} usable ({SOURCES}), {len(todo)} to extract",
          flush=True)
    t0, ok, errs = time.time(), 0, []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, (rid, res) in enumerate(
                    pool.map(_w, todo, chunksize=2)):
                if res.startswith("error"):
                    errs.append((rid, res))
                else:
                    ok += 1
                if (i + 1) % 50 == 0:
                    print(f"{i + 1}/{len(todo)} "
                          f"({(time.time() - t0) / 60:.1f} min)",
                          flush=True)
    else:
        for i, r in enumerate(todo):
            rid, res = _w(r)
            if res.startswith("error"):
                errs.append((rid, res))
            else:
                ok += 1
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(todo)}", flush=True)
    print(f"done: ok={ok} errors={len(errs)} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    for rid, res in errs[:20]:
        print(" ", rid, res, flush=True)


if __name__ == "__main__":
    main()
