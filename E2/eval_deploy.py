"""Evaluate deploy/*.pt TorchScript models on the Pi test_minutes cache.

Ground truth: the 1:10 AM boundary (common.TEST_BOUNDARY_FOLDER) — minutes
before it are occupied, at/after are empty. Reports, per model:

  * window accuracy / macro-F1 at the archive's `threshold`
    (the "chunk level" gate — each 50-frame window is one chunk-scale
    decision),
  * partial-minute accuracy: the on-device partial_minute protocol —
    after each new window completes, top-2 over the windows seen so far
    is compared against 0.5 (the runtime's argmax rule),
  * minute accuracy / macro-F1: top-2 aggregation over all windows,
    decision at 0.5 (runtime rule) and at `minute_threshold`.

Usage: python E2/eval_deploy.py [--glob "deploy/*.pt"]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import train as T

ROOT = C.ROOT


def load_ts_model(path):
    extra = {"meta.json": ""}
    model = torch.jit.load(str(path), map_location="cpu",
                           _extra_files=extra)
    model.eval()
    meta = json.loads(extra["meta.json"]) if extra["meta.json"] else {}
    return model, meta


def norm_block(radar, csi, valid, norm):
    """Apply archive normalization to raw cached windows -> float32."""
    r = radar.astype(np.float32)
    rm = np.asarray(norm["radar_mean"], np.float32)[None, None, :, None, None]
    rs = np.asarray(norm["radar_std"], np.float32)[None, None, :, None, None]
    r = (r - rm) / rs
    c = np.log1p(csi.astype(np.float32))
    cm = np.asarray(norm["csi_mean"], np.float32)[None, None, :]
    cs = np.asarray(norm["csi_std"], np.float32)[None, None, :]
    c = (c - cm) / cs
    c[~valid] = 0.0
    return r, c


def minute_probs(probs):
    """top-2 aggregated minute probability per recording."""
    out = []
    for p in probs:
        pm = np.sort(p)
        out.append(float(pm[-2:].mean()) if len(pm) >= 2 else float(pm.mean()))
    return np.asarray(out)


def partial_minute_preds(p, thr=0.5):
    """On-device partial_minute decisions for one recording: after window
    k completes, top-2 over windows[0..k] >= thr -> occupied."""
    preds = []
    for k in range(1, len(p) + 1):
        pool = np.sort(p[:k])
        mp = pool[-2:].mean() if pool.size >= 2 else pool.mean()
        preds.append(int(mp >= thr))
    return np.asarray(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "deploy" / "*.pt"))
    args = ap.parse_args()

    index = T.load_index()
    test = index[index.split == "test_minutes"].sort_values("rec_id")
    print(f"test_minutes: {len(test)} cached recordings "
          f"({int((test.label == 1).sum())} occupied / "
          f"{int((test.label == 0).sum())} empty)")

    # raw cached windows per recording
    recs = []
    for _, row in test.iterrows():
        d = np.load(row["path"])
        recs.append({
            "rec_id": row["rec_id"], "y": int(row["label"]),
            "radar": d["radar"], "csi": d["csi"],
            "csi_valid": d["csi_valid"].astype(bool),
        })
    y_true = np.asarray([r["y"] for r in recs])

    summary = {}
    for pt in sorted(glob.glob(args.glob)):
        name = Path(pt).name
        try:
            model, meta = load_ts_model(pt)
        except Exception as e:
            print(f"{name}: load failed: {e}")
            continue
        modality = meta.get("modality", "radar")
        norm = meta.get("norm") or {}
        thr = float(meta.get("threshold", 0.5))
        mthr = float(meta.get("minute_threshold", thr))

        win_probs, win_y, win_rec = [], [], []
        minute_p = []
        for i, r in enumerate(recs):
            radar, csi = norm_block(r["radar"], r["csi"], r["csi_valid"], norm)
            with torch.inference_mode():
                rt = torch.from_numpy(radar)
                if modality == "fusion":
                    out = model(rt, torch.from_numpy(csi))
                elif modality == "radar":
                    out = model(rt)
                else:
                    out = model(torch.from_numpy(csi))
            p = torch.sigmoid(out.reshape(-1)).numpy()
            win_probs.append(p)
            win_y += [r["y"]] * len(p)
            win_rec += [i] * len(p)
            minute_p.append(p)

        win_y = np.asarray(win_y)
        all_p = np.concatenate(win_probs) if win_probs else np.array([])
        win = T.compute_metrics(win_y, all_p, thr)

        # partial-minute (chunk-scale) decisions
        pm_pred, pm_true = [], []
        for i, r in enumerate(recs):
            preds = partial_minute_preds(minute_p[i], 0.5)
            pm_pred.extend(preds)
            pm_true.extend([r["y"]] * len(preds))
        pm = T.compute_metrics(pm_true, pm_pred, 0.5)

        mp = minute_probs(minute_p)
        minute_rt = T.compute_metrics(y_true, mp, 0.5)     # runtime rule
        minute_thr = T.compute_metrics(y_true, mp, mthr)   # tuned threshold

        summary[name] = {
            "modality": modality, "threshold": thr, "minute_threshold": mthr,
            "window": {k: win[k] for k in ("accuracy", "macro_f1", "n")},
            "partial_minute": {k: pm[k] for k in ("accuracy", "macro_f1", "n")},
            "minute@0.5": {k: minute_rt[k] for k in ("accuracy", "macro_f1", "n")},
            "minute@tuned": {k: minute_thr[k] for k in ("accuracy", "macro_f1", "n")},
        }
        print(f"{name:42s} win={win['accuracy']:.3f} "
              f"partial={pm['accuracy']:.3f} "
              f"min@0.5={minute_rt['accuracy']:.3f} "
              f"min@{mthr:.2f}={minute_thr['accuracy']:.3f}", flush=True)

    out_path = C.OUTPUT_DIR / "deploy_eval_test_minutes.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
