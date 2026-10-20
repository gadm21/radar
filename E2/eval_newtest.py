"""Evaluate the whispy occ-rf-hier model on the newtest captures.

Replays each capture's csi/radar JSONL through the same code path the
model uses live (hier.build_seconds -> window_features -> bundle
predict_proba), then scores 15-s windows and minutes (mean of 4 window
probabilities, paper protocol) against the capture's ground truth.

Ground truth: occupied1/occupied2 -> occupied(1), empty -> empty(0).
capture3 has no declared occupancy label -> predictions reported
without accuracy.

Run:  python E2/eval_newtest.py
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEWTEST = ROOT / "newtest"
sys.path.insert(0, str(ROOT.parent / "thothcraft" / "whispy" / "packages"
                       / "whispy-model-occ"))
sys.path.insert(0, str(ROOT.parent / "thothcraft" / "whispy" / "packages"
                       / "whispy"))
from whispy_model_occ import hier  # noqa: E402

CAPS = [
    ("occupied1_b696fd860ca3", "b696fd860ca3", 1),
    ("occupied2_be490a59db9d", "be490a59db9d", 1),
    ("empty_2fa6d3f1bf90", "2fa6d3f1bf90", 0),
    ("capture3_768c3d688716", "768c3d688716", None),
]


def load_samples(path, ptype=None):
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if ptype and d.get("payload_type") != ptype:
                continue
            out.append((float(d["timestamp"]), d["payload"]))
    return out


def eval_capture(cap_dir: Path, model):
    man = json.loads((cap_dir / "manifest.json").read_text())
    t0 = float(man["started_at"])
    dur = float(man.get("stopped_at") or 0) - t0
    csi_path = cap_dir / "csi-c50e.jsonl"
    rad_path = cap_dir / "radar-a316.jsonl"
    nsec = int(np.ceil(dur))
    csi_samples = (load_samples(csi_path, "csi_raw")
                   if csi_path.exists() else [])
    rad_samples = load_samples(rad_path) if rad_path.exists() else []
    grid = hier.build_seconds(csi_samples, rad_samples, t0, nsec)
    m = model._load()
    rows = []
    for start in range(0, nsec - hier.WIN + 1, hier.WIN):
        feats = hier.window_features(grid, nsec, m["level1"],
                                     m["hierarchy"], start)
        p_csi = model._proba_view(feats, "csi")
        p_rad = model._proba_view(feats, "radar")
        p_early = model._proba_view(feats, "early")
        rows.append({
            "win": start // hier.WIN,
            "minute": start // 60,
            "p_csi": p_csi, "p_rad": p_rad, "p_early": p_early,
            "p_late": (p_csi + p_rad) / 2,
        })
    return rows, {"t0": t0, "dur": dur, "n_csi": len(csi_samples),
                  "n_rad": len(rad_samples),
                  "csi_secs": len(grid.csi_secs),
                  "rad_secs": len(grid.rd_n)}


def minute_probs(rows, key):
    by_min = defaultdict(list)
    for r in rows:
        by_min[r["minute"]].append(r[key])
    return {mn: float(np.mean(ps)) for mn, ps in by_min.items()}


def acc(rows_key_probs, truth, thr):
    if truth is None:
        return None
    y = np.array(list(rows_key_probs.keys()) * 0 + [truth] * len(rows_key_probs))
    pred = (np.array(list(rows_key_probs.values())) >= thr).astype(int)
    return float((pred == y).mean()), int(pred.sum()), len(pred)


def main():
    model = hier.OccHierModel()
    model._load()
    thrs = model._models["thresholds"]
    thr_late = thrs["late"]["mean"]
    report = {"thresholds": {k: round(v["mean"], 4)
                             for k, v in thrs.items()}}
    all_min, all_win = [], []
    t_start = time.time()
    for name, cid, truth in CAPS:
        cap_dir = NEWTEST / name / cid
        rows, info = eval_capture(cap_dir, model)
        mins = {k: minute_probs(rows, k)
                for k in ("p_csi", "p_rad", "p_late", "p_early")}
        crep = {"cap": name, "truth": truth, **info,
                "windows": len(rows), "minutes": len(mins["p_late"])}
        for view, key in (("late", "p_late"), ("radar", "p_rad"),
                          ("csi", "p_csi"), ("early", "p_early")):
            mp = mins[key]
            res = acc(mp, truth, thr_late)
            wacc = None
            if truth is not None:
                wp = np.array([r[key] for r in rows])
                wacc = float(((wp >= thr_late).astype(int) == truth).mean())
            crep[view] = {"min_acc": res[0] if res else None,
                          "n_occ_pred": res[1] if res else
                          int(sum(v >= thr_late
                                  for v in mp.values())),
                          "win_acc": wacc,
                          "mean_p": float(np.mean(list(mp.values())))}
        all_min.append((name, truth, mins["p_late"]))
        all_win += [(name, truth, r["p_late"]) for r in rows]
        report[name] = crep
        print(f"{name}: {len(rows)} windows, {len(mins['p_late'])} "
              f"min | late min_acc={crep['late']['min_acc']} "
              f"win_acc={crep['late']['win_acc']} "
              f"(csi {info['csi_secs']}s / rad {info['rad_secs']}s)",
              flush=True)
    # pooled minute accuracy over labeled captures
    truth_v, prob_v = [], []
    for name, truth, mp in all_min:
        if truth is None:
            continue
        truth_v += [truth] * len(mp)
        prob_v += list(mp.values())
    y = np.array(truth_v)
    p = np.array(prob_v)
    overall = {"labeled_minutes": int(len(y)),
               "minute_accuracy": float(((p >= thr_late) == y).mean()),
               "balanced_accuracy": float(
                   ((p >= thr_late)[y == 1].mean()
                    + (p < thr_late)[y == 0].mean()) / 2)
               if len(np.unique(y)) == 2 else None,
               "n_occ_pred": int((p >= thr_late).sum())}
    report["overall_late"] = overall
    report["elapsed_s"] = round(time.time() - t_start, 1)
    out = NEWTEST / "eval_newtest_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(overall, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
