"""icc_multimodal2 frozen_v3 models evaluated on the newtest dataset.

Protocol (per user):
  * weights: frozen_v3 trained models (unchanged)
  * validation = old validation minutes + old multilink test minutes
  * thresholds calibrated on that validation data ONLY
  * newtest = new test dataset -> report window and minute accuracy

A "window" is a 15-s hierarchical-PCA window; a "minute" is the mean of
its 4 window probabilities (non-overlapping windows, complete minutes
only). Late fusion = mean of the same model's csi + radar probabilities.
Decision: prob >= threshold, where the threshold maximises balanced
accuracy on the (new) validation minutes, ties broken toward 0.5 —
the same fit_threshold rule used for the frozen artifacts.

Run: python E2/eval_newtest_v2.py
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NEWTEST = ROOT / "newtest"
FROZEN = ROOT / "E2" / "outputs" / "frozen_v3"
sys.path.insert(0, str(ROOT.parent / "thothcraft" / "whispy" / "packages"
                       / "whispy-model-occ"))
sys.path.insert(0, str(ROOT.parent / "thothcraft" / "whispy" / "packages"
                       / "whispy"))
from whispy_model_occ import hier  # noqa: E402

MODELS = ["RF", "GBM", "MLP", "SVM"]
VIEWS = ["CSI", "radar", "early_fusion"]          # late derived

CAPS = [
    ("empty_2fa6d3f1bf90", "2fa6d3f1bf90", 0),
    ("empty_whispy", None, 0),
    ("occupied_whispy", None, 1),
    ("occupied_whispy2", None, 1),
    ("occupied_whispy11", None, 1),
    ("occupied_whispy12", None, 1),
    ("capture3_768c3d688716", "768c3d688716", None),
]


def load_models():
    out = {m: {v: [] for v in VIEWS} for m in MODELS}
    for m in MODELS:
        for v in VIEWS:
            for s in range(5):
                p = FROZEN / "models" / f"{m}__{v}__s{s}.joblib"
                out[m][v].append(joblib.load(p))
    return out


def win_proba(objs, x):
    """Seed-mean window probability for one model/view."""
    ps = []
    for obj in objs:
        xi = np.asarray(x, float)
        xi[~np.isfinite(xi)] = np.nan
        xi = np.where(np.isnan(xi), obj["fill"], xi)
        z = obj["scaler"].transform(xi[None])
        ps.append(float(obj["model"].predict_proba(z)[0, 1]))
    return float(np.mean(ps))


def fit_threshold(probs, labels):
    """Max balanced-accuracy threshold, ties toward 0.5."""
    probs = np.asarray(probs, float)
    y = np.asarray(labels, int)
    best = (-1.0, -1.0, 0.5)
    for t in np.unique(probs):
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        tpr = tp / max(tp + int(((pred == 0) & (y == 1)).sum()), 1)
        tnr = tn / max(tn + int(((pred == 1) & (y == 0)).sum()), 1)
        score = ((tpr + tnr) / 2, -abs(float(t) - .5), float(t))
        if score > best:
            best = score
    return best[2]


# ---------------------------------------------------------------- validation
def validation_minutes(models):
    """Per (model, view incl. late) -> (minute probs, labels) on
    old validation + old test minutes."""
    vw = pd.read_csv(FROZEN / "validation_windows.csv")
    val = {}                                    # (m,v) -> probs, labels
    for m in MODELS:
        for v in VIEWS:
            cols = models[m][v][0]["columns"]
            sub = vw[cols].to_numpy(float)
            probs = []
            for obj in models[m][v]:
                xi = np.where(np.isnan(sub), obj["fill"], sub)
                z = obj["scaler"].transform(xi)
                probs.append(obj["model"].predict_proba(z)[:, 1])
            vw[f"p_{m}_{v}"] = np.mean(probs, axis=0)
        vw[f"p_{m}_late"] = (vw[f"p_{m}_CSI"] + vw[f"p_{m}_radar"]) / 2
    for m in MODELS:
        for v in ["CSI", "radar", "early_fusion", "late"]:
            g = vw.groupby(["unit", "minute"]).agg(
                p=(f"p_{m}_{v}", "mean"), y=("label", "first"))
            val[(m, v)] = (g["p"].to_numpy(), g["y"].to_numpy())
    # fold in the old multilink test minutes (selected-model CSVs)
    for f in FROZEN.glob("test_*_probabilities.csv"):
        stem = f.name[len("test_"):-len("_probabilities.csv")]
        mname, vname = stem.split("__", 1)
        vkey = {"CSI": "CSI", "radar": "radar",
                "early fusion": "early_fusion",
                "late probability fusion": "late"}[vname]
        df = pd.read_csv(f)
        old_p, old_y = val.get((mname, vkey),
                               (np.array([]), np.array([])))
        val[(mname, vkey)] = (
            np.concatenate([old_p, df["probability"].to_numpy()]),
            np.concatenate([old_y, df["label"].to_numpy()]))
        print(f"  + {f.name}: {len(df)} old-test minutes -> "
              f"validation[{mname}|{vkey}]")
    return val


# ------------------------------------------------------------------- newtest
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


def newtest_windows(cap_dir, models, m_bundle):
    """Per (model, view incl late) window probs + minute index."""
    man = json.loads((cap_dir / "manifest.json").read_text())
    t0 = float(man["started_at"])
    dur = float(man.get("stopped_at") or man.get("duration_s", 0)) - t0 \
        if "stopped_at" in man else float(man["duration_s"])
    nsec = int(np.ceil(dur))
    csi = load_samples(cap_dir / "csi-c50e.jsonl", "csi_raw")
    rad = load_samples(cap_dir / "radar-a316.jsonl")
    grid = hier.build_seconds(csi, rad, t0, nsec)
    lvl1, hie = m_bundle["level1"], m_bundle["hierarchy"]
    rows = []
    for start in range(0, nsec - hier.WIN + 1, hier.WIN):
        feats = hier.window_features(grid, nsec, lvl1, hie, start)
        row = {"minute": start // 60}
        for m in MODELS:
            p = {v: win_proba(models[m][v], feats[v_key])
                 for v, v_key in (("CSI", "csi"), ("radar", "radar"),
                                  ("early_fusion", "early"))}
            p["late"] = (p["CSI"] + p["radar"]) / 2
            for v in p:
                row[f"p_{m}_{v}"] = p[v]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    t_start = time.time()
    models = load_models()
    import whispy_model_occ.hier as _h
    mdl = _h.OccHierModel()
    m_bundle = mdl._load()          # level1 pcas + hierarchy for feats

    print("building validation minute probs ...", flush=True)
    val = validation_minutes(models)

    print("scoring newtest captures ...", flush=True)
    caps = {}
    for name, cid, truth in CAPS:
        cap_dir = NEWTEST / name if cid is None else NEWTEST / name / cid
        df = newtest_windows(cap_dir, models, m_bundle)
        caps[name] = (df, truth)
        print(f"  {name}: {len(df)} windows, truth={truth}", flush=True)

    report = {"protocol": {
        "validation": "old validation minutes + old multilink test minutes",
        "test": "newtest dataset", "window_seconds": 15,
        "windows_per_minute": 4}}
    table = []
    for m in MODELS:
        for v in ["CSI", "radar", "early_fusion", "late"]:
            vp, vy = val[(m, v)]
            thr = fit_threshold(vp, vy)
            win_p, win_y, min_p, min_y = [], [], [], []
            per_cap = {}
            for name, (df, truth) in caps.items():
                p = df[f"p_{m}_{v}"].to_numpy()
                g = df.groupby("minute")[f"p_{m}_{v}"].mean()
                g4 = df.groupby("minute").size()
                full = g[g4[g4 == 4].index]
                crep = {"minutes": int(len(full)),
                        "occ_minutes_pred": int((full >= thr).sum()),
                        "mean_p": float(p.mean())}
                if truth is not None:
                    crep["window_acc"] = float(
                        ((p >= thr).astype(int) == truth).mean())
                    crep["minute_acc"] = float(
                        ((full >= thr).astype(int) == truth).mean())
                    win_p += list(p); win_y += [truth] * len(p)
                    min_p += list(full); min_y += [truth] * len(full)
                per_cap[name] = crep
            wacc = float((np.array(win_p) >= thr).astype(int)
                         .__eq__(np.array(win_y)).mean())
            my = np.array(min_y); mp = np.array(min_p)
            macc = float(((mp >= thr).astype(int) == my).mean())
            bacc = float((((mp >= thr)[my == 1]).mean()
                          + ((mp < thr)[my == 0]).mean()) / 2) \
                if len(np.unique(my)) == 2 else None
            rec = {"model": m, "view": v, "threshold": round(thr, 4),
                   "val_minutes": int(len(vy)),
                   "window_acc": wacc, "minute_acc": macc,
                   "minute_balanced_acc": bacc,
                   "n_windows": len(win_p), "n_minutes": len(min_p),
                   "per_capture": per_cap}
            table.append(rec)
            print(f"{m:4s} | {v:13s} | thr {thr:.3f} | "
                  f"win {wacc:.3f} | min {macc:.3f} "
                  f"(bal {bacc if bacc is None else round(bacc,3)})",
                  flush=True)
    report["results"] = table
    report["elapsed_s"] = round(time.time() - t_start, 1)
    out = NEWTEST / "eval_newtest_v2_report.json"
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
