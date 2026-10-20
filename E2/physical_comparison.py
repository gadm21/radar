"""Leakage-resistant physical-descriptor occupancy comparison.

Uses E1's shared physical feature extraction, not the per-cohort PCA or
cluster coordinates in historical model_compare. Fits supervised models on
legacy train minutes, validates on placement-disjoint legacy minutes, and
tests on four labeled multilink captures. Test features/probabilities may be
rescaled or calibrated using unlabeled test-batch statistics only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "E2/outputs/revision"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figs"
FIG.mkdir(exist_ok=True)
P = ROOT / "E1/outputs"

RAD = ["snr_mean", "snr_max", "rd_mean", "rd_p90", "rd_std90", "rd_delta",
       "rd_peak", "ra_mean", "ra_p90", "ra_std90", "ra_delta", "ra_peak",
       "re_mean", "re_p90", "re_std90", "re_delta", "re_peak",
       "xy_mean", "xy_p90", "xy_std90", "xy_delta", "xy_peak"]
CSI = ["csi_amp_mean", "csi_amp_std", "csi_amp_q90", "csi_rv_mean",
       "csi_rv_q90", "csi_rv_max", "csi_tv", "csi_dop_frac"]
STABLE = ["ra_std90", "re_std90", "xy_std90", "re_delta", "ra_delta", "ra_p90"]
VIEWS = {"CSI": CSI, "radar": RAD, "early fusion": RAD + CSI,
         "stable radar": STABLE}


def load():
    old = pd.read_csv(P / "features_old_5s_ext.csv")
    ml = pd.read_csv(P / "features_multilink_5s.csv")
    # The legacy table calls the held-out validation split 'test'.
    train = old[old.rec_id.str.startswith("train/")].copy()
    val = old[old.rec_id.str.startswith("test_minutes/")].copy()
    test = ml.copy()
    for suffix in [s.removeprefix("csi_") for s in CSI]:
        test[f"csi_{suffix}"] = test[[f"csi1_{suffix}",
                                      f"csi2_{suffix}"]].mean(axis=1)
    for name, d in (("train", train), ("val", val), ("test", test)):
        d["split"] = name
        if name == "test":
            d["unit"] = d.cap.astype(str)
            d["minute"] = (d.t_rel//60).astype(int)
        else:
            d["unit"] = d.rec_id.astype(str)
            d["minute"] = 0
    return train, val, test


def matrix(df, cols, fill):
    x = df[cols].to_numpy(float).copy()
    x[~np.isfinite(x)] = np.nan
    return np.where(np.isnan(x), fill[None], x)


def cal(y, p):
    med = float(np.median(p))
    # Exact global two-means solution in one dimension: minimize within-
    # cluster squared error over every distinct sorted split.
    x = np.sort(np.asarray(p, float))
    n = len(x)
    if n < 2 or x[0] == x[-1]:
        km = med
    else:
        s, q = np.cumsum(x), np.cumsum(x*x)
        count_l = np.arange(1,n)
        count_r = n-count_l
        sum_l, sum_r = s[:-1], s[-1]-s[:-1]
        sq_l, sq_r = q[:-1], q[-1]-q[:-1]
        cost = (sq_l-sum_l**2/count_l + sq_r-sum_r**2/count_r)
        cost[x[:-1] == x[1:]] = np.inf
        i = int(np.argmin(cost))
        km = float((sum_l[i]/count_l[i] + sum_r[i]/count_r[i])/2)
    out = {}
    for name, threshold in (("med", med), ("km", km)):
        pred = (p >= threshold).astype(int)
        out[f"acc_{name}"] = accuracy_score(y, pred)
        out[f"ba_{name}"] = balanced_accuracy_score(y, pred)
        out[f"cm_{name}"] = confusion_matrix(y, pred, labels=[0,1]).tolist()
    return out


def aggregate(df, p):
    g = pd.DataFrame({"unit":df.unit.to_numpy(), "minute":df.minute.to_numpy(),
                      "label":df.label.to_numpy().astype(int), "p":p}).groupby(
                          ["unit", "minute"], sort=True).agg(
                              label=("label", "first"), p=("p", "mean"))
    return g.label.to_numpy(), g.p.to_numpy(), g.index


def score(model, view, scale, split, df, p):
    rows = []
    for level in ("window", "minute"):
        if level == "minute":
            y, pp, ids = aggregate(df, p)
        else:
            y, pp = df.label.to_numpy().astype(int), p
            ids = None
        m = cal(y, pp)
        rows.append({"model": model, "view": view, "scale": scale,
                     "split": split, "level": level, "n": len(y),
                     "acc_raw": accuracy_score(y, pp>=.5),
                     "acc_med": m["acc_med"], "acc_km": m["acc_km"],
                     "ba_med": m["ba_med"], "ba_km": m["ba_km"]})
    return rows


def main():
    train, val, test = load()
    print("windows/minutes:", [(n,len(d),d.unit.nunique()) for n,d in
                                (("train",train),("val",val),("test",test))],
          flush=True)
    results = []
    save_probs = {}
    # Fixed model definitions and train-only imputation/scaling.
    for view, cols in VIEWS.items():
        fill = np.nanmedian(train[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float),axis=0)
        Xtr = matrix(train, cols, fill)
        sc = StandardScaler().fit(Xtr)
        Ztr = sc.transform(Xtr)
        ytr = train.label.to_numpy().astype(int)
        models = {
            "RF": RandomForestClassifier(n_estimators=200, max_depth=12,
                 class_weight="balanced", random_state=42, n_jobs=1),
            "GBM": HistGradientBoostingClassifier(max_iter=200,
                 learning_rate=.07, random_state=42),
            "SVM": SVC(C=4, kernel="rbf", probability=True,
                 class_weight="balanced", random_state=42),
            "MLP": MLPClassifier((64,32), alpha=1e-3, max_iter=400,
                 random_state=42),
        }
        for model_name, model in models.items():
            model.fit(Ztr, ytr)
            for split, df in (("val",val),("test",test)):
                X = matrix(df, cols, fill)
                for scale in ("raw", "rescaled"):
                    # Eval-set transformation uses no labels. The classifier
                    # always sees train-scaled coordinates at fit time.
                    Z = sc.transform(X) if scale == "raw" else StandardScaler().fit_transform(X)
                    p = model.predict_proba(Z)[:,1]
                    results += score(model_name,view,scale,split,df,p)
                    save_probs[(model_name,view,scale,split)] = p
            print(view, model_name, flush=True)
            pd.DataFrame(results).to_csv(OUT / "physical_progress.csv",index=False)
    # Late fusion: same underlying RF experts, mean probabilities. Fixed
    # equal weighting avoids fitting a gate on validation labels.
    for scale in ("raw", "rescaled"):
        for split, df in (("val",val),("test",test)):
            p = (save_probs[("RF","CSI",scale,split)] +
                 save_probs[("RF","radar",scale,split)])/2
            results += score("RF", "late probability fusion", scale, split, df, p)
            save_probs[("RF","late probability fusion",scale,split)] = p
    out = pd.DataFrame(results)
    out.to_csv(OUT / "physical_all.csv", index=False)
    good = []
    for key, g in out.groupby(["model","view","scale","level"]):
        if set(g.split) != {"val","test"}:
            continue
        pair = g.set_index("split")
        if any(pair.loc["val",f"acc_{rule}"]>.70 and
               pair.loc["test",f"acc_{rule}"]>.70 for rule in ("med","km")):
            good.append(g)
    passed = pd.concat(good,ignore_index=True) if good else out.iloc[:0]
    passed.to_csv(OUT / "physical_over70.csv",index=False)
    np.savez_compressed(OUT / "physical_probs.npz",
        **{"|".join(k):v for k,v in save_probs.items()})
    print("passing",len(passed), flush=True)
    print(passed[passed.split.eq("test")].sort_values("acc_km",ascending=False)[
        ["model","view","scale","level","acc_med","acc_km"]].to_string(index=False),
        flush=True)


if __name__ == "__main__":
    main()
