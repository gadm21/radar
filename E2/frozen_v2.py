"""Frozen comparison, revision 2: PCA window descriptors, balanced cohorts,
and multi-seed model fits.

Changes from frozen_evaluation.py
  * Twelve window-level PCA descriptors (mean/std of the three training-basis
    CSI and radar principal-component coordinates over each five-second
    window) join the physical descriptor set in every view.
  * Training and validation cohorts are class-balanced by synthesising
    occupied minutes from real occupied minutes of the same cohort
    (gauss / gain / mixup / temporal perturbations at the window level).
    Test captures are never augmented.
  * Every model is fitted with SEEDS random states; selection uses the mean
    validation-minute AUC and reported numbers are mean +- sd over seeds.
    Seed 0 models provide the LLM reference probabilities.

Run ``--phase select`` first (never opens a multilink file), then
``--phase evaluate``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from physical_comparison import ROOT, RAD, CSI, matrix

OLD = ROOT / "E1/outputs/features_old_5s_ext.csv"
TEST = ROOT / "E1/outputs/features_multilink_5s.csv"
DOTS = ROOT / "E1/outputs/pca_deep"
OUT = ROOT / "E2/outputs/frozen_v2"
OUT.mkdir(parents=True, exist_ok=True)

CSI_PC = [f"csi_pc{j}_{s}" for s in ("mean", "std") for j in (1, 2, 3)]
RAD_PC = [f"rad_pc{j}_{s}" for s in ("mean", "std") for j in (1, 2, 3)]
CSI_ALL = CSI + CSI_PC
RAD_ALL = RAD + RAD_PC
VIEWS = {"CSI": CSI_ALL, "radar": RAD_ALL, "early fusion": RAD_ALL + CSI_ALL}
MODEL_NAMES = ("RF", "GBM", "SVM", "MLP")
SEEDS = tuple(range(5))
AUG_SEED = 20261015
NOISE = ("gauss", "gain", "mixup", "temporal")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_model(name, seed):
    return {
        "RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=12,
            class_weight="balanced", random_state=seed, n_jobs=1),
        "GBM": lambda: HistGradientBoostingClassifier(max_iter=200,
            learning_rate=.07, random_state=seed),
        "SVM": lambda: SVC(C=4, kernel="rbf", probability=True,
            class_weight="balanced", random_state=seed),
        "MLP": lambda: MLPClassifier((64, 32), alpha=1e-3,
            max_iter=400, random_state=seed),
    }[name]()


# ---------------------------------------------------------------- descriptors
def pc_window_descriptors(dots, keys, wins):
    """Mean/std of per-second PC coordinates inside each five-second window.

    dots[key] is a (seconds, 7) array: column 0 is a validity flag, columns
    1-3 CSI PCs, columns 4-6 radar PCs, all on training-fitted bases.
    """
    out = np.full((len(keys), 12), np.nan)
    for i, (key, w) in enumerate(zip(keys, wins)):
        a = dots.get(key)
        if a is None:
            continue
        seg = a[5 * int(w):5 * int(w) + 5]
        good = (seg[:, 0] > .5) & np.isfinite(seg[:, 1:7]).all(axis=1)
        if good.sum() < 2:
            continue
        v = seg[good, 1:7]
        out[i, 0:3] = v[:, 0:3].mean(axis=0); out[i, 3:6] = v[:, 0:3].std(axis=0)
        out[i, 6:9] = v[:, 3:6].mean(axis=0); out[i, 9:12] = v[:, 3:6].std(axis=0)
    return pd.DataFrame(out, columns=CSI_PC + RAD_PC)


def load_dots(name):
    with np.load(DOTS / f"pc_dots_{name}.npz") as z:
        return {k: z[k].copy() for k in z.files}


def read_old():
    d = pd.read_csv(OLD)
    frames = {}
    for split, prefix, dot in (("train", "train/", "train"),
                               ("validation", "test_minutes/", "val")):
        f = d[d.rec_id.str.startswith(prefix)].copy().reset_index(drop=True)
        f["unit"] = f.rec_id.astype(str)
        f["minute"] = 0
        f["aug"] = 0
        pcs = pc_window_descriptors(load_dots(dot), f.unit.tolist(), f.win.tolist())
        frames[split] = pd.concat([f, pcs], axis=1)
    return frames["train"], frames["validation"]


def read_test():
    t = pd.read_csv(TEST)
    for suffix in [s.removeprefix("csi_") for s in CSI]:
        t[f"csi_{suffix}"] = t[[f"csi1_{suffix}", f"csi2_{suffix}"]].mean(axis=1)
    t["unit"] = t.cap.astype(str)
    t["minute"] = (t.t_rel // 60).astype(int)
    t["aug"] = 0
    pcs = pc_window_descriptors(load_dots("testml_trainbasis"),
                                t.unit.tolist(), t.win.tolist())
    return pd.concat([t.reset_index(drop=True), pcs], axis=1)


# --------------------------------------------------------------- augmentation
def perturb(x, pool, sig, rng, kind):
    if kind == "gauss":
        return x + rng.normal(0, .15, x.shape) * sig
    if kind == "gain":
        return x * (1 + rng.normal(0, .08, (1, x.shape[1])) + rng.normal(0, .03, x.shape))
    if kind == "mixup":
        other = pool[rng.integers(len(pool))]
        lam = rng.uniform(.25, .75)
        w = min(len(x), len(other))
        out = x.copy()
        out[:w] = lam * x[:w] + (1 - lam) * other[:w]
        return out
    if kind == "temporal":
        idx = np.sort(rng.integers(0, len(x), len(x)))
        return x[idx] + rng.normal(0, .05, x.shape) * sig
    raise ValueError(kind)


def balance(frame, cols, rng, tag):
    """Synthesise minority-class minutes until both classes have equal minute
    counts. Each synthetic minute derives from one real minute of the same
    cohort and class; NaN descriptors stay NaN."""
    per_class = frame.groupby("label").unit.nunique()
    minority = int(per_class.idxmin())
    n_new = int(per_class.max() - per_class.min())
    real = frame[frame.label == minority]
    units = sorted(real.unit.unique())
    pool = [g.sort_values("win")[cols].to_numpy(float) for _, g in real.groupby("unit")]
    sig = np.nanstd(frame[cols].to_numpy(float), axis=0) + 1e-9
    rows = []
    for i in range(n_new):
        src = units[i % len(units)]
        g = real[real.unit == src].sort_values("win")
        x = g[cols].to_numpy(float)
        kind = NOISE[i % len(NOISE)]
        xa = perturb(x, pool, sig, rng, kind)
        xa[~np.isfinite(x)] = np.nan
        meta = g[["win", "label"]].copy()
        meta["unit"] = f"{src}__{tag}{i:03d}"
        meta["minute"] = 0
        meta["aug"] = 1
        meta["aug_type"] = kind
        meta["source_unit"] = src
        rows.append(pd.concat([meta.reset_index(drop=True),
                               pd.DataFrame(xa, columns=cols)], axis=1))
    frame = frame.copy()
    frame["aug_type"] = "real"
    frame["source_unit"] = frame.unit
    keep = ["unit", "minute", "win", "label", "aug", "aug_type", "source_unit"] + cols
    return pd.concat([frame[keep]] + [r[keep] for r in rows], ignore_index=True)


# -------------------------------------------------------------------- helpers
def grouped(df, p):
    return pd.DataFrame({"unit": df.unit.to_numpy(), "minute": df.minute.to_numpy(),
        "label": df.label.to_numpy(int), "probability": p}).groupby(
            ["unit", "minute"], sort=True).agg(label=("label", "first"),
                                               probability=("probability", "mean"))


def fit_threshold(y, p):
    candidates = np.unique(p)
    scored = [(balanced_accuracy_score(y, p >= t), -abs(float(t) - .5), float(t))
              for t in candidates]
    return max(scored)[2]


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, p)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def model_path(name, view, seed):
    return OUT / "models" / f"{name}__{view.replace(' ', '_')}__s{seed}.joblib"


def keys_all():
    return [f"{n}|{v}" for n in MODEL_NAMES for v in (*VIEWS, "late probability fusion")]


def predict(df, name, view, seed):
    if view == "late probability fusion":
        return (predict(df, name, "CSI", seed) + predict(df, name, "radar", seed)) / 2
    obj = joblib.load(model_path(name, view, seed))
    x = matrix(df, obj["columns"], obj["fill"])
    return obj["model"].predict_proba(obj["scaler"].transform(x))[:, 1]


def summarize(values):
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "per_seed": [float(v) for v in a]}


# --------------------------------------------------------------------- phases
def select():
    target = OUT / "selection.json"
    if target.exists():
        raise FileExistsError("Selection is frozen; refusing to overwrite it")
    (OUT / "models").mkdir(exist_ok=True)
    cols_all = RAD_ALL + CSI_ALL
    train_raw, val_raw = read_old()
    rng = np.random.default_rng(AUG_SEED)
    train = balance(train_raw, cols_all, rng, "aug")
    val = balance(val_raw, cols_all, rng, "aug")
    train.to_csv(OUT / "train_windows.csv", index=False)
    val.to_csv(OUT / "validation_windows.csv", index=False)
    ytr = train.label.to_numpy(int)
    pred = {}
    for view, cols in VIEWS.items():
        fill = np.nanmedian(train[cols].replace([np.inf, -np.inf], np.nan).to_numpy(float), axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        xtr = matrix(train, cols, fill)
        xva = matrix(val, cols, fill)
        scaler = StandardScaler().fit(xtr)
        ztr, zva = scaler.transform(xtr), scaler.transform(xva)
        for name in MODEL_NAMES:
            for seed in SEEDS:
                model = make_model(name, seed).fit(ztr, ytr)
                joblib.dump({"model": model, "scaler": scaler, "fill": fill,
                             "columns": cols}, model_path(name, view, seed))
                pred[(name, view, seed)] = model.predict_proba(zva)[:, 1]
            print("trained", name, view, flush=True)
    for name in MODEL_NAMES:
        for seed in SEEDS:
            pred[(name, "late probability fusion", seed)] = (
                pred[(name, "CSI", seed)] + pred[(name, "radar", seed)]) / 2
    records = []
    minute_frames = {}
    for key in keys_all():
        name, view = key.split("|", 1)
        per_seed = []
        for seed in SEEDS:
            frame = grouped(val, pred[(name, view, seed)])
            y = frame.label.to_numpy(int)
            pp = frame.probability.to_numpy(float)
            th = fit_threshold(y, pp)
            m = metrics(y, pp, th)
            per_seed.append({"seed": seed, "threshold": th, **m,
                             "window": metrics(val.label.to_numpy(int),
                                               pred[(name, view, seed)], th)})
            minute_frames[(key, seed)] = frame
        records.append({"key": key, "family": "DL" if name == "MLP" else "ML",
                        "model": name, "view": view, "n": int(len(frame)),
                        "roc_auc": summarize([s["roc_auc"] for s in per_seed]),
                        "accuracy": summarize([s["accuracy"] for s in per_seed]),
                        "balanced_accuracy": summarize([s["balanced_accuracy"] for s in per_seed]),
                        "window_accuracy": summarize([s["window"]["accuracy"] for s in per_seed]),
                        "seeds": per_seed})
    ranked = sorted(records, key=lambda r: (-r["roc_auc"]["mean"], r["key"]))
    selected = {view: next(r for r in ranked if r["view"] == view)
                for view in (*VIEWS, "late probability fusion")}
    references = {fam: next(r for r in ranked if r["family"] == fam) for fam in ("ML", "DL")}
    counts = lambda f: {"windows": int(len(f)), "minutes": int(f.unit.nunique()),
                        "real_minutes_by_class": {int(k): int(v) for k, v in f[f.aug == 0].groupby("label").unit.nunique().items()},
                        "synthetic_minutes_by_class": {int(k): int(v) for k, v in f[f.aug == 1].groupby("label").unit.nunique().items()}}
    manifest = {
        "protocol": ("physical + training-basis PCA window descriptors; occupied minutes "
                     "synthesised to balance train and validation (gauss/gain/mixup/temporal, "
                     f"seed {AUG_SEED}); training-only imputation and standardisation; "
                     f"{len(SEEDS)} model seeds; mean validation-minute AUC selects view leaders "
                     "and ML/DL references; validation labels fix each seed's threshold; "
                     "seed 0 supplies LLM reference probabilities; no test-batch statistics"),
        "seeds": list(SEEDS), "descriptors": {"CSI": CSI_ALL, "radar": RAD_ALL},
        "train": counts(train), "validation": counts(val),
        "source_sha256": {"train_validation_features": sha256(OLD),
                          "pc_dots_train": sha256(DOTS / "pc_dots_train.npz"),
                          "pc_dots_val": sha256(DOTS / "pc_dots_val.npz")},
        "selected_views": {k: v["key"] for k, v in selected.items()},
        "references": {k: {"key": v["key"], "threshold": v["seeds"][0]["threshold"],
                           "balanced_accuracy": v["seeds"][0]["balanced_accuracy"]}
                       for k, v in references.items()},
        "all_validation_candidates": ranked}
    target.write_text(json.dumps(manifest, indent=2))
    for fam, r in references.items():
        minute_frames[(r["key"], 0)].to_csv(OUT / f"validation_{fam.lower()}_probabilities.csv")
    print(json.dumps({"selected_views": manifest["selected_views"],
                      "references": manifest["references"]}, indent=2), flush=True)


def evaluate():
    frozen = json.loads((OUT / "selection.json").read_text())
    cand = {r["key"]: r for r in frozen["all_validation_candidates"]}
    test = read_test()
    y_min = None
    outcomes = {}
    for key in keys_all():
        name, view = key.split("|", 1)
        r = cand[key]
        per_seed = []
        for seed in SEEDS:
            p = predict(test, name, view, seed)
            frame = grouped(test, p)
            th = r["seeds"][seed]["threshold"]
            m = metrics(frame.label.to_numpy(int), frame.probability.to_numpy(float), th)
            by_cap = {str(u): float(accuracy_score(g.label.to_numpy(int),
                (g.probability.to_numpy(float) >= th).astype(int)))
                for u, g in frame.groupby(level=0)}
            per_seed.append({"seed": seed, "threshold": th, **m,
                             "window": metrics(test.label.to_numpy(int), p, th),
                             "capture_accuracy": by_cap})
            if seed == 0 and key in (frozen["references"]["ML"]["key"], frozen["references"]["DL"]["key"]):
                frame.to_csv(OUT / ("test_" + key.replace("|", "__") + "_probabilities.csv"))
            if seed == 0 and key in frozen["selected_views"].values():
                frame.to_csv(OUT / ("test_" + key.replace("|", "__") + "_probabilities.csv"))
        outcomes[key] = {
            "validation": {k: r[k] for k in ("roc_auc", "accuracy", "balanced_accuracy", "window_accuracy")},
            "validation_threshold": summarize([s["threshold"] for s in r["seeds"]]),
            "test": {"roc_auc": summarize([s["roc_auc"] for s in per_seed]),
                     "accuracy": summarize([s["accuracy"] for s in per_seed]),
                     "balanced_accuracy": summarize([s["balanced_accuracy"] for s in per_seed]),
                     "window_accuracy": summarize([s["window"]["accuracy"] for s in per_seed]),
                     "confusion_matrix_seed0": per_seed[0]["confusion_matrix"],
                     "seeds": per_seed}}
        print(key, "test acc %.3f +- %.3f" % (outcomes[key]["test"]["accuracy"]["mean"],
                                               outcomes[key]["test"]["accuracy"]["sd"]), flush=True)
    report = {"selection_sha256": sha256(OUT / "selection.json"),
              "test_source_sha256": sha256(TEST), "test_windows": int(len(test)),
              "test_minutes": int(test.groupby(["unit", "minute"]).ngroups),
              "selected_views": frozen["selected_views"],
              "references": {k: v["key"] for k, v in frozen["references"].items()},
              "results": outcomes}
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("select", "evaluate"))
    {"select": select, "evaluate": evaluate}[ap.parse_args().phase]()
