"""Frozen comparison, revision 3: three-level hierarchical PCA descriptors on
fifteen-second windows.

Changes from frozen_v2.py
  * The twelve single-basis PC mean/std descriptors are replaced by a
    hierarchical PCA per modality. Level 1 (1 s) is the training-fitted
    first-level basis from E1 (PC1-PC2 of pc_dots_*.npz). Level 2 (5 s) is a
    two-component PCA fitted on the training level-1 output of five
    consecutive seconds; level 3 (15 s) is a two-component PCA fitted on the
    training level-2 output of three consecutive blocks. Validation and test
    seconds are only projected through the frozen bases.
  * Classifier windows are fifteen seconds (three consecutive five-second
    descriptor rows averaged); each window carries ten hierarchical values
    per modality: mean and std of the level-1 PC1/PC2 over the window's
    seconds, mean and std of the level-2 PC1/PC2 over its three blocks, and
    the level-3 PC1/PC2. Minute probabilities average four windows.
  * Balancing, seeds, selection and the frozen protocol are unchanged.

Run ``--phase select`` first (fits and saves the hierarchy before any
multilink file is opened), then ``--phase evaluate``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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
OUT = ROOT / "E2/outputs/frozen_v3"
OUT.mkdir(parents=True, exist_ok=True)

SEC, BLOCK, WIN = 1, 5, 15          # level units in seconds
BLOCKS_PER_WIN = WIN // BLOCK       # 3
WINS_PER_MIN = 60 // WIN            # 4
ROWS_PER_WIN = WIN // 5             # three five-second descriptor rows
MOD_COLS = {"csi": slice(1, 3), "rad": slice(4, 6)}   # level-1 PC1/PC2 columns


def hier_cols(mod):
    return ([f"{mod}_l{lvl}_pc{j}_{s}" for lvl in (1, 2) for s in ("mean", "std") for j in (1, 2)]
            + [f"{mod}_l3_pc{j}" for j in (1, 2)])


CSI_H, RAD_H = hier_cols("csi"), hier_cols("rad")
CSI_ALL = CSI + CSI_H
RAD_ALL = RAD + RAD_H
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


# ---------------------------------------------------------------- hierarchy
def block5(a, cols, start):
    """Level-1 (5, 2) array for seconds start..start+5, missing seconds
    interpolated from valid ones inside the block; None if < 2 valid (the
    same minimum frozen_v2 used for a five-second window)."""
    seg = a[start:start + BLOCK]
    if len(seg) < BLOCK:
        return None
    good = (seg[:, 0] > .5) & np.isfinite(seg[:, cols]).all(axis=1)
    if good.sum() < 2:
        return None
    v = seg[:, cols].astype(float)
    return np.column_stack([np.interp(np.arange(BLOCK), np.flatnonzero(good), v[good, j])
                            for j in range(v.shape[1])])


def window_blocks(a, cols, start):
    return [block5(a, cols, start + BLOCK * b) for b in range(BLOCKS_PER_WIN)]


def level2(blocks, h):
    """Level-2 coordinates (3, 2) for a window's blocks; NaN rows for
    missing blocks."""
    out = np.full((BLOCKS_PER_WIN, 2), np.nan)
    present = [i for i, b in enumerate(blocks) if b is not None]
    if present:
        x = np.stack([blocks[i].reshape(-1) for i in present])
        out[present] = h["p5"].transform(h["sc5"].transform(x))
    return out


def level3_input(l2):
    """(6,) level-2 trajectory with a missing block interpolated; None if
    fewer than two blocks are present."""
    present = np.flatnonzero(np.isfinite(l2).all(axis=1))
    if len(present) < 2:
        return None
    return np.column_stack([np.interp(np.arange(BLOCKS_PER_WIN), present, l2[present, j])
                            for j in range(2)]).reshape(-1)


def mean_std(points, min_points):
    """[mean pc1, mean pc2, std pc1, std pc2] over the finite rows of an
    (n, 2) array; NaN when fewer than min_points rows are present."""
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) < min_points:
        return np.full(4, np.nan)
    return np.r_[pts.mean(axis=0), pts.std(axis=0)]


def window_vector(a, cols, start, h):
    """Ten hierarchical values for one fifteen-second window of one modality:
    level-1 mean/std, level-2 mean/std, level-3 coordinates."""
    blocks = window_blocks(a, cols, start)
    l1 = np.concatenate([b for b in blocks if b is not None]) if any(
        b is not None for b in blocks) else np.full((0, 2), np.nan)
    l2 = level2(blocks, h)
    x3 = level3_input(l2)
    l3 = (h["p15"].transform(h["sc15"].transform(x3[None]))[0]
          if x3 is not None else np.full(2, np.nan))
    return np.r_[mean_std(l1, 2), mean_std(l2, 2), l3]


def fit_hierarchy(train_dots, keys, cols):
    blocks = [b for k in keys for s in range(0, len(train_dots[k]) - BLOCK + 1, BLOCK)
              if (b := block5(train_dots[k], cols, s)) is not None]
    x5 = np.stack([b.reshape(-1) for b in blocks])
    sc5 = StandardScaler().fit(x5)
    p5 = PCA(n_components=2, random_state=0).fit(sc5.transform(x5))
    h = {"sc5": sc5, "p5": p5}
    x15 = []
    for k in keys:
        a = train_dots[k]
        for s in range(0, len(a) - WIN + 1, WIN):
            x3 = level3_input(level2(window_blocks(a, cols, s), h))
            if x3 is not None:
                x15.append(x3)
    x15 = np.stack(x15)
    sc15 = StandardScaler().fit(x15)
    p15 = PCA(n_components=2, random_state=0).fit(sc15.transform(x15))
    h.update({"sc15": sc15, "p15": p15, "n5": int(len(x5)), "n15": int(len(x15)),
              "explained5": p5.explained_variance_ratio_.tolist(),
              "explained15": p15.explained_variance_ratio_.tolist()})
    return h


def hierarchical_descriptors(dots, keys, wins, hierarchy):
    """DataFrame of CSI_H + RAD_H for (unit, fifteen-second window) pairs."""
    out = np.full((len(keys), len(CSI_H) + len(RAD_H)), np.nan)
    for i, (key, w) in enumerate(zip(keys, wins)):
        a = dots.get(key)
        if a is None:
            continue
        start = WIN * int(w)
        out[i] = np.r_[window_vector(a, MOD_COLS["csi"], start, hierarchy["csi"]),
                       window_vector(a, MOD_COLS["rad"], start, hierarchy["rad"])]
    return pd.DataFrame(out, columns=CSI_H + RAD_H)


def load_dots(name):
    with np.load(DOTS / f"pc_dots_{name}.npz") as z:
        return {k: z[k].copy() for k in z.files}


def load_hierarchy():
    return joblib.load(OUT / "hierarchy.joblib")


# ---------------------------------------------------------------- descriptors
def aggregate15(df, cols):
    """Collapse five-second descriptor rows into fifteen-second windows."""
    df = df.copy()
    df["win"] = (df.win // ROWS_PER_WIN).astype(int)
    g = df.groupby(["unit", "minute", "win"], sort=True)
    agg = g[cols].mean()
    agg["label"] = g.label.first()
    agg["aug"] = 0
    return agg.reset_index()


def read_old(hierarchy):
    d = pd.read_csv(OLD)
    frames = {}
    for split, prefix, dot in (("train", "train/", "train"),
                               ("validation", "test_minutes/", "val")):
        f = d[d.rec_id.str.startswith(prefix)].copy()
        f["unit"] = f.rec_id.astype(str)
        f["minute"] = 0
        f = aggregate15(f, CSI + RAD)
        pcs = hierarchical_descriptors(load_dots(dot), f.unit.tolist(), f.win.tolist(), hierarchy)
        frames[split] = pd.concat([f, pcs], axis=1)
    return frames["train"], frames["validation"]


def read_test(hierarchy=None):
    hierarchy = hierarchy or load_hierarchy()
    t = pd.read_csv(TEST)
    for suffix in [s.removeprefix("csi_") for s in CSI]:
        t[f"csi_{suffix}"] = t[[f"csi1_{suffix}", f"csi2_{suffix}"]].mean(axis=1)
    t["unit"] = t.cap.astype(str)
    t["minute"] = (t.t_rel // 60).astype(int)
    t = aggregate15(t, CSI + RAD)
    pcs = hierarchical_descriptors(load_dots("testml_trainbasis"),
                                   t.unit.tolist(), t.win.tolist(), hierarchy)
    return pd.concat([t, pcs], axis=1)


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


def nan_share(frame, prefix):
    cols = [c for c in frame.columns if c.startswith(prefix)]
    return float(frame[cols].isna().to_numpy().mean())


# --------------------------------------------------------------------- phases
def select():
    target = OUT / "selection.json"
    if target.exists():
        raise FileExistsError("Selection is frozen; refusing to overwrite it")
    (OUT / "models").mkdir(exist_ok=True)
    cols_all = RAD_ALL + CSI_ALL
    train_dots = load_dots("train")
    train_keys = sorted(k for k in train_dots if k.startswith("train/"))
    hierarchy = {mod: fit_hierarchy(train_dots, train_keys, cols) for mod, cols in MOD_COLS.items()}
    joblib.dump(hierarchy, OUT / "hierarchy.joblib")
    (OUT / "hierarchy.json").write_text(json.dumps({
        "levels": {"1": "1 s: E1 training-fitted first-level basis, PC1-PC2",
                   "2": "5 s: PCA(2) on five consecutive level-1 pairs, fitted on train",
                   "3": "15 s: PCA(2) on three consecutive level-2 pairs, fitted on train"},
        "train_minutes": len(train_keys),
        **{mod: {k: hierarchy[mod][k] for k in ("n5", "n15", "explained5", "explained15")}
           for mod in MOD_COLS}}, indent=2))
    train_raw, val_raw = read_old(hierarchy)
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
                        "synthetic_minutes_by_class": {int(k): int(v) for k, v in f[f.aug == 1].groupby("label").unit.nunique().items()},
                        "nan_share": {m: nan_share(f, f"{m}_l") for m in MOD_COLS}}
    manifest = {
        "protocol": ("physical descriptors averaged over fifteen-second windows plus "
                     "three-level hierarchical PCA (1 s / 5 s / 15 s, two components per "
                     "level, every level fitted on training output of the level below); "
                     "occupied minutes synthesised to balance train and validation "
                     f"(gauss/gain/mixup/temporal, seed {AUG_SEED}); training-only imputation "
                     f"and standardisation; {len(SEEDS)} model seeds; mean validation-minute "
                     "AUC selects view leaders and ML/DL references; validation labels fix "
                     "each seed's threshold; minute probability = mean of four windows; "
                     "seed 0 supplies LLM reference probabilities; no test-batch statistics"),
        "window_seconds": WIN, "windows_per_minute": WINS_PER_MIN,
        "seeds": list(SEEDS), "descriptors": {"CSI": CSI_ALL, "radar": RAD_ALL},
        "train": counts(train), "validation": counts(val),
        "source_sha256": {"train_validation_features": sha256(OLD),
                          "pc_dots_train": sha256(DOTS / "pc_dots_train.npz"),
                          "pc_dots_val": sha256(DOTS / "pc_dots_val.npz"),
                          "hierarchy": sha256(OUT / "hierarchy.joblib")},
        "selected_views": {k: v["key"] for k, v in selected.items()},
        "references": {k: {"key": v["key"], "threshold": v["seeds"][0]["threshold"],
                           "balanced_accuracy": v["seeds"][0]["balanced_accuracy"]}
                       for k, v in references.items()},
        "all_validation_candidates": ranked}
    target.write_text(json.dumps(manifest, indent=2))
    for fam, r in references.items():
        minute_frames[(r["key"], 0)].to_csv(OUT / f"validation_{fam.lower()}_probabilities.csv")
    print(json.dumps({"selected_views": manifest["selected_views"],
                      "references": manifest["references"],
                      "train": manifest["train"], "validation": manifest["validation"]},
                     indent=2), flush=True)


def evaluate():
    frozen = json.loads((OUT / "selection.json").read_text())
    cand = {r["key"]: r for r in frozen["all_validation_candidates"]}
    test = read_test()
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
            if seed == 0 and (key in (frozen["references"]["ML"]["key"], frozen["references"]["DL"]["key"])
                              or key in frozen["selected_views"].values()):
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
              "test_nan_share": {m: nan_share(test, f"{m}_l") for m in MOD_COLS},
              "selected_views": frozen["selected_views"],
              "references": {k: v["key"] for k, v in frozen["references"].items()},
              "results": outcomes}
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("select", "evaluate"))
    {"select": select, "evaluate": evaluate}[ap.parse_args().phase]()
