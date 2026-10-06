"""Iterative separability screen: which physical descriptors + features
distinguish empty vs occupied, per dataset and per modality group.

Pipeline per (dataset, window, modality):
  0. keep rows whose modality ok-flags are set; drop cols >30% NaN or
     zero-variance; median-impute.
  1. univariate screen: ROC-AUC (direction-free), Cohen's d,
     Mann-Whitney p per feature.
  2. collinearity prune: greedy keep of best-ranked features with
     |Pearson r| <= 0.95 against everything already kept.
  3. RF importance (300 trees) on the pruned set.
  4. combined rank (univariate + RF), take top-K; grouped 5-fold CV
     (LogReg scaled + RF) on top-K -> separation quality estimate.
Groups for CV: capture id (multilink) / minute rec_id (legacy) —
windows from one recording never leak across folds.

Outputs: outputs/analysis.json, outputs/rankings_*.csv
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

warnings.filterwarnings("ignore")
OUT = C.OUTPUT_DIR
TOP_K = 15
CORR_CUT = 0.95
NAN_CUT = 0.30


def _uni_stats(X, y, names):
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    rows = []
    e, o = y == 0, y == 1
    for j, n in enumerate(names):
        a, b = X[e, j], X[o, j]
        try:
            auc = roc_auc_score(y, X[:, j])
        except Exception:
            auc = np.nan
        try:
            p = mannwhitneyu(a, b).pvalue
        except Exception:
            p = np.nan
        rows.append({"feature": n, "auc": auc,
                     "auc_dir": max(auc, 1 - auc) if np.isfinite(auc) else .5,
                     "cohen_d": C.cohen_d(a, b), "mw_p": p,
                     "mean_empty": float(np.nanmean(a)),
                     "mean_occ": float(np.nanmean(b)),
                     "std_empty": float(np.nanstd(a)),
                     "std_occ": float(np.nanstd(b))})
    return pd.DataFrame(rows)


def screen(X, y, groups, names, top_k=TOP_K):
    """Iterative screen. Returns (summary dict, ranking df)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # step 0: drop junk cols
    nanf = np.isnan(X).mean(axis=0)
    keep0 = np.flatnonzero((nanf <= NAN_CUT)
                           & (np.nanstd(X, axis=0) > 0))
    X = X[:, keep0]
    names = [names[i] for i in keep0]
    med = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(med, inds[1])

    # step 1: univariate
    uni = _uni_stats(X, y, names).sort_values("auc_dir", ascending=False)

    # step 2: collinearity prune (order = univariate rank)
    order = list(uni.feature.values)
    pos = {f: i for i, f in enumerate(order)}
    r = np.corrcoef(X[:, [names.index(f) for f in order]].T)
    kept, dropped = [], []
    for i, f in enumerate(order):
        if any(abs(r[i, pos[k]]) > CORR_CUT for k in kept):
            dropped.append(f)
        else:
            kept.append(f)

    # step 3: RF importance on pruned set
    idx = [names.index(f) for f in kept]
    rf = RandomForestClassifier(300, n_jobs=-1, random_state=0,
                                class_weight="balanced")
    rf.fit(X[:, idx], y)
    imp = dict(zip(kept, rf.feature_importances_))

    # step 4: combined rank -> top-k -> grouped CV probe
    uni_r = uni.set_index("feature").auc_dir.rank(ascending=False)
    rf_r = pd.Series(imp).rank(ascending=False)
    comb = (uni_r.reindex(kept) + rf_r.reindex(kept)).sort_values()
    topk = comb.index[:top_k].tolist()
    tk = [names.index(f) for f in topk]

    from sklearn.model_selection import StratifiedKFold
    n_groups = len(np.unique(groups))
    accs_lr, accs_rf = [], []
    if n_groups >= 10:
        # plenty of recording groups -> honest grouped CV
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        splits = cv.split(X[:, tk], y, groups)
    else:
        # few groups (e.g. 4 captures): window-level probe + LOCO below
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        splits = cv.split(X[:, tk], y)
    for tr, te in splits:
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000,
                                              class_weight="balanced"))
        lr.fit(X[tr][:, tk], y[tr])
        accs_lr.append(float(balanced_accuracy_score(
            y[te], lr.predict(X[te][:, tk]))))
        r2 = RandomForestClassifier(200, n_jobs=-1, random_state=0,
                                    class_weight="balanced")
        r2.fit(X[tr][:, tk], y[tr])
        accs_rf.append(float(balanced_accuracy_score(
            y[te], r2.predict(X[te][:, tk]))))

    # leave-one-recording-out recall per class (honest for few groups)
    loco = {"lr_recall_empty": np.nan, "lr_recall_occ": np.nan,
            "rf_recall_empty": np.nan, "rf_recall_occ": np.nan}
    if 2 <= n_groups < 10:
        recs = {}
        for gi, g in enumerate(np.unique(groups)):
            tr = groups != g
            te = groups == g
            if len(np.unique(y[tr])) < 2:
                continue
            lr = make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000,
                                                  class_weight="balanced"))
            lr.fit(X[tr][:, tk], y[tr])
            r2 = RandomForestClassifier(200, n_jobs=-1, random_state=0,
                                        class_weight="balanced")
            r2.fit(X[tr][:, tk], y[tr])
            for name, mdl in (("lr", lr), ("rf", r2)):
                recs.setdefault(f"{name}_recall_cls"
                                f"{y[te][0]}", []).append(
                    float((mdl.predict(X[te][:, tk]) == y[te]).mean()))
        loco = {"lr_recall_empty":
                float(np.mean(recs.get("lr_recall_cls0", [np.nan]))),
                "lr_recall_occ":
                float(np.mean(recs.get("lr_recall_cls1", [np.nan]))),
                "rf_recall_empty":
                float(np.mean(recs.get("rf_recall_cls0", [np.nan]))),
                "rf_recall_occ":
                float(np.mean(recs.get("rf_recall_cls1", [np.nan])))}

    uni = uni.set_index("feature")
    rank = pd.DataFrame({
        "feature": kept,
        "auc_dir": uni.loc[kept, "auc_dir"].values,
        "cohen_d": uni.loc[kept, "cohen_d"].values,
        "mw_p": uni.loc[kept, "mw_p"].values,
        "rf_imp": [imp[f] for f in kept],
        "comb_rank": comb.loc[kept].values,
        "selected": [f in topk for f in kept],
    }).sort_values("comb_rank")
    return {
        "n_windows": int(len(y)), "n_empty": int((y == 0).sum()),
        "n_occ": int((y == 1).sum()),
        "n_features_in": len(names), "n_pruned": len(kept),
        "n_dropped_collinear": len(dropped),
        "top_k": topk,
        "cv_acc_logreg": float(np.mean(accs_lr)) if accs_lr else np.nan,
        "cv_acc_rf": float(np.mean(accs_rf)) if accs_rf else np.nan,
        **loco,
        "best_single_auc": float(uni.auc_dir.max()),
        "best_single": str(uni.auc_dir.idxmax()),
    }, rank


# ---------------------------------------------------------------------------
# dataset/group definitions
# ---------------------------------------------------------------------------
def multilink_groups(df):
    """Modality -> (feature cols, row mask)."""
    suf = C.CSI_LINK_COLS
    c1 = C.csi_cols("csi1_")
    c2 = C.csi_cols("csi2_")
    df = df.copy()
    for s in suf:                                   # 2-link mean channel
        df[f"csiM_{s}"] = df[[f"csi1_{s}", f"csi2_{s}"]].mean(axis=1)
    cM = C.csi_cols("csiM_")
    radok = df.rad_ok == 1
    return df, {
        "csi_link1": (c1, df.csi1_ok == 1),
        "csi_link2": (c2, df.csi2_ok == 1),
        "csi_2link_mean": (cM, (df.csi1_ok == 1) & (df.csi2_ok == 1)),
        "radar": (list(C.RADAR_COLS), radok),
        "fusion": (list(C.RADAR_COLS) + cM,
                   radok & (df.csi1_ok == 1)),
    }


def old_groups(df):
    csi = [c for c in df.columns if c.startswith("csi_")
           and c not in ("csi_ok",)]
    radar = [c for c in C.RADAR_COLS if c in df.columns]
    radok = df.rad_ok == 1
    return df, {
        "csi": (csi, df.csi_ok == 1),
        "radar": (radar, radok),
        "fusion": (radar + csi, radok & (df.csi_ok == 1)),
    }


def run_table(name, df, groups_fn, results, tag_prefix):
    df, groups = groups_fn(df)
    for mod, (cols, mask) in groups.items():
        cols = [c for c in cols if c in df.columns]
        sub = df[mask.fillna(False) if hasattr(mask, "fillna") else mask]
        sub = sub[np.isfinite(sub.label)]
        if len(sub) < 40 or sub.label.nunique() < 2:
            print(f"  {name}/{mod}: skipped ({len(sub)} rows)", flush=True)
            continue
        X = sub[cols].to_numpy(np.float64)
        y = sub.label.astype(int).values
        g = (sub["cap"] if "cap" in sub else sub.rec_id).astype(str).values
        summ, rank = screen(X, y, g, cols)
        results[f"{name}/{mod}"] = summ
        rank.to_csv(OUT / f"rankings_{tag_prefix}_{mod}.csv", index=False)
        loco = (f" loco[lr e/o={summ['lr_recall_empty']:.2f}/"
                f"{summ['lr_recall_occ']:.2f}]"
                if np.isfinite(summ.get('lr_recall_empty', np.nan))
                else "")
        print(f"  {name:>16}/{mod:<14} n={summ['n_windows']:>5} "
              f"top={summ['best_single']:<14} "
              f"auc={summ['best_single_auc']:.3f} "
              f"cv_lr={summ['cv_acc_logreg']:.3f} "
              f"cv_rf={summ['cv_acc_rf']:.3f}{loco}", flush=True)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.time()
    for W in C.WIN_LIST:
        print(f"== W={W}s ==", flush=True)
        ml = pd.read_csv(OUT / f"features_multilink_{W}s.csv")
        ml = run_table(f"multilink_{W}s", ml, multilink_groups,
                       results, f"ml{W}s")
        old = pd.read_csv(OUT / f"features_old_{W}s_ext.csv")
        for ds in ("train", "test"):
            run_table(f"old_{ds}_{W}s", old[old.dataset == ds].copy(),
                      old_groups, results, f"old{ds}{W}s")
    (OUT / "analysis.json").write_text(json.dumps(results, indent=2))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
