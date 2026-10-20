"""Five-seed RF benchmark ablation with real fit minutes only.

Uses the saved 5 s descriptor table and the historical balanced test IDs.
The benchmark test is retrospective and class is confounded with session.
"""
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import occupancy_pipeline as P


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "E2/outputs/occupancy/features_5s.csv"
OUT = ROOT / "E2/outputs/revision/reviewer_audit"
OUT.mkdir(parents=True, exist_ok=True)


def aggregate(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"rec_id": df.rec_id.to_numpy(),
                         "label": df.label.to_numpy().astype(int),
                         "probability": p}).groupby("rec_id", sort=True).agg(
                             label=("label", "first"), probability=("probability", "mean"))


def two_means_threshold(p: np.ndarray) -> float:
    """Exact one-dimensional split; identical ties stay in one cluster."""
    x = np.sort(np.asarray(p, float))
    if len(x) < 2 or x[0] == x[-1]:
        return float(np.median(x))
    n = len(x)
    s, q = np.cumsum(x), np.cumsum(x*x)
    left = np.arange(1, n)
    right = n-left
    cost = q[:-1]-s[:-1]**2/left + (q[-1]-q[:-1])-(s[-1]-s[:-1])**2/right
    cost[x[:-1] == x[1:]] = np.inf
    k = int(np.argmin(cost))
    return float((s[k]/left[k] + (s[-1]-s[k])/right[k])/2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-aug", action="store_true",
                        help="Reproduce the augmented reference with the same code")
    args = parser.parse_args()
    prefix = "with_augmentation" if args.include_aug else "no_augmentation"
    f = pd.read_csv(SOURCE)
    f["is_val"] = f.is_val.astype(bool)
    f["aug"] = f.aug.astype(bool)
    fit = f[(f.dataset == "train") & ~f.is_val &
            (args.include_aug | ~f.aug)].copy()
    val = f[(f.dataset == "train") & f.is_val & ~f.aug].copy()
    test = f[(f.dataset == "test") & ~f.aug].copy()
    assert fit.loc[~fit.aug, "rec_id"].nunique() == 919
    assert fit.loc[fit.aug, "rec_id"].nunique() == (601 if args.include_aug else 0)
    assert val.rec_id.nunique() == 162
    rng = np.random.default_rng(42)
    tmins = test.groupby("rec_id").label.first()
    keep = []
    for _, group in tmins.groupby(tmins):
        keep.extend(rng.choice(group.index, size=tmins.value_counts().min(), replace=False))
    test = test[test.rec_id.isin(keep)].copy()
    assert test.rec_id.nunique() == 160

    metrics, predictions, weights = [], [], []
    for seed in range(42, 47):
        prob = {}
        for view in ("csi", "radar"):
            cols = P.FEATURE_SETS[view]
            model = make_pipeline(SimpleImputer(strategy="median"),
                                  StandardScaler(),
                                  RandomForestClassifier(
                                      n_estimators=300,
                                      class_weight="balanced_subsample",
                                      n_jobs=4, random_state=seed))
            model.fit(fit[cols], fit.label.astype(int))
            prob[(view, "val")] = model.predict_proba(val[cols])[:, 1]
            prob[(view, "test")] = model.predict_proba(test[cols])[:, 1]
        best_weight, best_f1 = 0.5, -1.0
        for weight in np.linspace(0, 1, 21):
            pv = weight*prob[("csi", "val")] + (1-weight)*prob[("radar", "val")]
            score = f1_score(val.label, pv >= .5)
            if score > best_f1:
                best_weight, best_f1 = float(weight), score
        weights.append({"seed": seed, "w_csi": best_weight, "validation_f1": best_f1})
        for view in ("csi", "moe"):
            if view == "csi":
                pv, pt = prob[("csi", "val")], prob[("csi", "test")]
            else:
                pv = best_weight*prob[("csi", "val")] + (1-best_weight)*prob[("radar", "val")]
                pt = best_weight*prob[("csi", "test")] + (1-best_weight)*prob[("radar", "test")]
            v = aggregate(val, pv)
            t = aggregate(test, pt)
            # Match the historical validation-threshold selection at minute level.
            threshold = float(P._best_thr(v.label.to_numpy(), v.probability.to_numpy()))
            p = t.probability.to_numpy()
            y = t.label.to_numpy()
            row = {"seed": seed, "model": "rf", "view": view,
                   "fit_real_minutes": fit.loc[~fit.aug, "rec_id"].nunique(),
                   "fit_augmented_minutes": fit.loc[fit.aug, "rec_id"].nunique(),
                   "validation_minutes": len(v), "test_minutes": len(t),
                   "validation_threshold": threshold,
                   "accuracy_validation_threshold": accuracy_score(y, p >= threshold),
                   "accuracy_median_batch": accuracy_score(y, p >= np.median(p)),
                   "accuracy_two_means_batch": accuracy_score(y, p >= two_means_threshold(p)),
                   "auc": roc_auc_score(y, p),
                   "balanced_accuracy_validation_threshold": balanced_accuracy_score(y, p >= threshold)}
            metrics.append(row)
            predictions.extend({"seed": seed, "view": view, "rec_id": rec_id,
                                "label": int(lab), "probability": float(pp),
                                "validation_threshold": threshold}
                               for rec_id, lab, pp in zip(t.index, y, p))
        print(f"seed {seed} complete", flush=True)
    pd.DataFrame(metrics).to_csv(OUT / f"{prefix}_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / f"{prefix}_predictions.csv", index=False)
    pd.DataFrame(weights).to_csv(OUT / f"{prefix}_gate_weights.csv", index=False)
    print(pd.DataFrame(metrics).groupby("view")[["accuracy_validation_threshold",
        "accuracy_median_batch", "accuracy_two_means_batch", "auc"]].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
