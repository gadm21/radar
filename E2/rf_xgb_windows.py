"""RandomForest / XGBoost on radar SNR + CSI PCA projections, swept over
segment (window-group) size.

Protocol
--------
- Input units are the cached 50-radar-frame windows; a segment groups W
  consecutive windows of one minute (W in {1,2,4,8} plus 'all' = whole
  minute).
- Features per segment:
    radar  snr_max_db, snr_mean_db   -- clutter-masked peak/median of the
           range-Doppler channel (DC rows + near-range bins masked)
    csi    pc{1..5}_mean, pc{1..5}_std -- PCA projections of per-step
           log1p CSI amplitude, pooled over the segment (basis fitted on
           train/ only), plus has_csi indicator
- Models: sklearn RandomForest (balanced), XGBoost (scale_pos_weight).
  Trained on train/ segments ONLY; calibration_minutes is never used for
  supervised fitting. Validation used for reporting only (no tuning).
- Reported: segment accuracy / balanced accuracy on validation_minutes
  and test_minutes, plus minute-level majority-vote accuracy.

Usage: python E2/rf_xgb_windows.py
Outputs: outputs/rf_xgb_window_results.csv, outputs/figs/rf_xgb_windows.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import train as T

OUT = C.OUTPUT_DIR
FIG = OUT / "figs"
SEED = 42
N_PC = 5
WINDOW_SIZES = [1, 2, 4, 8, "all"]
_DC_ROWS = (10, 14)
_NEAR_RANGE = 2


# ---------------------------------------------------------------------------
# Feature extraction per segment
# ---------------------------------------------------------------------------
def snr_segment(rd_log):
    """rd_log: (nF,24,24) log1p RD frames -> (snr_max_db, snr_mean_db)."""
    rd = np.expm1(rd_log.astype(np.float32))
    m = np.zeros((24, 24), bool)
    m[_DC_ROWS[0]:_DC_ROWS[1], :] = True
    m[:, :_NEAR_RANGE] = True
    flat = rd.reshape(len(rd), -1)
    med = np.median(flat, axis=1) + 1e-9
    peak = np.where(m[None], 0.0, rd).reshape(len(rd), -1).max(axis=1)
    s = 10.0 * np.log10(peak / med)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.nan, np.nan
    return float(s.max()), float(s.mean())


def segments_for_minute(d, W):
    """Yield (radar_frames, csi_steps) per segment of W windows."""
    n = d["radar"].shape[0]
    ends = [n] if W == "all" else list(range(W, n + 1, W))
    start = 0
    for e in ends:
        if e <= start:
            continue
        yield d["radar"][start:e], d["csi"][start:e], d["csi_valid"][start:e]
        start = e


def extract(index, pca_mean, pca_comp):
    """Build segment-level feature table for all splits."""
    rows = []
    for i, row in enumerate(index.sort_values("rec_id").itertuples(index=False)):
        d = np.load(row.path)
        minute = Path(row.path).stem
        for W in WINDOW_SIZES:
            for k, (rad, csi, cv) in enumerate(
                    segments_for_minute(d, W)):
                snr_max, snr_mean = snr_segment(
                    rad[:, :, 0].reshape(-1, 24, 24))
                csteps = csi[cv.astype(bool)]
                feats = {"has_csi": 0}
                if len(csteps) >= 4:
                    amp = np.log1p(csteps.reshape(-1, 52))
                    z = (amp - pca_mean) @ pca_comp.T          # (steps,5)
                    for j in range(N_PC):
                        feats[f"pc{j+1}_mean"] = float(z[:, j].mean())
                        feats[f"pc{j+1}_std"] = float(z[:, j].std())
                    feats["has_csi"] = 1
                else:
                    for j in range(N_PC):
                        feats[f"pc{j+1}_mean"] = np.nan
                        feats[f"pc{j+1}_std"] = np.nan
                rows.append({"split": row.split, "minute": minute,
                             "W": str(W), "seg": k,
                             "label": int(row.label),
                             "snr_max_db": snr_max,
                             "snr_mean_db": snr_mean, **feats})
        if (i + 1) % 400 == 0:
            print(f"[{i+1}/{len(index)}]", flush=True)
    return pd.DataFrame(rows)


def fit_csi_pca(index):
    """PCA basis on per-step log1p amplitude from train/ windows only."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(SEED)
    vecs = []
    tr = index[index.split == "train"]
    for row in tr.itertuples(index=False):
        d = np.load(row.path)
        c, v = d["csi"], d["csi_valid"].astype(bool)
        if v.any():
            vecs.append(np.log1p(c[v].astype(np.float32)
                                 .reshape(-1, 52)))
    X = np.concatenate(vecs)
    if len(X) > 200_000:
        X = X[rng.choice(len(X), 200_000, replace=False)]
    pca = PCA(n_components=N_PC, random_state=SEED).fit(X)
    print(f"CSI PCA fit on {len(X)} train steps; "
          f"explained={pca.explained_variance_ratio_.round(3)}")
    return pca.mean_, pca.components_


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------
def evaluate(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from xgboost import XGBClassifier

    feats = ["snr_max_db", "snr_mean_db", "has_csi"] + [
        f"pc{i+1}_{s}" for i in range(N_PC) for s in ("mean", "std")]
    results = []
    for W in WINDOW_SIZES:
        sub = df[df.W == str(W)]
        trm = sub["split"].eq("train").to_numpy()
        Xtr = sub.loc[trm, feats].to_numpy(np.float32)
        ytr = sub.loc[trm, "label"].to_numpy()
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=400, class_weight="balanced",
                n_jobs=-1, random_state=SEED),
            "xgboost": XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.1,
                scale_pos_weight=spw, n_jobs=-1, random_state=SEED,
                eval_metric="logloss"),
        }
        for name, model in models.items():
            model.fit(np.nan_to_num(Xtr, nan=0.0), ytr)
            for sname in ("validation_minutes", "test_minutes"):
                m = sub["split"].eq(sname).to_numpy()
                Xs = np.nan_to_num(sub.loc[m, feats].to_numpy(np.float32),
                                   nan=0.0)
                ys = sub.loc[m, "label"].to_numpy()
                pred = model.predict(Xs)
                results.append({
                    "model": name, "W": str(W), "split": sname,
                    "level": "segment", "n": int(m.sum()),
                    "accuracy": float(accuracy_score(ys, pred)),
                    "balanced_acc": float(balanced_accuracy_score(ys, pred))})
                # minute-level majority vote
                mins = sub.loc[m].copy()
                mins["pred"] = pred
                vote = (mins.groupby("minute")
                        .agg(label=("label", "first"),
                             pred=("pred", lambda p: int(p.mean() > 0.5))))
                results.append({
                    "model": name, "W": str(W), "split": sname,
                    "level": "minute", "n": len(vote),
                    "accuracy": float(accuracy_score(
                        vote.label, vote.pred)),
                    "balanced_acc": float(balanced_accuracy_score(
                        vote.label, vote.pred))})
            print(f"  W={W} {name} done", flush=True)
    out = pd.DataFrame(results)
    out.to_csv(OUT / "rf_xgb_window_results.csv", index=False)
    return out


def plot(res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    colors = {"random_forest": "#1b9e77", "xgboost": "#d95f02"}
    order = [str(w) for w in WINDOW_SIZES]
    for ax, split in zip(axes, ["validation_minutes", "test_minutes"]):
        for model, col in colors.items():
            for lvl, mk in (("segment", "o"), ("minute", "s")):
                g = res[(res.model == model) & (res.split == split)
                        & (res.level == lvl)]
                g = g.set_index("W").reindex(order)
                ax.plot(order, g.accuracy, marker=mk, color=col,
                        ls="-" if lvl == "segment" else "--",
                        label=f"{model} ({lvl})")
        ax.set(title=split.replace("_minutes", ""), xlabel="windows per segment",
               ylim=(0.3, 1.0))
        ax.grid(alpha=0.25)
        ax.axhline(0.5, ls=":", c="#bbb")
    axes[0].set_ylabel("accuracy")
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("RF / XGBoost — radar SNR + CSI PC projections vs segment size")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "rf_xgb_windows.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "rf_xgb_windows.png")


def main():
    index = T.load_index()
    pca_mean, pca_comp = fit_csi_pca(index)
    df = extract(index, pca_mean, pca_comp)
    df.to_csv(OUT / "rf_xgb_window_features.csv", index=False)
    print(f"{len(df)} segments", flush=True)
    res = evaluate(df)
    print(res.to_string(index=False,
                        float_format=lambda x: f"{x:.3f}"))
    plot(res)


if __name__ == "__main__":
    main()
