"""Feature-driven E2 experiment: per-minute features -> rules / ML / DL.

Features (from E2/cache windows, aggregated to minute level):
  radar:
    snr_db / snr_mean_db     peak-vs-median clutter-masked SNR (E2 cache
                           ch0 = range-Doppler; rows 10:14 + near-range
                           cols masked), per-minute max / mean of frames
    xy_*                   "xy energy map" = range-azimuth channel (ch1,
                           azimuth x range ~ x-y plane): temporal-std map
                           energy per window -> per-minute stats
    rd_std_*               temporal-std energy of the RD channel (extra)
  csi:
    csi_amp_*              amplitude mean / std per minute
    csi_rollvar_*          causal trailing-window (w=20) amplitude
                           variance (WifiSensingESP32HAR-style), per
                           window -> minute mean / q90
    csi_stft_*             STFT band powers of detrended amplitude
                           (0-0.5, 0.5-2, 2-8 Hz at ~23 Hz grid rate)

Split protocol (post-cleanup semantics):
  train        = train/ only (supervised fitting + scaler stats)
  calibration  = calibration_minutes/ (pretraining / calibration only —
                 used here for the feature autoencoder, never supervised)
  val          = validation_minutes/ (threshold + model selection)
  test         = test_minutes/ (final metrics)

Models x feature sets:
  rules (thresholds calibrated on validation):
    rule_radar   snr_db >= t
    rule_csi     csi_rollvar_q90 >= t
    rule_fusion  OR rule over (snr_db, csi_rollvar_q90)
  ML (fit on train, threshold on val): logistic regression,
    HistGradientBoosting, MLP — on every feature set
  DL: ae_calib — feature autoencoder pretrained on calibration_minutes,
    logistic head trained on train/ embeddings

Outputs: feature_experiments_features.csv, feature_experiments_metrics.csv,
feature_experiments_predictions.csv, feature_experiments_selected.json,
figs/feature_exp_{test_ba,scatter,roc}.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import train as T

OUT = C.OUTPUT_DIR
FIG = OUT / "figs"
SEED = 42

SPLITS = ["calibration_minutes", "train", "validation_minutes",
          "test_minutes"]
SPLIT_XLABELS = {"calibration_minutes": "calibration", "train": "train",
                 "validation_minutes": "validation",
                 "test_minutes": "test"}
ACT_COLORS = {"empty": "#000000", "sleep": "#F2C500",
              "present": "#D62728"}

_DC_ROWS = (10, 14)     # RD doppler-DC rows masked (24x24 cache maps)
_NEAR_RANGE = 2
ROLL_W = 20             # causal rolling-variance window on the 128-pt grid
STFT_BANDS = [(0.0, 0.5), (0.5, 2.0), (2.0, 8.0)]

RADAR_SNR_COLS = ["snr_db", "snr_mean_db"]
RADAR_XY_COLS = ["xy_std_q90_mean", "xy_std_q90_max", "xy_energy_mean"]
RADAR_RD_COLS = ["rd_std_q90_mean", "rd_std_q90_max"]
RADAR_ALL = RADAR_SNR_COLS + RADAR_XY_COLS + RADAR_RD_COLS
CSI_AMP_COLS = ["csi_amp_mean", "csi_amp_std"]
CSI_VAR_COLS = ["csi_rollvar_mean", "csi_rollvar_q90"]
CSI_STFT_COLS = ["csi_stft_low", "csi_stft_med", "csi_stft_high"]
CSI_ALL = CSI_AMP_COLS + CSI_VAR_COLS + CSI_STFT_COLS
FEATURE_SETS = {
    "csi_amp": CSI_AMP_COLS,
    "csi_rollvar": CSI_VAR_COLS,
    "csi_stft": CSI_STFT_COLS,
    "csi_all": CSI_ALL,
    "radar_snr": RADAR_SNR_COLS,
    "radar_xy": RADAR_XY_COLS,
    "radar_all": RADAR_ALL,
    "fusion": CSI_ALL + RADAR_ALL,
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _rollvar(x, w):
    """Causal trailing variance along axis 1. x: (nW,T,S)."""
    z = np.zeros((x.shape[0], 1, x.shape[2]), dtype=np.float32)
    cs = np.concatenate([z, np.cumsum(x, axis=1)], axis=1)
    cs2 = np.concatenate([z, np.cumsum(x * x, axis=1)], axis=1)
    hi = np.arange(1, x.shape[1] + 1)
    lo = np.clip(hi - w, 0, None)
    cnt = (hi - lo).astype(np.float32)[None, :, None]
    mean = (cs[:, hi] - cs[:, lo]) / cnt
    msq = (cs2[:, hi] - cs2[:, lo]) / cnt
    return np.clip(msq - mean * mean, 0, None)


def radar_features(d):
    """Per-minute radar features from one cache npz."""
    rd = np.expm1(d["radar"][:, :, 0].astype(np.float32))   # (nW,50,24,24)
    ra = np.expm1(d["radar"][:, :, 1].astype(np.float32))   # (nW,50,24,24)
    nw, nf = rd.shape[0], rd.shape[1]
    flat = rd.reshape(-1, 24, 24)
    m = np.zeros((24, 24), bool)
    m[_DC_ROWS[0]:_DC_ROWS[1], :] = True
    m[:, :_NEAR_RANGE] = True
    med = np.median(flat.reshape(len(flat), -1), axis=1) + 1e-9
    peak = np.where(m[None], 0.0, flat).reshape(len(flat), -1).max(axis=1)
    snr = 10.0 * np.log10(peak / med)
    snr = snr[np.isfinite(snr)]
    sd = rd.std(axis=1)[:, ~m]                       # per-bin temporal std
    sa = ra.std(axis=1).reshape(nw, -1)
    en = ra.mean(axis=1).reshape(nw, -1)
    out = {
        "snr_db": float(snr.max()) if snr.size else np.nan,
        "snr_mean_db": float(snr.mean()) if snr.size else np.nan,
        "rd_std_q90_mean": float(np.quantile(sd, 0.9, axis=1).mean()),
        "rd_std_q90_max": float(np.quantile(sd, 0.9, axis=1).max()),
        "xy_std_q90_mean": float(np.quantile(sa, 0.9, axis=1).mean()),
        "xy_std_q90_max": float(np.quantile(sa, 0.9, axis=1).max()),
        "xy_energy_mean": float(np.quantile(en, 0.9, axis=1).mean()),
    }
    return out


def csi_features(d):
    """Per-minute CSI features from one cache npz."""
    x = d["csi"].astype(np.float32)                     # (nW,128,52)
    v = d["csi_valid"].astype(bool)
    dur = d["win_dur"].astype(np.float32)
    if not v.any():
        return {c: np.nan for c in CSI_ALL}
    xv = x[v]
    dv = dur[v]
    rv = _rollvar(np.log1p(xv), ROLL_W)
    # STFT band powers: detrended amplitude, per subcarrier, mean |Z|^2
    xc = xv - xv.mean(axis=1, keepdims=True)
    stft_bands = np.zeros((len(xv), len(STFT_BANDS)), np.float32)
    for i, (sig, dur_i) in enumerate(zip(xc, dv)):
        fs = 128.0 / max(dur_i, 1e-3)
        f, _, Z = sps.stft(sig.T, fs=fs, nperseg=32, noverlap=24,
                           padded=False, boundary=None)
        pw = (np.abs(Z) ** 2).mean(axis=(0, 2))         # (freq,)
        for b, (lo, hi) in enumerate(STFT_BANDS):
            band = (f >= lo) & (f < hi)
            stft_bands[i, b] = pw[band].mean() if band.any() else 0.0
    out = {
        "csi_amp_mean": float(xv.mean()),
        "csi_amp_std": float(xv.std()),
        "csi_rollvar_mean": float(rv.mean()),
        "csi_rollvar_q90": float(np.quantile(rv, 0.9)),
        "csi_stft_low": float(np.log1p(stft_bands[:, 0].mean())),
        "csi_stft_med": float(np.log1p(stft_bands[:, 1].mean())),
        "csi_stft_high": float(np.log1p(stft_bands[:, 2].mean())),
    }
    return out


def extract_features():
    """One row per cached minute: features + split/label."""
    index = T.load_index()
    acts = pd.read_csv(OUT / "snr_minutes.csv")[
        ["split", "minute", "activity"]]
    rows = []
    paths = list(index.sort_values("rec_id").itertuples(index=False))
    for i, row in enumerate(paths):
        d = np.load(row.path)
        rec = {"split": row.split, "minute": Path(row.path).stem,
               "label": int(row.label), "placement": row.placement}
        rec.update(radar_features(d))
        rec.update(csi_features(d))
        rows.append(rec)
        if (i + 1) % 300 == 0:
            print(f"[{i+1}/{len(paths)}]", flush=True)
    df = pd.DataFrame(rows)
    df = df.merge(acts, on=["split", "minute"], how="left")
    df["activity"] = df["activity"].where(
        df["activity"].notna(),
        pd.Series(np.where(df["label"] == 1, "occupied", "empty"),
                  index=df.index))
    df.to_csv(OUT / "feature_experiments_features.csv", index=False)
    print("wrote", OUT / "feature_experiments_features.csv",
          f"({len(df)} minutes)")
    return df


# ---------------------------------------------------------------------------
# Rules / metrics helpers
# ---------------------------------------------------------------------------
def _ba_threshold(y, score):
    """Threshold maximizing balanced accuracy; picks middle of the tie
    band and the better of score-up / score-down direction."""
    s = np.asarray(score, float)
    y = np.asarray(y, int)
    best = (-1.0, 1.0, 0.0)      # ba, sign, threshold
    for sign in (1.0, -1.0):
        u = np.unique(s)
        ts = np.r_[-np.inf, (u[:-1] + u[1:]) / 2, np.inf]
        pred = (sign * s[:, None]) >= ts[None, :]
        tpr = pred[y == 1].mean(axis=0)
        fpr = pred[y == 0].mean(axis=0)
        ba = (tpr + 1 - fpr) / 2
        i = np.flatnonzero(np.isclose(ba, ba.max()))
        i = i[len(i) // 2]
        if ba[i] > best[0]:
            best = (float(ba[i]), sign, float(ts[i]))
    return best[1], best[2]      # sign, threshold


def or_rule_fit(Xv, yv, scales):
    """Two-threshold OR rule calibrated on validation."""
    u0 = np.unique(Xv[:, 0]); u1 = np.unique(Xv[:, 1])
    t0s = np.r_[-np.inf, (u0[:-1] + u0[1:]) / 2, np.inf]
    t1s = np.r_[-np.inf, (u1[:-1] + u1[1:]) / 2, np.inf]
    best = (-1.0, np.inf, 0.0, 0.0)
    for a in t0s:
        pr = Xv[:, 0] >= a
        for b in t1s:
            pred = pr | (Xv[:, 1] >= b)
            tp = pred[yv == 1].mean(); fp = pred[yv == 0].mean()
            ba = (tp + 1 - fp) / 2
            if (ba, -fp) > (best[0], -best[1]):
                best = (ba, fp, float(a), float(b))
    ta, tb = best[2], best[3]

    def score(X):
        parts = []
        for j, t in enumerate((ta, tb)):
            if np.isneginf(t):
                return np.ones(len(X))
            parts.append((X[:, j] - t) / scales[j])
        return np.max(np.stack(parts), axis=0)
    return score, (ta, tb)


def evaluate_score(y, score, thr, sign=1.0):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, roc_auc_score,
                                 precision_recall_fscore_support)
    y = np.asarray(y, int)
    pred = (sign * np.asarray(score) >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, sign * np.asarray(score)))
        if len(np.unique(y)) == 2 else np.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float((f1[0] + f1[1]) / 2),
        "empty_recall": float(rc[0]), "occupied_recall": float(rc[1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def run(df):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = df["split"].eq("train").to_numpy()
    va = df["split"].eq("validation_minutes").to_numpy()
    te = df["split"].eq("test_minutes").to_numpy()
    ca = df["split"].eq("calibration_minutes").to_numpy()
    y = df["label"].to_numpy()

    metrics, preds_extra = [], {}
    results_rows = []

    def record(name, fset, scores, thr, sign=1.0):
        for sname, mask in (("validation_minutes", va),
                            ("test_minutes", te)):
            m = evaluate_score(y[mask], scores[mask], thr, sign)
            results_rows.append({"model": name, "feature_set": fset,
                                 "split": sname, **m})

    # ---- rules -------------------------------------------------------
    Xr = df["snr_db"].to_numpy()
    sgn, t = _ba_threshold(y[va], Xr[va])
    record("rule_radar", "radar_snr", sgn * Xr, t)
    preds_extra["rule_radar"] = sgn * Xr

    Xc = df["csi_rollvar_q90"].to_numpy()
    sgn, t = _ba_threshold(y[va], Xc[va])
    record("rule_csi", "csi_rollvar", sgn * Xc, t)
    preds_extra["rule_csi"] = sgn * Xc

    Xf = np.c_[df["snr_db"].to_numpy(), df["csi_rollvar_q90"].to_numpy()]
    sc = np.maximum(np.nanstd(Xf[tr], axis=0), 1e-9)
    orf, (ta, tb) = or_rule_fit(Xf[va], y[va], sc)
    record("rule_fusion_or", "fusion", orf(Xf), 0.0)
    preds_extra["rule_fusion_or"] = orf(Xf)
    print(f"OR rule: snr_db>={ta:.2f} | csi_rollvar_q90>={tb:.4f}")

    # ---- ML on feature sets ------------------------------------------
    models = {
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced",
                               max_iter=3000, random_state=SEED)),
        "gbdt": lambda: HistGradientBoostingClassifier(
            max_iter=300, random_state=SEED),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                          random_state=SEED)),
    }
    med = {c: np.nanmedian(df.loc[tr, c]) for c in
           df.columns if c.startswith(("csi_", "snr_", "xy_", "rd_"))}

    def fill(Xdf):
        return Xdf.fillna(med)

    for fset, cols in FEATURE_SETS.items():
        X = fill(df[cols]).to_numpy()
        for mname, ctor in models.items():
            try:
                model = ctor()
                model.fit(X[tr], y[tr])
                p = model.predict_proba(X)[:, 1]
            except Exception as e:
                print(f"  {mname}/{fset} failed: {e}")
                continue
            sgn, t = _ba_threshold(y[va], p[va])
            record(f"{mname}", fset, sgn * p, t)
            if fset == "fusion":
                preds_extra[f"{mname}_fusion"] = sgn * p
        print(f"  feature set done: {fset}", flush=True)

    # ---- DL: autoencoder pretrained on calibration --------------------
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    cols = FEATURE_SETS["fusion"]
    X = fill(df[cols]).to_numpy().astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = torch.tensor((X - mu) / sd)

    ae = nn.Sequential(nn.Linear(Z.shape[1], 32), nn.ReLU(),
                       nn.Linear(32, 8), nn.ReLU(),
                       nn.Linear(8, 32), nn.ReLU(),
                       nn.Linear(32, Z.shape[1]))
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    xa = Z[ca] if ca.any() else Z[tr]
    for ep in range(300):
        opt.zero_grad()
        loss = ((ae(xa) - xa) ** 2).mean()
        loss.backward(); opt.step()
    print(f"  AE pretrained on calibration, recon loss={loss.item():.4f}")
    with torch.no_grad():
        emb = nn.Sequential(*list(ae.children())[:3])(Z).numpy()
    head = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced",
                                            max_iter=3000,
                                            random_state=SEED))
    head.fit(emb[tr], y[tr])
    p = head.predict_proba(emb)[:, 1]
    sgn, t = _ba_threshold(y[va], p[va])
    record("ae_calib_logreg", "fusion", sgn * p, t)
    preds_extra["ae_calib"] = sgn * p

    met = pd.DataFrame(results_rows)
    met.to_csv(OUT / "feature_experiments_metrics.csv", index=False)

    pred_df = df[["split", "minute", "activity", "label"]].copy()
    for k, v in preds_extra.items():
        pred_df[k + "_score"] = v
    pred_df.to_csv(OUT / "feature_experiments_predictions.csv",
                   index=False)
    return met, pred_df, preds_extra


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_bars(met):
    val = met[met.split == "validation_minutes"].set_index(
        ["feature_set", "model"])
    te = met[met.split == "test_minutes"].set_index(
        ["feature_set", "model"])
    fsets = list(FEATURE_SETS)
    mnames = ["rule_radar", "rule_csi", "rule_fusion_or",
              "logreg", "gbdt", "mlp", "ae_calib_logreg"]
    colors = {"rule_radar": "#8c8c8c", "rule_csi": "#8c8c8c",
              "rule_fusion_or": "#8c8c8c", "logreg": "#1b9e77",
              "gbdt": "#7570b3", "mlp": "#d95f02",
              "ae_calib_logreg": "#e7298a"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (tbl, ttl) in zip(axes, [(val, "validation"),
                                   (te, "test")]):
        bw = 0.12
        for mi, m in enumerate(mnames):
            xs, ys = [], []
            for fs in fsets:
                key = (fs, m)
                # rules only live on their own feature sets
                if key in tbl.index:
                    xs.append(fsets.index(fs) + (mi - 3) * bw)
                    ys.append(tbl.loc[key, "balanced_acc"])
            if xs:
                ax.bar(xs, ys, bw * 0.9, color=colors[m], label=m,
                       alpha=0.9)
        ax.set_xticks(range(len(fsets)))
        ax.set_xticklabels(fsets, rotation=30, ha="right")
        ax.set_title(f"balanced accuracy — {ttl}")
        ax.axhline(0.5, ls="--", c="#bbb")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("balanced accuracy")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Feature-set x model grid (minute level)")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "feature_exp_ba.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "feature_exp_ba.png")


def plot_scatter(df, or_cuts):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for ax, sp in zip(axes, SPLITS):
        g = df[df.split == sp]
        for act, col in ACT_COLORS.items():
            h = g[g.activity == act]
            ax.scatter(h["snr_db"],
                       h["csi_rollvar_q90"].clip(lower=1e-6), c=col,
                       s=13, alpha=0.6, label=act)
        if or_cuts:
            ta, tb = or_cuts
            ax.axvline(ta, ls="--", c="#555", lw=1)
            ax.axhline(tb, ls="--", c="#555", lw=1)
        ax.set_title(SPLIT_XLABELS[sp])
        ax.set_xlabel("radar SNR (dB)")
        ax.set_xlim(1, None)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("CSI rolling-var q90")
    axes[0].set_yscale("log")
    axes[-1].legend(markerscale=1.6)
    fig.suptitle("Radar SNR vs CSI rolling variance "
                 "(dashed = fusion OR rule)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "feature_exp_scatter.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "feature_exp_scatter.png")


def plot_roc(df, scores, met):
    from sklearn.metrics import roc_auc_score, roc_curve
    names = ["rule_radar", "rule_csi", "rule_fusion_or",
             "logreg_fusion", "mlp_fusion", "ae_calib"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, sp in zip(axes, ["validation_minutes", "test_minutes"]):
        mask = (df.split == sp).to_numpy()
        yy = df["label"].to_numpy()[mask]
        for n in names:
            if n not in scores:
                continue
            s = scores[n][mask]
            fpr, tpr, _ = roc_curve(yy, s)
            ax.plot(fpr, tpr,
                    label=f"{n}: {roc_auc_score(yy, s):.3f}")
        ax.plot([0, 1], [0, 1], "--", c="#aaa")
        ax.set(title=SPLIT_XLABELS[sp], xlabel="FPR", ylabel="TPR",
               xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("ROC — selected models (minute level)")
    fig.tight_layout()
    fig.savefig(FIG / "feature_exp_roc.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "feature_exp_roc.png")


def main():
    df = extract_features()
    met, pred_df, scores = run(df)
    # recover OR cut for the scatter plot
    va = df["split"].eq("validation_minutes").to_numpy()
    y = df["label"].to_numpy()
    Xf = np.c_[df["snr_db"].to_numpy(),
               df["csi_rollvar_q90"].to_numpy()]
    tr = df["split"].eq("train").to_numpy()
    sc = np.maximum(np.nanstd(Xf[tr], axis=0), 1e-9)
    _, or_cuts = or_rule_fit(Xf[va], y[va], sc)
    plot_bars(met)
    plot_scatter(df, or_cuts)
    plot_roc(df, scores, met)
    best = (met[met.split == "validation_minutes"]
            .sort_values("balanced_acc", ascending=False).head(10))
    print("\nTop-10 on validation:")
    print(best[["model", "feature_set", "balanced_acc", "auc"]]
          .to_string(index=False))
    sel = best.iloc[0]
    row = met[(met.model == sel.model) & (met.feature_set == sel.feature_set)
              & (met.split == "test_minutes")]
    (OUT / "feature_experiments_selected.json").write_text(json.dumps({
        "selected_by": "validation balanced_acc",
        "model": sel.model, "feature_set": sel.feature_set,
        "val_balanced_acc": float(sel.balanced_acc),
        "test": row.iloc[0].to_dict() if len(row) else None}, indent=2))
    print("wrote", OUT / "feature_experiments_selected.json")


if __name__ == "__main__":
    main()
