"""CSI processing-pipeline comparison + fusion with radar SNR.

Pipelines (each produces one per-minute motion score from the cached
(nW,128,52) CSI amplitude windows):
  raw_amp    temporal std of amplitude (baseline)
  rollvar    causal trailing-window variance (w=20 on the 128-pt grid)
  stft       STFT band power of detrended amplitude (0.5-8 Hz motion band)
  median3    std of (amp - trailing median-3) residual
  ema05      std of (amp - EMA alpha=.5) residual
  hampel5    std of (amp - Hampel-5, 3*1.4826*MAD) residual
  diff       std of adjacent-step differences

Each pipeline score is tested per split: AUC + accuracy/balanced-acc at a
threshold fitted on train only. Minutes without valid CSI are excluded
from per-pipeline metrics (coverage reported); for fusion they fall back
to the radar rule.

Fusion: the validation-best CSI pipeline score + radar snr_db —
  fusion_or      two-threshold OR rule calibrated on validation
  fusion_logreg  logistic regression trained on train paired minutes

Outputs: csi_pipelines_minutes.csv, csi_pipelines_metrics.csv,
csi_pipelines_fusion.csv, figs/csi_pipelines_ba.png,
figs/csi_pipelines_fusion.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import train as T

OUT = C.OUTPUT_DIR
FIG = OUT / "figs"
SEED = 42
ROLL_W = 20
MOTION_BAND = (0.5, 8.0)          # Hz, at ~23 Hz effective grid rate
_DC_ROWS = (10, 14)
_NEAR_RANGE = 2
PIPELINES = ["raw_amp", "rollvar", "stft", "median3", "ema05",
             "hampel5", "diff"]
SPLITS = ["train", "validation_minutes", "test_minutes"]
ACT_COLORS = {"empty": "#000000", "sleep": "#F2C500",
              "present": "#D62728"}


# ---------------------------------------------------------------------------
# Per-window pipeline scores
# ---------------------------------------------------------------------------
def _rollvar(x, w):
    """Causal trailing variance along axis 0. x: (T,S)."""
    z = np.zeros((1, x.shape[1]), np.float32)
    cs = np.concatenate([z, np.cumsum(x, axis=0)])
    cs2 = np.concatenate([z, np.cumsum(x * x, axis=0)])
    hi = np.arange(1, x.shape[0] + 1)
    lo = np.clip(hi - w, 0, None)
    cnt = (hi - lo).astype(np.float32)[:, None]
    mean = (cs[hi] - cs[lo]) / cnt
    msq = (cs2[hi] - cs2[lo]) / cnt
    return np.clip(msq - mean * mean, 0, None)


def _ema(x, a=0.5):
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out


def _hampel(x, w=5, k=3.0):
    """Trailing-window Hampel on axis 0. x: (T,S)."""
    out = x.copy()
    for t in range(w - 1, len(x)):
        win = x[t - w + 1:t + 1]
        med = np.median(win, axis=0)
        mad = np.median(np.abs(win - med), axis=0)
        bad = np.abs(x[t] - med) > k * 1.4826 * np.maximum(mad, 1e-6)
        out[t] = np.where(bad, med, x[t])
    return out


def window_scores(x, dur):
    """x: (128,52) amplitude of one window; dur: seconds. -> dict."""
    a = np.log1p(x)
    s = {
        "raw_amp": float(a.std(axis=0).mean()),
        "rollvar": float(_rollvar(a, ROLL_W).mean()),
        "diff": float(np.diff(a, axis=0).std(axis=0).mean()),
    }
    xc = a - a.mean(axis=0, keepdims=True)
    fs = 128.0 / max(float(dur), 1e-3)
    f, _, Z = sps.stft(xc.T, fs=fs, nperseg=32, noverlap=24,
                       padded=False, boundary=None)
    pw = (np.abs(Z) ** 2).mean(axis=(0, 2))
    band = (f >= MOTION_BAND[0]) & (f < MOTION_BAND[1])
    s["stft"] = float(np.log1p(pw[band].mean())) if band.any() else 0.0
    for name, filt in (("median3", lambda v: median_filter(
                            v, size=(3, 1), mode="nearest")),
                       ("ema05", _ema),
                       ("hampel5", _hampel)):
        s[name] = float((a - filt(a)).std(axis=0).mean())
    return s


def snr_minute(rd_log):
    rd = np.expm1(rd_log.astype(np.float32))
    m = np.zeros((24, 24), bool)
    m[_DC_ROWS[0]:_DC_ROWS[1], :] = True
    m[:, :_NEAR_RANGE] = True
    flat = rd.reshape(len(rd), -1)
    med = np.median(flat, axis=1) + 1e-9
    peak = np.where(m[None], 0.0, rd).reshape(len(rd), -1).max(axis=1)
    s = 10.0 * np.log10(peak / med)
    s = s[np.isfinite(s)]
    return float(s.max()) if s.size else np.nan


def extract():
    index = T.load_index()
    acts = pd.read_csv(OUT / "snr_minutes.csv")[["split", "minute",
                                               "activity"]]
    rows = []
    recs = list(index[index.split.isin(SPLITS + ["calibration_minutes"])]
                .sort_values("rec_id").itertuples(index=False))
    for i, row in enumerate(recs):
        d = np.load(row.path)
        minute = Path(row.path).stem
        rec = {"split": row.split, "minute": minute,
               "label": int(row.label),
               "snr_db": snr_minute(d["radar"][:, :, 0].reshape(-1, 24, 24)),
               "n_windows": int(d["radar"].shape[0]),
               "n_csi_valid": int(d["csi_valid"].sum())}
        per_win = {p: [] for p in PIPELINES}
        for w in np.flatnonzero(d["csi_valid"].astype(bool)):
            s = window_scores(d["csi"][w].astype(np.float32),
                              d["win_dur"][w])
            for p, v in s.items():
                per_win[p].append(v)
        for p in PIPELINES:
            rec[p] = float(np.mean(per_win[p])) if per_win[p] else np.nan
        rows.append(rec)
        if (i + 1) % 300 == 0:
            print(f"[{i+1}/{len(recs)}]", flush=True)
    df = pd.DataFrame(rows).merge(acts, on=["split", "minute"], how="left")
    df.to_csv(OUT / "csi_pipelines_minutes.csv", index=False)
    print(f"wrote csi_pipelines_minutes.csv ({len(df)} minutes)")
    return df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _threshold_train(y, score):
    """BA-max threshold + direction fitted on train."""
    y = np.asarray(y, int); s = np.asarray(score, float)
    best = (-1.0, 1.0, 0.0)
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
    return best[1], best[2]


def _metrics(y, score, thr, sign):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, roc_auc_score)
    y = np.asarray(y, int)
    pred = (sign * np.asarray(score) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)),
            "auc": float(roc_auc_score(y, sign * np.asarray(score)))
            if len(np.unique(y)) == 2 else np.nan,
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_acc": float(balanced_accuracy_score(y, pred)),
            "empty_recall": float(tn / max(tn + fp, 1)),
            "occupied_recall": float(tp / max(tp + fn, 1))}


def evaluate_pipelines(df):
    y = df["label"].to_numpy()
    tr = df["split"].eq("train").to_numpy()
    rows = []
    for p in PIPELINES:
        ok = df[p].notna().to_numpy()
        sgn, thr = _threshold_train(y[tr & ok], df.loc[tr & ok, p])
        for sp in SPLITS:
            m = df["split"].eq(sp).to_numpy() & ok
            r = _metrics(y[m], df.loc[m, p], thr, sgn)
            rows.append({"pipeline": p, "split": sp, "thr": thr,
                         "sign": sgn, **r})
        rows.append({"pipeline": p, "split": "_coverage",
                     "thr": thr, "sign": sgn, "n": int(ok.sum()),
                     "auc": np.nan, "accuracy": np.nan,
                     "balanced_acc": np.nan, "empty_recall": np.nan,
                     "occupied_recall": np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "csi_pipelines_metrics.csv", index=False)
    return res


def fusion(df, best):
    """OR rule (val-calibrated) + logreg (train-fit): snr_db x best csi."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = df["label"].to_numpy()
    tr = df["split"].eq("train").to_numpy()
    va = df["split"].eq("validation_minutes").to_numpy()
    both = df["snr_db"].notna() & df[best].notna()
    X = np.c_[df["snr_db"], df[best]]
    # scales for OR margin score
    sc = np.maximum(np.nanstd(X[tr], axis=0), 1e-9)

    # radar-only rule threshold (val-calibrated) for fallback
    sgn_r, thr_r = _threshold_train(y[va], df.loc[va, "snr_db"])

    # --- OR rule on validation (paired minutes only for fitting) ------
    mv = va & both.to_numpy()
    u0 = np.unique(X[mv, 0]); u1 = np.unique(X[mv, 1])
    t0s = np.r_[-np.inf, (u0[:-1] + u0[1:]) / 2, np.inf]
    t1s = np.r_[-np.inf, (u1[:-1] + u1[1:]) / 2, np.inf]
    best_or = (-1.0, 0.0, 0.0)
    for a in t0s:
        pr = X[mv, 0] >= a
        for b in t1s:
            pred = pr | (X[mv, 1] >= b)
            ba = (pred[y[mv] == 1].mean() + 1 - pred[y[mv] == 0].mean()) / 2
            if ba > best_or[0]:
                best_or = (ba, float(a), float(b))
    ta, tb = best_or[1], best_or[2]
    print(f"fusion OR cuts: snr>={ta:.2f} | {best}>={tb:.4f}")

    def or_score(Xm):
        parts = []
        for j, t in enumerate((ta, tb)):
            col = Xm[:, j]
            if np.isneginf(t):
                return np.ones(len(Xm))
            v = (col - t) / sc[j]
            v[~np.isfinite(v)] = -np.inf
            parts.append(v)
        return np.max(np.stack(parts), axis=0)

    # --- logistic on train paired minutes -----------------------------
    mt = tr & both.to_numpy()
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, class_weight="balanced",
                                           max_iter=3000,
                                           random_state=SEED))
    clf.fit(X[mt], y[mt])
    p_all = clf.predict_proba(np.nan_to_num(X, nan=0.0))[:, 1]
    sgn_l, thr_l = _threshold_train(y[va & both.to_numpy()],
                                    p_all[va & both.to_numpy()])

    rows = []
    # radar rule alone (all minutes)
    for sp in ("validation_minutes", "test_minutes"):
        m = df["split"].eq(sp).to_numpy()
        rows.append({"model": "rule_radar", "split": sp,
                     **_metrics(y[m], df.loc[m, "snr_db"], thr_r, sgn_r)})
        # OR rule with radar fallback where CSI missing
        s = or_score(X[m].copy())
        miss = ~np.isfinite(X[m, 1])
        s[miss] = sgn_r * X[m, 0][miss] - thr_r  # radar-only fallback
        rows.append({"model": "fusion_or+fallback", "split": sp,
                     **_metrics(y[m], s, 0.0, 1.0)})
        # logreg on paired minutes only
        mp = m & both.to_numpy()
        rows.append({"model": "fusion_logreg (paired)", "split": sp,
                     **_metrics(y[mp], p_all[mp], thr_l, sgn_l)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "csi_pipelines_fusion.csv", index=False)
    return out, (ta, tb)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_ba(res):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(PIPELINES))
    cols = {"train": "#4c72b0", "validation_minutes": "#55a868",
            "test_minutes": "#c44e52"}
    for k, sp in enumerate(SPLITS):
        g = res[(res.split == sp)].set_index("pipeline").reindex(PIPELINES)
        ax.bar(x + (k - 1) * 0.26, g.balanced_acc, 0.24,
               label=sp.replace("_minutes", ""), color=cols[sp])
        for xi, v in zip(x + (k - 1) * 0.26, g.balanced_acc):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7,
                    color=cols[sp])
    ax.set_xticks(x); ax.set_xticklabels(PIPELINES)
    ax.axhline(0.5, ls="--", c="#bbb")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("balanced accuracy (train-fitted threshold)")
    ax.set_title("CSI pipelines — minute-level presence prediction")
    ax.legend()
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "csi_pipelines_ba.png", dpi=150)
    plt.close(fig)


def plot_fusion(df, best, or_cuts):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, sp in zip(axes, SPLITS):
        g = df[df.split == sp]
        for act, col in ACT_COLORS.items():
            h = g[g.activity == act]
            ax.scatter(h["snr_db"], h[best].clip(lower=1e-6), c=col,
                       s=14, alpha=0.65, label=act)
        ta, tb = or_cuts
        ax.axvline(ta, ls="--", c="#555", lw=1)
        ax.axhline(tb, ls="--", c="#555", lw=1)
        ax.set_title(sp.replace("_minutes", ""))
        ax.set_xlabel("radar SNR (dB)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(f"CSI {best} score")
    axes[0].set_yscale("log")
    axes[-1].legend(markerscale=1.5)
    fig.suptitle(f"Fusion: SNR + best CSI pipeline ({best}); "
                 "dashed = OR rule")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "csi_pipelines_fusion.png", dpi=150)
    plt.close(fig)


def main():
    df = extract()
    res = evaluate_pipelines(df)
    print(res[res.split != "_coverage"].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    print("\ncoverage:", res[res.split == "_coverage"]
          [["pipeline", "n"]].to_dict("records"))
    # best pipeline = highest validation BA
    v = res[res.split == "validation_minutes"]
    best = v.sort_values("balanced_acc", ascending=False).iloc[0].pipeline
    print(f"\nbest pipeline on validation: {best}")
    fus, or_cuts = fusion(df, best)
    print(fus.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    plot_ba(res)
    plot_fusion(df, best, or_cuts)
    print("figs ->", FIG)


if __name__ == "__main__":
    main()
