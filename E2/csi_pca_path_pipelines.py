"""PCA-path-length scores on top of each CSI processing pipeline.

For every pipeline the raw CSI amplitude window (128 steps x 52 subcarriers)
is first transformed into a per-step matrix, then each step is projected
onto a 5-PC basis (standardized; PCA fit on train/ steps ONLY). The score
for a window is the total distance of the projected trajectory; the minute
score is the mean over valid CSI windows.

Pipelines:
  raw_amp    log1p amplitude (baseline)
  rollvar    log1p causal trailing variance (w=20)
  stft       log1p STFT motion-band (0.5-8 Hz) power per time bin
  median3    amp - trailing median-3 residual
  ema05      amp - EMA(a=.5) residual
  hampel5    amp - Hampel-5 residual
  diff       adjacent-step differences

Rule-based models only:
  thr_ba   threshold maximizing balanced accuracy on train (either sign)
  thr_q    quantile rule: occupied = score beyond the train-empty 90th pct
           (or below the 10th pct if direction is inverted on train)
  vote     majority vote of the 7 thr_ba pipeline decisions
  fusion_or  OR rule: radar snr_db >= ta OR best-path-score >= tb
             (both cutoffs picked on validation; best pipeline = val BA)

Outputs: csi_pca_path_minutes.csv, csi_pca_path_metrics.csv,
figs/csi_pca_path_ba.png, figs/csi_pca_path_fusion.png
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
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import train as T

OUT = C.OUTPUT_DIR
FIG = OUT / "figs"
SEED = 42
N_PC = 5
PCA_CAP = 200_000          # rows to fit each PCA on
STEP_CAP = 2_000           # rows sampled per minute for PCA fitting
ROLL_W = 20
MOTION_BAND = (0.5, 8.0)
_DC_ROWS = (10, 14)
_NEAR_RANGE = 2
PIPELINES = ["raw_amp", "rollvar", "stft", "median3", "ema05",
             "hampel5", "diff"]
SPLITS = ["train", "validation_minutes", "test_minutes"]
ACT_COLORS = {"empty": "#000000", "sleep": "#F2C500",
              "present": "#D62728"}
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Transforms: (128,52) amplitude -> (T',52) per-step feature matrix
# ---------------------------------------------------------------------------
def _rollvar(x, w):
    z = np.zeros((1, x.shape[1]), np.float32)
    cs = np.concatenate([z, np.cumsum(x, axis=0)])
    cs2 = np.concatenate([z, np.cumsum(x * x, axis=0)])
    hi = np.arange(1, x.shape[0] + 1)
    lo = np.clip(hi - w, 0, None)
    cnt = (hi - lo).astype(np.float32)[:, None]
    mean = (cs[hi] - cs[lo]) / cnt
    return np.clip((cs2[hi] - cs2[lo]) / cnt - mean * mean, 0, None)


def _ema(x, a=0.5):
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out


def _hampel(x, w=5, k=3.0):
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(x, w, axis=0)          # (T-w+1, S, w)
    med = np.median(win, axis=2)
    mad = np.median(np.abs(win - med[:, :, None]), axis=2)
    bad = np.abs(x[w - 1:] - med) > k * 1.4826 * np.maximum(mad, 1e-6)
    out = x.copy()
    out[w - 1:] = np.where(bad, med, x[w - 1:])
    return out


def transforms(x, dur):
    a = np.log1p(x)
    tr = {"raw_amp": a,
          "rollvar": np.log1p(_rollvar(a, ROLL_W)),
          "median3": a - median_filter(a, size=(3, 1), mode="nearest"),
          "ema05": a - _ema(a),
          "hampel5": a - _hampel(a),
          "diff": np.diff(a, axis=0)}
    xc = a - a.mean(axis=0, keepdims=True)
    fs = 128.0 / max(float(dur), 1e-3)
    f, _, Z = sps.stft(xc.T, fs=fs, nperseg=32, noverlap=24,
                       padded=False, boundary=None)
    band = (f >= MOTION_BAND[0]) & (f < MOTION_BAND[1])
    if band.any() and Z.shape[2] > 0:
        tr["stft"] = np.log1p(np.abs(Z[:, band, :]).mean(axis=1).T)  # (T',52)
    else:
        tr["stft"] = np.zeros((0, a.shape[1]), np.float32)
    return tr


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


def index_windows(row):
    d = np.load(row.path)
    return d, np.flatnonzero(d["csi_valid"].astype(bool))


# ---------------------------------------------------------------------------
# PCA basis per pipeline (fit on train steps only)
# ---------------------------------------------------------------------------
def fit_pca(index):
    train = index[index.split == "train"]
    buf = {p: [] for p in PIPELINES}
    for i, row in enumerate(train.itertuples(index=False)):
        d, ok = index_windows(row)
        for w in ok:
            for p, mat in transforms(d["csi"][w].astype(np.float32),
                                     d["win_dur"][w]).items():
                if len(mat) > STEP_CAP:
                    sel = rng.choice(len(mat), STEP_CAP, replace=False)
                    mat = mat[sel]
                buf[p].append(mat)
        if (i + 1) % 100 == 0:
            print(f"  pca sample [{i+1}/{len(train)}]", flush=True)
    bases = {}
    for p in PIPELINES:
        X = np.concatenate(buf[p]).astype(np.float32)
        X = X[np.isfinite(X).all(axis=1)]
        if len(X) > PCA_CAP:
            X = X[rng.choice(len(X), PCA_CAP, replace=False)]
        mu, sd = X.mean(0), np.maximum(X.std(0), 1e-9)
        pca = PCA(n_components=N_PC, random_state=SEED)
        pca.fit((X - mu) / sd)
        bases[p] = (mu, sd, pca)
        ev = " ".join(f"{v:.2f}" for v in pca.explained_variance_ratio_)
        print(f"  {p}: PCA on {len(X)} steps  ev=[{ev}]", flush=True)
    return bases


def path_length(mat, basis):
    mu, sd, pca = basis
    z = pca.transform((mat - mu) / sd)
    return float(np.linalg.norm(np.diff(z, axis=0), axis=1).sum())


def extract(index, bases):
    acts = pd.read_csv(OUT / "snr_minutes.csv")[["split", "minute",
                                               "activity"]]
    rows = []
    recs = list(index[index.split.isin(SPLITS + ["calibration_minutes"])]
                .sort_values("rec_id").itertuples(index=False))
    for i, row in enumerate(recs):
        d, ok = index_windows(row)
        rec = {"split": row.split, "minute": Path(row.path).stem,
               "label": int(row.label),
               "snr_db": snr_minute(d["radar"][:, :, 0].reshape(-1, 24, 24)),
               "n_csi_valid": int(ok.size)}
        acc = {p: [] for p in PIPELINES}
        for w in ok:
            for p, mat in transforms(d["csi"][w].astype(np.float32),
                                     d["win_dur"][w]).items():
                if len(mat) >= 2 and np.isfinite(mat).all():
                    acc[p].append(path_length(mat, bases[p]))
        for p in PIPELINES:
            rec[p] = float(np.mean(acc[p])) if acc[p] else np.nan
        rows.append(rec)
        if (i + 1) % 300 == 0:
            print(f"[{i+1}/{len(recs)}]", flush=True)
    df = pd.DataFrame(rows).merge(acts, on=["split", "minute"], how="left")
    df.to_csv(OUT / "csi_pca_path_minutes.csv", index=False)
    print(f"wrote csi_pca_path_minutes.csv ({len(df)} minutes)")
    return df


# ---------------------------------------------------------------------------
# Rule-based evaluation
# ---------------------------------------------------------------------------
def _thr_ba(y, s):
    """Threshold + direction maximizing BA on train."""
    y = np.asarray(y, int); s = np.asarray(s, float)
    best = (-1.0, 1.0, 0.0)
    for sign in (1.0, -1.0):
        u = np.unique(s)
        ts = np.r_[-np.inf, (u[:-1] + u[1:]) / 2, np.inf]
        pred = (sign * s[:, None]) >= ts[None, :]
        tpr = pred[y == 1].mean(axis=0)
        fpr = pred[y == 0].mean(axis=0)
        ba = (tpr + 1 - fpr) / 2
        i = np.flatnonzero(np.isclose(ba, ba.max()))[0]
        if ba[i] > best[0]:
            best = (float(ba[i]), sign, float(ts[i]))
    return best[1], best[2]


def _thr_quantile(y, s):
    """Direction from train class medians; cutoff at the empty-class
    90th (or 10th) percentile on train."""
    y = np.asarray(y, int); s = np.asarray(s, float)
    sign = 1.0 if np.median(s[y == 1]) >= np.median(s[y == 0]) else -1.0
    q = 0.9 if sign > 0 else 0.1
    return sign, float(np.quantile(s[y == 0], q))


def _metrics(y, s, thr, sign):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, roc_auc_score)
    y = np.asarray(y, int)
    pred = (sign * np.asarray(s) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)),
            "auc": float(roc_auc_score(y, sign * np.asarray(s)))
            if len(np.unique(y)) == 2 else np.nan,
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_acc": float(balanced_accuracy_score(y, pred)),
            "empty_recall": float(tn / max(tn + fp, 1)),
            "occupied_recall": float(tp / max(tp + fn, 1))}


def evaluate(df):
    y = df["label"].to_numpy()
    tr = df["split"].eq("train").to_numpy()
    va = df["split"].eq("validation_minutes").to_numpy()
    te = df["split"].eq("test_minutes").to_numpy()
    rows, preds = [], {}
    for p in PIPELINES:
        ok = df[p].notna().to_numpy()
        preds[p] = {}
        for rule, fit in (("thr_ba", _thr_ba), ("thr_q", _thr_quantile)):
            sgn, thr = fit(y[tr & ok], df.loc[tr & ok, p])
            for sp, m in (("train", tr), ("validation_minutes", va),
                          ("test_minutes", te)):
                mm = m & ok
                rows.append({"pipeline": p, "rule": rule, "split": sp,
                             "thr": thr, "sign": sgn,
                             **_metrics(y[mm], df.loc[mm, p], thr, sgn)})
            if rule == "thr_ba":
                preds[p] = (sgn, thr)
    # majority vote of the 7 thr_ba decisions
    votes = np.zeros(len(df))
    for p, (sgn, thr) in preds.items():
        ok = df[p].notna().to_numpy()
        votes[ok] += (sgn * df.loc[ok, p] >= thr).astype(int)
    vote_score = votes / len(preds)
    for sp, m in (("train", tr), ("validation_minutes", va),
                  ("test_minutes", te)):
        rows.append({"pipeline": "_vote7", "rule": "majority", "split": sp,
                     "thr": 0.5, "sign": 1.0,
                     **_metrics(y[m], vote_score[m], 0.5, 1.0)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "csi_pca_path_metrics.csv", index=False)
    return res


def fusion(df, best):
    y = df["label"].to_numpy()
    tr = df["split"].eq("train").to_numpy()
    va = df["split"].eq("validation_minutes").to_numpy()
    te = df["split"].eq("test_minutes").to_numpy()
    X = np.c_[df["snr_db"], df[best]]
    sc = np.maximum(np.nanstd(X[tr], axis=0), 1e-9)
    sgn_r, thr_r = _thr_ba(y[va], df.loc[va, "snr_db"])

    mv = va & np.isfinite(X).all(axis=1)
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
    print(f"fusion OR cuts: snr>={ta:.2f} | {best}>={tb:.2f}")

    def or_score(Xm):
        v = np.stack([(Xm[:, 0] - ta) / sc[0],
                      (Xm[:, 1] - tb) / sc[1]])
        v[1, ~np.isfinite(v[1])] = -np.inf
        return np.max(v, axis=0)

    rows = []
    for sp, m in (("validation_minutes", va), ("test_minutes", te)):
        rows.append({"model": "rule_radar", "split": sp,
                     **_metrics(y[m], df.loc[m, "snr_db"], thr_r, sgn_r)})
        s = or_score(X[m].copy())
        miss = ~np.isfinite(X[m, 1])
        s[miss] = sgn_r * X[m, 0][miss] - thr_r
        rows.append({"model": "fusion_or+fallback", "split": sp,
                     **_metrics(y[m], s, 0.0, 1.0)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "csi_pca_path_fusion.csv", index=False)
    return out, (ta, tb)


# ---------------------------------------------------------------------------
def plot_ba(res):
    sub = res[(res.rule == "thr_ba") & (res.split.isin(SPLITS))]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(PIPELINES))
    cols = {"train": "#4c72b0", "validation_minutes": "#55a868",
            "test_minutes": "#c44e52"}
    for k, sp in enumerate(SPLITS):
        g = sub[sub.split == sp].set_index("pipeline").reindex(PIPELINES)
        ax.bar(x + (k - 1) * 0.26, g.balanced_acc, 0.24,
               label=sp.replace("_minutes", ""), color=cols[sp])
        for xi, v in zip(x + (k - 1) * 0.26, g.balanced_acc):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7,
                    color=cols[sp])
    ax.set_xticks(x); ax.set_xticklabels(PIPELINES)
    ax.axhline(0.5, ls="--", c="#bbb")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("balanced accuracy (thr_ba, fit on train)")
    ax.set_title("PCA path length per CSI pipeline — presence prediction")
    ax.legend(); ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "csi_pca_path_ba.png", dpi=150)
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
        ax.set_xlabel("radar SNR (dB)"); ax.grid(alpha=0.2)
    axes[0].set_ylabel(f"CSI {best} path length")
    axes[0].set_yscale("log")
    axes[-1].legend(markerscale=1.5)
    fig.suptitle(f"Fusion: SNR + {best} PCA path (dashed = OR rule)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "csi_pca_path_fusion.png", dpi=150)
    plt.close(fig)


def main():
    index = T.load_index()
    bases = fit_pca(index)
    df = extract(index, bases)
    res = evaluate(df)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    v = res[(res.rule == "thr_ba") & (res.split == "validation_minutes")]
    best = v.sort_values("balanced_acc", ascending=False).iloc[0].pipeline
    print(f"\nbest pipeline on validation (thr_ba): {best}")
    fus, or_cuts = fusion(df, best)
    print(fus.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    plot_ba(res)
    plot_fusion(df, best, or_cuts)
    print("figs ->", FIG)


if __name__ == "__main__":
    main()
