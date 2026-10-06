"""E1 figures: PCA dot clouds, compressed-PCA trajectories, descriptor
distributions, importances, mean maps, time series, correlations.

    python E1/plots.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np              # noqa: E402
import pandas as pd             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C              # noqa: E402

FIGS = C.FIGS_DIR
OUT = C.OUTPUT_DIR
CLS = ["empty", "occupied"]
CMAP = {0: "#4c9bd6", 1: "#e4572e"}


def _add_csim(df):
    """2-link mean channel columns (same derivation as analyze.py)."""
    df = df.copy()
    for s in C.CSI_LINK_COLS:
        df[f"csiM_{s}"] = df[[f"csi1_{s}", f"csi2_{s}"]].mean(axis=1)
    return df


def _save(fig, name):
    fig.savefig(FIGS / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(" ", name, flush=True)


# ---------------------------------------------------------------------------
def fig_pc_scatter():
    """PC1 vs PC2 dot clouds, subsampled, per dataset x modality."""
    import joblib
    rng = np.random.default_rng(0)
    # build panels: (row, title, array[label,sec,pc1,pc2,pc3])
    ml = np.load(OUT / "pc_dots_multilink.npz")
    old = np.load(OUT / "pc_dots_old.npz")
    rows = []
    for p in sorted(C.ML_CACHE.glob("*.npz")):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        ok = z["csi2_ok"]
        if ok.any():
            pcas = joblib.load(OUT / "pcas_multilink.joblib")
            pr = pcas["csi2"].transform(z["csi2_dots"][ok])
            rows.append(np.c_[np.full(ok.sum(),
                                      meta["label"] == "occupied"),
                              np.flatnonzero(ok),
                              np.full(ok.sum(), 0), pr])
    link2 = np.concatenate(rows)
    panels = [
        (0, "multilink — CSI link1 dots", ml["csi"]),
        (0, "multilink — radar RD dots", ml["rad"]),
        (0, "multilink — CSI link2 dots", link2),
        (1, "legacy minutes — CSI dots", old["csi"]),
        (1, "legacy minutes — radar RD dots", old["rad"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    col = {0: 0, 1: 0}
    for r, ttl, a in panels:
        ax = axes[r, col[r]]
        col[r] += 1
        take = rng.choice(len(a), min(12000, len(a)), replace=False)
        a = a[take]
        for lab in (0, 1):
            m = a[:, 0] == lab
            ax.scatter(a[m, 3], a[m, 4], s=2, alpha=.25,
                       c=CMAP[lab], label=CLS[lab], rasterized=True)
        ax.set_title(ttl)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(markerscale=6, fontsize=8)
    axes[1, 2].axis("off")
    fig.suptitle("Per-second PCA dots — class separation in PC space")
    _save(fig, "pc_dots_scatter.png")


def fig_trajectories():
    """Compressed-PCA dot trajectories over time (2-D PC1/PC2 paths,
    color = second within window) for a few windows of each class."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    W = 5
    for r, (tag, caches) in enumerate(
            (("pc_dots_multilink.npz", "ml"),
             ("pc_dots_old.npz", "old"))):
        z = np.load(OUT / tag)
        rng = np.random.default_rng(1)
        for lab in (0, 1):
            for k in range(2):
                ax = axes[r, k * 2 + (lab)]
                a = z["rad"] if k else z["csi"]
                # pick ONE recording (src col) of this class, then a
                # W-second run inside it
                srcs = np.unique(a[a[:, 0] == lab, 2])
                seg = np.empty((0, a.shape[1]))
                for _ in range(300):
                    s_ = srcs[rng.integers(len(srcs))]
                    rec = a[(a[:, 0] == lab) & (a[:, 2] == s_)]
                    if len(rec) == 0:
                        continue
                    t0 = rec[rng.integers(len(rec)), 1]
                    seg = rec[(rec[:, 1] >= t0) & (rec[:, 1] < t0 + W)]
                    if len(seg) >= max(3, W - 1):
                        break
                sc = ax.scatter(seg[:, 3], seg[:, 4],
                                c=seg[:, 1] - seg[:, 1].min(),
                                cmap="viridis", s=30)
                ax.plot(seg[:, 3], seg[:, 4], "k-", lw=.5, alpha=.5)
                ax.set_title(f"{'multilink' if tag.startswith('pc_dots_m') else 'legacy'} "
                             f"{'CSI' if k == 0 else 'radar'} — {CLS[lab]} "
                             f"({W}s window)", fontsize=9)
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("Compressed-PCA dot trajectories per 5 s window "
                 "(colour = second index)")
    _save(fig, "pc_trajectories.png")


def fig_pcv_box():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for r, W in enumerate(C.WIN_LIST):
        ml = pd.read_csv(OUT / f"features_multilink_{W}s.csv")
        old = pd.read_csv(OUT / f"features_old_{W}s_ext.csv")
        for c, (df, cols, lab_ok, ttl) in enumerate((
                (ml, [f"csi1_pcv{i}" for i in (1, 2, 3)],
                 ml.csi1_ok == 1, "ml CSI link1"),
                (ml, [f"rad_pcv{i}" for i in (1, 2, 3)],
                 ml.rad_ok == 1, "ml radar"),
                (old, [f"csi_pcv{i}" for i in (1, 2, 3)],
                 old.csi_ok == 1, "legacy CSI"))):
            ax = axes[r, c]
            d = df[lab_ok]
            data = [d.loc[d.label == lab, cols].values.ravel()
                    for lab in (0, 1)]
            data = [x[np.isfinite(x)] for x in data]
            ax.boxplot(data, tick_labels=CLS, showfliers=False)
            ax.set_title(f"PCV (window-dot variance), {ttl}, W={W}s")
            ax.set_ylabel("variance of PC projection")
    _save(fig, "pcv_box.png")


def fig_descriptor_dists():
    res = json.loads((OUT / "analysis.json").read_text())
    for ds_file, key, tag in (("features_multilink_5s.csv",
                               "multilink_5s/fusion", "multilink"),
                              ("features_old_5s_ext.csv",
                               "old_train_5s/fusion", "old_train"),
                              ("features_old_5s_ext.csv",
                               "old_test_5s/fusion", "old_test")):
        if key not in res:
            continue
        top = res[key]["top_k"][:8]
        df = pd.read_csv(OUT / ds_file)
        if tag == "multilink":
            df = _add_csim(df)
        top = [f for f in top if f in df.columns]
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        for ax, f in zip(axes.ravel(), top):
            for lab in (0, 1):
                v = df.loc[df.label == lab, f].dropna()
                ax.hist(v, bins=60, alpha=.55, density=True,
                        color=CMAP[lab], label=CLS[lab])
            ax.set_title(f, fontsize=9)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Top descriptor distributions — {tag} (5 s)")
        _save(fig, f"descriptor_dists_{tag}.png")


def fig_importance():
    for tag in ("ml5s", "oldtrain5s", "oldtest5s"):
        p = OUT / f"rankings_{tag}_fusion.csv"
        if not p.exists():
            continue
        r = pd.read_csv(p).head(18)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(r.feature[::-1], r.rf_imp[::-1], color="#3a6ea5")
        ax.set_title(f"RF importance, fusion features — {tag}")
        ax.set_xlabel("importance")
        _save(fig, f"importance_{tag}.png")


def fig_mean_maps():
    """Mean log-power RD/XY maps per class + class-difference panels.
    vsec/maps are already log1p — shown directly (expm1 saturates)."""
    fig, axes = plt.subplots(2, 6, figsize=(19, 6.5))
    # --- multilink row
    means = {0: np.zeros((2, 24, 24)), 1: np.zeros((2, 24, 24))}
    cnt = {0: 0, 1: 0}
    for p in sorted(C.ML_CACHE.glob("*.npz")):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        lab = int(meta["label"] == "occupied")
        m = z["radar_vsec"].astype(np.float32).mean(axis=0)
        means[lab] += np.stack([m[0], m[3]])
        cnt[lab] += 1
    for lab in (0, 1):
        means[lab] /= cnt[lab]
    for c, (lab, vi) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        ax = axes[0, c]
        im = ax.imshow(means[lab][vi], cmap="inferno", origin="lower")
        ax.set_title(f"ml {'RD' if vi == 0 else 'XY'} — {CLS[lab]}",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046)
    for c, vi in enumerate((0, 1)):
        ax = axes[0, 4 + c]
        d = means[1][vi] - means[0][vi]
        im = ax.imshow(d, cmap="RdBu_r", origin="lower",
                       vmin=-np.abs(d).max(), vmax=np.abs(d).max())
        ax.set_title(f"ml {'RD' if vi == 0 else 'XY'} — occ − empty",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046)
    # --- legacy test row
    means = {0: np.zeros((2, 24, 24)), 1: np.zeros((2, 24, 24))}
    cnt = {0: 0, 1: 0}
    for p in sorted((C.OLD_CACHE / "test_minutes").glob("*.npz")):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        lab = int(meta["label"])
        m = z["maps"].astype(np.float32).mean(axis=0)
        means[lab] += np.stack([m[0], m[3]])
        cnt[lab] += 1
    for lab in (0, 1):
        means[lab] /= cnt[lab]
    for c, (lab, vi) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        ax = axes[1, c]
        im = ax.imshow(means[lab][vi], cmap="inferno", origin="lower")
        ax.set_title(f"legacy {'RD' if vi == 0 else 'XY'} — {CLS[lab]}",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046)
    for c, vi in enumerate((0, 1)):
        ax = axes[1, 4 + c]
        d = means[1][vi] - means[0][vi]
        im = ax.imshow(d, cmap="RdBu_r", origin="lower",
                       vmin=-np.abs(d).max(), vmax=np.abs(d).max())
        ax.set_title(f"legacy {'RD' if vi == 0 else 'XY'} — occ − empty",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle("Mean log-power maps per class and class-difference maps")
    _save(fig, "mean_maps.png")


def fig_timeseries():
    ml = pd.read_csv(OUT / "features_multilink_5s.csv")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    for ax, col in zip(axes, ("snr_mean", "csi1_rv_mean", "csi1_pcv1")):
        for cap, g in ml.groupby("cap"):
            lab = "occupied" if g.label.iloc[0] else "empty"
            ax.plot(g.t_rel / 60, g[col].rolling(12, min_periods=1)
                    .median(), lw=.8, label=f"{cap.split('_')[1]} {lab}",
                    color=CMAP[g.label.iloc[0]], alpha=.75)
        ax.set_ylabel(col)
        ax.set_title(f"{col} over capture time (5 s windows, "
                     "1 min rolling median)")
        ax.legend(fontsize=7, ncol=4)
        ax.set_xlabel("minutes")
    _save(fig, "timeseries_ml.png")


def fig_corr_heatmap():
    for tag, file, key in (("multilink", "features_multilink_5s.csv",
                            "multilink_5s/fusion"),
                           ("old_train", "features_old_5s_ext.csv",
                            "old_train_5s/fusion")):
        res = json.loads((OUT / "analysis.json").read_text())
        if key not in res:
            continue
        cols = res[key]["top_k"]
        df = pd.read_csv(OUT / file)
        if tag == "multilink":
            df = _add_csim(df)
        df = df[[c for c in cols if c in df.columns]]
        r = df.corr()
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(df.columns)), df.columns, rotation=90,
                      fontsize=7)
        ax.set_yticks(range(len(df.columns)), df.columns, fontsize=7)
        fig.colorbar(im)
        ax.set_title(f"Top-K descriptor correlation — {tag}")
        _save(fig, f"corr_{tag}.png")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    print("[figs]", flush=True)
    fig_pc_scatter()
    fig_trajectories()
    fig_pcv_box()
    fig_descriptor_dists()
    fig_importance()
    fig_mean_maps()
    fig_timeseries()
    fig_corr_heatmap()


if __name__ == "__main__":
    main()
