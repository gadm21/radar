"""Deep-PCA figures + report for E1/outputs/pca_deep.

Reads desc_<split>_5s.csv, sep_<split>.csv, analysis_pca.json,
pc_dots_<split>.npz  ->  figs_pca/*.png + REPORT_PCA.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402
import pandas as pd                           # noqa: E402

E1 = Path(__file__).resolve().parent
OUT = E1 / "outputs" / "pca_deep"
FIG = OUT / "figs_pca"
FIG.mkdir(parents=True, exist_ok=True)
CLS = ["empty", "occ"]
CMAP = {0: "#4c9be0", 1: "#e0704c"}
SPLITS = ["train", "val", "calib", "testml"]


def load_desc():
    return {s: pd.read_csv(OUT / f"desc_{s}_5s.csv")
            for s in SPLITS if (OUT / f"desc_{s}_5s.csv").exists()}


# ---------------------------------------------------------------------------
def fig_ev():
    ana = json.loads((OUT / "analysis_pca.json").read_text())
    ds = [k for k in ("train", "testml") if k in ana]
    fig, axes = plt.subplots(len(ds), 2, figsize=(11, 4 * len(ds)),
                             squeeze=False)
    for r, k in enumerate(ds):
        for c, view in enumerate(("csi", "radar")):
            ev = np.array(ana[k]["ev"][view])
            ax = axes[r][c]
            ax.bar(range(1, len(ev) + 1), ev, color="#555")
            ax.plot(range(1, len(ev) + 1), np.cumsum(ev), "o-",
                    color=CMAP[1], label="cum")
            ax.set_title(f"{k} / {view}  (PC1-3 = "
                         f"{ev[:3].sum():.0%})")
            ax.set_xlabel("PC"); ax.set_ylabel("explained var")
            ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "ev_scree.png", dpi=140)
    plt.close(fig)


def fig_scatter(dfs):
    views = [("c", "csi"), ("r", "radar"), ("j", "joint")]
    sp = [s for s in SPLITS if s in dfs]
    fig, axes = plt.subplots(len(views), len(sp),
                             figsize=(4.2 * len(sp), 4 * len(views)),
                             squeeze=False)
    for c, (p, vn) in enumerate(views):
        for r, s in enumerate(sp):
            ax = axes[c][r]
            df = dfs[s]
            sub = df.sample(min(4000, len(df)), random_state=0)
            xa = f"{p}_mean0"
            ya = f"{p}_mean3" if p == "j" else f"{p}_mean1"
            for lab in (0, 1):
                d = sub[sub.label == lab]
                ax.scatter(d[xa], d[ya], s=4,
                           alpha=.35, color=CMAP[lab],
                           label=CLS[lab])
            ax.set_title(f"{s} / {vn}" +
                         (" csi-PC1 vs rad-PC1" if p == "j"
                          else " PC1-PC2"))
            if r == 0:
                ax.set_ylabel("rad-PC1" if p == "j" else "PC2")
            if c == len(views) - 1:
                ax.set_xlabel("csi-PC1" if p == "j" else "PC1")
            if r == len(sp) - 1:
                ax.legend(markerscale=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "pc_scatter.png", dpi=140)
    plt.close(fig)


def fig_pcv(dfs):
    cols = ["j_std0", "j_std3", "j_speed"]
    sp = [s for s in SPLITS if s in dfs]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4.5),
                             sharey=False)
    for ax, col in zip(axes, cols):
        data, ticks, labs = [], [], []
        for s in sp:
            for lab in (0, 1):
                data.append(dfs[s][dfs[s].label == lab][col].dropna())
                ticks.append(f"{s}\n{CLS[lab]}")
                labs.append(lab)
        bp = ax.boxplot(data, tick_labels=ticks, showfliers=False,
                        patch_artist=True, widths=.6)
        for patch, lab in zip(bp["boxes"], labs):
            patch.set_facecolor(CMAP[lab])
        ax.set_title(f"windowed {col} (compressed PC var)")
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "pcv_box.png", dpi=140)
    plt.close(fig)


def fig_traj():
    sp = [s for s in SPLITS if (OUT / f"pc_dots_{s}.npz").exists()]
    fig, axes = plt.subplots(3, len(sp), figsize=(4 * len(sp), 9),
                             squeeze=False)
    for c, s in enumerate(sp):
        z = np.load(OUT / f"pc_dots_{s}.npz")
        # pick one rec per class (label via id/name or desc csv)
        d = pd.read_csv(OUT / f"desc_{s}_5s.csv")
        labs = d.groupby("id").label.first()
        for r, lab in enumerate((0, 1)):
            ax = axes[r][c]
            ids = labs[labs == lab].index
            if not len(ids):
                ax.set_title(f"{s} {CLS[lab]} — none", fontsize=8)
                ax.axis("off")
                continue
            rid = ids[len(ids) // 2]
            a = z[rid]                     # (T,7) ok+6pc
            ok = a[:, 0] > 0
            seg = a[ok][60:60 + 90]        # a ~90s stretch
            ax = axes[r][c]
            sc = ax.scatter(seg[:, 1], seg[:, 2], c=np.arange(len(seg)),
                            cmap="viridis", s=8)
            ax.plot(seg[:, 1], seg[:, 2], "k-", lw=.4, alpha=.5)
            ax.set_title(f"{s} {CLS[lab]}\n{rid.split('/')[-1][:22]}",
                         fontsize=8)
            if r == 0:
                ax.set_ylabel("csi-PC2")
            ax.set_xlabel("csi-PC1")
        # third row: joint PC (csi1 vs rad1)
        ax = axes[2][c]
        for lab, mk in ((0, "o"), (1, "s")):
            ids = labs[labs == lab].index
            if not len(ids):
                continue
            rid = ids[len(ids) // 3]
            a = z[rid]
            ok = a[:, 0] > 0
            seg = a[ok][:400]
            ax.scatter(seg[:, 1], seg[:, 4], s=4, alpha=.4,
                       color=CMAP[lab], label=f"{CLS[lab]}")
        ax.set_title(f"{s}: csi-PC1 vs rad-PC1", fontsize=8)
        ax.set_xlabel("csi-PC1"); ax.set_ylabel("rad-PC1")
        ax.legend(markerscale=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "trajectories.png", dpi=140)
    plt.close(fig)


def fig_clusters(dfs):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                             gridspec_kw={"height_ratios": [1, 2]})
    ana = json.loads((OUT / "analysis_pca.json").read_text())
    ks = [k for k in SPLITS if k in ana]
    ax = axes[0]
    x = np.arange(len(ks))
    ax.bar(x - .15, [ana[k]["purity"] for k in ks], .3,
           label="kmeans purity")
    ax.bar(x + .15, [ana[k]["ari"] for k in ks], .3, label="ARI")
    ax.bar(x + .45, [ana[k]["silhouette"] for k in ks], .3,
           label="silhouette")
    ax.set_xticks(x, ks); ax.legend(); ax.set_ylim(0, 1)
    ax.set_title("unsupervised cluster quality (kmeans/gmm on train "
                 "joint-PC space)")
    ax = axes[1]
    df = dfs.get("val", dfs["train"])
    df = df[df.id == df.id.unique()[len(df.id.unique()) // 2]]
    t = df.s0.values
    ax.scatter(t, df.clu_kmmargin, c=df.clu_kmlabel, cmap="coolwarm",
               s=25)
    ax.plot(t, df.clu_kmmargin, "k-", lw=.4, alpha=.4)
    ax.set_ylabel("|dist1-dist0| (kmeans margin)")
    for i, (_, row) in enumerate(df.iterrows()):
        ax.axvspan(row.s0, row.s0 + 5,
                   color=CMAP[int(row.label)], alpha=.12)
    ax.set_title(f"cluster assignment over time — {df.id.iloc[0]} "
                 "(shading = true class)")
    ax.set_xlabel("second in recording"); ax.set_ylabel("joint-PC1 mean")
    fig.tight_layout()
    fig.savefig(FIG / "clusters.png", dpi=140)
    plt.close(fig)


def fig_fft(dfs):
    cols = [c for c in dfs["train"].columns if c.startswith("fft")]
    sp = [s for s in SPLITS if s in dfs]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        data, ticks, labs = [], [], []
        for s in sp:
            if col not in dfs[s]:
                continue
            for lab in (0, 1):
                data.append(dfs[s][dfs[s].label == lab][col].dropna())
                ticks.append(f"{s[:4]}\n{CLS[lab]}")
                labs.append(lab)
        if not data:
            ax.axis("off"); continue
        bp = ax.boxplot(data, tick_labels=ticks, showfliers=False,
                        patch_artist=True, widths=.6)
        for patch, lab in zip(bp["boxes"], labs):
            patch.set_facecolor(CMAP[lab])
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fft_bands.png", dpi=140)
    plt.close(fig)


def fig_auc():
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(4.5 * len(SPLITS),
                                                    5),
                             squeeze=False)
    for ax, s in zip(axes[0], SPLITS):
        f = OUT / f"sep_{s}.csv"
        if not f.exists():
            ax.axis("off"); continue
        df = pd.read_csv(f).head(12)
        ax.barh(range(len(df)), df.auc_eff,
                color=[CMAP[int(a < .5)] for a in df.auc])
        ax.set_yticks(range(len(df)), df.col, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(.5, color="k", lw=.5)
        ax.set_title(f"{s} top AUCs")
        ax.set_xlim(.45, 1.02)
    fig.tight_layout()
    fig.savefig(FIG / "desc_auc.png", dpi=140)
    plt.close(fig)


def fig_corr(dfs):
    df = dfs["train"]
    cols = [c for c in df.columns if c.startswith(("j_", "clu_", "fft"))]
    cols = [c for c in cols if df[c].std() > 0][:28]
    C = np.corrcoef(df[cols].values.T)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cols)), cols, fontsize=6)
    fig.colorbar(im, fraction=.045)
    ax.set_title("descriptor correlation (train)")
    fig.tight_layout()
    fig.savefig(FIG / "corr.png", dpi=140)
    plt.close(fig)


def fig_timeline(dfs):
    df = dfs.get("val", dfs["train"])
    ids = df.id.unique()
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    for ax, rid in zip(axes, ids[:3]):
        d = df[df.id == rid]
        ax.plot(d.s0, d.j_std0, ".-", ms=4, label="j_pcv1")
        ax.plot(d.s0, d.fft_snr_b0, ".-", ms=4, label="fft_snr_b0")
        ax.set_title(f"{rid} (label={CLS[int(d.label.iloc[0])]})",
                     fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "timeline.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def report(dfs):
    ana = json.loads((OUT / "analysis_pca.json").read_text())
    L = ["# E1 deep-PCA analysis (E1/pca_deep)",
         "",
         "## Preprocessing",
         "- per-second validity masks; invalid seconds excluded (no "
         "fabricated motion via interpolation)",
         "- StandardScaler + PCA fit on **train** seconds only "
         "(multilink: unsupervised fit on its own seconds)",
         "- windows W=5 s, hop 2 s, min coverage 3 s",
         "- FFT descriptors: radar SNR stream (all sets) + 100 Hz CSI "
         "amplitude FFT (legacy only)",
         "",
         "## Variance explained (first 3 PCs)"]
    for k, a in ana.items():
        L.append(f"- **{k}**: csi "
                 f"{np.sum(a['ev']['csi'][:3]):.1%} / "
                 f"{np.sum(a['ev']['csi']):.1%}@8,  radar "
                 f"{np.sum(a['ev']['radar'][:3]):.1%} / "
                 f"{np.sum(a['ev']['radar']):.1%}@8")
    L += ["", "## Unsupervised cluster quality (train joint-PC space)",
          "| split | kmeans purity | ARI | silhouette | dbscan noise "
          "| dbscan clusters |",
          "|---|---|---|---|---|---|"]
    for k, a in ana.items():
        L.append(f"| {k} | {a['purity']:.3f} | {a['ari']:.3f} | "
                 f"{a['silhouette']:.3f} | {a['dbscan_noise']:.3f} | "
                 f"{a['dbscan_clusters']} |")
    L += ["", "## Top descriptors per split (|AUC|)"]
    for s in SPLITS:
        f = OUT / f"sep_{s}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).head(10)
        L.append(f"### {s}")
        L.append("| descriptor | auc (eff) | dir | cohen d |")
        L.append("|---|---|---|---|")
        for _, r in df.iterrows():
            L.append(f"| {r.col} | {r.auc_eff:.3f} | "
                     f"{'+occ' if r.auc >= .5 else '+empty'} | "
                     f"{r.cohend:+.2f} |")
        L.append("")
    L += ["## Figures", "- figs_pca/ev_scree.png",
          "- figs_pca/pc_scatter.png", "- figs_pca/pcv_box.png",
          "- figs_pca/trajectories.png", "- figs_pca/clusters.png",
          "- figs_pca/fft_bands.png", "- figs_pca/desc_auc.png",
          "- figs_pca/corr.png", "- figs_pca/timeline.png", "",
          "## Caveats",
          "- multilink PCA is an unsupervised fit (no labels); cluster "
          "metrics there measure self-consistency.",
          "- legacy val/calib share the train-fit space: their "
          "purity/ARI are *transfer* metrics.",
          "- fftc_* exists only on legacy (raw 100 Hz CSI stream).",
          "- t_sleep/t1_sleep count as occupied (E2/common mapping)."]
    (OUT / "REPORT_PCA.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT / "REPORT_PCA.md")


def main():
    dfs = load_desc()
    fig_ev(); fig_scatter(dfs); fig_pcv(dfs); fig_traj()
    fig_clusters(dfs); fig_fft(dfs); fig_auc(); fig_corr(dfs)
    fig_timeline(dfs)
    report(dfs)


if __name__ == "__main__":
    main()
