"""Two-by-three hierarchical PCA scatter for the v3 frozen descriptors:
rows CSI / Radar, columns 1 s / 5 s / 15 s, multilink test points projected
through the train-fitted bases stored in frozen_v3's hierarchy.joblib."""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frozen_v3 import OUT, MOD_COLS, block5, window_vector
from physical_comparison import ROOT

DOTS = ROOT / "E1/outputs/pca_deep"
ASSETS = ROOT / "paper/icc_assets_v3"
ASSETS.mkdir(exist_ok=True)
BLUE, ORANGE = "#2C6B9E", "#C65A2E"
LEVELS = ("1 second", "5 seconds", "15 seconds")


def level_points(a, cols, h):
    """Per-level (n, 2) point arrays for one capture's dot matrix."""
    good = (a[:, 0] > .5) & np.isfinite(a[:, cols]).all(axis=1)
    l1 = a[good][:, cols]
    l2, l3 = [], []
    for s in range(0, len(a) - 4, 5):
        b = block5(a, cols, s)
        if b is not None:
            l2.append(h["p5"].transform(h["sc5"].transform(b.reshape(1, -1)))[0])
    for s in range(0, len(a) - 14, 15):
        v = window_vector(a, cols, s, h)[-2:]
        if np.isfinite(v).all():
            l3.append(v)
    return l1, np.asarray(l2).reshape(-1, 2), np.asarray(l3).reshape(-1, 2)


def main():
    hierarchy = joblib.load(OUT / "hierarchy.joblib")
    with np.load(DOTS / "pc_dots_testml_trainbasis.npz") as z:
        dots = {k: z[k].copy() for k in z.files}
    labels = pd.read_csv(DOTS / "desc_testml_5s.csv",
                         usecols=["id", "label"]).groupby("id").label.first()

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    manifest = {}
    for row, (mod, cols) in enumerate((("CSI", MOD_COLS["csi"]),
                                       ("Radar", MOD_COLS["rad"]))):
        h = hierarchy["csi" if mod == "CSI" else "rad"]
        pts = [[], [], []]
        for key, a in dots.items():
            lab = int(labels[key])
            for j, p in enumerate(level_points(a, cols, h)):
                pts[j].append(np.column_stack([p, np.full(len(p), lab)]))
        for j in range(3):
            ax = axes[row, j]
            data = np.concatenate(pts[j]) if pts[j] else np.empty((0, 3))
            rng = np.random.default_rng(104 + j)
            for lab, color in ((0, BLUE), (1, ORANGE)):
                sub = data[data[:, 2] == lab]
                if len(sub) > 2500:
                    sub = sub[rng.choice(len(sub), 2500, replace=False)]
                ax.scatter(sub[:, 0], sub[:, 1], s=5 if j < 2 else 13,
                           alpha=.42 if j < 2 else .75, color=color,
                           marker=("o", "^")[lab],
                           rasterized=True, label=("Empty", "Occupied")[lab])
            ax.set_xlabel("PC1")
            ax.set_ylabel(f"{mod}  PC2" if j == 0 else "PC2")
            if row == 0:
                ax.set_title(LEVELS[j])
            if row == 0 and j == 2:
                ax.legend(fontsize=8)
        manifest[mod] = {"points": {lvl: int(sum(len(p) for p in pts[j]))
                                    for j, lvl in enumerate(LEVELS)},
                         "explained5": h["explained5"],
                         "explained15": h["explained15"]}
    fig.tight_layout()
    fig.savefig(ASSETS / "hierarchical_pca.pdf")
    fig.savefig(ASSETS / "hierarchical_pca.png", dpi=300)
    plt.close(fig)
    (OUT / "hierarchical_pca.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
