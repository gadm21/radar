"""One-second PC coordinates projected through training-fitted bases.

Two panels (CSI, Radar) show PC1 vs PC2 for the multilink test captures,
colored by class. Mirrors the first level of the earlier hierarchical plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEEP = ROOT / "E1/outputs/pca_deep"
ASSETS = ROOT / "paper/icc_assets"
BLUE, ORANGE = "#2C6B9E", "#C65A2E"

labels = pd.read_csv(DEEP / "desc_testml_5s.csv",
                     usecols=["id", "label"]).groupby("id").label.first()
z = np.load(DEEP / "pc_dots_testml_trainbasis.npz")

fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6))
rng = np.random.default_rng(0)
for ax, (name, cols) in zip(axes, (("CSI", (1, 2)), ("Radar", (4, 5)))):
    for lab, color in ((0, BLUE), (1, ORANGE)):
        pts = []
        for k in z.files:
            if k not in labels.index or int(labels[k]) != lab:
                continue
            a = z[k]
            good = (a[:, 0] > .5) & np.isfinite(a[:, cols]).all(axis=1)
            pts.append(a[good][:, cols])
        a = np.concatenate(pts)
        if len(a) > 3000:
            a = a[rng.choice(len(a), 3000, replace=False)]
        ax.scatter(a[:, 0], a[:, 1], s=4, alpha=.35, color=color,
                   marker=("o", "^")[lab], rasterized=True,
                   label=("Empty", "Occupied")[lab])
    ax.set_xlabel(f"{name} PC1")
    ax.set_ylabel("PC2")
    ax.yaxis.grid(True, color="#E7ECEF", linewidth=.5)
    ax.set_axisbelow(True)
axes[0].legend(fontsize=7, loc="best", markerscale=2)
fig.tight_layout()
fig.savefig(ASSETS / "pca_scatter.pdf")
fig.savefig(ASSETS / "pca_scatter.png", dpi=300)
print("saved", ASSETS / "pca_scatter.pdf")
