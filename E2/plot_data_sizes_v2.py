"""Cohort sizes for the v2 comparison: real and synthetic minutes per class.
Counts come from the saved v2 cohort tables and the multilink feature file."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "E2/outputs/frozen_v2"
TEST = ROOT / "E1/outputs/features_multilink_5s.csv"
OUT = ROOT / "paper/icc_assets"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
EMPTY, OCC, INK = "#4C8DB5", "#C05B22", "#213542"


def counts(frame):
    g = frame.groupby(["label", "aug"]).unit.nunique()
    return {(int(l), int(a)): int(v) for (l, a), v in g.items()}


tr = counts(pd.read_csv(V2 / "train_windows.csv"))
va = counts(pd.read_csv(V2 / "validation_windows.csv"))
t = pd.read_csv(TEST)
t["minute"] = t.win // 12
te = {(int(l), 0): int(v) for l, v in t.groupby("label").apply(
    lambda g: g.groupby("cap").minute.nunique().sum()).items()}
rows = [("Training\n(wall 2 + wall 4)", tr), ("Validation\n(wall 3)", va),
        ("Test, 4 captures\n(wall 3, multilink)", te)]

fig, ax = plt.subplots(figsize=(3.45, 1.95))
y = np.arange(len(rows))[::-1]
for i, (name, c) in enumerate(rows):
    left = 0
    for label, color in ((0, EMPTY), (1, OCC)):
        real, syn = c.get((label, 0), 0), c.get((label, 1), 0)
        ax.barh(y[i], real, left=left, color=color, height=.6)
        ax.text(left + real / 2, y[i], str(real), ha="center", va="center",
                color="white", fontsize=6.6, weight="bold")
        left += real
        if syn:
            ax.barh(y[i], syn, left=left, color="white", edgecolor=color,
                    hatch="////", height=.6, linewidth=.8)
            ax.text(left + syn / 2, y[i], f"+{syn}", ha="center", va="center",
                    color=color, fontsize=6.6, weight="bold")
            left += syn
    ax.text(left + 8, y[i], f"{left} min", va="center", fontsize=6.6, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
ax.set_xlabel("Minutes", fontsize=7)
ax.set_xlim(0, 440)
ax.tick_params(axis="x", labelsize=6.5)
ax.legend(handles=[Patch(color=EMPTY, label="empty, original"),
                   Patch(color=OCC, label="occupied, original"),
                   Patch(facecolor="white", edgecolor=OCC, hatch="////",
                         label="occupied, synthetic")],
          loc="lower center", bbox_to_anchor=(.45, 1.0), ncol=3, frameon=False,
          fontsize=6.0, handlelength=1.4, columnspacing=.9)
ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=.3, right=.98, bottom=.2, top=.9)
fig.savefig(OUT / "data_sizes.pdf", bbox_inches="tight", pad_inches=.03)
fig.savefig(OUT / "data_sizes.png", dpi=300, bbox_inches="tight", pad_inches=.03)
print(rows)
