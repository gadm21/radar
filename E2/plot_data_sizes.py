"""Cohort sizes for the frozen comparison: minutes per class, excluded
sources, and the synthetic minutes created by the legacy benchmark but not
used. Counts are read from the same feature files as frozen_evaluation.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "E1/outputs/features_old_5s_ext.csv"
TEST = ROOT / "E1/outputs/features_multilink_5s.csv"
AUDIT = ROOT / "E2/outputs/revision/reviewer_audit/audit.json"
OUT = ROOT / "paper/icc_assets"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
EMPTY, OCC, GREY, INK = "#4C8DB5", "#C05B22", "#B7C2C8", "#213542"

d = pd.read_csv(OLD)
d["split"] = d.rec_id.str.split("/").str[0]
t = pd.read_csv(TEST)
synthetic = json.loads(AUDIT.read_text())["augmentation"][
    "planned_and_observed_synthetic_minutes"]

def minutes(frame, label):
    return int(frame[frame.label == label].rec_id.nunique())

def windows(frame, label):
    return int((frame.label == label).sum())

tr, va, cal = (d[d.split == s] for s in ("train", "test_minutes",
                                           "calibration_minutes"))
t["minute"] = t.win // 12
test_min = t.groupby(["cap", "label"]).minute.nunique().reset_index()
rows = [
    ("Training\n(wall 2 + wall 4)", minutes(tr, 0), minutes(tr, 1),
     windows(tr, 0), windows(tr, 1), int((tr.csi_ok == 0).sum())),
    ("Validation\n(wall 3)", minutes(va, 0), minutes(va, 1),
     windows(va, 0), windows(va, 1), int((va.csi_ok == 0).sum())),
    ("Test, 4 captures\n(wall 3, multilink)",
     int(test_min[test_min.label == 0].minute.sum()),
     int(test_min[test_min.label == 1].minute.sum()),
     windows(t, 0), windows(t, 1), 0),
]
excluded = [
    ("Calibration session\n(occupied only, excluded)", 0, minutes(cal, 1)),
    ("Synthetic empty minutes\n(legacy benchmark, not used)", synthetic, 0),
]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 2.2),
                             gridspec_kw={"width_ratios": [1.45, 1]})

# (a) minutes per cohort, used sets in colour, unused sources in grey.
labels = [r[0] for r in rows] + [e[0] for e in excluded]
y = np.arange(len(labels))[::-1]
for i, r in enumerate(rows):
    ax.barh(y[i], r[1], color=EMPTY, height=.62)
    ax.barh(y[i], r[2], left=r[1], color=OCC, height=.62)
    ax.text(r[1] / 2, y[i], f"{r[1]} E", ha="center", va="center",
            color="white", fontsize=6.8, weight="bold")
    ax.text(r[1] + r[2] / 2, y[i], f"{r[2]} O", ha="center", va="center",
            color="white", fontsize=6.8, weight="bold")
    ax.text(r[1] + r[2] + 12, y[i], f"{r[1] + r[2]} min", va="center",
            fontsize=6.8, color=INK)
for j, e in enumerate(excluded):
    k = len(rows) + j
    n = e[1] + e[2]
    ax.barh(y[k], n, color=GREY, height=.62, hatch="///", edgecolor="white")
    ax.text(n + 12, y[k], f"{n} min", va="center", fontsize=6.8, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=6.8)
ax.set_xlabel("Minutes")
ax.set_xlim(0, 1000)
ax.axhline(1.5, color=GREY, linewidth=.8, linestyle=":")
ax.text(990, 1.62, "not in the frozen comparison", ha="right", va="bottom",
        fontsize=6.2, color="#687C87", style="italic")
ax.legend(handles=[Patch(color=EMPTY, label="empty"),
                   Patch(color=OCC, label="occupied"),
                   Patch(facecolor=GREY, hatch="///", edgecolor="white",
                         label="removed / unused")],
          loc="upper right", bbox_to_anchor=(1, 1.0), frameon=False,
          fontsize=6.4, ncol=1)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(a) Minutes by cohort and class", fontsize=8, loc="left")

# (b) five-second windows per cohort with the CSI-missing share.
x = np.arange(len(rows))
tot = [r[3] + r[4] for r in rows]
miss = [r[5] for r in rows]
bx.bar(x, [r[3] for r in rows], color=EMPTY, width=.6)
bx.bar(x, [r[4] for r in rows], bottom=[r[3] for r in rows], color=OCC,
       width=.6)
bx.bar(x, miss, color="none", edgecolor=INK, hatch="xxx", width=.6,
       linewidth=.6)
for i in range(len(rows)):
    bx.text(x[i], tot[i] + 60, f"{tot[i]:,}", ha="center", fontsize=6.8,
            color=INK)
    if miss[i]:
        bx.text(x[i], miss[i] / 2, f"{miss[i]:,}\nno CSI", ha="center",
                va="center", fontsize=6.0, color="white", weight="bold")
bx.set_xticks(x)
bx.set_xticklabels(["Train", "Validation", "Test"], fontsize=6.8)
bx.set_ylabel("5-s windows")
bx.set_ylim(0, 5200)
bx.legend(handles=[Patch(facecolor="white", edgecolor=INK, hatch="xxx",
                         label="CSI missing, radar only\n(kept; median-imputed)")],
          loc="upper left", frameon=False, fontsize=6.2)
bx.spines[["top", "right"]].set_visible(False)
bx.set_title("(b) Windows and CSI coverage", fontsize=8, loc="left")

fig.subplots_adjust(left=.2, right=.99, bottom=.2, top=.88, wspace=.35)
fig.savefig(OUT / "data_sizes.pdf", bbox_inches="tight", pad_inches=.03)
fig.savefig(OUT / "data_sizes.png", dpi=300, bbox_inches="tight",
            pad_inches=.03)
summary = {"rows": rows, "excluded": excluded}
(OUT / "data_sizes.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
