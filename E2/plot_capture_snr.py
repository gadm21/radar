"""Per-capture radar SNR box plots for labeling the stray newtest capture.

Fifteen-second window SNR (mean and max) for every wall-1 newtest capture,
including the unlabeled capture3_768c3d688716, beside pooled training and
combined-validation references per class. Output:
newtest/paper_eval/capture_snr_boxplot.{pdf,png}
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from eval_newtest_paper import FROZEN, NEWTEST, OUT, window_rows

STRAY = ("capture3_768c3d688716", "768c3d688716")
EMPTY, OCC, UNK = "#3d85b8", "#c65d22", "#8a8a8a"


def stray_windows() -> pd.DataFrame:
    cache = OUT / "capture3_windows.csv"
    if not cache.exists():
        import whispy_model_occ.hier as h
        df, _ = window_rows(NEWTEST / STRAY[0] / STRAY[1], STRAY[0], -1,
                            h.OccHierModel()._load())
        df.to_csv(cache, index=False)
    return pd.read_csv(cache)


def start_time(name: str) -> dt.datetime:
    p = NEWTEST / name / "manifest.json"
    if not p.exists():
        p = next((NEWTEST / name).glob("*/manifest.json"))
    return dt.datetime.fromtimestamp(json.loads(p.read_text())["started_at"])


def main():
    test = pd.read_csv(OUT / "windows.csv")
    if STRAY[0] not in set(test.unit):
        test = pd.concat([test, stray_windows()])
    train = pd.read_csv(FROZEN / "train_windows.csv").query("aug == 0")
    val = pd.read_csv(OUT / "validation_windows_combined.csv").query("aug == 0")

    groups = []
    for split, df, wall in (("train", train, "walls 2+4"),
                            ("validation", val, "wall 3")):
        for lab, cls in ((0, "empty"), (1, "occupied")):
            groups.append((f"{split} {cls}\n({wall}, pooled)",
                           df[df.label == lab], lab))
    for name in sorted(test.unit.unique(), key=start_time):
        d = test[test.unit == name]
        lab = int(d.label.iloc[0])
        tag = start_time(name).strftime("%b %d %H:%M")
        groups.append((f"{name}\n(wall 1, {tag})", d, lab))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    for ax, col, title in ((axes[0], "snr_mean", "Radar SNR mean (dB)"),
                           (axes[1], "snr_max", "Radar SNR max (dB)")):
        data = [g[1][col].dropna().values for g in groups]
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=.6,
                        showfliers=True, flierprops={"markersize": 2})
        for patch, (_, _, lab) in zip(bp["boxes"], groups):
            patch.set_facecolor({0: EMPTY, 1: OCC}.get(lab, UNK))
            patch.set_alpha(.85)
        for med in bp["medians"]:
            med.set_color("black")
        ax.axhline(4.5, color="k", lw=.6, ls="--")
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=.3)
    axes[0].set_yticks(range(1, len(groups) + 1))
    axes[0].set_yticklabels([g[0] for g in groups], fontsize=7.5)
    axes[0].invert_yaxis()
    for i, (label, d, _) in enumerate(groups, start=1):
        axes[1].text(1.01, i, f"n={len(d)}", transform=axes[1]
                     .get_yaxis_transform(), va="center", fontsize=7)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=EMPTY, label="empty"),
                        Patch(color=OCC, label="occupied"),
                        Patch(color=UNK, label="unlabeled (capture3)")],
               loc="upper center", bbox_to_anchor=(.5, .955), ncol=3,
               frameon=False, fontsize=9)
    fig.suptitle("Per-window (15 s) radar SNR by capture; references above "
                 "the dashed line, wall-1 test captures below in time order",
                 y=.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, .97, .91))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"capture_snr_boxplot.{ext}", dpi=200)

    rows = [(g[0].split("\n")[0], len(g[1]), g[1].snr_mean.median(),
             g[1].snr_max.median(), g[1].snr_max.quantile(.75),
             (g[1].snr_max > 12).mean()) for g in groups]
    print(pd.DataFrame(rows, columns=["group", "n_win", "snr_mean_med",
                                      "snr_max_med", "snr_max_q75",
                                      "frac_max_gt12"]).round(3).to_string())


if __name__ == "__main__":
    main()
