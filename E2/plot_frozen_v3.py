"""Comparison and confusion figures for the v3 frozen evaluation, plus the
LaTeX rows for the model table. No panel or figure titles."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs/frozen_v3"
ASSETS = OUT.parent.parent.parent / "paper/icc_assets_v3"
BLUE, ORANGE = "#247BA0", "#D27B35"
VIEW_ORDER = ("CSI", "radar", "early fusion", "late probability fusion")
VIEW_SHORT = {"CSI": "CSI", "radar": "Radar", "early fusion": "Early fusion",
              "late probability fusion": "Late fusion"}
MODELS = ("RF", "GBM", "SVM", "MLP")


def fmt(s, dec=3):
    m = f"{s['mean']:.{dec}f}".lstrip("0") or "0"
    if s["sd"] < 5e-4:
        return m
    return f"{m}$\\pm${s['sd']:.{dec}f}".replace("$\\pm$0.", "$\\pm$.")


def comparison(data):
    keys = [data["selected_views"][v] for v in VIEW_ORDER]
    labels = [f"{VIEW_SHORT[v]}\n{k.split('|')[0]}" for v, k in zip(VIEW_ORDER, keys)]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.0), sharey=True)
    x = np.arange(4)
    for ax, unit in zip(axes, ("window", "minute")):
        vk = "window_accuracy" if unit == "window" else "accuracy"
        v = [data["results"][k]["validation"][vk] for k in keys]
        t = [data["results"][k]["test"][vk] for k in keys]
        ax.bar(x - .17, [s["mean"] for s in v], .34, yerr=[s["sd"] for s in v],
               color=BLUE, label="Validation", capsize=2, error_kw={"linewidth": .7})
        ax.bar(x + .17, [s["mean"] for s in t], .34, yerr=[s["sd"] for s in t],
               color=ORANGE, label="Multilink test", capsize=2, error_kw={"linewidth": .7})
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylim(0, 1.07)
        ax.set_ylabel(f"{unit.capitalize()} accuracy", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    h, n = axes[0].get_legend_handles_labels()
    fig.legend(h, n, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(.55, 1.0), fontsize=7)
    fig.subplots_adjust(left=.16, right=.99, bottom=.1, top=.92, hspace=.55)
    fig.savefig(ASSETS / "frozen_comparison.pdf", bbox_inches="tight")
    fig.savefig(ASSETS / "frozen_comparison.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def confusion(data):
    keys = [data["selected_views"][v] for v in VIEW_ORDER]
    fig, axes = plt.subplots(1, 4, figsize=(3.48, 1.25), layout="constrained")
    for ax, v, key in zip(axes, VIEW_ORDER, keys):
        arr = np.array(data["results"][key]["test"]["confusion_matrix_seed0"])
        ax.imshow(arr, cmap="Blues", vmin=0, vmax=180)
        for (i, j), val in np.ndenumerate(arr):
            ax.text(j, i, str(val), ha="center", va="center", fontsize=8,
                    color="white" if val > 90 else "#1E3441")
        ax.set_xticks([0, 1], ["E", "O"], fontsize=6.5)
        ax.set_yticks([0, 1], ["E", "O"], fontsize=6.5)
        ax.set_xlabel(f"{VIEW_SHORT[v].replace(' fusion', '')}\n{key.split('|')[0]}", fontsize=6.5)
    axes[0].set_ylabel("Actual", fontsize=7)
    fig.savefig(ASSETS / "primary_cm.pdf", bbox_inches="tight")
    fig.savefig(ASSETS / "primary_cm.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def table_rows(data):
    lines = []
    for v in VIEW_ORDER:
        for i, m in enumerate(MODELS):
            r = data["results"][f"{m}|{v}"]
            lead = f"{m}" + ("$^\\ast$" if data["selected_views"][v] == f"{m}|{v}" else "")
            first = ("\\multirow{4}{*}{" + VIEW_SHORT[v] + "}") if i == 0 else ""
            lines.append(" & ".join([first, lead, fmt(r["validation"]["roc_auc"], 2),
                                     fmt(r["validation"]["accuracy"], 2),
                                     fmt(r["test"]["accuracy"], 2),
                                     fmt(r["test"]["roc_auc"], 2)]) + " \\\\")
        if v != VIEW_ORDER[-1]:
            lines.append("\\midrule")
    lines.append(r"\bottomrule")
    (ASSETS / "model_table_rows.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    data = json.loads((OUT / "evaluation.json").read_text())
    comparison(data)
    confusion(data)
    table_rows(data)
    for v in VIEW_ORDER:
        k = data["selected_views"][v]
        r = data["results"][k]
        print(k, "val acc", fmt(r["validation"]["accuracy"]), "test acc", fmt(r["test"]["accuracy"]),
              "test auc", fmt(r["test"]["roc_auc"]), "thr", fmt(r["validation_threshold"]))
