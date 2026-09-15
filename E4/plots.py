"""Figures for the E4 hierarchical pipeline -> outputs/figs/*.png."""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

FIGS = C.OUTPUT_DIR / "figs"
LABELS = ["empty", "sleep", "present(left)", "present(right)"]


def plot_confusion(rows):
    cm = np.zeros((4, 4), dtype=int)
    for r in rows:
        t = r["true_combined"]
        p = r["pred_combined"]
        if t not in LABELS or p not in LABELS:
            continue
        cm[LABELS.index(t), LABELS.index(p)] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("predict_capture combined-label confusion (test_minutes)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGS / "combined_confusion.png", dpi=150)
    plt.close(fig)


def plot_stage_accuracy(results, metrics):
    stages = ["occupancy\n(empty/occupied)", "sleep/present", "left/right\n(E3 5-fold CV)"]
    win = [results["occupancy"]["test"]["window"]["accuracy"],
           results["sleep_present"]["test"]["window"]["accuracy"],
           results["position"]["cv_summary"]["window_accuracy_mean"]]
    minute = [results["occupancy"]["test"]["minute"]["accuracy"],
              results["sleep_present"]["test"]["minute"]["accuracy"],
              results["position"]["cv_summary"]["minute_accuracy_mean"]]
    x = np.arange(3); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, win, w, label="window-level")
    ax.bar(x + w / 2, minute, w, label="minute-level")
    ax.axhline(0.9, color="red", ls="--", lw=1, label="0.9 target")
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Per-stage test accuracy (placement t)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "stage_accuracy.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    ev = json.loads((C.OUTPUT_DIR / "predict_capture_eval.json").read_text())
    results = json.loads((C.OUTPUT_DIR / "results_e4.json").read_text())
    plot_confusion(ev["rows"])
    plot_stage_accuracy(results, ev["metrics"])
    print(f"Figures written to {FIGS}")


if __name__ == "__main__":
    main()
