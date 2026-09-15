"""Figures for the E3 left/right pipeline -> outputs/figs/*.png."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

FIGS = C.OUTPUT_DIR / "figs"


def plot_kfold_accuracy(results):
    folds = results["folds"]
    summary = results["summary"]
    x = np.arange(len(folds))
    w = [f["test"]["window"]["accuracy"] for f in folds]
    m = [f["test"]["minute"]["accuracy"] for f in folds]

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    ax.bar(x - width / 2, w, width, label="window-level")
    ax.bar(x + width / 2, m, width, label="minute-level")
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1, label="0.9 target")
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {f['fold']}" for f in folds])
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"Left/right test accuracy per fold "
        f"(mean window={summary['window_accuracy_mean']:.3f}, "
        f"mean minute={summary['minute_accuracy_mean']:.3f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "kfold_accuracy.png", dpi=150)
    plt.close(fig)


def plot_confusion(results):
    cm_total = np.zeros((2, 2), dtype=int)
    for f in results["folds"]:
        cm_total += np.asarray(f["test"]["window"]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_total, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(C.CLASS_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(C.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Aggregated window-level confusion (all folds)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_total[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGS / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_hp_search(hp):
    results = hp["results"]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"lr={r['cfg']['lr']}\nemb={r['cfg']['embed_dim']}\ndrop={r['cfg']['dropout']}" for r in results]
    accs = [r["val_window"]["accuracy"] for r in results]
    ax.bar(range(len(results)), accs)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Hyperparameter search (fold-0 held-out validation)")
    ax.set_ylim(0.5, 1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "hp_search.png", dpi=150)
    plt.close(fig)


def plot_example_map(index_row_path):
    d = np.load(index_row_path)
    radar = d["radar"].astype(np.float32)  # (nW, T, 2, S, S)
    meta = json.loads(str(d["meta"]))
    mean_rd = radar[0, :, 0].mean(0)
    mean_ra = radar[0, :, 1].mean(0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(mean_rd, aspect="auto", cmap="viridis")
    axes[0].set_title("Range-Doppler (time-mean)")
    axes[0].set_xlabel("range bin"); axes[0].set_ylabel("doppler bin")
    axes[1].imshow(mean_ra, aspect="auto", cmap="viridis")
    axes[1].set_title("Range-Azimuth (time-mean)")
    axes[1].set_xlabel("range bin"); axes[1].set_ylabel("azimuth bin")
    fig.suptitle(f"Example window - {meta['rec_id']} (label={C.CLASS_NAMES[meta['label']]})")
    fig.tight_layout()
    fig.savefig(FIGS / "example_maps.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    results = json.loads((C.OUTPUT_DIR / "results_kfold.json").read_text())
    plot_kfold_accuracy(results)
    plot_confusion(results)

    hp_path = C.OUTPUT_DIR / "hp_search.json"
    if hp_path.exists():
        plot_hp_search(json.loads(hp_path.read_text()))

    caches = sorted(C.CACHE_DIR.glob("*.npz"))
    if caches:
        plot_example_map(caches[0])

    print(f"Figures written to {FIGS}")


if __name__ == "__main__":
    main()
