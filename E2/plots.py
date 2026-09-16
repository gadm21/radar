"""Generate all figures for the E2 occupancy pipeline into outputs/figs/."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import train as T

FIGS = C.OUTPUT_DIR / "figs"
CLASS_NAMES = ["empty", "occupied"]


def _load(name):
    p = C.OUTPUT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_window_durations(index):
    """Histogram of 50-frame window durations + CSI counts per split."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for p, color in (("train_minutes", "tab:blue"),
                     ("val_minutes", "tab:orange"),
                     ("test_minutes", "tab:green")):
        durs, cnts = [], []
        for f in index[index.split == p]["path"]:
            d = np.load(f)
            durs.extend(d["win_dur"].tolist())
            cnts.extend(d["csi_count"].tolist())
        axes[0].hist(durs, bins=60, alpha=0.5, density=True, color=color,
                     label=f"{p} (med={np.median(durs):.2f}s)")
        axes[1].hist(cnts, bins=60, alpha=0.5, density=True, color=color,
                     label=f"{p} (med={np.median(cnts):.0f})")
    axes[0].set_xlabel("50-frame window duration (s)")
    axes[0].set_ylabel("density"); axes[0].legend(); axes[0].set_title("Radar window durations")
    axes[1].set_xlabel("CSI samples inside window")
    axes[1].set_ylabel("density"); axes[1].legend(); axes[1].set_title("CSI samples per window")
    fig.tight_layout()
    fig.savefig(FIGS / "window_timing.png", dpi=160)
    plt.close(fig)


def fig_class_balance(index):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(3); w = 0.35
    for i, lab in enumerate((0, 1)):
        vals = [int((index[index.split == p]["label"] == lab).sum())
                for p in ("train_minutes", "val_minutes", "test_minutes")]
        ax.bar(x + (i - 0.5) * w, vals, w, label=CLASS_NAMES[lab])
    ax.set_xticks(x, ["train", "val", "test"])
    ax.set_ylabel("minutes"); ax.legend(); ax.set_title("Class balance per split")
    fig.tight_layout(); fig.savefig(FIGS / "class_balance.png", dpi=160)
    plt.close(fig)


def _cm_ax(ax, cm, title):
    cm = np.asarray(cm)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASS_NAMES); ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    return im


def fig_e2(results):
    mods = [m for m in ("fusion", "radar", "csi") if m in results]
    # confusion matrices: window + minute per modality
    fig, axes = plt.subplots(2, len(mods), figsize=(4 * len(mods), 8))
    if len(mods) == 1:
        axes = axes.reshape(2, 1)
    for j, m in enumerate(mods):
        _cm_ax(axes[0, j], results[m]["test"]["window"]["confusion_matrix"],
               f"{m} — window level")
        _cm_ax(axes[1, j], results[m]["test"]["minute"]["confusion_matrix"],
               f"{m} — minute level")
    fig.suptitle("E2 test confusion matrices (test_minutes)")
    fig.tight_layout(); fig.savefig(FIGS / "e2_confusion.png", dpi=160)
    plt.close(fig)

    # metrics bar chart
    metrics = ["accuracy", "macro_f1", "empty_recall", "occupied_recall",
               "false_empty_rate"]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(metrics)); w = 0.8 / len(mods)
    for j, m in enumerate(mods):
        vals = [results[m]["test"]["window"][k] for k in metrics]
        ax.bar(x + (j - (len(mods) - 1) / 2) * w, vals, w, label=m)
    ax.set_xticks(x, metrics); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("E2 test metrics by modality (window level)")
    fig.tight_layout(); fig.savefig(FIGS / "e2_metrics.png", dpi=160)
    plt.close(fig)

    # training curve (best-val modality, fallback fusion)
    curve_mod = max((m for m in mods if m != "csi"),
                    key=lambda m: results[m]["val"]["window"]["accuracy"],
                    default=None)
    if curve_mod:
        hist = results[curve_mod]["history"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([h["epoch"] for h in hist], [h["val_macro_f1"] for h in hist],
                marker="o", label="val macro-F1")
        ax.plot([h["epoch"] for h in hist], [h["val_acc"] for h in hist],
                marker="s", label="val accuracy")
        ax.set_xlabel("epoch"); ax.set_ylim(0, 1.05); ax.legend()
        ax.set_title(f"E2 {curve_mod} training curve (val = val_minutes)")
        fig.tight_layout(); fig.savefig(FIGS / "e2_training.png", dpi=160)
        plt.close(fig)


def fig_e3(results):
    strats = ["zero_shot"] + [s for s in ("head", "fusion_head", "full")
                              if s in results]
    metrics = ["accuracy", "macro_f1", "false_empty_rate"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(strats)); w = 0.25
    for j, lev in enumerate(("window", "minute")):
        ax = axes[j]
        for i, k in enumerate(metrics):
            vals = [results[s][lev][k] for s in strats]
            ax.bar(x + (i - 1) * w, vals, w, label=k)
        ax.set_xticks(x, strats, rotation=15)
        ax.set_ylim(0, 1.05); ax.set_title(f"E3 few-shot ({lev} level)")
        ax.legend()
    fig.suptitle(f"E3: {results['n_support']} support minutes from "
                 f"val_minutes ({', '.join(results['support_minutes'][:4])}...)")
    fig.tight_layout(); fig.savefig(FIGS / "e3_fewshot.png", dpi=160)
    plt.close(fig)


def fig_hp_search(search):
    res = search["results"]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"lr={r['cfg']['lr']} e={r['cfg']['embed_dim']} d={r['cfg']['dropout']}"
              for r in res]
    vals = [r["val_window"]["accuracy"] for r in res]
    order = np.argsort(vals)
    ax.barh(np.arange(len(res)), [vals[i] for i in order])
    ax.set_yticks(np.arange(len(res)), [labels[i] for i in order], fontsize=8)
    ax.set_xlabel("val accuracy (val_minutes)"); ax.set_xlim(0, 1.05)
    ax.set_title("Hyperparameter search (radar, train→val)")
    fig.tight_layout(); fig.savefig(FIGS / "hp_search.png", dpi=160)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    index = T.load_index()
    fig_window_durations(index)
    fig_class_balance(index)
    e2 = _load("results_e2.json")
    if e2:
        fig_e2(e2)
    e3 = _load("results_e3.json")
    if e3:
        fig_e3(e3)
    hp = _load("hp_search.json")
    if hp:
        fig_hp_search(hp)
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
