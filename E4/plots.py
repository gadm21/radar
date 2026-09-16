"""Figures for the E4 hierarchical pipeline -> outputs/figs/*.png.

Figures produced:
  * stage_accuracy.png          — per-stage window/minute accuracy with CV
                                  error bars + end-to-end accuracy markers
  * combined_confusion.png      — row-normalized 4-way confusion (counts + %)
  * activity_confusion.png      — row-normalized 3-way activity confusion
  * sleep_present_cv.png        — per-fold CV accuracy for both training
                                  variants + the transfer diagnostic
  * probability_distributions.png — per-recording stage-1/2 score
                                  distributions vs decision thresholds
  * training_curves.png         — validation macro-F1 per epoch (stage-2
                                  folds + deployment + position)
"""
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
COMBINED_LABELS = ["empty", "sleep", "present(left)", "present(right)"]
ACT_LABELS = ["empty", "sleep", "present"]

C_WIN = "#4C9BD6"
C_MIN = "#E8833A"
C_TARGET = "#C0392B"


def _bar_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.015, fmt.format(h),
                ha="center", va="bottom", fontsize=8)


def _confusion_ax(ax, cm, labels, title):
    """Row-normalized confusion heatmap annotated with count + row %."""
    cm = np.asarray(cm, dtype=float)
    row_sum = cm.sum(1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            n = int(cm[i, j])
            if n == 0:
                continue
            frac = norm[i, j]
            ax.text(j, i, f"{n}\n({frac:.0%})", ha="center", va="center",
                    fontsize=8,
                    color="white" if frac > 0.55 else "black")
    return im


def plot_confusions(rows):
    """Combined 4-way + activity 3-way confusion matrices."""
    cm4 = np.zeros((4, 4), dtype=int)
    cm3 = np.zeros((3, 3), dtype=int)
    for r in rows:
        t, p = r["true_combined"], r["pred_combined"]
        if t in COMBINED_LABELS and p in COMBINED_LABELS:
            cm4[COMBINED_LABELS.index(t), COMBINED_LABELS.index(p)] += 1
        ta, pa = r["true_activity"], r["pred_activity"]
        if ta is not None and pa in ACT_LABELS:
            cm3[int(ta), ACT_LABELS.index(pa)] += 1

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = _confusion_ax(ax, cm4, COMBINED_LABELS,
                       "predict_capture combined-label confusion\n"
                       "(test_minutes, row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    fig.tight_layout()
    fig.savefig(FIGS / "combined_confusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = _confusion_ax(ax, cm3, ACT_LABELS,
                       "Activity confusion (empty/sleep/present)\n"
                       "(test_minutes, row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    fig.tight_layout()
    fig.savefig(FIGS / "activity_confusion.png", dpi=150)
    plt.close(fig)


def plot_stage_accuracy(results, metrics):
    sp = results["sleep_present"]
    sp_cv = sp["cv"][sp["winner"]]
    stages = ["occupancy\n(empty/occupied)",
              f"sleep/present\n({sp['winner']}, {sp_cv['n_folds']}-fold CV)",
              "left/right\n(E3 5-fold CV)"]
    win = [results["occupancy"]["test"]["window"]["accuracy"],
           sp_cv["window_accuracy_mean"],
           results["position"]["cv_summary"]["window_accuracy_mean"]]
    win_err = [0, sp_cv["window_accuracy_std"],
               results["position"]["cv_summary"]["window_accuracy_std"]]
    minute = [results["occupancy"]["test"]["minute"]["accuracy"],
              sp_cv["minute_accuracy_mean"],
              results["position"]["cv_summary"]["minute_accuracy_mean"]]
    min_err = [0, sp_cv["minute_accuracy_std"],
               results["position"]["cv_summary"]["minute_accuracy_std"]]

    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar(x - w / 2, win, w, yerr=win_err, capsize=4,
                color=C_WIN, label="window-level")
    b2 = ax.bar(x + w / 2, minute, w, yerr=min_err, capsize=4,
                color=C_MIN, label="minute-level")
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    ax.axhline(0.9, color=C_TARGET, ls="--", lw=1, label="0.9 target")
    # end-to-end markers
    for val, name, mk in ((metrics["combined_accuracy"], "end-to-end combined", "D"),
                          (metrics["activity_accuracy"], "end-to-end activity", "s")):
        if val is not None:
            ax.scatter([2.62], [val], marker=mk, zorder=5,
                       label=f"{name} = {val:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-stage accuracy on placement t (held out / CV)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "stage_accuracy.png", dpi=150)
    plt.close(fig)


def plot_sleep_present_cv(results):
    sp = results["sleep_present"]
    variants = list(sp["cv"].keys())
    folds = range(sp["cv"][variants[0]]["n_folds"])
    x = np.arange(len(folds))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, level, key in ((axes[0], "window", "window_accuracy_per_fold"),
                           (axes[1], "minute", "minute_accuracy_per_fold")):
        for i, name in enumerate(variants):
            vals = sp["cv"][name][key]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=name,
                          color=[C_WIN, "#7CB668"][i % 2])
            _bar_labels(ax, bars)
        tr = sp["transfer_diagnostic"]["test"][level]["accuracy"]
        ax.axhline(tr, color=C_TARGET, ls=":", lw=1.4,
                   label=f"transfer t1+t2→t = {tr:.2f}")
        ax.axhline(0.9, color="gray", ls="--", lw=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"fold {k}" for k in folds])
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{level}-level accuracy per fold")
        ax.set_xlabel("held-out t fold")
    axes[0].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Sleep/present stage — grouped 5-fold CV on occupied t "
                 "recordings", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "sleep_present_cv.png", dpi=150)
    plt.close(fig)


def plot_probability_distributions(rows):
    """Per-recording stage scores vs thresholds, split by true class."""
    occ_p, occ_y = [], []
    sp_p, sp_y = [], []
    pos_p, pos_y = [], []
    for r in rows:
        if r.get("occupancy_prob") is not None:
            occ_p.append(r["occupancy_prob"])
            occ_y.append(1 if r["true_activity"] > 0 else 0)
        if r.get("activity_probs") and r["true_activity"] in (1, 2):
            sp_p.append(r["activity_probs"]["present"])
            sp_y.append(1 if r["true_activity"] == 2 else 0)
        if r.get("position_probs") and r["true_position"] is not None:
            pos_p.append(r["position_probs"]["right"])
            pos_y.append(int(r["true_position"]))

    panels = [
        ("Stage 1 — P(occupied)", occ_p, occ_y, ["empty", "occupied"]),
        ("Stage 2 — P(present)", sp_p, sp_y, ["sleep", "present"]),
        ("Stage 3 — P(right)", pos_p, pos_y, ["left", "right"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    rng = np.random.RandomState(0)
    for ax, (title, p, y, names) in zip(axes, panels):
        p = np.asarray(p); y = np.asarray(y)
        for cls, color in ((0, C_WIN), (1, C_MIN)):
            m = y == cls
            if not m.any():
                continue
            jitter = rng.uniform(-0.06, 0.06, m.sum())
            ax.scatter(p[m], cls + jitter, s=18, alpha=0.65, color=color,
                       label=f"true {names[cls]} (n={m.sum()})")
        ax.axvline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("predicted probability")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper center")
    fig.suptitle("Per-recording stage scores (predict_capture on test_minutes)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "probability_distributions.png", dpi=150)
    plt.close(fig)


def plot_training_curves(results):
    sp = results["sleep_present"]
    winner = sp["winner"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    curves = []
    for f in sp["folds"][winner]:
        ep = [h["epoch"] for h in f["history"]]
        f1 = [h["val_macro_f1"] for h in f["history"]]
        curves.append((ep, f1))
        ax.plot(ep, f1, alpha=0.55, lw=1.2,
                label=f"sleep/present fold {f['fold']}")
    pos_hist = results["position"]["history"]
    ax.plot([h["epoch"] for h in pos_hist],
            [h["val_macro_f1"] for h in pos_hist],
            color="black", lw=2.0, ls="--", label="position (deployment)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Training curves — sleep/present ({winner} CV folds) "
                 "+ position")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    ev = json.loads((C.OUTPUT_DIR / "predict_capture_eval.json").read_text())
    results = json.loads((C.OUTPUT_DIR / "results_e4.json").read_text())
    plot_confusions(ev["rows"])
    plot_stage_accuracy(results, ev["metrics"])
    plot_sleep_present_cv(results)
    plot_probability_distributions(ev["rows"])
    plot_training_curves(results)
    print(f"Figures written to {FIGS}")


if __name__ == "__main__":
    main()
