"""Revised icc_multimodal2 assets for the newtest (wall-1) protocol.

Reads newtest/paper_eval artifacts and writes PDFs + .tex rows into
paper/icc_multimodal2_overleaf/icc_assets/:
  data_sizes.pdf, stable_descriptor_box_scatter.pdf, hierarchical_pca.pdf,
  frozen_comparison.pdf, llm_cm.pdf, model_table_rows.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "E2" / "outputs" / "frozen_v3"
EVAL = ROOT / "newtest" / "paper_eval"
ASSETS = ROOT / "paper/icc_multimodal2_overleaf/icc_assets"
sys.path.insert(0, str(ROOT.parent / "thothcraft" / "whispy" / "packages"
                       / "whispy-model-occ"))
from whispy_model_occ import hier          # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
BLUE, ORANGE = "#247BA0", "#D27B35"
EMPTY, OCC, INK = "#4C8DB5", "#C05B22", "#213542"
VIEW_ORDER = ("CSI", "radar", "early fusion", "late probability fusion")
VIEW_SHORT = {"CSI": "CSI", "radar": "Radar", "early fusion": "Early fusion",
              "late probability fusion": "Late fusion"}
LEVELS = ("1 second", "5 seconds", "15 seconds")


def fmt(s, dec=3):
    m = f"{s['mean']:.{dec}f}".lstrip("0") or "0"
    if s["sd"] < 5e-4:
        return m
    return f"{m}$\\pm${s['sd']:.{dec}f}".replace("$\\pm$0.", "$\\pm$.")


# ------------------------------------------------------------------ fig 2
def data_sizes():
    c = json.loads((EVAL / "capture_summary.json").read_text())
    rows = [
        ("Training\n(wall 2 + wall 4)",
         {0: c["train"]["empty"], 1: c["train"]["occupied"],
          "syn": c["train"]["occupied_synth"]}),
        ("Validation\n(wall 3)",
         {0: c["validation"]["empty"], 1: c["validation"]["occupied"],
          "syn": c["validation"]["occupied_synth"]}),
        ("Test\n(wall 1)",
         {0: c["test"]["empty"], 1: c["test"]["occupied"], "syn": 0}),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 1.55))
    y = np.arange(len(rows))[::-1]
    for i, (name, cnt) in enumerate(rows):
        left = 0
        for label, color in ((0, EMPTY), (1, OCC)):
            real = cnt[label]
            ax.barh(y[i], real, left=left, color=color, height=.6)
            ax.text(left + real / 2, y[i], str(real), ha="center",
                    va="center", color="white", fontsize=6.6,
                    weight="bold")
            left += real
        if cnt["syn"]:
            ax.barh(y[i], cnt["syn"], left=left, color="white",
                    edgecolor=OCC, hatch="////", height=.6, linewidth=.8)
            if cnt["syn"] >= 60:
                ax.text(left + cnt["syn"] / 2, y[i], f"+{cnt['syn']}",
                        ha="center", va="center", color=OCC, fontsize=6.6,
                        weight="bold")
            left += cnt["syn"]
        total = f"{left} min" + (f" (+{cnt['syn']})"
                                 if 0 < cnt["syn"] < 60 else "")
        ax.text(left + 12, y[i], total, va="center", fontsize=6.6,
                color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xlabel("Minutes", fontsize=7)
    ax.set_xlim(0, 760)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend(handles=[Patch(color=EMPTY, label="empty, original"),
                       Patch(color=OCC, label="occupied, original"),
                       Patch(facecolor="white", edgecolor=OCC,
                             hatch="////", label="occupied, synthetic")],
              loc="lower center", bbox_to_anchor=(.45, 1.0), ncol=3,
              frameon=False, fontsize=6.0, handlelength=1.4,
              columnspacing=.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.3, right=.98, bottom=.2, top=.9)
    fig.savefig(ASSETS / "data_sizes.pdf", bbox_inches="tight",
                pad_inches=.03)
    fig.savefig(ASSETS / "data_sizes.png", dpi=300, bbox_inches="tight",
                pad_inches=.03)
    plt.close(fig)


# ------------------------------------------------------------------ fig 3
def stable_descriptors():
    train = pd.read_csv(FROZEN / "train_windows.csv")
    train = train[train.aug == 0]
    val = pd.read_csv(EVAL / "validation_windows_combined.csv")
    val = val[val.aug == 0]
    test = pd.read_csv(EVAL / "windows.csv")
    groups = {"Train": train, "Validation": val, "Test": test}
    cols = ["ra_std90", "re_std90"]
    aucs = {}
    fig, axes = plt.subplots(len(cols), 3,
                             figsize=(10.8, 2.0 * len(cols)),
                             squeeze=False)
    rng = np.random.default_rng(17)
    for r, col in enumerate(cols):
        for c_, (name, d) in enumerate(groups.items()):
            ax = axes[r, c_]
            vals = [d.loc[d.label.eq(k), col]
                    .replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                    for k in (0, 1)]
            pooled = np.concatenate(vals)
            lo, hi = np.quantile(pooled, [.01, .99])
            boxes = ax.boxplot(vals, positions=[0, 1], widths=.42,
                               showfliers=False, patch_artist=True,
                               medianprops={"color": "black"},
                               boxprops={"linewidth": 1.05})
            for patch, fill in zip(boxes["boxes"],
                                   ("#CBE6F2", "#F9D9C7")):
                patch.set_facecolor(fill)
                patch.set_edgecolor("#46545D")
            for k, a in enumerate(vals):
                take = rng.choice(len(a), min(130, len(a)), replace=False)
                ax.scatter(k + rng.normal(0, .095, len(take)), a[take],
                           s=6, alpha=.36, color=(EMPTY, OCC)[k],
                           rasterized=True)
            ax.set_ylim(lo - .08 * (hi - lo), hi + .08 * (hi - lo))
            ax.set_xticks([0, 1], ["Empty", "Occupied"])
            ax.yaxis.grid(True, color="#E7ECEF", linewidth=.55)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c_ == 0:
                ax.set_ylabel(col)
            lab = np.concatenate([np.zeros(len(vals[0])),
                                  np.ones(len(vals[1]))])
            auc = roc_auc_score(lab, pooled)
            aucs[f"{col}|{name}"] = round(float(auc), 3)
            ax.text(.98, .96, f"AUC {auc:.2f}", ha="right", va="top",
                    transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "stable_descriptor_box_scatter.pdf",
                bbox_inches="tight")
    fig.savefig(ASSETS / "stable_descriptor_box_scatter.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    return aucs


# ------------------------------------------------------------------ fig 4
def level_points(a, cols, h):
    good = (a[:, 0] > .5) & np.isfinite(a[:, cols]).all(axis=1)
    l1 = a[good][:, cols]
    l2, l3 = [], []
    for s in range(0, len(a) - 4, 5):
        b = hier._block5(a, cols, s)
        if b is not None:
            l2.append(h["p5"].transform(
                h["sc5"].transform(b.reshape(1, -1)))[0])
    for s in range(0, len(a) - 14, 15):
        v = hier.window_hier(a, cols, s, h)[-2:]
        if np.isfinite(v).all():
            l3.append(v)
    return l1, np.asarray(l2).reshape(-1, 2), np.asarray(l3).reshape(-1, 2)


def hierarchical_pca(bundle):
    labels = pd.read_csv(EVAL / "windows.csv", usecols=["unit", "label"]) \
        .drop_duplicates().set_index("unit").label.to_dict()
    with np.load(EVAL / "dots.npz") as z:
        dots = {k: z[k].copy() for k in z.files}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    manifest = {}
    for row, (mod, cols) in enumerate(
            (("CSI", hier.MOD_COLS["csi"]),
             ("Radar", hier.MOD_COLS["rad"]))):
        h = bundle["hierarchy"]["csi" if mod == "CSI" else "rad"]
        pts = [[], [], []]
        for key, a in dots.items():
            lab = labels[key]
            for j, p in enumerate(level_points(a, cols, h)):
                pts[j].append(np.column_stack(
                    [p, np.full(len(p), lab)]))
        for j in range(3):
            ax = axes[row, j]
            data = (np.concatenate(pts[j]) if pts[j]
                    else np.empty((0, 3)))
            rng = np.random.default_rng(104 + j)
            for lab, color in ((0, BLUE), (1, ORANGE)):
                sub = data[data[:, 2] == lab]
                if len(sub) > 2500:
                    sub = sub[rng.choice(len(sub), 2500, replace=False)]
                ax.scatter(sub[:, 0], sub[:, 1],
                           s=5 if j < 2 else 13,
                           alpha=.42 if j < 2 else .75, color=color,
                           marker=("o", "^")[lab], rasterized=True,
                           label=("Empty", "Occupied")[lab])
            ax.set_xlabel("PC1")
            ax.set_ylabel(f"{mod}  PC2" if j == 0 else "PC2")
            if row == 0:
                ax.set_title(LEVELS[j])
            if row == 0 and j == 2:
                ax.legend(fontsize=8)
        manifest[mod] = {"points": {lvl: int(
            sum(len(p) for p in pts[j])) for j, lvl in enumerate(LEVELS)}}
    fig.tight_layout()
    fig.savefig(ASSETS / "hierarchical_pca.pdf")
    fig.savefig(ASSETS / "hierarchical_pca.png", dpi=300)
    plt.close(fig)
    return manifest


# ------------------------------------------------------------------ fig 5
def frozen_comparison(data):
    keys = [data["selected_views"][v] for v in VIEW_ORDER]
    labels = [f"{VIEW_SHORT[v]}\n{k.split('|')[0]}"
              for v, k in zip(VIEW_ORDER, keys)]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.2), sharey=True)
    x = np.arange(4)
    for ax, unit in zip(axes, ("window", "minute")):
        vk = "window_accuracy" if unit == "window" else "accuracy"
        v = [data["results"][k]["validation"][vk] for k in keys]
        t = [data["results"][k]["test"][vk] for k in keys]
        ax.bar(x - .17, [s["mean"] for s in v], .34,
               yerr=[s["sd"] for s in v], color=BLUE, label="Validation",
               capsize=2, error_kw={"linewidth": .7})
        ax.bar(x + .17, [s["mean"] for s in t], .34,
               yerr=[s["sd"] for s in t], color=ORANGE, label="Test",
               capsize=2, error_kw={"linewidth": .7})
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylim(0, 1.07)
        ax.set_ylabel(f"{unit.capitalize()} accuracy", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    h, n = axes[0].get_legend_handles_labels()
    fig.legend(h, n, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(.55, 1.0), fontsize=7)
    fig.subplots_adjust(left=.16, right=.99, bottom=.1, top=.92,
                        hspace=.55)
    fig.savefig(ASSETS / "frozen_comparison.pdf", bbox_inches="tight")
    fig.savefig(ASSETS / "frozen_comparison.png", dpi=350,
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 7
def llm_cm():
    fig, axes = plt.subplots(1, 2, figsize=(2.5, 1.35),
                             layout="constrained")
    vmax = 0
    cms = {}
    for model, title in (("gpt-4o-mini", "GPT-4o mini"),
                         ("gpt-5.4", "GPT-5.4")):
        s = json.loads((EVAL / "llm" / f"{model}_summary.json").read_text())
        arr = np.array(s["mean_vote_confusion"])
        cms[model] = arr
        vmax = max(vmax, int(arr.max()))
    for ax, (model, title) in zip(axes, (("gpt-4o-mini", "GPT-4o mini"),
                                        ("gpt-5.4", "GPT-5.4"))):
        arr = cms[model]
        ax.imshow(arr, cmap="Blues", vmin=0, vmax=vmax)
        for (i, j), val in np.ndenumerate(arr):
            ax.text(j, i, str(val), ha="center", va="center", fontsize=8,
                    color="white" if val > vmax / 2 else "#1E3441")
        ax.set_xticks([0, 1], ["E", "O"], fontsize=6.5)
        ax.set_yticks([0, 1], ["E", "O"], fontsize=6.5)
        ax.set_xlabel(title, fontsize=7)
    axes[0].set_ylabel("Actual", fontsize=7)
    fig.savefig(ASSETS / "llm_cm.pdf", bbox_inches="tight")
    fig.savefig(ASSETS / "llm_cm.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- table
def table_rows(data):
    lines = []
    for v in VIEW_ORDER:
        for i, m in enumerate(("RF", "GBM", "SVM", "MLP")):
            r = data["results"][f"{m}|{v}"]
            lead = f"{m}" + ("$^\\ast$" if data["selected_views"][v]
                             == f"{m}|{v}" else "")
            first = ("\\multirow{4}{*}{" + VIEW_SHORT[v] + "}") \
                if i == 0 else ""
            lines.append(" & ".join(
                [first, lead, fmt(r["validation"]["roc_auc"], 2),
                 fmt(r["validation"]["accuracy"], 2),
                 fmt(r["test"]["accuracy"], 2),
                 fmt(r["test"]["roc_auc"], 2)]) + " \\\\")
        if v != VIEW_ORDER[-1]:
            lines.append("\\midrule")
    lines.append(r"\bottomrule")
    (ASSETS / "model_table_rows.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = json.loads((EVAL / "evaluation.json").read_text())
    import whispy_model_occ.hier as _h
    bundle = _h.OccHierModel()._load()
    data_sizes()
    aucs = stable_descriptors()
    manifest = hierarchical_pca(bundle)
    frozen_comparison(data)
    table_rows(data)
    try:
        llm_cm()
    except FileNotFoundError:
        print("llm summaries not present yet; skipping llm_cm")
    print(json.dumps({"descriptor_aucs": aucs, "pca_points": manifest},
                     indent=2))
    for v in VIEW_ORDER:
        k = data["selected_views"][v]
        r = data["results"][k]
        print(k, "val acc", fmt(r["validation"]["accuracy"]),
              "test acc", fmt(r["test"]["accuracy"]),
              "test auc", fmt(r["test"]["roc_auc"]),
              "thr", fmt(r["validation_threshold"]))
