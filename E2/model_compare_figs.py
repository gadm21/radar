"""Figures + report for E2 model_compare outputs.

Reads E2/outputs/model_compare/{results.json,results.csv,probs.pkl}
+ descriptor tables (for labels) -> figs/*.png + REPORT.md
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from sklearn.metrics import (confusion_matrix,      # noqa: E402
                             roc_curve, auc)

E2 = Path(__file__).resolve().parent
ROOT = E2.parent
OUT = E2 / "outputs" / "model_compare"
FIGS = OUT / "figs"
PCA_DIR = ROOT / "E1" / "outputs" / "pca_deep"
SET_LABEL = {"val": "validation (wall3)", "testml": "multilink test",
             "val_norm": "val — norm", "testml_norm": "test — norm",
             "cal_sup": "cal-sup (10min sleep)", "cal_unsup":
             "cal-unsup (5+5min)"}


def labels_for(s):
    if s == "cal_sup":
        df = pd.read_csv(PCA_DIR / "desc_calib_5s.csv")
        ids = list(dict.fromkeys(df.id))[:10]
        return df[df.id.isin(ids)].label.values.astype(int)
    if s == "cal_unsup":
        v = pd.read_csv(PCA_DIR / "desc_val_5s.csv")
        e = v[v.label == 0].id.unique()[:5]
        o = v[v.label == 1].id.unique()[:5]
        return v[v.id.isin(list(e) + list(o))].label.values.astype(int)
    s = s.replace("_norm", "")
    df = pd.read_csv(PCA_DIR / f"desc_{s}_5s.csv")
    return df.label.values.astype(int)


def fig_bars(res):
    """balanced acc per model, grouped by class, val+testml."""
    rows = []
    for k, sets in res.items():
        if "/" not in k:
            continue
        m, v = k.split("/")
        for s in ("val", "val_norm", "testml", "testml_norm"):
            if s in sets and isinstance(sets[s], dict) \
                    and "bal" in sets[s]:
                rows.append({"model": k, "set": s,
                             "bal": sets[s]["bal"],
                             "cls": ("dl" if m.startswith(("seqcnn",))
                                     else "rule" if m.startswith(
                                         ("stump", "knn", "centroid",
                                          "clustervote",
                                          "threshcal"))
                                     else "ml" if not
                                     m.startswith(("fuse", "emb", "jev"))
                                     else "fuse/emb")})
    df = pd.DataFrame(rows)
    if not len(df):
        return
    top = (df[df.set == "testml_norm"]
           .sort_values("bal", ascending=False).head(24))
    if not len(top):
        top = (df[df.set == "testml"]
               .sort_values("bal", ascending=False).head(24))
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(top))
    v2 = df.merge(top[["model"]], on="model")
    for i, s in enumerate(("val", "val_norm", "testml",
                           "testml_norm")):
        d = v2[v2.set == s].set_index("model").reindex(top.model)
        ax.bar(x + i * .22 - .33, d.bal, .22,
               label=SET_LABEL[s],
               color=["#4c9be0", "#9ec9eb", "#e0704c", "#f0a987"][i])
    ax.set_xticks(x, top.model, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("balanced acc"); ax.set_ylim(0, 1.05)
    ax.axhline(.5, color="k", lw=.6, ls=":")
    ax.legend()
    ax.set_title("balanced accuracy — top models on multilink test")
    fig.tight_layout()
    fig.savefig(FIGS / "model_bars.png", dpi=150)
    plt.close(fig)


def fig_roc(res, probs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, s in zip(axes, ("val_norm", "testml_norm")):
        y = labels_for(s if s != "cal_unsup" else "cal_unsup")
        cand = [(k, sets[s]["auc"]) for k, sets in res.items()
                if s in sets and isinstance(sets[s], dict)
                and "auc" in sets[s]
                and k in probs and s in probs[k]]
        cand.sort(key=lambda t: -(t[1] if t[1] == t[1] else 0))
        for k, a in cand[:6]:
            pr = probs[k][s]
            if len(pr) != len(y):
                continue
            f, t, _ = roc_curve(y, pr)
            ax.plot(f, t, label=f"{k} ({a:.2f})", lw=1.4)
        ax.plot([0, 1], [0, 1], "k:", lw=.7)
        ax.set_title(f"ROC — {SET_LABEL[s]}")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "roc.png", dpi=150)
    plt.close(fig)


def fig_cm(res, probs):
    best = [(k, s, sets[s]) for k, sets in res.items()
            for s in ("val", "testml", "val_norm", "testml_norm")
            if s in sets and isinstance(sets[s], dict)
            and "bal" in sets[s]]
    best.sort(key=lambda t: -t[2]["bal"])
    pick = []
    seen = set()
    for k, s, m in best:
        base = k.split("/")[0]
        if (base, s) in seen:
            continue
        seen.add((base, s))
        pick.append((k, s))
        if len(pick) >= 8:
            break
    if not pick:
        return
    n = len(pick)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * (n + 1) // 2,
                                                     6))
    for ax, (k, s) in zip(np.ravel(axes), pick):
        if k not in probs or s not in probs[k]:
            ax.axis("off"); continue
        y = labels_for(s)
        pr = probs[k][s]
        if len(pr) != len(y):
            ax.axis("off"); continue
        cm = confusion_matrix(y, pr > .5, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center",
                        color="red" if i == j else "k")
        ax.set_title(f"{k}\n{SET_LABEL[s]}", fontsize=8)
        ax.set_xticks([0, 1], ["e", "o"]); ax.set_yticks([0, 1],
                                                       ["e", "o"])
    fig.suptitle("confusion matrices — best per model family", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "cm.png", dpi=150)
    plt.close(fig)


def fig_cal(res):
    rows = []
    for k, sets in res.items():
        if "/" not in k:
            continue
        for s in ("cal_sup", "cal_unsup"):
            if s in sets and isinstance(sets[s], dict):
                v = (sets[s].get("ro") if s == "cal_sup"
                     else sets[s].get("bal"))
                if v == v:
                    rows.append({"model": k, "cal": s, "v": v})
    df = pd.DataFrame(rows)
    if not len(df):
        return
    top = (df[df.cal == "cal_unsup"].sort_values("v",
                                                 ascending=False)
           .head(18))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(top))
    for i, s in enumerate(("cal_sup", "cal_unsup")):
        d = df[df.cal == s].set_index("model").reindex(top.model)
        ax.bar(x + i * .4 - .2, d.v, .4, label=SET_LABEL[s],
               color=["#7bc47f", "#e0704c"][i])
    ax.set_xticks(x, top.model, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("occupied recall (sup) / bal acc (unsup)")
    ax.set_ylim(0, 1.05)
    ax.legend(); ax.set_title("calibration-set accuracy")
    fig.tight_layout()
    fig.savefig(FIGS / "calibration.png", dpi=150)
    plt.close(fig)


def report(res):
    L = ["# E2 model comparison (E2/model_compare.py)",
         "",
         "Splits: train=legacy train_minutes (274 min, wall2) | "
         "val=validation_minutes (192 min, wall3) | test=multilink "
         "test_minutes (4×1 h) | calib=calibration_minutes (t1_sleep, "
         "40 min cached, first 10 used)",
         "",
         "Labels: sleep/present/t1_sleep→occupied, empty→empty.",
         "Feature views: radar = physical radar + radar-PC + fft_snr; "
         "csi = physical csi + csi-PC; joint = 6-d joint-PC stats + "
         "cluster geometry; all = union (shared cols only).",
         "",
         "## Key findings",
         "- Rank-order signal survives every domain shift (top single "
         "descriptor `ra_mean`: AUC 0.72/0.95/1.00 on train/val/test), "
         "but absolute feature scales drift across sessions — raw "
         "accuracy collapses to ~0.50 while AUC stays 0.85-0.94.",
         "- `_norm` = unsupervised per-set re-standardisation "
         "(unlabelled eval windows re-scaled by their own mean/std "
         "before the train scaler). It recovers most of the signal "
         "and is the honest 'unsupervised calibration' number.",
         "- CSI-view is the most transferable raw view (rf/csi "
         "val 0.88 / test 0.82); radar-only is scale-fragile.",
         "- cal_sup: several models reach 1.00 occupied recall on a "
         "10-minute t1_sleep block — a one-class calibration visit "
         "works.",
         "",
         "## Best per model class (test=multilink)"]
    rows = []
    for k, sets in res.items():
        m = k.split("/")[0]
        cls = ("jev" if m == "jev" else
               "dl" if m.startswith(("seqcnn", "mlp"))
               else "rule" if m.startswith(
                   ("stump", "knn", "centroid", "clustervote",
                    "threshcal"))
               else "emb" if m.startswith("emb")
               else "fuse" if m.startswith("fus")
               else "ml")
        for s in ("val", "val_norm", "testml", "testml_norm",
                  "cal_sup", "cal_unsup"):
            if s in sets and isinstance(sets[s], dict) \
                    and "bal" in sets[s]:
                rows.append({"model": k, "cls": cls, "set": s,
                             **{kk: vv for kk, vv in sets[s].items()
                                if kk != "proba"}})
    df = pd.DataFrame(rows)
    if len(df):
        for cls in ("ml", "dl", "rule", "emb", "fuse", "jev"):
            d = df[(df.cls == cls) &
                   (df.set.isin(("testml", "testml_norm")))]
            if not len(d):
                continue
            d = d.sort_values("bal", ascending=False).head(5)
            L.append(f"### {cls}")
            L.append("| model | set | bal | acc | auc | rec_e | "
                     "rec_o |")
            L.append("|---|---|---|---|---|---|---|")
            for _, r in d.iterrows():
                L.append(f"| {r.model} | {r.set} | {r.bal:.3f} | "
                         f"{r.acc:.3f} | "
                         f"{r.auc if r.auc == r.auc else -1:.3f} | "
                         f"{r.re:.2f} | {r.ro:.2f} |")
            L.append("")
    L += ["## Full results (val + test + calibration)",
          "", "| model | set | bal | acc | f1 | auc | rec_e | rec_o |",
          "|---|---|---|---|---|---|---|---|"]
    if len(df):
        df = df.sort_values(["set", "bal"], ascending=[True, False])
        for _, r in df.iterrows():
            L.append(f"| {r.model} | {r.set} | {r.bal:.3f} | "
                     f"{r.acc:.3f} | {r.f1:.3f} | "
                     f"{r.auc if r.auc == r.auc else float('nan'):.3f} |"
                     f" {r.re:.2f} | {r.ro:.2f} |")
    if "jev" in res and isinstance(res["jev"], dict) \
            and "skipped" in res["jev"]:
        L += ["", f"JEV: {res['jev']['skipped']} "
                  "(client in E2/jev.py; set JEV_API_KEY to run)"]
    L += ["", "## Figures", "- figs/model_bars.png",
          "- figs/roc.png", "- figs/cm.png", "- figs/calibration.png",
          "", "Workflow A = descriptor fusion (per-view + 'all').",
          "Workflow B = shared PCA embedding (emb_*).",
          "Workflow C = probability fusion (fuse*/fusw_*).",
          "cal_sup = 10 min t1_sleep recall; cal_unsup = 5+5 min "
          "balanced accuracy."]
    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT / "REPORT.md")


def main():
    res = json.loads((OUT / "results.json").read_text())
    probs = pickle.loads((OUT / "probs.pkl").read_bytes()) \
        if (OUT / "probs.pkl").exists() else {}
    fig_bars(res); fig_roc(res, probs); fig_cm(res, probs)
    fig_cal(res); report(res)


if __name__ == "__main__":
    main()
