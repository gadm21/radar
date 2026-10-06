"""Build E1/outputs/REPORT.md from analysis.json + rankings + tables."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.OUTPUT_DIR


def _md_table(rows, cols, headers=None):
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols)
                   + " |")
    return "\n".join(out)


def main():
    res = json.loads((OUT / "analysis.json").read_text())
    L = []
    L.append("# E1 — Class-Separability Analysis: Empty vs Occupied\n")
    L.append("Physical descriptors and compressed-PCA features that "
             "distinguish the two classes, computed identically on the "
             "new **multilink** captures (`multilink_train/`, 2 CSI links "
             "+ BGT60TR13C radar, JSONL streams) and the **legacy** "
             "train/test minutes (1 CSI link + radar, E2 cache + "
             "published feature tables extended with the same physical "
             "descriptors).\n")

    L.append("## Datasets\n")
    ml5 = pd.read_csv(OUT / "features_multilink_5s.csv")
    o5 = pd.read_csv(OUT / "features_old_5s_ext.csv")
    rows = [
        {"dataset": "multilink (4 captures)",
         "windows_5s": len(ml5),
         "empty": int((ml5.label == 0).sum()),
         "occupied": int((ml5.label == 1).sum()),
         "csi_links": 2, "grouping": "capture"},
        {"dataset": "legacy train (wall1+wall2 mins)",
         "windows_5s": int((o5.dataset == "train").sum()),
         "empty": int(((o5.dataset == "train") & (o5.label == 0)).sum()),
         "occupied": int(((o5.dataset == "train") & (o5.label == 1)).sum()),
         "csi_links": 1, "grouping": "minute"},
        {"dataset": "legacy test (wall3 mins)",
         "windows_5s": int((o5.dataset == "test").sum()),
         "empty": int(((o5.dataset == "test") & (o5.label == 0)).sum()),
         "occupied": int(((o5.dataset == "test") & (o5.label == 1)).sum()),
         "csi_links": 1, "grouping": "minute"},
    ]
    L.append(_md_table(rows, list(rows[0])) + "\n")

    L.append("## Method\n")
    L.append("Per 5 s and 10 s non-overlapping windows:\n"
             "- **CSI (per link)**: amplitude mean/std/q90; 0.2 s causal "
             "rolling-variance mean/q90/max (log1p); total variation of "
             "the mean-amplitude trace; high-band PSD fraction "
             "(micro-Doppler proxy); `pcv1-3` — variance of the 3-D PCA "
             "projection of the per-second 104-d amplitude 'dots'.\n"
             "- **radar**: clutter-masked `snr_max/mean`; `rad_pcv1-3` — "
             "variance of the 3-D PCA projection of the per-second mean "
             "RD-map 'dots'; per-view RD/RA/RE/XY descriptors "
             "(mean/p90/std90/delta/peak); physical descriptors "
             "`rd_range_cm` (range-bin centroid), `rd_dop_cm` (doppler "
             "centroid), `rd_dop_spread`, `rd_dca` (hot-pixel area "
             "fraction), `rd_energy`.\n\n"
             "Screening is iterative: junk-column drop -> univariate "
             "ROC-AUC/Cohen's-d/Mann-Whitney -> collinearity prune "
             "(|r|>0.95) -> RF importance -> top-15 grouped 5-fold CV "
             "probe (LogReg and RF; folds grouped by capture/minute so "
             "windows never leak across recordings).\n")

    L.append("## Separation quality (W=5s)\n")
    L.append("`cv_*` = 5-fold stratified grouped CV (by minute) for "
             "legacy, window-stratified CV for multilink (only 4 capture "
             "groups). `loco_e/o` = leave-one-capture-out recall, "
             "empty/occupied — the honest cross-session estimate.\n")
    rows = []
    for k, v in res.items():
        if "_5s/" not in k:
            continue
        loco = "-"
        if np.isfinite(v.get("lr_recall_empty", np.nan)):
            loco = (f"{v['lr_recall_empty']:.2f}/"
                    f"{v['lr_recall_occ']:.2f}")
        rows.append({"group": k,
                     "n": v["n_windows"],
                     "best_feature": v["best_single"],
                     "best_auc": f"{v['best_single_auc']:.3f}",
                     "cv_logreg": f"{v['cv_acc_logreg']:.3f}",
                     "cv_rf": f"{v['cv_acc_rf']:.3f}",
                     "loco_e/o": loco})
    L.append(_md_table(rows, list(rows[0])) + "\n")

    for W in (5, 10):
        L.append(f"## Top descriptors (combined rank), W={W}s\n")
        for name, tag in (("multilink", f"ml{W}s"),
                          ("legacy train", f"oldtrain{W}s"),
                          ("legacy test", f"oldtest{W}s")):
            p = OUT / f"rankings_{tag}_fusion.csv"
            if not p.exists():
                continue
            r = pd.read_csv(p).head(12)
            rows = [{"feature": f, "auc": f"{a:.3f}",
                     "d": f"{d:+.2f}", "rf_imp": f"{i:.4f}"}
                    for f, a, d, i in zip(r.feature, r.auc_dir,
                                          r.cohen_d, r.rf_imp)]
            L.append(f"**{name}**\n")
            L.append(_md_table(rows, ["feature", "auc", "d", "rf_imp"])
                     + "\n")

    L.append("## Cross-dataset check — shared descriptors\n")
    L.append("Same feature, univariate direction-free AUC, multilink "
             "vs legacy train vs legacy test (W=5s):\n")
    from sklearn.metrics import roc_auc_score
    shared = {}
    ml = pd.read_csv(OUT / "features_multilink_5s.csv")
    for s in C.CSI_LINK_COLS:                       # csiM_ derivation
        ml[f"csiM_{s}"] = ml[[f"csi1_{s}", f"csi2_{s}"]].mean(axis=1)
    old = pd.read_csv(OUT / "features_old_5s_ext.csv")
    shared["multilink"] = ml
    shared["legacy train"] = old[old.dataset == "train"]
    shared["legacy test"] = old[old.dataset == "test"]

    def auc_of(df, col):
        if col not in df.columns:
            return np.nan
        d = df[[col, "label"]].dropna()
        if len(d) < 40 or d.label.nunique() < 2:
            return np.nan
        try:
            a = roc_auc_score(d.label, d[col])
            return max(a, 1 - a)
        except Exception:
            return np.nan

    # rows: (display name, multilink col, legacy col)
    keys = [("snr_mean", "snr_mean", "snr_mean"),
            ("snr_max", "snr_max", "snr_max"),
            ("rad_pcv1", "rad_pcv1", "rad_pcv1"),
            ("rd_delta", "rd_delta", "rd_delta"),
            ("rd_std90", "rd_std90", "rd_std90"),
            ("rd_dca", "rd_dca", "rd_dca"),
            ("rd_dop_spread", "rd_dop_spread", "rd_dop_spread"),
            ("rd_range_cm", "rd_range_cm", "rd_range_cm"),
            ("csi rv_mean", "csiM_rv_mean", "csi_rv_mean"),
            ("csi pcv1", "csiM_pcv1", "csi_pcv1"),
            ("csi amp_std", "csiM_amp_std", "csi_amp_std"),
            ("csi amp_q90", "csiM_amp_q90", "csi_amp_q90"),
            ("csi tv", "csiM_tv", "csi_tv"),
            ("csi dop_frac", "csiM_dop_frac", "csi_dop_frac")]
    rows = []
    for disp, kml, kold in keys:
        row = {"descriptor": disp}
        for ds, df_ in shared.items():
            kk = kml if ds == "multilink" else kold
            v = auc_of(df_, kk)
            row[ds] = f"{v:.3f}" if np.isfinite(v) else "-"
        rows.append(row)
    L.append(_md_table(rows, ["descriptor", *shared]) + "\n")
    L.append("`-` = descriptor not present for that dataset.\n")

    L.append("## Figures\n")
    for f in sorted((OUT / "figs").glob("*.png")):
        L.append(f"- `figs/{f.name}`")
    L.append("\n")
    L.append("## Caveats\n"
             "- Multilink link2 (thoth-toronto) stopped streaming ~70 min "
             "into the Oct-5 occupied capture; affected windows are "
             "masked by `csi2_ok`.\n"
             "- PCAs are fit unsupervised per dataset (multilink) or "
             "reused from E2 (legacy) — pcv magnitudes are not directly "
             "comparable across datasets; compare AUCs, not raw values.\n"
             "- `csi_tv`/`csi_dop_frac`/`rd_*_cm` units are descriptor-"
             "internal (bins/log-power), not calibrated physics units.\n")

    (OUT / "REPORT.md").write_text("\n".join(L))
    print("wrote", OUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
