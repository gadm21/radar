"""Validation-only interventional SHAP estimate for frozen early-fusion GBM."""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frozen_evaluation import OLD, OUT
from frozen_shap import predictor, explain

key = "GBM|early fusion"
data = pd.read_csv(OLD)
train = data[data.rec_id.str.startswith("train/")].copy()
val = data[data.rec_id.str.startswith("test_minutes/")].copy()
rng = np.random.default_rng(73)
idx = np.concatenate([rng.choice(np.flatnonzero(val.label.to_numpy() == c),
                                 12, replace=False) for c in (0, 1)])
foreground = val.iloc[idx].reset_index(drop=True)
cols, fn = predictor(key)
phi = explain(foreground, train, cols, fn, seed=73, orders=24)
scores = np.abs(phi).mean(axis=0)
order = np.argsort(scores)[::-1]
rows = [{"model":key,"descriptor":cols[j],"mean_abs_shap":float(scores[j])}
        for j in order]
pd.DataFrame(rows).to_csv(OUT / "early_fusion_shap.csv", index=False)

names = {"csi_amp_q90":"CSI amplitude 90th pct.",
         "csi_amp_mean":"CSI amplitude mean",
         "csi_amp_std":"CSI amplitude std.",
         "ra_std90":"RA temporal std. 90th pct.",
         "re_std90":"RE temporal std. 90th pct.",
         "xy_std90":"XY temporal std. 90th pct."}
top = order[:7][::-1]
fig, ax = plt.subplots(figsize=(3.42, 2.32))
ax.barh([names.get(cols[j], cols[j].replace("_", " ")) for j in top],
        scores[top], color=["#176C99" if cols[j].startswith("csi_")
                            else "#C05B22" for j in top])
ax.set_xlabel("Mean absolute probability contribution", fontsize=7.5)
ax.tick_params(labelsize=7)
ax.grid(axis="x", alpha=.18)
ax.set_axisbelow(True)
fig.tight_layout()
dest = OUT / "figs" / "early_fusion_shap.pdf"
fig.savefig(dest, bbox_inches="tight")
fig.savefig(dest.with_suffix(".png"), dpi=350, bbox_inches="tight")
plt.close(fig)
print(pd.DataFrame(rows).head(10).to_string(index=False))
