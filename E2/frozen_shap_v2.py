"""Interventional Shapley importance for the v2 references over every
descriptor, with a LaTeX table for the paper."""
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from frozen_v2 import (OUT, RAD_ALL, CSI_ALL, VIEWS, matrix, model_path,
                       grouped)

ASSETS = OUT.parent.parent.parent / "paper/icc_assets"
LABEL = {
    "csi_amp_mean": "CSI amplitude mean", "csi_amp_std": "CSI amplitude std",
    "csi_amp_q90": "CSI amplitude P90", "csi_rv_mean": "CSI rolling-var. mean",
    "csi_rv_q90": "CSI rolling-var. P90", "csi_rv_max": "CSI rolling-var. max",
    "csi_tv": "CSI total variation", "csi_dop_frac": "CSI Doppler fraction",
    "snr_mean": "Radar SNR mean", "snr_max": "Radar SNR max",
}
for m, name in (("rd", "RD"), ("ra", "RA"), ("re", "RE"), ("xy", "XY")):
    LABEL.update({f"{m}_mean": f"{name} energy mean", f"{m}_p90": f"{name} energy P90",
                  f"{m}_std90": f"{name} variability P90", f"{m}_delta": f"{name} frame delta",
                  f"{m}_peak": f"{name} peak/background"})
for mod, name in (("csi", "CSI"), ("rad", "Radar")):
    for j in (1, 2, 3):
        LABEL[f"{mod}_pc{j}_mean"] = f"{name} PC{j} mean"
        LABEL[f"{mod}_pc{j}_std"] = f"{name} PC{j} std"


def predictor(key, seed=0):
    name, view = key.split("|", 1)
    def single(v):
        obj = joblib.load(model_path(name, v, seed))
        cols = obj["columns"]
        def call(frame):
            return obj["model"].predict_proba(
                obj["scaler"].transform(matrix(frame, cols, obj["fill"])))[:, 1]
        return call
    if view == "late probability fusion":
        a, b = single("CSI"), single("radar")
        return RAD_ALL + CSI_ALL, lambda f: (a(f) + b(f)) / 2
    return VIEWS[view], single(view)


def explain(fore, back, cols, fn, seed=73, orders=24):
    rng = np.random.default_rng(seed)
    phi = np.zeros((len(fore), len(cols)))
    for _ in range(orders):
        cur = back.iloc[rng.integers(len(back), size=len(fore))][cols].reset_index(drop=True).copy()
        base = fn(cur)
        for j in rng.permutation(len(cols)):
            cur[cols[j]] = fore[cols[j]].to_numpy()
            after = fn(cur)
            phi[:, j] += after - base
            base = after
    return phi / orders


def main():
    sel = json.loads((OUT / "selection.json").read_text())
    train = pd.read_csv(OUT / "train_windows.csv")
    val = pd.read_csv(OUT / "validation_windows.csv")
    real_val = val[val.aug == 0].reset_index(drop=True)
    rng = np.random.default_rng(73)
    idx = np.concatenate([rng.choice(np.flatnonzero(real_val.label.to_numpy() == c), 12, replace=False)
                          for c in (0, 1)])
    fore = real_val.iloc[idx].reset_index(drop=True)
    targets = {"ML": sel["references"]["ML"]["key"], "DL": sel["references"]["DL"]["key"],
               "Early": sel["selected_views"]["early fusion"]}
    table = pd.DataFrame(index=RAD_ALL + CSI_ALL)
    for col, key in targets.items():
        cols, fn = predictor(key)
        phi = explain(fore, train[train.aug == 0], cols, fn)
        table.loc[cols, col] = np.abs(phi).mean(axis=0)
        print(col, key, flush=True)
    table.index.name = "descriptor"
    table.to_csv(OUT / "shap_table.csv")
    (OUT / "shap_method.json").write_text(json.dumps({
        "targets": targets, "seed_models": 0, "orders": 24, "seed": 73,
        "foreground": "24 real validation windows (12 per class)",
        "background": "random real training window per foreground row and order",
        "estimand": "Monte Carlo interventional Shapley contribution to occupied probability"}, indent=2))
    # LaTeX: two side-by-side blocks, descriptors ordered by group.
    order = CSI_ALL + RAD_ALL
    half = (len(order) + 1) // 2
    def cell(v):
        return "--" if pd.isna(v) else f"{v:.3f}".lstrip("0")
    lines = []
    for i in range(half):
        parts = []
        for j in (i, i + half):
            if j < len(order):
                d = order[j]
                parts.append(" & ".join([LABEL.get(d, d)] + [cell(table.loc[d, c]) for c in targets]))
            else:
                parts.append(" & & & ")
        lines.append(" & ".join(parts) + r" \\")
    lines.append(r"\bottomrule")
    (ASSETS / "shap_table_rows.tex").write_text("\n".join(lines))
    print(table.sort_values("ML", ascending=False).head(12))


if __name__ == "__main__":
    main()
