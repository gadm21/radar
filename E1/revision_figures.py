"""Paper figures from frozen E1 outputs; never fit a PCA on validation/test.

Run with the repository's Python dependencies on PYTHONPATH.  The 60-second
level uses six consecutive 10-second vectors (ten would span 100 seconds).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figs"
DEEP = OUT / "pca_deep"
BLUE, ORANGE = "#0072B2", "#D55E00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.edgecolor": "#4C5963", "axes.linewidth": .7,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=360, bbox_inches="tight")
    fig.savefig(FIG / Path(name).with_suffix(".pdf"), dpi=360,
                bbox_inches="tight")
    plt.close(fig)


def stable_descriptors():
    old = pd.read_csv(OUT / "features_old_5s_ext.csv")
    development = {"Train": old[old.dataset.eq("train")],
                   "Validation": old[old.dataset.eq("test")]}
    shared = sorted(c for c in set.intersection(*(set(d.columns) for d in development.values()))
                    if pd.api.types.is_numeric_dtype(old[c]))
    preferred = ["ra_mean", "re_mean", "xy_std90", "re_std90", "xy_mean",
                 "snr_mean", "ra_std90", "xy_p90", "re_p90"]
    results = []
    for col in preferred + [x for x in shared if x not in preferred]:
        if col not in shared or col in {"label", "win", "aug", "is_val"}:
            continue
        row = {"descriptor": col}
        aucs = []
        for name, d in development.items():
            v = d[[col, "label"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(v) < 30 or v.label.nunique() < 2 or v[col].nunique() < 2:
                break
            auc = roc_auc_score(v.label, v[col])
            row[f"auc_{name}"] = auc
            row[f"n_{name}"] = len(v)
            aucs.append(auc)
        if len(aucs) == 2:
            row["consistent"] = (all(x > .5 for x in aucs) or
                                 all(x < .5 for x in aucs))
            row["min_directional_auc"] = min(max(x, 1-x) for x in aucs)
            results.append(row)
    summary = pd.DataFrame(results)
    selected = summary[(summary.consistent) &
                       (summary.min_directional_auc >= .55)].sort_values(
                           "min_directional_auc", ascending=False).head(6)
    if selected.empty:
        raise RuntimeError("No descriptors pass train/validation stability")
    # Freeze selected descriptors before opening multilink labels.
    (OUT / "stable_descriptor_selection.json").write_text(json.dumps({
        "selection_splits":["train","validation"],
        "selected":selected.descriptor.tolist(),
        "rule":"same AUC direction and minimum direction-adjusted AUC >= 0.55"
    },indent=2))
    ml = pd.read_csv(OUT / "features_multilink_5s.csv")
    groups = {**development, "Multilink test":ml}
    for i,row in summary.iterrows():
        col=row.descriptor
        if col not in ml.columns: continue
        v=ml[[col,"label"]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(v)>=30 and v.label.nunique()==2 and v[col].nunique()>1:
            auc=roc_auc_score(v.label,v[col])
            summary.loc[i,"auc_Multilink test"]=auc
            summary.loc[i,"n_Multilink test"]=len(v)
            summary.loc[i,"test_same_direction"]=(auc>.5)==(row.auc_Train>.5)
    summary.to_csv(OUT / "stable_descriptor_screen.csv",index=False)
    # Present the physically interpretable movement-variability family.
    # The choice is based on descriptor type and development-set screening;
    # multilink labels remain assessment-only and never select a model input.
    display = selected.set_index("descriptor").loc[
        ["ra_std90", "re_std90"]].reset_index()
    fig, axes = plt.subplots(len(display), 3, figsize=(10.8, 2.25*len(display)),
                             squeeze=False)
    rng = np.random.default_rng(17)
    for r, col in enumerate(display.descriptor):
        for c, (name, d) in enumerate(groups.items()):
            ax = axes[r, c]
            vals = [d.loc[d.label.eq(k), col].replace([np.inf, -np.inf], np.nan)
                    .dropna().to_numpy() for k in (0, 1)]
            # Use central quantiles for display only; AUC uses every finite value.
            pooled = np.concatenate(vals)
            lo, hi = np.quantile(pooled, [.01, .99])
            boxes = ax.boxplot(vals, positions=[0, 1], widths=.42, showfliers=False,
                       patch_artist=True, medianprops={"color": "black"},
                       boxprops={"linewidth": 1.05})
            for patch, fill in zip(boxes["boxes"], ("#CBE6F2", "#F9D9C7")):
                patch.set_facecolor(fill)
                patch.set_edgecolor("#46545D")
            for k, a in enumerate(vals):
                take = rng.choice(len(a), min(130, len(a)), replace=False)
                ax.scatter(k + rng.normal(0, .095, len(take)), a[take], s=6,
                           alpha=.36, color=(BLUE, ORANGE)[k], rasterized=True)
            ax.set_ylim(lo - .08*(hi-lo), hi + .08*(hi-lo))
            ax.set_xticks([0, 1], ["Empty", "Occupied"])
            ax.yaxis.grid(True, color="#E7ECEF", linewidth=.55)
            ax.set_axisbelow(True)
            if r == 0: ax.set_title(name, fontsize=9)
            if c == 0: ax.set_ylabel(col)
            auc = summary.loc[summary.descriptor.eq(col), f"auc_{name}"].iloc[0]
            ax.text(.98, .96, f"AUC {auc:.2f}", ha="right", va="top",
                    transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    _save(fig, "stable_descriptor_box_scatter.png")
    import shutil
    shutil.copy2(FIG / "stable_descriptor_box_scatter.pdf",
                 ROOT.parent / "paper/icc_assets/stable_descriptor_box_scatter.pdf")
    return selected.descriptor.tolist()


def _records(split):
    labels = pd.read_csv(DEEP / f"desc_{split}_5s.csv",
                         usecols=["id", "label"]).groupby("id").label.first()
    if split != "testml":
        z = np.load(DEEP / f"pc_dots_{split}.npz")
        return [(k, z[k], int(labels[k])) for k in z.files if k in labels.index]
    # Historical testml PC dots were fitted on the test captures themselves.
    # Reproject raw per-second dots through the *train* first-level bases.
    import joblib
    models = joblib.load(DEEP / "pcas_deep.joblib")["train"]
    sc_c, p_c = models["sc_c"], models["pca_c"]
    sc_r, p_r = models["sc_r"], models["pca_r"]
    out, dump = [], {}
    for path in sorted((ROOT / "cache/multilink").glob("*.npz")):
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            key = meta["cap"]
            if key not in labels.index: continue
            n = min(len(z["csi1_ok"]), len(z["csi2_ok"]),
                    len(z["radar_rsec"]))
            ok1, ok2 = z["csi1_ok"][:n].astype(bool), z["csi2_ok"][:n].astype(bool)
            okc = ok1 | ok2
            okr = z["radar_rsec"][:n].astype(bool)
            d1, d2 = z["csi1_dots"][:n], z["csi2_dots"][:n]
            denom = ok1.astype(float) + ok2.astype(float)
            csi = np.where(denom[:,None] > 0,
                (d1*ok1[:,None] + d2*ok2[:,None])/np.maximum(denom[:,None],1),
                sc_c.mean_[None])
            # First RD map only: same 576 coordinates as legacy train.
            rd = z["radar_vsec"][:n,0].reshape(n,-1).astype(float)
            csi = np.where(np.isfinite(csi), csi, sc_c.mean_[None])
            rd = np.where(np.isfinite(rd), rd, sc_r.mean_[None])
            pc_c = p_c.transform(sc_c.transform(csi))[:,:3]
            pc_r = p_r.transform(sc_r.transform(rd))[:,:3]
            a = np.column_stack([(okc & okr).astype(float), pc_c, pc_r])
            dump[key] = a
            out.append((key, a, int(labels[key])))
    np.savez_compressed(DEEP / "pc_dots_testml_trainbasis.npz", **dump)
    return out


def hierarchical_pca():
    recs = {s: _records(s) for s in ("train", "val", "testml")}
    fig, axes = plt.subplots(2, 3, figsize=(11.3, 6.7))
    manifest = {}
    for row, (mod, cols) in enumerate((("CSI", slice(1, 4)),
                                      ("Radar", slice(4, 7)))):
        raw = {}
        for split, items in recs.items():
            for key, a, lab in items:
                for start in range(0, len(a)-9, 10):
                    seg = a[start:start+10]
                    good = (seg[:, 0] > .5) & np.isfinite(seg[:, cols]).all(axis=1)
                    if good.sum() >= 5:
                        # Reconstruct missing seconds only from observed PCA
                        # dots within the same ten-second block.
                        v = seg[:, cols].copy()
                        for dim in range(v.shape[1]):
                            v[:, dim] = np.interp(np.arange(10),
                                                  np.flatnonzero(good), v[good, dim])
                        raw[(split, key, start)] = (v.reshape(-1), lab)
        fit10 = np.stack([x for (s, _, _), (x, _) in raw.items() if s == "train"])
        sc10 = StandardScaler().fit(fit10)
        p10 = PCA(n_components=3, random_state=0).fit(sc10.transform(fit10))
        dots10 = {k: (p10.transform(sc10.transform(x[None]))[0], lab)
                  for k, (x, lab) in raw.items()}
        raw60 = {}
        for split, items in recs.items():
            for key, a, lab in items:
                for start in range(0, len(a)-59, 60):
                    ks = [(split, key, start + 10*i) for i in range(6)]
                    present = [i for i, k in enumerate(ks) if k in dots10]
                    if len(present) >= 4:
                        v = np.stack([dots10[ks[i]][0] for i in present])
                        filled = np.column_stack([
                            np.interp(np.arange(6), present, v[:, dim])
                            for dim in range(3)])
                        raw60[(split, key, start)] = (filled.reshape(-1), lab)
        fit60 = np.stack([x for (s, _, _), (x, _) in raw60.items() if s == "train"])
        sc60 = StandardScaler().fit(fit60)
        p60 = PCA(n_components=2, random_state=0).fit(sc60.transform(fit60))
        dots60 = {k: (p60.transform(sc60.transform(x[None]))[0], lab)
                  for k, (x, lab) in raw60.items()}
        pts1 = [(a[t, cols], lab) for _, a, lab in recs["testml"]
                for t in range(len(a)) if a[t, 0] > .5 and
                np.isfinite(a[t, cols]).all()]
        pts2 = [v for k, v in dots10.items() if k[0] == "testml"]
        pts3 = [v for k, v in dots60.items() if k[0] == "testml"]
        for j, pts in enumerate((pts1, pts2, pts3)):
            ax = axes[row, j]
            rng = np.random.default_rng(104+j)
            for lab, color in ((0, BLUE), (1, ORANGE)):
                a = np.array([x[:2] for x, y in pts if y == lab])
                if len(a) > 2500: a = a[rng.choice(len(a), 2500, replace=False)]
                ax.scatter(a[:, 0], a[:, 1], s=5 if j < 2 else 13,
                           alpha=.42 if j < 2 else .75, color=color,
                           marker=("o", "^")[lab],
                           rasterized=True, label=("Empty", "Occupied")[lab])
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            if row == 0: ax.set_title(("1 second", "10 seconds", "1 minute")[j])
            if j == 0: ax.annotate(mod, (-.32, .5), xycoords="axes fraction",
                                   rotation=90, va="center", fontsize=11)
            if row == 0 and j == 2: ax.legend(fontsize=8)
        manifest[mod] = {"fit_10s": len(fit10), "fit_1m": len(fit60),
                         "test_10s": len(pts2), "test_1m": len(pts3),
                         "explained_10s": p10.explained_variance_ratio_[:2].tolist(),
                         "explained_1m": p60.explained_variance_ratio_[:2].tolist()}
    fig.suptitle("Hierarchical PCA: fixed train bases, independent multilink projection")
    fig.tight_layout()
    _save(fig, "hierarchical_pca.png")
    (OUT / "hierarchical_pca.json").write_text(json.dumps(manifest, indent=2))


def timeseries():
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.1), sharex="col")
    for j, (split, title) in enumerate((("train", "Train"), ("val", "Validation"),
                                       ("testml", "Multilink test"))):
        d = pd.read_csv(DEEP / f"desc_{split}_5s.csv",
                        usecols=["id", "s0", "label", "csi_amp_mean", "snr_mean"])
        # Hop is two seconds; a ten-point mean spans 20 seconds. No rolling
        # window crosses a minute/capture boundary.
        d = d.sort_values(["id", "s0"])
        for col, row in (("csi_amp_mean", 0), ("snr_mean", 1)):
            d[f"smooth_{col}"] = d.groupby("id")[col].transform(
                lambda x: x.rolling(10, min_periods=2).mean())
            d["minute_bin"] = (d.s0 // 60).astype(int)
            m = d.groupby(["id", "minute_bin"], sort=True).agg(label=("label", "first"),
                    value=(f"smooth_{col}", "mean"))
            ax = axes[row, j]
            x = np.arange(len(m))
            ax.plot(x, m.value, color="#65727C", linewidth=.7, alpha=.7)
            for lab, color in ((0, BLUE), (1, ORANGE)):
                ix = np.flatnonzero(m.label.to_numpy() == lab)
                ax.scatter(ix, m.value.to_numpy()[ix], s=9, color=color,
                           alpha=.68, rasterized=True,
                           marker=("o", "^")[lab],
                           label=("Empty", "Occupied")[lab])
            if row == 0: ax.set_title(title)
            if j == 0: ax.set_ylabel(("CSI mean amplitude", "Radar SBR proxy (dB)")[row])
            if row == 1: ax.set_xlabel("Minute / capture order")
            ax.yaxis.grid(True, color="#E7ECEF", linewidth=.5)
            ax.set_axisbelow(True)
    axes[0, 2].legend(fontsize=7)
    fig.suptitle("20-second rolling means aggregated by labeled recording")
    fig.tight_layout()
    _save(fig, "timeseries_ml.png")


def movement_maps():
    # Derive motion solely from labeled occupied and empty samples.  Taking
    # absolute consecutive differences suppresses static room reflectors.
    from common import ML_CACHE, OLD_CACHE
    fig, axes = plt.subplots(2, 4, figsize=(11.3, 5.7))
    for row, (title, files, field) in enumerate((
        ("Multilink", sorted(ML_CACHE.glob("*.npz")), "radar_vsec"),
        ("Validation", sorted((OLD_CACHE / "test_minutes").glob("*.npz")), "maps"))):
        sums = {lab: np.zeros((2,24,24)) for lab in (0,1)}
        counts = {0:0, 1:0}
        for p in files:
            with np.load(p) as z:
                meta = json.loads(str(z["meta"]))
                lab = int(meta["label"] == "occupied") if row == 0 else int(meta["label"])
                a = z[field].astype(float)
                if a.ndim != 4 or len(a) < 2: continue
                a = a[:, [0,3]]
                motion = np.abs(np.diff(a, axis=0)).mean(axis=0)
                if np.isfinite(motion).all():
                    sums[lab] += motion; counts[lab] += 1
        if min(counts.values()) == 0: continue
        maps = {lab: sums[lab]/counts[lab] for lab in (0,1)}
        for v, name in enumerate(("RD", "XY")):
            vmax = np.quantile(np.stack([maps[0][v], maps[1][v]]), .995)
            for lab in (0,1):
                ax = axes[row, 2*v+lab]
                im = ax.imshow(maps[lab][v], origin="lower", cmap="magma",
                               vmin=0, vmax=vmax)
                ax.set_title(f"{title} {name}: " + ("empty", "occupied")[lab],
                             fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle("Mean per-second map change from labeled samples")
    fig.tight_layout()
    _save(fig, "mean_maps_movement.png")


if __name__ == "__main__":
    import sys
    print("Stable:", stable_descriptors(), flush=True)
    if "--stable-only" not in sys.argv:
        hierarchical_pca()
        timeseries()
        movement_maps()
