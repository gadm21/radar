"""icc_multimodal2 revision artifacts: frozen_v3 models on the newtest set.

Protocol:
  * weights: frozen_v3 trained models (unchanged)
  * validation = old validation minutes (224, balanced) + old multilink
    test minutes (360) = 584 minutes
  * per-seed thresholds fitted on the combined validation minutes
    (max balanced accuracy, ties toward 0.5)
  * test = labeled newtest captures (wall 1), minutes = mean of their four
    15-s window probabilities; only complete 4-window minutes count
  * view leaders / ML-DL references selected by mean validation-minute AUC

Outputs under newtest/paper_eval/:
  windows.csv              newtest 15-s windows, all 50 descriptors + probs
  dots.npz                 per-capture level-1 joint dots (nsec, 7)
  {val,test}_minute_probabilities.npz   per (model,view,seed) minute probs
  evaluation.json          same schema as frozen_v3/evaluation.json
  selection.json           view leaders + ML/DL references
  capture_summary.json     minute counts + date ranges per data set
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
NEWTEST = ROOT / "newtest"
FROZEN = ROOT / "E2" / "outputs" / "frozen_v3"
OUT = NEWTEST / "paper_eval"
sys.path.insert(0, str(ROOT.parent / "thothcraft" / "whispy" / "packages"
                       / "whispy-model-occ"))
sys.path.insert(0, str(ROOT / "E2"))
from whispy_model_occ import hier          # noqa: E402
from frozen_v3 import read_test            # noqa: E402

MODELS = ["RF", "GBM", "SVM", "MLP"]
VIEWS = ["CSI", "radar", "early_fusion"]
VIEW4 = VIEWS + ["late"]
VIEW_LONG = {"CSI": "CSI", "radar": "radar", "early_fusion": "early fusion",
             "late": "late probability fusion"}
CAPS = [
    ("empty_2fa6d3f1bf90", "2fa6d3f1bf90", 0),
    ("empty_whispy", None, 0),
    ("occupied_whispy", None, 1),
    ("occupied_whispy2", None, 1),
    ("occupied_whispy11", None, 1),
    ("occupied_whispy12", None, 1),
    ("capture3_768c3d688716", "768c3d688716", 1),
]


def load_models():
    out = {m: {v: [] for v in VIEWS} for m in MODELS}
    for m in MODELS:
        for v in VIEWS:
            for s in range(5):
                p = FROZEN / "models" / f"{m}__{v}__s{s}.joblib"
                out[m][v].append(joblib.load(p))
    return out


def load_samples(path, ptype=None):
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if ptype and d.get("payload_type") != ptype:
                continue
            out.append((float(d["timestamp"]), d["payload"]))
    return out


def proba_matrix(models, X, cols_view):
    """(n_windows, 5) seed probabilities for one model/view combo."""
    x = np.asarray(X, float)
    x[~np.isfinite(x)] = np.nan
    ps = []
    for obj in models:
        xi = np.where(np.isnan(x), obj["fill"], x)
        z = obj["scaler"].transform(xi)
        ps.append(obj["model"].predict_proba(z)[:, 1])
    return np.stack(ps, axis=1)


def window_rows(cap_dir, name, truth, bundle):
    man = json.loads((cap_dir / "manifest.json").read_text())
    t0 = float(man["started_at"])
    nsec = int(np.ceil((float(man["stopped_at"]) - t0)
                       if "stopped_at" in man else float(man["duration_s"])))
    csi_raw = load_samples(cap_dir / "csi-c50e.jsonl", "csi_raw")
    rad = load_samples(cap_dir / "radar-a316.jsonl")
    grid = hier.build_seconds(csi_raw, rad, t0, nsec)
    csi = grid.csi_arrays(nsec)
    radar = grid.radar_arrays(nsec)
    dots = hier.joint_dots(csi, radar, nsec, bundle["level1"])
    hie = bundle["hierarchy"]
    rows = []
    for start in range(0, nsec - hier.WIN + 1, hier.WIN):
        csi_rows = [hier._csi_5s(csi, start + b * hier.BLOCK,
                                 min(start + (b + 1) * hier.BLOCK, nsec))
                    for b in range(hier.BLOCKS_PER_WIN)]
        rad_rows = [hier._radar_5s(grid.frames, start + b * hier.BLOCK,
                                   start + (b + 1) * hier.BLOCK)
                    for b in range(hier.BLOCKS_PER_WIN)]
        with np.errstate(invalid="ignore"):
            csi_phys = np.nanmean(np.stack(csi_rows), axis=0)
            rad_phys = np.nanmean(np.stack(rad_rows), axis=0)
        csi_x = np.r_[csi_phys, hier.window_hier(
            dots, hier.MOD_COLS["csi"], start, hie["csi"])]
        rad_x = np.r_[rad_phys, hier.window_hier(
            dots, hier.MOD_COLS["rad"], start, hie["rad"])]
        rows.append({"unit": name, "minute": start // 60,
                     "win": (start % 60) // hier.WIN, "label": truth,
                     "start_s": start,
                     **dict(zip(hier.CSI_ALL, csi_x)),
                     **dict(zip(hier.RAD_ALL, rad_x))})
    return pd.DataFrame(rows), dots


def fit_threshold(probs, labels):
    probs = np.asarray(probs, float)
    y = np.asarray(labels, int)
    best = (-1.0, -1.0, 0.5)
    for t in np.unique(probs):
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        tpr = tp / max(tp + int(((pred == 0) & (y == 1)).sum()), 1)
        tnr = tn / max(tn + int(((pred == 1) & (y == 0)).sum()), 1)
        score = ((tpr + tnr) / 2, -abs(float(t) - .5), float(t))
        if score > best:
            best = score
    return best[2]


def minute_probs(win_df, prob_cols):
    """Complete 4-window minutes -> (n_min, n_seeds) probs + labels."""
    g = win_df.groupby(["unit", "minute"], sort=True)
    sizes = g.size()
    full_idx = sizes[sizes == 4].index
    sub = win_df.set_index(["unit", "minute"]).loc[full_idx]
    agg = sub.groupby(["unit", "minute"], sort=True)[prob_cols].mean()
    labels = sub.groupby(["unit", "minute"], sort=True)["label"].first()
    return agg.to_numpy(), labels.to_numpy(int)


def metrics(win_p, win_y, min_p, min_y, thr):
    pred = (min_p >= thr).astype(int)
    wpred = (win_p >= thr).astype(int)
    cm = np.zeros((2, 2), int)
    for a in (0, 1):
        for p in (0, 1):
            cm[a, p] = int(((min_y == a) & (pred == p)).sum())
    return {
        "accuracy": float((pred == min_y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(min_y, pred)),
        "roc_auc": float(roc_auc_score(min_y, min_p)),
        "window_accuracy": float((wpred == win_y).mean()),
        "window_roc_auc": float(roc_auc_score(win_y, win_p)),
        "confusion_matrix": cm.tolist()}


def agg(vals):
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=0)),
            "per_seed": [float(v) for v in a]}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    models = load_models()
    import whispy_model_occ.hier as _h
    bundle = _h.OccHierModel()._load()

    # ------------------------------------------------- newtest windows
    print("scoring newtest captures ...", flush=True)
    frames, dots = [], {}
    for name, cid, truth in CAPS:
        cap_dir = NEWTEST / name if cid is None else NEWTEST / name / cid
        df, d = window_rows(cap_dir, name, truth, bundle)
        dots[name] = d
        frames.append(df)
        print(f"  {name}: {len(df)} windows", flush=True)
    tw = pd.concat(frames, ignore_index=True)
    np.savez_compressed(OUT / "dots.npz", **dots)

    # ------------------------------------------------- window probabilities
    print("predicting window probabilities ...", flush=True)
    for m in MODELS:
        for v in VIEWS:
            cols = models[m][v][0]["columns"]
            pm = proba_matrix(models[m][v], tw[cols].to_numpy(float), v)
            for s in range(5):
                tw[f"p_{m}_{v}_{s}"] = pm[:, s]
    for m in MODELS:
        for s in range(5):
            tw[f"p_{m}_late_{s}"] = (tw[f"p_{m}_CSI_{s}"]
                                    + tw[f"p_{m}_radar_{s}"]) / 2
    tw.to_csv(OUT / "windows.csv", index=False)

    # ------------------------------------------------- combined validation
    print("building combined-validation probabilities ...", flush=True)
    vw = pd.read_csv(FROZEN / "validation_windows.csv")
    vw["source"] = "val"
    mw = read_test()
    mw["source"] = "ml"
    val_frames = {}
    for split, frame in (("val", vw), ("ml", mw)):
        for m in MODELS:
            for v in VIEWS:
                cols = models[m][v][0]["columns"]
                pm = proba_matrix(models[m][v],
                                  frame[cols].to_numpy(float), v)
                for s in range(5):
                    frame[f"p_{m}_{v}_{s}"] = pm[:, s]
        for m in MODELS:
            for s in range(5):
                frame[f"p_{m}_late_{s}"] = (frame[f"p_{m}_CSI_{s}"]
                                            + frame[f"p_{m}_radar_{s}"]) / 2
        val_frames[split] = frame
    valw = pd.concat([val_frames["val"], val_frames["ml"]],
                     ignore_index=True)
    valw.to_csv(OUT / "validation_windows_combined.csv", index=False)

    prob_cols = {m: [f"p_{m}_{v}_{s}" for v in VIEW4 for s in range(5)]
                 for m in MODELS}
    all_cols = [c for m in MODELS for c in prob_cols[m]]

    vmp, vmy = minute_probs(valw, all_cols)
    tmp, tmy = minute_probs(tw, all_cols)
    np.savez_compressed(OUT / "val_minute_probabilities.npz",
                        probs=vmp, labels=vmy, columns=all_cols)
    np.savez_compressed(OUT / "test_minute_probabilities.npz",
                        probs=tmp, labels=tmy, columns=all_cols)

    # ------------------------------------------------- metrics / selection
    print("computing metrics ...", flush=True)
    col_idx = {f"{m}|{VIEW_LONG[v]}":
               [all_cols.index(f"p_{m}_{v}_{s}") for s in range(5)]
               for m in MODELS for v in VIEW4}
    results, thresholds = {}, {}
    inv = {v: k for k, v in VIEW_LONG.items()}
    tw_idx = tw.set_index(["unit", "minute"])
    for key, idxs in col_idx.items():
        mname, vname = key.split("|")[0], inv[key.split("|", 1)[1]]
        seed_rec = {"val": [], "test": [], "thr": []}
        win_p_all = tw[[f"p_{mname}_{vname}_{s}" for s in range(5)]].to_numpy()
        win_y = tw["label"].to_numpy(int)
        for s_i, s in enumerate(range(5)):
            thr = fit_threshold(vmp[:, idxs[s_i]], vmy)
            vm = metrics(win_p_all[:, s_i], win_y, tmp[:, idxs[s_i]], tmy, thr)
            # validation metrics for this seed
            vw_prob_cols = [f"p_{mname}_{vname}_{s}"]
            vwp = valw[vw_prob_cols[0]].to_numpy()
            vwy = valw["label"].to_numpy(int)
            vp = (vwp >= thr).astype(int)
            v_min = vmp[:, idxs[s_i]]
            vpred = (v_min >= thr).astype(int)
            vv = {"accuracy": float((vpred == vmy).mean()),
                  "balanced_accuracy": float(
                      balanced_accuracy_score(vmy, vpred)),
                  "roc_auc": float(roc_auc_score(vmy, v_min)),
                  "window_accuracy": float((vp == vwy).mean()),
                  "window_roc_auc": float(roc_auc_score(vwy, vwp))}
            seed_rec["val"].append(vv)
            seed_rec["test"].append(vm)
            seed_rec["thr"].append(float(thr))
        results[key] = {
            "validation": {k: agg([r[k] for r in seed_rec["val"]])
                           for k in ("roc_auc", "accuracy",
                                     "balanced_accuracy", "window_accuracy")},
            "validation_threshold": agg(seed_rec["thr"]),
            "test": {k: agg([r[k] for r in seed_rec["test"]])
                     for k in ("roc_auc", "accuracy", "balanced_accuracy",
                               "window_accuracy")},
            }
        results[key]["test"]["confusion_matrix_seed0"] = \
            seed_rec["test"][0]["confusion_matrix"]
        results[key]["test"]["seeds"] = [
            {"seed": i, "threshold": seed_rec["thr"][i],
             **{k: seed_rec["test"][i][k]
                for k in ("accuracy", "balanced_accuracy", "roc_auc",
                          "window_accuracy", "confusion_matrix")}}
            for i in range(5)]
        thresholds[key] = seed_rec["thr"]

    selected = {}
    for v in VIEW4:
        best = max((m for m in MODELS),
                   key=lambda m: results[f"{m}|{VIEW_LONG[v]}"]
                                      ["validation"]["roc_auc"]["mean"])
        selected[VIEW_LONG[v]] = f"{best}|{VIEW_LONG[v]}"
    ml_ref = max((f"{m}|{VIEW_LONG[v]}" for m in ("RF", "GBM", "SVM")
                  for v in VIEW4),
                 key=lambda k: results[k]["validation"]["roc_auc"]["mean"])
    dl_ref = max((f"MLP|{VIEW_LONG[v]}" for v in VIEW4),
                 key=lambda k: results[k]["validation"]["roc_auc"]["mean"])

    def ref_block(key):
        return {"key": key, "threshold": thresholds[key][0],
                "balanced_accuracy":
                    results[key]["validation"]["balanced_accuracy"]
                    ["per_seed"][0]}

    eval_doc = {"protocol": {
        "validation": "balanced old validation (224) + old multilink test "
                      "(360) = 584 minutes",
        "test": "labeled newtest captures (wall 1), complete minutes",
        "window_seconds": 15, "windows_per_minute": 4,
        "n_val_minutes": int(len(vmy)), "n_test_minutes": int(len(tmy)),
        "n_test_windows": int(len(tw))},
        "selected_views": selected,
        "references": {"ML": ref_block(ml_ref), "DL": ref_block(dl_ref)},
        "results": results}
    (OUT / "evaluation.json").write_text(json.dumps(eval_doc, indent=2))
    (OUT / "selection.json").write_text(json.dumps(
        {"selected_views": selected,
         "references": {"ML": ref_block(ml_ref), "DL": ref_block(dl_ref)}},
        indent=2))

    # ------------------------------------------------- cohort summary
    train = pd.read_csv(FROZEN / "train_windows.csv")
    tr = train.drop_duplicates(["unit", "minute"]).groupby(
        ["label", "aug"]).size()
    vc = valw.drop_duplicates(["unit", "minute"]).groupby(
        ["label", "aug"]).size()
    nte = tw.groupby(["unit", "minute"]).size()
    full = nte[nte == 4]
    tl = tw.set_index(["unit", "minute"]).loc[full.index] \
        .groupby(["unit", "minute"]).label.first()
    summary = {
        "train": {"empty": int(tr.get((0, 0), 0)),
                  "occupied": int(tr.get((1, 0), 0)),
                  "occupied_synth": int(tr.get((1, 1), 0)),
                  "dates": "2026-09-06 to 2026-09-08"},
        "validation": {"empty": int(vc.get((0, 0), 0)),
                       "occupied": int(vc.get((1, 0), 0)),
                       "occupied_synth": int(vc.get((1, 1), 0)),
                       "dates": "2026-09-15; 2026-10-03 to 2026-10-05"},
        "test": {"empty": int((tl == 0).sum()),
                 "occupied": int((tl == 1).sum()),
                 "occupied_synth": 0,
                 "dates": "2026-10-19 to 2026-10-20"}}
    (OUT / "capture_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"selected: {selected}", flush=True)
    print(f"refs: ML={ml_ref} DL={dl_ref}", flush=True)
    print(f"val minutes {len(vmy)}  test minutes {len(tmy)}  "
          f"windows {len(tw)}", flush=True)
    print(f"elapsed {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
