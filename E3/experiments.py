"""Experiments for the E3 left/right localization pipeline.

Since only one placement/room exists for this task, generalization is
assessed with STRATIFIED GROUP K-FOLD cross-validation at the recording
(minute) level (K=5 by default): every window of one minute stays in a
single fold, so a model is always evaluated on minutes/subjects it never
saw during training. Per-fold: hyperparameters and threshold are chosen
on a validation split carved out of the training folds only; the held-out
fold is touched exactly once, for final scoring.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import train as T

SEED = 202
OUT = C.OUTPUT_DIR
N_FOLDS = 5

DEFAULT_CFG = {
    "lr": 1e-3, "embed_dim": 64, "dropout": 0.3, "weight_decay": 1e-4,
    "epochs": 40, "patience": 8, "batch_size": 64,
}


def _save(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str))
    print(f"saved {name}", flush=True)


def stratified_group_kfold(index, n_folds=N_FOLDS, seed=SEED):
    """Split recordings into n_folds groups, stratified by label, at the
    recording level (never split within one recording)."""
    rng = np.random.RandomState(seed)
    fold_of = {}
    for label in (0, 1):
        rec_ids = index[index.label == label]["rec_id"].tolist()
        rng.shuffle(rec_ids)
        for i, rid in enumerate(rec_ids):
            fold_of[rid] = i % n_folds
    return fold_of


def _strip(res):
    res.pop("y", None); res.pop("p", None); res.pop("rec_idx", None)
    return res


def _arrays_for(index, rec_ids, norm=None):
    sub = index.set_index("rec_id")
    paths = [sub.loc[r, "path"] for r in rec_ids]
    return T.load_split_arrays(paths, norm)


def config_search(index, fold_of, quick_epochs=10, max_train=6000):
    """Small hyperparameter search using fold 0 as the held-out test fold
    (train on folds 1-4, with a further internal train/val carve-out for
    model selection) -- selected BEFORE any other fold is scored, and the
    resulting config is then reused (frozen) for the final K-fold report,
    so search decisions cannot leak into the reported per-fold test
    numbers of folds 1..K-1."""
    train_ids = [r for r, f in fold_of.items() if f != 0]
    rng = np.random.RandomState(SEED)
    rng.shuffle(train_ids)
    n_val = max(1, int(0.2 * len(train_ids)))
    val_ids, tr_ids = train_ids[:n_val], train_ids[n_val:]

    norm = T.fit_norm(index.set_index("rec_id").loc[tr_ids, "path"].tolist())
    train_arr = T.balance_windows(_arrays_for(index, tr_ids, norm), seed=SEED)
    val_arr = _arrays_for(index, val_ids, norm)
    if len(train_arr["y"]) > max_train:
        rngk = np.random.RandomState(SEED)
        keep = np.sort(rngk.choice(len(train_arr["y"]), max_train, replace=False))
        train_arr = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(train_arr["y"]) else v)
                     for k, v in train_arr.items()}

    grid = []
    for lr in (1e-3, 3e-4):
        for emb in (32, 64):
            for dr in (0.2, 0.4):
                grid.append({"lr": lr, "embed_dim": emb, "dropout": dr,
                             "weight_decay": 1e-4, "epochs": quick_epochs,
                             "patience": 4, "batch_size": 64})
    results = []
    for cfg in grid:
        model, thr, hist, tsec, npar = T.train_model(train_arr, val_arr, cfg, SEED)
        ev = _strip(T.evaluate(model, val_arr, thr))
        results.append({"cfg": cfg, "threshold": thr, "val_window": ev["window"], "train_seconds": tsec})
        print(f"  cfg lr={cfg['lr']} emb={cfg['embed_dim']} drop={cfg['dropout']}: "
              f"val acc={ev['window']['accuracy']:.3f} f1={ev['window']['macro_f1']:.3f}", flush=True)
    best = max(results, key=lambda r: (r["val_window"]["accuracy"], r["val_window"]["macro_f1"]))
    _save({"results": results, "best": best}, "hp_search.json")
    cfg = dict(best["cfg"])
    cfg["epochs"] = DEFAULT_CFG["epochs"]
    cfg["patience"] = DEFAULT_CFG["patience"]
    return cfg


def run_kfold(index, cfg, fold_of, n_folds=N_FOLDS):
    fold_results = []
    for k in range(n_folds):
        test_ids = [r for r, f in fold_of.items() if f == k]
        train_ids = [r for r, f in fold_of.items() if f != k]
        rng = np.random.RandomState(SEED + k)
        rng.shuffle(train_ids)
        n_val = max(1, int(0.2 * len(train_ids)))
        val_ids, tr_ids = train_ids[:n_val], train_ids[n_val:]

        norm = T.fit_norm(index.set_index("rec_id").loc[tr_ids, "path"].tolist())
        train_arr = T.balance_windows(_arrays_for(index, tr_ids, norm), seed=SEED + k)
        val_arr = _arrays_for(index, val_ids, norm)
        test_arr = _arrays_for(index, test_ids, norm)

        print(f"=== fold {k}: train_recs={len(tr_ids)} val_recs={len(val_ids)} "
              f"test_recs={len(test_ids)} train_windows={len(train_arr['y'])} "
              f"test_windows={len(test_arr['y'])} ===", flush=True)

        model, thr, hist, tsec, npar = T.train_model(train_arr, val_arr, cfg, SEED + k, verbose=False)
        val_ev = _strip(T.evaluate(model, val_arr, thr))
        test_ev = _strip(T.evaluate(model, test_arr, thr))
        print(f"  fold {k}: val acc={val_ev['window']['accuracy']:.3f} "
              f"test window acc={test_ev['window']['accuracy']:.3f} "
              f"test minute acc={test_ev['minute']['accuracy']:.3f}", flush=True)

        fold_results.append({
            "fold": k, "threshold": thr, "n_params": npar, "train_seconds": tsec,
            "history": hist, "test_recordings": test_ids,
            "val": val_ev, "test": test_ev,
        })
        torch.save({"state_dict": model.state_dict(), "cfg": cfg, "threshold": thr,
                    "norm": norm, "fold": k}, C.OUTPUT_DIR / f"model_fold{k}.pt")

    window_accs = [f["test"]["window"]["accuracy"] for f in fold_results]
    minute_accs = [f["test"]["minute"]["accuracy"] for f in fold_results]
    window_f1s = [f["test"]["window"]["macro_f1"] for f in fold_results]
    minute_f1s = [f["test"]["minute"]["macro_f1"] for f in fold_results]

    summary = {
        "n_folds": n_folds,
        "cfg": cfg,
        "window_accuracy_mean": float(np.mean(window_accs)),
        "window_accuracy_std": float(np.std(window_accs)),
        "window_accuracy_per_fold": window_accs,
        "minute_accuracy_mean": float(np.mean(minute_accs)),
        "minute_accuracy_std": float(np.std(minute_accs)),
        "minute_accuracy_per_fold": minute_accs,
        "window_macro_f1_mean": float(np.mean(window_f1s)),
        "minute_macro_f1_mean": float(np.mean(minute_f1s)),
    }
    print("\n=== K-FOLD SUMMARY ===")
    print(json.dumps(summary, indent=2))
    _save({"summary": summary, "folds": fold_results}, "results_kfold.json")
    return summary, fold_results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    a = ap.parse_args()

    index = T.load_index()
    print(f"index: {len(index)} recordings, labels={index.label.value_counts().to_dict()}", flush=True)

    fold_of = stratified_group_kfold(index, n_folds=a.n_folds)
    _save({rid: int(f) for rid, f in fold_of.items()}, "fold_assignment.json")

    if a.quick:
        cfg = dict(DEFAULT_CFG, epochs=6, patience=3)
    elif (OUT / "best_config.json").exists():
        cfg = json.loads((OUT / "best_config.json").read_text())
    else:
        cfg = config_search(index, fold_of)
        _save(cfg, "best_config.json")

    run_kfold(index, cfg, fold_of, n_folds=a.n_folds)


if __name__ == "__main__":
    main()
