"""Experiments for the E2 occupancy pipeline (4-folder design).

Splits are the dataset folders:
  train = train_minutes/ (placements t1+t2) + train2_minutes/
      (placement t, Sept 6-8)
  val   = validation_minutes/ (placement t, Sept 15) — early stopping,
      threshold selection, hyperparameter search
  test  = test_minutes/ (Pi captures, Sept 16-17 night, 1:10 AM
      ground-truth boundary) — touched once for final metrics

The training set is class-balanced by subsampling majority-class
windows (empty/occupied). Windows are built per recording, so they
never cross recording / placement / label boundaries. Normalization
uses train statistics only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import train as T

SEED = 101
OUT = C.OUTPUT_DIR
CKPT = OUT / "best_model.pt"

DEFAULT_CFG = {
    "lr": 1e-3, "embed_dim": 64, "dropout": 0.3, "weight_decay": 1e-4,
    "epochs": 40, "patience": 8, "batch_size": 256,
}


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------
def _paths(index, split):
    return list(index[index.split == split].sort_values("rec_id")
                ["path"])


def load_e2_splits(index, norm_mode="global"):
    """train=train_minutes+train2_minutes, val=validation_minutes,
    test=test_minutes; norm stats from the train pool only (global)."""
    train_paths = _paths(index, "train_minutes") + \
        _paths(index, "train2_minutes")
    norm = T.fit_norm(train_paths) if norm_mode == "global" else None
    return {
        "norm": norm, "norm_mode": norm_mode,
        "train": T.load_split_arrays(train_paths, norm, norm_mode),
        "val": T.load_split_arrays(_paths(index, "validation_minutes"),
                                   norm, norm_mode),
        "test": T.load_split_arrays(_paths(index, "test_minutes"),
                                    norm, norm_mode),
    }


def balance_windows(arrays, seed=SEED):
    """Subsample majority-class windows so classes are equal."""
    rng = np.random.RandomState(seed)
    y = arrays["y"]
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    keep = np.sort(np.concatenate([
        rng.choice(idx0, n, replace=False),
        rng.choice(idx1, n, replace=False)]))
    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(y)
               else v) for k, v in arrays.items()}
    print(f"  balanced train: {len(y)} -> {len(keep)} windows "
          f"({n}/class)", flush=True)
    return out


def _strip(res):
    res.pop("y", None); res.pop("p", None); res.pop("rec_idx", None)
    return res


def _save(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str))
    print(f"saved {name}", flush=True)


# ---------------------------------------------------------------------------
# E2: train t1 / val t2 / test t
# ---------------------------------------------------------------------------
def _subsample(arrays, n, seed=SEED):
    if len(arrays["y"]) <= n:
        return arrays
    rng = np.random.RandomState(seed)
    keep = np.sort(rng.choice(len(arrays["y"]), n, replace=False))
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(arrays["y"])
                else v) for k, v in arrays.items()}


def config_search(splits, modality="radar", quick_epochs=8,
                  max_train=12000):
    """Small improvement loop on train->val accuracy/F1.
    Runs on a subsample of the balanced train set to bound CPU time."""
    train_sub = _subsample(splits["train"], max_train)
    grid = []
    for lr in (1e-3, 3e-4):
        for emb in (64, 128):
            for dr in (0.2, 0.4):
                grid.append({"lr": lr, "embed_dim": emb, "dropout": dr,
                             "weight_decay": 1e-4, "epochs": quick_epochs,
                             "patience": 4, "batch_size": 256})
    results = []
    for cfg in grid:
        model, thr, mthr, hist, tsec, npar = T.train_model(
            modality, train_sub, splits["val"], cfg, SEED)
        ev = _strip(T.evaluate(model, splits["val"], modality, thr, mthr))
        results.append({"cfg": cfg, "threshold": thr,
                        "val_window": ev["window"], "train_seconds": tsec})
        print(f"  cfg lr={cfg['lr']} emb={cfg['embed_dim']} "
              f"drop={cfg['dropout']}: val acc={ev['window']['accuracy']:.3f} "
              f"f1={ev['window']['macro_f1']:.3f}", flush=True)
    best = max(results, key=lambda r: (r["val_window"]["accuracy"],
                                       r["val_window"]["macro_f1"]))
    _save({"results": results, "best": best}, "hp_search.json")
    cfg = dict(best["cfg"])
    cfg["epochs"] = DEFAULT_CFG["epochs"]
    cfg["patience"] = DEFAULT_CFG["patience"]
    return cfg


def run_e2(index, cfg=None, modalities=("fusion", "radar", "csi"),
           norm_mode="global"):
    splits = load_e2_splits(index, norm_mode)
    splits["train"] = balance_windows(splits["train"])
    if cfg is None:
        cfg = config_search(splits)
        _save(cfg, "best_config.json")
    results = {}
    best = {"val_acc": -1.0, "mod": None, "model": None, "thr": None}
    for mod in modalities:
        print(f"=== E2 {mod} ===", flush=True)
        tr_arr, val_arr = splits["train"], splits["val"]
        model, thr, mthr, hist, tsec, npar = T.train_model(
            mod, tr_arr, val_arr, cfg, SEED, verbose=True)
        res = {"modality": mod, "cfg": cfg, "seed": SEED, "threshold": thr,
               "minute_threshold": mthr,
               "n_params": npar, "train_seconds": tsec, "history": hist,
               "val_split": "validation_minutes"}
        res["val"] = _strip(T.evaluate(model, val_arr, mod, thr, mthr))
        res["test"] = _strip(T.evaluate(model, splits["test"], mod, thr,
                                        mthr))
        results[mod] = res
        print(f"  {mod}: val acc={res['val']['window']['accuracy']:.3f} "
              f"test acc={res['test']['window']['accuracy']:.3f} "
              f"f1={res['test']['window']['macro_f1']:.3f}", flush=True)
        torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                    "threshold": thr, "minute_threshold": mthr,
                    "norm": splits["norm"],
                    "norm_mode": splits["norm_mode"], "modality": mod},
                   C.OUTPUT_DIR / f"model_{mod}.pt")
        vacc = res["val"]["window"]["accuracy"]
        # the exported checkpoint is the radar-only architecture
        if mod == "radar" and vacc > best["val_acc"]:
            best = {"val_acc": vacc, "mod": mod, "model": model,
                    "thr": thr, "mthr": mthr}
    if best["model"] is not None:
        torch.save({"state_dict": best["model"].state_dict(), "cfg": cfg,
                    "threshold": best["thr"],
                    "minute_threshold": best["mthr"],
                    "norm": splits["norm"],
                    "norm_mode": splits["norm_mode"],
                    "modality": best["mod"]}, CKPT)
        print(f"saved {CKPT} (modality={best['mod']})", flush=True)
    _save(results, "results_e2.json")
    return results, splits


# ---------------------------------------------------------------------------
def load_index():
    return T.load_index()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="e2")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--norm-mode", default="global",
                    choices=("global", "local"))
    a = ap.parse_args()
    index = load_index()
    print(f"index: {len(index)} recordings", flush=True)
    cfg = None
    if a.quick:
        cfg = dict(DEFAULT_CFG, epochs=8, patience=3)
    elif (OUT / "best_config.json").exists():
        cfg = json.loads((OUT / "best_config.json").read_text())
    if "e2" in a.steps:
        run_e2(index, cfg, norm_mode=a.norm_mode)


if __name__ == "__main__":
    main()
