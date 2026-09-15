"""Experiments for the E2 occupancy pipeline (revised design).

Splits are by placement, no sessions:
  E2: train = all t1 minutes, val = all t2 minutes, test = all t minutes.
      The training set is class-balanced by subsampling majority-class
      windows (empty/occupied). A small config search on t1->t2 selects
      the best hyperparameters; the best model is saved to
      outputs/best_model.pt.
  E3: few-shot adaptation. The E2 model is fine-tuned with 10 support
      minutes from t (5 per class) and evaluated on the remaining query
      minutes; compared against the 0-shot baseline. Query minutes are
      never used for training or selection.

Windows are built per recording, so they never cross recording /
placement / label boundaries. Normalization uses t1 statistics only.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
def _paths(index, placement):
    return list(index[index.placement == placement].sort_values("rec_id")
                ["path"])


def load_e2_splits(index, norm_mode="global"):
    """train=t1, val=t2, test=t; norm stats from t1 only (global mode)."""
    norm = T.fit_norm(_paths(index, "t1")) if norm_mode == "global" else None
    return {
        "norm": norm, "norm_mode": norm_mode,
        "train": T.load_split_arrays(_paths(index, "t1"), norm, norm_mode),
        "val": T.load_split_arrays(_paths(index, "t2"), norm, norm_mode),
        "test": T.load_split_arrays(_paths(index, "t"), norm, norm_mode),
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
    """Small improvement loop on t1->t2 validation accuracy/F1.
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
        model, thr, hist, tsec, npar = T.train_model(
            modality, train_sub, splits["val"], cfg, SEED)
        ev = _strip(T.evaluate(model, splits["val"], modality, thr))
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


def _drop_recs(arrays, rec_ids):
    keep = ~np.isin(arrays["rec_idx"], list(rec_ids))
    return {k: (v[keep] if isinstance(v, np.ndarray)
                and len(v) == len(arrays["y"]) else v)
            for k, v in arrays.items()}


def run_e2(index, cfg=None, modalities=("fusion", "radar", "csi"),
           norm_mode="global"):
    splits = load_e2_splits(index, norm_mode)
    splits["train"] = balance_windows(splits["train"])
    if cfg is None:
        cfg = config_search(splits)
        _save(cfg, "best_config.json")
    # t2 has no CSI coverage at all, so the csi-only baseline cannot
    # early-stop on t2 — hold out the last 10% of t1 recordings that
    # actually have valid CSI windows
    t1_paths = _paths(index, "t1")
    t1_idx = index[index.placement == "t1"].sort_values("rec_id")
    csi_ok = set(t1_idx[t1_idx.n_csi_valid > 0]["path"])
    n_hold = max(1, int(0.1 * len(csi_ok)))
    hold_paths = list(t1_idx[t1_idx.n_csi_valid > 0]["path"])[-n_hold:]
    hold_ids = {i for i, p in enumerate(t1_paths) if p in set(hold_paths)}
    csi_train = balance_windows(_drop_recs(
        T.load_split_arrays(t1_paths, splits["norm"], norm_mode), hold_ids))
    csi_val = T.load_split_arrays(hold_paths, splits["norm"], norm_mode)
    results = {}
    best = {"val_acc": -1.0, "mod": None, "model": None, "thr": None}
    for mod in modalities:
        print(f"=== E2 {mod} ===", flush=True)
        if mod == "csi":
            tr_arr, val_arr = csi_train, csi_val
        else:
            tr_arr, val_arr = splits["train"], splits["val"]
        model, thr, hist, tsec, npar = T.train_model(
            mod, tr_arr, val_arr, cfg, SEED, verbose=True)
        res = {"modality": mod, "cfg": cfg, "seed": SEED, "threshold": thr,
               "n_params": npar, "train_seconds": tsec, "history": hist,
               "val_split": "t1_holdout" if mod == "csi" else "t2"}
        res["val"] = _strip(T.evaluate(model, val_arr, mod, thr))
        res["test"] = _strip(T.evaluate(model, splits["test"], mod, thr))
        results[mod] = res
        print(f"  {mod}: val acc={res['val']['window']['accuracy']:.3f} "
              f"test acc={res['test']['window']['accuracy']:.3f} "
              f"f1={res['test']['window']['macro_f1']:.3f}", flush=True)
        torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                    "threshold": thr, "norm": splits["norm"],
                    "norm_mode": splits["norm_mode"], "modality": mod},
                   C.OUTPUT_DIR / f"model_{mod}.pt")
        vacc = res["val"]["window"]["accuracy"]
        if mod != "csi" and vacc > best["val_acc"]:
            best = {"val_acc": vacc, "mod": mod, "model": model, "thr": thr}
    if best["model"] is not None:
        torch.save({"state_dict": best["model"].state_dict(), "cfg": cfg,
                    "threshold": best["thr"], "norm": splits["norm"],
                    "norm_mode": splits["norm_mode"],
                    "modality": best["mod"]}, CKPT)
        print(f"saved {CKPT} (modality={best['mod']})", flush=True)
    _save(results, "results_e2.json")
    return results, splits


# ---------------------------------------------------------------------------
# E3: few-shot adaptation to placement t
# ---------------------------------------------------------------------------
def _finetune(model, strategy, sup_arrays, cfg, seed=SEED, epochs=10):
    for p in model.parameters():
        p.requires_grad = False
    if strategy == "head":
        for p in model.head.parameters():
            p.requires_grad = True
        lr = cfg["lr"] * 0.1
    elif strategy == "fusion_head":
        for p in model.head.parameters():
            p.requires_grad = True
        for p in model.fusion.parameters():
            p.requires_grad = True
        lr = cfg["lr"] * 0.1
    else:  # full
        for p in model.parameters():
            p.requires_grad = True
        lr = cfg["lr"] * 0.02
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    pos = float(sup_arrays["y"].sum()); neg = len(sup_arrays["y"]) - pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1)))
    loader = T.make_loader(sup_arrays, min(32, len(sup_arrays["y"])), True, seed)
    model.train()
    for _ in range(epochs):
        for r, c, y, _ in loader:
            out = model(r.float(), c.float())
            loss = crit(out, y.float())
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_e3(index, n_support=10, strategies=None):
    """Few-shot: fine-tune the saved E2 model on n_support minutes
    of t (half per class), evaluate on the remaining query minutes."""
    from models import build_model
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]; thr = ckpt["threshold"]
    norm = ckpt["norm"]; norm_mode = ckpt.get("norm_mode", "global")
    modality = ckpt.get("modality", "fusion")
    if strategies is None:
        strategies = (["head", "fusion_head", "full"] if modality == "fusion"
                      else ["head", "full"])
    base = build_model(modality, cfg["embed_dim"], cfg["dropout"])
    base.load_state_dict(ckpt["state_dict"])

    t_index = index[index.placement == "t"].sort_values("rec_id")
    rng = np.random.RandomState(SEED)
    support = []
    for label in (0, 1):
        ids = t_index[t_index.label == label]["rec_id"].tolist()
        rng.shuffle(ids)
        support.extend(ids[: n_support // 2])
    query = [r for r in t_index["rec_id"] if r not in support]
    _save({"support": support, "query": query, "n_support": n_support},
          "e3_partition.json")

    sub = t_index.set_index("rec_id")
    sup = T.load_split_arrays([sub.loc[r, "path"] for r in support],
                              norm, norm_mode)
    qry = T.load_split_arrays([sub.loc[r, "path"] for r in query],
                              norm, norm_mode)

    results = {"n_support": n_support, "support_minutes": support,
               "modality": modality}
    ev = _strip(T.evaluate(base, qry, modality, thr))
    results["zero_shot"] = {"window": ev["window"], "minute": ev["minute"]}
    print(f"  E3 0-shot: acc={ev['window']['accuracy']:.3f} "
          f"f1={ev['window']['macro_f1']:.3f}", flush=True)
    for strat in strategies:
        m = _finetune(copy.deepcopy(base), strat, sup, cfg)
        ev = _strip(T.evaluate(m, qry, modality, thr))
        results[strat] = {"window": ev["window"], "minute": ev["minute"]}
        print(f"  E3 {strat}: acc={ev['window']['accuracy']:.3f} "
              f"f1={ev['window']['macro_f1']:.3f}", flush=True)
        torch.save({"state_dict": m.state_dict(), "cfg": cfg,
                    "threshold": thr, "norm": norm,
                    "norm_mode": norm_mode, "modality": modality,
                    "e3_strategy": strat, "support_minutes": support},
                   C.OUTPUT_DIR / f"model_{modality}_e3_{strat}.pt")
    _save(results, "results_e3.json")
    return results


# ---------------------------------------------------------------------------
def load_index():
    return T.load_index()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="e2,e3")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-support", type=int, default=10)
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
    if "e3" in a.steps:
        run_e3(index, n_support=a.n_support)


if __name__ == "__main__":
    main()
