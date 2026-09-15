"""Training + evaluation for the E4 hierarchical pipeline.

  * Occupancy stage: NOT retrained — E2's exported
    `outputs/best_model.pt` (radar-only, trained t1 / val t2) is loaded
    and evaluated on t.
  * Sleep/present stage (binary): trained on occupied t1+t2 recordings
    (`minutes/`), tested on occupied t recordings (`test_minutes/`).
    Uses E2's cached radar windows; the sleep/present label is recovered
    from each recording's manifest via `common.scan_all_labels` (E2's
    cache only stores the binary occupancy label).
  * Position stage (binary left/right): trained and evaluated on t only
    (the only source with left/right labels). The reported accuracy is
    the E3 stratified group 5-fold CV result; the model saved for
    `predict_capture` is additionally retrained on ALL t recordings
    (standard practice: CV for honest evaluation, full-data fit for
    deployment).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from models import (build_occupancy_model, build_sleep_present_model,
                    build_position_model, count_params)

SEED = 303
SP_CFG = {"lr": 1e-3, "embed_dim": 64, "dropout": 0.3, "weight_decay": 1e-4,
          "epochs": 40, "patience": 8, "batch_size": 256,
          "norm_mode": "local"}
POS_CFG = {"lr": 1e-3, "embed_dim": 32, "dropout": 0.2, "weight_decay": 1e-4,
           "epochs": 40, "patience": 8, "batch_size": 64}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Index + arrays
# ---------------------------------------------------------------------------
def load_index(label_map):
    """Build a recording index from the E2 and E3 caches, annotated with
    the 3-class activity and binary position labels from `label_map`
    (produced by `common.scan_all_labels`)."""
    import pandas as pd
    rows = []
    for f in sorted(C.E2_CACHE.glob("minutes/*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        rec_id = meta["rec_id"]
        info = label_map.get(rec_id, {})
        rows.append({"rec_id": rec_id, "path": str(f), "cache": "E2",
                     "placement": info.get("placement"),
                     "activity": info.get("activity"),
                     "position": info.get("position"),
                     "n_windows": int(d["radar"].shape[0])})
    for f in sorted(C.E2_CACHE.glob("test_minutes/*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        rec_id = meta["rec_id"]
        info = label_map.get(rec_id, {})
        rows.append({"rec_id": rec_id, "path": str(f), "cache": "E2",
                     "placement": info.get("placement"),
                     "activity": info.get("activity"),
                     "position": info.get("position"),
                     "n_windows": int(d["radar"].shape[0])})
    for f in sorted(C.E3_CACHE.glob("*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        rec_id = meta["rec_id"]
        info = label_map.get(rec_id, {})
        rows.append({"rec_id": rec_id, "path": str(f), "cache": "E3",
                     "placement": info.get("placement"),
                     "activity": info.get("activity"),
                     "position": info.get("position"),
                     "n_windows": int(d["radar"].shape[0])})
    return pd.DataFrame(rows)


def fit_norm(rec_paths):
    rs = rss = rn = None
    for p in rec_paths:
        d = np.load(p)
        r = d["radar"].astype(np.float64)
        s = r.sum(axis=(0, 1, 3, 4))
        ss = (r ** 2).sum(axis=(0, 1, 3, 4))
        n = r.shape[0] * r.shape[1] * r.shape[3] * r.shape[4]
        rs = s if rs is None else rs + s
        rss = ss if rss is None else rss + ss
        rn = n if rn is None else rn + n
    rmean = rs / rn
    rstd = np.sqrt(np.maximum(rss / rn - rmean ** 2, 0)) + 1e-6
    return {"radar_mean": rmean.tolist(), "radar_std": rstd.tolist()}


def _norm_block(radar, norm, norm_mode="global"):
    r = radar.astype(np.float32)
    if norm_mode == "local":
        # per-recording z-score: removes inter-recording gain/offset
        # differences (placement, hardware) at the cost of absolute-level
        # signal — same option E2 exposes via --norm-mode local
        rm = r.mean(axis=(0, 1, 3, 4), keepdims=True)
        rs = r.std(axis=(0, 1, 3, 4), keepdims=True) + 1e-6
    else:
        rm = np.asarray(norm["radar_mean"], np.float32)[None, None, :, None, None]
        rs = np.asarray(norm["radar_std"], np.float32)[None, None, :, None, None]
    return ((r - rm) / rs).astype(np.float16)


def load_arrays(rec_paths, label_key, norm=None, norm_mode="global"):
    """Load cached windows; `label_key` is 'activity' or 'position'."""
    import pandas as pd
    blocks, total = [], 0
    for p in rec_paths:
        d = np.load(p)
        n = d["radar"].shape[0]
        blocks.append((p, d, n))
        total += n
    if total == 0:
        raise ValueError("no windows")
    r0 = blocks[0][1]["radar"]
    radar = np.empty((total,) + r0.shape[1:], dtype=np.float16)
    y = np.empty(total, dtype=np.int64)
    rec_idx = np.empty(total, dtype=np.int32)
    off = 0
    for i, (p, d, n) in enumerate(blocks):
        meta = json.loads(str(d["meta"]))
        if norm is not None or norm_mode == "local":
            r = _norm_block(d["radar"], norm, norm_mode)
        else:
            r = d["radar"]
        radar[off:off + n] = r
        # label comes from the manifest-derived map, not the cache meta
        # (cache meta only has the binary occupancy label)
        rec_id = meta["rec_id"]
        y[off:off + n] = _LABEL_LOOKUP[rec_id][label_key]
        rec_idx[off:off + n] = i
        off += n
    return {"radar": radar, "y": y, "rec_idx": rec_idx}


_LABEL_LOOKUP = {}  # rec_id -> {"activity": int|None, "position": int|None}


def set_label_lookup(label_map):
    global _LABEL_LOOKUP
    _LABEL_LOOKUP = label_map


def make_loader(arrays, batch_size, shuffle, seed=0):
    idx = np.arange(len(arrays["y"]))
    ds = TensorDataset(
        torch.from_numpy(arrays["radar"][idx]),
        torch.from_numpy(arrays["y"][idx]),
        torch.from_numpy(idx))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def balance_windows(arrays, seed=0):
    rng = np.random.RandomState(seed)
    y = arrays["y"]
    classes = np.unique(y)
    idxs = [np.where(y == c)[0] for c in classes]
    n = min(len(i) for i in idxs)
    keep = np.sort(np.concatenate([rng.choice(i, n, replace=False) for i in idxs]))
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(y) else v)
            for k, v in arrays.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, n_classes):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def per_minute_metrics(y_true, y_pred, rec_idx, n_classes):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); rec_idx = np.asarray(rec_idx)
    yt, yp = [], []
    for r in np.unique(rec_idx):
        m = rec_idx == r
        yt.append(int(np.round(y_true[m].mean())))
        vals, cnts = np.unique(y_pred[m], return_counts=True)
        yp.append(int(vals[np.argmax(cnts)]))
    return compute_metrics(yt, yp, n_classes)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def predict_proba(model, loader, device, n_classes):
    model.eval()
    probs, ys, idxs = [], [], []
    with torch.no_grad():
        for r, y, i in loader:
            r = r.float().to(device)
            out = model(r)
            if n_classes == 2:
                p = torch.stack([1 - torch.sigmoid(out), torch.sigmoid(out)], dim=1)
            else:
                p = torch.softmax(out, dim=1)
            probs.append(p.cpu().numpy())
            ys.append(y.numpy()); idxs.append(i.numpy())
    order = np.argsort(np.concatenate(idxs))
    return np.concatenate(ys)[order], np.concatenate(probs)[order]


def train_model(model, train_arrays, val_arrays, cfg, seed, n_classes,
                device="cpu", verbose=False):
    set_seed(seed)
    model = model.to(device)
    n_params = count_params(model)
    train_loader = make_loader(train_arrays, cfg["batch_size"], True, seed)

    val_sub = val_arrays
    yv = val_sub["y"]
    classes = np.unique(yv)
    idxs = [np.where(yv == c)[0] for c in classes]
    n = min(len(i) for i in idxs)
    if n > 0:
        rng = np.random.RandomState(seed)
        keep = np.sort(np.concatenate([rng.choice(i, n, replace=False) for i in idxs]))
        val_sub = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(yv) else v)
                   for k, v in val_sub.items()}
    val_loader = make_loader(val_sub, cfg["batch_size"], False)

    if n_classes == 2:
        pos = float(train_arrays["y"].sum()); neg = float(len(train_arrays["y"]) - pos)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1)))
    else:
        cnts = np.bincount(train_arrays["y"], minlength=n_classes).astype(np.float32)
        w = torch.tensor(cnts.sum() / np.maximum(cnts, 1))
        criterion = nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg.get("weight_decay", 1e-4))

    best = {"f1": -1.0, "state": None, "epoch": 0}
    history, bad = [], 0
    t0 = time.time()
    loss = torch.tensor(0.0)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for r, y, _ in train_loader:
            r = r.float().to(device)
            if n_classes == 2:
                # label-preserving augmentations only (see E3/train.py)
                if torch.rand(1).item() < 0.5:
                    shift = int(torch.randint(-2, 3, (1,)).item())
                    if shift != 0:
                        r = torch.roll(r, shifts=shift, dims=-1)
                if torch.rand(1).item() < 0.5:
                    r = torch.flip(r, dims=[1])
            else:
                # E2-style augmentation: doppler-sign flip (valid for
                # activity, which is label-preserving under this flip)
                flip = torch.rand(r.shape[0]) < 0.5
                r[flip] = torch.flip(r[flip], dims=[-2])
            y = y.to(device)
            out = model(r)
            loss = criterion(out, y.float() if n_classes == 2 else y)
            opt.zero_grad(); loss.backward(); opt.step()
        vy, vp = predict_proba(model, val_loader, device, n_classes)
        vm = compute_metrics(vy, vp.argmax(1), n_classes)
        history.append({"epoch": epoch, "val_macro_f1": vm["macro_f1"],
                        "val_acc": vm["accuracy"], "train_loss": float(loss.item())})
        if verbose:
            print(f"  ep{epoch}: val f1={vm['macro_f1']:.3f} acc={vm['accuracy']:.3f}", flush=True)
        if vm["macro_f1"] > best["f1"]:
            best = {"f1": vm["macro_f1"], "epoch": epoch,
                    "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    train_seconds = time.time() - t0
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, history, train_seconds, n_params


def evaluate(model, arrays, n_classes, device="cpu", batch_size=256):
    loader = make_loader(arrays, batch_size, False)
    y, p = predict_proba(model, loader, device, n_classes)
    pred = p.argmax(1)
    rec_idx = arrays["rec_idx"]
    return {"window": compute_metrics(y, pred, n_classes),
            "minute": per_minute_metrics(y, pred, rec_idx, n_classes),
            "y": y.tolist(), "p": p.tolist(), "rec_idx": rec_idx.tolist()}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _strip(res):
    res.pop("y", None); res.pop("p", None); res.pop("rec_idx", None)
    return res


def _save(obj, name):
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.OUTPUT_DIR / name).write_text(json.dumps(obj, indent=2, default=str))
    print(f"saved {name}", flush=True)


def run_occupancy(index):
    """Evaluate E2's exported occupancy model (best_model.pt) on t.
    The model is NOT retrained — it was trained on t1 / validated on t2
    by the E2 pipeline."""
    ckpt_path = C.ROOT / "E2" / "outputs" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_occupancy_model(ckpt["cfg"]["embed_dim"], ckpt["cfg"]["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    thr = ckpt["threshold"]
    norm = ckpt["norm"]

    e2 = index[index.cache == "E2"]
    test_ids = e2[(e2.placement == "t") & e2.activity.notna()]["rec_id"].tolist()
    sub = e2.set_index("rec_id")
    test_arr = load_arrays(sub.loc[test_ids, "path"].tolist(), "activity", norm)
    # binarize: empty=0, sleep/present=1
    test_arr["y"] = (test_arr["y"] > 0).astype(np.int64)

    loader = make_loader(test_arr, 256, False)
    y, p = predict_proba(model, loader, "cpu", 2)
    pred = (p[:, 1] >= thr).astype(int)
    win = compute_metrics(y, pred, 2)
    minute = per_minute_metrics(y, pred, test_arr["rec_idx"], 2)
    print(f"  occupancy (E2 export, thr={thr:.2f}): test window acc={win['accuracy']:.3f} "
          f"minute acc={minute['accuracy']:.3f}", flush=True)
    return {"test": {"window": win, "minute": minute}, "threshold": thr,
            "source": "E2/outputs/best_model.pt (trained t1, val t2)"}


def run_sleep_present(index):
    """Train binary sleep/present model on occupied t1+t2, test on
    occupied t."""
    e2 = index[index.cache == "E2"]
    train_ids = e2[e2.placement.isin(["t1", "t2"]) & (e2.activity > 0)]["rec_id"].tolist()
    test_ids = e2[(e2.placement == "t") & (e2.activity > 0)]["rec_id"].tolist()
    print(f"sleep/present: {len(train_ids)} train recs (occupied t1+t2), "
          f"{len(test_ids)} test recs (occupied t)", flush=True)

    sub = e2.set_index("rec_id")
    norm = fit_norm(sub.loc[train_ids, "path"].tolist())
    rng = np.random.RandomState(SEED)
    tr_ids = train_ids.copy(); rng.shuffle(tr_ids)
    n_val = max(1, int(0.15 * len(tr_ids)))
    val_ids, tr_ids = tr_ids[:n_val], tr_ids[n_val:]

    def _sp_arrays(ids):
        arr = load_arrays(sub.loc[ids, "path"].tolist(), "activity", norm,
                          norm_mode=SP_CFG.get("norm_mode", "global"))
        arr["y"] = (arr["y"] == 2).astype(np.int64)  # sleep=0, present=1
        return arr

    train_arr = balance_windows(_sp_arrays(tr_ids), SEED)
    val_arr = _sp_arrays(val_ids)
    test_arr = _sp_arrays(test_ids)

    model = build_sleep_present_model(SP_CFG["embed_dim"], SP_CFG["dropout"])
    model, hist, tsec, npar = train_model(model, train_arr, val_arr, SP_CFG, SEED, n_classes=2, verbose=True)
    val_ev = _strip(evaluate(model, val_arr, 2))
    test_ev = _strip(evaluate(model, test_arr, 2))
    print(f"  sleep/present: val acc={val_ev['window']['accuracy']:.3f} "
          f"test window acc={test_ev['window']['accuracy']:.3f} "
          f"test minute acc={test_ev['minute']['accuracy']:.3f}", flush=True)

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": SP_CFG, "norm": norm,
                "norm_mode": SP_CFG.get("norm_mode", "global"),
                "n_classes": 2, "window_frames": C.E2_WINDOW_FRAMES,
                "map_size": C.E2_MAP_SIZE, "az_bins": C.E2_AZ_BINS,
                "range_bins": C.E2_RANGE_BINS},
               C.MODELS_DIR / "sleep_present_model.pt")
    return {"val": val_ev, "test": test_ev, "cfg": SP_CFG, "n_params": npar,
            "train_seconds": tsec, "history": hist, "norm": norm}


def run_position(index):
    """Train binary left/right model on all t recordings (deployment model).
    Reported accuracy is the E3 5-fold CV result (loaded from
    E3/outputs/results_kfold.json)."""
    e3 = index[index.cache == "E3"]
    t_ids = e3[e3.position.notna()]["rec_id"].tolist()
    print(f"position: {len(t_ids)} t recs for deployment training", flush=True)

    sub = e3.set_index("rec_id")
    norm = fit_norm(sub.loc[t_ids, "path"].tolist())
    train_arr = balance_windows(load_arrays(sub.loc[t_ids, "path"].tolist(), "position", norm), SEED)
    # small internal val split for early stopping
    rng = np.random.RandomState(SEED)
    tr_ids = t_ids.copy(); rng.shuffle(tr_ids)
    n_val = max(1, int(0.15 * len(tr_ids)))
    val_ids, tr_ids = tr_ids[:n_val], tr_ids[n_val:]
    train_arr = balance_windows(load_arrays(sub.loc[tr_ids, "path"].tolist(), "position", norm), SEED)
    val_arr = load_arrays(sub.loc[val_ids, "path"].tolist(), "position", norm)

    model = build_position_model(POS_CFG["embed_dim"], POS_CFG["dropout"])
    model, hist, tsec, npar = train_model(model, train_arr, val_arr, POS_CFG, SEED, n_classes=2, verbose=True)
    val_ev = _strip(evaluate(model, val_arr, 2))
    print(f"  position: val acc={val_ev['window']['accuracy']:.3f}", flush=True)

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": POS_CFG, "norm": norm,
                "n_classes": 2, "window_frames": C.E3_WINDOW_FRAMES,
                "map_size": C.E3_MAP_SIZE, "az_bins": C.E3_AZ_BINS,
                "range_bins": C.E3_RANGE_BINS},
               C.MODELS_DIR / "position_model.pt")

    cv = {}
    e3_results = C.ROOT / "E3" / "outputs" / "results_kfold.json"
    if e3_results.exists():
        cv = json.loads(e3_results.read_text())["summary"]
    return {"val": val_ev, "cv_summary": cv, "cfg": POS_CFG, "n_params": npar,
            "train_seconds": tsec, "history": hist, "norm": norm}


def main():
    label_map = C.scan_all_labels()
    set_label_lookup(label_map)
    index = load_index(label_map)
    print(f"index: {len(index)} cached recordings "
          f"(E2={len(index[index.cache=='E2'])}, E3={len(index[index.cache=='E3'])})", flush=True)

    results = {}
    results["occupancy"] = run_occupancy(index)
    results["sleep_present"] = run_sleep_present(index)
    results["position"] = run_position(index)
    _save(results, "results_e4.json")


if __name__ == "__main__":
    main()
