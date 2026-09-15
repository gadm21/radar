"""Dataset assembly, normalization, training and evaluation for E3.

Only one placement/room exists in this dataset for the left/right task
(test_minutes), so there is no cross-placement split to exploit. Instead
we use STRATIFIED GROUP K-FOLD cross-validation at the recording (minute)
level: windows from one minute always stay together in the same fold, so
no fold ever leaks information about a specific minute's radar signature
between train and validation/test.
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
from models import build_model, count_params


# ---------------------------------------------------------------------------
# Recording index + arrays
# ---------------------------------------------------------------------------
def load_index():
    import pandas as pd
    rows = []
    for f in sorted(C.CACHE_DIR.glob("*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        meta["path"] = str(f)
        meta["n_windows"] = int(d["radar"].shape[0])
        rows.append(meta)
    return pd.DataFrame(rows)


def fit_norm(rec_paths):
    """Radar-map normalization statistics from the given recordings only
    (must be the training recordings of the current fold)."""
    rs = rss = rn = None
    for p in rec_paths:
        d = np.load(p)
        r = d["radar"].astype(np.float64)  # (n,T,2,S,S)
        s = r.sum(axis=(0, 1, 3, 4))
        ss = (r ** 2).sum(axis=(0, 1, 3, 4))
        n = r.shape[0] * r.shape[1] * r.shape[3] * r.shape[4]
        rs = s if rs is None else rs + s
        rss = ss if rss is None else rss + ss
        rn = n if rn is None else rn + n
    rmean = rs / rn
    rstd = np.sqrt(np.maximum(rss / rn - rmean ** 2, 0)) + 1e-6
    return {"radar_mean": rmean.tolist(), "radar_std": rstd.tolist()}


def _norm_block(radar, norm):
    r = radar.astype(np.float32)
    rm = np.asarray(norm["radar_mean"], np.float32)[None, None, :, None, None]
    rs = np.asarray(norm["radar_std"], np.float32)[None, None, :, None, None]
    r = (r - rm) / rs
    return r.astype(np.float16)


def load_split_arrays(rec_paths, norm=None):
    blocks = []
    total = 0
    for p in rec_paths:
        d = np.load(p)
        n = d["radar"].shape[0]
        blocks.append((p, d, n))
        total += n
    if total == 0:
        raise ValueError("no windows in split")
    r0 = blocks[0][1]["radar"]
    radar = np.empty((total,) + r0.shape[1:], dtype=np.float16)
    y = np.empty(total, dtype=np.int64)
    rec_idx = np.empty(total, dtype=np.int32)
    dur = np.empty(total, dtype=np.float32)
    off = 0
    for i, (p, d, n) in enumerate(blocks):
        meta = json.loads(str(d["meta"]))
        r = _norm_block(d["radar"], norm) if norm is not None else d["radar"]
        radar[off:off + n] = r
        y[off:off + n] = meta["label"]
        rec_idx[off:off + n] = i
        dur[off:off + n] = d["win_dur"]
        off += n
    return {"radar": radar, "y": y, "rec_idx": rec_idx, "win_dur": dur,
            "rec_paths": list(rec_paths)}


def make_loader(arrays, batch_size, shuffle, seed=0):
    idx = np.arange(len(arrays["y"]))
    ds = TensorDataset(
        torch.from_numpy(arrays["radar"][idx]),
        torch.from_numpy(arrays["y"][idx]),
        torch.from_numpy(idx),
    )
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=False)


def balance_windows(arrays, seed=0):
    rng = np.random.RandomState(seed)
    y = arrays["y"]
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    if n == 0:
        return arrays
    keep = np.sort(np.concatenate([
        rng.choice(idx0, n, replace=False),
        rng.choice(idx1, n, replace=False)]))
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(y) else v)
            for k, v in arrays.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, prob, threshold=0.5):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    y_true = np.asarray(y_true, dtype=int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float((f1[0] + f1[1]) / 2),
        "left_precision": float(pr[0]), "left_recall": float(rc[0]), "left_f1": float(f1[0]),
        "right_precision": float(pr[1]), "right_recall": float(rc[1]), "right_f1": float(f1[1]),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def per_minute_metrics(y_true, prob, rec_idx, threshold=0.5):
    y_true = np.asarray(y_true); prob = np.asarray(prob); rec_idx = np.asarray(rec_idx)
    yt, yp = [], []
    for r in np.unique(rec_idx):
        m = rec_idx == r
        yt.append(int(np.round(y_true[m].mean())))
        yp.append(float(prob[m].mean()))
    return compute_metrics(yt, yp, threshold)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, loader, device):
    model.eval()
    probs, ys, idxs = [], [], []
    with torch.no_grad():
        for r, y, i in loader:
            r = r.float().to(device)
            out = model(r)
            probs.append(torch.sigmoid(out).cpu().numpy())
            ys.append(y.numpy()); idxs.append(i.numpy())
    if not idxs:
        return np.array([], dtype=int), np.array([])
    order = np.argsort(np.concatenate(idxs))
    p = np.concatenate(probs)[order]
    y = np.concatenate(ys)[order]
    return y, p


def train_model(train_arrays, val_arrays, cfg, seed, device="cpu", verbose=False):
    """Train one model; early stopping + threshold selection on validation
    only (never on the held-out test fold)."""
    set_seed(seed)
    model = build_model(cfg["embed_dim"], cfg["dropout"]).to(device)
    n_params = count_params(model)
    train_loader = make_loader(train_arrays, cfg["batch_size"], True, seed)

    val_sub = val_arrays
    yv = val_sub["y"]
    i0, i1 = np.where(yv == 0)[0], np.where(yv == 1)[0]
    if len(i0) and len(i1):
        rng = np.random.RandomState(seed)
        n = min(len(i0), len(i1))
        keep = np.sort(np.concatenate([rng.choice(i0, n, replace=False),
                                       rng.choice(i1, n, replace=False)]))
        val_sub = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(yv) else v)
                   for k, v in val_sub.items()}
    val_loader = make_loader(val_sub, cfg["batch_size"], False)

    pos = float(train_arrays["y"].sum())
    neg = float(len(train_arrays["y"]) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1)))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))

    best = {"f1": -1.0, "state": None, "epoch": 0}
    history = []
    bad = 0
    t0 = time.time()
    loss = torch.tensor(0.0)
    augment = cfg.get("augment", True)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for r, y, _ in train_loader:
            r = r.float().to(device)
            if augment:
                # left-right mirror flip of the azimuth axis (last dim of
                # the maps) is NOT a valid augmentation here (it would
                # flip the very label we're predicting) -- so we only use
                # label-preserving augmentations: random small circular
                # shift along the range axis (sensor jitter / distance
                # tolerance) and random time-reversal (does not change
                # spatial content).
                if torch.rand(1).item() < 0.5:
                    shift = int(torch.randint(-2, 3, (1,)).item())
                    if shift != 0:
                        r = torch.roll(r, shifts=shift, dims=-1)
                if torch.rand(1).item() < 0.5:
                    r = torch.flip(r, dims=[1])  # reverse time order
            y = y.to(device).float()
            out = model(r)
            loss = criterion(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        vy, vp = predict(model, val_loader, device)
        vm = compute_metrics(vy, vp, 0.5)
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

    vy, vp = predict(model, val_loader, device)
    thr_grid = np.arange(0.30, 0.71, 0.02)
    scores = [compute_metrics(vy, vp, t)["macro_f1"] for t in thr_grid]
    best_thr = float(thr_grid[int(np.argmax(scores))])
    return model, best_thr, history, train_seconds, n_params


def evaluate(model, arrays, threshold, device="cpu", batch_size=256):
    loader = make_loader(arrays, batch_size, False)
    y, p = predict(model, loader, device)
    rec_idx = arrays["rec_idx"]
    win = compute_metrics(y, p, threshold)
    minute = per_minute_metrics(y, p, rec_idx, threshold)
    return {"window": win, "minute": minute, "y": y.tolist(), "p": [float(x) for x in p],
            "rec_idx": rec_idx.tolist()}
