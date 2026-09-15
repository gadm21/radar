"""Dataset assembly, normalization, training and evaluation for E2.

Memory model: cached windows are stored/loaded as float16 (radar is
log1p-compressed FFT maps; CSI is log1p amplitude). Normalization uses
TRAIN-split statistics only and is applied per recording block, storing
the result as float16; batches are cast to float32 inside the loop.
"""
import json
import os
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
    """Scan the cache and return a DataFrame of recordings."""
    import pandas as pd
    rows = []
    for source in ("minutes", "test_minutes"):
        for f in sorted((C.CACHE_DIR / source).glob("*.npz")):
            d = np.load(f)
            meta = json.loads(str(d["meta"]))
            meta["path"] = str(f)
            meta["n_windows"] = int(d["radar"].shape[0])
            meta["n_csi_valid"] = int(d["csi_valid"].sum())
            rows.append(meta)
    return pd.DataFrame(rows)


def _norm_block(radar, csi, valid, norm, norm_mode="global"):
    """Normalize one recording block; returns float16 radar/csi.

    norm_mode="global": train-split statistics (may carry placement gain).
    norm_mode="local":  per-recording z-score — removes inter-recording
                        gain/offset differences (placement, receiver
                        hardware) at the cost of absolute-level signal.
    """
    r = radar.astype(np.float32)
    c = csi.astype(np.float32)
    np.log1p(c, out=c)
    if norm_mode == "local":
        rm = r.mean(axis=(0, 1, 3, 4), keepdims=True)
        rs = r.std(axis=(0, 1, 3, 4), keepdims=True) + 1e-6
        r = (r - rm) / rs
        cm = c.reshape(-1, c.shape[-1]).mean(axis=0)[None, None, :]
        cs = c.reshape(-1, c.shape[-1]).std(axis=0)[None, None, :] + 1e-6
        c = (c - cm) / cs
    else:
        rm = np.asarray(norm["radar_mean"], np.float32)[None, None, :, None, None]
        rs = np.asarray(norm["radar_std"], np.float32)[None, None, :, None, None]
        r = (r - rm) / rs
        cm = np.asarray(norm["csi_mean"], np.float32)[None, None, :]
        cs = np.asarray(norm["csi_std"], np.float32)[None, None, :]
        c = (c - cm) / cs
    c[~valid] = 0.0
    return r.astype(np.float16), c.astype(np.float16)


def load_split_arrays(rec_paths, norm=None, norm_mode="global"):
    """Load cached windows for recordings; optionally normalize (train
    statistics) block-by-block. Arrays are float16 to bound RAM.
    Output arrays are preallocated (no concatenate peak)."""
    blocks = []
    total = 0
    for p in rec_paths:
        d = np.load(p)
        n = d["radar"].shape[0]
        blocks.append((p, d, n))
        total += n
    if total == 0:
        raise ValueError("no windows in split")
    r0, c0 = blocks[0][1]["radar"], blocks[0][1]["csi"]
    radar = np.empty((total,) + r0.shape[1:], dtype=np.float16)
    csi = np.empty((total,) + c0.shape[1:], dtype=np.float16)
    valid = np.empty(total, dtype=bool)
    y = np.empty(total, dtype=np.int64)
    rec_idx = np.empty(total, dtype=np.int32)
    dur = np.empty(total, dtype=np.float32)
    off = 0
    for i, (p, d, n) in enumerate(blocks):
        meta = json.loads(str(d["meta"]))
        v = d["csi_valid"].astype(bool)
        if norm is not None or norm_mode == "local":
            r, c = _norm_block(d["radar"], d["csi"], v, norm, norm_mode)
        else:
            r = d["radar"]
            c = d["csi"].astype(np.float32)
            np.log1p(c, out=c)
            c[~v] = 0.0
            c = c.astype(np.float16)
        radar[off:off + n] = r
        csi[off:off + n] = c
        valid[off:off + n] = v
        y[off:off + n] = meta["label"]
        rec_idx[off:off + n] = i
        dur[off:off + n] = d["win_dur"]
        off += n
    return {
        "radar": radar, "csi": csi, "csi_valid": valid, "y": y,
        "rec_idx": rec_idx, "win_dur": dur,
        "rec_paths": list(rec_paths),
    }


def fit_norm(rec_paths):
    """Normalization statistics from TRAIN split only, accumulated
    per recording file (no large concatenation).

    Radar cache is already log1p-compressed; CSI cache is raw amplitude,
    so stats are computed on log1p(csi) to match _norm_block.
    """
    rs = rss = rn = None
    cs = css = cn = None
    for p in rec_paths:
        d = np.load(p)
        r = d["radar"].astype(np.float32)          # (n,5,2,S,S)
        c = np.log1p(d["csi"].astype(np.float32))  # (n,T,52)
        s = r.sum(axis=(0, 1, 3, 4), dtype=np.float64)
        ss = (r.astype(np.float64) ** 2).sum(axis=(0, 1, 3, 4))
        n = r.shape[0] * r.shape[1] * r.shape[3] * r.shape[4]
        rs = s if rs is None else rs + s
        rss = ss if rss is None else rss + ss
        rn = n if rn is None else rn + n
        s = c.sum(axis=(0, 1), dtype=np.float64)
        ss = (c.astype(np.float64) ** 2).sum(axis=(0, 1))
        n = c.shape[0] * c.shape[1]
        cs = s if cs is None else cs + s
        css = ss if css is None else css + ss
        cn = n if cn is None else cn + n
    rmean = rs / rn
    rstd = np.sqrt(np.maximum(rss / rn - rmean ** 2, 0)) + 1e-6
    cmean = cs / cn
    cstd = np.sqrt(np.maximum(css / cn - cmean ** 2, 0)) + 1e-6
    return {"radar_mean": rmean.tolist(), "radar_std": rstd.tolist(),
            "csi_mean": cmean.tolist(), "csi_std": cstd.tolist()}


def make_loader(arrays, batch_size, shuffle, seed=0, csi_only_valid=False):
    idx = np.arange(len(arrays["y"]))
    if csi_only_valid:
        idx = idx[arrays["csi_valid"]]
    ds = TensorDataset(
        torch.from_numpy(arrays["radar"][idx]),
        torch.from_numpy(arrays["csi"][idx]),
        torch.from_numpy(arrays["y"][idx]),
        torch.from_numpy(idx),
    )
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=g, drop_last=False)


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
    occ = y_true == 1
    false_empty = float((pred[occ] == 0).mean()) if occ.any() else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float((f1[0] + f1[1]) / 2),
        "empty_precision": float(pr[0]), "empty_recall": float(rc[0]),
        "empty_f1": float(f1[0]),
        "occupied_precision": float(pr[1]), "occupied_recall": float(rc[1]),
        "occupied_f1": float(f1[1]),
        "false_empty_rate": false_empty,
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def per_minute_metrics(y_true, prob, rec_idx, threshold=0.5):
    """Aggregate window probabilities per recording (mean prob -> label)."""
    y_true = np.asarray(y_true); prob = np.asarray(prob)
    rec_idx = np.asarray(rec_idx)
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


def _forward(model, modality, r, c):
    if modality == "fusion":
        return model(r, c)
    if modality == "radar":
        return model(r)
    return model(c)


def predict(model, loader, device, modality):
    model.eval()
    probs, ys, idxs, gates = [], [], [], []
    with torch.no_grad():
        for r, c, y, i in loader:
            r = r.float().to(device); c = c.float().to(device)
            out = _forward(model, modality, r, c)
            probs.append(torch.sigmoid(out).cpu().numpy())
            ys.append(y.numpy()); idxs.append(i.numpy())
            if modality == "fusion" and model.last_gate is not None:
                gates.append(model.last_gate.cpu().numpy())
    if not idxs:
        return np.array([], dtype=int), np.array([]), None
    order = np.argsort(np.concatenate(idxs))
    p = np.concatenate(probs)[order]
    y = np.concatenate(ys)[order]
    g = np.concatenate(gates)[order] if gates else None
    return y, p, g


def train_model(modality, train_arrays, val_arrays, cfg, seed, device="cpu",
                verbose=False):
    """Train one model; early stopping + threshold on validation only."""
    set_seed(seed)
    n_frames = int(train_arrays["radar"].shape[1])
    model = build_model(modality, cfg["embed_dim"], cfg["dropout"],
                        n_frames=n_frames).to(device)
    n_params = count_params(model)
    csi_only = modality == "csi"
    train_loader = make_loader(train_arrays, cfg["batch_size"], True, seed,
                               csi_only_valid=csi_only)
    # early stopping uses a val subsample (full val is used afterwards
    # for threshold selection) — full-val eval every epoch is too slow
    val_sub = val_arrays
    max_val = cfg.get("max_val_windows", 8000)
    if len(val_arrays["y"]) > max_val:
        rng = np.random.RandomState(seed)
        keep = np.sort(rng.choice(len(val_arrays["y"]), max_val,
                                  replace=False))
        val_sub = {k: (v[keep] if isinstance(v, np.ndarray)
                       and len(v) == len(val_arrays["y"]) else v)
                   for k, v in val_arrays.items()}
    # class-balance the val subsample: early stopping and threshold
    # selection must not inherit the split's class prior (t2 is ~80%
    # occupied, t is ~68% empty — a prior-tuned threshold miscalibrates)
    yv = val_sub["y"]
    i0, i1 = np.where(yv == 0)[0], np.where(yv == 1)[0]
    if len(i0) and len(i1):
        rng = np.random.RandomState(seed)
        n = min(len(i0), len(i1))
        keep = np.sort(np.concatenate([rng.choice(i0, n, replace=False),
                                       rng.choice(i1, n, replace=False)]))
        val_sub = {k: (v[keep] if isinstance(v, np.ndarray)
                       and len(v) == len(yv) else v)
                   for k, v in val_sub.items()}
    val_loader = make_loader(val_sub, cfg["batch_size"], False,
                             csi_only_valid=csi_only)
    pos = float(train_arrays["y"].sum())
    neg = float(len(train_arrays["y"]) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1)))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg.get("weight_decay", 1e-4))

    best = {"f1": -1.0, "state": None, "epoch": 0}
    history = []
    bad = 0
    t0 = time.time()
    loss = torch.tensor(0.0)
    augment = cfg.get("augment", True) and modality != "csi"
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for r, c, y, _ in train_loader:
            r = r.float().to(device); c = c.float().to(device)
            if augment:
                # flip dim -2 (doppler axis of RD map = radial-velocity
                # sign; azimuth axis of RA map = left/right mirror) for a
                # random half of the batch — both are physical symmetries
                flip = torch.rand(r.shape[0]) < 0.5
                r[flip] = torch.flip(r[flip], dims=[-2])
            if modality == "fusion":
                # CSI dropout: zero the CSI input on 30% of windows that
                # have CSI, so the gate learns a radar-only fallback and
                # can't over-fit to t1's receiver-specific CSI patterns
                has = c.abs().sum(dim=(1, 2)) > 0
                drop = (torch.rand(c.shape[0]) < 0.3) & has
                c[drop] = 0.0
            y = y.to(device).float()
            out = _forward(model, modality, r, c)
            loss = criterion(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        vy, vp, _ = predict(model, val_loader, device, modality)
        vm = compute_metrics(vy, vp, 0.5)
        history.append({"epoch": epoch, "val_macro_f1": vm["macro_f1"],
                        "val_acc": vm["accuracy"],
                        "train_loss": float(loss.item())})
        if verbose:
            print(f"  ep{epoch}: val f1={vm['macro_f1']:.3f} acc={vm['accuracy']:.3f}",
                  flush=True)
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

    vy, vp, _ = predict(model, val_loader, device, modality)
    thr_grid = np.arange(0.30, 0.71, 0.02)
    scores = [compute_metrics(vy, vp, t)["macro_f1"] for t in thr_grid]
    best_thr = float(thr_grid[int(np.argmax(scores))])
    return model, best_thr, history, train_seconds, n_params


def evaluate(model, arrays, modality, threshold, device="cpu", batch_size=256):
    loader = make_loader(arrays, batch_size, False,
                         csi_only_valid=(modality == "csi"))
    y, p, g = predict(model, loader, device, modality)
    idx = np.arange(len(arrays["y"]))
    if modality == "csi":
        idx = idx[arrays["csi_valid"]]
    rec_idx = arrays["rec_idx"][idx]
    win = compute_metrics(y, p, threshold)
    minute = per_minute_metrics(y, p, rec_idx, threshold)
    gate_mean = float(g.mean()) if g is not None else None
    return {"window": win, "minute": minute, "gate_radar_mean": gate_mean,
            "y": y.tolist(), "p": [float(x) for x in p],
            "rec_idx": rec_idx.tolist()}
