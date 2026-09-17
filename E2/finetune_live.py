"""Fine-tune the E2 radar occupancy model on live-device empty-room data.

The deployed model over-fires on the Pi because the device's radar produces
a different feature distribution (azimuth channel / temporal-variance peaks)
than the training placement. We recalibrate by fine-tuning on:
  - live empty-room windows (label 0)   <- domain adaptation target
  - cached empty windows   (label 0)
  - cached occupied windows (label 1)   <- preserve occupied detection

Usage: python E2/finetune_live.py <live_empty_dir> [live_occ_dir]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from models import build_model

# reuse the exact deployed preprocessing
sys.path.insert(0, str(Path.home() / "Desktop/thothcraft/thoth/src"))
try:
    from backend.model_runtime import _e2_frames_to_maps, E2_FRAME_PAYLOAD_BYTES
except Exception:
    # fallback: local copy of the same routine
    from preprocess import frames_to_maps as _fm  # type: ignore
    _e2_frames_to_maps = None

CKPT = C.OUTPUT_DIR / "model_radar.pt"
OUT = C.OUTPUT_DIR / "model_radar_live.pt"


def live_windows(npz_path, mean, std):
    """Raw capture.npz -> normalized (nW,50,2,24,24) float32 windows."""
    d = np.load(npz_path, allow_pickle=False)
    off = d["radar_sample_offsets"]; raw = d["radar_sample_bytes"]
    n = len(off) - 1
    seq = d["radar_sample_sequence"].astype(np.int64)
    order = np.argsort(seq, kind="stable")
    pay = []
    for i in order:
        pkt = bytes(raw[off[i]:off[i+1]])
        pl = pkt[12:] if len(pkt) >= 12 else pkt
        if len(pl) == E2_FRAME_PAYLOAD_BYTES:
            pay.append(bytes(pl))
    nW = len(pay) // 50
    if nW == 0:
        return None
    m = _e2_frames_to_maps(pay[:nW*50]).reshape(nW, 50, 2, 24, 24)
    return ((m - mean.reshape(1, 1, 2, 1, 1)) / std.reshape(1, 1, 2, 1, 1)).astype(np.float32)


def cached_windows(split):
    """All cached windows for a split -> (X, y)."""
    Xs, ys = [], []
    for f in sorted((C.CACHE_DIR / split).glob("*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        label = int(meta.get("label", -1))
        if label < 0:
            continue
        Xs.append(d["radar"].astype(np.float32))
        ys += [label] * d["radar"].shape[0]
    return np.concatenate(Xs), np.asarray(ys)


def main():
    live_dir = Path(sys.argv[1])
    occ_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    norm = ckpt["norm"]
    mean = np.asarray(norm["radar_mean"], dtype=np.float32)
    std = np.asarray(norm["radar_std"], dtype=np.float32)

    # ---- dataset ----
    Xs, ys, ws = [], [], []
    n_live = 0
    for f in sorted(live_dir.glob("*.npz")):
        w = live_windows(f, mean, std)
        if w is None:
            continue
        Xs.append(w); ys += [0] * w.shape[0]
        ws += [6.0] * w.shape[0]          # upweight live-empty (domain target)
        n_live += w.shape[0]
    if occ_dir and occ_dir.exists():
        for f in sorted(occ_dir.glob("*.npz")):
            w = live_windows(f, mean, std)
            if w is None:
                continue
            Xs.append(w); ys += [1] * w.shape[0]
            ws += [3.0] * w.shape[0]      # upweight live-occupied too
            n_live += w.shape[0]
    for split in ("train_minutes", "val_minutes", "test_minutes"):
        Xc, yc = cached_windows(split)
        Xs.append(Xc); ys += yc.tolist()
        ws += [1.0] * len(yc)
    X = np.concatenate(Xs); y = np.asarray(ys, dtype=np.float32)
    w = np.asarray(ws, dtype=np.float32)
    print(f"dataset: {len(y)} windows | live-empty={n_live} | occ={int(y.sum())} empty={int((y==0).sum())}")

    # ---- model ----
    model = build_model("radar", cfg["embed_dim"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(reduction="none")

    Xt = torch.from_numpy(X); yt = torch.from_numpy(y); wt = torch.from_numpy(w)
    n = len(y); idx = np.arange(n)
    for epoch in range(45):
        np.random.shuffle(idx)
        tot = 0.0
        for i in range(0, n, 256):
            b = idx[i:i+256]
            logit = model(Xt[b])
            loss = (lossf(logit, yt[b]) * wt[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if epoch % 5 == 0:
            print(f"  epoch {epoch}: loss {tot/n:.4f}")

    # ---- eval: per-minute top2 on live-empty + cached ----
    model.eval()
    def minute_top2(w):
        with torch.inference_mode():
            p = torch.sigmoid(model(torch.from_numpy(w))).numpy()
        return float(np.sort(p)[-2:].mean()) if len(p) >= 2 else float(p.mean())

    live_top2 = []
    for f in sorted(live_dir.glob("*.npz")):
        w = live_windows(f, mean, std)
        if w is not None:
            live_top2.append(minute_top2(w))
    print("live-empty minute top2:", np.round(live_top2, 3))
    if occ_dir and occ_dir.exists():
        locc = []
        for f in sorted(occ_dir.glob("*.npz")):
            w = live_windows(f, mean, std)
            if w is not None:
                locc.append(minute_top2(w))
        print("live-occupied minute top2:", np.round(locc, 3))

    # occupied reference: a few cached occupied minutes
    occ_top2 = []
    for f in sorted((C.CACHE_DIR / "val_minutes").glob("*.npz")):
        d = np.load(f); meta = json.loads(str(d["meta"]))
        if int(meta.get("label", -1)) == 1:
            occ_top2.append(minute_top2(d["radar"].astype(np.float32)))
    print("val-occupied minute top2:", np.round(occ_top2, 3))

    # pick threshold separating live-empty from occupied
    thr = (max(live_top2) + min(occ_top2)) / 2 if occ_top2 and live_top2 else 0.7
    thr = float(np.clip(thr, 0.5, 0.95))
    print("chosen minute_threshold:", round(thr, 3))

    ckpt["state_dict"] = model.state_dict()
    ckpt["minute_threshold"] = thr
    torch.save(ckpt, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
