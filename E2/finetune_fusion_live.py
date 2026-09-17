"""Fine-tune the E2 fusion occupancy model on live-device empty-room data.

Same domain-adaptation idea as finetune_live.py but for the fusion model
(radar encoder + CSI stats encoder + gated fusion + head).

Usage: python E2/finetune_fusion_live.py <live_empty_dir> [live_occ_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from models import build_model

sys.path.insert(0, str(Path.home() / "Desktop/thothcraft/thoth/src"))
from backend.model_runtime import (
    _e2_frames_to_maps, e2_csi_windows, _e2_parse_csi_amplitude,
    E2_FRAME_PAYLOAD_BYTES,
)


def live_csi_norm(dirs):
    """Per-subcarrier mean/std of log1p(amplitude) over all live captures.
    The training device's CSI amplitude scale is ~16x larger, so the cached
    csi_std crushes live CSI to ~0.06. Normalizing with live stats restores
    the occupied>empty separation the CSI encoder needs."""
    amps = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            if "csi_sample_offsets" not in z.files:
                continue
            coff = z["csi_sample_offsets"]; craw = z["csi_sample_bytes"]
            for i in range(len(coff) - 1):
                v = _e2_parse_csi_amplitude(
                    bytes(craw[coff[i]:coff[i+1]]).decode("utf-8", "replace"))
                if v is not None:
                    amps.append(v)
    A = np.log1p(np.stack(amps))
    return {"csi_mean": A.mean(0).tolist(),
            "csi_std": (A.std(0) + 1e-6).tolist()}

CKPT = C.OUTPUT_DIR / "model_fusion.pt"
OUT = C.OUTPUT_DIR / "model_fusion_live.pt"


def live_windows(npz_path, rmean, rstd, cnorm):
    """capture.npz -> (radar_w, csi_w) normalized, aligned."""
    d = np.load(npz_path, allow_pickle=False)
    off = d["radar_sample_offsets"]; raw = d["radar_sample_bytes"]
    n = len(off) - 1
    seq = d["radar_sample_sequence"].astype(np.int64)
    order = np.argsort(seq, kind="stable")
    pay, times = [], []
    rts = d["radar_sample_unix_ns"].astype(np.float64)[order] / 1e9
    for idx, i in enumerate(order):
        pkt = bytes(raw[off[i]:off[i+1]])
        pl = pkt[12:] if len(pkt) >= 12 else pkt
        if len(pl) == E2_FRAME_PAYLOAD_BYTES:
            pay.append(bytes(pl)); times.append(float(rts[idx]))
    nW = len(pay) // 50
    if nW == 0:
        return None, None
    m = _e2_frames_to_maps(pay[:nW*50]).reshape(nW, 50, 2, 24, 24)
    radar = ((m - rmean.reshape(1, 1, 2, 1, 1)) / rstd.reshape(1, 1, 2, 1, 1)).astype(np.float32)
    kept = np.asarray(times[:nW*50]).reshape(nW, 50)
    win_t0, win_t1 = kept[:, 0], kept[:, -1]
    # csi samples -> (receiver, t, line)
    csi_samples = []
    if "csi_sample_offsets" in d.files:
        coff = d["csi_sample_offsets"]; craw = d["csi_sample_bytes"]
        crx = d["csi_sample_receiver_index"]; cts = d["csi_sample_unix_ns"].astype(np.float64) / 1e9
        for i in range(len(coff) - 1):
            csi_samples.append((int(crx[i]), float(cts[i]),
                                bytes(craw[coff[i]:coff[i+1]]).decode("utf-8", "replace")))
    csi, _valid = e2_csi_windows(csi_samples, win_t0, win_t1, cnorm)
    return radar, csi


def cached_windows(split, cnorm, live_norm):
    """Cached windows with CSI re-normalized from cached scale to live scale:
    live = (cached*cached_std + cached_mean - live_mean)/live_std."""
    cm = np.asarray(cnorm["csi_mean"], dtype=np.float32)
    cs = np.asarray(cnorm["csi_std"], dtype=np.float32)
    lm = np.asarray(live_norm["csi_mean"], dtype=np.float32)
    ls = np.asarray(live_norm["csi_std"], dtype=np.float32)
    Xs, Cs, ys = [], [], []
    for f in sorted((C.CACHE_DIR / split).glob("*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        label = int(meta.get("label", -1))
        if label < 0:
            continue
        Xs.append(d["radar"].astype(np.float32))
        c = d["csi"].astype(np.float32)
        c = (c * cs + cm - lm) / ls          # cached-norm -> live-norm
        Cs.append(c.astype(np.float32))
        ys += [label] * d["radar"].shape[0]
    return np.concatenate(Xs), np.concatenate(Cs), np.asarray(ys)


def main():
    live_dir = Path(sys.argv[1])
    occ_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    norm = ckpt["norm"]
    rmean = np.asarray(norm["radar_mean"], dtype=np.float32)
    rstd = np.asarray(norm["radar_std"], dtype=np.float32)
    # live CSI norm: fixes the cross-device amplitude-scale crush
    live_norm = dict(norm)
    live_norm.update(live_csi_norm([live_dir] + ([occ_dir] if occ_dir else [])))
    print("live csi_std[:4]:", np.round(np.asarray(live_norm["csi_std"])[:4], 3),
          "vs cached", np.round(np.asarray(norm["csi_std"])[:4], 3))

    Xs, Cs, ys, ws = [], [], [], []
    n_live = 0
    for f in sorted(live_dir.glob("*.npz")):
        r, c = live_windows(f, rmean, rstd, live_norm)
        if r is None:
            continue
        Xs.append(r); Cs.append(c); ys += [0] * r.shape[0]
        ws += [3.0] * r.shape[0]; n_live += r.shape[0]
    if occ_dir and occ_dir.exists():
        for f in sorted(occ_dir.glob("*.npz")):
            r, c = live_windows(f, rmean, rstd, live_norm)
            if r is None:
                continue
            Xs.append(r); Cs.append(c); ys += [1] * r.shape[0]
            ws += [3.0] * r.shape[0]; n_live += r.shape[0]
    for split in ("train_minutes", "val_minutes", "test_minutes"):
        Xc, Cc, yc = cached_windows(split, norm, live_norm)
        Xs.append(Xc); Cs.append(Cc); ys += yc.tolist(); ws += [1.0] * len(yc)
    X = np.concatenate(Xs); Cc = np.concatenate(Cs)
    y = np.asarray(ys, dtype=np.float32); w = np.asarray(ws, dtype=np.float32)
    print(f"dataset: {len(y)} | live-empty={n_live} | occ={int(y.sum())} empty={int((y==0).sum())}")

    model = build_model("fusion", cfg["embed_dim"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    Xt = torch.from_numpy(X); Ct = torch.from_numpy(Cc)
    yt = torch.from_numpy(y); wt = torch.from_numpy(w)
    n = len(y); idx = np.arange(n)
    for epoch in range(30):
        np.random.shuffle(idx); tot = 0.0
        for i in range(0, n, 256):
            b = idx[i:i+256]
            loss = (lossf(model(Xt[b], Ct[b]), yt[b]) * wt[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if epoch % 5 == 0:
            print(f"  epoch {epoch}: loss {tot/n:.4f}")

    model.eval()
    def minute_top2(r, c):
        with torch.inference_mode():
            p = torch.sigmoid(model(torch.from_numpy(r), torch.from_numpy(c))).numpy()
        return float(np.sort(p)[-2:].mean()) if len(p) >= 2 else float(p.mean())

    live_top2 = []
    for f in sorted(live_dir.glob("*.npz")):
        r, c = live_windows(f, rmean, rstd, live_norm)
        if r is not None:
            live_top2.append(minute_top2(r, c))
    print("live-empty minute top2:", np.round(live_top2, 3))
    if occ_dir and occ_dir.exists():
        locc = []
        for f in sorted(occ_dir.glob("*.npz")):
            r, c = live_windows(f, rmean, rstd, live_norm)
            if r is not None:
                locc.append(minute_top2(r, c))
        print("live-occupied minute top2:", np.round(locc, 3))
    occ_top2 = []
    for f in sorted((C.CACHE_DIR / "val_minutes").glob("*.npz")):
        d = np.load(f); meta = json.loads(str(d["meta"]))
        if int(meta.get("label", -1)) == 1:
            cm = np.asarray(norm["csi_mean"], np.float32); cs = np.asarray(norm["csi_std"], np.float32)
            lm = np.asarray(live_norm["csi_mean"], np.float32); ls = np.asarray(live_norm["csi_std"], np.float32)
            c = (d["csi"].astype(np.float32) * cs + cm - lm) / ls
            occ_top2.append(minute_top2(d["radar"].astype(np.float32), c.astype(np.float32)))
    print("val-occupied minute top2:", np.round(occ_top2, 3))
    thr = float(np.clip((max(live_top2) + min(occ_top2)) / 2, 0.5, 0.95)) if occ_top2 and live_top2 else 0.75
    print("chosen minute_threshold:", round(thr, 3))
    ckpt["state_dict"] = model.state_dict()
    ckpt["minute_threshold"] = thr
    # embed live CSI norm so the deployed model normalizes live CSI correctly
    ckpt["norm"] = dict(norm)
    ckpt["norm"]["csi_mean"] = live_norm["csi_mean"]
    ckpt["norm"]["csi_std"] = live_norm["csi_std"]
    torch.save(ckpt, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
