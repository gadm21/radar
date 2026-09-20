"""Scene-feature occupancy model — all preprocessing inside the TorchScript.

The E2 encoders pool temporal-std maps into a few statistics, discarding
the static-scene reflection pattern (body in bed vs empty mattress) that
separates occupied vs empty on a fixed device. This model computes the
full feature set inside ``forward`` so the exported .pt is self-contained:

  radar (B,50,2,24,24) log1p maps -> per-window mean-map & std-map ->
      spatial stats + range/azimuth profiles (12-bin pooled) + argmax
  csi   (B,128,52) log1p grid    -> per-subcarrier mean/std stats + valid

Features are z-scored with buffers fitted on the training data, then an
MLP head emits one logit per window. The runtime's own normalization is
bypassed by exporting identity stats in meta.json (the model owns all
preprocessing); the runtime still applies log1p to CSI grids itself, so
training data must be log1p-transformed to match (see ``prep_csi``).

Protocol: train on train_minutes + train2_minutes, tune thresholds on
validation_minutes, evaluate once on test_minutes (Pi night, 1:10 AM
boundary labels). The exported .pt is the deployment artifact.

Usage:
    python E2/scene_model.py            # train + tune + test + export both
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import train as T

DEPLOY = C.OUTPUT_DIR.parent.parent / "deploy"
PROFILE_BINS = 12
N_RADAR_FEATS = 112          # 2 ch * (6 stats + 4 profiles*12 + 2 argmax)
N_CSI_FEATS = 6
EPOCHS = 60
HIDDEN = 128


# ---------------------------------------------------------------- features

def scene_radar_features(radar: torch.Tensor) -> torch.Tensor:
    """(B,50,2,S,S) log1p maps -> (B,112) scene features."""
    mu = radar.mean(dim=1)
    sd = radar.std(dim=1, unbiased=False)
    b = radar.shape[0]
    cols = []
    blocks = []
    for ch in range(2):
        m = mu[:, ch]                                # (B,S,S)
        s = sd[:, ch]
        cols += [
            m.mean(dim=(1, 2)), m.std(dim=(1, 2), unbiased=False),
            m.amax(dim=(1, 2)),
            s.mean(dim=(1, 2)), s.std(dim=(1, 2), unbiased=False),
            s.amax(dim=(1, 2)),
        ]
        rp_m = m.mean(dim=2)                         # (B,S) range profile
        rp_s = s.mean(dim=2)
        ap_m = m.mean(dim=1)                         # (B,S) azimuth profile
        ap_s = s.mean(dim=1)
        for prof in (rp_m, rp_s, ap_m, ap_s):
            blocks.append(prof.reshape(b, PROFILE_BINS, 2).mean(dim=2))
        cols += [rp_m.argmax(dim=1).to(radar.dtype),
                 rp_s.argmax(dim=1).to(radar.dtype)]
    return torch.cat([torch.stack(cols, dim=1)] + blocks, dim=1)


def scene_csi_features(csi: torch.Tensor) -> torch.Tensor:
    """(B,128,52) log1p grid -> (B,6); invalid (zeroed) windows -> valid=0."""
    cmean = csi.mean(dim=1)                          # (B,52)
    cstd = csi.std(dim=1, unbiased=False)
    valid = (csi.abs().sum(dim=(1, 2)) > 0).to(csi.dtype)
    return torch.stack([
        cmean.mean(dim=1), cmean.std(dim=1, unbiased=False),
        cstd.mean(dim=1), cstd.std(dim=1, unbiased=False),
        cstd.amax(dim=1), valid,
    ], dim=1)


class SceneRadarModel(nn.Module):
    def __init__(self, n_feat: int = N_RADAR_FEATS, hidden: int = HIDDEN):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(n_feat))
        self.register_buffer("feat_std", torch.ones(n_feat))
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, radar: torch.Tensor) -> torch.Tensor:
        f = scene_radar_features(radar)
        f = (f - self.feat_mean) / self.feat_std
        return self.mlp(f).squeeze(1)


class SceneFusionModel(nn.Module):
    def __init__(self, n_feat: int = N_RADAR_FEATS + N_CSI_FEATS,
                 hidden: int = HIDDEN):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(n_feat))
        self.register_buffer("feat_std", torch.ones(n_feat))
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, radar: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        f = torch.cat([scene_radar_features(radar), scene_csi_features(csi)],
                      dim=1)
        f = (f - self.feat_mean) / self.feat_std
        return self.mlp(f).squeeze(1)


# ------------------------------------------------------------------- data

def prep_csi(d) -> np.ndarray:
    """Match the runtime's e2_csi_windows output: log1p + zero invalid."""
    c = np.log1p(d["csi"].astype(np.float32))
    c[~d["csi_valid"].astype(bool)] = 0.0
    return c


def load_split_arrays(splits):
    """Cached windows for the given split names: raw log1p radar +
    runtime-matched CSI (log1p + zero invalid)."""
    index = T.load_index()
    sub = index[index.split.isin(splits)].sort_values("rec_id")
    rad, csi, y, rec = [], [], [], []
    for i, (_, row) in enumerate(sub.iterrows()):
        d = np.load(row["path"])
        n = d["radar"].shape[0]
        rad.append(d["radar"].astype(np.float32))
        csi.append(prep_csi(d))
        y += [int(row["label"])] * n
        rec += [i] * n
    return (np.concatenate(rad), np.concatenate(csi),
            np.array(y), np.array(rec), sub)


def features_for(model: nn.Module, rad: np.ndarray,
                 csi: np.ndarray) -> torch.Tensor:
    """Run the model's own feature extractor -> (N,F) float32 tensor."""
    with torch.inference_mode():
        if isinstance(model, SceneFusionModel):
            return torch.cat([
                scene_radar_features(torch.from_numpy(rad)),
                scene_csi_features(torch.from_numpy(csi))], dim=1)
        return scene_radar_features(torch.from_numpy(rad))


# ------------------------------------------------------------------ train

def fit(model: nn.Module, rad: np.ndarray, csi: np.ndarray, y: np.ndarray,
        seed: int = 0) -> None:
    """Fit feature scaler + train MLP head on all given windows."""
    torch.manual_seed(seed)
    feats = features_for(model, rad, csi)
    model.feat_mean.copy_(feats.mean(dim=0))
    model.feat_std.copy_(feats.std(dim=0, unbiased=False).clamp_min(1e-6))
    x = (feats - model.feat_mean) / model.feat_std
    yt = torch.from_numpy(y.astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float((y == 0).sum() / max(int((y == 1).sum()), 1))
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos, dtype=torch.float32))
    n = len(x)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            loss = crit(model.mlp(x[b]).squeeze(1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()


def predict_probs(model: nn.Module, rad: np.ndarray,
                  csi: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        if isinstance(model, SceneFusionModel):
            out = model(torch.from_numpy(rad), torch.from_numpy(csi))
        else:
            out = model(torch.from_numpy(rad))
    return torch.sigmoid(out).numpy()


def minute_probs(p: np.ndarray, rec: np.ndarray):
    """top2 aggregation per recording -> (minutes,) probs + labels."""
    mins = {}
    for r in np.unique(rec):
        m = rec == r
        pm = np.sort(p[m])
        mins[r] = pm[-2:].mean() if pm.size >= 2 else pm.mean()
    return mins


def eval_split(model, rad, csi, y, rec, thr, mthr):
    """Window + minute metrics for one split at the given thresholds."""
    p = predict_probs(model, rad, csi)
    win = T.compute_metrics(y, p, thr)
    mins = minute_probs(p, rec)
    yt = np.array([y[rec == r][0] for r in mins])
    yp = np.array(list(mins.values()))
    minute = T.compute_metrics(yt, yp, mthr)
    return {"window": win, "minute": minute}


# ----------------------------------------------------------------- export

def tune_thresholds(p: np.ndarray, y: np.ndarray,
                    rec: np.ndarray) -> tuple[float, float]:
    """(window_threshold, minute_threshold) maximizing accuracy."""
    thr = max(np.arange(0.05, 0.95, 0.01),
              key=lambda t: ((p >= t).astype(int) == y).mean())
    mins = minute_probs(p, rec)
    yt = np.array([y[rec == r][0] for r in mins])
    yp = np.array(list(mins.values()))
    mthr = max(np.arange(0.05, 0.95, 0.01),
               key=lambda t: ((yp >= t).astype(int) == yt).mean())
    return float(thr), float(mthr)


IDENTITY_NORM = {
    "radar_mean": [0.0, 0.0], "radar_std": [1.0, 1.0],
    "csi_mean": [0.0] * 52, "csi_std": [1.0] * 52,
}

INPUT_SPEC = {
    "radar": {"shape": "(B, 50, 2, 24, 24)",
              "desc": "50-frame window; ch0 = range-Doppler, ch1 = "
                      "range-azimuth; raw log1p maps (identity norm — "
                      "the model owns all preprocessing)"},
    "csi": {"shape": "(B, 128, 52)",
            "desc": "52-subcarrier amplitude interpolated to 128 steps; "
                    "runtime log1p grid, invalid windows zeroed "
                    "(identity norm — the model owns all preprocessing)"},
}


def export(model: nn.Module, modality: str, dst_name: str,
           threshold: float, minute_threshold: float) -> Path:
    model.eval()
    r = torch.randn(2, 50, 2, 24, 24)
    c = torch.randn(2, 128, 52)
    with torch.no_grad():
        if modality == "fusion":
            scripted = torch.jit.trace(model, (r, c))
        else:
            scripted = torch.jit.trace(model, (r,))
    meta = {
        "modality": modality,
        "arch": "scene_features_mlp",
        "n_features": N_RADAR_FEATS + (N_CSI_FEATS if modality == "fusion" else 0),
        "threshold": threshold,
        "minute_threshold": minute_threshold,
        "minute_aggregation": "top2",
        "norm": IDENTITY_NORM,
        "norm_mode": "identity",
        "input_spec": {k: INPUT_SPEC[k] for k in
                       (("radar", "csi") if modality == "fusion"
                        else ("radar",))},
        "n_params": sum(p.numel() for p in model.parameters()),
        "trained_on": "train_minutes + train2_minutes "
                      "(thresholds tuned on validation_minutes)",
    }
    dst = DEPLOY / dst_name
    torch.jit.save(scripted, str(dst),
                   _extra_files={"meta.json": json.dumps(meta)})
    print(f"exported {dst.name} ({dst.stat().st_size // 1024} KB, "
          f"thr={threshold:.3f} mthr={minute_threshold:.3f})")
    return dst


def registry_metadata(name: str, modality: str,
                      minute_threshold: float) -> dict:
    """thoth-model/v1 registry metadata for upload/seed."""
    radar_in = {"sensor": "radar", "representation": "e2_maps", "frames": 50,
                "shape": [1, 50, 2, 24, 24], "fit": "left_pad_latest",
                "normalization": {"kind": "none"}}
    inputs = [radar_in]
    if modality == "fusion":
        inputs.append({"sensor": "csi", "representation": "e2_grid",
                       "samples": 128, "shape": [1, 128, 52],
                       "fit": "left_pad_latest",
                       "normalization": {"kind": "none"}})
    return {
        "schema": "thoth-model/v1",
        "name": name,
        "version": "scene1",
        "inputs": inputs,
        "output": {"kind": "logits", "path": []},
        "class_names": ["empty", "occupied"],
        "execution": "minute",
        "aggregation": {"kind": "top2", "threshold": minute_threshold},
        "binary_output": True,
        "output_dimension": 1,
    }


def main():
    DEPLOY.mkdir(exist_ok=True)
    tr_r, tr_c, tr_y, tr_rec, _ = load_split_arrays(
        ("train_minutes", "train2_minutes"))
    va_r, va_c, va_y, va_rec, _ = load_split_arrays(("validation_minutes",))
    te_r, te_c, te_y, te_rec, te_idx = load_split_arrays(("test_minutes",))
    print(f"train: {len(tr_y)} win ({int((tr_y == 1).sum())} occ) | "
          f"val: {len(va_y)} win | test: {len(te_y)} win "
          f"({int((te_y == 1).sum())} occ / {int((te_y == 0).sum())} empty)")

    results = {}
    for modality, cls in (("radar", SceneRadarModel),
                          ("fusion", SceneFusionModel)):
        model = cls()
        fit(model, tr_r, tr_c, tr_y)
        # thresholds tuned on validation_minutes only
        vp = predict_probs(model, va_r, va_c)
        thr, mthr = tune_thresholds(vp, va_y, va_rec)
        val = eval_split(model, va_r, va_c, va_y, va_rec, thr, mthr)
        test = eval_split(model, te_r, te_c, te_y, te_rec, thr, mthr)
        print(f"[{modality}] val win={val['window']['accuracy']:.3f} "
              f"min={val['minute']['accuracy']:.3f} | "
              f"test win={test['window']['accuracy']:.3f} "
              f"min={test['minute']['accuracy']:.3f} "
              f"(thr={thr:.2f} mthr={mthr:.2f})")

        dst = export(model, modality, f"{modality}_occupancy_scene.pt",
                     thr, mthr)
        meta = registry_metadata(
            f"{'Fusion' if modality == 'fusion' else 'Radar'} Occupancy Scene",
            modality, mthr)
        (DEPLOY / f"{modality}_occupancy_scene.meta.json").write_text(
            json.dumps(meta, indent=2))
        results[modality] = {"threshold": thr, "minute_threshold": mthr,
                             "val": val, "test": test, "file": dst.name}

    out = C.OUTPUT_DIR / "scene_model_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
