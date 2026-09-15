"""Inference function for the E4 hierarchical pipeline.

    from E4.predict import predict_capture
    label = predict_capture("test_minutes/20260906_2317/capture.npz")
    # -> "empty" | "sleep" | "present(left)" | "present(right)"

The function decodes the radar frames in a `capture.npz` and runs a
3-stage hierarchy:

  1. Occupancy (E2's exported `best_model.pt`, trained t1 / val t2):
     empty vs occupied. Empty -> "empty", stop.
  2. Sleep/present (trained on occupied t1+t2 minutes): occupied ->
     "sleep" or "present". Sleep -> "sleep", stop (sleep recordings
     carry no left/right ground truth).
  3. Left/right (E3-style model, trained on t): present ->
     "present(left)" | "present(right)".

Stages 1-2 share the E2 preprocessing spec (50-frame windows, 24x24
maps); stage 3 uses the E3 spec (30-frame windows, 32x32 maps).
Window-level probabilities are averaged across the recording before the
final decision (minute-level), matching how the models are evaluated.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from models import (build_occupancy_model, build_sleep_present_model,
                    build_position_model)

_MODELS = {}


def _load_models(models_dir=None):
    """Lazy-load the occupancy (E2 export) + sleep/present + position
    models (once per process)."""
    global _MODELS
    if _MODELS:
        return _MODELS
    models_dir = Path(models_dir) if models_dir else C.MODELS_DIR

    # stage 1: E2's exported occupancy model (trained t1 / val t2)
    occ_ckpt = torch.load(C.ROOT / "E2" / "outputs" / "best_model.pt",
                          map_location="cpu", weights_only=False)
    occ = build_occupancy_model(occ_ckpt["cfg"]["embed_dim"],
                                occ_ckpt["cfg"]["dropout"])
    occ.load_state_dict(occ_ckpt["state_dict"])
    occ.eval()

    # stage 2: sleep/present (trained on occupied t1+t2)
    sp_ckpt = torch.load(models_dir / "sleep_present_model.pt",
                         map_location="cpu", weights_only=False)
    sp = build_sleep_present_model(sp_ckpt["cfg"]["embed_dim"],
                                   sp_ckpt["cfg"]["dropout"])
    sp.load_state_dict(sp_ckpt["state_dict"])
    sp.eval()

    # stage 3: left/right (trained on all t)
    pos_ckpt = torch.load(models_dir / "position_model.pt",
                          map_location="cpu", weights_only=False)
    pos = build_position_model(pos_ckpt["cfg"]["embed_dim"],
                               pos_ckpt["cfg"]["dropout"])
    pos.load_state_dict(pos_ckpt["state_dict"])
    pos.eval()

    _MODELS = {"occupancy": (occ, occ_ckpt), "sleep_present": (sp, sp_ckpt),
               "position": (pos, pos_ckpt)}
    return _MODELS


def _normalize(windows, norm, norm_mode="global"):
    r = windows.astype(np.float32)
    if norm_mode == "local":
        rm = r.mean(axis=(0, 1, 3, 4), keepdims=True)
        rs = r.std(axis=(0, 1, 3, 4), keepdims=True) + 1e-6
    else:
        rm = np.asarray(norm["radar_mean"], np.float32)[None, None, :, None, None]
        rs = np.asarray(norm["radar_std"], np.float32)[None, None, :, None, None]
    return (r - rm) / rs


def predict_capture(npz_path, models_dir=None, return_probs=False):
    """Predict the combined label for one capture.npz recording.

    Returns the label string, or (label, details_dict) if return_probs.
    """
    models = _load_models(models_dir)
    occ, occ_ckpt = models["occupancy"]
    sp, sp_ckpt = models["sleep_present"]
    pos, pos_ckpt = models["position"]

    payloads, _ts = C.decode_npz_frames(npz_path)
    if len(payloads) < C.E2_WINDOW_FRAMES:
        raise ValueError(f"{npz_path}: only {len(payloads)} frames, "
                         f"need >= {C.E2_WINDOW_FRAMES}")

    # E2-spec windows are shared by stages 1 and 2
    maps_a = C.frames_to_maps(payloads, C.E2_MAP_SIZE, C.E2_AZ_BINS, C.E2_RANGE_BINS)
    win_a = C.build_windows(maps_a, C.E2_WINDOW_FRAMES)

    # --- stage 1: occupancy (E2 exported model + its tuned threshold) ---
    win_occ = _normalize(win_a, occ_ckpt["norm"])
    with torch.no_grad():
        logits = occ(torch.from_numpy(win_occ).float())
        p_occ = torch.sigmoid(logits).mean().item()
    occupied = p_occ >= occ_ckpt["threshold"]

    details = {
        "occupancy": "occupied" if occupied else "empty",
        "occupancy_prob": float(p_occ),
        "occupancy_threshold": float(occ_ckpt["threshold"]),
        "n_occupancy_windows": int(win_occ.shape[0]),
    }

    if not occupied:
        details.update(activity="empty", activity_probs=None, position=None,
                       position_probs=None, n_position_windows=0)
        details["label"] = "empty"
        return ("empty", details) if return_probs else "empty"

    # --- stage 2: sleep vs present ---
    win_sp = _normalize(win_a, sp_ckpt["norm"],
                        sp_ckpt.get("norm_mode", "global"))
    with torch.no_grad():
        logits = sp(torch.from_numpy(win_sp).float())
        p_present = torch.sigmoid(logits).mean().item()
    activity = 2 if p_present >= 0.5 else 1  # sleep=1, present=2
    details["activity"] = C.ACTIVITY_NAMES[activity]
    details["activity_probs"] = {"sleep": float(1 - p_present),
                                 "present": float(p_present)}

    if activity != 2:  # sleep -> no position (only t_present recordings
        # carry a left/right ground-truth label)
        details.update(position=None, position_probs=None, n_position_windows=0)
        details["label"] = "sleep"
        return ("sleep", details) if return_probs else "sleep"

    # --- stage 3: left/right (E3 spec: 30-frame windows, 32x32 maps) ---
    maps_p = C.frames_to_maps(payloads, C.E3_MAP_SIZE, C.E3_AZ_BINS, C.E3_RANGE_BINS)
    win_p = _normalize(C.build_windows(maps_p, C.E3_WINDOW_FRAMES), pos_ckpt["norm"])
    if win_p.shape[0] == 0:
        details.update(position=None, position_probs=None, n_position_windows=0)
        details["label"] = "present"
        return ("present", details) if return_probs else "present"
    with torch.no_grad():
        logits = pos(torch.from_numpy(win_p).float())
        p_right = torch.sigmoid(logits).mean().item()
    position = int(p_right >= 0.5)
    details["position"] = C.POSITION_NAMES[position]
    details["position_probs"] = {"left": float(1 - p_right), "right": float(p_right)}
    details["n_position_windows"] = int(win_p.shape[0])
    label = f"present({C.POSITION_NAMES[position]})"
    details["label"] = label
    return (label, details) if return_probs else label


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="path to a capture.npz file")
    a = ap.parse_args()
    label, details = predict_capture(a.npz, return_probs=True)
    print(label)
    for k, v in details.items():
        if k != "label":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
