#!/usr/bin/env python3
"""Run inference with one of the trained models on a captured minute folder.

Usage example:
    python infer.py presence dataset2/present/20260626_1410
    python infer.py localize_r dataset2/present/20260626_1410 --output result.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from ml_utils import PreprocessingConfig, extract_radar_sample, extract_single_csi_features, find_csi_file, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Model name/prefix (e.g. presence, localize_rcs, localize_r) or .pth file")
    parser.add_argument("minute_dir", help="Path to a captured minute folder")
    parser.add_argument("--models_dir", default="output", help="Directory containing saved .pth models")
    parser.add_argument("--output", help="Optional JSON output file; if omitted, prints to stdout")
    parser.add_argument("--device", default="cpu", help="torch device")
    return parser.parse_args()


def _resolve_model_path(model_name, models_dir):
    model_name = str(model_name)
    if model_name.endswith(".pth"):
        candidates = [Path(model_name)]
    else:
        models_dir = Path(models_dir)
        candidates = [
            models_dir / f"{model_name}_model.pth",
            models_dir / f"{model_name}.pth",
        ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find model for {model_name}; tried {candidates}")


def _format_presence_result(logit):
    conf = float(F.sigmoid(logit))
    return {"present": conf >= 0.5, "confidence": round(conf, 6)}


def _format_targets(raw_out):
    # raw_out: (K, 6) -> x, y, orientation, confidence, id, class
    confs = F.sigmoid(raw_out[:, 3]).cpu().numpy()
    raw_out = raw_out.cpu().numpy()
    targets = []
    for i, conf in enumerate(confs):
        if conf < 0.5:
            continue
        targets.append(
            {
                "id": max(0, int(round(float(raw_out[i, 4])))),
                "x": float(raw_out[i, 0]),
                "y": float(raw_out[i, 1]),
                "orientation": float(raw_out[i, 2]),
                "confidence": round(float(conf), 6),
            }
        )
    return targets


def main():
    args = parse_args()
    model_path = _resolve_model_path(args.model_name, args.models_dir)
    minute_dir = Path(args.minute_dir)
    if minute_dir.is_file():
        minute_dir = minute_dir.parent
    minute_id = minute_dir.name

    model, preprocessing_config = load_model(model_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    cfg = PreprocessingConfig(**preprocessing_config)
    class_name = model.__class__.__name__

    out = {"minute": minute_id, "model": model_path.stem, "generated_at": datetime.utcnow().isoformat() + "Z"}

    if class_name == "PresenceModel":
        csi_file = find_csi_file(minute_dir)
        if csi_file is None:
            raise FileNotFoundError(f"No CSI file in {minute_dir}")
        csi_sample = torch.from_numpy(extract_single_csi_features(csi_file, cfg)).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(csi_sample).squeeze()
        out["presence"] = _format_presence_result(logit)
    elif class_name == "RadarOnlyLocalizationModel":
        radar_sample = torch.from_numpy(extract_radar_sample(minute_dir, cfg)).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model(radar_sample).squeeze(0)
        out["targets"] = _format_targets(raw)
    elif class_name == "DualLocalizationModel":
        csi_file = find_csi_file(minute_dir)
        if csi_file is None:
            raise FileNotFoundError(f"No CSI file in {minute_dir}")
        radar_sample = torch.from_numpy(extract_radar_sample(minute_dir, cfg)).unsqueeze(0).to(device)
        csi_sample = torch.from_numpy(extract_single_csi_features(csi_file, cfg)).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model(radar_sample, csi_sample).squeeze(0)
        out["targets"] = _format_targets(raw)
    else:
        raise ValueError(f"Unsupported model class {class_name}")

    json_text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(json_text, encoding="utf-8")
        print(f"Wrote inference output to {args.output}")
    else:
        print(json_text)


if __name__ == "__main__":
    main()
