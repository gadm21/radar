"""Export trained occupancy models as TorchScript .pt files -> deploy/.

Each export is a traced torch.jit module — no Python dependency needed
for inference. Metadata (threshold, normalization stats, input spec) is
embedded in the archive as `meta.json` (readable via
torch.jit.load(..., _extra_files=...) or any zip reader).

Usage:
    python E2/export_models.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from models import build_model

DEPLOY = C.OUTPUT_DIR.parent.parent / "deploy"

INPUT_SPEC = {
    "radar": {
        "shape": "(B, 50, 2, 24, 24)",
        "desc": "50-frame window; ch0 = range-Doppler, ch1 = range-azimuth; "
                "log1p then z-score with norm['radar_mean'/'radar_std']",
    },
    "csi": {
        "shape": "(B, 128, 52)",
        "desc": "52-subcarrier amplitude interpolated to 128 steps; "
                "log1p then z-score with norm['csi_mean'/'csi_std']; "
                "rolling variance (w=20) is applied inside the model",
    },
}


def _example_inputs(modality):
    r = torch.randn(2, 50, 2, 24, 24)
    c = torch.randn(2, 128, 52)
    if modality == "fusion":
        return (r, c)
    if modality == "radar":
        return (r,)
    return (c,)


def export(src_pt, dst_name, extra=None):
    ckpt = torch.load(src_pt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(ckpt["modality"], cfg["embed_dim"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    modality = ckpt["modality"]
    with torch.no_grad():
        scripted = torch.jit.trace(model, _example_inputs(modality))

    meta = {
        "modality": modality,
        "cfg": cfg,
        "threshold": ckpt["threshold"],
        "norm": ckpt["norm"],
        "norm_mode": ckpt.get("norm_mode", "global"),
        "input_spec": {k: INPUT_SPEC[k] for k in
                       (("radar", "csi") if modality == "fusion"
                        else (modality,))},
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    if extra:
        meta.update(extra)

    dst = DEPLOY / dst_name
    torch.jit.save(scripted, str(dst),
                   _extra_files={"meta.json": json.dumps(meta)})
    print(f"exported {dst} ({dst.stat().st_size // 1024} KB, "
          f"{meta['n_params']} params, thr={meta['threshold']:.3f})")
    return dst


def main():
    DEPLOY.mkdir(exist_ok=True)
    out = C.OUTPUT_DIR

    # radar model fine-tuned on t support minutes (E3 'full' strategy)
    e3 = out / "model_radar_e3_full.pt"
    if e3.exists():
        export(e3, "radar_occupancy_e3_finetuned.pt",
               {"finetune": "E3 full fine-tune on 10 support minutes of t"})
    else:
        print(f"WARNING: {e3} missing — run experiments.py --steps e2,e3")

    # fusion model trained on t1 (E2)
    fus = out / "model_fusion.pt"
    if fus.exists():
        export(fus, "fusion_occupancy_e2.pt")
    else:
        print(f"WARNING: {fus} missing — run experiments.py --steps e2,e3")


if __name__ == "__main__":
    main()
