"""Companion panels of actual sensor excerpts for six LLM explanation minutes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from E1.common import decode_csi_iq, decode_radar_views

SRC = ROOT / "E2/outputs/frozen/llm"
DEST = ROOT / "paper/icc_assets"
selection = pd.read_csv(SRC / "explainability_selection.csv")

def records(path, lo, hi):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            ts = float(row["ts"])
            if ts < lo:
                continue
            if ts >= hi:
                break
            yield row

examples = []
for item in selection.itertuples(index=False):
    folder = ROOT / "multilink_test_minutes" / item.unit
    manifest = json.loads((folder / "manifest.json").read_text())
    lo = manifest["started_at"] + 60 * int(item.minute)
    csi_file = folder / manifest["sensors"]["csi"]["file"]
    rad_file = folder / manifest["sensors"]["radar"]["file"]
    csi = np.array([np.abs(decode_csi_iq(r["payload"]))
                    for r in records(csi_file, lo, lo+5)], dtype=np.float32)
    radar = list(records(rad_file, lo, lo+5))
    if csi.ndim != 2 or not radar:
        raise RuntimeError(f"No signals for {item.id}")
    rd = decode_radar_views(radar[len(radar)//2]["payload"])[0]
    with np.load(ROOT / "E1/cache/multilink" / (item.unit + ".npz")) as cache:
        start = 60 * int(item.minute)
        valid = cache["radar_rsec"][start:start+60]
        re_seconds = cache["radar_vsec"][start:start+60, 2].astype(np.float32)
        re_std = re_seconds[valid].std(axis=0)
    examples.append((item, csi, rd, re_std))

csi_all = np.concatenate([x[1].ravel() for x in examples])
rd_all = np.concatenate([x[2].ravel() for x in examples])
re_all = np.concatenate([x[3].ravel() for x in examples])
cmin,cmax = np.quantile(csi_all,[.01,.99])
rmin,rmax = np.quantile(rd_all,[.01,.99])
emin,emax = np.quantile(re_all,[.01,.99])
fig,axes = plt.subplots(6,3,figsize=(10.2,9.1),layout="constrained")
for i,(item,csi,rd,re_std) in enumerate(examples):
    ax1,ax2,ax3 = axes[i]
    # Display at most 320 packets while retaining the full 5-s span.
    take = np.linspace(0,len(csi)-1,min(len(csi),320)).round().astype(int)
    ax1.imshow(csi[take].T,aspect="auto",origin="lower",cmap="Blues",
               vmin=cmin,vmax=cmax,interpolation="none")
    ax2.imshow(rd,origin="lower",cmap="Oranges",vmin=rmin,vmax=rmax,
               interpolation="none")
    ax3.imshow(re_std,origin="lower",cmap="magma",vmin=emin,vmax=emax,
               interpolation="none")
    label = "empty" if item.label == 0 else "occupied"
    ax1.set_title(f"{item.id} | {label} | CSI amplitude, first 5 s",fontsize=9)
    ax2.set_title(f"{item.id} | {label} | logged radar RD map",fontsize=9)
    ax3.set_title(f"{item.id} | {label} | RE temporal std (60 s)",fontsize=9)
    ax1.set_ylabel("subcarrier",fontsize=8)
    ax2.set_ylabel("range bin",fontsize=8)
    ax3.set_ylabel("range bin",fontsize=8)
    ax1.set_xlabel("packet (subsampled)",fontsize=8)
    ax2.set_xlabel("Doppler bin",fontsize=8)
    ax3.set_xlabel("elevation bin",fontsize=8)
    for ax in (ax1,ax2,ax3):
        ax.tick_params(labelsize=7,length=2)
fig.suptitle("Six explanation-check minutes: sensor excerpts, not the complete 60-s model input",
             fontsize=11)
DEST.mkdir(exist_ok=True)
fig.savefig(DEST/"explainability_signals.pdf",bbox_inches="tight")
fig.savefig(DEST/"explainability_signals.png",dpi=220,bbox_inches="tight")
plt.close(fig)
