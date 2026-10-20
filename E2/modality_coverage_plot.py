"""Valid/invalid modality (radar / CSI) coverage per minute across splits.

A modality is "valid" for a minute when it produced usable data in the E2
preprocessing cache (`radar.shape[0] > 0` windows, `csi_count > 0`). For
the ~50 folders without a cache entry the raw files are inspected instead:
radar = capture.npz `radar_sample_offsets` non-empty or a non-empty
`radar_*.bin`; CSI = capture.npz `csi_sample_offsets` non-empty or a
`wifi_csi_*.csv` larger than the header (~200 B; empty files are ~54 B).

Outputs:
  E2/outputs/modality_coverage.csv
  E2/outputs/figs/modality_coverage.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SPLITS = ["calibration_minutes", "train", "validation_minutes",
          "test_minutes"]
SPLIT_XLABELS = {"calibration_minutes": "calibration", "train": "train",
                 "validation_minutes": "validation",
                 "test_minutes": "test"}

MIN_CSI_CSV_BYTES = 200   # wifi_csi header-only files are ~54 B
MIN_RADAR_BIN_BYTES = 12  # one MMW-HAT frame header


def raw_validity(folder):
    """(radar_ok, csi_ok) from raw files — fallback for uncached minutes."""
    npz_path = folder / "capture.npz"
    if npz_path.exists():
        try:
            d = np.load(npz_path)
            radar_ok = ("radar_sample_offsets" in d
                        and len(d["radar_sample_offsets"]) > 0)
            csi_ok = ("csi_sample_offsets" in d
                      and len(d["csi_sample_offsets"]) > 0)
            return bool(radar_ok), bool(csi_ok)
        except Exception:
            pass  # fall through to file checks
    radar_ok = any(p.stat().st_size > MIN_RADAR_BIN_BYTES
                   for p in folder.glob("radar_*.bin"))
    csi_ok = any(p.stat().st_size > MIN_CSI_CSV_BYTES
                 for p in folder.glob("wifi_csi_*.csv"))
    return radar_ok, csi_ok


def minute_validity(rec):
    """(radar_ok, csi_ok, via_cache) for one recording."""
    cache = C.CACHE_DIR / rec.source / f"{rec.folder.name}.npz"
    if cache.exists():
        try:
            d = np.load(cache)
            radar_ok = d["radar"].shape[0] > 0
            csi_ok = int(d["csi_count"]) > 0 if "csi_count" in d \
                else d["csi"].shape[0] > 0
            return bool(radar_ok), bool(csi_ok), True
        except Exception:
            pass
    radar_ok, csi_ok = raw_validity(rec.folder)
    return radar_ok, csi_ok, False


def collect():
    recs, _ = C.discover_recordings()
    rows = []
    for i, rec in enumerate(recs):
        radar_ok, csi_ok, via_cache = minute_validity(rec)
        rows.append({
            "split": rec.source,
            "minute": rec.folder.name,
            "radar_valid": radar_ok,
            "csi_valid": csi_ok,
            "via_cache": via_cache,
        })
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(recs)}]", flush=True)
    return pd.DataFrame(rows)


def plot(df, out_png):
    modalities = [("radar_valid", "radar"), ("csi_valid", "csi")]
    colors_valid = {"radar": "#1B9E77", "csi": "#7570B3"}
    color_invalid = "#C44E52"
    width, group_gap = 0.34, 0.55
    fig, ax = plt.subplots(figsize=(9, 5.5))
    tick_pos, tick_lbl = [], []
    for si, split in enumerate(SPLITS):
        center = si * (len(modalities) + group_gap)
        sub = df[df["split"] == split]
        for mi, (col, name) in enumerate(modalities):
            x = center + mi
            n_ok = int(sub[col].sum())
            n_bad = int((~sub[col]).sum())
            ax.bar(x, n_ok, width, color=colors_valid[name],
                   label=f"{name} valid" if si == 0 else None)
            ax.bar(x, n_bad, width, bottom=n_ok, color=color_invalid,
                   alpha=0.85, label="invalid" if si == 0 and mi == 0
                   else None)
            # counts next to the bar: outside-left for the left bar of the
            # pair (radar), outside-right for the right bar (csi), so that
            # near-zero invalid segments are still readable
            off = -0.28 if mi == 0 else 0.28
            ha = "right" if mi == 0 else "left"
            ax.text(x + off, max(n_ok / 2, 10), str(n_ok), ha=ha,
                    va="center", fontsize=8, fontweight="bold",
                    color=colors_valid[name])
            bad_y = n_ok + (n_bad / 2 if n_bad else 8)
            ax.text(x + off, bad_y, str(n_bad), ha=ha, va="center",
                    fontsize=8, fontweight="bold", color=color_invalid)
        tick_pos.append(center + (len(modalities) - 1) / 2)
        tick_lbl.append(SPLIT_XLABELS[split])
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl)
    ax.set_xlabel("minute folder")
    ax.set_ylabel("minutes")
    ax.set_title("Valid / invalid modalities per minute "
                 "(paired bars: radar, csi)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print("wrote", out_png)


def main():
    df = collect()
    csv_path = C.OUTPUT_DIR / "modality_coverage.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print("wrote", csv_path)
    summary = (df.groupby("split")[["radar_valid", "csi_valid"]]
               .agg(["sum", "count"]))
    print(summary)
    plot(df, C.OUTPUT_DIR / "figs" / "modality_coverage.png")


if __name__ == "__main__":
    main()
