"""Per-minute radar SNR scatter plot across the four minute-folder splits.

SNR definition (uniform across splits, computed from the E2 preprocessing
cache in `E2/cache/<split>/<minute>.npz`):
  * channel 0 of each cached window is the log1p-compressed range-Doppler
    map (24 x 24, doppler x range). It is de-compressed with expm1.
  * Zero-Doppler rows (static clutter) and the two nearest range bins are
    masked out; the remaining peak power is compared against the map
    median (noise floor): SNR_frame = 10*log10(peak / median).
  * The per-minute value is the max over all frames in the minute (the
    strongest observed scatterer, mirroring the manifest's target snr_db
    semantics); the frame mean is also written to the CSV.

The recorder-side snr_db from `minute_summary.targets` / chunk `targets`
(calibration_minutes / train only) is additionally collected into the
CSV as `device_snr_db` for reference.

Outputs:
  E2/outputs/snr_minutes.csv
  E2/outputs/figs/snr_minutes_scatter.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SPLITS = ["calibration_minutes", "train", "validation_minutes",
          "test_minutes"]
SPLIT_XLABELS = {"calibration_minutes": "calibration", "train": "train",
                 "validation_minutes": "validation", "test_minutes": "test"}

# 24x24 range-Doppler: doppler DC is row 12; bilinear resize smears it
# into the adjacent rows -> mask 11..13. First two range bins are the
# near-range leakage region.
_DC_ROWS = (10, 14)   # rows [10,14) masked  -> rows 11,12,13
_NEAR_RANGE = 2       # cols [0,2) masked


def activity_of(rec):
    """empty / sleep / present color label for one recording."""
    for lab in rec.orig_labels:
        if lab.endswith("empty") or lab == "empty":
            return "empty"
        if lab.endswith("sleep"):
            return "sleep"
        if lab.endswith("present") or lab == "present":
            return "present"
    return "empty" if rec.label == 0 else "present"


def device_snr(rec):
    """Recorder-side target snr_db (max over targets), NaN if absent."""
    vals = []
    try:
        man, _ = C.load_manifest(rec.folder / "manifest.json")
    except Exception:
        return float("nan")
    ms = man.get("minute_summary") or {}
    for t in ms.get("targets") or []:
        if isinstance(t, dict) and isinstance(t.get("snr_db"), (int, float)):
            vals.append(t["snr_db"])
    radar = (man.get("outputs") or {}).get("radar") or {}
    for ch in radar.get("chunks") or []:
        for t in ch.get("targets") or []:
            if isinstance(t, dict) and isinstance(t.get("snr_db"), (int, float)):
                vals.append(t["snr_db"])
    return float(np.max(vals)) if vals else float("nan")


def snr_from_cache(rec):
    """Return (snr_max_db, snr_mean_db, n_frames) or None."""
    path = C.CACHE_DIR / rec.source / f"{rec.folder.name}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    rd = np.asarray(d["radar"][:, :, 0], dtype=np.float32)  # (nW,50,24,24)
    if rd.size == 0:
        return None
    power = np.expm1(rd).reshape(-1, rd.shape[-2], rd.shape[-1])
    flat = power.reshape(power.shape[0], -1)
    med = np.median(flat, axis=1) + 1e-9
    masked = power.copy()
    masked[:, _DC_ROWS[0]:_DC_ROWS[1], :] = 0.0
    masked[:, :, :_NEAR_RANGE] = 0.0
    peak = masked.reshape(power.shape[0], -1).max(axis=1)
    snr = 10.0 * np.log10(peak / med)
    snr = snr[np.isfinite(snr)]
    if snr.size == 0:
        return None
    return float(snr.max()), float(snr.mean()), int(power.shape[0])


def collect():
    recs, _ = C.discover_recordings()
    rows, skipped = [], 0
    for i, rec in enumerate(recs):
        s = snr_from_cache(rec)
        if s is None:
            skipped += 1
            continue
        snr_max, snr_mean, n_frames = s
        rows.append({
            "split": rec.source,
            "minute": rec.folder.name,
            "activity": activity_of(rec),
            "label": rec.label,
            "placement": rec.placement,
            "snr_db": snr_max,
            "snr_mean_db": snr_mean,
            "n_frames": n_frames,
            "device_snr_db": device_snr(rec),
        })
        if (i + 1) % 250 == 0:
            print(f"[{i+1}/{len(recs)}]", flush=True)
    print(f"{len(rows)} minutes with SNR, {skipped} skipped (no cache)")
    return rows


def plot(rows, out_png):
    acts = ["empty", "sleep", "present"]
    colors = {"empty": "#000000", "sleep": "#F2C500", "present": "#D62728"}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for act in acts:
        xs, ys = [], []
        for r in rows:
            if r["activity"] != act:
                continue
            xs.append(SPLITS.index(r["split"]) + rng.uniform(-0.3, 0.3))
            ys.append(r["snr_db"])
        if xs:
            ax.scatter(xs, ys, s=9, alpha=0.45, color=colors[act],
                       label=f"{act} ({len(xs)})", linewidths=0)
    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels([SPLIT_XLABELS[s] for s in SPLITS])
    ax.set_xlabel("minute folder")
    ax.set_ylabel("SNR (dB) — peak vs noise floor, clutter-masked")
    ax.set_ylim(bottom=1)
    ax.set_title("Radar SNR per captured minute")
    ax.legend(title="label", markerscale=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print("wrote", out_png)


def plot_box(rows, out_png):
    """Grouped box plots: one box per label inside each split."""
    acts = ["empty", "sleep", "present"]
    colors = {"empty": "#000000", "sleep": "#F2C500", "present": "#D62728"}
    positions, data, facecolors = [], [], []
    for si, split in enumerate(SPLITS):
        base = si * 4.0
        for ai, act in enumerate(acts):
            vals = [r["snr_db"] for r in rows
                    if r["split"] == split and r["activity"] == act]
            if not vals:
                continue
            positions.append(base + ai)
            data.append(vals)
            facecolors.append(colors[act])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(data, positions=positions, widths=0.65,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=4),
                    medianprops=dict(color="white", linewidth=1.6),
                    whiskerprops=dict(color="#555555"),
                    capprops=dict(color="#555555"),
                    flierprops=dict(marker="o", markersize=2.5,
                                    markerfacecolor="#999999",
                                    markeredgecolor="none", alpha=0.5))
    for patch, fc in zip(bp["boxes"], facecolors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.8)
    ax.set_ylim(bottom=1)
    # annotations: n just above the axis floor, median above each box
    for pos, vals, fc in zip(positions, data, facecolors):
        ax.text(pos, 1.2, f"n={len(vals)}", ha="center", va="bottom",
                fontsize=7, color="#444444")
        ax.text(pos, np.median(vals) + 0.4, f"{np.median(vals):.1f}",
                ha="center", va="bottom", fontsize=7.5,
                color=fc if fc != "#000000" else "#333333",
                fontweight="bold")
    centers = [si * 4.0 + 1.0 for si in range(len(SPLITS))]
    ax.set_xticks(centers)
    ax.set_xticklabels([SPLIT_XLABELS[s] for s in SPLITS])
    ax.set_xlabel("minute folder")
    ax.set_ylabel("SNR (dB) — peak vs noise floor, clutter-masked")
    ax.set_title("Radar SNR per captured minute (box: median/quartiles, "
                 "diamond: mean)")
    handles = [plt.Line2D([], [], marker="s", color=colors[a],
                          linestyle="", markersize=9,
                          label=f"{a} ({sum(1 for r in rows if r['activity']==a)})")
               for a in acts]
    ax.legend(handles=handles, title="label", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print("wrote", out_png)


def main():
    import pandas as pd
    rows = collect()
    df = pd.DataFrame(rows)
    csv_path = C.OUTPUT_DIR / "snr_minutes.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print("wrote", csv_path)
    plot(rows, C.OUTPUT_DIR / "figs" / "snr_minutes_scatter.png")
    plot_box(rows, C.OUTPUT_DIR / "figs" / "snr_minutes_boxplot.png")


if __name__ == "__main__":
    main()
