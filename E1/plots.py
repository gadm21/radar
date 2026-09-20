"""Figures for the E1 dataset audit.

Reads outputs/inspection.json + outputs/per_minute.csv and writes
figures to outputs/figs/.

Usage:  python E1/plots.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def _load():
    rep = json.loads((C.OUTPUT_DIR / "inspection.json").read_text())
    df = pd.read_csv(C.OUTPUT_DIR / "per_minute.csv")
    return rep, df


def fig_label_distribution(rep, df):
    """Task-label counts per source, split into placement/activity/position."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title) in zip(axes, (("activity_counts", "Activity (task labels)"),
                                     ("placement_counts", "Placement / split"),
                                     ("position_counts", "Position (train2_minutes)"))):
        sources = C.SOURCES
        labels = sorted({k for s in sources for k in rep[s][key] if k != "none"})
        x = np.arange(len(labels))
        w = 0.8 / len(sources)
        for i, s in enumerate(sources):
            vals = [rep[s][key].get(l, 0) for l in labels]
            ax.bar(x + (i - (len(sources) - 1) / 2) * w, vals, w, label=s)
            for xi, v in zip(x + (i - (len(sources) - 1) / 2) * w, vals):
                if v:
                    ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ground-truth label distribution")
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "label_distribution.png", dpi=140)
    plt.close(fig)


def fig_raw_labels(rep):
    """All raw manifest labels (incl. auxiliary) per source."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    for ax, s in zip(axes.flat, C.SOURCES):
        counts = rep[s]["raw_label_counts"]
        items = sorted(counts.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in items], [v for _, v in items])
        for i, (_, v) in enumerate(items):
            ax.text(v, i, f" {v}", va="center", fontsize=8)
        ax.set_title(f"{s} — raw manifest labels")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "raw_labels.png", dpi=140)
    plt.close(fig)


def fig_timeline(rep, df):
    """Folders per day + cumulative coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    for s in C.SOURCES:
        days = rep[s]["folders_per_day"]
        xs = list(days.keys())
        ax.bar(xs, [days[x] for x in xs], alpha=0.7, label=s)
    ax.set_title("Minute folders per day")
    ax.set_ylabel("folders")
    ax.tick_params(axis="x", rotation=45)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for s in C.SOURCES:
        d = df[(df["source"] == s) & np.isfinite(df["folder_ts"])]
        if d.empty:
            continue
        ts = np.sort(d["folder_ts"].to_numpy())
        ax.plot((ts - ts[0]) / 86400.0, np.arange(1, len(ts) + 1), label=s)
    ax.set_title("Cumulative folders over collection time")
    ax.set_xlabel("days since first folder")
    ax.set_ylabel("cumulative folders")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "collection_timeline.png", dpi=140)
    plt.close(fig)


def fig_file_completeness(rep):
    """Presence of each expected artefact per source (percent of folders)."""
    keys = [("n_with_manifest", "manifest.json"),
            ("n_with_radar_bin", "radar_*.bin"),
            ("n_with_npz", "capture.npz"),
            ("n_with_csi", "CSI samples"),
            ("n_with_xytracking", "xy-tracking.json"),
            ("n_with_ha_status", ".home_assistant_status.json"),
            ("n_with_sense_error", "sense_hat.error.json")]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(keys))
    w = 0.8 / len(C.SOURCES)
    for i, s in enumerate(C.SOURCES):
        n = rep[s]["n_folders"]
        vals = [100.0 * rep[s][k] / max(n, 1) for k, _ in keys]
        bars = ax.bar(x + (i - (len(C.SOURCES) - 1) / 2) * w, vals, w,
                      label=f"{s} (n={n})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in keys], rotation=25, ha="right")
    ax.set_ylabel("% of folders containing artefact")
    ax.set_ylim(0, 115)
    ax.set_title("Artefact completeness per dataset")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "file_completeness.png", dpi=140)
    plt.close(fig)


def fig_manifest_health(rep):
    """Manifest parse modes + capture status per source."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (key, title) in zip(axes, (("manifest_parse_modes", "Manifest parse mode"),
                                     ("manifest_status", "Capture status field"))):
        sources = C.SOURCES
        cats = sorted({k for s in sources for k in rep[s][key]})
        x = np.arange(len(cats))
        w = 0.8 / len(sources)
        for i, s in enumerate(sources):
            vals = [rep[s][key].get(c, 0) for c in cats]
            ax.bar(x + (i - (len(sources) - 1) / 2) * w, vals, w, label=s)
            for xi, v in zip(x + (i - (len(sources) - 1) / 2) * w, vals):
                if v:
                    ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=25, ha="right")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "manifest_health.png", dpi=140)
    plt.close(fig)


def fig_radar_frames(df):
    """Radar frame-count distributions (bin folders and npz folders)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for s in C.SOURCES:
        d = df[(df["source"] == s) & (df["n_radar_bin"] > 0)]
        if not d.empty:
            ax.hist(d["radar_frames"], bins=40, alpha=0.6, label=f"{s} (bin)")
    d = df[df["npz_radar_frames"].apply(lambda v: isinstance(v, (int, float)) and v > 0)]
    if not d.empty:
        ax.hist(d["npz_radar_frames"].astype(float), bins=40, alpha=0.6,
                label="npz sources (capture.npz)")
    ax.set_title("Radar frames per minute folder")
    ax.set_xlabel("frames (10 Hz nominal -> ~600 = full minute)")
    ax.set_ylabel("folders")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for s in C.SOURCES:
        d = df[(df["source"] == s) & (df["n_radar_bin"] > 0)]
        if not d.empty:
            ax.hist(d["n_radar_bin"], bins=40, alpha=0.6, label=s)
    ax.set_title("radar_*.bin files per folder")
    ax.set_xlabel("bin files (nominal ~58-60)")
    ax.set_ylabel("folders")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "radar_frames.png", dpi=140)
    plt.close(fig)


def fig_csi_coverage(df):
    """CSI sample counts per folder."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in C.SOURCES:
        d = df[(df["source"] == s) & (df["csi_lines"] > 0)]
        if not d.empty:
            ax.hist(d["csi_lines"], bins=40, alpha=0.6,
                    label=f"{s} (n={len(d)})")
    ax.set_title("CSI samples per folder (folders with >0 samples)")
    ax.set_xlabel("CSI_DATA lines (~100 Hz nominal)")
    ax.set_ylabel("folders")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "csi_coverage.png", dpi=140)
    plt.close(fig)


def fig_problems(rep):
    """Problem-category counts."""
    cats = rep.get("problem_categories", {})
    if not cats:
        return
    items = sorted(cats.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.35 * len(items))))
    ax.barh([k for k, _ in items], [v for _, v in items])
    for i, (_, v) in enumerate(items):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.set_title(f"Detected problems by category (n={sum(cats.values())})")
    ax.set_xlabel("instances")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "problems.png", dpi=140)
    plt.close(fig)


def fig_duration(df):
    """Capture duration distribution."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in C.SOURCES:
        d = df[(df["source"] == s) & np.isfinite(df["duration_s"])]
        if not d.empty:
            ax.hist(d["duration_s"], bins=40, alpha=0.6, label=s)
    ax.axvline(60, color="k", ls="--", lw=1, label="nominal 60 s")
    ax.set_title("Capture duration per folder")
    ax.set_xlabel("seconds")
    ax.set_ylabel("folders")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "durations.png", dpi=140)
    plt.close(fig)


def main():
    C.FIGS_DIR.mkdir(parents=True, exist_ok=True)
    rep, df = _load()
    fig_label_distribution(rep, df)
    fig_raw_labels(rep)
    fig_timeline(rep, df)
    fig_file_completeness(rep)
    fig_manifest_health(rep)
    fig_radar_frames(df)
    fig_csi_coverage(df)
    fig_problems(rep)
    fig_duration(df)
    print(f"Wrote figures to {C.FIGS_DIR}")


if __name__ == "__main__":
    main()
