"""Dataset inspection for the E3 left/right pipeline.

Scans test_minutes/, validates structure, counts left/right recordings,
storage formats, and radar timing. Writes outputs/inspection.json.

Usage:  python E3/inspect_dataset.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    OUTPUT = C.OUTPUT_DIR
    OUTPUT.mkdir(parents=True, exist_ok=True)

    recordings, problems = C.discover_recordings()
    report = {"problems": problems}

    print(f"Discovered {len(recordings)} left/right recordings, {len(problems)} problems")

    by_label = Counter(C.CLASS_NAMES[r.label] for r in recordings)
    by_format = Counter(r.storage_format for r in recordings)
    by_label_format = Counter((C.CLASS_NAMES[r.label], r.storage_format) for r in recordings)
    print("  labels:", dict(by_label))
    print("  storage formats:", dict(by_format))
    print("  label x format:", dict(by_label_format))

    report["n_recordings"] = len(recordings)
    report["by_label"] = dict(by_label)
    report["by_storage_format"] = dict(by_format)
    report["by_label_and_format"] = {f"{k[0]}/{k[1]}": v for k, v in by_label_format.items()}

    print("\n=== radar timing (sampling up to 30 recordings) ===")
    rng = np.random.RandomState(0)
    idx = rng.choice(len(recordings), min(30, len(recordings)), replace=False)
    frame_counts, intervals = [], []
    for i in idx:
        r = recordings[int(i)]
        try:
            payloads, ts = C.recording_radar_frames(r)
        except Exception as e:
            problems.append(f"{r.rec_id}: radar read failed: {e}")
            continue
        frame_counts.append(len(payloads))
        d = np.diff(ts)
        d = d[(d > 0) & (d < 5)]
        intervals.extend(d.tolist())
    intervals = np.asarray(intervals)
    fps = 1.0 / intervals if len(intervals) else np.array([np.nan])
    report["timing"] = {
        "n_recordings_sampled": len(frame_counts),
        "radar_frames_per_recording": {
            "mean": float(np.mean(frame_counts)) if frame_counts else None,
            "min": int(np.min(frame_counts)) if frame_counts else None,
            "max": int(np.max(frame_counts)) if frame_counts else None,
        },
        "radar_fps": {
            "mean": float(np.nanmean(fps)),
            "median": float(np.nanmedian(fps)),
            "std": float(np.nanstd(fps)),
        },
    }
    t = report["timing"]
    print(f"  frames/rec={t['radar_frames_per_recording']}, fps mean={t['radar_fps']['mean']:.2f} "
          f"med={t['radar_fps']['median']:.2f} std={t['radar_fps']['std']:.2f}")

    out = OUTPUT / "inspection.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out}")
    if problems:
        print(f"{len(problems)} problems recorded (first 10):")
        for p in problems[:10]:
            print("  -", p)


if __name__ == "__main__":
    main()
