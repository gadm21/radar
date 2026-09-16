"""Dataset inspection for the E2 occupancy pipeline.

Scans train_minutes/, val_minutes/ and test_minutes/, validates
structure, counts data by split / session / label, measures radar
timing, and writes outputs/inspection.json plus a printed report.

Usage:  python E2/inspect_dataset.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    OUTPUT = C.OUTPUT_DIR
    OUTPUT.mkdir(parents=True, exist_ok=True)

    recordings, problems = C.discover_recordings()
    recordings = C.assign_sessions(recordings)

    report = {"problems": problems, "splits": {}}
    usable = [r for r in recordings if r.status == "ok"]
    print(f"Discovered {len(recordings)} recordings, {len(usable)} usable, "
          f"{len(problems)} problems (see outputs/inspection.json)")

    # ---- counts by split / label / session ----
    for p in ("train_minutes", "val_minutes", "test_minutes"):
        recs = [r for r in usable if r.source == p]
        by_label = Counter(C.CLASS_NAMES[r.label] for r in recs)
        by_orig = Counter(l for r in recs for l in r.orig_labels)
        sessions = defaultdict(list)
        for r in recs:
            sessions[r.session_id].append(r)
        sess_summary = {
            s: {"n_minutes": len(v),
                "label": C.CLASS_NAMES[v[0].label],
                "start": min(x.start_ts for x in v),
                "end": max(x.end_ts for x in v)}
            for s, v in sorted(sessions.items())
        }
        report["splits"][p] = {
            "n_recordings": len(recs),
            "by_binary_label": dict(by_label),
            "by_original_label": dict(by_orig),
            "n_sessions": len(sessions),
            "sessions": sess_summary,
            "parse_modes": dict(Counter(r.parse_mode for r in recs)),
        }
        print(f"\n=== {p}: {len(recs)} recordings, "
              f"{len(sessions)} sessions ===")
        print("  binary labels:", dict(by_label))
        print("  original labels:", dict(by_orig))
        for s, info in sess_summary.items():
            print(f"  {s}: {info['n_minutes']:3d} min  {info['label']:8s} "
                  f"{info['start']:.0f} .. {info['end']:.0f}")

    # ---- radar frame counts + timing on a sample of recordings ----
    print("\n=== radar timing (sampling up to 40 recordings/split) ===")
    timing = {}
    for p in ("train_minutes", "val_minutes", "test_minutes"):
        recs = [r for r in usable if r.source == p]
        rng = np.random.RandomState(0)
        idx = rng.choice(len(recs), min(40, len(recs)), replace=False) if recs else []
        frame_counts, intervals, win5 = [], [], []
        csi_counts = []
        for i in idx:
            r = recs[int(i)]
            try:
                payloads, ts = C.recording_radar_frames(r)
            except Exception as e:
                problems.append(f"{r.rec_id}: radar read failed: {e}")
                continue
            frame_counts.append(len(payloads))
            d = np.diff(ts)
            d = d[(d > 0) & (d < 5)]
            intervals.extend(d.tolist())
            # non-overlapping 50-frame window durations
            for k in range(0, len(ts) - 49, 50):
                win5.append(float(ts[k + 49] - ts[k]))
            try:
                csi, cts = C.recording_csi(r)
                if csi is not None:
                    csi_counts.append(len(csi))
            except Exception as e:
                problems.append(f"{r.rec_id}: csi read failed: {e}")
        intervals = np.asarray(intervals)
        win5 = np.asarray(win5)
        fps = 1.0 / intervals if len(intervals) else np.array([np.nan])
        timing[p] = {
            "n_recordings_sampled": len(frame_counts),
            "radar_frames_per_recording": {
                "mean": float(np.mean(frame_counts)) if frame_counts else None,
                "min": int(np.min(frame_counts)) if frame_counts else None,
                "max": int(np.max(frame_counts)) if frame_counts else None,
            },
            "csi_samples_per_recording": {
                "mean": float(np.mean(csi_counts)) if csi_counts else None,
                "min": int(np.min(csi_counts)) if csi_counts else None,
                "max": int(np.max(csi_counts)) if csi_counts else None,
            },
            "radar_fps": {
                "mean": float(np.nanmean(fps)),
                "median": float(np.nanmedian(fps)),
                "std": float(np.nanstd(fps)),
            },
            "frame_interval_s": {
                "mean": float(np.nanmean(intervals)) if len(intervals) else None,
                "median": float(np.nanmedian(intervals)) if len(intervals) else None,
                "std": float(np.nanstd(intervals)) if len(intervals) else None,
                "p5": float(np.nanpercentile(intervals, 5)) if len(intervals) else None,
                "p95": float(np.nanpercentile(intervals, 95)) if len(intervals) else None,
            },
            "window_50f_s": {
                "mean": float(np.nanmean(win5)) if len(win5) else None,
                "median": float(np.nanmedian(win5)) if len(win5) else None,
                "std": float(np.nanstd(win5)) if len(win5) else None,
                "p5": float(np.nanpercentile(win5, 5)) if len(win5) else None,
                "p95": float(np.nanpercentile(win5, 95)) if len(win5) else None,
                "min": float(np.nanmin(win5)) if len(win5) else None,
                "max": float(np.nanmax(win5)) if len(win5) else None,
            },
        }
        t = timing[p]
        print(f"  {p}: frames/rec={t['radar_frames_per_recording']}, "
              f"fps mean={t['radar_fps']['mean']:.2f} "
              f"med={t['radar_fps']['median']:.2f} std={t['radar_fps']['std']:.2f}, "
              f"50-frame window mean={t['window_50f_s']['mean']:.3f}s "
              f"med={t['window_50f_s']['median']:.3f}s")
    report["timing"] = timing

    out = OUTPUT / "inspection.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out}")
    if problems:
        print(f"{len(problems)} problems recorded (first 10):")
        for p in problems[:10]:
            print("  -", p)


if __name__ == "__main__":
    main()
