"""Dataset audit for the E1 inspection stage.

Scans every minute folder in `train_minutes/`, `train2_minutes/`,
`validation_minutes/` and `test_minutes/`, checks file
completeness and integrity (manifest, radar .bin chunks / capture.npz,
CSI csv, xy-tracking, home-assistant status, sense-hat), tabulates the
label taxonomy, measures timing/coverage, and catalogs every missing or
corrupt instance.

Outputs (under E1/outputs/):
  inspection.json  -- aggregate report (counts, timing, problem summary)
  per_minute.csv   -- one row per minute folder with every check result
  problems.csv     -- one row per detected problem instance

Usage:  python E1/inspect_dataset.py [--full-json-parse N]
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def scan_folder(folder, source, rng, full_json_budget):
    """Inspect one minute folder -> (record dict, problems list)."""
    rec = {
        "source": source,
        "folder": folder.name,
        "folder_ts": C.folder_start_ts(folder.name),
        "n_files": 0,
        "total_bytes": 0,
        "n_tmp_files": 0,
        # manifest
        "has_manifest": False,
        "manifest_parse": "",
        "manifest_status": "",
        "manifest_schema": "",
        "labels": "",
        "placement": "",
        "activity": "",
        "position": "",
        "label_status": "",
        "duration_s": float("nan"),
        "expected_chunks": "",
        "manifest_chunks": "",
        "n_warnings": 0,
        "n_errors": 0,
        # radar
        "n_radar_bin": 0,
        "radar_frames": 0,
        "radar_bytes": 0,
        "radar_bin_bad": 0,
        "has_npz": False,
        "npz_ok": "",
        "npz_radar_frames": "",
        "npz_csi_samples": "",
        "npz_sense_samples": "",
        "npz_camera_frames": "",
        # csi
        "n_csi_files": 0,
        "csi_lines": 0,
        "csi_bytes": 0,
        "csi_nonempty_files": 0,
        # other sensors
        "has_xytracking": False,
        "xytracking_ok": "",
        "has_ha_status": False,
        "has_sense_error": False,
    }
    problems = []

    def prob(category, detail):
        problems.append({"source": source, "folder": folder.name,
                         "category": category, "detail": detail})

    files = [p for p in folder.iterdir() if p.is_file()]
    rec["n_files"] = len(files)
    rec["total_bytes"] = sum(p.stat().st_size for p in files)
    rec["n_tmp_files"] = sum(1 for p in files if p.name.endswith(".tmp"))
    if rec["n_tmp_files"]:
        prob("leftover_tmp", f"{rec['n_tmp_files']} .tmp file(s): "
             + ", ".join(p.name for p in files if p.name.endswith(".tmp")))
    if not files:
        prob("empty_folder", "folder contains no files")
        return rec, problems

    # ---- manifest ----
    man = {}
    mpath = folder / "manifest.json"
    if not mpath.exists():
        prob("missing_manifest", "no manifest.json")
    else:
        rec["has_manifest"] = True
        man, mode = C.load_manifest(mpath)
        rec["manifest_parse"] = mode
        if mode != "json":
            prob("malformed_manifest", f"parse_mode={mode}")
        rec["manifest_status"] = str(man.get("status") or "")
        rec["manifest_schema"] = str(man.get("schema") or "")
        labels = man.get("labels", []) or []
        rec["labels"] = "|".join(labels)
        cls = C.classify_labels(labels, source, folder.name)
        rec["placement"] = cls["placement"]
        rec["activity"] = cls["activity"]
        rec["position"] = cls["position"]
        rec["label_status"] = cls["status"]
        if cls["status"] == "ambiguous":
            prob("ambiguous_labels", f"labels={labels}")
        for o in cls["other"]:
            prob("unknown_label", f"label '{o}' not in taxonomy")
        rec["n_warnings"] = len(man.get("warnings") or [])
        rec["n_errors"] = len(man.get("errors") or [])
        for e in (man.get("errors") or []):
            prob("manifest_error", str(e)[:120])
        start = C.parse_iso(man.get("capture_started") or man.get("scheduled_start"))
        end = C.parse_iso(man.get("capture_finished"))
        if np.isfinite(start) and np.isfinite(end) and end > start:
            rec["duration_s"] = end - start
        elif np.isfinite(start):
            rec["duration_s"] = float(man.get("duration_seconds") or float("nan"))
        rec["expected_chunks"] = man.get("expected_chunks", "")
        chunks = man.get("outputs", {}).get("radar", {}).get("chunks", []) \
            if isinstance(man.get("outputs"), dict) else []
        rec["manifest_chunks"] = len(chunks)

    # ---- radar: chunked .bin files ----
    bin_files = sorted(folder.glob("radar_*.bin"))
    rec["n_radar_bin"] = len(bin_files)
    bad_ts = 0
    for bp in bin_files:
        n, ok, note = C.check_bin_file(bp)
        rec["radar_frames"] += n
        rec["radar_bytes"] += bp.stat().st_size
        if not ok:
            rec["radar_bin_bad"] += 1
            prob("corrupt_bin", f"{bp.name}: {note}")
        if not np.isfinite(C.radar_fname_ts(bp.name)):
            bad_ts += 1
    if bad_ts:
        prob("bad_bin_name", f"{bad_ts} radar_*.bin name(s) without a parseable timestamp")

    # ---- radar: capture.npz (train2 / validation / test_minutes) ----
    npz = folder / "capture.npz"
    if npz.exists():
        rec["has_npz"] = True
        info, ok, note = C.check_npz(npz)
        rec["npz_ok"] = bool(ok)
        rec["npz_radar_frames"] = info.get("npz_radar_frames", "")
        rec["npz_csi_samples"] = info.get("npz_csi_samples", "")
        rec["npz_sense_samples"] = info.get("npz_sense_samples", "")
        rec["npz_camera_frames"] = info.get("npz_camera_frames", "")
        if not ok:
            prob("corrupt_npz", note)
    elif source in C.NPZ_SOURCES and not bin_files:
        prob("missing_radar", "no capture.npz and no radar_*.bin")
    if source == "train_minutes" and not bin_files:
        prob("missing_radar", "no radar_*.bin files")

    # ---- CSI ----
    csi_files = sorted(folder.glob("wifi_csi*.csv"))
    rec["n_csi_files"] = len(csi_files)
    for cp in csi_files:
        size = cp.stat().st_size
        rec["csi_bytes"] += size
        if size <= 64:  # header-only file (~54 B)
            continue
        try:
            lines = C.count_csv_lines(cp)
        except OSError as exc:
            prob("csi_read_error", f"{cp.name}: {exc}")
            continue
        rec["csi_lines"] += max(0, lines - 1)  # minus header
        rec["csi_nonempty_files"] += 1
    if source == "train_minutes" and files and not csi_files:
        prob("missing_csi", "no wifi_csi*.csv files")

    # ---- other sensor outputs ----
    xy = folder / "xy-tracking.json"
    if xy.exists():
        rec["has_xytracking"] = True
        do_full = full_json_budget[0] > 0
        if do_full:
            full_json_budget[0] -= 1
        ok, note = C.check_json_file(xy, full_parse=do_full)
        rec["xytracking_ok"] = bool(ok)
        if not ok:
            prob("corrupt_xytracking", note)
    elif files:
        prob("missing_xytracking", "no xy-tracking.json")
    rec["has_ha_status"] = (folder / ".home_assistant_status.json").exists()
    rec["has_sense_error"] = (folder / "sense_hat.error.json").exists()

    # ---- cross-checks between manifest and files ----
    if rec["has_manifest"] and rec["manifest_parse"] == "json":
        if isinstance(rec["expected_chunks"], int) and rec["n_radar_bin"]:
            if rec["n_radar_bin"] != rec["expected_chunks"]:
                prob("chunk_count_mismatch",
                     f"expected_chunks={rec['expected_chunks']} but {rec['n_radar_bin']} radar_*.bin on disk")
    if "radar-missing" in (man.get("labels") or []):
        prob("flagged_radar_missing", "manifest label 'radar-missing'")
    return rec, problems


def aggregate(records, problems):
    """Build the aggregate inspection report from per-minute records."""
    rep = {"n_problems": len(problems), "problem_categories": dict(
        Counter(p["category"] for p in problems))}
    for source in C.SOURCES:
        recs = [r for r in records if r["source"] == source]
        s = {
            "n_folders": len(recs),
            "n_empty_folders": sum(1 for r in recs if r["n_files"] == 0),
            "n_with_manifest": sum(1 for r in recs if r["has_manifest"]),
            "manifest_parse_modes": dict(Counter(r["manifest_parse"] or "none" for r in recs)),
            "manifest_status": dict(Counter(r["manifest_status"] or "none" for r in recs)),
            "manifest_schema": dict(Counter(r["manifest_schema"] or "none" for r in recs)),
            "label_status": dict(Counter(r["label_status"] or "none" for r in recs)),
            "raw_label_counts": dict(Counter(
                l for r in recs for l in (r["labels"].split("|") if r["labels"] else []))),
            "placement_counts": dict(Counter(r["placement"] or "none" for r in recs)),
            "activity_counts": dict(Counter(r["activity"] or "none" for r in recs)),
            "position_counts": dict(Counter(r["position"] or "none" for r in recs)),
            "n_with_radar_bin": sum(1 for r in recs if r["n_radar_bin"] > 0),
            "n_with_npz": sum(1 for r in recs if r["has_npz"]),
            "n_with_csi": sum(1 for r in recs if r["csi_lines"] > 0),
            "n_with_xytracking": sum(1 for r in recs if r["has_xytracking"]),
            "n_with_ha_status": sum(1 for r in recs if r["has_ha_status"]),
            "n_with_sense_error": sum(1 for r in recs if r["has_sense_error"]),
            "total_radar_bin_files": sum(r["n_radar_bin"] for r in recs),
            "total_radar_frames_bin": sum(r["radar_frames"] for r in recs),
            "total_csi_lines": sum(r["csi_lines"] for r in recs),
            "total_bytes": sum(r["total_bytes"] for r in recs),
            "total_tmp_files": sum(r["n_tmp_files"] for r in recs),
        }
        # per-day folder counts (from folder-name timestamps)
        days = Counter()
        for r in recs:
            if np.isfinite(r["folder_ts"]):
                days[datetime_day(r["folder_ts"])] += 1
        s["folders_per_day"] = dict(sorted(days.items()))
        # frame-count stats
        fr = [r["radar_frames"] for r in recs if r["n_radar_bin"] > 0]
        nz = [r["npz_radar_frames"] for r in recs
              if isinstance(r["npz_radar_frames"], int) and r["npz_radar_frames"] > 0]
        s["bin_frames_per_folder"] = _stats(fr)
        s["npz_frames_per_folder"] = _stats(nz)
        dur = [r["duration_s"] for r in recs if np.isfinite(r["duration_s"])]
        s["duration_s"] = _stats(dur)
        rep[source] = s
    return rep


def datetime_day(ts):
    from datetime import datetime as _dt
    return _dt.fromtimestamp(ts).strftime("%Y-%m-%d")


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(len(a)), "mean": float(a.mean()), "min": float(a.min()),
            "max": float(a.max()), "median": float(np.median(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-json-parse", type=int, default=40,
                    help="number of xy-tracking.json files to fully json.loads (sampled)")
    args = ap.parse_args()

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    full_json_budget = [args.full_json_parse]

    records, problems = [], []
    for source in C.SOURCES:
        root = C.SOURCE_ROOTS[source]
        if not root.exists():
            problems.append({"source": source, "folder": "",
                             "category": "missing_root", "detail": str(root)})
            continue
        folders = sorted(p for p in root.iterdir() if p.is_dir())
        print(f"Scanning {source}/ ({len(folders)} folders)...", flush=True)
        for i, folder in enumerate(folders):
            if (i + 1) % 200 == 0:
                print(f"  {source}: {i + 1}/{len(folders)}", flush=True)
            rec, probs = scan_folder(folder, source, rng, full_json_budget)
            records.append(rec)
            problems.extend(probs)

    report = aggregate(records, problems)
    report["problems"] = problems

    out = C.OUTPUT_DIR / "inspection.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out}")

    with open(C.OUTPUT_DIR / "per_minute.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {C.OUTPUT_DIR / 'per_minute.csv'} ({len(records)} rows)")

    with open(C.OUTPUT_DIR / "problems.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source", "folder", "category", "detail"])
        writer.writeheader()
        writer.writerows(problems)
    print(f"Wrote {C.OUTPUT_DIR / 'problems.csv'} ({len(problems)} rows)")

    # ---- console summary ----
    for source in C.SOURCES:
        s = report[source]
        print(f"\n=== {source}: {s['n_folders']} folders ===")
        print("  empty:", s["n_empty_folders"], "| with manifest:", s["n_with_manifest"])
        print("  manifest parse:", s["manifest_parse_modes"])
        print("  manifest status:", s["manifest_status"])
        print("  placements:", s["placement_counts"])
        print("  activities:", s["activity_counts"])
        print("  positions:", s["position_counts"])
        print("  radar: bin folders", s["n_with_radar_bin"], "| npz folders",
              s["n_with_npz"], "| bin frames", s["total_radar_frames_bin"])
        print("  csi folders:", s["n_with_csi"], "| xy-tracking:", s["n_with_xytracking"],
              "| sense errors:", s["n_with_sense_error"])
    print(f"\n{len(problems)} problems:", dict(Counter(p['category'] for p in problems)))


if __name__ == "__main__":
    main()
