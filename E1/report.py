"""Generate E1/outputs/REPORT.md from the inspection outputs.

Usage:  python E1/report.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def _md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "-"


def main():
    rep = json.loads((C.OUTPUT_DIR / "inspection.json").read_text())
    df = pd.read_csv(C.OUTPUT_DIR / "per_minute.csv")
    problems = rep.get("problems", [])

    L = []
    L.append("# E1 — Dataset Audit Report")
    L.append("")
    L.append("Exploratory inspection of `train_minutes/`, `val_minutes/` "
             "and `test_minutes/`: file completeness, manifest health, "
             "label taxonomy, sensor coverage, timing, and a catalog of "
             "every missing/corrupt instance.")
    L.append("")

    # ---- overview ----
    L.append("## 1. Overview")
    rows = []
    for s in C.SOURCES:
        r = rep[s]
        rows.append([s, r["n_folders"], r["n_empty_folders"],
                     r["n_with_manifest"], f"{r['total_bytes'] / 1e9:.2f} GB",
                     r["total_radar_bin_files"], r["total_radar_frames_bin"],
                     r["total_csi_lines"], r["total_tmp_files"]])
    L.append(_md_table(rows, ["dataset", "folders", "empty", "with manifest",
                              "size", "radar .bin files", "radar frames (bin)",
                              "CSI lines", ".tmp leftovers"]))
    L.append("")

    # ---- labels ----
    L.append("## 2. Labels")
    L.append("Task labels: `t1_*`/`t2_*` in `train_minutes/` (placements "
             "t1/t2), `t_*` in `val_minutes/` (placement t), and plain "
             "`empty`/`present` in `test_minutes/` (placement t, newer "
             "naming). `left`/`right` position labels and the "
             "`radar-missing` flag appear in `val_minutes/`. "
             "`absent`/`occupied` are auxiliary auto-labels written by "
             "the recorder (stripped from train/val manifests by "
             "`E2/clean_minutes.py`).")
    L.append("")
    for s in C.SOURCES:
        r = rep[s]
        L.append(f"### {s} ({r['n_folders']} folders)")
        L.append("")
        L.append("**Activity (task) labels**")
        L.append(_md_table(sorted(r["activity_counts"].items()),
                           ["activity", "folders"]))
        L.append("")
        L.append("**Placement**")
        L.append(_md_table(sorted(r["placement_counts"].items()),
                           ["placement", "folders"]))
        L.append("")
        if any(k != "none" for k in r["position_counts"]):
            L.append("**Position**")
            L.append(_md_table(sorted(r["position_counts"].items()),
                               ["position", "folders"]))
            L.append("")
        L.append("**Raw manifest labels**")
        L.append(_md_table(sorted(r["raw_label_counts"].items(),
                                  key=lambda kv: -kv[1]),
                           ["label", "occurrences"]))
        L.append("")
        L.append(f"Label status: `{json.dumps(r['label_status'])}`")
        L.append("")

    # ---- manifest health ----
    L.append("## 3. Manifest health")
    for s in C.SOURCES:
        r = rep[s]
        L.append(f"### {s}")
        L.append(f"- Parse modes: `{json.dumps(r['manifest_parse_modes'])}`")
        L.append(f"- Capture status: `{json.dumps(r['manifest_status'])}`")
        L.append(f"- Schema versions: `{json.dumps(r['manifest_schema'])}`")
        L.append("")

    # ---- sensor coverage ----
    L.append("## 4. Sensor / artefact coverage")
    rows = []
    for s in C.SOURCES:
        r = rep[s]
        n = r["n_folders"]
        rows.append([s, n,
                     f"{r['n_with_radar_bin']} ({_pct(r['n_with_radar_bin'], n)})",
                     f"{r['n_with_npz']} ({_pct(r['n_with_npz'], n)})",
                     f"{r['n_with_csi']} ({_pct(r['n_with_csi'], n)})",
                     f"{r['n_with_xytracking']} ({_pct(r['n_with_xytracking'], n)})",
                     f"{r['n_with_ha_status']} ({_pct(r['n_with_ha_status'], n)})",
                     f"{r['n_with_sense_error']} ({_pct(r['n_with_sense_error'], n)})"])
    L.append(_md_table(rows, ["dataset", "folders", "radar .bin", "capture.npz",
                              "CSI >0 samples", "xy-tracking", "HA status",
                              "sense-hat error"]))
    L.append("")
    for s in C.SOURCES:
        r = rep[s]
        L.append(f"- **{s}** bin frames/folder: `{json.dumps(r['bin_frames_per_folder'])}`")
        if r["npz_frames_per_folder"].get("n"):
            L.append(f"- **{s}** npz frames/folder: `{json.dumps(r['npz_frames_per_folder'])}`")
        L.append(f"- **{s}** duration s: `{json.dumps(r['duration_s'])}`")
    L.append("")

    # ---- timeline ----
    L.append("## 5. Collection timeline")
    for s in C.SOURCES:
        days = rep[s]["folders_per_day"]
        L.append(f"### {s}")
        L.append(_md_table(sorted(days.items()), ["day", "folders"]))
        L.append("")

    # ---- problems ----
    L.append("## 6. Missing / corrupt instances")
    cats = Counter(p["category"] for p in problems)
    L.append(f"Total problem instances: **{len(problems)}**")
    L.append("")
    L.append(_md_table(sorted(cats.items(), key=lambda kv: -kv[1]),
                       ["category", "count"]))
    L.append("")
    # per-category detail tables (bounded)
    for cat, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        L.append(f"### {cat} ({n})")
        rows = [[p["source"], p["folder"], p["detail"][:110]]
                for p in problems if p["category"] == cat]
        L.append(_md_table(rows[:60], ["source", "folder", "detail"]))
        if len(rows) > 60:
            L.append(f"\n*... {len(rows) - 60} more in `problems.csv`*")
        L.append("")

    # ---- figures ----
    L.append("## 7. Figures")
    for name, desc in [
        ("label_distribution.png", "task-label distribution (activity / placement / position)"),
        ("raw_labels.png", "all raw manifest labels incl. auxiliary auto-labels"),
        ("collection_timeline.png", "folders per day + cumulative coverage"),
        ("file_completeness.png", "artefact presence per dataset"),
        ("manifest_health.png", "manifest parse modes + capture status"),
        ("radar_frames.png", "radar frame / bin-file counts per folder"),
        ("csi_coverage.png", "CSI sample counts per folder"),
        ("durations.png", "capture duration distribution"),
        ("problems.png", "problem categories"),
    ]:
        L.append(f"![{name}](figs/{name}) — {desc}")
        L.append("")

    out = C.OUTPUT_DIR / "REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
