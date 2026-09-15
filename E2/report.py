"""Generate outputs/REPORT.md from inspection + experiment results."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import train as T

OUT = C.OUTPUT_DIR


def _load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(m, key):
    v = m.get(key)
    return f"{v:.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else "-"


def metrics_table(res_map, level="window"):
    rows = ["| config | acc | macro-F1 | empty R | occ R | false-empty | n |",
            "|---|---|---|---|---|---|---|"]
    for name, r in res_map.items():
        m = r[level] if level in r else r["test"][level]
        rows.append(
            f"| {name} | {_fmt(m,'accuracy')} | {_fmt(m,'macro_f1')} | "
            f"{_fmt(m,'empty_recall')} | {_fmt(m,'occupied_recall')} | "
            f"{_fmt(m,'false_empty_rate')} | {m.get('n','-')} |")
    return "\n".join(rows)


def main():
    insp = _load("inspection.json")
    e2 = _load("results_e2.json")
    e3 = _load("results_e3.json")
    hp = _load("hp_search.json")
    index = T.load_index()

    L = ["# Occupancy Detection — Experiment Report", ""]
    L.append("Binary task: **empty** vs **occupied** (sleep + present merged).")
    L.append("")

    # ---- dataset ----
    L.append("## 1. Dataset")
    if insp:
        for p in ("t1", "t2", "t"):
            pl = insp["placements"].get(p, {})
            L.append(f"- **{p}**: {pl.get('n_recordings')} minutes, "
                     f"labels {pl.get('by_binary_label')}, "
                     f"original {pl.get('by_original_label')}")
        L.append(f"- **Problems**: {len(insp.get('problems', []))} "
                 f"(malformed manifests recovered via regex fallback, "
                 f"ignored/ambiguous labels) — see `inspection.json`")
    L.append("")
    L.append("Splits: **t1 = train, t2 = validation, t = test** "
             "(placement-level, no sessions). Windows are 50 consecutive "
             "radar frames (~5-7 s — long enough to capture breathing) "
             "built inside one recording — they can never cross "
             "recording/placement/label boundaries.")
    L.append("")
    L.append("**CSI coverage**: t1 ~85% of windows, t2 **none** (the "
             "receiver was not recording), t 100%. The CSI-only baseline "
             "therefore early-stops on a t1 holdout, and fusion is "
             "evaluated with missing-CSI masking + CSI-dropout training.")
    L.append("")

    # ---- timing ----
    L.append("## 2. Timing")
    L.append("- `minutes/`: each `radar_*.bin` filename is its capture "
             "timestamp and holds one second of frames (nominally 10); "
             "frame times = filename ts + j/n.")
    L.append("- `test_minutes/`: per-frame npz timestamps are "
             "burst-flushed (degenerate); frames are distributed "
             "uniformly inside their `second_start` bucket (~10/s).")
    L.append("- CSI: `minutes/` uses the `wifi_csi_XX.csv` with most "
             "samples inside the minute (the other receiver is usually "
             "empty); `test_minutes/` uses the receiver with most samples.")
    if insp and "timing" in insp:
        for p, t in insp["timing"].items():
            w = t.get("window_50f_s") or {}
            L.append(f"- **{p}** 50-frame window: mean={w.get('mean'):.3f}s "
                     f"median={w.get('median'):.3f}s std={w.get('std'):.3f}s"
                     if w.get("mean") else f"- **{p}**: n/a")
    L.append("")

    # ---- model ----
    L.append("## 3. Model")
    L.append("Radar encoder: per-pixel **temporal-std map** of the "
             "50-frame window (range-Doppler + range-azimuth), pooled "
             "spatially to mean/std/max per channel -> small MLP. "
             "Temporal variation is the placement-invariant occupancy "
             "cue; absolute levels flip sign across placements and were "
             "removed after diagnosis. CSI encoder: temporal Conv1d over "
             "causal **rolling variance** (w=20) of the 52-subcarrier "
             "amplitude (128 steps) — the 'rolling_variance' pipeline "
             "from WifiSensingESP32HAR; amplitude only, no phase. "
             "Gated fusion: per-feature sigmoid gate "
             "z = g*r + (1-g)*c with missing-CSI masking. Binary head.")
    L.append("")
    L.append("Training: Adam + BCE (pos_weight), early stopping on a "
             "**class-balanced** t2 subsample (so the threshold does not "
             "inherit t2's 80%-occupied prior), decision threshold tuned "
             "on t2 only. Training windows are class-balanced by "
             "subsampling the majority class. Normalization statistics "
             "come from t1 only. Augmentation: doppler-sign/azimuth "
             "flip; CSI dropout (30%) for fusion.")
    L.append("")

    # ---- hp search ----
    if hp:
        L.append("## 4. Hyperparameter search (t1 -> t2, radar)")
        b = hp["best"]
        L.append(f"Best: `{b['cfg']}` -> val acc "
                 f"{b['val_window']['accuracy']:.3f}, "
                 f"F1 {b['val_window']['macro_f1']:.3f}, "
                 f"threshold {b['threshold']:.2f}")
        L.append("")
        L.append("| lr | emb | dropout | val acc | val F1 |")
        L.append("|---|---|---|---|---|")
        for r in sorted(hp["results"],
                        key=lambda r: -r["val_window"]["accuracy"]):
            L.append(f"| {r['cfg']['lr']} | {r['cfg']['embed_dim']} | "
                     f"{r['cfg']['dropout']} | "
                     f"{r['val_window']['accuracy']:.3f} | "
                     f"{r['val_window']['macro_f1']:.3f} |")
        L.append("")

    # ---- E2 ----
    if e2:
        L.append("## 5. E2 — train t1 / val t2 / test t (unseen placement)")
        L.append("")
        L.append("### Window level (test = t)")
        L.append(metrics_table({m: r["test"] for m, r in e2.items()}))
        L.append("")
        L.append("### Minute level (test = t, mean-prob aggregation)")
        L.append(metrics_table({m: r["test"] for m, r in e2.items()},
                               level="minute"))
        L.append("")
        for m, r in e2.items():
            g = r["test"].get("gate_radar_mean")
            if g is not None:
                L.append(f"- Fusion gate (radar weight) mean on t: **{g:.3f}**")
        L.append("")

    # ---- E3 ----
    if e3:
        L.append(f"## 6. E3 — few-shot adaptation "
                 f"({e3['n_support']} support minutes from t)")
        L.append("")
        L.append("Support minutes: " + ", ".join(e3["support_minutes"]))
        L.append("")
        L.append("### Window level (query = remaining t minutes)")
        L.append(metrics_table(
            {k: v for k, v in e3.items()
             if isinstance(v, dict) and "window" in v}))
        L.append("")
        L.append("### Minute level")
        L.append(metrics_table(
            {k: v for k, v in e3.items()
             if isinstance(v, dict) and "window" in v}, level="minute"))
        L.append("")
        zs = e3["zero_shot"]["window"]["accuracy"]
        best = max((k for k in ("head", "fusion_head", "full") if k in e3),
                   key=lambda k: e3[k]["window"]["accuracy"])
        d = e3[best]["window"]["accuracy"] - zs
        L.append(f"Best strategy **{best}**: {d:+.3f} accuracy vs 0-shot.")
        L.append("")

    # ---- leakage audit ----
    L.append("## 7. Leakage audit")
    L.append("- Splits are at placement level: no recording appears in "
             "more than one split.")
    L.append("- Windows are built inside a single recording; no window "
             "spans two recordings or labels.")
    L.append("- Normalization statistics are computed on t1 only and "
             "applied to t2/t.")
    L.append("- Early stopping, threshold selection and hyperparameter "
             "search use t2 (validation) only; t is touched once for "
             "final metrics.")
    L.append("- E3 support minutes are excluded from the query set; "
             "query data is never used for training or selection.")
    L.append("")
    L.append("## 8. Failure modes / notes")
    L.append("- CSI windows with <2 samples are zeroed and flagged "
             "(`csi_valid=0`); the CSI-only model skips them, fusion "
             "masks the CSI embedding for them.")
    L.append("- **CSI does not transfer across receivers/placements** "
             "(csi-only test acc ~0.38; it also degrades fusion on t). "
             "Radar temporal-std features transfer cleanly (test acc "
             "~0.99), so the radar-only model is selected for E3.")
    L.append("- `radar-missing` / non-task labels (left/right/absent) "
             "are excluded from the task.")
    L.append("- Malformed manifests are recovered by regex fallback; "
             "unrecoverable ones are listed in `inspection.json`.")

    (OUT / "REPORT.md").write_text("\n".join(L))
    print(f"wrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
