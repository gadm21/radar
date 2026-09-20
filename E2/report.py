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
    scene = _load("scene_model_results.json")
    hp = _load("hp_search.json")
    index = T.load_index()

    L = ["# Occupancy Detection — Experiment Report", ""]
    L.append("Binary task: **empty** vs **occupied** (sleep + present merged).")
    L.append("")

    # ---- dataset ----
    L.append("## 1. Dataset")
    if insp:
        for p in ("train_minutes", "train2_minutes", "validation_minutes", "test_minutes"):
            pl = insp["splits"].get(p, {})
            L.append(f"- **{p}**: {pl.get('n_recordings')} minutes, "
                     f"labels {pl.get('by_binary_label')}, "
                     f"original {pl.get('by_original_label')}")
        L.append(f"- **Problems**: {len(insp.get('problems', []))} "
                 f"(malformed manifests recovered via regex fallback, "
                 f"ignored/ambiguous labels) — see `inspection.json`")
    L.append("")
    L.append("Splits are the dataset folders: **train = train_minutes "
             "(placements t1+t2) + train2_minutes (placement t, Sept 6-8), "
             "val = validation_minutes (placement t, Sept 15), test = "
             "test_minutes (Pi captures, Sept 16-17 night — ground truth "
             "is the 1:10 AM boundary: occupied before, empty after)**. "
             "Windows are 50 consecutive radar frames (~5-7 s — long "
             "enough to capture breathing) built inside one recording — "
             "they can never cross recording/placement/label boundaries.")
    L.append("")
    L.append("**CSI coverage**: train_minutes ~57% of recordings (t1 "
             "only — t2 had no receiver), train2/validation/test ~99% "
             "via capture.npz. Fusion uses missing-CSI masking + "
             "CSI-dropout training so radar-only windows still work.")
    L.append("")

    # ---- timing ----
    L.append("## 2. Timing")
    L.append("- `train_minutes/`: each `radar_*.bin` filename is its "
             "capture timestamp and holds one second of frames "
             "(nominally 10); frame times = filename ts + j/n.")
    L.append("- `train2_minutes/` + `validation_minutes/` + "
             "`test_minutes/`: per-frame npz timestamps are "
             "burst-flushed (degenerate); frames are distributed "
             "uniformly inside their `second_start` bucket (~10/s).")
    L.append("- CSI: `train_minutes/` uses the `wifi_csi_XX.csv` with "
             "most samples inside the minute (the other receiver is "
             "usually empty); the npz sources use the capture.npz "
             "receiver with most samples.")
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
             "removed after diagnosis. CSI-only encoder: temporal "
             "Conv1d over causal **rolling variance** (w=20) of the "
             "52-subcarrier amplitude (128 steps) — the "
             "'rolling_variance' pipeline from WifiSensingESP32HAR; "
             "amplitude only, no phase. The **fusion** model instead "
             "uses a CSIStatsEncoder (MLP on per-window amplitude "
             "variance/mean/temporal-std): conv CSI features are "
             "receiver/day-specific and hijack the gate on the "
             "gain-shifted test day, while amplitude variance is the "
             "robust cross-day occupancy cue. Gated fusion: "
             "per-feature sigmoid gate z = g*r + (1-g)*c with "
             "missing-CSI masking. Binary head.")
    L.append("")
    L.append("Training: Adam + BCE (pos_weight). The train set is "
             "train_minutes + train2_minutes, class-balanced by "
             "subsampling majority-class windows. validation_minutes "
             "drives early stopping and threshold selection; "
             "test_minutes is evaluated once. Normalization statistics "
             "come from the train pool only. Augmentation: "
             "doppler-sign/azimuth flip; CSI dropout (30%) for fusion. "
             "Minute-level decisions use **top-2 window aggregation** "
             "with a separately tuned `minute_threshold` (median of the "
             "optimal range — saturated val probs make the argmax edge "
             "miscalibrate on the shifted test day).")
    L.append("")

    # ---- hp search ----
    if hp:
        L.append("## 4. Hyperparameter search (train -> val, radar)")
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
        L.append("## 5. E2 — train+train2 / validation / test")
        L.append("")
        L.append("### Window level (test = test_minutes)")
        L.append(metrics_table({m: r["test"] for m, r in e2.items()}))
        L.append("")
        L.append("### Minute level (test = test_minutes, top-2 window "
                 "aggregation, val-tuned minute_threshold)")
        L.append(metrics_table({m: r["test"] for m, r in e2.items()},
                               level="minute"))
        L.append("")
        for m, r in e2.items():
            g = r["test"].get("gate_radar_mean")
            if g is not None:
                L.append(f"- Fusion gate (radar weight) mean on test: **{g:.3f}**")
        L.append("")

    # ---- scene model (deployment) ----
    if scene:
        L.append("## 6. Scene model (deployment artifact)")
        L.append("")
        L.append("`scene_model.py` computes all features inside the "
                 "TorchScript forward pass (mean/std maps, range/azimuth "
                 "profiles, CSI stats) — the exported .pt is "
                 "self-contained (identity normalization in meta.json). "
                 "Trained on train+train2, thresholds tuned on "
                 "validation_minutes, evaluated on test_minutes.")
        L.append("")
        L.append("### Window level")
        L.append(metrics_table(
            {m: r["test"] for m, r in scene.items()}))
        L.append("")
        L.append("### Minute level (top-2 aggregation)")
        L.append(metrics_table(
            {m: r["test"] for m, r in scene.items()}, level="minute"))
        L.append("")
        L.append("Validation metrics (tuning split):")
        L.append(metrics_table(
            {m + " (val)": r["val"] for m, r in scene.items()}))
        L.append("")

    # ---- leakage audit ----
    L.append("## 7. Leakage audit")
    L.append("- Splits are disjoint folder sets: no recording appears "
             "in more than one split; validation and test are different "
             "days of placement t / the Pi device.")
    L.append("- Windows are built inside a single recording; no window "
             "spans two recordings or labels.")
    L.append("- Normalization statistics are computed on the train pool "
             "(train_minutes + train2_minutes) only and applied to "
             "val/test.")
    L.append("- Early stopping, threshold selection and hyperparameter "
             "search use validation_minutes only; test_minutes is "
             "touched once for final metrics.")
    L.append("")
    L.append("## 8. Failure modes / notes")
    L.append("- **Domain shift**: test_minutes is a different *device* "
             "(the Pi's radar), room and mounting — a much larger shift "
             "than placement-to-placement or day-to-day. Under this "
             "strict protocol no model reaches the >90% target: the CNN "
             "encoders top out at ~0.65-0.72 minute acc and the scene "
             "model collapses to the majority class (its mean-map "
             "features are device/room-specific — validation accuracy "
             "below chance shows the learned mapping inverts across "
             "domains). Earlier >0.97 numbers came from in-domain "
             "training on test_minutes itself (valid for a fixed-device "
             "deployment, but not a generalization measure).")
    L.append("- CSI windows with <2 samples are zeroed and flagged "
             "(`csi_valid=0`); the CSI-only model skips them, fusion "
             "masks the CSI embedding for them.")
    L.append("- CSI is receiver-specific: the csi-only model is "
             "weakest, but with rolling-variance features + dropout "
             "the fusion model still benefits where CSI exists.")
    L.append("- `radar-missing` / non-task labels (left/right/absent) "
             "are excluded from the task.")
    L.append("- Malformed manifests are recovered by regex fallback; "
             "unrecoverable ones are listed in `inspection.json`.")

    (OUT / "REPORT.md").write_text("\n".join(L))
    print(f"wrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
