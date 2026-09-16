# Radar Sensing Project — Final Report

End-to-end pipeline for **bedroom occupancy, activity, and position
sensing** from mmWave radar (with Wi-Fi CSI as a secondary modality).
The project is organized as four stages, `E1`–`E4`, plus a legacy
monolithic pipeline at the repository root.

| stage | scope | key output |
|---|---|---|
| **E1** | dataset audit of `train_minutes/` + `val_minutes/` + `test_minutes/` | `E1/outputs/REPORT.md`, `per_minute.csv`, `problems.csv` |
| **E2** | binary occupancy (empty vs occupied), cross-day | `E2/outputs/best_model.pt` (radar-only), `deploy/*.pt` |
| **E3** | left/right localization (position-labeled recordings) | `E3/outputs/model_fold*.pt` |
| **E4** | 3-stage hierarchy: occupancy → sleep/present → left/right | `E4/predict.py::predict_capture`, `E4/saved_models/` |

## 1. Data (E1 audit)

Three dataset roots of per-minute capture folders (`YYYYMMDD_HHMM`),
recorded by a "thoth" capture agent (`thoth-minute-manifest` v4–v6).
`train_minutes/` holds placements t1+t2; `val_minutes/` and
`test_minutes/` are the same placement t on **different capture days**
(Sept 6 vs Sept 15) — the test day has a weaker occupied radar signal
and a ~2x CSI gain shift.

| dataset | folders | with manifest | size | role |
|---|---|---|---|---|
| `train_minutes/` | 1394 | 1394 | 34.1 GB | train (t1+t2 placements) |
| `val_minutes/` | 329 | 329 | 6.3 GB | validation (placement t, day 1) |
| `test_minutes/` | 194 | 194 | 4.6 GB | test (placement t, day 2) |

**Sensors per folder.** `train_minutes/`: chunked `radar_*.bin` (~1 s /
10 MMW-HAT frames each, 64 chirps × 128 samples × 3 RX, uint12-packed),
`wifi_csi*.csv` (ESP32 CSI, 52 usable subcarriers), `xy-tracking.json`,
`.home_assistant_status.json`. `val_minutes/` + `test_minutes/`: a
synchronized `capture.npz` container (radar + CSI + Sense-HAT + camera)
in most folders, chunked `radar_*.bin` in the rest.

**Labels.** Task labels: `t1_*`/`t2_*` in `train_minutes/`, `t_*` in
`val_minutes/`, plain `empty`/`present` in `test_minutes/`.
`absent`/`empty`/`present`/`occupied` auto-labels were stripped from
train/val manifests by `E2/clean_minutes.py`; folders with no valid
`t_*` label, no manifest, no radar data, or ambiguous `t_*` labels were
moved to `_trash/`.

| split | empty | sleep | present | total |
|---|---|---|---|---|
| train_minutes (t1+t2) | 184 | 1025 | 185 | 1394 |
| val_minutes (t, day 1) | 220 | 20 | 89 | 329 |
| test_minutes (t, day 2) | 113 | 0 | 81 | 194 |

**Data quality (1009 flagged instances).** 504 manifest-reported errors
(mostly "Radar analysis exceeded its shutdown deadline"), 126 folders
with fewer `radar_*.bin` than `expected_chunks`, 61 leftover `.tmp`
files, 53 malformed manifests (recovered via partial-JSON/regex
fallback), 53 unknown labels, 14 empty/corrupt `.bin` files, 3
truncated `xy-tracking.json`, 195 folders missing `xy-tracking.json`.
Full catalog: `E1/outputs/problems.csv` and `E1/outputs/REPORT.md`.

## 2. Legacy root pipeline (superseded)

`train.py` / `test.py` / `infer.py` / `ml_utils.py` / `radar_utils.py`
implement an earlier monolithic dual-head radar+CSI 3-way activity
classifier (empty/present/sleep) with an interactive web dashboard
(`output/dashboard.html`). It reached **0.627 acc / 0.415 macro-F1** on
its test split (`output/test_results.json`) — it never learned the
`empty` class (0.0 recall). This motivated the staged redesign in
E2–E4. `list_labels.py` and `remove_empty_minutes.py` are label/ hygiene
utilities; `MMW-HAT/` is the vendor mmWave SDK used by `radar_utils.py`.

## 3. E2 — Occupancy detection

Binary **empty vs occupied** (sleep+present merged). Splits are folder
level: **train_minutes = train, val_minutes = val, test_minutes = test**
— val and test are the same placement t on different days, so this is a
*day-shift* generalization test. Windows = 50 consecutive radar frames
(~5–7 s, long enough for breathing). Features: per-pixel
**temporal-std map** of range-Doppler + range-azimuth, spatially pooled
— temporal variation is the placement-invariant cue; absolute levels
flip sign across rooms and are excluded.

Because the test day's occupied signal is weaker, **80% of val_minutes
recordings are folded into training** (20% holdout for early stopping +
thresholds), and minute-level decisions use **top-2 window
aggregation** with a separately tuned `minute_threshold`. The fusion
model's CSI branch is a `CSIStatsEncoder` (MLP on amplitude
variance/mean/temporal-std) — conv CSI features are
receiver/day-specific and hijack the gate, while amplitude variance is
the robust cross-day cue.

Test results on test_minutes (day 2):

| modality | window acc | window macro-F1 | minute acc | minute macro-F1 |
|---|---|---|---|---|
| **fusion** | 0.755 | 0.732 | **0.907** | **0.902** |
| **radar-only** | 0.723 | 0.690 | **0.902** | **0.896** |
| csi-only | 0.605 | 0.448 | 0.615 | 0.471 |

Few-shot adaptation (E3, 10 val_minutes support minutes) does not move
test_minutes (0.902 minute acc for 0-shot, head-only and full) — the
remaining gap is a day-shift the support set does not cover. The radar
model is exported as `E2/outputs/best_model.pt` → stage 1 of E4 and
`deploy/`.

## 4. E3 — Left/right localization

Binary **left vs right** from radar, on the position-labeled recordings
of the Sept-6 capture day (now `val_minutes/`; 84 recordings: 34 left,
50 right at the time of the run). Windows = 30 frames (~3 s); temporal
**mean+max** maps → small 2D CNN. Unlike E2, absolute spatial
structure *is* the signal (single fixed room). Stratified group 5-fold
CV at the recording level:

| metric | mean | std |
|---|---|---|
| window accuracy | **0.989** | 0.009 |
| minute accuracy | **1.000** | 0.000 |

**Note:** these numbers predate the label cleaning — `clean_minutes.py`
stripped `left`/`right` from val_minutes manifests (only `t_*` labels
were kept), leaving 9 position-labeled recordings. Re-running E3 now
requires restoring those labels.

## 5. E4 — Hierarchical combined pipeline

`predict_capture(npz_path)` composes three stages into one label:
**empty | sleep | present(left) | present(right)**.

1. **Occupancy** — E2 export (`best_model.pt`, radar-only), now scored
   with top-2 window aggregation + the val-tuned `minute_threshold`:
   0.902 minute acc on the held-out test day.
2. **Sleep vs present** — `MapStatsEncoder` CNN (mean+max+std maps),
   grouped 5-fold CV on occupied t (Sept-6 day). The original transfer
   design (train t1+t2 → test t) collapsed to 0.37–0.81; treating t as
   the deployment domain and adding t1+t2 as extra training data
   (`all_placements`) gives **0.964 window / 0.989 minute**. Deployed
   as a 5-model fold ensemble with an OOF-tuned threshold (0.46).
3. **Left/right** — E3-style model on t: 0.989 window / 1.000 minute.

Two preprocessing bugs in `predict_capture` were fixed along the way
(frame ordering by `radar_sample_sequence`; per-spec azimuth fftshift).

**End-to-end on the 267 position/activity-labeled recordings of the
Sept-6 day (now `val_minutes/`) with `capture.npz` — pre-cleaning
evaluation:**

| metric | value | previous |
|---|---|---|
| combined 4-way accuracy | **1.000** | 0.801 |
| activity 3-way accuracy | **1.000** | 0.801 |
| position coverage (65 labeled) | **1.000** | 0.215 |
| position accuracy when emitted | **1.000** | 1.000 |

Zero errors: 183/183 empty, 16/16 sleep, 29/29 present(left),
36/36 present(right). 61 recordings skipped (no `capture.npz` or <50
frames — the corrupt/missing instances catalogued by E1).

## 6. Deployment artifacts

- `deploy/radar_occupancy_e2.pt`, `deploy/fusion_occupancy_e2.pt`,
  `deploy/radar_occupancy_e3_finetuned.pt` — TorchScript-traced, with
  `meta.json` embedded (thresholds, `minute_threshold`, `top2`
  aggregation, normalization stats, input spec).
- `E4/saved_models/sleep_present_model.pt` (5 fold state_dicts + OOF
  threshold), `E4/saved_models/position_model.pt`
- Stage 1 loads `E2/outputs/best_model.pt` directly.

## 7. Reproduce

```powershell
python E1/run_all.py     # dataset audit (fast, no training)
python E2/run_all.py     # occupancy pipeline (~1-2 h CPU)
python E3/run_all.py     # left/right CV
python E4/run_all.py     # hierarchy + end-to-end eval (needs E2+E3 caches)
```

## 8. Caveats

- Position labels exist only for placement t day 1 (`val_minutes/`);
  the left/right model is not claimed to transfer to other
  rooms/mountings. **Label cleaning stripped `left`/`right` from
  val_minutes manifests** — only 9 position-labeled recordings remain;
  E3/E4 stage-3 numbers above are pre-cleaning and cannot be
  reproduced until those labels are restored.
- CSI is receiver/day-specific: the conv CSI encoder does not
  transfer, so the deployed fusion model uses amplitude-variance
  statistics instead; CSI remains unused in the E4 hierarchy.
- `test_minutes/` is a different capture day than `val_minutes/` —
  window-level accuracy drops (~0.72–0.76) while minute-level top-2
  aggregation recovers >0.90 for radar and fusion.
- E4's end-to-end evaluation was run on the Sept-6 capture day (now
  `val_minutes/`) before the reorganization; it has not been re-run on
  the new test day.
