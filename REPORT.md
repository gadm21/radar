# Radar Occupancy Detection — Report

Binary **empty vs occupied** (sleep + present merged) from mmWave radar
(BGT60TR13C via MMW-HAT) + Wi-Fi CSI (ESP32), for on-device deployment
on a Raspberry Pi ("thoth"). Two stages: **E1** dataset audit, **E2**
occupancy pipeline. Full detail: `E1/outputs/REPORT.md`,
`E2/outputs/REPORT.md`.

## 1. Datasets

One folder per captured minute (`YYYYMMDD_HHMM`).

| dataset | folders | size | role |
|---|---|---|---|
| `train_minutes/` | 1394 | 34.1 GB | train — placements t1 (971) + t2 (423); chunked `radar_*.bin` + `wifi_csi*.csv` |
| `train2_minutes/` | 329 | 6.3 GB | train — placement t, Sept 6-8; mostly `capture.npz` |
| `validation_minutes/` | 194 | 4.6 GB | validation — placement t, Sept 15 |
| `test_minutes/` | 275 | 8.0 GB | test — Pi captures, Sept 16-17 night |

**Labels.** `t1_*`/`t2_*` (train_minutes), `t_*` (train2_minutes),
`empty`/`present` (validation_minutes). `test_minutes` has no manifest
labels — ground truth is the 1:10 AM boundary: minutes before
`20260917_0110` are occupied (176), at/after empty (98). Auto-labels
(`absent`/`occupied`) were stripped by `E2/clean_minutes.py`; folders
with no valid label, no manifest, or no radar data were moved to
`_trash/`.

**Class balance (minutes).** train_minutes: 184 empty / 1025 sleep /
185 present. train2_minutes: 220 empty / 20 sleep / 89 present.
validation_minutes: 113 empty / 81 present. test_minutes: 98 empty /
176 occupied.

**Data quality (1662 flagged instances).** 800 manifest-reported errors
(mostly "Radar analysis exceeded its shutdown deadline"), 470 folders
missing `xy-tracking.json`, 151 `expected_chunks` mismatches, 94 unknown
labels, 73 leftover `.tmp` files, 53 malformed manifests (recovered via
partial-JSON/regex fallback), 15 corrupt `.bin` files, 3 truncated
`xy-tracking.json`, 2 folders with no radar data, 1 missing manifest.
Full catalog: `E1/outputs/problems.csv`.

## 2. Method (E2)

- **Windows**: 50 consecutive radar frames (~5-7 s — long enough for
  breathing), non-overlapping, inside one recording.
- **Radar maps**: per frame, Hann-windowed FFT -> range-Doppler (64x64)
  + range-azimuth (16x64), log1p, resized to 24x24.
- **CSI**: best receiver per recording; 52-subcarrier amplitude, log1p,
  interpolated to a 128-step grid per window; <2 samples -> zeroed +
  flagged.
- **CNN models** (`models.py`): radar encoder = per-pixel temporal-std
  map -> spatial mean/std/max -> MLP (placement-invariant cue; absolute
  levels excluded). Fusion adds a `CSIStatsEncoder` (amplitude
  var/mean/temporal-std) with per-feature gated fusion + missing-CSI
  masking. CSI-only baseline = Conv1d over causal rolling variance.
- **Scene model** (`scene_model.py`): all preprocessing inside the
  TorchScript `forward` — mean+std map stats, 12-bin range/azimuth
  profiles, argmax, CSI stats -> MLP. Self-contained deployment
  artifact (identity normalization in meta.json).
- **Training**: Adam + BCE(pos_weight); train pool class-balanced by
  subsampling; normalization from train pool only; doppler/azimuth flip
  augmentation; 30% CSI dropout (fusion). Early stopping, thresholds
  and HP search on `validation_minutes` only.
- **Minute aggregation**: mean of top-2 window probabilities with a
  separately tuned `minute_threshold`.

## 3. Results — test_minutes (Pi night, held out)

CNN encoders (HP search best: lr=1e-3, embed=128, dropout=0.2):

| modality | window acc | window F1 | minute acc | minute F1 |
|---|---|---|---|---|
| fusion | 0.507 | 0.477 | 0.567 | 0.564 |
| radar | 0.544 | 0.518 | 0.652 | 0.652 |
| csi | 0.647 | 0.630 | 0.715 | 0.714 |

Scene model (self-contained TorchScript):

| modality | window acc | window F1 | minute acc | minute F1 |
|---|---|---|---|---|
| radar | 0.783 | 0.471 | 0.652 | 0.432 |
| fusion | 0.780 | 0.448 | 0.641 | 0.400 |

## 4. Findings

- **Cross-device transfer fails.** test_minutes is a different radar
  device, room and mounting — a much larger shift than the
  placement/day shifts seen earlier. Under the strict
  train+train2 -> validation -> test protocol, no model reaches the
  >90% target; the scene model collapses to the majority class
  (validation accuracy 0.43 < chance — the learned mapping inverts
  across domains).
- **In-domain training works.** The same scene model trained on
  test_minutes itself reaches 0.98-0.99 minute accuracy (5-fold grouped
  CV) and 0.99 through the on-device runtime — viable for a
  fixed-device deployment, but it is not a generalization measure.
- **CSI is receiver/day-specific**: conv CSI features do not transfer;
  amplitude-variance statistics are the robust cue (csi-only minute acc
  0.715 is the best CNN result here — it predicts empty reliably:
  empty recall 1.0).
- **Top-2 minute aggregation** consistently improves over window-level
  decisions on every model.

## 5. Deployment artifacts

`deploy/`: `radar_occupancy_scene.pt`, `fusion_occupancy_scene.pt`
(self-contained preprocessing, identity norm, thoth-model/v1 registry
`*.meta.json`), plus `radar_occupancy_e2.pt`, `fusion_occupancy_e2.pt`
(CNN encoders, runtime-side normalization). Note: the clean-protocol
artifacts above do **not** meet the >90% gate on the Pi night — see
Findings.

## 6. Reproduce

```powershell
python E1/run_all.py     # dataset audit (minutes, no training)
python E2/run_all.py     # occupancy pipeline (~1-2 h CPU)
```
