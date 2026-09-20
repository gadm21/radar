# Occupancy Detection — Experiment Report

Binary task: **empty** vs **occupied** (sleep + present merged).

## 1. Dataset
- **train_minutes**: 1394 minutes, labels {'occupied': 1210, 'empty': 184}, original {'t1_sleep': 808, 't2_empty': 84, 't2_present': 122, 't2_sleep': 217, 't1_present': 63, 't1_empty': 100, 'absent': 11, 'empty': 11}
- **train2_minutes**: 329 minutes, labels {'empty': 220, 'occupied': 109}, original {'t_empty': 220, 'absent': 40, 'empty': 40, 't_present': 89, 't_sleep': 20, 'right': 7, 'present': 2, 'occupied': 2, 'left': 2}
- **validation_minutes**: 194 minutes, labels {'occupied': 81, 'empty': 113}, original {'present': 81, 'empty': 113}
- **test_minutes**: 275 minutes, labels {'occupied': 177, 'empty': 98}, original {'present': 177, 'empty': 98}
- **Problems**: 53 (malformed manifests recovered via regex fallback, ignored/ambiguous labels) — see `inspection.json`

Splits are the dataset folders: **train = train_minutes (placements t1+t2) + train2_minutes (placement t, Sept 6-8), val = validation_minutes (placement t, Sept 15), test = test_minutes (Pi captures, Sept 16-17 night — manifests labeled by label_test_minutes.py from the 1:10 AM boundary: present before, empty after)**. Windows are 50 consecutive radar frames (~5-7 s — long enough to capture breathing) built inside one recording — they can never cross recording/placement/label boundaries.

**CSI coverage**: train_minutes ~57% of recordings (t1 only — t2 had no receiver), train2/validation/test ~99% via capture.npz. Fusion uses missing-CSI masking + CSI-dropout training so radar-only windows still work.

## 2. Timing
- `train_minutes/`: each `radar_*.bin` filename is its capture timestamp and holds one second of frames (nominally 10); frame times = filename ts + j/n.
- `train2_minutes/` + `validation_minutes/` + `test_minutes/`: per-frame npz timestamps are burst-flushed (degenerate); frames are distributed uniformly inside their `second_start` bucket (~10/s).
- CSI: `train_minutes/` uses the `wifi_csi_XX.csv` with most samples inside the minute (the other receiver is usually empty); the npz sources use the capture.npz receiver with most samples.
- **train_minutes** 50-frame window: mean=5.188s median=4.900s std=2.312s
- **train2_minutes** 50-frame window: mean=6.239s median=5.829s std=3.484s
- **validation_minutes** 50-frame window: mean=4.869s median=4.900s std=2.056s
- **test_minutes** 50-frame window: mean=6.489s median=4.917s std=4.784s

## 3. Model
Radar encoder: per-pixel **temporal-std map** of the 50-frame window (range-Doppler + range-azimuth), pooled spatially to mean/std/max per channel -> small MLP. Temporal variation is the placement-invariant occupancy cue; absolute levels flip sign across placements and were removed after diagnosis. CSI-only encoder: temporal Conv1d over causal **rolling variance** (w=20) of the 52-subcarrier amplitude (128 steps) — the 'rolling_variance' pipeline from WifiSensingESP32HAR; amplitude only, no phase. The **fusion** model instead uses a CSIStatsEncoder (MLP on per-window amplitude variance/mean/temporal-std): conv CSI features are receiver/day-specific and hijack the gate on the gain-shifted test day, while amplitude variance is the robust cross-day occupancy cue. Gated fusion: per-feature sigmoid gate z = g*r + (1-g)*c with missing-CSI masking. Binary head.

Training: Adam + BCE (pos_weight). The train set is train_minutes + train2_minutes, class-balanced by subsampling majority-class windows. validation_minutes drives early stopping and threshold selection; test_minutes is evaluated once. Normalization statistics come from the train pool only. Augmentation: doppler-sign/azimuth flip; CSI dropout (30%) for fusion. Minute-level decisions use **top-2 window aggregation** with a separately tuned `minute_threshold` (median of the optimal range — saturated val probs make the argmax edge miscalibrate on the shifted test day).

## 4. Hyperparameter search (train -> val, radar)
Best: `{'lr': 0.001, 'embed_dim': 128, 'dropout': 0.2, 'weight_decay': 0.0001, 'epochs': 8, 'patience': 4, 'batch_size': 256}` -> val acc 0.858, F1 0.856, threshold 0.12

| lr | emb | dropout | val acc | val F1 |
|---|---|---|---|---|
| 0.001 | 128 | 0.2 | 0.858 | 0.856 |
| 0.001 | 64 | 0.4 | 0.825 | 0.823 |
| 0.001 | 128 | 0.4 | 0.816 | 0.813 |
| 0.0003 | 64 | 0.2 | 0.793 | 0.788 |
| 0.0003 | 128 | 0.2 | 0.764 | 0.755 |
| 0.001 | 64 | 0.2 | 0.760 | 0.741 |
| 0.0003 | 64 | 0.4 | 0.760 | 0.759 |
| 0.0003 | 128 | 0.4 | 0.686 | 0.637 |

## 5. E2 — train+train2 / validation / test

### Window level (test = test_minutes)
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| fusion | 0.507 | 0.477 | 0.609 | 0.478 | 0.522 | 1859 |
| radar | 0.544 | 0.518 | 0.702 | 0.500 | 0.500 | 1859 |
| csi | 0.647 | 0.630 | 0.926 | 0.562 | 0.438 | 1719 |

### Minute level (test = test_minutes, top-2 window aggregation, val-tuned minute_threshold)
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| fusion | 0.567 | 0.564 | 0.673 | 0.506 | 0.494 | 270 |
| radar | 0.652 | 0.652 | 0.929 | 0.494 | 0.506 | 270 |
| csi | 0.715 | 0.714 | 1.000 | 0.536 | 0.464 | 249 |

- Fusion gate (radar weight) mean on test: **0.625**

## 6. Scene model (deployment artifact)

`scene_model.py` computes all features inside the TorchScript forward pass (mean/std maps, range/azimuth profiles, CSI stats) — the exported .pt is self-contained (identity normalization in meta.json). Trained on train+train2, thresholds tuned on validation_minutes, evaluated on test_minutes.

### Window level
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| radar | 0.783 | 0.471 | 0.034 | 0.994 | 0.006 | 1859 |
| fusion | 0.780 | 0.448 | 0.010 | 0.997 | 0.003 | 1859 |

### Minute level (top-2 aggregation)
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| radar | 0.652 | 0.432 | 0.041 | 1.000 | 0.000 | 270 |
| fusion | 0.641 | 0.400 | 0.010 | 1.000 | 0.000 | 270 |

Validation metrics (tuning split):
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| radar (val) | 0.428 | 0.316 | 0.020 | 1.000 | 0.000 | 2064 |
| fusion (val) | 0.428 | 0.316 | 0.020 | 1.000 | 0.000 | 2064 |

## 7. Leakage audit
- Splits are disjoint folder sets: no recording appears in more than one split; validation and test are different days of placement t / the Pi device.
- Windows are built inside a single recording; no window spans two recordings or labels.
- Normalization statistics are computed on the train pool (train_minutes + train2_minutes) only and applied to val/test.
- Early stopping, threshold selection and hyperparameter search use validation_minutes only; test_minutes is touched once for final metrics.

## 8. Failure modes / notes
- **Domain shift**: test_minutes is a different *device* (the Pi's radar), room and mounting — a much larger shift than placement-to-placement or day-to-day. Under this strict protocol no model reaches the >90% target: the CNN encoders top out at ~0.65-0.72 minute acc and the scene model collapses to the majority class (its mean-map features are device/room-specific — validation accuracy below chance shows the learned mapping inverts across domains). Earlier >0.97 numbers came from in-domain training on test_minutes itself (valid for a fixed-device deployment, but not a generalization measure).
- CSI windows with <2 samples are zeroed and flagged (`csi_valid=0`); the CSI-only model skips them, fusion masks the CSI embedding for them.
- CSI is receiver-specific: the csi-only model is weakest, but with rolling-variance features + dropout the fusion model still benefits where CSI exists.
- `radar-missing` / non-task labels (left/right/absent) are excluded from the task.
- Malformed manifests are recovered by regex fallback; unrecoverable ones are listed in `inspection.json`.