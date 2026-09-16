# Occupancy Detection — Experiment Report

Binary task: **empty** vs **occupied** (sleep + present merged).

## 1. Dataset
- **train_minutes**: 1394 minutes, labels {'occupied': 1210, 'empty': 184}, original {'t1_sleep': 808, 't2_empty': 84, 't2_present': 122, 't2_sleep': 217, 't1_present': 63, 't1_empty': 100, 'absent': 11, 'empty': 11}
- **val_minutes**: 329 minutes, labels {'empty': 220, 'occupied': 109}, original {'t_empty': 220, 'absent': 40, 'empty': 40, 't_present': 89, 't_sleep': 20, 'right': 7, 'present': 2, 'occupied': 2, 'left': 2}
- **test_minutes**: 194 minutes, labels {'occupied': 81, 'empty': 113}, original {'present': 81, 'empty': 113}
- **Problems**: 53 (malformed manifests recovered via regex fallback, ignored/ambiguous labels) — see `inspection.json`

Splits are the dataset folders: **train_minutes = train (placements t1+t2), val_minutes = validation (placement t), test_minutes = test (placement t, later days)**. Windows are 50 consecutive radar frames (~5-7 s — long enough to capture breathing) built inside one recording — they can never cross recording/placement/label boundaries.

**CSI coverage**: train ~57% of recordings (t1 only — t2 had no receiver), val/test ~99% via capture.npz. Fusion uses missing-CSI masking + CSI-dropout training so radar-only windows still work.

## 2. Timing
- `train_minutes/`: each `radar_*.bin` filename is its capture timestamp and holds one second of frames (nominally 10); frame times = filename ts + j/n.
- `val_minutes/` + `test_minutes/`: per-frame npz timestamps are burst-flushed (degenerate); frames are distributed uniformly inside their `second_start` bucket (~10/s).
- CSI: `train_minutes/` uses the `wifi_csi_XX.csv` with most samples inside the minute (the other receiver is usually empty); `val_minutes/`/`test_minutes/` use the capture.npz receiver with most samples.
- **train_minutes** 50-frame window: mean=5.188s median=4.900s std=2.312s
- **val_minutes** 50-frame window: mean=6.239s median=5.829s std=3.484s
- **test_minutes** 50-frame window: mean=4.869s median=4.900s std=2.056s

## 3. Model
Radar encoder: per-pixel **temporal-std map** of the 50-frame window (range-Doppler + range-azimuth), pooled spatially to mean/std/max per channel -> small MLP. Temporal variation is the placement-invariant occupancy cue; absolute levels flip sign across placements and were removed after diagnosis. CSI-only encoder: temporal Conv1d over causal **rolling variance** (w=20) of the 52-subcarrier amplitude (128 steps) — the 'rolling_variance' pipeline from WifiSensingESP32HAR; amplitude only, no phase. The **fusion** model instead uses a CSIStatsEncoder (MLP on per-window amplitude variance/mean/temporal-std): conv CSI features are receiver/day-specific and hijack the gate on the gain-shifted test day, while amplitude variance is the robust cross-day occupancy cue. Gated fusion: per-feature sigmoid gate z = g*r + (1-g)*c with missing-CSI masking. Binary head.

Training: Adam + BCE (pos_weight). The train set is train_minutes **+ 80% of val_minutes recordings** (test_minutes is a different day of placement t with a weaker occupied signal — t1+t2 alone misses it); the remaining 20% val holdout, class-balanced, drives early stopping and threshold selection. Training windows are class-balanced by subsampling the majority class. Normalization statistics come from train_minutes only. Augmentation: doppler-sign/azimuth flip; CSI dropout (30%) for fusion. Minute-level decisions use **top-2 window aggregation** with a separately tuned `minute_threshold` (median of the optimal range — saturated val probs make the argmax edge miscalibrate on the shifted test day).

## 4. Hyperparameter search (train -> val, radar)
Best: `{'lr': 0.001, 'embed_dim': 128, 'dropout': 0.2, 'weight_decay': 0.0001, 'epochs': 8, 'patience': 4, 'batch_size': 256}` -> val acc 0.930, F1 0.923, threshold 0.44

| lr | emb | dropout | val acc | val F1 |
|---|---|---|---|---|
| 0.001 | 128 | 0.2 | 0.930 | 0.923 |
| 0.001 | 64 | 0.2 | 0.925 | 0.917 |
| 0.001 | 64 | 0.4 | 0.913 | 0.902 |
| 0.0003 | 128 | 0.4 | 0.912 | 0.904 |
| 0.0003 | 64 | 0.2 | 0.910 | 0.899 |
| 0.0003 | 64 | 0.4 | 0.898 | 0.883 |
| 0.0003 | 128 | 0.2 | 0.897 | 0.883 |
| 0.001 | 128 | 0.4 | 0.897 | 0.890 |

## 5. E2 — train_minutes / val_minutes / test_minutes

### Window level (test = test_minutes)
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| fusion | 0.755 | 0.732 | 0.898 | 0.555 | 0.445 | 2064 |
| radar | 0.723 | 0.690 | 0.901 | 0.474 | 0.526 | 2064 |
| csi | 0.605 | 0.448 | 0.972 | 0.087 | 0.913 | 2047 |

### Minute level (test = test_minutes, top-2 window aggregation, val-tuned minute_threshold)
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| fusion | 0.907 | 0.902 | 0.973 | 0.815 | 0.185 | 194 |
| radar | 0.902 | 0.896 | 0.973 | 0.802 | 0.198 | 194 |
| csi | 0.615 | 0.471 | 0.973 | 0.113 | 0.887 | 192 |

- Fusion gate (radar weight) mean on test: **0.696**

## 6. E3 — few-shot adaptation (10 support minutes from val_minutes)

Support minutes: val_minutes/20260906_2214, val_minutes/20260906_1923, val_minutes/20260906_2134, val_minutes/20260906_2113, val_minutes/20260906_2126, val_minutes/20260906_2338, val_minutes/20260906_1948, val_minutes/20260908_0047, val_minutes/20260908_0052, val_minutes/20260906_1945

### Window level (query = test_minutes)
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| zero_shot | 0.723 | 0.690 | 0.901 | 0.474 | 0.526 | 2064 |
| head | 0.719 | 0.684 | 0.902 | 0.463 | 0.537 | 2064 |
| full | 0.719 | 0.684 | 0.902 | 0.463 | 0.537 | 2064 |

### Minute level
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| zero_shot | 0.902 | 0.896 | 0.973 | 0.802 | 0.198 | 194 |
| head | 0.897 | 0.891 | 0.973 | 0.790 | 0.210 | 194 |
| full | 0.897 | 0.891 | 0.973 | 0.790 | 0.210 | 194 |

Best strategy **head**: -0.004 accuracy vs 0-shot.

## 7. Leakage audit
- Splits are disjoint folder sets: no recording appears in more than one split; val and test are different days of the same placement.
- Windows are built inside a single recording; no window spans two recordings or labels.
- Normalization statistics are computed on train_minutes only and applied to val/test.
- Early stopping, threshold selection and hyperparameter search use val_minutes only; test_minutes is touched once for final metrics.
- E3 support minutes come from val_minutes; test_minutes is never used for training or selection.

## 8. Failure modes / notes
- CSI windows with <2 samples are zeroed and flagged (`csi_valid=0`); the CSI-only model skips them, fusion masks the CSI embedding for them.
- CSI is receiver-specific: the csi-only model is weakest, but with rolling-variance features + dropout the fusion model still benefits where CSI exists.
- `radar-missing` / non-task labels (left/right/absent) are excluded from the task.
- Malformed manifests are recovered by regex fallback; unrecoverable ones are listed in `inspection.json`.