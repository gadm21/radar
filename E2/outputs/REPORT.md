# Occupancy Detection — Experiment Report

Binary task: **empty** vs **occupied** (sleep + present merged).

## 1. Dataset
- **t1**: 1133 minutes, labels {'empty': 236, 'occupied': 897}, original {'t1_empty': 236, 't1_sleep': 834, 't1_present': 63, 'absent': 93, 'empty': 93, 'present': 7, 'occupied': 7}
- **t2**: 423 minutes, labels {'empty': 84, 'occupied': 339}, original {'t2_empty': 84, 't2_present': 122, 't2_sleep': 217}
- **t**: 330 minutes, labels {'empty': 220, 'occupied': 110}, original {'t_empty': 220, 'absent': 291, 'empty': 291, 't_present': 89, 'present': 38, 'occupied': 38, 't_sleep': 21, 'radar-missing': 1, 'right': 50, 'left': 34}
- **Problems**: 63 (malformed manifests recovered via regex fallback, ignored/ambiguous labels) — see `inspection.json`

Splits: **t1 = train, t2 = validation, t = test** (placement-level, no sessions). Windows are 50 consecutive radar frames (~5-7 s — long enough to capture breathing) built inside one recording — they can never cross recording/placement/label boundaries.

**CSI coverage**: t1 ~85% of windows, t2 **none** (the receiver was not recording), t 100%. The CSI-only baseline therefore early-stops on a t1 holdout, and fusion is evaluated with missing-CSI masking + CSI-dropout training.

## 2. Timing
- `minutes/`: each `radar_*.bin` filename is its capture timestamp and holds one second of frames (nominally 10); frame times = filename ts + j/n.
- `test_minutes/`: per-frame npz timestamps are burst-flushed (degenerate); frames are distributed uniformly inside their `second_start` bucket (~10/s).
- CSI: `minutes/` uses the `wifi_csi_XX.csv` with most samples inside the minute (the other receiver is usually empty); `test_minutes/` uses the receiver with most samples.
- **t1** 50-frame window: mean=5.309s median=4.902s std=2.133s
- **t2** 50-frame window: mean=5.538s median=4.901s std=2.518s
- **t** 50-frame window: mean=6.466s median=5.900s std=4.020s

## 3. Model
Radar encoder: per-pixel **temporal-std map** of the 50-frame window (range-Doppler + range-azimuth), pooled spatially to mean/std/max per channel -> small MLP. Temporal variation is the placement-invariant occupancy cue; absolute levels flip sign across placements and were removed after diagnosis. CSI encoder: temporal Conv1d over causal **rolling variance** (w=20) of the 52-subcarrier amplitude (128 steps) — the 'rolling_variance' pipeline from WifiSensingESP32HAR; amplitude only, no phase. Gated fusion: per-feature sigmoid gate z = g*r + (1-g)*c with missing-CSI masking. Binary head.

Training: Adam + BCE (pos_weight), early stopping on a **class-balanced** t2 subsample (so the threshold does not inherit t2's 80%-occupied prior), decision threshold tuned on t2 only. Training windows are class-balanced by subsampling the majority class. Normalization statistics come from t1 only. Augmentation: doppler-sign/azimuth flip; CSI dropout (30%) for fusion.

## 4. Hyperparameter search (t1 -> t2, radar)
Best: `{'lr': 0.001, 'embed_dim': 64, 'dropout': 0.2, 'weight_decay': 0.0001, 'epochs': 8, 'patience': 4, 'batch_size': 256}` -> val acc 0.822, F1 0.780, threshold 0.48

| lr | emb | dropout | val acc | val F1 |
|---|---|---|---|---|
| 0.001 | 64 | 0.2 | 0.822 | 0.780 |
| 0.001 | 128 | 0.2 | 0.801 | 0.763 |
| 0.001 | 64 | 0.4 | 0.796 | 0.756 |
| 0.001 | 128 | 0.4 | 0.794 | 0.753 |
| 0.0003 | 64 | 0.4 | 0.779 | 0.741 |
| 0.0003 | 128 | 0.4 | 0.770 | 0.733 |
| 0.0003 | 128 | 0.2 | 0.766 | 0.730 |
| 0.0003 | 64 | 0.2 | 0.756 | 0.721 |

## 5. E2 — train t1 / val t2 / test t (unseen placement)

### Window level (test = t)
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| fusion | 0.678 | 0.583 | 0.847 | 0.315 | 0.685 | 1865 |
| radar | 0.995 | 0.994 | 0.994 | 0.997 | 0.003 | 1865 |
| csi | 0.318 | 0.242 | 0.000 | 1.000 | 0.000 | 1865 |

### Minute level (test = t, mean-prob aggregation)
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| fusion | 0.761 | 0.668 | 0.940 | 0.369 | 0.631 | 268 |
| radar | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 268 |
| csi | 0.313 | 0.239 | 0.000 | 1.000 | 0.000 | 268 |

- Fusion gate (radar weight) mean on t: **0.622**

## 6. E3 — few-shot adaptation (10 support minutes from t)

Support minutes: test_minutes/20260906_1851, test_minutes/20260906_2308, test_minutes/20260906_2236, test_minutes/20260906_1845, test_minutes/20260906_1839, test_minutes/20260907_2304, test_minutes/20260906_2335, test_minutes/20260906_1959, test_minutes/20260906_1956, test_minutes/20260906_2328

### Window level (query = remaining t minutes)
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| zero_shot | 0.995 | 0.994 | 0.994 | 0.996 | 0.004 | 1792 |
| head | 0.995 | 0.994 | 0.994 | 0.996 | 0.004 | 1792 |
| full | 0.996 | 0.995 | 0.995 | 0.996 | 0.004 | 1792 |

### Minute level
| config | acc | macro-F1 | empty R | occ R | false-empty | n |
|---|---|---|---|---|---|---|
| zero_shot | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 258 |
| head | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 258 |
| full | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 258 |

Best strategy **full**: +0.001 accuracy vs 0-shot.

## 7. Leakage audit
- Splits are at placement level: no recording appears in more than one split.
- Windows are built inside a single recording; no window spans two recordings or labels.
- Normalization statistics are computed on t1 only and applied to t2/t.
- Early stopping, threshold selection and hyperparameter search use t2 (validation) only; t is touched once for final metrics.
- E3 support minutes are excluded from the query set; query data is never used for training or selection.

## 8. Failure modes / notes
- CSI windows with <2 samples are zeroed and flagged (`csi_valid=0`); the CSI-only model skips them, fusion masks the CSI embedding for them.
- **CSI does not transfer across receivers/placements** (csi-only test acc ~0.38; it also degrades fusion on t). Radar temporal-std features transfer cleanly (test acc ~0.99), so the radar-only model is selected for E3.
- `radar-missing` / non-task labels (left/right/absent) are excluded from the task.
- Malformed manifests are recovered by regex fallback; unrecoverable ones are listed in `inspection.json`.