# E3 — Left/Right Localization Report

Binary left/right classification of the occupant's position from mmWave radar range-Doppler + range-azimuth maps, using only `test_minutes/` (the only source with a left/right ground-truth label). A small CNN is trained end-to-end (model-based approach, not a hand-derived physics estimate) and evaluated with stratified group 5-fold cross-validation at the recording (minute) level, so no fold ever sees windows from a test minute during training.

## Dataset

- Recordings with a left/right label: **84**
- By label: `{'right': 50, 'left': 34}`
- By storage format: `{'npz': 67, 'bin': 17}`
- By label x format: `{'right/npz': 38, 'right/bin': 12, 'left/npz': 29, 'left/bin': 5}`
- Radar frame rate: mean 70987.38 Hz, median 9000.65 Hz (sampled)

## Method

- **Per-frame maps**: Hann-windowed FFT -> range-Doppler (64x64, antenna-averaged) and range-azimuth (32x64, zero-padded angle FFT across the 3 RX antennas), log1p-compressed, resized to 32x32.
- **Windows**: 30 consecutive radar frames (~3 s), non-overlapping, never crossing a recording boundary.
- **Encoder**: per window, a temporal *mean* map and a temporal *max* map are computed per channel (4 maps total) and fed to a small 2D CNN -> 32-d embedding -> linear head -> left/right logit. Unlike the E2 occupancy pipeline (which must transfer across an unseen room and therefore discards absolute levels via a temporal-std-only pooling), here there is a single fixed room/radar placement, so absolute spatial structure (line-of-sight azimuth position, plus any consistent multipath signature) is exactly the discriminative signal and is preserved.
- **Cross-validation**: stratified group 5-fold at the recording level. Within each fold's training recordings, a further 80/20 train/val split is used for early stopping (best macro-F1) and decision-threshold selection; the held-out fold's recordings are never used for either.
- **Balancing**: training windows are subsampled to equal left/right counts; the validation subsample used for early stopping/threshold selection is also class-balanced.
- **Augmentation** (label-preserving only): random +/-2 bin circular shift along the range axis (sensor/distance jitter), random time-reversal of the window. A left-right mirror flip is deliberately NOT used, since it would flip the label itself.
- **Hyperparameter search**: small grid over learning rate, embedding size, and dropout, selected on an internal validation split carved out of fold 0's training recordings only; the resulting config is frozen and reused, unchanged, for every fold's final training run.

- **Selected config**: `{"lr": 0.001, "embed_dim": 32, "dropout": 0.2, "weight_decay": 0.0001, "epochs": 40, "patience": 8, "batch_size": 64}`

## Results

| Metric | Mean | Std | Per-fold |
|---|---|---|---|
| Window-level accuracy | **0.9894** | 0.0090 | ['0.980', '1.000', '1.000', '0.986', '0.980'] |
| Minute-level accuracy | **1.0000** | 0.0000 | ['1.000', '1.000', '1.000', '1.000', '1.000'] |
| Window-level macro-F1 | 0.9889 | | |
| Minute-level macro-F1 | 1.0000 | | |


All 5 folds individually clear the 0.9 accuracy target at both window and minute level; minute-level accuracy is 100% in every fold (majority-voting over ~7-15 windows per minute erases the rare per-window errors).

### Per-fold detail

| Fold | Train recs | Test recs | Test windows | Window acc | Minute acc |
|---|---|---|---|---|---|
| 0 | - | 17 | 204 | 0.980 | 1.000 |
| 1 | - | 17 | 152 | 1.000 | 1.000 |
| 2 | - | 17 | 220 | 1.000 | 1.000 |
| 3 | - | 16 | 222 | 0.986 | 1.000 |
| 4 | - | 15 | 200 | 0.980 | 1.000 |

## Figures

- `figs/kfold_accuracy.png` — per-fold window/minute accuracy vs the 0.9 target.
- `figs/confusion_matrix.png` — aggregated window-level confusion matrix across folds.
- `figs/hp_search.png` — hyperparameter search validation accuracy.
- `figs/example_maps.png` — example time-averaged range-Doppler / range-azimuth maps.

## Caveats

- All left/right data comes from one fixed room and radar placement; the model's spatial signature (whichever combination of line-of-sight azimuth and multipath it learns) is not claimed to transfer to a different room or radar mounting.
- The dataset is class-imbalanced at the recording level ({'right': 50, 'left': 34}); training balances windows, and cross-validation is stratified, to keep this from biasing the reported metrics.
