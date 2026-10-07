# E2 model comparison (E2/model_compare.py)

Splits: train=legacy train_minutes (274 min, wall2) | val=validation_minutes (192 min, wall3) | test=multilink test_minutes (4×1 h) | calib=calibration_minutes (t1_sleep, 40 min cached, first 10 used)

Labels: sleep/present/t1_sleep→occupied, empty→empty.
Feature views: radar = physical radar + radar-PC + fft_snr; csi = physical csi + csi-PC; joint = 6-d joint-PC stats + cluster geometry; all = union (shared cols only).

## Key findings
- Rank-order signal survives every domain shift (top single descriptor `ra_mean`: AUC 0.72/0.95/1.00 on train/val/test), but absolute feature scales drift across sessions — raw accuracy collapses to ~0.50 while AUC stays 0.85-0.94.
- `_norm` = unsupervised per-set re-standardisation (unlabelled eval windows re-scaled by their own mean/std before the train scaler). It recovers most of the signal and is the honest 'unsupervised calibration' number.
- CSI-view is the most transferable raw view (rf/csi val 0.88 / test 0.82); radar-only is scale-fragile.
- cal_sup: several models reach 1.00 occupied recall on a 10-minute t1_sleep block — a one-class calibration visit works.

## Best per model class (test=multilink)
### ml
| model | set | bal | acc | auc | rec_e | rec_o |
|---|---|---|---|---|---|---|
| svm/csi | testml_norm | 0.865 | 0.857 | 0.933 | 0.81 | 0.92 |
| svm/joint | testml_norm | 0.865 | 0.870 | 0.940 | 0.90 | 0.83 |
| gbm/csi | testml_norm | 0.830 | 0.817 | 0.909 | 0.75 | 0.91 |
| rf/csi | testml_norm | 0.823 | 0.816 | 0.882 | 0.78 | 0.87 |
| rf/csi | testml | 0.822 | 0.834 | 0.901 | 0.90 | 0.74 |

### dl
| model | set | bal | acc | auc | rec_e | rec_o |
|---|---|---|---|---|---|---|
| mlp/joint | testml_norm | 0.745 | 0.757 | 0.775 | 0.82 | 0.67 |
| mlp/csi | testml_norm | 0.742 | 0.740 | 0.823 | 0.73 | 0.75 |
| mlp/sel | testml_norm | 0.736 | 0.721 | 0.827 | 0.64 | 0.83 |
| mlp/sel | testml | 0.599 | 0.650 | 0.481 | 0.92 | 0.27 |
| mlp/all | testml_norm | 0.541 | 0.542 | 0.564 | 0.55 | 0.53 |

### rule
| model | set | bal | acc | auc | rec_e | rec_o |
|---|---|---|---|---|---|---|
| knn/csi | testml_norm | 0.697 | 0.731 | 0.787 | 0.91 | 0.48 |
| knn/sel | testml_norm | 0.646 | 0.689 | 0.648 | 0.92 | 0.38 |
| knn/all | testml_norm | 0.588 | 0.651 | 0.602 | 0.98 | 0.19 |
| clustervote/joint | testml | 0.561 | 0.578 | 0.561 | 0.67 | 0.45 |
| knn/joint | testml_norm | 0.561 | 0.604 | 0.679 | 0.83 | 0.29 |

### emb
| model | set | bal | acc | auc | rec_e | rec_o |
|---|---|---|---|---|---|---|
| emb_mlp/all | testml | 0.538 | 0.611 | 0.637 | 1.00 | 0.08 |
| emb_gbm/all | testml | 0.519 | 0.590 | 0.260 | 0.96 | 0.08 |
| emb_logreg/all | testml | 0.500 | 0.579 | 0.322 | 1.00 | 0.00 |
| emb_svm/all | testml | 0.500 | 0.421 | 0.740 | 0.00 | 1.00 |
| emb_rf/all | testml | 0.406 | 0.464 | 0.163 | 0.77 | 0.04 |

### fuse
| model | set | bal | acc | auc | rec_e | rec_o |
|---|---|---|---|---|---|---|
| fusw_gbm/prob | testml_norm | 0.834 | 0.820 | 0.857 | 0.74 | 0.92 |
| fusw_rf/prob | testml_norm | 0.809 | 0.795 | 0.839 | 0.72 | 0.89 |
| fusemean_gbm/prob | testml_norm | 0.786 | 0.777 | 0.831 | 0.73 | 0.85 |
| fusemean_rf/prob | testml_norm | 0.750 | 0.725 | 0.769 | 0.59 | 0.91 |
| fusw_gbm/prob | testml | 0.737 | 0.709 | 0.785 | 0.56 | 0.91 |

### jev
| model | set | bal | acc | auc | rec_e | rec_o |
|---|---|---|---|---|---|---|
| jev | testml | 0.500 | 0.623 | 0.468 | 1.00 | 0.00 |
| jev_emb | testml | 0.066 | 0.053 | 0.023 | 0.02 | 0.12 |

## Full results (val + test + calibration)

| model | set | bal | acc | f1 | auc | rec_e | rec_o |
|---|---|---|---|---|---|---|---|
| svm/radar | cal_sup | 1.000 | 1.000 | 1.000 | nan | nan | 1.00 |
| svm/joint | cal_sup | 1.000 | 1.000 | 1.000 | nan | nan | 1.00 |
| centroid_cal/all | cal_sup | 1.000 | 1.000 | 1.000 | nan | nan | 1.00 |
| fusemax_mlp/prob | cal_sup | 1.000 | 1.000 | 1.000 | nan | nan | 1.00 |
| mlp/joint | cal_sup | 0.996 | 0.996 | 0.998 | nan | nan | 1.00 |
| knn/sel | cal_sup | 0.950 | 0.950 | 0.974 | nan | nan | 0.95 |
| svm/csi | cal_sup | 0.924 | 0.924 | 0.961 | nan | nan | 0.92 |
| fusemean_mlp/prob | cal_sup | 0.874 | 0.874 | 0.933 | nan | nan | 0.87 |
| fusw_mlp/prob | cal_sup | 0.849 | 0.849 | 0.918 | nan | nan | 0.85 |
| mlp/radar | cal_sup | 0.769 | 0.769 | 0.869 | nan | nan | 0.77 |
| fusemax_rf/prob | cal_sup | 0.723 | 0.723 | 0.839 | nan | nan | 0.72 |
| rf/radar | cal_sup | 0.693 | 0.693 | 0.819 | nan | nan | 0.69 |
| mlp/csi | cal_sup | 0.454 | 0.454 | 0.624 | nan | nan | 0.45 |
| threshcal/all | cal_sup | 0.454 | 0.454 | 0.624 | nan | nan | 0.45 |
| emb_svm/all | cal_sup | 0.441 | 0.441 | 0.612 | nan | nan | 0.44 |
| jev | cal_sup | 0.435 | 0.435 | 0.606 | nan | nan | 0.43 |
| knn/csi | cal_sup | 0.399 | 0.399 | 0.571 | nan | nan | 0.40 |
| knn/joint | cal_sup | 0.214 | 0.214 | 0.353 | nan | nan | 0.21 |
| fusemax_gbm/prob | cal_sup | 0.185 | 0.185 | 0.312 | nan | nan | 0.18 |
| svm/all | cal_sup | 0.122 | 0.122 | 0.217 | nan | nan | 0.12 |
| gbm/joint | cal_sup | 0.109 | 0.109 | 0.197 | nan | nan | 0.11 |
| gbm/csi | cal_sup | 0.105 | 0.105 | 0.190 | nan | nan | 0.11 |
| fusw_gbm/prob | cal_sup | 0.092 | 0.092 | 0.169 | nan | nan | 0.09 |
| rf/joint | cal_sup | 0.084 | 0.084 | 0.155 | nan | nan | 0.08 |
| fusemean_gbm/prob | cal_sup | 0.063 | 0.063 | 0.119 | nan | nan | 0.06 |
| svm/sel | cal_sup | 0.055 | 0.055 | 0.104 | nan | nan | 0.05 |
| fusemean_rf/prob | cal_sup | 0.042 | 0.042 | 0.081 | nan | nan | 0.04 |
| rf/all | cal_sup | 0.025 | 0.025 | 0.049 | nan | nan | 0.03 |
| fusw_rf/prob | cal_sup | 0.025 | 0.025 | 0.049 | nan | nan | 0.03 |
| rf/csi | cal_sup | 0.021 | 0.021 | 0.041 | nan | nan | 0.02 |
| gbm/radar | cal_sup | 0.017 | 0.017 | 0.033 | nan | nan | 0.02 |
| emb_rf/all | cal_sup | 0.013 | 0.013 | 0.025 | nan | nan | 0.01 |
| logreg/csi | cal_sup | 0.008 | 0.008 | 0.017 | nan | nan | 0.01 |
| emb_gbm/all | cal_sup | 0.008 | 0.008 | 0.017 | nan | nan | 0.01 |
| logreg/joint | cal_sup | 0.004 | 0.004 | 0.008 | nan | nan | 0.00 |
| rf/sel | cal_sup | 0.004 | 0.004 | 0.008 | nan | nan | 0.00 |
| logreg/radar | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| stump/radar | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| knn/radar | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| stump/csi | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| stump/joint | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| logreg/all | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| gbm/all | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| mlp/all | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| stump/all | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| knn/all | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| logreg/sel | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| gbm/sel | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| mlp/sel | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| stump/sel | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| clustervote/joint | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| emb_logreg/all | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| emb_mlp/all | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| seqcnn/jointseq | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| jev_emb | cal_sup | 0.000 | 0.000 | 0.000 | nan | nan | 0.00 |
| gbm/csi | cal_unsup | 0.750 | 0.758 | 0.808 | 0.927 | 0.51 | 0.99 |
| rf/csi | cal_unsup | 0.737 | 0.744 | 0.800 | 0.924 | 0.48 | 0.99 |
| stump/csi | cal_unsup | 0.700 | 0.709 | 0.780 | 0.930 | 0.40 | 1.00 |
| stump/joint | cal_unsup | 0.700 | 0.709 | 0.780 | 0.930 | 0.40 | 1.00 |
| fusw_gbm/prob | cal_unsup | 0.678 | 0.687 | 0.766 | 0.927 | 0.36 | 0.99 |
| jev_emb | cal_unsup | 0.607 | 0.500 | 0.593 | 0.862 | 0.21 | 1.00 |
| jev | cal_unsup | 0.562 | 0.545 | 0.500 | 0.679 | 0.50 | 0.62 |
| mlp/all | cal_unsup | 0.526 | 0.533 | 0.629 | 0.532 | 0.28 | 0.77 |
| mlp/csi | cal_unsup | 0.505 | 0.520 | 0.682 | 0.523 | 0.01 | 1.00 |
| knn/joint | cal_unsup | 0.505 | 0.520 | 0.682 | 0.505 | 0.01 | 1.00 |
| fusemean_mlp/prob | cal_unsup | 0.505 | 0.520 | 0.682 | 0.525 | 0.01 | 1.00 |
| fusw_mlp/prob | cal_unsup | 0.505 | 0.520 | 0.682 | 0.516 | 0.01 | 1.00 |
| mlp/radar | cal_unsup | 0.503 | 0.489 | 0.079 | 0.539 | 0.96 | 0.04 |
| emb_mlp/all | cal_unsup | 0.503 | 0.489 | 0.079 | 0.503 | 0.96 | 0.04 |
| logreg/radar | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| svm/radar | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| rf/radar | cal_unsup | 0.500 | 0.515 | 0.680 | 0.580 | 0.00 | 1.00 |
| gbm/radar | cal_unsup | 0.500 | 0.515 | 0.680 | 0.498 | 0.00 | 1.00 |
| stump/radar | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| knn/radar | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| svm/csi | cal_unsup | 0.500 | 0.515 | 0.680 | 0.002 | 0.00 | 1.00 |
| knn/csi | cal_unsup | 0.500 | 0.515 | 0.680 | 0.505 | 0.00 | 1.00 |
| logreg/joint | cal_unsup | 0.500 | 0.515 | 0.680 | 0.906 | 0.00 | 1.00 |
| svm/joint | cal_unsup | 0.500 | 0.515 | 0.680 | 0.035 | 0.00 | 1.00 |
| rf/joint | cal_unsup | 0.500 | 0.515 | 0.680 | 0.909 | 0.00 | 1.00 |
| gbm/joint | cal_unsup | 0.500 | 0.515 | 0.680 | 0.899 | 0.00 | 1.00 |
| mlp/joint | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| logreg/all | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| svm/all | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| rf/all | cal_unsup | 0.500 | 0.515 | 0.680 | 0.921 | 0.00 | 1.00 |
| gbm/all | cal_unsup | 0.500 | 0.515 | 0.680 | 0.892 | 0.00 | 1.00 |
| stump/all | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| knn/all | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| logreg/sel | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| svm/sel | cal_unsup | 0.500 | 0.515 | 0.680 | 0.495 | 0.00 | 1.00 |
| rf/sel | cal_unsup | 0.500 | 0.515 | 0.680 | 0.818 | 0.00 | 1.00 |
| gbm/sel | cal_unsup | 0.500 | 0.515 | 0.680 | 0.831 | 0.00 | 1.00 |
| mlp/sel | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| stump/sel | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| knn/sel | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| centroid_cal/all | cal_unsup | 0.500 | 0.485 | 0.000 | 0.422 | 1.00 | 0.00 |
| threshcal/all | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| clustervote/joint | cal_unsup | 0.500 | 0.485 | 0.000 | 0.500 | 1.00 | 0.00 |
| emb_logreg/all | cal_unsup | 0.500 | 0.485 | 0.000 | 0.041 | 1.00 | 0.00 |
| emb_svm/all | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| emb_rf/all | cal_unsup | 0.500 | 0.515 | 0.680 | 0.660 | 0.00 | 1.00 |
| emb_gbm/all | cal_unsup | 0.500 | 0.515 | 0.680 | 0.505 | 0.00 | 1.00 |
| fusemean_rf/prob | cal_unsup | 0.500 | 0.515 | 0.680 | 0.922 | 0.00 | 1.00 |
| fusemax_rf/prob | cal_unsup | 0.500 | 0.515 | 0.680 | 0.912 | 0.00 | 1.00 |
| fusw_rf/prob | cal_unsup | 0.500 | 0.515 | 0.680 | 0.924 | 0.00 | 1.00 |
| fusemean_gbm/prob | cal_unsup | 0.500 | 0.515 | 0.680 | 0.927 | 0.00 | 1.00 |
| fusemax_gbm/prob | cal_unsup | 0.500 | 0.515 | 0.680 | 0.841 | 0.00 | 1.00 |
| fusemax_mlp/prob | cal_unsup | 0.500 | 0.515 | 0.680 | 0.500 | 0.00 | 1.00 |
| seqcnn/jointseq | cal_unsup | 0.497 | 0.511 | 0.671 | 0.274 | 0.03 | 0.97 |
| logreg/csi | cal_unsup | 0.495 | 0.480 | 0.000 | 0.170 | 0.99 | 0.00 |
| rf/csi | testml | 0.822 | 0.834 | 0.790 | 0.901 | 0.90 | 0.74 |
| gbm/csi | testml | 0.786 | 0.778 | 0.759 | 0.871 | 0.74 | 0.83 |
| fusw_gbm/prob | testml | 0.737 | 0.709 | 0.724 | 0.785 | 0.56 | 0.91 |
| fusw_rf/prob | testml | 0.715 | 0.695 | 0.698 | 0.761 | 0.59 | 0.84 |
| svm/sel | testml | 0.685 | 0.650 | 0.684 | 0.617 | 0.47 | 0.90 |
| mlp/sel | testml | 0.599 | 0.650 | 0.396 | 0.481 | 0.92 | 0.27 |
| clustervote/joint | testml | 0.561 | 0.578 | 0.475 | 0.561 | 0.67 | 0.45 |
| stump/csi | testml | 0.543 | 0.580 | 0.381 | 0.367 | 0.78 | 0.31 |
| stump/joint | testml | 0.543 | 0.580 | 0.381 | 0.367 | 0.78 | 0.31 |
| mlp/joint | testml | 0.539 | 0.560 | 0.439 | 0.552 | 0.67 | 0.41 |
| emb_mlp/all | testml | 0.538 | 0.611 | 0.141 | 0.637 | 1.00 | 0.08 |
| svm/joint | testml | 0.520 | 0.444 | 0.601 | 0.797 | 0.05 | 0.99 |
| emb_gbm/all | testml | 0.519 | 0.590 | 0.134 | 0.260 | 0.96 | 0.08 |
| knn/all | testml | 0.506 | 0.584 | 0.023 | 0.696 | 1.00 | 0.01 |
| logreg/csi | testml | 0.500 | 0.421 | 0.592 | 0.729 | 0.00 | 1.00 |
| svm/csi | testml | 0.500 | 0.421 | 0.592 | 0.500 | 0.00 | 1.00 |
| mlp/csi | testml | 0.500 | 0.421 | 0.592 | 0.500 | 0.00 | 1.00 |
| knn/csi | testml | 0.500 | 0.579 | 0.000 | 0.365 | 1.00 | 0.00 |
| logreg/all | testml | 0.500 | 0.421 | 0.592 | 0.419 | 0.00 | 1.00 |
| svm/all | testml | 0.500 | 0.421 | 0.592 | 0.491 | 0.00 | 1.00 |
| mlp/all | testml | 0.500 | 0.421 | 0.592 | 0.500 | 0.00 | 1.00 |
| centroid_cal/all | testml | 0.500 | 0.421 | 0.592 | 0.321 | 0.00 | 1.00 |
| emb_logreg/all | testml | 0.500 | 0.579 | 0.000 | 0.322 | 1.00 | 0.00 |
| emb_svm/all | testml | 0.500 | 0.421 | 0.592 | 0.740 | 0.00 | 1.00 |
| fusemax_mlp/prob | testml | 0.500 | 0.421 | 0.592 | 0.500 | 0.00 | 1.00 |
| jev | testml | 0.500 | 0.623 | 0.000 | 0.468 | 1.00 | 0.00 |
| gbm/all | testml | 0.500 | 0.420 | 0.592 | 0.185 | 0.00 | 1.00 |
| fusemax_gbm/prob | testml | 0.500 | 0.420 | 0.592 | 0.080 | 0.00 | 1.00 |
| fusemax_rf/prob | testml | 0.499 | 0.420 | 0.591 | 0.125 | 0.00 | 1.00 |
| svm/radar | testml | 0.498 | 0.419 | 0.590 | 0.006 | 0.00 | 1.00 |
| rf/radar | testml | 0.497 | 0.418 | 0.590 | 0.001 | 0.00 | 0.99 |
| gbm/radar | testml | 0.495 | 0.417 | 0.588 | 0.056 | 0.00 | 0.99 |
| fusemean_gbm/prob | testml | 0.481 | 0.405 | 0.576 | 0.537 | 0.00 | 0.96 |
| fusemean_rf/prob | testml | 0.453 | 0.381 | 0.552 | 0.430 | 0.00 | 0.91 |
| rf/all | testml | 0.440 | 0.370 | 0.541 | 0.238 | 0.00 | 0.88 |
| logreg/radar | testml | 0.433 | 0.499 | 0.031 | 0.058 | 0.85 | 0.02 |
| gbm/sel | testml | 0.414 | 0.348 | 0.517 | 0.449 | 0.00 | 0.83 |
| logreg/joint | testml | 0.407 | 0.343 | 0.510 | 0.293 | 0.00 | 0.81 |
| emb_rf/all | testml | 0.406 | 0.464 | 0.058 | 0.163 | 0.77 | 0.04 |
| stump/radar | testml | 0.385 | 0.324 | 0.490 | 0.523 | 0.00 | 0.77 |
| stump/all | testml | 0.385 | 0.324 | 0.490 | 0.523 | 0.00 | 0.77 |
| stump/sel | testml | 0.385 | 0.324 | 0.490 | 0.523 | 0.00 | 0.77 |
| knn/radar | testml | 0.348 | 0.393 | 0.077 | 0.164 | 0.64 | 0.06 |
| logreg/sel | testml | 0.343 | 0.395 | 0.023 | 0.440 | 0.67 | 0.02 |
| rf/sel | testml | 0.332 | 0.279 | 0.436 | 0.007 | 0.00 | 0.66 |
| gbm/joint | testml | 0.307 | 0.259 | 0.410 | 0.411 | 0.00 | 0.61 |
| rf/joint | testml | 0.273 | 0.230 | 0.373 | 0.182 | 0.00 | 0.54 |
| knn/joint | testml | 0.271 | 0.271 | 0.238 | 0.155 | 0.27 | 0.27 |
| fusw_mlp/prob | testml | 0.247 | 0.208 | 0.345 | 0.125 | 0.00 | 0.49 |
| fusemean_mlp/prob | testml | 0.244 | 0.205 | 0.341 | 0.217 | 0.00 | 0.49 |
| seqcnn/jointseq | testml | 0.169 | 0.142 | 0.249 | 0.012 | 0.00 | 0.34 |
| mlp/radar | testml | 0.101 | 0.085 | 0.157 | 0.015 | 0.00 | 0.20 |
| jev_emb | testml | 0.066 | 0.053 | 0.084 | 0.023 | 0.02 | 0.12 |
| knn/sel | testml | 0.035 | 0.031 | 0.053 | 0.013 | 0.01 | 0.06 |
| threshcal/all | testml | 0.018 | 0.015 | 0.030 | 0.005 | 0.00 | 0.04 |
| svm/csi | testml_norm | 0.865 | 0.857 | 0.843 | 0.933 | 0.81 | 0.92 |
| svm/joint | testml_norm | 0.865 | 0.870 | 0.843 | 0.940 | 0.90 | 0.83 |
| fusw_gbm/prob | testml_norm | 0.834 | 0.820 | 0.812 | 0.857 | 0.74 | 0.92 |
| gbm/csi | testml_norm | 0.830 | 0.817 | 0.807 | 0.909 | 0.75 | 0.91 |
| rf/csi | testml_norm | 0.823 | 0.816 | 0.798 | 0.882 | 0.78 | 0.87 |
| fusw_rf/prob | testml_norm | 0.809 | 0.795 | 0.786 | 0.839 | 0.72 | 0.89 |
| fusemean_gbm/prob | testml_norm | 0.786 | 0.777 | 0.761 | 0.831 | 0.73 | 0.85 |
| logreg/joint | testml_norm | 0.771 | 0.761 | 0.746 | 0.849 | 0.71 | 0.83 |
| svm/sel | testml_norm | 0.763 | 0.765 | 0.729 | 0.847 | 0.78 | 0.75 |
| fusemean_rf/prob | testml_norm | 0.750 | 0.725 | 0.735 | 0.769 | 0.59 | 0.91 |
| mlp/joint | testml_norm | 0.745 | 0.757 | 0.700 | 0.775 | 0.82 | 0.67 |
| gbm/joint | testml_norm | 0.745 | 0.758 | 0.697 | 0.797 | 0.83 | 0.66 |
| mlp/csi | testml_norm | 0.742 | 0.740 | 0.708 | 0.823 | 0.73 | 0.75 |
| mlp/sel | testml_norm | 0.736 | 0.721 | 0.714 | 0.827 | 0.64 | 0.83 |
| knn/csi | testml_norm | 0.697 | 0.731 | 0.600 | 0.787 | 0.91 | 0.48 |
| rf/joint | testml_norm | 0.694 | 0.695 | 0.656 | 0.753 | 0.70 | 0.69 |
| logreg/csi | testml_norm | 0.689 | 0.708 | 0.623 | 0.701 | 0.80 | 0.57 |
| rf/all | testml_norm | 0.651 | 0.627 | 0.645 | 0.615 | 0.50 | 0.81 |
| knn/sel | testml_norm | 0.646 | 0.689 | 0.504 | 0.648 | 0.92 | 0.38 |
| logreg/all | testml_norm | 0.645 | 0.646 | 0.603 | 0.706 | 0.65 | 0.64 |
| fusemean_mlp/prob | testml_norm | 0.627 | 0.637 | 0.565 | 0.705 | 0.69 | 0.56 |
| fusw_mlp/prob | testml_norm | 0.624 | 0.634 | 0.563 | 0.655 | 0.69 | 0.56 |
| knn/all | testml_norm | 0.588 | 0.651 | 0.318 | 0.602 | 0.98 | 0.19 |
| knn/joint | testml_norm | 0.561 | 0.604 | 0.380 | 0.679 | 0.83 | 0.29 |
| gbm/all | testml_norm | 0.549 | 0.501 | 0.589 | 0.599 | 0.25 | 0.85 |
| svm/all | testml_norm | 0.541 | 0.556 | 0.460 | 0.542 | 0.63 | 0.45 |
| mlp/all | testml_norm | 0.541 | 0.542 | 0.495 | 0.564 | 0.55 | 0.53 |
| logreg/sel | testml_norm | 0.535 | 0.556 | 0.434 | 0.448 | 0.67 | 0.40 |
| fusemax_mlp/prob | testml_norm | 0.524 | 0.451 | 0.601 | 0.750 | 0.07 | 0.98 |
| logreg/radar | testml_norm | 0.505 | 0.525 | 0.400 | 0.521 | 0.63 | 0.38 |
| fusemax_gbm/prob | testml_norm | 0.497 | 0.418 | 0.589 | 0.596 | 0.00 | 0.99 |
| fusemax_rf/prob | testml_norm | 0.491 | 0.413 | 0.585 | 0.474 | 0.00 | 0.98 |
| stump/csi | testml_norm | 0.489 | 0.512 | 0.373 | 0.367 | 0.63 | 0.34 |
| stump/joint | testml_norm | 0.489 | 0.512 | 0.373 | 0.367 | 0.63 | 0.34 |
| stump/radar | testml_norm | 0.455 | 0.436 | 0.464 | 0.523 | 0.33 | 0.58 |
| stump/all | testml_norm | 0.455 | 0.436 | 0.464 | 0.523 | 0.33 | 0.58 |
| stump/sel | testml_norm | 0.455 | 0.436 | 0.464 | 0.523 | 0.33 | 0.58 |
| gbm/sel | testml_norm | 0.429 | 0.366 | 0.522 | 0.342 | 0.03 | 0.82 |
| gbm/radar | testml_norm | 0.391 | 0.329 | 0.494 | 0.333 | 0.00 | 0.78 |
| svm/radar | testml_norm | 0.387 | 0.393 | 0.323 | 0.242 | 0.43 | 0.34 |
| rf/radar | testml_norm | 0.379 | 0.319 | 0.483 | 0.227 | 0.00 | 0.76 |
| rf/sel | testml_norm | 0.341 | 0.301 | 0.415 | 0.294 | 0.09 | 0.59 |
| knn/radar | testml_norm | 0.306 | 0.331 | 0.154 | 0.173 | 0.47 | 0.15 |
| mlp/radar | testml_norm | 0.168 | 0.154 | 0.201 | 0.138 | 0.08 | 0.25 |
| rf/csi | val | 0.881 | 0.877 | 0.860 | 0.936 | 0.86 | 0.90 |
| stump/csi | val | 0.856 | 0.862 | 0.832 | 0.863 | 0.89 | 0.82 |
| stump/joint | val | 0.856 | 0.862 | 0.832 | 0.863 | 0.89 | 0.82 |
| gbm/csi | val | 0.821 | 0.800 | 0.799 | 0.955 | 0.69 | 0.95 |
| fusw_gbm/prob | val | 0.749 | 0.712 | 0.739 | 0.955 | 0.52 | 0.98 |
| jev_emb | val | 0.590 | 0.527 | 0.636 | 0.885 | 0.20 | 0.98 |
| fusw_mlp/prob | val | 0.586 | 0.527 | 0.625 | 0.663 | 0.23 | 0.94 |
| fusemean_mlp/prob | val | 0.585 | 0.526 | 0.624 | 0.665 | 0.23 | 0.94 |
| mlp/csi | val | 0.580 | 0.524 | 0.618 | 0.655 | 0.24 | 0.92 |
| mlp/radar | val | 0.524 | 0.594 | 0.172 | 0.441 | 0.95 | 0.10 |
| fusw_rf/prob | val | 0.518 | 0.440 | 0.598 | 0.946 | 0.04 | 1.00 |
| logreg/csi | val | 0.514 | 0.593 | 0.054 | 0.307 | 1.00 | 0.03 |
| emb_mlp/all | val | 0.510 | 0.582 | 0.125 | 0.522 | 0.95 | 0.07 |
| emb_logreg/all | val | 0.507 | 0.588 | 0.028 | 0.195 | 1.00 | 0.01 |
| emb_gbm/all | val | 0.503 | 0.421 | 0.590 | 0.479 | 0.01 | 1.00 |
| knn/joint | val | 0.502 | 0.421 | 0.590 | 0.512 | 0.01 | 1.00 |
| clustervote/joint | val | 0.502 | 0.584 | 0.008 | 0.502 | 1.00 | 0.00 |
| emb_rf/all | val | 0.502 | 0.420 | 0.590 | 0.703 | 0.00 | 1.00 |
| centroid_cal/all | val | 0.501 | 0.583 | 0.002 | 0.659 | 1.00 | 0.00 |
| rf/joint | val | 0.500 | 0.418 | 0.589 | 0.911 | 0.00 | 1.00 |
| logreg/sel | val | 0.500 | 0.418 | 0.589 | 0.499 | 0.00 | 1.00 |
| fusemean_gbm/prob | val | 0.500 | 0.418 | 0.589 | 0.954 | 0.00 | 1.00 |
| svm/radar | val | 0.500 | 0.418 | 0.589 | 0.500 | 0.00 | 1.00 |
| gbm/radar | val | 0.500 | 0.418 | 0.589 | 0.583 | 0.00 | 1.00 |
| stump/radar | val | 0.500 | 0.418 | 0.589 | 0.010 | 0.00 | 1.00 |
| svm/csi | val | 0.500 | 0.418 | 0.589 | 0.033 | 0.00 | 1.00 |
| logreg/joint | val | 0.500 | 0.418 | 0.589 | 0.925 | 0.00 | 1.00 |
| svm/joint | val | 0.500 | 0.418 | 0.589 | 0.055 | 0.00 | 1.00 |
| mlp/joint | val | 0.500 | 0.418 | 0.589 | 0.500 | 0.00 | 1.00 |
| svm/all | val | 0.500 | 0.418 | 0.589 | 0.501 | 0.00 | 1.00 |
| rf/all | val | 0.500 | 0.418 | 0.589 | 0.911 | 0.00 | 1.00 |
| gbm/all | val | 0.500 | 0.418 | 0.589 | 0.899 | 0.00 | 1.00 |
| stump/all | val | 0.500 | 0.418 | 0.589 | 0.010 | 0.00 | 1.00 |
| knn/all | val | 0.500 | 0.418 | 0.589 | 0.500 | 0.00 | 1.00 |
| svm/sel | val | 0.500 | 0.418 | 0.589 | 0.497 | 0.00 | 1.00 |
| mlp/sel | val | 0.500 | 0.418 | 0.589 | 0.500 | 0.00 | 1.00 |
| stump/sel | val | 0.500 | 0.418 | 0.589 | 0.010 | 0.00 | 1.00 |
| knn/sel | val | 0.500 | 0.418 | 0.589 | 0.500 | 0.00 | 1.00 |
| threshcal/all | val | 0.500 | 0.418 | 0.589 | 0.500 | 0.00 | 1.00 |
| emb_svm/all | val | 0.500 | 0.418 | 0.589 | 0.501 | 0.00 | 1.00 |
| fusemean_rf/prob | val | 0.500 | 0.418 | 0.589 | 0.950 | 0.00 | 1.00 |
| fusemax_rf/prob | val | 0.500 | 0.418 | 0.589 | 0.942 | 0.00 | 1.00 |
| fusemax_gbm/prob | val | 0.500 | 0.418 | 0.589 | 0.900 | 0.00 | 1.00 |
| fusemax_mlp/prob | val | 0.500 | 0.418 | 0.589 | 0.500 | 0.00 | 1.00 |
| jev | val | 0.500 | 0.580 | 0.000 | 0.491 | 1.00 | 0.00 |
| gbm/joint | val | 0.500 | 0.418 | 0.589 | 0.937 | 0.00 | 1.00 |
| rf/radar | val | 0.500 | 0.417 | 0.589 | 0.597 | 0.00 | 1.00 |
| knn/radar | val | 0.500 | 0.417 | 0.589 | 0.499 | 0.00 | 1.00 |
| gbm/sel | val | 0.500 | 0.417 | 0.589 | 0.855 | 0.00 | 1.00 |
| rf/sel | val | 0.499 | 0.417 | 0.589 | 0.865 | 0.00 | 1.00 |
| logreg/all | val | 0.499 | 0.417 | 0.588 | 0.499 | 0.00 | 1.00 |
| logreg/radar | val | 0.499 | 0.417 | 0.588 | 0.499 | 0.00 | 1.00 |
| knn/csi | val | 0.496 | 0.414 | 0.586 | 0.491 | 0.00 | 0.99 |
| seqcnn/jointseq | val | 0.484 | 0.409 | 0.571 | 0.564 | 0.03 | 0.94 |
| mlp/all | val | 0.428 | 0.428 | 0.386 | 0.408 | 0.43 | 0.43 |
| mlp/joint | val_norm | 0.893 | 0.899 | 0.876 | 0.918 | 0.93 | 0.86 |
| stump/csi | val_norm | 0.861 | 0.879 | 0.838 | 0.863 | 0.97 | 0.75 |
| stump/joint | val_norm | 0.861 | 0.879 | 0.838 | 0.863 | 0.97 | 0.75 |
| rf/all | val_norm | 0.841 | 0.849 | 0.815 | 0.837 | 0.89 | 0.79 |
| rf/joint | val_norm | 0.830 | 0.854 | 0.796 | 0.824 | 0.98 | 0.68 |
| gbm/joint | val_norm | 0.758 | 0.795 | 0.684 | 0.809 | 0.98 | 0.53 |
| svm/joint | val_norm | 0.738 | 0.758 | 0.681 | 0.798 | 0.86 | 0.62 |
| knn/csi | val_norm | 0.732 | 0.771 | 0.642 | 0.755 | 0.97 | 0.49 |
| knn/sel | val_norm | 0.724 | 0.747 | 0.658 | 0.769 | 0.87 | 0.58 |
| knn/joint | val_norm | 0.692 | 0.741 | 0.559 | 0.761 | 0.99 | 0.39 |
| fusemean_mlp/prob | val_norm | 0.687 | 0.678 | 0.658 | 0.768 | 0.63 | 0.74 |
| fusw_mlp/prob | val_norm | 0.656 | 0.643 | 0.632 | 0.700 | 0.58 | 0.73 |
| mlp/sel | val_norm | 0.648 | 0.610 | 0.654 | 0.832 | 0.41 | 0.88 |
| knn/all | val_norm | 0.642 | 0.699 | 0.446 | 0.750 | 0.99 | 0.29 |
| svm/radar | val_norm | 0.607 | 0.651 | 0.451 | 0.600 | 0.87 | 0.34 |
| knn/radar | val_norm | 0.592 | 0.657 | 0.323 | 0.785 | 0.99 | 0.20 |
| mlp/all | val_norm | 0.564 | 0.569 | 0.509 | 0.580 | 0.59 | 0.53 |
| svm/all | val_norm | 0.555 | 0.578 | 0.451 | 0.508 | 0.70 | 0.41 |
| fusemean_rf/prob | val_norm | 0.549 | 0.510 | 0.574 | 0.759 | 0.31 | 0.79 |
| svm/csi | val_norm | 0.544 | 0.530 | 0.527 | 0.607 | 0.46 | 0.63 |
| rf/sel | val_norm | 0.543 | 0.475 | 0.603 | 0.826 | 0.13 | 0.96 |
| logreg/sel | val_norm | 0.534 | 0.525 | 0.510 | 0.561 | 0.48 | 0.59 |
| fusemax_mlp/prob | val_norm | 0.525 | 0.455 | 0.592 | 0.787 | 0.10 | 0.94 |
| fusemean_gbm/prob | val_norm | 0.519 | 0.483 | 0.544 | 0.720 | 0.30 | 0.74 |
| gbm/all | val_norm | 0.518 | 0.453 | 0.584 | 0.829 | 0.12 | 0.92 |
| fusw_rf/prob | val_norm | 0.512 | 0.471 | 0.547 | 0.703 | 0.26 | 0.77 |
| gbm/sel | val_norm | 0.511 | 0.435 | 0.590 | 0.944 | 0.05 | 0.97 |
| fusw_gbm/prob | val_norm | 0.507 | 0.467 | 0.540 | 0.697 | 0.26 | 0.75 |
| gbm/csi | val_norm | 0.502 | 0.461 | 0.538 | 0.635 | 0.25 | 0.75 |
| rf/radar | val_norm | 0.501 | 0.419 | 0.590 | 0.701 | 0.00 | 1.00 |
| fusemax_rf/prob | val_norm | 0.500 | 0.418 | 0.589 | 0.826 | 0.00 | 1.00 |
| stump/radar | val_norm | 0.499 | 0.417 | 0.588 | 0.010 | 0.00 | 0.99 |
| stump/all | val_norm | 0.499 | 0.417 | 0.588 | 0.010 | 0.00 | 0.99 |
| stump/sel | val_norm | 0.499 | 0.417 | 0.588 | 0.010 | 0.00 | 0.99 |
| mlp/radar | val_norm | 0.498 | 0.517 | 0.398 | 0.508 | 0.61 | 0.38 |
| fusemax_gbm/prob | val_norm | 0.493 | 0.414 | 0.582 | 0.719 | 0.01 | 0.97 |
| gbm/radar | val_norm | 0.482 | 0.407 | 0.569 | 0.682 | 0.03 | 0.94 |
| rf/csi | val_norm | 0.474 | 0.432 | 0.517 | 0.641 | 0.22 | 0.73 |
| logreg/joint | val_norm | 0.464 | 0.434 | 0.488 | 0.528 | 0.28 | 0.65 |
| logreg/csi | val_norm | 0.419 | 0.436 | 0.318 | 0.359 | 0.52 | 0.31 |
| svm/sel | val_norm | 0.417 | 0.373 | 0.475 | 0.404 | 0.16 | 0.68 |
| mlp/csi | val_norm | 0.415 | 0.373 | 0.472 | 0.538 | 0.16 | 0.67 |
| logreg/radar | val_norm | 0.408 | 0.411 | 0.354 | 0.369 | 0.43 | 0.39 |
| logreg/all | val_norm | 0.174 | 0.152 | 0.234 | 0.160 | 0.04 | 0.31 |

## Figures
- figs/model_bars.png
- figs/roc.png
- figs/cm.png
- figs/calibration.png

Workflow A = descriptor fusion (per-view + 'all').
Workflow B = shared PCA embedding (emb_*).
Workflow C = probability fusion (fuse*/fusw_*).
cal_sup = 10 min t1_sleep recall; cal_unsup = 5+5 min balanced accuracy.