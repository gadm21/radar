# E2 multilink_eval — ML/DL results (W=5s)
Train/validation on legacy train_minutes (+test_minutes as validation for thresholding), test on the 4 multilink captures (2-CSI-link + radar). Multilink CSI = 2-link mean.

## within-legacy reference (train_minutes → test_minutes)
| model/set | acc | bal_acc | auc | rec_e | rec_o | thr |
|---|---|---|---|---|---|---|
| logreg/csi | 0.563 | 0.568 | 0.612 | 0.54 | 0.60 | 0.50 |
| gbm/radar | 0.586 | 0.527 | 0.599 | 0.88 | 0.17 | 0.50 |
| gbm/fusion | 0.584 | 0.525 | 0.566 | 0.88 | 0.17 | 0.50 |
| svm/radar | 0.436 | 0.517 | 0.528 | 0.03 | 1.00 | 0.50 |
| mlp/radar | 0.435 | 0.516 | 0.502 | 0.03 | 1.00 | 0.50 |
| logreg/radar | 0.578 | 0.513 | 0.873 | 0.90 | 0.12 | 0.50 |

## cross-domain: legacy train → multilink test
| model/set | acc | bal_acc | auc | rec_e | rec_o | thr |
|---|---|---|---|---|---|---|
| rf/fusion | 0.953 | 0.953 | 0.943 | 1.00 | 0.91 | 0.62 |
| rf/radar | 0.941 | 0.941 | 0.944 | 0.97 | 0.91 | 0.63 |
| svm/radar | 0.917 | 0.917 | 0.970 | 0.93 | 0.90 | 0.70 |
| mlp/radar | 0.907 | 0.907 | 0.948 | 0.94 | 0.87 | 0.64 |
| logreg/radar | 0.882 | 0.882 | 0.945 | 0.98 | 0.79 | 0.95 |
| gbm/radar | 0.877 | 0.877 | 0.902 | 0.87 | 0.88 | 0.85 |

## multilink LOCO — mean/min balanced accuracy
| model/set | bal_mean | bal_min | rec_e | rec_o |
|---|---|---|---|---|
| logreg/radar | 0.851 | 0.546 | 0.99 | 0.71 |
| rf/fusion | 0.749 | 0.000 | 0.50 | 1.00 |
| rf/radar | 0.748 | 0.000 | 0.50 | 1.00 |
| logreg/csi | 0.719 | 0.000 | 0.50 | 0.94 |
| mlp/radar | 0.714 | 0.560 | 0.86 | 0.57 |
| svm/radar | 0.609 | 0.015 | 0.37 | 0.85 |
| mlp/fusion | 0.558 | 0.074 | 0.53 | 0.58 |
| logreg/fusion | 0.535 | 0.006 | 0.50 | 0.57 |
| gbm/radar | 0.499 | 0.000 | 0.01 | 0.99 |
| gbm/fusion | 0.498 | 0.000 | 0.01 | 0.99 |


## multilink LOCO — per-capture detail (fusion set)
| cap | rf/fusion | gbm/fusion | mlp/fusion | seqcnn/fusion | rf/radar | logreg/radar |
|---|---|---|---|---|---|---|
| capture_empty_20261003_195314 | 1.00 | 0.02 | 1.00 | 0.00 | 1.00 | 1.00 |
| capture_empty_20261004_171234 | 0.00 | 0.00 | 0.07 | 0.02 | 0.00 | 0.98 |
| capture_occupied_20261003_183049 | 1.00 | 0.99 | 0.68 | 1.00 | 0.99 | 0.55 |
| capture_occupied_20261005_124135 | 1.00 | 0.99 | 0.48 | 0.93 | 1.00 | 0.88 |

Note: single-class test captures leave the absent-class recall undefined; aggregates use nan-aware means. The empty-Oct-4 capture is the hard fold — most models score ~0 on it (predicted occupied). Its `ra_mean`/`snr_mean` sit near empty-Oct-3, but `csi2_amp_mean` shifted 22.7 vs 41.2 — a cross-session CSI amplitude drift the trained models don't tolerate (cf. E1 LOCO recall).
