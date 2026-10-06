# E1 — Class-Separability Analysis: Empty vs Occupied

Physical descriptors and compressed-PCA features that distinguish the two classes, computed identically on the new **multilink** captures (`multilink_train/`, 2 CSI links + BGT60TR13C radar, JSONL streams) and the **legacy** train/test minutes (1 CSI link + radar, E2 cache + published feature tables extended with the same physical descriptors).

## Datasets

| dataset | windows_5s | empty | occupied | csi_links | grouping |
|---|---|---|---|---|---|
| multilink (4 captures) | 4320 | 2160 | 2160 | 2 | capture |
| legacy train (wall1+wall2 mins) | 12972 | 2244 | 10728 | 1 | minute |
| legacy test (wall3 mins) | 2304 | 1344 | 960 | 1 | minute |

## Method

Per 5 s and 10 s non-overlapping windows:
- **CSI (per link)**: amplitude mean/std/q90; 0.2 s causal rolling-variance mean/q90/max (log1p); total variation of the mean-amplitude trace; high-band PSD fraction (micro-Doppler proxy); `pcv1-3` — variance of the 3-D PCA projection of the per-second 104-d amplitude 'dots'.
- **radar**: clutter-masked `snr_max/mean`; `rad_pcv1-3` — variance of the 3-D PCA projection of the per-second mean RD-map 'dots'; per-view RD/RA/RE/XY descriptors (mean/p90/std90/delta/peak); physical descriptors `rd_range_cm` (range-bin centroid), `rd_dop_cm` (doppler centroid), `rd_dop_spread`, `rd_dca` (hot-pixel area fraction), `rd_energy`.

Screening is iterative: junk-column drop -> univariate ROC-AUC/Cohen's-d/Mann-Whitney -> collinearity prune (|r|>0.95) -> RF importance -> top-15 grouped 5-fold CV probe (LogReg and RF; folds grouped by capture/minute so windows never leak across recordings).

## Separation quality (W=5s)

`cv_*` = 5-fold stratified grouped CV (by minute) for legacy, window-stratified CV for multilink (only 4 capture groups). `loco_e/o` = leave-one-capture-out recall, empty/occupied — the honest cross-session estimate.

| group | n | best_feature | best_auc | cv_logreg | cv_rf | loco_e/o |
|---|---|---|---|---|---|---|
| multilink_5s/csi_link1 | 4315 | csi1_pcv1 | 0.859 | 0.971 | 0.989 | 0.97/0.72 |
| multilink_5s/csi_link2 | 3696 | csi2_amp_mean | 0.925 | 0.997 | 0.999 | 0.50/0.30 |
| multilink_5s/csi_2link_mean | 3691 | csiM_amp_mean | 0.950 | 0.996 | 0.996 | 0.50/0.87 |
| multilink_5s/radar | 4314 | ra_mean | 0.999 | 1.000 | 1.000 | 0.97/0.75 |
| multilink_5s/fusion | 4313 | ra_mean | 0.999 | 1.000 | 1.000 | 0.85/0.76 |
| old_train_5s/csi | 9190 | csi_amp_std | 0.858 | 0.966 | 0.954 | - |
| old_train_5s/radar | 11049 | xy_peak | 0.970 | 0.998 | 0.999 | - |
| old_train_5s/fusion | 7896 | xy_peak | 0.970 | 0.999 | 1.000 | - |
| old_test_5s/csi | 1492 | csi_amp_q90 | 0.974 | 0.976 | 0.983 | - |
| old_test_5s/radar | 2201 | ra_peak | 0.880 | 0.671 | 0.964 | - |
| old_test_5s/fusion | 1492 | csi_amp_q90 | 0.974 | 0.989 | 0.995 | - |

## Top descriptors (combined rank), W=5s

**multilink**

| feature | auc | d | rf_imp |
|---|---|---|---|
| ra_mean | 0.999 | +3.07 | 0.1981 |
| re_mean | 0.992 | +2.91 | 0.1492 |
| xy_std90 | 0.976 | +0.99 | 0.0945 |
| re_std90 | 0.971 | +1.14 | 0.1159 |
| xy_mean | 0.941 | +2.04 | 0.0818 |
| re_p90 | 0.919 | +1.98 | 0.0701 |
| csiM_amp_mean | 0.875 | -1.47 | 0.0698 |
| csiM_pcv1 | 0.881 | +0.36 | 0.0379 |
| csiM_dop_frac | 0.883 | -1.51 | 0.0159 |
| xy_peak | 0.897 | +1.87 | 0.0153 |
| ra_p90 | 0.853 | +1.24 | 0.0290 |
| re_delta | 0.844 | +1.09 | 0.0222 |

**legacy train**

| feature | auc | d | rf_imp |
|---|---|---|---|
| xy_peak | 0.970 | -2.36 | 0.1880 |
| xy_p90 | 0.969 | +2.74 | 0.1668 |
| re_std90 | 0.946 | +0.62 | 0.0765 |
| ra_peak | 0.929 | -1.77 | 0.1470 |
| re_p90 | 0.932 | +2.25 | 0.0391 |
| rd_mean | 0.928 | -1.78 | 0.0666 |
| re_peak | 0.870 | -1.34 | 0.0999 |
| xy_mean | 0.901 | +2.28 | 0.0417 |
| re_delta | 0.826 | +0.71 | 0.0358 |
| xy_std90 | 0.835 | +0.51 | 0.0223 |
| ra_p90 | 0.835 | +1.30 | 0.0192 |
| csi_amp_std | 0.854 | -1.23 | 0.0099 |

**legacy test**

| feature | auc | d | rf_imp |
|---|---|---|---|
| csi_amp_q90 | 0.974 | +3.57 | 0.2920 |
| ra_peak | 0.929 | -0.10 | 0.1801 |
| xy_std90 | 0.924 | +0.04 | 0.1848 |
| ra_delta | 0.916 | +0.34 | 0.1321 |
| csi_pcv1 | 0.842 | +0.18 | 0.0370 |
| csi_amp_std | 0.821 | +1.37 | 0.0662 |
| csi_tv | 0.694 | +0.69 | 0.0200 |
| rd_p90 | 0.635 | +0.46 | 0.0123 |
| rd_range_cm | 0.651 | +0.36 | 0.0051 |
| rd_std90 | 0.607 | +0.33 | 0.0033 |
| snr_max | 0.581 | +0.46 | 0.0068 |
| rd_delta | 0.624 | +0.42 | 0.0025 |

## Top descriptors (combined rank), W=10s

**multilink**

| feature | auc | d | rf_imp |
|---|---|---|---|
| ra_mean | 1.000 | +3.19 | 0.1700 |
| re_mean | 0.992 | +3.02 | 0.1746 |
| xy_std90 | 0.989 | +1.14 | 0.1059 |
| re_std90 | 0.982 | +1.29 | 0.1108 |
| xy_mean | 0.944 | +2.08 | 0.1035 |
| csiM_pcv1 | 0.920 | +0.48 | 0.0520 |
| re_p90 | 0.919 | +2.00 | 0.0387 |
| csiM_amp_mean | 0.876 | -1.49 | 0.0543 |
| re_delta | 0.868 | +1.18 | 0.0505 |
| csiM_dop_frac | 0.908 | -1.60 | 0.0141 |
| xy_peak | 0.897 | +1.88 | 0.0158 |
| ra_p90 | 0.858 | +1.33 | 0.0333 |

**legacy train**

| feature | auc | d | rf_imp |
|---|---|---|---|
| xy_peak | 0.972 | -2.36 | 0.1759 |
| xy_p90 | 0.971 | +2.79 | 0.1676 |
| re_std90 | 0.951 | +0.61 | 0.1014 |
| rd_mean | 0.939 | -1.89 | 0.0919 |
| ra_peak | 0.934 | -1.82 | 0.1112 |
| re_p90 | 0.934 | +2.30 | 0.0390 |
| xy_mean | 0.903 | +2.35 | 0.0417 |
| csi_amp_std | 0.889 | -1.38 | 0.0480 |
| re_peak | 0.867 | -1.33 | 0.0566 |
| xy_std90 | 0.864 | +0.48 | 0.0277 |
| re_delta | 0.842 | +0.74 | 0.0290 |
| rd_p90 | 0.878 | -1.20 | 0.0166 |

**legacy test**

| feature | auc | d | rf_imp |
|---|---|---|---|
| csi_amp_q90 | 0.979 | +4.07 | 0.2560 |
| csi_amp_std | 0.960 | +2.10 | 0.1282 |
| ra_peak | 0.911 | -0.18 | 0.1514 |
| re_std90 | 0.903 | +0.03 | 0.1656 |
| ra_delta | 0.906 | +0.54 | 0.1186 |
| csi_pcv1 | 0.895 | +0.28 | 0.0509 |
| csi_tv | 0.780 | +1.06 | 0.0212 |
| rd_p90 | 0.676 | +0.53 | 0.0116 |
| csi_rv_q90 | 0.650 | -0.53 | 0.0209 |
| csi_pcv3 | 0.646 | -0.45 | 0.0063 |
| csi_rv_max | 0.634 | -0.46 | 0.0037 |
| ra_p90 | 0.581 | +0.14 | 0.0122 |

## Cross-dataset check — shared descriptors

Same feature, univariate direction-free AUC, multilink vs legacy train vs legacy test (W=5s):

| descriptor | multilink | legacy train | legacy test |
|---|---|---|---|
| snr_mean | 0.707 | 0.515 | 0.531 |
| snr_max | 0.705 | 0.595 | 0.567 |
| rad_pcv1 | 0.512 | 0.651 | 0.553 |
| rd_delta | 0.604 | 0.586 | 0.595 |
| rd_std90 | 0.662 | 0.589 | 0.577 |
| rd_dca | 0.503 | 0.504 | 0.503 |
| rd_dop_spread | 0.535 | 0.500 | 0.508 |
| rd_range_cm | 0.519 | 0.536 | 0.636 |
| csi rv_mean | 0.653 | 0.755 | 0.506 |
| csi pcv1 | 0.881 | 0.680 | 0.842 |
| csi amp_std | 0.752 | 0.858 | 0.821 |
| csi amp_q90 | 0.820 | 0.639 | 0.974 |
| csi tv | 0.552 | 0.787 | 0.693 |
| csi dop_frac | 0.883 | 0.762 | 0.589 |

`-` = descriptor not present for that dataset.

## Figures

- `figs/corr_multilink.png`
- `figs/corr_old_train.png`
- `figs/descriptor_dists_multilink.png`
- `figs/descriptor_dists_old_test.png`
- `figs/descriptor_dists_old_train.png`
- `figs/importance_ml5s.png`
- `figs/importance_oldtest5s.png`
- `figs/importance_oldtrain5s.png`
- `figs/mean_maps.png`
- `figs/pc_dots_scatter.png`
- `figs/pc_trajectories.png`
- `figs/pcv_box.png`
- `figs/timeseries_ml.png`


## Caveats
- Multilink link2 (thoth-toronto) stopped streaming ~70 min into the Oct-5 occupied capture; affected windows are masked by `csi2_ok`.
- PCAs are fit unsupervised per dataset (multilink) or reused from E2 (legacy) — pcv magnitudes are not directly comparable across datasets; compare AUCs, not raw values.
- `csi_tv`/`csi_dop_frac`/`rd_*_cm` units are descriptor-internal (bins/log-power), not calibrated physics units.
