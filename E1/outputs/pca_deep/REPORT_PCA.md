# E1 deep-PCA analysis (E1/pca_deep)

## Preprocessing
- per-second validity masks; invalid seconds excluded (no fabricated motion via interpolation)
- StandardScaler + PCA fit on **train** seconds only (multilink: unsupervised fit on its own seconds)
- windows W=5 s, hop 2 s, min coverage 3 s
- FFT descriptors: radar SNR stream (all sets) + 100 Hz CSI amplitude FFT (legacy only)

## Variance explained (first 3 PCs)
- **train**: csi 81.6% / 90.7%@8,  radar 6.8% / 12.0%@8
- **val**: csi 81.6% / 90.7%@8,  radar 6.8% / 12.0%@8
- **calib**: csi 81.6% / 90.7%@8,  radar 6.8% / 12.0%@8
- **testml**: csi 90.6% / 95.3%@8,  radar 39.4% / 54.0%@8

## Unsupervised cluster quality (train joint-PC space)
| split | kmeans purity | ARI | silhouette | dbscan noise | dbscan clusters |
|---|---|---|---|---|---|
| train | 0.748 | 0.183 | 0.261 | 0.006 | 1 |
| val | 0.584 | 0.001 | 0.573 | 0.006 | 1 |
| calib | 1.000 | 1.000 | nan | 0.006 | 1 |
| testml | 0.579 | 0.023 | 0.429 | 0.000 | 1 |

## Top descriptors per split (|AUC|)
### train
| descriptor | auc (eff) | dir | cohen d |
|---|---|---|---|
| fftc_b0 | 0.881 | +occ | -1.13 |
| clu_kmdist1 | 0.855 | +occ | -1.32 |
| fftc_domf | 0.853 | +empty | +1.31 |
| r_min1 | 0.825 | +empty | +1.43 |
| j_min4 | 0.825 | +empty | +1.43 |
| ra_peak | 0.821 | +occ | -1.30 |
| j_mean4 | 0.817 | +empty | +1.38 |
| r_mean1 | 0.817 | +empty | +1.38 |
| fftc_entr | 0.817 | +empty | +1.02 |
| rd_peak | 0.812 | +occ | -0.90 |

### val
| descriptor | auc (eff) | dir | cohen d |
|---|---|---|---|
| ra_peak | 0.991 | +empty | +0.28 |
| csi_amp_q90 | 0.974 | +occ | -3.33 |
| clu_kmdist1 | 0.972 | +occ | -3.61 |
| clu_kmdist0 | 0.972 | +occ | -3.39 |
| c_min0 | 0.970 | +occ | -2.89 |
| j_min0 | 0.970 | +occ | -2.89 |
| csi_amp_mean | 0.969 | +occ | -2.95 |
| c_mean0 | 0.969 | +occ | -2.65 |
| j_mean0 | 0.969 | +occ | -2.65 |
| xy_peak | 0.962 | +empty | +0.36 |

### calib
| descriptor | auc (eff) | dir | cohen d |
|---|---|---|---|
| c_mean0 | nan | +empty | +nan |
| c_std0 | nan | +empty | +nan |
| c_iqr0 | nan | +empty | +nan |
| c_min0 | nan | +empty | +nan |
| c_max0 | nan | +empty | +nan |
| c_mean1 | nan | +empty | +nan |
| c_std1 | nan | +empty | +nan |
| c_iqr1 | nan | +empty | +nan |
| c_min1 | nan | +empty | +nan |
| c_max1 | nan | +empty | +nan |

### testml
| descriptor | auc (eff) | dir | cohen d |
|---|---|---|---|
| ra_mean | 1.000 | +occ | -3.01 |
| r_max1 | 1.000 | +occ | -3.58 |
| j_max4 | 1.000 | +occ | -3.58 |
| r_mean1 | 0.999 | +occ | -3.52 |
| j_mean4 | 0.999 | +occ | -3.52 |
| j_min4 | 0.999 | +occ | -3.37 |
| r_min1 | 0.999 | +occ | -3.37 |
| re_mean | 0.990 | +occ | -2.69 |
| rd_std90 | 0.981 | +occ | -1.37 |
| re_p90 | 0.966 | +occ | -2.41 |

## Figures
- figs_pca/ev_scree.png
- figs_pca/pc_scatter.png
- figs_pca/pcv_box.png
- figs_pca/trajectories.png
- figs_pca/clusters.png
- figs_pca/fft_bands.png
- figs_pca/desc_auc.png
- figs_pca/corr.png
- figs_pca/timeline.png

## Caveats
- multilink PCA is an unsupervised fit (no labels); cluster metrics there measure self-consistency.
- legacy val/calib share the train-fit space: their purity/ARI are *transfer* metrics.
- fftc_* exists only on legacy (raw 100 Hz CSI stream).
- t_sleep/t1_sleep count as occupied (E2/common mapping).