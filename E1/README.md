# E1 — Class-Separability Analysis (empty vs occupied)

Physical descriptors and compressed-PCA features that distinguish the
two classes, applied to two corpora:

| corpus | data | modalities |
|---|---|---|
| `multilink` | `multilink_train/` — 4 JSONL captures (Oct 3–5): chen `csi-bb8b`, peer `csi-8b45` (april→toronto), chen `radar-a316` | 2 CSI links + radar |
| `legacy` | `train_minutes/train` + `test_minutes` minutes via `E1/cache/win` + `E2/outputs/occupancy/features_{5,10}s.csv` | 1 CSI link + radar |

## Descriptors (per 5 s / 10 s window)

- **CSI per link**: amplitude stats, 0.2 s rolling-variance stats,
  total variation, high-band PSD fraction, `pcv1-3` (variance of the
  3-D PCA projection of per-second 104-d mean|std "dots").
- **Radar**: SNR, `rad_pcv1-3` (variance of 3-D PCA projection of
  per-second mean-RD "dots"), RD/RA/RE/XY map descriptors, physical
  RD descriptors (range/doppler centroids, doppler spread, changed-
  area density, energy).
- **Screening**: iterative — univariate AUC / Cohen's d / Mann-Whitney,
  collinearity prune, RF importance, top-15 grouped-CV probe.

## Run

```powershell
python E1/run_all.py                 # full pipeline (extract + analyze)
python E1/run_all.py --skip-extract  # reuse E1/cache
```

Outputs land in `E1/outputs/`: `REPORT.md`, `analysis.json`,
`features_{multilink,old}_{5,10}s*.csv`, `rankings_*.csv`,
`pc_dots_*.npz`, `pcas_multilink.joblib`, `figs/`.

## Deep-PCA stage (`pca_deep.py` + `pca_deep_figs.py`)

Second-pass analysis producing the window descriptor tables used by
`E2/model_compare.py`:

- **Splits**: `train` = train_minutes/train, `val` = validation_minutes,
  `calib` = calibration_minutes (all `t1_sleep` → occupied),
  `testml` = multilink captures.
- **Preprocessing**: per-second validity masks (no interpolation),
  StandardScaler + PCA fit on train seconds (multilink: unsupervised
  self-fit), CSI=104-d dots (multilink = 2-link availability mean),
  radar = flattened per-second maps.
- **Window descriptors (5 s, hop 2 s)**: PC stats (mean/std/iqr/min/max
  per component), windowed PC variance, temporal-trajectory features
  (speed, path, displacement, tortuosity, reversals, lag-1 autocorr,
  slope), cluster geometry (kmeans/GMM distances + labels fit on train
  joint-PC space), FFT band descriptors (radar SNR stream all sets;
  100-Hz CSI amplitude FFT legacy-only), plus shared physical
  descriptors (rd/ra/re/xy stats, centroids, SNR, CSI amp/rv/tv/dop).
- **Outputs**: `E1/outputs/pca_deep/` — `desc_<split>_5s.csv`,
  `sep_<split>.csv`, `analysis_pca.json`, `pcas_deep.joblib`,
  `pc_dots_<split>.npz`, `REPORT_PCA.md`, `figs_pca/`.
