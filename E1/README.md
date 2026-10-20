# E1 — class-consistent descriptors and visual evidence

`revision_figures.py` generates the current paper figures from the retained
feature tables and per-second PCA dots. It screens class direction in the
training, validation, and multilink test cohorts. The display
requires the same occupied-versus-empty direction in all three and a minimum
direction-adjusted AUC of 0.55. The complete screen is in
`outputs/stable_descriptor_screen.csv`.

The six passing descriptors are `ra_std90`, `re_std90`, `xy_std90`,
`re_delta`, `ra_delta`, and `ra_p90`. The figure displays the strongest three.
`ra_mean`, `re_mean`, and `xy_mean` reverse direction in validation
cohort and are excluded from the main display.

Current figures in `outputs/figs/`:

- `stable_descriptor_box_scatter.png`: boxplots and scatter in three cohorts.
- `hierarchical_pca.png`: one-second dots, PCA of ten consecutive dots, and
  PCA of six consecutive ten-second vectors (one minute). The higher bases
  fit training data only; partial blocks are interpolated as documented in code.
- `timeseries_ml.png`: 20-second rolling means of `csi_amp_mean` and
  `snr_mean` in train/validation/test columns and two rows.
- `mean_maps_movement.png`: class-conditional consecutive-second RD/XY map
  changes from labeled samples, with paired color scales.

The old `pc_trajectories` figure is no longer generated. Historical feature
tables and caches remain for reproduction. To rebuild with offline packages:

```powershell
$env:PYTHONPATH = (Resolve-Path E2/.deps).Path
python E1/revision_figures.py
```
