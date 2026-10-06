# Guaranteed-rate CSI preprocessing

Run from the top-level radar folder:

```powershell
python run_csi_100hz.py --self-test
python run_csi_100hz.py
python E2/build_conference_paper.py --100hz
```

Use the project's scientific Python environment. The experiment writes only to `E2/outputs/paper_100hz_variance3s`; the paper writes to `output/pdf/multimodal_wireless_sensing_100hz_paper.pdf` and its source folder. Earlier experiment outputs and raw recordings are preserved.

The script executes `CSI_Loader._resample_equal_intervals` from the desktop `WifiSensingESP32HAR/src/train/utils.py` with `guaranteed_sr=100`. The method's exact source and SHA-256 are stored with the experiment. It bins magnitude samples at 10 ms intervals, averages colliding samples, and linearly interpolates empty bins. The first observed sample anchors the grid; endpoint behavior is unchanged from upstream. Relative host timestamps are supplied in microseconds.

Processing order:

1. Load the selected receiver, keep finite observations inside the nominal minute, and sort/deduplicate timestamps.
2. Apply the upstream guaranteed-rate routine to CSI amplitudes at 100 Hz.
3. Compute 52 amplitude means and 52 population standard deviations per accepted second, using exactly 100 resampled values. Keep the original observation-quality requirements (at least five packets, at least 0.5 seconds of span, and no gap above 0.5 seconds). Partial boundary seconds are omitted.
4. Refit centered, unwhitened 2D PCA on calibration and training only.
5. Sum PC1 and PC2 population variances across three consecutive one-second dots; take the median over valid windows within each minute. Classifiers use log1p of this feature.
6. Reuse radar features and the exact prior balanced minute IDs: validation 80/80 and test 96/96. Refit the same models, select thresholds on validation, and evaluate frozen decisions on test.

The 100 Hz guarantee concerns the processed grid, not packet acquisition. PCA dots remain one second apart; this change does not redefine the feature as 300 raw packet projections. `resampling_audit.csv` records actual packet rate, target samples, interpolation, raw gaps, and accepted seconds. Upstream integer microsecond timestamps may introduce up to one microsecond rounding between grid steps.

Extraction is checkpointed per minute and can be resumed by rerunning the same command. Existing caches must not be reused after changing the algorithm or its parameters. The script checks the previous cohort for any lost CSI feature rather than silently substituting different minutes.
