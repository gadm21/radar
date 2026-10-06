# ICC-style native LaTeX manuscript

`icc_multimodal.tex` is the root manuscript. Compile from the radar folder with `./build_icc.ps1`, or use `latexmk -pdf icc_multimodal.tex` with a full TeX distribution. The build uses `IEEEtran` in 10-point, letter-paper conference mode. All plotted figures are native PGFPlots; the processing overview is native TikZ. No ReportLab rendering or rasterized equations are used.

`icc_assets/` contains generated plot definitions, measured result rows, and the overview diagram. `export_icc_data.py` exports the verified 100 Hz experiment into these files. Do not manually change reported scores in the TeX source.

The WiFi channel formulation follows the user-supplied manuscript. The FMCW chirp/data-cube formulation cites HOOD, arXiv:2308.02396v2. Its RF parameters, E-RESPD processing, and neural architecture are not claimed as this experiment's configuration. Equations describe the implemented FFT, beamforming, map pooling, CSI resampling, PCA variance, fusion, and classification operations.

Current measurements: `E2/outputs/paper_100hz_variance3s/`. Validation contains 80 empty and 80 occupied minutes; test contains 96 and 96. The PCA fit uses 51,287 calibration/training summaries. All 1,523 recordings and 52 result rows have been checked. The same evaluation minute IDs are retained from the preceding experiment.

Training clarification pending: the actual folder is `train`, not `train_minutes`; it contains 184 empty and 84 occupied complete cases. The current manuscript reports the class-weighted baseline. Augmenting empty would worsen this imbalance, so no synthetic examples have been generated or represented as collected recordings pending clarification of the intended class/folder.

The author list is retained from the supplied manuscript. Confirm affiliations and physical collection metadata before submission. The manuscript is an ICC-style draft, not a PDF eXpress-certified or submitted paper. Venue-year submission requirements should be checked when a target edition is selected.
