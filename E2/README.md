# E2 — Occupancy Detection Pipeline

Binary occupancy detection (**empty** vs **occupied** = sleep + present)
from mmWave radar + Wi-Fi CSI.

## Layout

| file | purpose |
|---|---|
| `common.py` | tolerant manifest parsing, recording discovery, label mapping, radar/CSI access |
| `inspect_dataset.py` | dataset counts + timing report -> `outputs/inspection.json` |
| `preprocess.py` | radar FFT maps + CSI windows -> per-recording cache in `cache/` |
| `models.py` | RadarEncoder (temporal-std map -> spatial pooling -> MLP), CSIEncoder (Conv1d over rolling variance of amplitude), CSIStatsEncoder (MLP on amplitude var/mean/temporal-std — the fusion CSI branch), gated fusion, binary head |
| `train.py` | split loading, train-only normalization, balancing helpers, training, metrics |
| `experiments.py` | E2 train/val/test runs + hyperparameter search |
| `scene_model.py` | deployment model: all preprocessing inside the TorchScript forward (scene-feature MLP) -> `deploy/` |
| `clean_minutes.py` | strip auto-labels, remove empty/unlabeled folders |
| `export_models.py` | TorchScript exports of the CNN models -> `deploy/` |
| `eval_deploy.py` | evaluate `deploy/*.pt` on test_minutes (window / partial-minute / minute) |
| `plots.py` | figures -> `outputs/figs/` |
| `report.py` | `outputs/REPORT.md` |
| `run_all.py` | one-command pipeline |

## Run

```powershell
pip install -r E2/requirements.txt
python E2/run_all.py            # full pipeline (~1-2 h on CPU)
python E2/run_all.py --skip-preprocess   # reuse cache
```

## Data rules (documented)

- **Splits**: train = `train_minutes/` (labels `t1_*`/`t2_*`, placements
  t1+t2) + `train2_minutes/` (labels `t_*`, placement t, Sept 6-8);
  validation = `validation_minutes/` (labels `empty`/`present`,
  placement t, Sept 15); test = `test_minutes/` (Pi captures, Sept 16-17
  night — manifests labeled by `label_test_minutes.py` from the 1:10 AM
  boundary: `present` before `20260917_0110`, `empty` at/after).
  Sleep counts as occupied.
- **Windows**: 50 consecutive radar frames (~5-7 s, long enough for
  breathing) inside one recording, non-overlapping. Never cross
  recordings/labels.
- **Radar timing**: `train_minutes/` — `.bin` filename is the capture
  timestamp; each file spans 1 s (frames at `ts + j/n`). The npz
  sources — per-frame timestamps are burst-flushed; frames are spread
  uniformly inside their `second_start` bucket.
- **CSI**: one receiver per recording — the `wifi_csi_XX.csv` (or npz
  receiver index) with most samples inside the minute. Amplitude of the
  52 usable subcarriers, log1p, linearly interpolated onto a 128-step
  grid spanning the radar window; windows with <2 samples are zeroed +
  flagged, never fabricated. The CSI encoder consumes **causal rolling
  variance (w=20) of amplitude** (WifiSensingESP32HAR pipeline; no
  phase).
- **Radar maps**: per frame, Hann-windowed FFT -> range-Doppler
  (64x64) and range-azimuth (16x64), log1p, resized to 24x24.
- **Features (CNN models)**: the encoder pools each window along time
  into a per-pixel **temporal-std map** (motion/breathing energy), then
  spatially to mean/std/max per channel. Absolute levels flip sign
  across placements and are excluded — this is what makes the model
  transfer to an unseen room.
- **Features (scene model)**: per-window mean-map AND std-map spatial
  stats + 12-bin range/azimuth profiles + argmax, computed inside the
  TorchScript `forward` — the exported `.pt` is self-contained
  (identity normalization in meta.json).
- **Normalization**: mean/std computed on the train pool only
  (`--norm-mode local` switches to per-recording z-score).
- **Balancing**: training windows subsampled to equal class counts;
  early stopping + threshold use a class-balanced val subsample.
- **Augmentation**: doppler-sign/azimuth flip; 30% CSI dropout (fusion).
- **Minute aggregation**: per-recording score = mean of the **top-2**
  window probabilities (a still person still moves occasionally within
  a minute; a plain mean is diluted by still windows). A separate
  `minute_threshold` is tuned on validation_minutes — the median of the
  optimal range, since saturated val probs make the argmax edge
  miscalibrate on the shifted test day.
- **Threshold/early stopping/HP search**: validation_minutes only;
  test_minutes is touched once for final metrics.

## Outputs

`outputs/`: `inspection.json`, `hp_search.json`, `best_config.json`,
`results_e2.json`, `scene_model_results.json`,
`deploy_eval_test_minutes.json`, `best_model.pt`, `model_{radar,fusion,csi}.pt`,
`REPORT.md`, `figs/*.png`. Deployment artifacts land in `deploy/`
(TorchScript + embedded `meta.json` + registry `*.meta.json`).
