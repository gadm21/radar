# E2 — Occupancy Detection Pipeline

Binary occupancy detection (**empty** vs **occupied** = sleep + present)
from mmWave radar + Wi-Fi CSI.

## Layout

| file | purpose |
|---|---|
| `common.py` | tolerant manifest parsing, recording discovery, label mapping, radar/CSI access |
| `inspect_dataset.py` | dataset counts + timing report -> `outputs/inspection.json` |
| `preprocess.py` | radar FFT maps + CSI windows -> per-recording cache in `cache/` |
| `models.py` | RadarEncoder (temporal-std map -> spatial pooling -> MLP), CSIEncoder (Conv1d, DC/scale-free), gated fusion, binary head |
| `train.py` | split loading, train-only normalization, balancing helpers, training, metrics |
| `experiments.py` | E2 (t1 train / t2 val / t test) + E3 (few-shot on t) |
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

- **Splits**: `minutes/` labels `t1_*` = train, `t2_*` = validation;
  `test_minutes/` labels `t_*` = test. No sessions; placement = split.
- **Windows**: 50 consecutive radar frames (~5-7 s, long enough for
  breathing) inside one recording, non-overlapping. Never cross
  recordings/labels.
- **Radar timing**: `minutes/` — `.bin` filename is the capture
  timestamp; each file spans 1 s (frames at `ts + j/n`).
  `test_minutes/` — per-frame npz timestamps are burst-flushed; frames
  are spread uniformly inside their `second_start` bucket.
- **CSI**: one receiver per recording — the `wifi_csi_XX.csv` (or npz
  receiver index) with most samples inside the minute. Amplitude of the
  52 usable subcarriers, log1p, linearly interpolated onto a 128-step
  grid spanning the radar window; windows with <2 samples are zeroed +
  flagged, never fabricated. **t2 has no CSI coverage at all** — the
  csi-only baseline early-stops on a t1 holdout instead.
- **Radar maps**: per frame, Hann-windowed FFT -> range-Doppler
  (64x64) and range-azimuth (16x64), log1p, resized to 24x24.
- **Features**: the encoder pools each window along time into a
  per-pixel **temporal-std map** (motion/breathing energy), then
  spatially to mean/std/max per channel. Absolute levels flip sign
  across placements and are excluded — this is what makes the model
  transfer to an unseen room.
- **Normalization**: mean/std computed on t1 only (`--norm-mode local`
  switches to per-recording z-score).
- **Balancing**: training windows subsampled to equal class counts;
  early stopping + threshold use a class-balanced t2 subsample.
- **Augmentation**: doppler-sign/azimuth flip; 30% CSI dropout (fusion).
- **Threshold/early stopping/HP search**: validation (t2) only.

## Experiments

- **E2**: train on t1, validate on t2, test on t (unseen placement).
  Modalities: fusion / radar-only / csi-only. The best-val non-CSI
  modality (radar) is saved to `outputs/best_model.pt` for E3.
  Result: radar test acc **0.995** (minute level 1.000); fusion/csi
  do not transfer because CSI is receiver/placement-specific.
- **E3**: fine-tune the saved E2 model on 10 support minutes of t
  (5 per class), evaluate on the remaining query minutes vs the
  0-shot baseline. Strategies: head-only, full (fusion+head when the
  base model is fusion). Result: 0-shot 0.995 -> full fine-tune
  **0.996**.

## Outputs

`outputs/`: `inspection.json`, `hp_search.json`, `best_config.json`,
`results_e2.json`, `results_e3.json`, `e3_partition.json`,
`best_model.pt`, `REPORT.md`, `figs/*.png`.
