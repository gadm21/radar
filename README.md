# Radar Occupancy Detection

Binary room-occupancy detection (**empty** vs **occupied**) from mmWave
radar (Infineon BGT60TR13C via MMW-HAT) + Wi-Fi CSI (ESP32), targeting
on-device deployment on a Raspberry Pi ("thoth").

## Datasets

One folder per captured minute (`YYYYMMDD_HHMM`):

| split | folders | role |
|---|---|---|
| `train_minutes/` | 1394 | train — placements t1+t2 (chunked `radar_*.bin` + `wifi_csi*.csv`) |
| `train2_minutes/` | 329 | train — placement t, Sept 6-8 (`capture.npz`) |
| `validation_minutes/` | 194 | validation — placement t, Sept 15 |
| `test_minutes/` | 275 | test — Pi captures, Sept 16-17 night (labeled `present`/`empty` by the 1:10 AM boundary) |

Data directories are gitignored (large). See `REPORT.md` for results.

## Pipeline stages

- **E1** — dataset audit: file/manifest integrity, label taxonomy,
  sensor coverage, timing, problem catalog (`E1/outputs/REPORT.md`).
- **E2** — occupancy pipeline: preprocessing cache, CNN encoders
  (radar / CSI / gated fusion), the self-contained scene-model export,
  evaluation, figures, report (`E2/outputs/REPORT.md`).

## Reproduce

```powershell
pip install -r E2/requirements.txt
python E1/run_all.py                        # dataset audit (minutes)
python E2/run_all.py                        # full pipeline (~1-2 h CPU)
python E2/run_all.py --skip-preprocess      # reuse E2/cache/
```

## Deployment

`deploy/` holds TorchScript `.pt` artifacts with embedded `meta.json`
(thresholds, normalization, input spec) plus `*.meta.json` thoth-model
registry metadata. The `*_scene.pt` models compute all preprocessing
inside `forward`, so the runtime only feeds raw log1p maps.
