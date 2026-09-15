# E4 — Hierarchical Activity + Position Pipeline

Single inference function mapping a `capture.npz` recording to a combined
label: **empty** | **sleep** | **present(left)** | **present(right)**.

```python
from E4.predict import predict_capture
label = predict_capture("test_minutes/20260906_2317/capture.npz")
```

## Task

Compose three radar models into a hierarchy:

1. **Occupancy** (empty vs occupied) — E2's exported `best_model.pt`
   (radar-only, trained on t1 / validated on t2). Not retrained.
2. **Sleep vs present** — binary model trained on occupied t1+t2
   recordings, run only when stage 1 says occupied.
3. **Left/right** — E3-style model trained on t (the only placement with
   left/right labels), run only when stage 2 says present. Only
   `t_present` recordings carry left/right ground truth, so they are the
   only data that can validate location.

## Layout

| file | purpose |
|---|---|
| `common.py` | label parsing, `capture.npz` frame decode, parameterized FFT maps + windowing |
| `models.py` | `OccupancyModel` (loads E2 export), `SleepPresentModel`, `PositionModel` (E3-style CNN) |
| `train.py` | evaluates E2 export on t; trains sleep/present on occupied t1+t2; trains position on all t |
| `predict.py` | `predict_capture(npz_path)` — the 3-stage inference function |
| `evaluate.py` | runs `predict_capture` on every `test_minutes` npz and reports accuracy |
| `run_all.py` | one-command pipeline |

## Run

```powershell
# prerequisites: E2 and E3 caches + E2 exported model must exist
python E2/run_all.py
python E3/run_all.py

python E4/run_all.py
```

## Results

Per-stage accuracy on `test_minutes/` (placement t, held out from stages
1-2):

| stage | task | train | test | window acc | minute acc |
|---|---|---|---|---|---|
| 1 | empty vs occupied | t1 (val t2) | t | **0.995** | **1.000** |
| 2 | sleep vs present | occupied t1+t2 | occupied t | 0.374 | 0.369 |
| 3 | left vs right | t (5-fold CV) | t | **0.989** | **1.000** |

End-to-end `predict_capture` on the 267 `test_minutes` recordings with a
`capture.npz`:

| metric | value |
|---|---|
| combined 4-way accuracy | 0.801* |
| activity 3-way accuracy (empty/sleep/present) | 0.801 |
| position coverage (position emitted on the 65 labeled `t_present` recs) | 0.215 |
| position accuracy when emitted | **1.000** |

\* dominated by stage-2 errors: the occupancy stage is nearly perfect and
the position stage is perfect when invoked, but sleep/present does not
transfer across placements — the occupied class prior inverts (train
≈85% sleep, test ≈82% present) and the temporal-std features that make
occupancy transfer do not separate sleep from present on an unseen
placement. This is the pipeline's documented weak point.

### Plots

![Per-stage accuracy](outputs/figs/stage_accuracy.png)
![Combined confusion](outputs/figs/combined_confusion.png)

## Outputs

`saved_models/`: `sleep_present_model.pt`, `position_model.pt` (weights +
norm stats + preprocessing spec). Stage 1 loads `E2/outputs/best_model.pt`
directly.
`outputs/`: `results_e4.json`, `predict_capture_eval.json`.
