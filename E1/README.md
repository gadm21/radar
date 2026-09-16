# E1 — Dataset Audit

Exploratory inspection of the three dataset roots, `train_minutes/`,
`val_minutes/` and `test_minutes/`. Produces detailed tables and figures
on folder/file
completeness, manifest health, the label taxonomy, sensor coverage
(radar / Wi-Fi CSI / xy-tracking / Sense-HAT / camera), collection
timing, and a catalog of every missing or corrupt instance.

## Layout

| file | purpose |
|---|---|
| `common.py` | paths, tolerant manifest parsing, label taxonomy, `.bin`/`capture.npz`/CSV/JSON integrity helpers |
| `inspect_dataset.py` | full scan -> `outputs/inspection.json`, `outputs/per_minute.csv`, `outputs/problems.csv` |
| `plots.py` | figures -> `outputs/figs/` |
| `report.py` | `outputs/REPORT.md` |
| `run_all.py` | one-command pipeline |

## Run

```powershell
pip install -r E1/requirements.txt
python E1/run_all.py
```

## What is checked (per minute folder)

- **Folder**: file count, total bytes, leftover `*.tmp` files, empty folders.
- **Manifest**: presence; parse mode (`json` / `json_partial` / `regex` /
  `unparsable`); `status` field (`success` / `partial` / `collecting`);
  schema version; `warnings`/`errors`; `expected_chunks` vs on-disk
  `radar_*.bin` count; capture duration.
- **Labels**: full raw counts plus mapping to the task taxonomy —
  placement (`t1`/`t2`/`t`), activity (`empty`/`sleep`/`present`),
  position (`left`/`right`), `radar-missing` flag, auxiliary auto-labels
  (`absent`/`empty`/`present`/`occupied`); unlabeled and ambiguous
  folders are flagged.
- **Radar**: `radar_*.bin` — count, byte size, frame count
  (size / 36876 B frame), truncated/partial-frame files, unparseable
  filename timestamps. `capture.npz` — zip opens, required arrays
  present, radar/CSI/sense/camera sample counts.
- **CSI**: `wifi_csi*.csv` count, sizes, `CSI_DATA` line counts
  (header-only ~54 B receiver files detected).
- **Other sensors**: `xy-tracking.json` presence + truncation check
  (full `json.loads` on a sample), `.home_assistant_status.json`,
  `sense_hat.error.json`.

## Outputs

`outputs/`: `inspection.json` (aggregate), `per_minute.csv` (one row per
folder), `problems.csv` (one row per problem instance), `REPORT.md`,
`figs/*.png`.
