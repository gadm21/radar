# Revision plan: `icc_multimodal2_overleaf` (repositioning + stray capture)

Status: **implemented with choice A (capture3 = occupied)**, 6 pages. Numbers in §0 are pre-implementation; final: GPT-4o mini 88.2±1.4%, GPT-5.4 76.6±0.7% on 324 minutes. Pre-capture3 artifacts are backed up in `newtest/paper_eval/_pre_capture3/`.
Target: 6 pages (current source compiles to 7: two references spill onto p.7).

---

## 0. Decision needed first: label of `capture3_768c3d688716`

Evidence figure: `newtest/paper_eval/capture_snr_boxplot.png` (made by `E2/plot_capture_snr.py`)

Facts:
- Wall 1, recorded Oct 19 06:08–06:28 (20 min, 80 windows). It falls between `empty_2fa6d3f1bf90` (04:26) and `empty_whispy` (07:02).
- Its manifest label is the literal string `"capture3"`; no class is recorded anywhere.

Radar SNR per 15-s window (median / 75th percentile of SNR max, and the share of windows with SNR max > 12 dB):

| group | n | SNR-max median | SNR-max q75 | share > 12 dB |
|---|---|---|---|---|
| train empty (pooled) | 748 | 9.70 | 10.02 | 0.000 |
| validation empty (pooled) | 1168 | 9.78 | 10.18 | 0.004 |
| empty_2fa6d3f1bf90 | 240 | 9.88 | 10.10 | 0.000 |
| empty_whispy | 207 | 9.99 | 10.21 | 0.000 |
| **capture3 (unlabeled)** | 80 | 10.23 | **21.12** | **0.325** |
| train occupied (pooled) | 344 | 9.96 | 10.88 | 0.198 |
| validation occupied (pooled) | 1040 | 10.31 | 14.57 | 0.298 |
| occupied_whispy | 208 | 10.04 | 10.32 | 0.058 |
| occupied_whispy2 | 160 | 15.97 | 29.10 | 0.656 |
| occupied_whispy11 | 286 | 9.79 | 10.05 | 0.035 |
| occupied_whispy12 | 120 | 12.40 | 20.58 | 0.517 |

Interpretation:
- No empty capture, in any split, has more than 0.4% of windows above 12 dB. Capture3 has 32.5%.
- On this signal, capture3 looks occupied with movement.
- A low SNR does **not** prove a capture is empty: `occupied_whispy` and `occupied_whispy11` are occupied but look empty-like (likely a still subject).
- **The label is your call. It must come from your collection notes, not from this plot.**

Options:
- **A. Occupied:** the test set becomes 7 captures, 324 min (111 empty / 213 occupied), Oct 19–20.
- **B. Empty:** the test set becomes 7 captures, 324 min (131 empty / 193 occupied).
- **C. Keep excluded:** add one sentence saying the capture is excluded because no label was recorded. This conflicts with your "no capture excluded" requirement.

Work after A or B (one script pass, plus the API cost of 20 minutes × 3 runs × 2 LLMs):
1. Add capture3 with its label to `CAPS` in `E2/eval_newtest_paper.py`.
   - Rerun it. This regenerates test windows, minute probabilities, `evaluation.json`, and `capture_summary.json`.
   - Thresholds and model selection do not change, because they are fit on validation only.
2. `E2/llm_newtest.py --prepare-only`, then the runs for both models. Test-only, per your earlier instruction.
   - The six explainability samples stay as they are, since they are already chosen and answered.
3. `E2/plot_newtest_paper.py` regenerates Fig. 2 (sizes), Fig. 3 (RA/RE), Fig. 4 (hierarchical PCA), Fig. 5 (accuracy), Fig. 7 (LLM confusion matrices), and Table IV rows.
4. Update every number in the text:
   - §V-B counts.
   - Table II test row.
   - Every test accuracy/AUC in the abstract, §VIII, and the conclusion.
   - "six captures" → "seven".
   - "304" → "324".
   - The limitations replication count.

---

## 1. Framing: calibrate per room, then placement-robust within the room

### What the paper can claim (consistent with the data)
| Workflow stage | Instantiated in this paper by |
|---|---|
| Pretrained occupancy model (from previous environments) | The frozen models trained here (`E2/outputs/frozen_v3/models`: RF/GBM/SVM/MLP × CSI/radar/fusion × 5 seeds, plus hierarchical PCA bases, scalers, and medians), **released on GitHub** as the starting point for a new room. |
| New room: few-session calibration | Walls 2+4: about 6 empty and 6 occupied sessions, 273 recorded minutes (86 occupied), Sep 6–8. Wall 3 sets the decision thresholds. |
| Placement-robust inference | Wall 1: unseen position, **no** refit, threshold change, or test-batch statistics. |

Caveats to state, briefly, in Limitations:
- Adapting the released model to a *different* room (fine-tuning from it) is enabled but not evaluated here. This paper evaluates the calibrate-then-move stage.
- The claim is placement robustness for radar-based views and the LLM. CSI-only does **not** transfer (.373). Present that as a finding, not something to hide.
- One held-out placement.
- 32 synthetic occupied minutes are in the threshold-setting (validation) set.
- Use "few-session" calibration rather than "few-shot". 273 minutes is not few-shot in the ML sense. The selling point is the cost: about a dozen sessions per room.

### Text edits
- **Title** (no "Calibrate Once per Room:" prefix):
  *Placement-Robust Occupancy Detection with WiFi CSI, FMCW Radar, and LLM Reasoning*
- **Abstract:**
  - Replace "robust and explainable placement-invariant indoor occupancy detection system" with: "an occupancy detector that is calibrated per room from a few labeled empty and occupied sessions and then operates, without recalibration, at sensor placements unseen during calibration."
  - Add one sentence on the calibration cost (about six sessions per class, 86 occupied minutes).
  - Add one sentence that CSI alone does not survive the placement change while radar, fusion, and GPT-4o mini do.
  - Add that the calibrated models are released as a pretrained starting point.
  - Cut the sentence-long PCA detail from the abstract; it saves about 3 lines.
- **Intro, contribution paragraph (¶4):**
  - Restate the contribution as the two-part workflow: (i) environment-specific calibration, (ii) placement robustness within the calibrated room.
  - Cite the new figure.
  - Mention the released pretrained models (GitHub link as a footnote).
- **Intro roadmap paragraph ("The remainder of the paper…"):** delete (about 6 lines).
- **§III Problem definition:**
  - Define the calibration set C = {walls 2, 4 (fit); wall 3 (thresholds)}.
  - Define the evaluation placement p* = wall 1, "without further calibration".
  - The existing math stays.
- **§VIII-G LLM evaluation:**
  - Delete "and its own .61 score under the previous references" and the "inverts the earlier protocol's" sentence. Readers never see that protocol.
  - Replace with the verified mechanism:
    - All GPT-5.4 errors are missed occupied minutes, and 61 of its 69 misses are in one capture (`occupied_whispy11`).
    - On that capture the ML reference votes occupied in only 10% of minutes.
    - GPT-5.4 tracks the references more closely (r = .95 vs .83 with the ML reference probability), so it inherits their error.
    - Without that capture the two LLMs tie (97.4% vs 96.6%).
- **§VIII-H explanation check:** add a half-sentence noting that both LLMs report the RF reference probability as their own in this prompt. Otherwise identical bold numbers in Table III look like a copy error.
- **§IX Limitations:**
  - Turn "requires per-user calibration" into the stated scope: a new room requires calibration.
  - Add the caveats listed above.
- **§X Conclusion:** mirror the abstract and end with: calibrate per room from a few sessions, then move the sensors without recalibration (radar-based views and the LLM).
- **Small fixes:**
  - "automation,security" → add the space.
  - "wearable require" → "wearables require".
  - Lowercase ml/llm/rpm → ML/LLM/RPM.
  - "$.806\pm.00$" and "$.791\pm.00$" → drop the ±.00, following the table convention of omitting sd < .005.

---

## 2. New figure: occupancy training/deployment workflow

Single column, TikZ (to match Fig. 1), about 3.3 cm tall. Placed `[!t]` on page 2 and cited from the intro.

```
 ┌──── Environment-specific calibration (per room) ────┐┌─ Placement invariance (within room) ─┐
 ┌──────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────────┐
 │ Pretrained occupancy │──▶│ New room: few-session    │──▶│ Placement-robust inference   │
 │ model                │   │ calibration              │   │                              │
 │ learned from previous│   │ short empty + occupied   │   │ unseen sensor positions,     │
 │ environments         │   │ captures; fine-tuning,   │   │ no further calibration       │
 │                      │   │ classifier retraining,   │   │                              │
 │                      │   │ or feature/threshold cal.│   │                              │
 └──────────────────────┘   └──────────────────────────┘   └──────────────────────────────┘
  this paper: released      this paper: walls 2+4 fit      this paper: wall 1, 7 captures,
  frozen models (GitHub)    (~12 sessions, 273 min),        324 min, no updates
                            wall 3 thresholds
```

Visual design:
- Two top braces label part 1:
  - The left brace spans boxes 1–2: "Environment-specific calibration: a new room may require calibration or retraining."
  - The right brace spans box 3: "Placement invariance: after calibration, sensors can move to new positions."
- The three boxes carry the part-2 text verbatim.
- A grey italic strip below maps each stage to this paper's data, so a reviewer sees what was evaluated.
- Colors reuse csiBlue, radarOrange, and fusionRed from Fig. 1: training placement blue, validation orange, test red. This ties the figure to the room diagram.
- Caption (2 lines): "Proposed workflow. A model pretrained in earlier rooms is calibrated once in a new room from a few labeled sessions; afterwards it is evaluated at sensor placements not used for calibration, with no further updates. Bottom: how each stage is instantiated in this study."

---

## 3. Fixes found during review

- *done:* Fig. 2 validation bar showed 114/82 (228 min) because `capture_summary.json` counted capture IDs, not minutes. It now shows 292/260 (+32) = 584 min, matching Table II. The fix is in `E2/eval_newtest_paper.py` and `E2/plot_newtest_paper.py`.
- Table V (SHAP) is 15 pt too wide. Shorten descriptor names or set `\tabcolsep` to 1.5 pt.
- Fig. 5 uses `[H]` and leaves a large gap at the bottom of p.4. Change it to `[!t]`.
- Fig. 3 (RA/RE box plots) is unreadable at its current size.

---

## 4. Page budget (from 7 pages → 6 with the new figure)

| Change | Approx. gain |
|---|---|
| Delete intro roadmap paragraph | +6 lines |
| Trim PCA detail from abstract | +3 lines |
| Remove old-protocol sentences in §VIII-G | +3 lines |
| Fig. 5 `[H]` → `[!t]` (removes the p.4 gap) | +5–8 lines |
| Fig. 3: shrink to 0.8 column or drop (Fig. 4 already shows the shift) | +6–12 lines |
| New workflow figure | −14 lines |
| New limitation/framing sentences | −5 lines |

Net: about 0 to +10 lines of slack. Remaining fallback if needed: drop the WiFi block of Table I into one text sentence (+4 lines).

---

## 5. Order of implementation (after you choose A/B/C)
1. Rerun the evaluation, LLM, and figure scripts with capture3 included (A/B).
2. Write the workflow TikZ figure.
3. Make the text edits in §1, then the fixes in §3.
4. Compile, trim to exactly 6 pages, then check every page visually and search for stale numbers (304, "six captures", .61, "earlier protocol").
