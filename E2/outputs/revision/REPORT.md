> Superseded exploratory analysis: it refits transforms or thresholds on evaluation batches and filters configurations using multilink test outcomes. Use E2/outputs/frozen/selection.json, valuation.json, and the current paper for the leakage-controlled results.

# Corrected E2 results

All primary models fit 3,276 real five-second windows from 273 training
minutes. Validation contains 2,304 windows / 192 minutes. The independent
multilink test contains 4,320 windows / 360 minute buckets from four
captures. Physical descriptors are aligned by name across all cohorts;
test-specific PCA or cluster fits are not used.

`physical_all.csv` has all 136 split-level comparisons. The 16 configurations
in `physical_over70_pairs.csv` exceed 70% on validation **and** test with
one shared unsupervised probability rule. Seven are minute decisions and
nine are window decisions. The rule in the table is median (`med`) or
two-means (`km`), applied separately to the unlabeled probability batch.

| Method / input | Level | Scale | Rule | Validation | Test |
|---|---|---|---|---:|---:|
| RF / late CSI + radar probability | minute | raw | med | .729 | 1.000 |
| RF / late CSI + radar probability | minute | rescaled | med | .906 | .856 |
| RF / early CSI + radar descriptors | minute | rescaled | med | .740 | .817 |
| SVM / early CSI + radar descriptors | minute | rescaled | med | .802 | .772 |
| MLP (64,32) / CSI | minute | raw | med | .917 | .733 |
| MLP (64,32) / stable radar | window | raw | med | .855 | .957 |
| SVM / stable radar | window | rescaled | med | .822 | .938 |
| GBM / early CSI + radar descriptors | window | rescaled | med | .859 | .848 |

The complete passing list, including additional RF, GBM, and MLP rows,
is in `physical_over70_pairs.csv`. The filtered figure is
`figs/e2_passing_models.png`. Descriptor importance for the best passing
ML and neural configurations is in `physical_shap.csv` and
`figs/physical_shap.png` (32 randomized feature orders, 32 balanced held-out
windows, training-background draws). These are Monte Carlo interventional
SHAP estimates of model reliance, not causal sensor importance.

## Separate 10% benchmark LLM pilot

Sixteen of 160 held-out minutes were fixed by seed 42, eight per class.
The three live runs produced:

| Model | Raw accuracy mean ± sample SD | Median calibrated | Two-means calibrated |
|---|---:|---:|---:|
| GPT-4o mini | .625 ± .000 | .688 ± .000 | .625 ± .000 |
| GPT-5.4 | .500 ± .000 | .375 ± .217 | .354 ± .095 |

On the same 16 minutes, median-calibrated RF MoE and RF CSI each scored
14/16 (.875), while radar LSTM scored 8/16 (.500). All pilot thresholds
use only the same 16 unlabeled probabilities. The first-run,
same-sample confusion matrices are in `figs/llm_vs_ml_dl_cm.png`.
The LLM pilot is small and balanced by design. Its source is a subset of the
same 192 recordings used for primary E2 validation, not an independent
evaluation cohort or part of the four-capture multilink test.

The separate five-seed RF/CSI five-second minute test remains .953 ± .006
under median calibration; RF MoE is .973 ± .010. These are separate
160-minute-cohort results from `legacy_over70.csv`.

All results are exploratory. The four multilink test captures confound
label and recording-session effects; nearby windows/minutes are correlated.
Unlabeled batch rescaling and thresholds need a deployment batch, and the
median rule presumes a roughly balanced mix. The >70% display filter
inspects test outcomes and must not be treated as prospective model selection.

## Reviewer audit and augmentation sensitivity

The primary E2 comparison uses no synthetic rows. The separate RF/MoE
benchmark splits original minutes before 601 synthetic empty fit minutes are
created. The saved lineage audit finds no augmented parent or mixup partner in
its internal validation split, and the saved PCA fit counts match real fit
seconds. However, all nine fit sessions also contribute internal validation
minutes.

In a same-code five-seed rerun on the same balanced 160-minute benchmark
holdout, median-calibrated RF/CSI accuracy is .9525 with augmentation and
.7500 without; RF/MoE is .9725 with and .6725 without. The comparison is
retrospective and does not measure transfer to a new session. See
`reviewer_audit/augmentation_lineage.csv`, `with_augmentation_metrics.csv`,
`no_augmentation_metrics.csv`, and the companion paper review response.

Each of the four multilink captures has only one class (two empty, two
occupied), and the 192-minute validation cohort is likewise split across one
empty and one occupied session. Thus capture and occupancy labels are
confounded. The scores cannot establish occupancy discrimination within a
capture; new mixed-class sessions across repeated placements are needed.
