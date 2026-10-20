# Validation-frozen occupancy evaluation

Training has 273 minutes (3,276 five-second windows), validation has 192 minutes (2,304 windows), and multilink test has 360 minute buckets (4,320 windows) in four single-class captures. No synthetic training minutes are used in this matched comparison. Multilink CSI features average the two links; training and validation have one link.

`selection.json` was saved before multilink features were opened by `frozen_evaluation.py --phase evaluate`. Missing-value medians, scalers, and models use training windows only. Validation-minute AUC selects each view leader and the ML/DL references. Validation-minute balanced accuracy fixes each threshold. Late RF uses an equal, fixed CSI/radar probability weight. Test labels are used only for reported metrics. No test-batch scaling, rank, clustering, or threshold calibration is used.

| View | Selected model | Fixed threshold | Validation minute accuracy | Multilink test minute accuracy |
|---|---|---:|---:|---:|
| CSI | MLP | .429 | .953 | .589 |
| Radar | RF | .605 | .740 | .989 |
| Early feature fusion | GBM | .083 | .656 | .986 |
| Late probability fusion | RF | .620 | .839 | .969 |

The full-cohort LLM protocol is in `llm/protocol.json`; input row IDs are opaque, and each call receives nine physical descriptors, training-class statistics and examples, and the two frozen reference probabilities and predictions. The decision cutoff is .5. Each model was run three times over all 192 validation and 360 test minutes. GPT-5.4 was run after the prespecified GPT-4o mini test-mean gate exceeded .80.

| LLM | Validation accuracy, mean ± sample SD | Multilink test accuracy, mean ± sample SD |
|---|---:|---:|
| GPT-4o mini | .655 ± .032 | .924 ± .007 |
| GPT-5.4 | .932 ± .016 | .834 ± .014 |

The first-run test confusion matrices, with rows actual empty/occupied and columns predicted empty/occupied, are GPT-4o mini `[[178,2],[28,152]]`, GPT-5.4 `[[180,0],[54,126]]`, late RF `[[180,0],[11,169]]`, and CSI MLP `[[178,2],[146,34]]`. The vector visualization is `figs/frozen_llm_cm.pdf`. Three runs use the same minutes; their standard deviation is not a confidence interval over new captures.

A separate six-example explanation check samples three empty and three occupied multilink minutes across all four captures. It reuses the nine descriptors and frozen reference probabilities but asks each LLM for a short reason; its answers are excluded from the full-cohort accuracy estimates. The concise complete responses and selected CSI/radar descriptor values appear in paper Table II; the corresponding signal excerpts are in `paper/icc_assets/explainability_signals.pdf`. The request, response JSON, and sample selection are in `llm/explainability_*`.

API records contain 168 calls per LLM. GPT-4o mini used 310,014 prompt and 38,588 completion tokens; GPT-5.4 used 309,846 prompt and 24,046 completion tokens. No API key is written to these outputs.

The frozen code path fits no parameter on the multilink test. Validation has one session of each class, and multilink test has two captures of each class; each session or capture has one label. The reported test accuracy evaluates both classes across held-out captures. Minute buckets within a capture are correlated, and this protocol does not measure transitions within a capture. Earlier exploratory work inspected these outcomes, so additional independent captures would strengthen a confirmatory estimate.
