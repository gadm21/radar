# E2 — occupancy model comparison

The current study is `physical_comparison.py` followed by
`summarize_physical.py` and `physical_shap.py`. It fits all models on the
same real training descriptor windows, evaluates on the same
validation windows, and tests on the same four multilink captures. The
descriptor columns have the same physical meaning across cohorts; the
historical separately fitted PCA and cluster coordinates are excluded.

`outputs/revision/physical_all.csv` contains every window/minute,
single-modality/early-fusion/late-fusion, model, scaling, and calibration
result. `physical_over70_pairs.csv` lists only configurations above 70% on
**both** validation and test with the **same** median or two-means rule.
`physical_over70.csv` contains the paired split rows. This test-inspecting
display filter is descriptive, not an unbiased selection estimate.

The leading minute result is raw RF late probability fusion, 72.9%
validation and 100.0% test with median calibration. The leading neural
result is the raw two-hidden-layer MLP on stable radar descriptors at
five-second window level, 85.5% validation and 95.7% test. The four
test captures, not their 360 minute buckets or 4,320 overlapping windows,
are the independent acquisition units. Both near-perfect scores require
prospective session replication.

The current LLM comparison is `llm_20pct.py`: 38 validation minutes for
prompt selection, then 72 multilink test minute buckets with the prompt
frozen. GPT-5.4 receives physical descriptors, training statistics and
labeled training examples, and probabilities from both RF late fusion and
CSI MLP. It scores 85.6% ± 3.5% across three multilink runs; a transparent
average of the two model ranks scores 94.4% on the same sample. The old
`llm_10pct.py` pilot used only a single RF-MoE probability and is superseded.
No API key is stored in the workspace.

Use the bundled Python environment with `E2/.deps` on `PYTHONPATH` to
reproduce the analysis. `outputs/revision/REPORT.md` gives the measured
values and limitations. Historical `model_compare.py` outputs that used
test-refitted PCA are superseded and must not be cited.

`audit_reviewer.py`, `no_augmentation_ablation.py`, and
`audit_paired_errors.py` provide the saved-data provenance and benchmark
augmentation audit. All four multilink captures and both validation sessions
are single-class, so current accuracy can reflect capture conditions. See
`paper/REVIEW_RESPONSE.md` before interpreting transfer claims.

## Revision 3 (hierarchical PCA)

`frozen_v3.py` replaces the twelve single-basis PC descriptors with a
1 s / 5 s / 15 s hierarchical PCA (two components per level, each fitted on
training output of the level below); classifiers use fifteen-second windows
(four per minute). Outputs are in `outputs/frozen_v3`, the paper draft is
`paper/icc_multimodal2.tex` with assets in `paper/icc_assets_v3`. LLM steps:
`llm_full_frozen_v3.py --prepare-only` (done), then `--model gpt-4o-mini`,
`--model gpt-5.4`, `prepare_llm_explanations_v3.py`,
`run_llm_explanations_v3.py --model ...`, `export_explainability_rows_v3.py`.
The v2 outputs are retained unchanged.
