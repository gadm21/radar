> Superseded exploratory pilot: its test-batch ranks are excluded from the current paper. See E2/outputs/frozen/llm/protocol.json and the full-cohort summaries for the fixed-input analysis.

# Two-reference LLM evaluation

The rerun uses 38 of 192 validation minutes (20% within each class) and 72 of
360 multilink test minute buckets (20% within each of four captures), selected
with seed 42. The source minutes and model probabilities are in
`E2/outputs/revision/llm_20pct_{val,test}_sample.csv`. The full prompt inputs
and model choices are recorded in `llm_20pct_protocol.json`; each response is
checkpointed in the corresponding predictions CSV. No API key is saved.

The two minute-level references are RF late probability fusion with unlabeled
batch descriptor rescaling and the raw CSI MLP. They were selected using the
full validation comparison. The prompt also includes nine physical CSI/radar
descriptors, training-class means and standard deviations, two labeled
training minutes per class, and each model's probability and unlabeled
full-batch percentile rank. GPT-5.4 returns one probability per minute.

Two prompt variants were run three times on the validation sample. The more
generic evidence-weighting prompt scored 97.4%, 92.1%, and 92.1%; the variant
that forced RF priority scored 68.4% in all three runs. The former was frozen
before any multilink test calls. Its multilink test accuracies were 81.9%,
86.1%, and 88.9% (mean 85.6%, sample SD 3.5 percentage points). The same
sample's RF late-fusion and CSI MLP percentile decisions scored 84.7% and
69.4%; an explicit average of their percentile ranks scored 94.4%.

The multilink sample contains 36 empty and 36 occupied minute buckets, but
each of the four captures is single-class. The 72 buckets therefore represent
four dependent capture units, not 72 independent transfer trials. Validation
is a development result after prompt selection; multilink test was queried
only with the frozen prompt. The direct rank average exceeds the LLM result,
so the rerun supports an improved LLM prompt, not an LLM fusion advantage.

To regenerate summaries and the vector confusion-matrix figure from cached
responses, run `E2/llm_20pct.py --summarize-only`. Live calls require
`JEV_API_KEY` or `--key-stdin` and were run with the user-provided credential.
