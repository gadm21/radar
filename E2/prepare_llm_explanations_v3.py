"""Choose six multilink minutes for the v3 descriptive LLM explanation check.
Selection uses only recorded class and capture ID with a fixed seed; the
request excludes both fields and carries the v3 descriptor set (nine physical
minute means plus twenty hierarchical PCA mean/std descriptors)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs/frozen_v3/llm"
protocol = json.loads((OUT / "protocol.json").read_text())
mapping = pd.read_csv(OUT / "test_mapping.csv")
rng = np.random.default_rng(20261015)

chosen = []
for label in (0, 1):
    groups = sorted(mapping.loc[mapping.label.eq(label), "unit"].unique())
    assert len(groups) == 2
    for group, count in zip(groups, (2, 1)):
        candidates = mapping[mapping.unit.eq(group)]
        chosen.extend(rng.choice(candidates.index.to_numpy(), count, replace=False))

selected = mapping.loc[chosen].copy().reset_index(drop=True)
selected[["id", "unit", "minute", "label"]].to_csv(OUT / "explainability_selection.csv", index=False)
refs = protocol["reference_models"]
num = lambda v: None if pd.isna(v) else round(float(v), 3)
minutes = []
for row in selected.itertuples(index=False):
    p_ml, p_dl = float(row.p_ml), float(row.p_dl)
    minutes.append({"id": row.id,
                    "descriptors": {c: num(getattr(row, c)) for c in protocol["descriptors"]},
                    "ml_probability": round(p_ml, 4), "ml_prediction": int(p_ml >= refs["ML"]["threshold"]),
                    "dl_probability": round(p_dl, 4), "dl_prediction": int(p_dl >= refs["DL"]["threshold"])})
request = {"reference_models": refs,
           "training_class_mean_std": protocol["training_class_mean_std"],
           "labeled_training_examples": protocol["labeled_training_examples"],
           "minutes": minutes}
(OUT / "explainability_request.json").write_text(json.dumps(request, indent=2))
print("Selected six sample IDs:", ", ".join(selected.id))
