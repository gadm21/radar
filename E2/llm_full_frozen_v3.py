"""Full-cohort LLM evaluation against the v3 frozen references.

Same request structure as llm_full_frozen_v2.py (one system message with the
rule, one user message with the JSON body). The descriptor list holds the
nine physical minute means plus the twenty hierarchical PCA descriptors
(per modality: 1-s PC1/PC2 mean and std, 5-s PC1/PC2 mean and std, 15-s
PC1/PC2), averaged over the minute's four fifteen-second windows. References
come from E2/outputs/frozen_v3 (seed-0 models, validation-fixed thresholds).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from frozen_v3 import OUT as FROZEN, CSI_H, RAD_H, read_test, sha256
from jev import _post

OUT = FROZEN / "llm"
OUT.mkdir(parents=True, exist_ok=True)
PHYS = ["csi_amp_mean", "csi_amp_std", "csi_rv_mean", "snr_mean",
        "ra_std90", "re_std90", "xy_std90", "ra_delta", "re_delta"]
DESC = PHYS + CSI_H + RAD_H
BATCH = 10
SYSTEM = (
    "Classify room occupancy (empty=0, occupied=1) using physical WiFi CSI "
    "and radar descriptors, hierarchical training-basis PCA descriptors, and "
    "two trained model references. The PCA descriptors give, for CSI and "
    "radar, the mean and standard deviation of one-second (l1) and "
    "five-second (l2) PC1/PC2 coordinates and the fifteen-second (l3) PC1/PC2 "
    "coordinates, each averaged over the minute. Return only a JSON object "
    "with predictions: an array of {id,occupied_probability} for every input "
    "ID. The decision cutoff on your output is fixed at 0.5. The supplied "
    "reference thresholds and validation performances were fixed before "
    "multilink evaluation. Respect agreement of the two reference class "
    "predictions; when they disagree, consider their margins from the fixed "
    "thresholds and the training-class descriptor statistics. Use one "
    "consistent rule for all cases. IDs are arbitrary. Do not infer class "
    "prevalence from the batch or ordering."
)


def load_split(split, reference):
    windows = pd.read_csv(FROZEN / "validation_windows.csv") if split == "validation" else read_test()
    frame = windows.groupby(["unit", "minute"], sort=True).agg(
        label=("label", "first"), **{c: (c, "mean") for c in DESC})
    for family in ("ML", "DL"):
        if split == "validation":
            source = FROZEN / f"validation_{family.lower()}_probabilities.csv"
        else:
            key = reference[family]["key"]
            source = FROZEN / ("test_" + key.replace("|", "__") + "_probabilities.csv")
        p = pd.read_csv(source).set_index(["unit", "minute"])
        if not frame.index.equals(p.index):
            raise ValueError(f"Probability row alignment failed for {family} {split}")
        if not np.array_equal(frame.label.to_numpy(int), p.label.to_numpy(int)):
            raise ValueError("Probability labels do not align")
        frame[f"p_{family.lower()}"] = p.probability.to_numpy(float)
    return frame


def num(v):
    return None if pd.isna(v) else round(float(v), 3)


def training_context():
    train = pd.read_csv(FROZEN / "train_windows.csv")
    train = train[train.aug == 0]
    stats, examples = {}, []
    for lab, group in train.groupby("label"):
        label = ("empty", "occupied")[int(lab)]
        stats[label] = {c: [num(group[c].mean()), num(group[c].std())] for c in DESC}
        for _, row in group.groupby("unit", sort=True).mean(numeric_only=True).head(2).iterrows():
            examples.append({"class": label, "descriptors": {c: num(row[c]) for c in DESC}})
    return stats, examples


def prepare():
    selection = json.loads((FROZEN / "selection.json").read_text())
    refs = selection["references"]
    stats, examples = training_context()
    rng = np.random.default_rng(20261014)
    for split in ("validation", "test"):
        frame = load_split(split, refs)
        frame = frame.iloc[rng.permutation(len(frame))].copy()
        frame.insert(0, "id", [f"m{i:04d}" for i in range(len(frame))])
        frame.reset_index().to_csv(OUT / f"{split}_mapping.csv", index=False)
    context = {"protocol": "Balanced validation cohort and 360-minute multilink test. Train-only model scalers and hierarchical PCA bases; mean validation AUC over seeds selects reference models; seed-0 models and validation-fixed thresholds supply probabilities; descriptors are minute means over four fifteen-second windows (missing values as null); no test-batch calibration, statistics, percentile rank, or source ID in API input.",
               "selection_sha256": sha256(FROZEN / "selection.json"),
               "evaluation_sha256": sha256(FROZEN / "evaluation.json"),
               "reference_models": {f: {"key": refs[f]["key"], "threshold": refs[f]["threshold"],
                                        "validation_balanced_accuracy": refs[f]["balanced_accuracy"]}
                                    for f in ("ML", "DL")},
               "descriptors": DESC, "training_class_mean_std": stats,
               "labeled_training_examples": examples, "system_prompt": SYSTEM,
               "decision_threshold": .5, "runs": 3, "batch_size": BATCH,
               "model_order": ["gpt-4o-mini", "gpt-5.4"],
               "second_model_gate": "mini mean multilink-test accuracy across three runs > 0.80"}
    (OUT / "protocol.json").write_text(json.dumps(context, indent=2))
    _, rows = rows_for("test", context)
    chars = len(json.dumps(request_body(rows[:BATCH], context), separators=(",", ":")))
    print(json.dumps({"references": context["reference_models"],
                      "first_test_batch_user_chars": chars,
                      "approx_input_tokens_per_request": int(chars / 3.2) + 200}, indent=2))


def rows_for(split, context):
    df = pd.read_csv(OUT / f"{split}_mapping.csv")
    refs = context["reference_models"]
    rows = []
    for r in df.itertuples(index=False):
        pml, pdl = float(r.p_ml), float(r.p_dl)
        rows.append({"id": r.id, "descriptors": {c: num(getattr(r, c)) for c in DESC},
                     "ml_probability": round(pml, 4), "ml_prediction": int(pml >= refs["ML"]["threshold"]),
                     "dl_probability": round(pdl, 4), "dl_prediction": int(pdl >= refs["DL"]["threshold"])})
    return df, rows


def request_body(batch, context):
    return {"reference_models": context["reference_models"],
            "training_class_mean_std": context["training_class_mean_std"],
            "labeled_training_examples": context["labeled_training_examples"],
            "minutes": batch}


def request_batch(key, model, run, batch, context):
    payload = {"model": model, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": json.dumps(request_body(batch, context),
                                                                   separators=(",", ":"))}],
               "max_completion_tokens": 3000}
    if model == "gpt-4o-mini":
        payload.update({"temperature": .2, "seed": 27000 + run})
    data = _post(os.environ.get("JEV_BASE_URL", "https://api.openai.com/v1") + "/chat/completions",
                 {"Authorization": "Bearer " + key, "Content-Type": "application/json"}, payload)
    pred = json.loads(data["choices"][0]["message"]["content"])["predictions"]
    lookup = {str(r["id"]): float(r["occupied_probability"]) for r in pred}
    if set(lookup) != {r["id"] for r in batch}:
        raise ValueError("API returned mismatched or duplicate IDs")
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in lookup.values()):
        raise ValueError("API probability outside [0,1]")
    return lookup, data.get("usage", {})


def result_split(model, split):
    df = pd.read_csv(OUT / f"{split}_mapping.csv")
    pred = pd.read_csv(OUT / f"{model}_{split}_predictions.csv")
    if len(pred) != 3 * len(df):
        raise ValueError("Incomplete predictions")
    result = []
    for run in (1, 2, 3):
        p = pred[pred.run == run].set_index("id").loc[df.id].probability.to_numpy(float)
        y = df.label.to_numpy(int)
        yp = (p >= .5).astype(int)
        result.append({"run": run, "n": len(df), "accuracy": float(accuracy_score(y, yp)),
                       "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
                       "confusion_matrix": confusion_matrix(y, yp, labels=[0, 1]).tolist()})
    a = np.array([r["accuracy"] for r in result])
    return {"runs": result, "mean_accuracy": float(a.mean()),
            "sample_std_accuracy": float(a.std(ddof=1))}


def run_model(model, key, waive_gate=False):
    waived = False
    if model == "gpt-5.4":
        pilot = json.loads((OUT / "gpt-4o-mini_summary.json").read_text())
        if pilot["test"]["mean_accuracy"] <= .80:
            if not waive_gate:
                raise RuntimeError("GPT-4o mini did not exceed the prespecified 80% test gate")
            waived = True
            print("WARNING: running GPT-5.4 although the 0.80 pilot gate failed "
                  f"(mini test mean {pilot['test']['mean_accuracy']:.3f})", flush=True)
    context = json.loads((OUT / "protocol.json").read_text())
    for split in ("validation", "test"):
        df, rows = rows_for(split, context)
        path = OUT / f"{model}_{split}_predictions.csv"
        stored = pd.read_csv(path).to_dict("records") if path.exists() else []
        done = {(int(x["run"]), str(x["id"])) for x in stored}
        usage_path = OUT / f"{model}_usage.jsonl"
        for run in (1, 2, 3):
            for j in range(0, len(rows), BATCH):
                batch = [x for x in rows[j:j + BATCH] if (run, x["id"]) not in done]
                if not batch:
                    continue
                got, usage = request_batch(key, model, run, batch, context)
                for r in batch:
                    stored.append({"run": run, "id": r["id"], "probability": got[r["id"]]})
                    done.add((run, r["id"]))
                pd.DataFrame(stored).to_csv(path, index=False)
                with usage_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"split": split, "run": run, "batch_start": j,
                                        "n": len(batch), "usage": usage}) + "\n")
            print(f"{model} {split} run {run}: {len(done)}/{3 * len(rows)}", flush=True)
        if len(done) != 3 * len(rows):
            raise ValueError("Incomplete model run")
    summary = {s: result_split(model, s) for s in ("validation", "test")}
    if waived:
        summary["gate_waived"] = True
    (OUT / f"{model}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({s: {"mean_accuracy": summary[s]["mean_accuracy"],
                          "sd": summary[s]["sample_std_accuracy"]}
                      for s in ("validation", "test")}, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--model", choices=("gpt-4o-mini", "gpt-5.4"))
    ap.add_argument("--key-stdin", action="store_true")
    ap.add_argument("--waive-gate", action="store_true",
                    help="run gpt-5.4 even if the mini pilot gate failed")
    args = ap.parse_args()
    if args.prepare_only:
        prepare()
    elif args.model:
        key = (getpass.getpass("API key: ") if sys.stdin.isatty() else
               sys.stdin.readline().strip()) if args.key_stdin else os.environ.get("JEV_API_KEY", "")
        if not key:
            raise RuntimeError("API key is required for live calls")
        run_model(args.model, key, waive_gate=args.waive_gate)
    else:
        ap.error("Select --prepare-only or --model")
