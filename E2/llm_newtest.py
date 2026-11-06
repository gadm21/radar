"""LLM evaluation on the newtest set (wall 1) under the revised protocol.

Validation = old validation + old multilink test (NOT re-evaluated by the
LLM per user instruction); test = labeled newtest minutes. Reference
probabilities are seed-0 frozen_v3 model outputs with thresholds refit on
the combined validation (newtest/paper_eval/selection.json).

Usage:
  python E2/llm_newtest.py --prepare-only
  JEV_API_KEY=... python E2/llm_newtest.py --model gpt-4o-mini
  JEV_API_KEY=... python E2/llm_newtest.py --model gpt-5.4 --waive-gate
  JEV_API_KEY=... python E2/llm_newtest.py --explain --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "E2" / "outputs" / "frozen_v3"
EVAL = ROOT / "newtest" / "paper_eval"
sys.path.insert(0, str(ROOT.parent / "thothcraft" / "whispy" / "packages"
                       / "whispy-model-occ"))
sys.path.insert(0, str(ROOT / "E2"))
from whispy_model_occ import hier          # noqa: E402
from jev import _post                       # noqa: E402

OUT = EVAL / "llm"
PHYS = ["csi_amp_mean", "csi_amp_std", "csi_rv_mean", "snr_mean",
        "ra_std90", "re_std90", "xy_std90", "ra_delta", "re_delta"]
DESC = PHYS + hier.hier_cols("csi") + hier.hier_cols("rad")
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
    "test evaluation. Respect agreement of the two reference class "
    "predictions; when they disagree, consider their margins from the fixed "
    "thresholds and the training-class descriptor statistics. Use one "
    "consistent rule for all cases. IDs are arbitrary. Do not infer class "
    "prevalence from the batch or ordering."
)
EXPLAIN_SYSTEM = (
    "You are assessing six preselected room-occupancy minutes from WiFi CSI "
    "and FMCW radar. Use only the supplied physical descriptors, the "
    "hierarchical training-basis PCA descriptors (mean/std of one-second and "
    "five-second PC1/PC2 and the fifteen-second PC1/PC2; null marks missing "
    "values), training-class statistics/examples, and two fixed model "
    "probabilities and predictions. The input does not contain raw waveforms "
    "or images, so do not claim to have inspected them. For each ID return "
    "occupied_probability (0..1), predicted_class (empty or occupied), and a "
    "reason of at most 22 words naming specific CSI evidence, radar evidence, "
    "and the reference-model outputs. Explain conflicts or weak evidence "
    "honestly. Do not use sample ordering or infer prevalence. Return a JSON "
    "object with a results array."
)

REFCOL = {"CSI": "CSI", "radar": "radar", "early fusion": "early_fusion",
          "late probability fusion": "late"}


def num(v):
    return None if pd.isna(v) else round(float(v), 3)


def test_frame():
    """Labeled newtest minutes -> descriptor means + reference probs."""
    w = pd.read_csv(EVAL / "windows.csv")
    sizes = w.groupby(["unit", "minute"]).size()
    full = sizes[sizes == 4].index
    sub = w.set_index(["unit", "minute"]).loc[full]
    frame = sub.groupby(["unit", "minute"], sort=True).agg(
        label=("label", "first"), **{c: (c, "mean") for c in DESC})
    z = np.load(EVAL / "test_minute_probabilities.npz")
    probs = pd.DataFrame(z["probs"], columns=list(z["columns"]),
                         index=frame.index)
    sel = json.loads((EVAL / "selection.json").read_text())
    refs = sel["references"]
    for fam in ("ML", "DL"):
        mname, vname = refs[fam]["key"].split("|")
        col = f"p_{mname}_{REFCOL[vname]}_0"
        frame[f"p_{fam.lower()}"] = probs[col].to_numpy(float)
    return frame, refs


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, refs = test_frame()
    old_proto = json.loads((FROZEN / "llm" / "protocol.json").read_text())
    rng = np.random.default_rng(20261014)
    path = OUT / "test_mapping.csv"
    if path.exists():
        # keep IDs (and stored predictions) of minutes already mapped;
        # newly added minutes get fresh shuffled IDs appended at the end
        old = pd.read_csv(path).set_index(["unit", "minute"])["id"]
        known = frame.index.isin(old.index)
        new = frame[~known]
        new = new.iloc[rng.permutation(len(new))].copy()
        new.insert(0, "id", [f"m{i:04d}" for i in
                             range(len(old), len(old) + len(new))])
        kept = frame[known].copy()
        kept.insert(0, "id", old.loc[kept.index].to_numpy())
        frame = pd.concat([kept.loc[old.index.intersection(kept.index,
                                                           sort=False)],
                           new])
    else:
        frame = frame.iloc[rng.permutation(len(frame))].copy()
        frame.insert(0, "id", [f"m{i:04d}" for i in range(len(frame))])
    frame.reset_index().to_csv(path, index=False)
    context = {
        "protocol": f"{len(frame)} labeled newtest minutes (wall 1). "
                    "Train-only model "
                    "scalers and hierarchical PCA bases; seed-0 reference "
                    "probabilities with thresholds refit on the combined "
                    "validation (old validation + old multilink test); "
                    "descriptors are minute means over four fifteen-second "
                    "windows (missing values as null); no test-batch "
                    "calibration, statistics, percentile rank, or source ID "
                    "in API input.",
        "reference_models": refs,
        "descriptors": DESC,
        "training_class_mean_std": old_proto["training_class_mean_std"],
        "labeled_training_examples": old_proto["labeled_training_examples"],
        "system_prompt": SYSTEM,
        "decision_threshold": .5, "runs": 3, "batch_size": BATCH,
        "model_order": ["gpt-4o-mini", "gpt-5.4"],
        "second_model_gate": "mini mean test accuracy across three runs "
                             "> 0.80"}
    (OUT / "protocol.json").write_text(json.dumps(context, indent=2))
    _, rows = rows_for(context)
    chars = len(json.dumps(request_body(rows[:BATCH], context),
                           separators=(",", ":")))
    print(json.dumps({"references": context["reference_models"],
                      "n_minutes": len(rows),
                      "first_test_batch_user_chars": chars,
                      "approx_input_tokens_per_request":
                          int(chars / 3.2) + 200}, indent=2))


def rows_for(context):
    df = pd.read_csv(OUT / "test_mapping.csv")
    refs = context["reference_models"]
    rows = []
    for r in df.itertuples(index=False):
        pml, pdl = float(r.p_ml), float(r.p_dl)
        rows.append({"id": r.id,
                     "descriptors": {c: num(getattr(r, c)) for c in DESC},
                     "ml_probability": round(pml, 4),
                     "ml_prediction": int(pml >= refs["ML"]["threshold"]),
                     "dl_probability": round(pdl, 4),
                     "dl_prediction": int(pdl >= refs["DL"]["threshold"])})
    return df, rows


def request_body(batch, context):
    return {"reference_models": context["reference_models"],
            "training_class_mean_std": context["training_class_mean_std"],
            "labeled_training_examples":
                context["labeled_training_examples"],
            "minutes": batch}


def request_batch(key, model, run, batch, context):
    payload = {"model": model,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user",
                             "content": json.dumps(
                                 request_body(batch, context),
                                 separators=(",", ":"))}],
               "max_completion_tokens": 3000}
    if model == "gpt-4o-mini":
        payload.update({"temperature": .2, "seed": 27000 + run})
    data = _post(os.environ.get("JEV_BASE_URL",
                                "https://api.openai.com/v1")
                 + "/chat/completions",
                 {"Authorization": "Bearer " + key,
                  "Content-Type": "application/json"}, payload)
    content = json.loads(data["choices"][0]["message"]["content"])
    arr = content.get("predictions") or content.get("results")
    got = {str(x["id"]): float(x["occupied_probability"]) for x in arr}
    missing = [x["id"] for x in batch if x["id"] not in got]
    if missing:
        raise ValueError(f"missing ids {missing}")
    return got, data.get("usage", {})


def result(model):
    df = pd.read_csv(OUT / "test_mapping.csv").set_index("id")
    pred = pd.read_csv(OUT / f"{model}_test_predictions.csv")
    wide = pred.pivot_table(index="id", columns="run",
                            values="probability")
    labels = df.loc[wide.index, "label"].to_numpy(int)
    accs, cms = [], []
    for run in sorted(wide.columns):
        p = (wide[run].to_numpy(float) >= .5).astype(int)
        accs.append(float(accuracy_score(labels, p)))
        cms.append(confusion_matrix(labels, p, labels=[0, 1]).tolist())
    mean_p = wide.mean(axis=1).to_numpy(float)
    mean_pred = (mean_p >= .5).astype(int)
    return {"per_run_accuracy": accs,
            "mean_accuracy": float(np.mean(accs)),
            "sample_std_accuracy": float(np.std(accs, ddof=1)),
            "per_run_confusion": cms,
            "mean_vote_confusion": confusion_matrix(
                labels, mean_pred, labels=[0, 1]).tolist(),
            "n_minutes": int(len(labels)),
            "mean_probability_file": f"{model}_test_predictions.csv"}


def run_model(model, key, waive_gate=False):
    waived = False
    if model == "gpt-5.4":
        pilot = json.loads((OUT / "gpt-4o-mini_summary.json").read_text())
        if pilot["mean_accuracy"] <= .80:
            if not waive_gate:
                raise RuntimeError(
                    "GPT-4o mini did not exceed the prespecified 80% "
                    "test gate")
            waived = True
            print("WARNING: running GPT-5.4 although the 0.80 pilot gate "
                  f"failed (mini test mean "
                  f"{pilot['mean_accuracy']:.3f})", flush=True)
    context = json.loads((OUT / "protocol.json").read_text())
    df, rows = rows_for(context)
    path = OUT / f"{model}_test_predictions.csv"
    stored = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(int(x["run"]), str(x["id"])) for x in stored}
    usage_path = OUT / f"{model}_usage.jsonl"
    for run in (1, 2, 3):
        for j in range(0, len(rows), BATCH):
            batch = [x for x in rows[j:j + BATCH]
                     if (run, x["id"]) not in done]
            if not batch:
                continue
            got, usage = request_batch(key, model, run, batch, context)
            for r in batch:
                stored.append({"run": run, "id": r["id"],
                               "probability": got[r["id"]]})
                done.add((run, r["id"]))
            pd.DataFrame(stored).to_csv(path, index=False)
            with usage_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"run": run, "batch_start": j,
                                    "n": len(batch),
                                    "usage": usage}) + "\n")
        print(f"{model} test run {run}: {len(done)}/{3 * len(rows)}",
              flush=True)
    if len(done) != 3 * len(rows):
        raise ValueError("Incomplete model run")
    summary = result(model)
    if waived:
        summary["gate_waived"] = True
    (OUT / f"{model}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"mean_accuracy": summary["mean_accuracy"],
                      "sd": summary["sample_std_accuracy"]}, indent=2),
          flush=True)


def prepare_explain():
    context = json.loads((OUT / "protocol.json").read_text())
    mapping = pd.read_csv(OUT / "test_mapping.csv")
    rng = np.random.default_rng(20261015)
    chosen = []
    for label in (0, 1):
        groups = sorted(mapping.loc[mapping.label.eq(label),
                                    "unit"].unique())
        counts = ([2, 1] if len(groups) == 2 else
                  [1] * (len(groups) - 1) + [0])
        for group, count in zip(groups, counts):
            candidates = mapping[mapping.unit.eq(group)]
            chosen.extend(rng.choice(candidates.index.to_numpy(),
                                     count, replace=False))
    selected = mapping.loc[chosen].copy().reset_index(drop=True)
    selected[["id", "unit", "minute", "label"]].to_csv(
        OUT / "explainability_selection.csv", index=False)
    refs = context["reference_models"]
    minutes = []
    for row in selected.itertuples(index=False):
        p_ml, p_dl = float(row.p_ml), float(row.p_dl)
        minutes.append({"id": row.id,
                        "descriptors": {c: num(getattr(row, c))
                                        for c in DESC},
                        "ml_probability": round(p_ml, 4),
                        "ml_prediction": int(
                            p_ml >= refs["ML"]["threshold"]),
                        "dl_probability": round(p_dl, 4),
                        "dl_prediction": int(
                            p_dl >= refs["DL"]["threshold"])})
    request = {"reference_models": refs,
               "training_class_mean_std":
                   context["training_class_mean_std"],
               "labeled_training_examples":
                   context["labeled_training_examples"],
               "minutes": minutes}
    (OUT / "explainability_request.json").write_text(
        json.dumps(request, indent=2))
    print("Selected:", ", ".join(
        f"{r.id}({r.unit}:{r.minute},{'O' if r.label else 'E'})"
        for r in selected.itertuples(index=False)))


def run_explain(model, key):
    request = json.loads((OUT / "explainability_request.json").read_text())
    payload = {"model": model,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": EXPLAIN_SYSTEM},
                            {"role": "user",
                             "content": json.dumps(
                                 request, separators=(",", ":"))}],
               "max_completion_tokens": 5000}
    if model == "gpt-4o-mini":
        payload["temperature"] = 0.2
    data = _post(os.environ.get("JEV_BASE_URL",
                                "https://api.openai.com/v1")
                 + "/chat/completions",
                 {"Authorization": "Bearer " + key,
                  "Content-Type": "application/json"}, payload)
    response = json.loads(data["choices"][0]["message"]["content"])
    received = [str(x["id"]) for x in response["results"]]
    expected = [str(x["id"]) for x in request["minutes"]]
    if set(received) != set(expected):
        raise ValueError("Response sample IDs do not match request")
    (OUT / f"explainability_{model}.json").write_text(json.dumps({
        "model": model,
        "purpose": "qualitative explanations only; excluded from accuracy",
        "system_prompt": EXPLAIN_SYSTEM,
        "request_sha256": hashlib.sha256(
            json.dumps(request).encode()).hexdigest(),
        "response": response, "usage": data.get("usage", {})}, indent=2))
    print(f"Saved {len(received)} explanations for {model}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--prepare-explain", action="store_true")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--model", choices=("gpt-4o-mini", "gpt-5.4"))
    ap.add_argument("--key-stdin", action="store_true")
    ap.add_argument("--waive-gate", action="store_true")
    args = ap.parse_args()
    if args.prepare_only:
        prepare()
    elif args.prepare_explain:
        prepare_explain()
    elif args.model:
        key = (getpass.getpass("API key: ") if sys.stdin.isatty() else
               sys.stdin.readline().strip()) if args.key_stdin \
            else os.environ.get("JEV_API_KEY", "")
        if not key:
            raise RuntimeError("API key is required for live calls")
        if args.explain:
            run_explain(args.model, key)
        else:
            run_model(args.model, key, waive_gate=args.waive_gate)
    else:
        ap.error("Select --prepare-only, --prepare-explain, or --model")
