"""Fixed, auditable 20% validation and multilink-test LLM evaluation.

The two reference models are chosen on full validation-minute accuracy:
RF late fusion with unlabeled batch rescaling, and MLP CSI without rescaling.
The validation sample develops the prompt. The multilink test is queried only
after the selected prompt is frozen. Predictions are checkpointed per batch.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from physical_comparison import load, aggregate
from jev import _post

OUT = Path(__file__).resolve().parent / "outputs/revision"
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)
DESC = ["csi_amp_mean", "csi_amp_std", "csi_rv_mean", "snr_mean",
        "ra_std90", "re_std90", "xy_std90", "ra_delta", "re_delta"]
MODELS = [("rf_late", "RF|late probability fusion|rescaled"),
          ("mlp_csi", "MLP|CSI|raw")]
MODEL_NAME = "gpt-5.4"


def make_cohorts():
    train, val, test = load()
    saved = np.load(OUT / "physical_probs.npz")
    stats = {}
    for lab, frame in train.groupby("label"):
        stats[("empty", "occupied")[int(lab)]] = {
            c: [round(float(frame[c].mean()), 3),
                round(float(frame[c].std()), 3)] for c in DESC}
    examples = []
    for lab, frame in train.groupby("label"):
        for _, row in frame.groupby("rec_id", sort=True).mean(numeric_only=True).head(2).iterrows():
            examples.append({"label": ("empty", "occupied")[int(lab)],
                             "x": {c: round(float(row[c]), 3) for c in DESC}})
    result = {}
    for split, windows in (("val", val), ("test", test)):
        minute = windows.groupby(["unit", "minute"], sort=True).agg(
            label=("label", "first"), **{c:(c,"mean") for c in DESC})
        for name, key in MODELS:
            y, p, ids = aggregate(windows, saved[key+"|"+split])
            assert list(ids) == list(minute.index) and np.array_equal(y, minute.label.to_numpy())
            minute[name] = p
            # An unlabeled percentile conveys the same batch calibration
            # available to the comparator, without using target labels.
            minute[name+"_rank"] = pd.Series(p, index=minute.index).rank(pct=True).to_numpy()
        minute["split"] = split
        result[split] = minute
    rng = np.random.default_rng(42)
    selected = {}
    for split, frame in result.items():
        picks = []
        groups = [g for _, g in frame.groupby("unit")] if split == "test" else [g for _,g in frame.groupby("label")]
        for group in groups:
            n = round(.2 * len(group))
            picks.extend(group.index[rng.choice(len(group), n, replace=False)].tolist())
        # numpy can flatten MultiIndex tuples; use membership through strings
        want = {str(x) for x in picks}
        sample = frame[[str(x) in want for x in frame.index]].copy()
        assert len(sample) == sum(round(.2*len(g)) for g in groups)
        selected[split] = sample
        sample.reset_index().to_csv(OUT / f"llm_20pct_{split}_sample.csv", index=False)
    return selected, stats, examples


def system_prompt(version):
    common = ("Classify binary room occupancy. Return a JSON object with key "
              "predictions: an array of {id, occupied_probability} in input order. "
              "Every probability must be 0..1. A reference model probability "
              "is evidence, not ground truth. Use the same decision rule for "
              "every item; do not infer labels from IDs, ordering, or capture. ")
    if version == "v1":
        return common + "Use both reference model probabilities, their unlabeled-batch ranks, physical descriptors, class statistics, and labeled training examples."
    return common + ("The RF late-fusion model is the more reliable cross-cohort reference. "
                     "Start with its probability rank: values above 0.50 favor occupied. "
                     "The CSI MLP is a secondary reference. When both agree, retain "
                     "their consensus. When they disagree, follow RF unless several "
                     "physical descriptors strongly support the MLP relative to the "
                     "training class statistics. Do not allow a verbal explanation "
                     "to overturn concordant reference predictions.")


def payload_rows(frame):
    rows = []
    for idx, row in frame.iterrows():
        rows.append({"id": str(idx), "x": {c:round(float(row[c]),3) for c in DESC},
                     "rf_late_probability": round(float(row.rf_late),4),
                     "rf_late_batch_percentile": round(float(row.rf_late_rank),4),
                     "mlp_csi_probability": round(float(row.mlp_csi),4),
                     "mlp_csi_batch_percentile": round(float(row.mlp_csi_rank),4)})
    return rows


def call_batch(key, version, rows, stats, examples):
    request = {"training_class_mean_std":stats,
               "labeled_training_calibration_examples":examples,
               "minutes":rows}
    data = _post(os.environ.get("JEV_BASE_URL", "https://api.openai.com/v1")+"/chat/completions",
                 {"Authorization":"Bearer "+key, "Content-Type":"application/json"},
                 {"model":MODEL_NAME, "response_format":{"type":"json_object"},
                  "messages":[{"role":"system","content":system_prompt(version)},
                              {"role":"user","content":json.dumps(request,separators=(",",":"))}],
                  "max_completion_tokens":3000})
    obj = json.loads(data["choices"][0]["message"]["content"])
    pred = obj["predictions"]
    by_id = {str(x["id"]):float(x["occupied_probability"]) for x in pred}
    if set(by_id) != {x["id"] for x in rows}:
        raise ValueError("Model response IDs do not match request")
    return {k:float(np.clip(v,0,1)) for k,v in by_id.items()}


def eval_split(key, split, frame, stats, examples, version, runs=3, batch_size=10):
    rows = payload_rows(frame)
    path = OUT / f"llm_20pct_{split}_{version}_predictions.csv"
    records = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(int(r["run"]),str(r["id"])) for r in records}
    for run in range(1,runs+1):
        for j in range(0,len(rows),batch_size):
            batch = [r for r in rows[j:j+batch_size] if (run,r["id"]) not in done]
            if not batch: continue
            if not key: raise RuntimeError(f"Missing cached predictions for {split} {version}")
            got = call_batch(key,version,batch,stats,examples)
            for item in batch:
                records.append({"run":run,"id":item["id"],"probability":got[item["id"]]})
                done.add((run,item["id"]))
            pd.DataFrame(records).to_csv(path,index=False)
            print(f"{split} {version} run {run}: {len(done)}/{runs*len(rows)}",flush=True)
    y = frame.label.to_numpy(int)
    vals = []
    for run in range(1,runs+1):
        p = np.array([next(float(r["probability"]) for r in records if int(r["run"])==run and str(r["id"])==x["id"]) for x in rows])
        pred = (p>=.5).astype(int)
        vals.append({"run":run,"accuracy":float(accuracy_score(y,pred)),
                     "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()})
    return vals


def baseline(frame):
    y = frame.label.to_numpy(int)
    out = {}
    for name, _ in MODELS:
        p = frame[name].to_numpy()
        pred = (frame[name+"_rank"].to_numpy()>=.5).astype(int)
        out[name] = {"accuracy":float(accuracy_score(y,pred)),
                     "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}
    # A transparent no-LLM ensemble benchmark on the same records.
    pred = ((frame.rf_late_rank.to_numpy()+frame.mlp_csi_rank.to_numpy())/2>=.5).astype(int)
    out["rank_average"] = {"accuracy":float(accuracy_score(y,pred)),
                           "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}
    return out


def write_figure(report):
    cases = [("LLM run 1",report["test"]["llm"][0]["confusion_matrix"])]
    cases += [(name.replace("_"," ").upper(),d["confusion_matrix"])
              for name,d in report["test"]["baselines"].items()]
    fig,axes=plt.subplots(1,len(cases),figsize=(2.5*len(cases),2.9),constrained_layout=True)
    vmax=max(np.max(x) for _,x in cases)
    for ax,(title,cm) in zip(axes,cases):
        ax.imshow(cm,cmap="YlGnBu",vmin=0,vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i][j]),ha="center",va="center",fontsize=11,
                        color="white" if cm[i][j] > .55*vmax else "#14202A")
        ax.set_xticks([0,1],["Empty","Occupied"],fontsize=7)
        ax.set_yticks([0,1],["Empty","Occupied"],fontsize=7)
        ax.set_xlabel("Predicted",fontsize=8)
        ax.set_title(title,fontsize=9)
    axes[0].set_ylabel("True",fontsize=8)
    fig.savefig(FIG/"llm_20pct_cm.pdf",bbox_inches="tight")
    fig.savefig(FIG/"llm_20pct_cm.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--key-stdin",action="store_true")
    ap.add_argument("--prepare-only",action="store_true")
    ap.add_argument("--summarize-only",action="store_true")
    args=ap.parse_args()
    cohorts,stats,examples=make_cohorts()
    report={s:{"n":len(f),"baselines":baseline(f)} for s,f in cohorts.items()}
    (OUT/"llm_20pct_protocol.json").write_text(json.dumps({"model":MODEL_NAME,"descriptors":DESC,
        "reference_models":MODELS,"sampling":"20% per validation class and multilink capture, seed 42",
        "prompt_versions":["v1","v2"],"training_stats":stats,"training_examples":examples},indent=2))
    if args.prepare_only:
        (OUT/"llm_20pct_baselines.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2)); return
    key="" if args.summarize_only else ((getpass.getpass("API key: ") if sys.stdin.isatty() else sys.stdin.readline().strip()) if args.key_stdin else os.environ.get("JEV_API_KEY",""))
    if not key and not args.summarize_only: raise RuntimeError("JEV_API_KEY is not configured")
    # Prompt selection uses validation labels. Test is queried once afterward.
    versions={}
    for version in ("v1","v2"):
        vals=eval_split(key,"val",cohorts["val"],stats,examples,version)
        versions[version]=vals
    chosen=max(versions,key=lambda v:np.mean([r["accuracy"] for r in versions[v]]))
    report["prompt_development"]={"validation_runs":versions,"selected":chosen}
    report["val"]["llm"]=versions[chosen]
    report["test"]["llm"]=eval_split(key,"test",cohorts["test"],stats,examples,chosen)
    for split in ("val","test"):
        scores=np.array([r["accuracy"] for r in report[split]["llm"]])
        report[split]["llm_summary"]={"accuracy_mean":float(scores.mean()),
            "accuracy_sd":float(scores.std(ddof=1)),"run_accuracies":scores.tolist()}
    (OUT/"llm_20pct_results.json").write_text(json.dumps(report,indent=2))
    write_figure(report)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
