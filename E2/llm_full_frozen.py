"""Full-cohort LLM evaluation with validation-frozen model references.

Opaque, shuffled IDs prevent capture names or label-bearing paths from being
sent to the API. No test-batch rank, mean, median, scaler, weight, threshold,
or calibration example enters the prompt. GPT-5.4 is gated on the completed
three-run GPT-4o-mini multilink-test mean accuracy exceeding 0.80.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from frozen_evaluation import ROOT, OUT as FROZEN, OLD, TEST, sha256
from jev import _post

OUT = ROOT / "E2/outputs/frozen/llm"
OUT.mkdir(parents=True,exist_ok=True)
DESC = ["csi_amp_mean","csi_amp_std","csi_rv_mean","snr_mean",
        "ra_std90","re_std90","xy_std90","ra_delta","re_delta"]
SYSTEM = (
    "Classify room occupancy (empty=0, occupied=1) using physical WiFi CSI "
    "and radar descriptors and two trained model references. Return only a "
    "JSON object with predictions: an array of {id,occupied_probability} "
    "for every input ID. The decision cutoff on your output is fixed at 0.5. "
    "The supplied reference thresholds and validation performances were "
    "fixed before multilink evaluation. Respect agreement of the two "
    "reference class predictions; when they disagree, consider their margins "
    "from the fixed thresholds and the training-class descriptor statistics. "
    "Use one consistent rule for all cases. IDs are arbitrary. Do not infer "
    "class prevalence from the batch or ordering."
)


def load_split(split, reference):
    path = OLD if split=="validation" else TEST
    windows=pd.read_csv(path)
    if split=="validation":
        windows=windows[windows.rec_id.str.startswith("test_minutes/")].copy()
        windows["unit"]=windows.rec_id.astype(str)
        windows["minute"]=0
    else:
        for suffix in ("amp_mean", "amp_std", "rv_mean"):
            windows[f"csi_{suffix}"]=windows[[f"csi1_{suffix}",f"csi2_{suffix}"]].mean(axis=1)
        windows["unit"]=windows.cap.astype(str)
        windows["minute"]=(windows.t_rel//60).astype(int)
    frame=windows.groupby(["unit","minute"],sort=True).agg(
        label=("label","first"),**{c:(c,"mean") for c in DESC})
    for family in ("ML","DL"):
        if split=="validation":
            source=FROZEN/f"validation_{family.lower()}_probabilities.csv"
        else:
            key=reference[family]["key"]
            source=FROZEN/("test_"+key.replace("|","__")+"_probabilities.csv")
        p=pd.read_csv(source).set_index(["unit","minute"])
        if not frame.index.equals(p.index):
            raise ValueError(f"Probability row alignment failed for {family} {split}")
        if not np.array_equal(frame.label.to_numpy(int),p.label.to_numpy(int)):
            raise ValueError("Probability labels do not align")
        frame[f"p_{family.lower()}"]=p.probability.to_numpy(float)
    return frame


def training_context():
    train=pd.read_csv(OLD)
    train=train[train.rec_id.str.startswith("train/")].copy()
    stats={}
    examples=[]
    for lab, group in train.groupby("label"):
        label=("empty","occupied")[int(lab)]
        stats[label]={c:[round(float(group[c].mean()),3),
                         round(float(group[c].std()),3)] for c in DESC}
        for _,row in group.groupby("rec_id",sort=True).mean(numeric_only=True).head(2).iterrows():
            examples.append({"class":label,"descriptors":{
                c:round(float(row[c]),3) for c in DESC}})
    return stats,examples


def prepare():
    selection=json.loads((FROZEN/"selection.json").read_text())
    evaluation=json.loads((FROZEN/"evaluation.json").read_text())
    refs=selection["references"]
    stats,examples=training_context()
    rng=np.random.default_rng(20261014)
    for split in ("validation","test"):
        frame=load_split(split,refs)
        order=rng.permutation(len(frame))
        frame=frame.iloc[order].copy()
        frame.insert(0,"id",[f"m{i:04d}" for i in range(len(frame))])
        frame.reset_index().to_csv(OUT/f"{split}_mapping.csv",index=False)
    context={"protocol":"Full 192-minute validation and 360-minute multilink test. Train-only model scalers; validation AUC selects reference models; validation labels fix operating thresholds; no test-batch calibration, statistics, percentile rank, or source ID in API input.",
             "selection_sha256":sha256(FROZEN/"selection.json"),
             "evaluation_sha256":sha256(FROZEN/"evaluation.json"),
             "reference_models":{f:{"key":refs[f]["key"],
                 "threshold":refs[f]["threshold"],
                 "validation_balanced_accuracy":refs[f]["balanced_accuracy"]}
                 for f in ("ML","DL")},
             "descriptors":DESC,"training_class_mean_std":stats,
             "labeled_training_examples":examples,"system_prompt":SYSTEM,
             "decision_threshold":.5,"runs":3,"batch_size":10,
             "model_order":["gpt-4o-mini","gpt-5.4"],
             "second_model_gate":"mini mean multilink-test accuracy across three runs > 0.80"}
    (OUT/"protocol.json").write_text(json.dumps(context,indent=2))
    print(json.dumps({"validation_minutes":192,"test_minutes":360,
                       "references":context["reference_models"]},indent=2))


def rows_for(split,context):
    df=pd.read_csv(OUT/f"{split}_mapping.csv")
    rows=[]
    refs=context["reference_models"]
    for r in df.itertuples(index=False):
        pml,pdl=float(r.p_ml),float(r.p_dl)
        rows.append({"id":r.id,"descriptors":{
            c:round(float(getattr(r,c)),3) for c in DESC},
            "ml_probability":round(pml,4),
            "ml_prediction":int(pml>=refs["ML"]["threshold"]),
            "dl_probability":round(pdl,4),
            "dl_prediction":int(pdl>=refs["DL"]["threshold"])})
    return df,rows


def request_batch(key,model,run,batch,context):
    body={"reference_models":context["reference_models"],
          "training_class_mean_std":context["training_class_mean_std"],
          "labeled_training_examples":context["labeled_training_examples"],
          "minutes":batch}
    payload={"model":model,"response_format":{"type":"json_object"},
             "messages":[{"role":"system","content":SYSTEM},
                         {"role":"user","content":json.dumps(body,separators=(",",":"))}],
             "max_completion_tokens":3000}
    if model=="gpt-4o-mini":
        payload.update({"temperature":.2,"seed":27000+run})
    data=_post(os.environ.get("JEV_BASE_URL","https://api.openai.com/v1")+"/chat/completions",
               {"Authorization":"Bearer "+key,"Content-Type":"application/json"},payload)
    pred=json.loads(data["choices"][0]["message"]["content"])["predictions"]
    lookup={str(r["id"]):float(r["occupied_probability"]) for r in pred}
    if set(lookup)!={r["id"] for r in batch}:
        raise ValueError("API returned mismatched or duplicate IDs")
    if not all(np.isfinite(x) and 0<=x<=1 for x in lookup.values()):
        raise ValueError("API probability outside [0,1]")
    return lookup,data.get("usage",{})


def result_split(model,split):
    df=pd.read_csv(OUT/f"{split}_mapping.csv")
    pred=pd.read_csv(OUT/f"{model}_{split}_predictions.csv")
    if len(pred)!=3*len(df): raise ValueError("Incomplete predictions")
    result=[]
    for run in (1,2,3):
        p=pred[pred.run==run].set_index("id").loc[df.id].probability.to_numpy(float)
        y=df.label.to_numpy(int)
        yp=(p>=.5).astype(int)
        result.append({"run":run,"n":len(df),
            "accuracy":float(accuracy_score(y,yp)),
            "balanced_accuracy":float(balanced_accuracy_score(y,yp)),
            "confusion_matrix":confusion_matrix(y,yp,labels=[0,1]).tolist()})
    a=np.array([r["accuracy"] for r in result])
    return {"runs":result,"mean_accuracy":float(a.mean()),
            "sample_std_accuracy":float(a.std(ddof=1))}


def run_model(model,key):
    if model=="gpt-5.4":
        pilot=json.loads((OUT/"gpt-4o-mini_summary.json").read_text())
        if pilot["test"]["mean_accuracy"]<=.80:
            raise RuntimeError("GPT-4o mini did not exceed the prespecified 80% test gate")
    context=json.loads((OUT/"protocol.json").read_text())
    for split in ("validation","test"):
        df,rows=rows_for(split,context)
        path=OUT/f"{model}_{split}_predictions.csv"
        stored=pd.read_csv(path).to_dict("records") if path.exists() else []
        done={(int(x["run"]),str(x["id"])) for x in stored}
        usage_path=OUT/f"{model}_usage.jsonl"
        for run in (1,2,3):
            for j in range(0,len(rows),10):
                batch=[x for x in rows[j:j+10] if (run,x["id"]) not in done]
                if not batch: continue
                got,usage=request_batch(key,model,run,batch,context)
                for r in batch:
                    stored.append({"run":run,"id":r["id"],
                                   "probability":got[r["id"]]})
                    done.add((run,r["id"]))
                pd.DataFrame(stored).to_csv(path,index=False)
                with usage_path.open("a",encoding="utf-8") as f:
                    f.write(json.dumps({"split":split,"run":run,"batch_start":j,
                                        "n":len(batch),"usage":usage})+"\n")
                print(f"{model} {split} run {run}: {len(done)}/{3*len(rows)}",flush=True)
        if len(done)!=3*len(rows): raise ValueError("Incomplete model run")
    summary={s:result_split(model,s) for s in ("validation","test")}
    (OUT/f"{model}_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepare-only",action="store_true")
    ap.add_argument("--model",choices=("gpt-4o-mini","gpt-5.4"))
    ap.add_argument("--key-stdin",action="store_true")
    args=ap.parse_args()
    if args.prepare_only:
        prepare()
    elif args.model:
        key=(getpass.getpass("API key: ") if sys.stdin.isatty() else
             sys.stdin.readline().strip()) if args.key_stdin else os.environ.get("JEV_API_KEY","")
        if not key: raise RuntimeError("API key is required for live calls")
        run_model(args.model,key)
    else:
        ap.error("Select --prepare-only or --model")
