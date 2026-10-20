"""Train/validation-only selection, followed by fixed multilink evaluation.

Run ``--phase select`` first. That phase never opens a multilink feature file
or test label. It fits each model and scaler on training windows, selects
models by validation-minute AUC, and freezes thresholds on validation labels.
Run ``--phase evaluate`` only after the selection JSON exists. That phase
applies the saved models and thresholds unchanged to multilink captures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from physical_comparison import ROOT, RAD, CSI, matrix

OLD = ROOT / "E1/outputs/features_old_5s_ext.csv"
TEST = ROOT / "E1/outputs/features_multilink_5s.csv"
OUT = ROOT / "E2/outputs/frozen"
OUT.mkdir(parents=True, exist_ok=True)
VIEWS = {"CSI": CSI, "radar": RAD, "early fusion": RAD + CSI}
MODEL_NAMES = ("RF", "GBM", "SVM", "MLP")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_model(name):
    return {
        "RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=12,
            class_weight="balanced", random_state=42, n_jobs=1),
        "GBM": lambda: HistGradientBoostingClassifier(max_iter=200,
            learning_rate=.07, random_state=42),
        "SVM": lambda: SVC(C=4, kernel="rbf", probability=True,
            class_weight="balanced", random_state=42),
        "MLP": lambda: MLPClassifier((64,32), alpha=1e-3,
            max_iter=400, random_state=42),
    }[name]()


def grouped(df, p):
    return pd.DataFrame({"unit":df.unit.to_numpy(),
        "minute":df.minute.to_numpy(),"label":df.label.to_numpy(int),
        "probability":p}).groupby(["unit","minute"],sort=True).agg(
            label=("label","first"),probability=("probability","mean"))


def fit_threshold(y, p):
    """Choose an occupied-probability cutoff using validation labels only."""
    candidates = np.unique(p)
    scored = [(balanced_accuracy_score(y, p >= t),
               -abs(float(t)-.5),float(t)) for t in candidates]
    return max(scored)[2]


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {"accuracy":float(accuracy_score(y,pred)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "roc_auc":float(roc_auc_score(y,p)),
            "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}


def read_old():
    d = pd.read_csv(OLD)
    train = d[d.rec_id.str.startswith("train/")].copy()
    val = d[d.rec_id.str.startswith("test_minutes/")].copy()
    for frame in (train,val):
        frame["unit"] = frame.rec_id.astype(str)
        frame["minute"] = 0
    return train,val


def select():
    target=OUT/"selection.json"
    if target.exists():
        raise FileExistsError("Selection is frozen; refusing to overwrite it")
    train,val=read_old()  # no multilink file or labels are opened in this phase
    ytr=train.label.to_numpy(int)
    pred={}
    for view,cols in VIEWS.items():
        fill=np.nanmedian(train[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float),axis=0)
        tr=matrix(train,cols,fill)
        va=matrix(val,cols,fill)
        scaler=StandardScaler().fit(tr)
        Xtr,Xva=scaler.transform(tr),scaler.transform(va)
        for name in MODEL_NAMES:
            key=f"{name}|{view}"
            model=make_model(name).fit(Xtr,ytr)
            joblib.dump({"model":model,"scaler":scaler,"fill":fill,
                         "columns":cols},OUT/(key.replace("|","__")+".joblib"))
            pred[key]=model.predict_proba(Xva)[:,1]
            print("trained",key,flush=True)
    pred["RF|late probability fusion"]=(pred["RF|CSI"]+pred["RF|radar"])/2
    records=[]
    minute_probs={}
    for key,p in pred.items():
        frame=grouped(val,p)
        y=frame.label.to_numpy(int)
        pp=frame.probability.to_numpy(float)
        threshold=fit_threshold(y,pp)
        result=metrics(y,pp,threshold)
        name,view=key.split("|",1)
        records.append({"key":key,"family":"DL" if name=="MLP" else "ML",
                        "view":view,"threshold":threshold,
                        "n":len(y),**result})
        minute_probs[key]=frame
    # Selection is by validation AUC only. Exact ties use the key name.
    ranked=sorted(records,key=lambda r:(-r["roc_auc"],r["key"]))
    selected={view:next(r for r in ranked if r["view"]==view)
              for view in (*VIEWS,"late probability fusion")}
    references={family:next(r for r in ranked if r["family"]==family)
                for family in ("ML","DL")}
    manifest={"protocol":"training-only imputation and standardization; validation-minute AUC selects view leaders and ML/DL references; validation labels fix each probability threshold; no test-batch statistics",
              "train_windows":len(train),"train_minutes":int(train.unit.nunique()),
              "validation_windows":len(val),"validation_minutes":int(val.unit.nunique()),
              "source_sha256":{"train_validation_features":sha256(OLD)},
              "selected_views":selected,"references":references,
              "all_validation_candidates":ranked}
    # Save selection before any evaluation opens multilink inputs.
    target.write_text(json.dumps(manifest,indent=2))
    for family,r in references.items():
        minute_probs[r["key"]].to_csv(OUT/f"validation_{family.lower()}_probabilities.csv")
    print(json.dumps({"selected_views":selected,"references":references},indent=2),flush=True)


def predict_view(df,key):
    if key=="RF|late probability fusion":
        return (predict_view(df,"RF|CSI")+predict_view(df,"RF|radar"))/2
    obj=joblib.load(OUT/(key.replace("|","__")+".joblib"))
    X=matrix(df,obj["columns"],obj["fill"])
    return obj["model"].predict_proba(obj["scaler"].transform(X))[:,1]


def evaluate():
    frozen=json.loads((OUT/"selection.json").read_text())
    # All choices, including operating thresholds, are loaded before test.
    selected={r["key"]:r for r in list(frozen["selected_views"].values())+
              list(frozen["references"].values())}
    test=pd.read_csv(TEST)
    for suffix in [s.removeprefix("csi_") for s in CSI]:
        test[f"csi_{suffix}"]=test[[f"csi1_{suffix}",f"csi2_{suffix}"]].mean(axis=1)
    test["unit"]=test.cap.astype(str)
    test["minute"]=(test.t_rel//60).astype(int)
    _,val=read_old()
    outcomes={}
    for key,r in selected.items():
        p_test=predict_view(test,key)
        p_val=predict_view(val,key)
        frame=grouped(test,p_test)
        frame.to_csv(OUT/("test_"+key.replace("|","__")+"_probabilities.csv"))
        pred=(frame.probability.to_numpy(float)>=r["threshold"]).astype(int)
        by_capture={str(unit):float(accuracy_score(g.label.to_numpy(int),
            (g.probability.to_numpy(float)>=r["threshold"]).astype(int)))
            for unit,g in frame.groupby(level=0)}
        outcomes[key]={"validation":{k:r[k] for k in ("threshold","accuracy","balanced_accuracy","roc_auc","confusion_matrix")},
                       "validation_window":metrics(val.label.to_numpy(int),p_val,r["threshold"]),
                       "test":metrics(frame.label.to_numpy(int),frame.probability.to_numpy(float),r["threshold"]),
                       "test_window":metrics(test.label.to_numpy(int),p_test,r["threshold"]),
                       "test_capture_accuracy":by_capture,
                       "test_minutes":len(frame)}
    report={"selection_sha256":sha256(OUT/"selection.json"),
            "test_source_sha256":sha256(TEST),"test_windows":len(test),
            "test_minutes":int(test.groupby(["unit","minute"]).ngroups),
            "results":outcomes,"selected_views":{k:v["key"] for k,v in frozen["selected_views"].items()},
            "references":{k:v["key"] for k,v in frozen["references"].items()}}
    (OUT/"evaluation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",required=True,choices=("select","evaluate"))
    args=ap.parse_args()
    {"select":select,"evaluate":evaluate}[args.phase]()
