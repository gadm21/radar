"""Separate exploratory 1-s/10-s/1-min PCA fusion rerun; not paper data.

Run --phase select, then --phase evaluate. The select phase cannot open the
multilink tables or PCA dots. All PCA bases and fitted components are saved
before the evaluate phase reads any multilink input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from frozen_evaluation import make_model, fit_threshold
from physical_comparison import CSI, RAD, ROOT

P = ROOT / "E1/outputs"
D = P / "pca_deep"
OUT = ROOT / "E2/outputs/hierarchical_pca_fusion"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ("RF", "GBM", "SVM", "MLP")


def old_physical():
    frame = pd.read_csv(P / "features_old_5s_ext.csv")
    frame = frame[frame.rec_id.str.startswith(("train/", "test_minutes/"))].copy()
    cols = CSI + RAD
    return frame.groupby("rec_id", sort=True)[cols].mean().assign(
        label=frame.groupby("rec_id").label.first())


def dots(split):
    path = D / f"pc_dots_{split}.npz"
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def ten_blocks(a, cols):
    out = []
    for start in range(0, len(a)-9, 10):
        seg = a[start:start+10]
        good = (seg[:, 0] > .5) & np.isfinite(seg[:, cols]).all(axis=1)
        if good.sum() < 5:
            out.append(None)
            continue
        v = np.column_stack([np.interp(np.arange(10), np.flatnonzero(good),
                                       seg[good, j]) for j in range(cols.start, cols.stop)])
        out.append(v.reshape(-1))
    return out


def features_for_minute(a, cols, sc10, p10, sc60, p60):
    good = (a[:, 0] > .5) & np.isfinite(a[:, cols]).all(axis=1)
    one = a[good, cols]
    one_stats = np.r_[one.mean(axis=0), one.std(axis=0)] if len(one) else np.full(6, np.nan)
    blocks = ten_blocks(a, cols)
    indices = [i for i, b in enumerate(blocks) if b is not None]
    if indices:
        ten = p10.transform(sc10.transform(np.stack([blocks[i] for i in indices])))
        ten_stats = np.r_[ten.mean(axis=0), ten.std(axis=0)]
    else:
        ten_stats = np.full(6, np.nan)
    if len(indices) >= 4 and len(blocks) == 6:
        filled = np.column_stack([np.interp(np.arange(6), indices, ten[:, j])
                                  for j in range(3)]).reshape(1, -1)
        minute = p60.transform(sc60.transform(filled))[0]
    else:
        minute = np.full(2, np.nan)
    return np.r_[one_stats, ten_stats, minute]


def fit_hierarchy(train_dots, keys, cols):
    blocks = [b for k in keys for b in ten_blocks(train_dots[k], cols) if b is not None]
    sc10 = StandardScaler().fit(np.stack(blocks))
    p10 = PCA(n_components=3, random_state=0).fit(sc10.transform(np.stack(blocks)))
    minute_vectors = []
    for k in keys:
        b = ten_blocks(train_dots[k], cols)
        indices = [i for i, v in enumerate(b) if v is not None]
        if len(b) == 6 and len(indices) >= 4:
            ten = p10.transform(sc10.transform(np.stack([b[i] for i in indices])))
            minute_vectors.append(np.column_stack([
                np.interp(np.arange(6), indices, ten[:, j])
                for j in range(3)]).reshape(-1))
    sc60 = StandardScaler().fit(np.stack(minute_vectors))
    p60 = PCA(n_components=2, random_state=0).fit(sc60.transform(np.stack(minute_vectors)))
    return sc10, p10, sc60, p60, len(blocks), len(minute_vectors)


def augmented(frame, arrays, hierarchy):
    rows = []
    for key in frame.index:
        a = arrays.get(key)
        if a is None:
            rows.append(np.full(28, np.nan))
        else:
            rows.append(np.r_[features_for_minute(a, slice(1,4), *hierarchy["csi"][:4]),
                              features_for_minute(a, slice(4,7), *hierarchy["radar"][:4])])
    a = np.stack(rows)
    names = [f"{mod}_pca_{level}_{stat}_pc{j}" for mod in ("csi", "radar")
             for level, stat, count in (("1s","mean",3),("1s","std",3),
                                        ("10s","mean",3),("10s","std",3),
                                        ("1min","value",2)) for j in range(1,count+1)]
    return pd.concat([frame, pd.DataFrame(a, index=frame.index, columns=names)], axis=1)


def xform(train, val, cols):
    a = train[cols].to_numpy(float)
    a[~np.isfinite(a)] = np.nan
    med = np.nanmedian(a, axis=0)
    med = np.where(np.isfinite(med), med, 0)
    a = np.where(np.isnan(a), med, a)
    b = val[cols].to_numpy(float)
    b[~np.isfinite(b)] = np.nan
    b = np.where(np.isnan(b), med, b)
    scaler = StandardScaler().fit(a)
    return scaler.transform(a), scaler.transform(b), med, scaler


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "auc": float(roc_auc_score(y, p))}


def select():
    if (OUT / "selection.json").exists():
        raise FileExistsError("Existing PCA selection is frozen; refusing overwrite")
    base = old_physical()
    train = base.loc[base.index.str.startswith("train/")]
    val = base.loc[base.index.str.startswith("test_minutes/")]
    tr_d, va_d = dots("train"), dots("val")
    # The cached first-level CSI/radar PC coordinates use training-fitted bases.
    hierarchy = {mod: fit_hierarchy(tr_d, [k for k in train.index if k in tr_d], cols)
                 for mod, cols in (("csi",slice(1,4)),("radar",slice(4,7)))}
    tr, va = augmented(train, tr_d, hierarchy), augmented(val, va_d, hierarchy)
    ccols = CSI + [c for c in tr if c.startswith("csi_pca_")]
    rcols = RAD + [c for c in tr if c.startswith("radar_pca_")]
    views = {"early": ccols+rcols, "csi": ccols, "radar": rcols}
    ytr, yva = tr.label.to_numpy(int), va.label.to_numpy(int)
    preds = {}
    for view, cols in views.items():
        xtr, xva, med, sc = xform(tr, va, cols)
        for name in NAMES:
            model = make_model(name).fit(xtr, ytr)
            preds[(view,name)] = model.predict_proba(xva)[:,1]
            joblib.dump({"model":model,"columns":cols,"median":med,"scaler":sc},
                        OUT / f"{view}_{name}.joblib")
            print("fit", view, name, flush=True)
    records=[]
    for name in NAMES:
        for stage in ("early", "late"):
            p = preds[("early",name)] if stage=="early" else (
                preds[("csi",name)] + preds[("radar",name)])/2
            th = fit_threshold(yva,p)
            records.append({"family":"DL" if name=="MLP" else "ML",
                            "model":name,"fusion":stage,"threshold":float(th),
                            "validation":metrics(yva,p,th)})
    joblib.dump(hierarchy, OUT/"hierarchy.joblib")
    manifest={"protocol":"minute-level physical descriptors plus 1-s PC mean/std, 10-s PC mean/std, and 1-min PC coordinates for each modality; all PC bases fit on train; fixed equal-weight late fusion; validation-fitted thresholds; no test fitting",
              "train_minutes":len(tr),"validation_minutes":len(va),
              "pca_training_samples":{m:{"10s":int(h[4]),"1min":int(h[5])} for m,h in hierarchy.items()},
              "results":records}
    (OUT/"selection.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


def evaluate():
    manifest=json.loads((OUT/"selection.json").read_text())
    hierarchy=joblib.load(OUT/"hierarchy.joblib")
    test=pd.read_csv(P/"features_multilink_5s.csv")
    for c in CSI:
        tail=c.removeprefix("csi_")
        test[c]=test[[f"csi1_{tail}",f"csi2_{tail}"]].mean(axis=1)
    test["minute"]=(test.t_rel//60).astype(int)
    base=test.groupby(["cap","minute"],sort=True)[CSI+RAD].mean().assign(
        label=test.groupby(["cap","minute"]).label.first())
    projected=dots("testml_trainbasis")
    rows=[]
    for cap,minute in base.index:
        a=projected.get(cap)
        rows.append(np.full(28,np.nan) if a is None else np.r_[
            features_for_minute(a[60*minute:60*(minute+1)],slice(1,4),*hierarchy["csi"][:4]),
            features_for_minute(a[60*minute:60*(minute+1)],slice(4,7),*hierarchy["radar"][:4])])
    a=np.stack(rows)
    names=[f"{mod}_pca_{level}_{stat}_pc{j}" for mod in ("csi","radar")
           for level,stat,count in (("1s","mean",3),("1s","std",3),
                                    ("10s","mean",3),("10s","std",3),
                                    ("1min","value",2)) for j in range(1,count+1)]
    base=pd.concat([base,pd.DataFrame(a,index=base.index,columns=names)],axis=1)
    y=base.label.to_numpy(int)
    preds={}
    for view in ("early","csi","radar"):
        for name in NAMES:
            obj=joblib.load(OUT/f"{view}_{name}.joblib")
            x=base[obj["columns"]].to_numpy(float)
            x[~np.isfinite(x)]=np.nan
            x=np.where(np.isnan(x),obj["median"],x)
            preds[(view,name)]=obj["model"].predict_proba(obj["scaler"].transform(x))[:,1]
    results=[]
    for r in manifest["results"]:
        name,stage=r["model"],r["fusion"]
        p=preds[("early",name)] if stage=="early" else (
            preds[("csi",name)]+preds[("radar",name)])/2
        results.append({**r,"test":metrics(y,p,r["threshold"])})
    report={"protocol":manifest["protocol"],"test_minutes":len(base),"results":results}
    (OUT/"evaluation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",required=True,choices=("select","evaluate"))
    {"select":select,"evaluate":evaluate}[ap.parse_args().phase]()
