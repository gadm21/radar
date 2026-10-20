"""Validation-selected interventional Shapley estimates, with train background."""
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frozen_evaluation import OUT, OLD, RAD, CSI, matrix

FIG=OUT/"figs"
FIG.mkdir(exist_ok=True)


def predictor(key):
    def single(name):
        obj=joblib.load(OUT/(name.replace("|","__")+".joblib"))
        cols=obj["columns"]
        def call(frame):
            x=matrix(frame,cols,obj["fill"])
            return obj["model"].predict_proba(obj["scaler"].transform(x))[:,1]
        return call
    if key=="RF|late probability fusion":
        a,b=single("RF|CSI"),single("RF|radar")
        return RAD+CSI,lambda frame:(a(frame)+b(frame))/2
    cols=(RAD if key.endswith("|radar") else CSI if key.endswith("|CSI") else RAD+CSI)
    return cols,single(key)


def explain(foreground,background,cols,fn,seed=73,orders=24):
    rng=np.random.default_rng(seed)
    phi=np.zeros((len(foreground),len(cols)))
    for _ in range(orders):
        current=background.iloc[rng.integers(len(background),size=len(foreground))][cols].reset_index(drop=True).copy()
        base=fn(current)
        perm=rng.permutation(len(cols))
        for j in perm:
            current[cols[j]]=foreground[cols[j]].to_numpy()
            after=fn(current)
            phi[:,j]+=after-base
            base=after
    return phi/orders


def main():
    sel=json.loads((OUT/"selection.json").read_text())
    data=pd.read_csv(OLD)
    train=data[data.rec_id.str.startswith("train/")].copy()
    val=data[data.rec_id.str.startswith("test_minutes/")].copy()
    rng=np.random.default_rng(73)
    idx=np.concatenate([rng.choice(np.flatnonzero(val.label.to_numpy()==c),12,replace=False) for c in (0,1)])
    foreground=val.iloc[idx].reset_index(drop=True)
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.65))
    for ax,fam,color in zip(axes,("ML","DL"),("#A84968","#247BA0")):
        key=sel["references"][fam]["key"]
        cols,fn=predictor(key)
        phi=explain(foreground,train,cols,fn)
        scores=np.abs(phi).mean(axis=0)
        order=np.argsort(scores)[::-1]
        rows.extend({"family":fam,"model":key,"descriptor":cols[j],"mean_abs_shap":float(scores[j])} for j in order)
        top=order[:9][::-1]
        ax.barh([cols[j] for j in top],scores[top],color=color)
        ax.set_title(f"{fam}: {key.replace('|', ' / ')}",fontsize=10)
        ax.set_xlabel("Mean absolute probability contribution")
        ax.grid(axis="x",alpha=.2)
    fig.tight_layout()
    fig.savefig(FIG/"frozen_shap.pdf",bbox_inches="tight")
    fig.savefig(FIG/"frozen_shap.png",dpi=320,bbox_inches="tight")
    pd.DataFrame(rows).to_csv(OUT/"frozen_shap.csv",index=False)
    (OUT/"frozen_shap_method.json").write_text(json.dumps({
        "selection":"ML and DL model by validation-minute AUC only",
        "foreground":"24 validation windows (12 per class)",
        "background":"random training window for each foreground row and order",
        "orders":24,"seed":73,
        "estimand":"Monte Carlo interventional Shapley contribution to occupied probability; descriptive, not causal"
    },indent=2))
    print(pd.DataFrame(rows).groupby("family").head(5).to_string(index=False))


if __name__=="__main__": main()
