"""Monte Carlo interventional SHAP for the best passing physical models."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
                     "axes.edgecolor":"#4C5963", "axes.linewidth":.7,
                     "figure.facecolor":"white", "savefig.facecolor":"white"})
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from physical_comparison import load, matrix, VIEWS, RAD, CSI, OUT, FIG


def make_model(name):
    return {
        "RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=12,
            class_weight="balanced", random_state=42, n_jobs=1),
        "GBM": lambda: HistGradientBoostingClassifier(max_iter=200,
            learning_rate=.07, random_state=42),
        "SVM": lambda: SVC(C=4, kernel="rbf", probability=True,
            class_weight="balanced", random_state=42),
        "MLP": lambda: MLPClassifier((64,32), alpha=1e-3, max_iter=400,
            random_state=42),
    }[name]()


class LateRF:
    def __init__(self, full_names, Xtr, ytr):
        self.ri = [full_names.index(c) for c in RAD]
        self.ci = [full_names.index(c) for c in CSI]
        self.radar = make_model("RF").fit(Xtr[:,self.ri],ytr)
        self.csi = make_model("RF").fit(Xtr[:,self.ci],ytr)

    def predict_proba(self, X):
        return (self.radar.predict_proba(X[:,self.ri])+
                self.csi.predict_proba(X[:,self.ci]))/2


def shap_estimate(model, foreground, background, seed=73, orders=32):
    """Estimate interventional SHAP with random orders and train backgrounds."""
    rng = np.random.default_rng(seed)
    n, d = foreground.shape
    values = np.zeros((n,d))
    for _ in range(orders):
        current = background[rng.integers(len(background),size=n)].copy()
        prev = model.predict_proba(current)[:,1]
        for j in rng.permutation(d):
            current[:,j] = foreground[:,j]
            new = model.predict_proba(current)[:,1]
            values[:,j] += new-prev
            prev = new
    return values/orders


def main():
    passed = pd.read_csv(OUT/"physical_over70.csv")
    q = passed[passed.split=="test"].copy()
    q["best"] = q[["acc_med","acc_km"]].max(axis=1)
    choices = []
    for family, subset in (("ML",q[q.model!="MLP"]),
                           ("DL",q[q.model=="MLP"])):
        if not subset.empty:
            choices.append((family,subset.sort_values("best",ascending=False).iloc[0]))
    if not choices:
        raise RuntimeError("No passing physical model for SHAP")
    train, _, test = load()
    rng = np.random.default_rng(73)
    idx = np.concatenate([rng.choice(np.flatnonzero(test.label.to_numpy()==lab),
                                      16,replace=False) for lab in (0,1)])
    all_rows = []
    fig, axes = plt.subplots(1,len(choices),figsize=(5.6*len(choices),4.4),squeeze=False)
    for ax,(family,row) in zip(axes.ravel(),choices):
        cols = RAD+CSI if row.view=="late probability fusion" else VIEWS[row.view]
        fill = np.nanmedian(train[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float),axis=0)
        tr = matrix(train,cols,fill)
        te = matrix(test,cols,fill)
        scaler = StandardScaler().fit(tr)
        Xtr = scaler.transform(tr)
        Xte = scaler.transform(te) if row.scale=="raw" else StandardScaler().fit_transform(te)
        ytr=train.label.to_numpy().astype(int)
        model = (LateRF(cols,Xtr,ytr) if row.view=="late probability fusion"
                 else make_model(row.model).fit(Xtr,ytr))
        values = shap_estimate(model,Xte[idx],Xtr,orders=32)
        mean = np.abs(values).mean(axis=0)
        rank = np.argsort(mean)[::-1]
        for j in rank:
            all_rows.append({"family":family,"model":row.model,"view":row.view,
                "scale":row.scale,"descriptor":cols[j],"mean_abs_shap":mean[j]})
        top = rank[:10][::-1]
        ax.barh(np.arange(len(top)),mean[top],color="#0072B2" if family=="ML" else "#D55E00")
        ax.set_yticks(np.arange(len(top)),[cols[j] for j in top])
        ax.set_xlabel("Mean |SHAP| in occupied probability")
        ax.set_title(f"{family}: {row.model}, {row.view}, {row.scale}")
    fig.tight_layout()
    fig.savefig(FIG/"physical_shap.png",dpi=360,bbox_inches="tight")
    fig.savefig(FIG/"physical_shap.pdf",dpi=360,bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(all_rows).to_csv(OUT/"physical_shap.csv",index=False)
    (OUT/"physical_shap_method.json").write_text(json.dumps({
        "foreground":"32 balanced held-out five-second windows",
        "background":"random training window per foreground row and order",
        "orders":32,"seed":73,"estimand":"interventional probability contribution",
        "selection":"highest test minute accuracy among configurations passing >70% on validation and test; exploratory"},indent=2))
    print(pd.DataFrame(all_rows).groupby("family").head(6).to_string(index=False))


if __name__=="__main__": main()
