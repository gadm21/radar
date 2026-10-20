"""Summarize only physical models passing one shared calibration rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
                     "axes.edgecolor":"#4C5963", "axes.linewidth":.7,
                     "figure.facecolor":"white", "savefig.facecolor":"white"})

from physical_comparison import OUT, FIG


def main():
    d = pd.read_csv(OUT/"physical_all.csv")
    keys = ["model","view","scale","level"]
    rows = []
    for key,g in d.groupby(keys,sort=False):
        if set(g.split)!={"val","test"}: continue
        a = g.set_index("split")
        rules = [r for r in ("med","km") if
                 a.loc["val",f"acc_{r}"]>.70 and a.loc["test",f"acc_{r}"]>.70]
        if not rules: continue
        model,view,scale,level = key
        rule = "med" if "med" in rules else "km"
        stage = "late probability" if view=="late probability fusion" else \
                "early descriptor" if view=="early fusion" else "single"
        rows.append({"model":model,"family":"DL" if model=="MLP" else "ML",
                     "view":view,"fusion_stage":stage,
                     "modality":"multimodal" if "fusion" in view else "single",
                     "scale":scale,"level":level,"qualifying_rule":rule,
                     "passing_rules":"+".join(rules),
                     "val_accuracy":a.loc["val",f"acc_{rule}"],
                     "test_accuracy":a.loc["test",f"acc_{rule}"],
                     "val_median":a.loc["val","acc_med"],
                     "test_median":a.loc["test","acc_med"],
                     "val_two_means":a.loc["val","acc_km"],
                     "test_two_means":a.loc["test","acc_km"]})
    q = pd.DataFrame(rows).sort_values(["level","test_accuracy"],ascending=[True,False])
    q.to_csv(OUT/"physical_over70_pairs.csv",index=False)
    passed = d.merge(q[keys+["qualifying_rule","passing_rules","family",
                             "fusion_stage","modality"]],on=keys,how="inner")
    passed.to_csv(OUT/"physical_over70.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.7),sharex=True)
    for ax,level in zip(axes,("minute","window")):
        p=q[q.level.eq(level)].sort_values("test_accuracy")
        yy=np.arange(len(p))
        ax.scatter(p.val_accuracy,yy,marker="o",s=35,color="#0072B2",label="Validation")
        ax.scatter(p.test_accuracy,yy,marker="s",s=35,color="#D55E00",label="Multilink test")
        for y,r in enumerate(p.itertuples()):
            ax.plot([r.val_accuracy,r.test_accuracy],[y,y],color="#b0bdc7",lw=1,zorder=0)
        view_short={"late probability fusion":"L", "early fusion":"E",
                    "stable radar":"S", "radar":"R", "CSI":"C"}
        ax.set_yticks(yy,[f"{r.model} {view_short[r.view]} · {'batch' if r.scale=='rescaled' else 'raw'} · {r.qualifying_rule}" for r in p.itertuples()],fontsize=8.5)
        ax.axvline(.70,color="black",linestyle="--",lw=.8)
        ax.set_xlim(.65,1.02)
        ax.set_title(f"{level.title()} decisions (n={len(p)})")
        ax.set_xlabel("Accuracy")
        ax.xaxis.grid(True,color="#E6EBEF",linewidth=.6)
        ax.set_axisbelow(True)
    axes[1].legend(loc="lower right",fontsize=8,framealpha=.95)
    fig.tight_layout()
    fig.savefig(FIG/"e2_passing_models.png",dpi=360,bbox_inches="tight")
    fig.savefig(FIG/"e2_passing_models.pdf",dpi=360,bbox_inches="tight")
    plt.close(fig)
    print(q[["family","model","view","scale","level","qualifying_rule",
             "val_accuracy","test_accuracy"]].to_string(index=False))


if __name__=="__main__": main()
