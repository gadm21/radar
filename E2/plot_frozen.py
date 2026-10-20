"""Publication plots for the validation-frozen comparison and full LLM cohort."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/"outputs/frozen"
FIG=OUT/"figs"
FIG.mkdir(exist_ok=True)
BLUE="#247BA0"
ORANGE="#D27B35"
RED="#A84968"

def models():
    data=json.loads((OUT/"evaluation.json").read_text())
    labels=["CSI\nMLP","Radar\nRF","Early fusion\nGBM","Late fusion\nRF"]
    keys=[data["selected_views"][v] for v in ("CSI","radar","early fusion","late probability fusion")]
    fig,axes=plt.subplots(2,1,figsize=(3.45,3.15),sharey=True)
    x=np.arange(4)
    for ax,unit in zip(axes,("window","minute")):
        v=[data["results"][k]["validation_window" if unit=="window" else "validation"]["accuracy"] for k in keys]
        t=[data["results"][k]["test_window" if unit=="window" else "test"]["accuracy"] for k in keys]
        ax.bar(x-.17,v,.34,color=BLUE,label="Validation")
        ax.bar(x+.17,t,.34,color=ORANGE,label="Multilink test")
        ax.set_xticks(x,labels,fontsize=7)
        ax.set_ylim(0,1.07)
        ax.set_title(unit.capitalize()+" accuracy",fontsize=8.5)
        ax.tick_params(axis="y",labelsize=7)
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
    fig.supylabel("Accuracy",fontsize=8,x=.025)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,frameon=False,loc="upper center",ncol=2,
               bbox_to_anchor=(.5,.99),fontsize=7)
    fig.subplots_adjust(left=.16,right=.99,bottom=.085,top=.80,hspace=.63)
    fig.savefig(FIG/"frozen_comparison.pdf",bbox_inches="tight")
    fig.savefig(FIG/"frozen_comparison.png",dpi=350,bbox_inches="tight")
    plt.close(fig)

def primary_cm():
    """One-column four-panel confusion figure for Table III's baseline models."""
    data=json.loads((OUT/"evaluation.json").read_text())
    labels=["CSI / MLP","Radar / RF","Early / GBM","Late / RF"]
    keys=[data["selected_views"][v] for v in
          ("CSI","radar","early fusion","late probability fusion")]
    fig,axes=plt.subplots(2,2,figsize=(3.48,2.72),layout="constrained")
    for ax,label,key in zip(axes.flat,labels,keys):
        arr=np.array(data["results"][key]["test"]["confusion_matrix"])
        ax.imshow(arr,cmap="Blues",vmin=0,vmax=180)
        for (i,j),v in np.ndenumerate(arr):
            ax.text(j,i,str(v),ha="center",va="center",fontsize=10,
                    color="white" if v>90 else "#1E3441")
        ax.set_title(label,fontsize=8)
        ax.set_xticks([0,1],["E","O"],fontsize=7)
        ax.set_yticks([0,1],["E","O"],fontsize=7)
    fig.supxlabel("Predicted",fontsize=8)
    fig.supylabel("Actual",fontsize=8)
    fig.savefig(FIG/"primary_cm.pdf",bbox_inches="tight")
    fig.savefig(FIG/"primary_cm.png",dpi=350,bbox_inches="tight")
    plt.close(fig)

def llm():
    path=OUT/"llm/gpt-4o-mini_summary.json"
    if not path.exists(): return
    mini=json.loads(path.read_text())
    evaluation=json.loads((OUT/"evaluation.json").read_text())
    cm=[("GPT-4o mini",np.array(mini["test"]["runs"][0]["confusion_matrix"])),
        ("ML: late RF",np.array(evaluation["results"]["RF|late probability fusion"]["test"]["confusion_matrix"])),
        ("DL: CSI MLP",np.array(evaluation["results"]["MLP|CSI"]["test"]["confusion_matrix"]))]
    gpt=OUT/"llm/gpt-5.4_summary.json"
    if gpt.exists():
        d=json.loads(gpt.read_text())
        cm.insert(1,("GPT-5.4",np.array(d["test"]["runs"][0]["confusion_matrix"])))
    fig,axes=plt.subplots(1,len(cm),figsize=(2.35*len(cm),2.55),layout="constrained")
    for ax,(name,arr) in zip(axes,cm):
        ax.imshow(arr,cmap="Blues",vmin=0,vmax=180)
        for (i,j),v in np.ndenumerate(arr):
            ax.text(j,i,str(v),ha="center",va="center",color="white" if v>85 else "#1E3441",fontsize=11)
        ax.set_xticks([0,1],["empty","occupied"],fontsize=7)
        ax.set_yticks([0,1],["empty","occupied"],fontsize=7)
        ax.set_title(name,fontsize=9)
        ax.set_xlabel("Predicted",fontsize=8)
    axes[0].set_ylabel("Actual",fontsize=8)
    fig.savefig(FIG/"frozen_llm_cm.pdf",bbox_inches="tight")
    fig.savefig(FIG/"frozen_llm_cm.png",dpi=350,bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    models(); primary_cm(); llm()
