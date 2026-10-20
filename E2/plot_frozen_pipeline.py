"""Compact vector system diagram using literal fields/instructions from llm_full_frozen.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "paper/icc_assets"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans", "figure.facecolor":"white",
                     "savefig.facecolor":"white"})
fig = plt.figure(figsize=(7.15, 3.1))
blue, orange, purple, ink = "#176C99", "#C05B22", "#673E91", "#213542"
teal, grey = "#41626B", "#687C87"
MONO = "DejaVu Sans Mono"

def card(x,y,w,h,edge,fill,lw=1.1,ls="-"):
    fig.add_artist(FancyBboxPatch((x,y),w,h,
        boxstyle="round,pad=.006,rounding_size=.016",transform=fig.transFigure,
        facecolor=fill,edgecolor=edge,linewidth=lw,linestyle=ls))

def txt(x,y,s,size=8,color=ink,weight=None,ha="left",family=None,va="center"):
    fig.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,
             family=family)

def arrow(x0,y0,x1,y1,color=grey,style="-|>",lw=1.15):
    fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,
        arrowstyle=style,mutation_scale=10,linewidth=lw,color=color))

def header(x,y,n,s,color):
    fig.add_artist(Circle((x,y),.013,transform=fig.transFigure,color=color))
    txt(x,y,str(n),6.8,"white","bold",ha="center")
    txt(x+.02,y,s,7.6,color,"bold")

# 1. Sensors: waveform and radar arcs.
card(.03,.55,.135,.37,blue,"#EDF6FB")
xx=np.linspace(.045,.153,90)
fig.lines.append(plt.Line2D(xx,.79+.045*np.sin(np.linspace(0,5*np.pi,90)),
    transform=fig.transFigure,color=blue,linewidth=1.5))
txt(.097,.66,"WiFi CSI",8.4,blue,"bold",ha="center")
txt(.097,.6,"52 subcarrier amplitudes",6.1,ink,ha="center")
card(.03,.08,.135,.37,orange,"#FEF2E9")
for r in (.025,.042,.059):
    fig.add_artist(Arc((.097,.3),2*r,2*r,theta1=25,theta2=155,
                       transform=fig.transFigure,color=orange,linewidth=1.3))
fig.add_artist(Circle((.097,.299),.006,transform=fig.transFigure,color=orange))
txt(.097,.19,"FMCW radar",8.4,orange,"bold",ha="center")
txt(.097,.13,"RD / RA / RE / XY maps",6.1,ink,ha="center")

# 2. Physical + PCA descriptors (fifteen minute means enter the prompt).
card(.2,.08,.15,.84,teal,"#EEF4F5")
for i,h in enumerate((.06,.105,.08,.125)):
    fig.add_artist(FancyBboxPatch((.234+i*.022,.66),.013,h,
        boxstyle="square,pad=0",transform=fig.transFigure,
        facecolor=blue if i<2 else orange,edgecolor="none"))
txt(.275,.58,"15 minute means",8,teal,"bold",ha="center")
txt(.275,.5,"CSI: amp mean / std,",6.3,ink,ha="center")
txt(.275,.455,"rolling var., PC1-3",6.3,ink,ha="center")
txt(.275,.385,"Radar: SNR, RA/RE/XY",6.3,ink,ha="center")
txt(.275,.34,"var. P90, RA/RE delta,",6.3,ink,ha="center")
txt(.275,.295,"PC1-3",6.3,ink,ha="center")
txt(.275,.2,"PCA bases and $z$ scores",6.0,grey,ha="center")
txt(.275,.155,"fitted on training only",6.0,grey,ha="center")
arrow(.17,.73,.195,.62,blue)
arrow(.17,.27,.195,.38,orange)

# 3. Frozen references with validation-fixed cutoffs.
card(.38,.08,.15,.84,teal,"#EEF4F5")
fig.add_artist(Circle((.42,.78),.03,transform=fig.transFigure,
                      facecolor="white",edgecolor=blue,linewidth=1.5))
fig.add_artist(Circle((.49,.78),.03,transform=fig.transFigure,
                      facecolor="white",edgecolor=orange,linewidth=1.5))
txt(.42,.78,"GBM",6.8,blue,"bold",ha="center")
txt(.49,.78,"MLP",6.8,orange,"bold",ha="center")
txt(.42,.69,"late fusion",6.0,blue,ha="center")
txt(.49,.69,"early fusion",6.0,orange,ha="center")
txt(.455,.58,"$p_{\\rm ML},\\ p_{\\rm DL}$",8.2,teal,"bold",ha="center")
txt(.455,.49,r"$\theta_{\rm ML}=.662$",6.8,blue,ha="center")
txt(.455,.43,r"$\theta_{\rm DL}=.484$",6.8,orange,ha="center")
txt(.455,.28,"cutoffs fixed on",6.0,grey,ha="center")
txt(.455,.235,"validation",6.0,grey,ha="center")
arrow(.352,.5,.376,.5)

# 4. One chat request: system rule and user JSON body in one prompt.
card(.56,.08,.285,.84,purple,"#F5F0FA")
txt(.572,.875,"Prompt",7.0,purple,"bold")
txt(.572,.815,"system:",6.2,grey,"bold")
for i,s in enumerate(('"Respect agreement of the two',
                      ' reference class predictions;',
                      ' when they disagree, consider',
                      ' their margins from the fixed',
                      ' thresholds and the training-',
                      ' class descriptor statistics."')):
    txt(.572,.775-i*.04,s,5.9,ink,family=MONO)
txt(.572,.5,"user (JSON):",6.2,grey,"bold")
for i,s in enumerate(('"reference_models"','"training_class_mean_std"',
                      '"labeled_training_examples"',
                      '"minutes": [{"id", "descriptors",',
                      '  "ml_probability", "ml_prediction",',
                      '  "dl_probability", "dl_prediction"}]')):
    txt(.572,.45-i*.045,s,5.9,ink,family=MONO)
arrow(.532,.5,.556,.5)

# 5. LLM and outputs.
card(.858,.55,.128,.37,purple,"#F5F0FA")
for dx in (-.026,0,.026):
    for dy in (-.026,0,.026):
        fig.add_artist(Circle((.922+dx,.79+dy),.0055,transform=fig.transFigure,
                              color=purple))
txt(.922,.67,"GPT-4o mini",6.8,purple,"bold",ha="center")
txt(.922,.615,"GPT-5.4",6.8,purple,"bold",ha="center")
arrow(.847,.735,.853,.735,purple)
card(.858,.08,.128,.37,purple,"#F5F0FA")
txt(.922,.37,"occupied_",6.0,ink,ha="center",family=MONO)
txt(.922,.325,"probability",6.0,ink,ha="center",family=MONO)
txt(.922,.25,"cutoff .5 $\\rightarrow$ 0 / 1",6.5,purple,"bold",ha="center")
arrow(.922,.54,.922,.46,purple)

fig.savefig(OUT/"frozen_pipeline.pdf",bbox_inches="tight",pad_inches=.035)
fig.savefig(OUT/"frozen_pipeline.png",dpi=350,bbox_inches="tight",pad_inches=.035)
plt.close(fig)
