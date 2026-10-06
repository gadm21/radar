"""Export measured results to native PGFPlots/LaTeX in the radar root."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.paper_packages'))
import numpy as np
import pandas as pd
EXP=ROOT/'E2/outputs/paper_100hz_variance3s'
OUT=ROOT/'icc_assets';OUT.mkdir(exist_ok=True)
d=pd.read_csv(EXP/'paired_minutes.csv');raw=pd.read_csv(EXP/'all_minutes.csv');res=pd.read_csv(EXP/'results.csv')
ranked=res[res.split=='validation_minutes'].sort_values(['balanced_accuracy','auc','model'],ascending=[False,False,True]).drop_duplicates('features').set_index('features')
def test(name):
    return res[(res.features==name)&(res.model==ranked.loc[name,'model'])&(res.split=='test_minutes')].iloc[0]
def tex(s):return str(s).replace('&',r'\&').replace('_',r'\_')
def f(x):return f'{float(x):.9g}'
selection=json.loads((EXP/'selection.json').read_text());winner=test(selection['features'])
method=json.loads((EXP/'method.json').read_text())
macros={'SelectedFeatures':tex(selection['features']),'SelectedModel':selection['model'],'ValBA':f'{100*selection["balanced_accuracy"]:.1f}','TestBA':f'{100*winner.balanced_accuracy:.1f}','TestAUC':f'{winner.auc:.3f}','ValAUC':f'{selection["auc"]:.3f}','SelectedThreshold':f'{selection["threshold"]:.4f}','FitDots':f'{method["pca_fit_dots"]:,}','TestTN':int(winner.tn),'TestFP':int(winner.fp),'TestFN':int(winner.fn),'TestTP':int(winner.tp)}
for key,split in [('Val','validation_minutes'),('Test','test_minutes')]:
    a=raw[raw.split==split];b=d[d.split==split]
    macros[key+'Packets']=f'{int(a.packets.sum()):,}';macros[key+'Frames']=f'{int(a.radar_frames.sum()):,}'
    macros[key+'SelectedPackets']=f'{int(b.packets.sum()):,}';macros[key+'SelectedFrames']=f'{int(b.radar_frames.sum()):,}'
    macros[key+'Interpolation']=f'{100*(a.used_interpolated_samples/a.used_resampled_samples).median():.1f}'
    macros[key+'ValidSeconds']=f'{a.valid_seconds.median():.0f}'
    macros[key+'PacketRate']=f'{a.original_rate_hz.median():.1f}'
macros.update(CsiBA=f'{100*test("CSI").balanced_accuracy:.1f}',SnrFusionBA=f'{100*test("CSI+SNR").balanced_accuracy:.1f}',RdBA=f'{100*test("CSI+SNR+RD").balanced_accuracy:.1f}',RdAUC=f'{test("CSI+SNR+RD").auc:.3f}',RaBA=f'{100*test("CSI+SNR+RA").balanced_accuracy:.1f}',AllBA=f'{100*test("CSI+radar all").balanced_accuracy:.1f}')
(OUT/'metrics.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items()),encoding='utf8')
lines=[]
for name in ranked.index:
    row=test(name);v=ranked.loc[name]
    lines.append(f'{tex(name)} & {row.model} & {v.balanced_accuracy:.3f} & {row.balanced_accuracy:.3f} & {row.auc:.3f} \\\\')
(OUT/'results_rows.tex').write_text('\\newcommand{\\ResultRows}{%\n'+'\n'.join(lines)+'\n}\n',encoding='utf8')

def boxplot(values,pos,color):
    v=np.asarray(values);q1,med,q3=np.quantile(v,[.25,.5,.75]);iqr=q3-q1
    low=v[v>=q1-1.5*iqr].min();high=v[v<=q3+1.5*iqr].max()
    out=v[(v<low)|(v>high)]
    props=f'lower whisker={f(low)},lower quartile={f(q1)},median={f(med)},upper quartile={f(q3)},upper whisker={f(high)}'
    points=' '.join(f'({pos},{f(x)})' for x in out)
    return rf'\addplot+[boxplot prepared={{{props}}},boxplot/draw position={pos},fill={color}!65,draw={color},mark=*,mark size=.55pt] coordinates {{{points}}};'

lines=[r'\begin{tikzpicture}',r'\begin{groupplot}[group style={group size=3 by 2,horizontal sep=1.1cm,vertical sep=1.05cm},width=.30\textwidth,height=3.3cm,boxplot/draw direction=y,xtick={1.5,4.5},xticklabels={Val.,Test},xmin=.3,xmax=5.7,tick label style={font=\scriptsize},title style={font=\footnotesize},label style={font=\scriptsize},ymajorgrids,grid style={gray!15}]']
for col,title,units in [('csi','CSI variance','variance'),('snr_db','Radar SNR proxy','dB'),('rd_mean','Range--Doppler','mean log power'),('ra_mean','Range--azimuth','mean log power'),('re_mean','Range--elevation','mean log power'),('xy_mean','XY energy','mean log power')]:
    lines.append(r'\nextgroupplot[title={'+title+'},ylabel={'+units+'}'+(',ymode=log' if col=='csi' else '')+']')
    for split,pos0 in [('validation_minutes',1),('test_minutes',4)]:
        for label,color in [(0,'emptycolor'),(1,'occColor')]:
            lines.append(boxplot(d[(d.split==split)&(d.label==label)][col],pos0+label,color))
lines.extend([r'\end{groupplot}',r'\end{tikzpicture}'])
(OUT/'boxplots.tex').write_text('\n'.join(lines),encoding='utf8')

names=['CSI','CSI+SNR','CSI+SNR+RD','CSI+SNR+RA','CSI+SNR+RE','CSI+SNR+XY','CSI+radar all']
lines=[r'\begin{tikzpicture}',r'\begin{axis}[width=.98\columnwidth,height=5.2cm,xbar,bar width=3pt,xmin=0,xmax=1,ytick={0,1,2,3,4,5,6},yticklabels={CSI,{+SNR},{+SNR+RD},{+SNR+RA},{+SNR+RE},{+SNR+XY},{+all radar}},y dir=reverse,xlabel={Balanced accuracy},tick label style={font=\scriptsize},label style={font=\footnotesize},legend style={font=\scriptsize,at={(.5,1.03)},anchor=south,legend columns=2,draw=none},enlarge y limits=.12]']
lines[1] = lines[1].replace('xlabel={Balanced accuracy}', 'xlabel={ROC AUC}')
for color,parts in [('valColor',[ranked.loc[n,'auc'] for n in names]),('testColor',[test(n).auc for n in names])]:
    lines.append(r'\addplot[fill='+color+',draw=none] coordinates {'+' '.join(f'({f(v)},{i})' for i,v in enumerate(parts))+'};')
lines.extend([r'\legend{Validation,Test}',r'\end{axis}',r'\end{tikzpicture}'])
(OUT/'performance.tex').write_text('\n'.join(lines),encoding='utf8')

lines=[r'\begin{tikzpicture}',r'\begin{axis}[width=\columnwidth,height=4.1cm,ybar,bar width=5pt,ymin=0,ymax=290,symbolic x coords={CSI,SNR,RD,RA,RE,XY,Joint},xtick=data,tick label style={font=\scriptsize},ylabel={Usable collected minutes},label style={font=\scriptsize},legend style={at={(.5,1.02)},anchor=south,font=\scriptsize,legend columns=2,draw=none},nodes near coords,nodes near coords style={font=\tiny},enlarge x limits=.1]']
for split,color in [('validation_minutes','valColor'),('test_minutes','testColor')]:
    a=raw[raw.split==split]
    cols={'CSI':['csi'],'SNR':['snr_db']}
    for m in ['RD','RA','RE','XY']:cols[m]=[c for c in a if c.startswith(m.lower()+'_')]
    cols['Joint']=sum(cols.values(),[])
    coords=' '.join(f'({k},{int(a[v].notna().all(axis=1).sum())})' for k,v in cols.items())
    lines.append(r'\addplot[fill='+color+',draw=none] coordinates {'+coords+'};')
lines.extend([r'\legend{Validation,Test}',r'\end{axis}',r'\end{tikzpicture}'])
(OUT/'coverage.tex').write_text('\n'.join(lines),encoding='utf8')
lines=[r'\begin{tikzpicture}',r'\begin{groupplot}[group style={group size=2 by 1,horizontal sep=.9cm},width=.45\columnwidth,height=3.5cm,boxplot/draw direction=y,xtick={1,2},xticklabels={Val.,Test},xmin=.4,xmax=2.6,tick label style={font=\scriptsize},title style={font=\scriptsize},ymajorgrids,grid style={gray!15}]']
for what,title in [('valid_seconds','Valid seconds'),('interpolation',r'Interpolated (\%)')]:
    lines.append(r'\nextgroupplot[title={'+title+'}]')
    for i,split in enumerate(['validation_minutes','test_minutes'],1):
        a=raw[raw.split==split];v=a.valid_seconds if what=='valid_seconds' else 100*a.used_interpolated_samples/a.used_resampled_samples
        lines.append(boxplot(v,i,'valColor' if i==1 else 'testColor'))
lines.extend([r'\end{groupplot}',r'\end{tikzpicture}'])
(OUT/'temporal.tex').write_text('\n'.join(lines),encoding='utf8')
print('Exported native LaTeX tables and PGFPlots figures to',OUT)
