"""Exploratory PCA-path + radar-SNR fusion; validation-only model selection.

Run after csi_pca_path.py: python E2/csi_radar_fusion.py
No deployment models or source labels are modified.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
import common as C
from snr_minutes_plot import SPLITS, SPLIT_XLABELS

OUT = C.OUTPUT_DIR
FIG = OUT / 'figs'
CSI = 'path2d_10s_median'
COLORS = {'empty':'#252525', 'sleep':'#cca600', 'present':'#d62728'}


def thresholds(values):
    unique = np.unique(values)
    return np.r_[-np.inf, (unique[:-1]+unique[1:])/2, np.inf]


def choose_threshold(y, score):
    candidates = thresholds(score)
    results = []
    for t in candidates:
        pred = score >= t
        tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
        results.append(((tp/(tp+fn)+tn/(tn+fp))/2, -fp))
    best = max(results)
    indices = [i for i,r in enumerate(results) if np.allclose(r,best)]
    return float(candidates[indices[len(indices)//2]])


def measure(y, score, pred):
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y), empty=int(tn+fp), occupied=int(tp+fn),
                auc=roc_auc_score(y,score) if len(np.unique(y))==2 else np.nan,
                balanced_accuracy=balanced_accuracy_score(y,pred),
                occupancy_recall=tp/(tp+fn) if tp+fn else np.nan,
                empty_specificity=tn/(tn+fp) if tn+fp else np.nan,
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def fit_rules(X,y,scales,fixed_radar=None):
    """Exact validation search over OR thresholds; labels never include test."""
    tr = thresholds(X[:,0]) if fixed_radar is None else np.array([fixed_radar])
    tc = thresholds(X[:,1])
    pr = X[:,0,None] >= tr
    pc = X[:,1,None] >= tc
    ranked = []
    for i,a in enumerate(tr):
        pred = pr[:,i,None] | pc
        tpr = pred[y==1].mean(axis=0)
        fpr = pred[y==0].mean(axis=0)
        for j,b in enumerate(tc):
            ranked.append(((float((tpr[j]+1-fpr[j])/2),float(-fpr[j])),float(a),float(b)))
    best = max(r[0] for r in ranked)
    ties = [r for r in ranked if np.allclose(r[0],best)]
    _,a,b = ties[len(ties)//2]
    def score(x):
        parts=[]
        for j,t in enumerate([a,b]):
            if np.isneginf(t):
                return np.ones(len(x))
            if np.isfinite(t):
                parts.append((x[:,j]-t)/scales[j])
        return np.max(parts,axis=0) if parts else -np.ones(len(x))
    return score, [a,b]


def plots(df, predictors, thresholds_by_model, selected):
    paired = df.dropna(subset=['snr_db',CSI]).copy()
    X = np.c_[paired.snr_db,np.log10(paired[CSI].clip(lower=1e-9))]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,split in zip(axes,['validation_minutes','test_minutes']):
        mask=(paired.split==split).to_numpy()
        for name,fun in predictors.items():
            score=fun(X[mask])
            fpr,tpr,_=roc_curve(paired.loc[mask,'label'],score)
            ax.plot(fpr,tpr,label=f'{name}: {roc_auc_score(paired.loc[mask,"label"],score):.3f}')
        ax.plot([0,1],[0,1],'--',c='#aaa')
        ax.set(title=SPLIT_XLABELS[split],xlabel='False-positive rate (empty called occupied)',ylabel='Occupied recall',xlim=(0,1),ylim=(0,1.02))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Radar + CSI fusion | matched minutes | ROC AUC')
    fig.tight_layout()
    fig.savefig(FIG/'csi_radar_fusion_roc.png',dpi=160)
    plt.close(fig)
    xmin,xmax=X[:,0].min()-.5,X[:,0].max()+.5
    ymin,ymax=X[:,1].min()-.15,X[:,1].max()+.15
    xx,yy=np.meshgrid(np.linspace(xmin,xmax,250),np.linspace(ymin,ymax,250))
    zz=predictors[selected](np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    cut=thresholds_by_model[selected]
    fig,axes=plt.subplots(1,4,figsize=(17,4.8),sharex=True,sharey=True)
    for ax,split in zip(axes,SPLITS):
        ax.contourf(xx,10**yy,(zz>=cut).astype(int),levels=[-.5,.5,1.5],colors=['#d9e6ee','#fce2d2'],alpha=.55)
        if zz.min()<cut<zz.max():
            ax.contour(xx,10**yy,zz,levels=[cut],colors=['#666'],linewidths=1)
        for act,color in COLORS.items():
            g=paired[(paired.split==split)&(paired.activity==act)]
            ax.scatter(g.snr_db,g[CSI],c=color,s=13,alpha=.65,label=act)
        ax.set_title(SPLIT_XLABELS[split])
        ax.set_xlabel('Radar SNR (dB)')
        ax.set_yscale('log')
    axes[0].set_ylabel('10 s 2D CSI path length (minute median)')
    axes[-1].legend()
    fig.suptitle(f'{selected}: fixed decision boundary | blue = empty, peach = occupied')
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(FIG/'csi_radar_fusion_boundary.png',dpi=160)
    plt.close(fig)


def main():
    df=pd.read_csv(OUT/'csi_pca_path_minutes.csv')
    df=df[df.label.isin([0,1]) & df.activity.isin(COLORS)].dropna(subset=['snr_db']).copy()
    paired=df.dropna(subset=[CSI]).copy()
    X=np.c_[paired.snr_db,np.log10(paired[CSI].clip(lower=1e-9))]
    y=paired.label.to_numpy()
    train=paired.split.isin(['calibration_minutes','train']).to_numpy()
    val=(paired.split=='validation_minutes').to_numpy()
    predictors={'radar':lambda x:x[:,0], 'csi':lambda x:x[:,1]}
    learned={
        'fusion_logistic':make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42)),
        'fusion_tree':DecisionTreeClassifier(max_depth=2,min_samples_leaf=20,class_weight='balanced',random_state=42),
    }
    for name,model in learned.items():
        model.fit(X[train],y[train])
        predictors[name]=lambda x,model=model:model.predict_proba(x)[:,1]
    cuts={name:choose_threshold(y[val],fun(X[val])) for name,fun in predictors.items()}
    predictors['fusion_or'],or_cuts=fit_rules(X[val],y[val],np.maximum(X[train].std(axis=0),1e-9))
    cuts['fusion_or']=0.0
    predictors['fusion_preserve_radar'],preserve_cuts=fit_rules(
        X[val],y[val],np.maximum(X[train].std(axis=0),1e-9),fixed_radar=cuts['radar'])
    cuts['fusion_preserve_radar']=0.0
    # Selection uses validation balanced accuracy, then specificity; fixed
    # candidate order breaks remaining ties. Do not select based on test.
    fusion_names=list(learned)+['fusion_or','fusion_preserve_radar']
    selected=max(fusion_names,key=lambda name:(
        balanced_accuracy_score(y[val],predictors[name](X[val])>=cuts[name]),
        np.mean(predictors[name](X[val & (y==0)])<cuts[name])))
    print('Validation-selected fusion:',selected,flush=True)
    prediction=paired[['split','minute','activity','label','snr_db',CSI]].copy()
    rows=[]
    for name,fun in predictors.items():
        score=fun(X)
        pred=score>=cuts[name]
        prediction[name+'_score']=score
        prediction[name+'_pred']=pred.astype(int)
        for split in ['validation_minutes','test_minutes']:
            mask=(paired.split==split).to_numpy()
            rows.append(dict(cohort='paired',model=name,split=split,**measure(y[mask],score[mask],pred[mask])))
    # All radar-eligible minutes; missing CSI invokes the same radar baseline.
    # Binary predictions only: avoid misleading AUC from incomparable fallback scores.
    full=df[['split','minute','activity','label','snr_db',CSI]].copy()
    full['csi_available']=full[CSI].notna()
    full['radar_pred']=(full.snr_db>=cuts['radar']).astype(int)
    full['fusion_pred']=full.radar_pred.copy()
    full['preserve_radar_pred']=full.radar_pred.copy()
    available=full.csi_available
    full.loc[available,'fusion_pred']=(predictors[selected](np.c_[full.loc[available,'snr_db'],np.log10(full.loc[available,CSI].clip(lower=1e-9))])>=cuts[selected]).astype(int)
    full.loc[available,'preserve_radar_pred']=(predictors['fusion_preserve_radar'](np.c_[full.loc[available,'snr_db'],np.log10(full.loc[available,CSI].clip(lower=1e-9))])>=0).astype(int)
    full.to_csv(OUT/'csi_radar_fusion_all_predictions.csv',index=False)
    for split in ['validation_minutes','test_minutes']:
        g=full[full.split==split]
        for name,col in [('radar','radar_pred'),(selected+'_with_radar_fallback','fusion_pred'),
                         ('preserve_radar_with_fallback','preserve_radar_pred')]:
            result=measure(g.label,g[col],g[col])
            result['auc']=np.nan
            rows.append(dict(cohort='all_radar',model=name,split=split,**result))
    metrics=pd.DataFrame(rows)
    metrics.to_csv(OUT/'csi_radar_fusion_metrics.csv',index=False)
    prediction.to_csv(OUT/'csi_radar_fusion_predictions.csv',index=False)
    changes=[]
    for split in ['validation_minutes','test_minutes']:
        g=prediction[prediction.split==split]
        r=g.radar_pred.to_numpy().astype(bool)
        f=g[selected+'_pred'].to_numpy().astype(bool)
        occupied=g.label.to_numpy()==1
        changes.append(dict(split=split,
            radar_misses_recovered=int((occupied & ~r & f).sum()),
            new_occupied_misses=int((occupied & r & ~f).sum()),
            radar_false_alarms_removed=int((~occupied & r & ~f).sum()),
            new_false_alarms=int((~occupied & ~r & f).sum())))
    change_df=pd.DataFrame(changes)
    change_df.to_csv(OUT/'csi_radar_fusion_changes.csv',index=False)
    def finite_json(x):
        return float(x) if np.isfinite(x) else str(x)
    config=dict(feature=CSI,selected_fusion=selected,
                thresholds={k:finite_json(v) for k,v in cuts.items()},
                or_rule=dict(snr_db=finite_json(or_cuts[0]),csi_path=finite_json(10**or_cuts[1])),
                preserve_radar_rule=dict(snr_db=finite_json(preserve_cuts[0]),csi_path=finite_json(10**preserve_cuts[1])),
                logistic_scaled_coefficients=learned['fusion_logistic'][-1].coef_.tolist(),
                logistic_intercept=learned['fusion_logistic'][-1].intercept_.tolist(),
                scaler_mean=learned['fusion_logistic'][0].mean_.tolist(),
                scaler_scale=learned['fusion_logistic'][0].scale_.tolist(),
                tree=export_text(learned['fusion_tree'],feature_names=['snr_db','log10_csi_path']))
    (OUT/'csi_radar_fusion_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    plots(df,predictors,cuts,selected)
    coverage=full.groupby('split').agg(radar_minutes=('minute','size'),paired_minutes=('csi_available','sum'))
    report='''# Radar SNR + CSI PCA path fusion

Feature pair: radar SNR in dB and log10 of median 10-second 2D PCA path length.
This follows the preceding PCA analysis. Ten seconds retains more usable
minutes than 20/30 seconds; 2D was stronger on validation than 3D. These
datasets have already been explored, so this is not a fresh untouched test.
Sleep is occupied for all binary results.

## Protocol

Use identical paired minutes for radar-only, CSI-only, and fusion comparisons.
Fit standardization and balanced logistic regression (fixed C=1) on calibration
+ train only. Also fit one balanced shallow tree (depth 2, minimum leaf 20) on
that same training pool. Baseline thresholds and learned-model probability
thresholds maximize validation balanced accuracy. The interpretable OR rule
selects two thresholds on validation: occupied if radar exceeds its threshold
OR CSI exceeds its threshold. This rule is calibrated directly on validation;
its validation performance is optimistic, especially after searching thresholds.
Threshold ties favor fewer false positives, then a central candidate.

An additional radar-preserving OR diagnostic fixes the radar threshold to the
radar-only baseline and chooses only the CSI threshold on validation. It was
added after finding that free OR calibration behaves like CSI-only; it tests
the operational constraint that CSI must not undo radar-positive decisions.

Select among the fusion candidates using validation balanced accuracy,
then specificity; test never selects candidates or thresholds. Export every
candidate's test result for transparency, but retain the validation-selected
candidate even if another performs better on test. ROC AUC for OR uses the max
signed threshold margin, scaled by training standard deviations; it is a
ranking score, not a probability. A threshold of zero implements the OR rule.

For all-radar evaluation, missing CSI paths fall back to the exact same radar
baseline and threshold. Report binary operating-point metrics there, not AUC
of a mixture of incompatible score scales. No zeros are substituted for CSI.
This coverage fallback does not detect sensor failures beyond the preceding
CSI quality filtering. Calibration is sleep-only and dominates training;
class/time/placement confounding and serially correlated minutes remain.
The PCA basis itself was trained exclusively on calibration + train.

## Results

'''
    report+=f'Validation-selected fusion: **{selected}**.\n\n'
    report+='Matched and all-radar metrics:\n\n```text\n'+metrics.to_string(index=False,float_format=lambda x:f'{x:.3f}')+'\n```\n\n'
    report+='Changes versus radar on paired minutes:\n\n```text\n'+change_df.to_string(index=False)+'\n```\n\n'
    report+='Coverage:\n\n```text\n'+coverage.to_string()+'\n```\n\n'
    report+='## Interpretation\n\n'
    for split in ['validation_minutes','test_minutes']:
        g=prediction[prediction.split==split]
        different=int((g[selected+'_pred']!=g.csi_pred).sum())
        report+=f'- {split}: selected fusion differs from CSI-only on {different}/{len(g)} paired minutes.\n'
    report+='\nImprovement over radar alone must be distinguished from improvement over CSI alone. '
    report+='A rule that reproduces CSI decisions has not established complementary sensor information. '
    report+='Inspect false positives and missed occupancy as well as balanced accuracy. '
    report+='The shallow tree uses only radar in its fitted splits; the logistic CSI coefficient is near zero in this run. '
    report+='Training calibration is predominantly sleep, and these models may learn collection conditions rather than a stable occupancy relationship.\n\n'
    report+='Configuration:\n\n```json\n'+json.dumps(config,indent=2)+'\n```\n\n'
    report+='''## Files and reproduction

Run `python E2/csi_radar_fusion.py` after `E2/csi_pca_path.py`, using
E2/requirements.txt. CSV files with prefix csi_radar_fusion contain metrics,
per-minute predictions, changes, and full-coverage fallback predictions.
csi_radar_fusion_config.json records selection, thresholds, scaler, and model
parameters. Figures: figs/csi_radar_fusion_roc.png and
figs/csi_radar_fusion_boundary.png. No deployed model was changed.
'''
    (OUT/'CSI_RADAR_FUSION_REPORT.md').write_text(report,encoding='utf-8')
    print(metrics.to_string(index=False),flush=True)
    print(change_df.to_string(index=False),flush=True)
    print(json.dumps(config,indent=2),flush=True)


if __name__=='__main__':
    main()
