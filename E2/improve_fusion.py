"""Coverage-aware CSI/radar fusion search, selected without test labels.

Requires csi_pca_path.py outputs. Run python E2/improve_fusion.py.
"""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import common as C
from csi_radar_fusion import choose_threshold, measure

OUT=C.OUTPUT_DIR
FIG=OUT/'figs'


def extract():
    meta=pd.read_csv(OUT/'csi_pca_coverage.csv')
    with np.load(OUT/'csi_pca_snapshots.npz') as d:
        X=d['snapshots'].astype(float)
    with np.load(OUT/'csi_pca_projections.npz') as d:
        Z=d['projections']
    assert len(meta)==len(X)==len(Z)
    rows=[]
    for i,row in meta.iterrows():
        x,z=X[i],Z[i]
        valid=np.isfinite(x).all(axis=1)
        edges=valid[:-1]&valid[1:]
        record=dict(split=row['split'],minute=row['minute'],activity=row['activity'],label=row['label'],
                    valid_seconds=int(valid.sum()),valid_edges=int(edges.sum()))
        if valid.sum()>=5 and edges.sum()>=3:
            amp=x[:,:52]
            std=x[:,52:]
            # Two versions of motion: fixed PCA and gain-normalized CSI shape.
            level=np.nanmean(amp,axis=1)
            shape=amp/np.maximum(level[:,None],1.)
            relative_std=std/np.maximum(amp,1.)
            series={
                'pc12_step':np.linalg.norm(np.diff(z[:,:2],axis=0)[edges],axis=1),
                'pc3_step':np.abs(np.diff(z[:,2])[edges]),
                'shape_step':np.linalg.norm(np.diff(shape,axis=0)[edges],axis=1),
                'gain_step':np.abs(np.diff(np.log(np.maximum(level,1.)))[edges]),
                'relative_std':np.median(relative_std[valid],axis=1),
                'relative_std_step':np.linalg.norm(np.diff(relative_std,axis=0)[edges],axis=1),
            }
            for name,values in series.items():
                record[name+'_median']=float(np.median(values))
                record[name+'_p90']=float(np.quantile(values,.9))
        rows.append(record)
    df=pd.DataFrame(rows)
    radar=pd.read_csv(OUT/'snr_minutes.csv')[['split','minute','snr_db','snr_mean_db']]
    df=df.merge(radar,on=['split','minute'],how='inner',validate='one_to_one')
    df=df[df.label.isin([0,1])&df.activity.isin(['empty','sleep','present'])].copy()
    df['snr_peak_minus_mean']=df.snr_db-df.snr_mean_db
    df=df.sort_values(['split','minute']).reset_index(drop=True)
    df.to_csv(OUT/'improved_fusion_features.csv',index=False)
    return df


def weights(df,policy):
    """Equal binary class mass, then equal source/activity groups per class."""
    if policy=='target_train':
        mask=df.split=='train'
    else:
        mask=df.split.isin(['calibration_minutes','train'])
    selected=df[mask]
    w=np.zeros(len(selected))
    for label in [0,1]:
        groups=selected[selected.label==label].groupby(['split','activity']).groups
        for idx in groups.values():
            positions=np.flatnonzero(selected.index.isin(idx))
            w[positions]=1/(2*len(groups)*len(positions))
    return mask.to_numpy(),w*len(w)


def smooth(scores,df,length):
    """Causal mean; reset at split boundaries or >90-second gaps."""
    if length==1:
        return np.array(scores)
    times=pd.to_datetime(df.minute,format='%Y%m%d_%H%M')
    out=np.empty(len(df))
    history=[]
    for i,(_,row) in enumerate(df.iterrows()):
        if i==0 or row['split']!=df.iloc[i-1]['split'] or (times.iloc[i]-times.iloc[i-1]).total_seconds()>90:
            history=[]
        history.append(float(scores[i]))
        history=history[-length:]
        out[i]=np.mean(history)
    return out


def metric_table(model_name,split,y,p,threshold):
    return dict(model=model_name,split=split,threshold=float(threshold),**measure(y,p,p>=threshold))


def main():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        df=extract()
    radar=['snr_db','snr_mean_db','snr_peak_minus_mean']
    motion=['pc12_step_median','pc12_step_p90','pc3_step_median','pc3_step_p90']
    relative=[c for c in df if c.startswith(('shape_','gain_','relative_std'))]
    feature_sets={'motion':radar+motion,'relative':radar+relative,'combined':radar+motion+relative}
    # Log scale CSI magnitudes only; fitted imputers/scalers remain train-only.
    feature_df=df.copy()
    for col in motion+relative:
        feature_df[col]=np.log10(feature_df[col].clip(lower=1e-9))
    val=(df.split=='validation_minutes').to_numpy()
    validation=df[val].reset_index(drop=True)
    yval=validation.label.to_numpy()
    # Fixed 15-minute groups; thresholds trained on other validation groups.
    times=pd.to_datetime(validation.minute,format='%Y%m%d_%H%M')
    groups=((times-times.min()).dt.total_seconds()//900).astype(int)
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(validation,yval,groups))
    assert all(len(np.unique(yval[a]))==2 and len(np.unique(yval[b]))==2 for a,b in folds)
    models={
        'logistic':lambda:make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=.3,max_iter=2000,random_state=42)),
        'boost_small':lambda:HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=42),
        'boost_medium':lambda:HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=42),
        'forest':lambda:make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestClassifier(n_estimators=150,max_depth=6,min_samples_leaf=10,max_features=1.,n_jobs=2,random_state=42)),
    }
    results=[]
    best=None
    for policy in ['group_balanced','target_train']:
        train,w=weights(df,policy)
        for feature_name,cols in feature_sets.items():
            X=feature_df[cols].to_numpy()
            for model_name,factory in models.items():
                model=factory()
                kwargs={'sample_weight':w}
                if hasattr(model,'steps'):
                    kwargs={model.steps[-1][0]+'__sample_weight':w}
                model.fit(X[train],df.loc[train,'label'],**kwargs)
                # Only validation predictions are accessed during selection.
                raw=model.predict_proba(X[val])[:,1]
                for length in [1,3]:
                    p=smooth(raw,validation,length)
                    oof=np.zeros(len(p),dtype=bool)
                    for fit_idx,hold_idx in folds:
                        t=choose_threshold(yval[fit_idx],p[fit_idx])
                        oof[hold_idx]=p[hold_idx]>=t
                    ba=balanced_accuracy_score(yval,oof)
                    spec=np.mean(~oof[yval==0])
                    name=f'{policy}_{feature_name}_{model_name}_causal{length}'
                    results.append(dict(candidate=name,validation_oof_balanced_accuracy=ba,
                                        validation_auc=roc_auc_score(yval,p),validation_oof_specificity=spec,
                                        training_minutes=int(train.sum()),features=len(cols)))
                    key=(ba,spec,-length,-len(cols))
                    if best is None or key>best['key']:
                        best=dict(key=key,name=name,model=model,cols=cols,length=length,policy=policy,
                                  threshold=choose_threshold(yval,p))
                print('Evaluated',policy,feature_name,model_name,flush=True)
    search=pd.DataFrame(results).sort_values('validation_oof_balanced_accuracy',ascending=False)
    search.to_csv(OUT/'improved_fusion_validation_search.csv',index=False)
    # Save the frozen selection before accessing test labels for metrics.
    config={k:v for k,v in best.items() if k not in ['key','model']}
    config['selection']='3-fold grouped validation out-of-fold threshold balanced accuracy'
    (OUT/'improved_fusion_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    with open(OUT/'improved_fusion_model.pkl','wb') as fh:
        pickle.dump(best['model'],fh)
    raw=best['model'].predict_proba(feature_df[best['cols']].to_numpy())[:,1]
    score=smooth(raw,df,best['length'])
    prediction=df[['split','minute','activity','label','snr_db','valid_edges']].copy()
    prediction['raw_probability']=raw
    prediction['score']=score
    prediction['prediction']=(score>=best['threshold']).astype(int)
    prediction.to_csv(OUT/'improved_fusion_predictions.csv',index=False)
    metrics=[]
    for split in ['validation_minutes','test_minutes']:
        m=(df.split==split).to_numpy()
        metrics.append(metric_table('improved_selected',split,df.loc[m,'label'],score[m],best['threshold']))
    # Compare with saved previous methods on identical exact minute keys.
    old=pd.read_csv(OUT/'csi_radar_fusion_all_predictions.csv')
    paired=prediction.merge(old[['split','minute','radar_pred','fusion_pred','preserve_radar_pred']],on=['split','minute'],how='inner',validate='one_to_one')
    assert len(paired)==len(prediction)
    for split in ['validation_minutes','test_minutes']:
        g=paired[paired.split==split]
        for name,col in [('radar','radar_pred'),('previous_selected','fusion_pred'),('previous_preserve_radar','preserve_radar_pred')]:
            entry=metric_table(name,split,g.label,g[col],.5)
            entry['auc']=np.nan
            metrics.append(entry)
    metrics=pd.DataFrame(metrics)
    metrics.to_csv(OUT/'improved_fusion_metrics.csv',index=False)
    coverage=df.groupby(['split','activity']).agg(minutes=('minute','size'),csi_usable=('pc12_step_median','count'))
    coverage.to_csv(OUT/'improved_fusion_coverage.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,split in zip(axes,['validation_minutes','test_minutes']):
        g=prediction[prediction.split==split]
        for act,color in [('empty','#252525'),('sleep','#cca600'),('present','#d62728')]:
            h=g[g.activity==act]
            ax.scatter(pd.to_datetime(h.minute,format='%Y%m%d_%H%M'),h.score,c=color,s=15,label=act,alpha=.7)
        ax.axhline(best['threshold'],c='#2468a0',ls='--',label='Frozen threshold')
        ax.set(title=split,ylabel='Occupancy score',ylim=(-.02,1.02))
        ax.tick_params(axis='x',rotation=25)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Selected improved fusion | chronological scores')
    fig.tight_layout()
    fig.savefig(FIG/'improved_fusion_scores.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8))
    test=metrics[metrics.split=='test_minutes']
    ax.bar(test.model,test.balanced_accuracy*100,color=['#287b8e','#777','#c59544','#b57843'])
    for i,(_,r) in enumerate(test.iterrows()):
        ax.text(i,r.balanced_accuracy*100+1,f'{r.balanced_accuracy:.1%}\nFN={r.fn}, FP={r.fp}',ha='center',fontsize=9)
    ax.set(ylim=(0,100),ylabel='Balanced accuracy (%)',title='Same test minutes | frozen validation choices')
    ax.tick_params(axis='x',rotation=12)
    fig.tight_layout()
    fig.savefig(FIG/'improved_fusion_comparison.png',dpi=160)
    plt.close(fig)
    report='''# Improved radar / CSI fusion

## Changes and protocol

Use valid consecutive one-second CSI summary pairs instead of requiring a
10-second uninterrupted path. Require at least 5 valid seconds and 3 adjacent
pairs; never bridge missing seconds. Extract median and p90 of PCA PC1/PC2
step magnitude, PC3 step magnitude, amplitude-shape change after dividing by
mean subcarrier amplitude, relative temporal std, relative-std change, and
log-amplitude gain change. Add radar mean SNR and peak-minus-mean SNR.
No packet counts, timestamps, class labels, placement IDs, or split IDs are
used as classifier inputs. Median-imputation models include missing indicators;
boosting handles NaNs directly. This can still learn missingness artifacts.

Log-transform positive CSI features. Train-only fitted imputation/scaling.
Compare group-balanced pooled training with train-directory-only fitting;
the latter tests whether sleep-only calibration harms transfer. Pooled weights
give equal mass to binary classes and then to source/activity groups within
each class. The PCA basis remains the preceding train-only fitted basis.

Bounded search: 2 training policies x 3 feature sets x 4 fixed classifiers
x 2 smoothing settings = 48 candidates. Classifiers: regularized logistic,
two small histogram gradient-boosting configurations, and a constrained random
forest. Validation uses 15-minute temporal groups and three stratified group
folds: fit thresholds on two folds, score held-fold predictions. Select by
out-of-fold balanced accuracy, then specificity and simpler processing.
After selection, fit one threshold on all validation and freeze the selection
before evaluating the test. Only the selected new candidate is evaluated on
test. Causal smoothing candidates: none or mean of current and two previous
minute scores, resetting at split boundaries and gaps >90 seconds. Smoothing
never looks ahead or uses labels, but can delay transitions by up to two minutes.

These sets have already been explored in preceding experiments; grouped
validation reduces threshold-search optimism but does not make validation or
test independent fresh data. Serial dependence, class/placement confounding,
and test labels assigned by a time boundary remain limitations. No deployment
model is replaced. The saved model requires this feature extraction, log
transforms, and optional causal state; it is not a drop-in deployment artifact.

## Frozen choice

'''
    report+='```json\n'+json.dumps(config,indent=2)+'\n```\n\n'
    report+='## Metrics\n\n```text\n'+metrics.to_string(index=False,float_format=lambda x:f'{x:.3f}')+'\n```\n\n'
    report+='## Coverage\n\n```text\n'+coverage.to_string()+'\n```\n\n'
    low=prediction[(prediction.split=='test_minutes')&(prediction.snr_db<=6)]
    low_metrics=measure(low.label,low.score,low.prediction)
    report+='## Low-SNR diagnostic\n\n'
    report+=f"At SNR <=6 dB: {len(low)} test minutes, AUC {low_metrics['auc']:.3f}, balanced accuracy {low_metrics['balanced_accuracy']:.3f}, {low_metrics['tp']}/{low_metrics['occupied']} occupied minutes detected, and {low_metrics['fp']}/{low_metrics['empty']} empty minutes falsely marked occupied. This subset does not identify radar field-of-view status.\n\n"
    report+='## Top validation candidates\n\n```text\n'+search.head(10).to_string(index=False)+'\n```\n\n'
    report+='Reproduce: `python E2/improve_fusion.py`. All files have the improved_fusion prefix. Source snapshots and previous models remain unchanged.\n'
    (OUT/'IMPROVED_FUSION_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(config,indent=2),flush=True)
    print(metrics.to_string(index=False),flush=True)
    print(coverage.to_string(),flush=True)


if __name__=='__main__':
    main()
