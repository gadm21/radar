"""Radar/CSI processing sweep with validation-only selection.

python E2/fusion_processing_sweep.py [--reuse-features]
Saved earlier experiments and deploy/ are unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as C
from improve_fusion import weights
from csi_radar_fusion import measure

OUT=C.OUTPUT_DIR
FIG=OUT/'figs'


def contiguous(valid):
    edges=np.diff(np.r_[False,valid,False].astype(int))
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))


def filter_csi(z,method):
    """Filter within valid runs; no crossing missing seconds/minutes."""
    out=np.full_like(z,np.nan)
    for start,end in contiguous(np.isfinite(z).all(axis=1)):
        x=z[start:end]
        if method=='raw':
            out[start:end]=x
            continue
        for j in range(len(x)):
            if method=='median3':
                out[start+j]=np.median(x[max(0,j-2):j+1],axis=0)
            elif method=='ema05':
                out[start+j]=x[j] if j==0 else .5*x[j]+.5*out[start+j-1]
            elif method=='hampel5':
                window=x[max(0,j-4):j+1]
                med=np.median(window,axis=0)
                mad=np.median(np.abs(window-med),axis=0)
                reject=(j>=4)&(np.abs(x[j]-med)>3*1.4826*np.maximum(mad,1e-6))
                out[start+j]=np.where(reject,med,x[j])
    return out


def csi_features(meta):
    with np.load(OUT/'csi_pca_snapshots.npz') as d:
        x=d['snapshots'].astype(float)
    with np.load(OUT/'csi_pca_projections.npz') as d:
        original=d['projections']
    valid=np.isfinite(x).all(axis=2)
    train=(meta.split=='train').to_numpy()
    gain=np.maximum(np.mean(x[:,:,:52],axis=2,keepdims=True),1.)
    normalized=x/gain
    bases={'original':original}
    parameters={}
    for name,data in [('target',x),('gain_normalized',normalized)]:
        pca=PCA(n_components=3,svd_solver='full').fit(data[train][valid[train]])
        z=np.full_like(original,np.nan)
        z[valid]=pca.transform(data[valid])
        bases[name]=z
        parameters[name+'_mean']=pca.mean_
        parameters[name+'_components']=pca.components_
        parameters[name+'_explained_variance_ratio']=pca.explained_variance_ratio_
    np.savez_compressed(OUT/'fusion_sweep_pca.npz',**parameters)
    recipes=[('original','raw'),('original','median3'),('original','ema05'),
             ('original','hampel5'),('target','raw'),('target','median3'),
             ('gain_normalized','ema05')]
    columns={}
    groups={}
    for basis,method in recipes:
        key=f'{basis}_{method}'
        values=[]
        for z in bases[basis]:
            z=filter_csi(z,method)
            ok=np.isfinite(z).all(axis=1)
            pair=ok[:-1]&ok[1:]
            if ok.sum()<5 or pair.sum()<3:
                values.append([np.nan]*4)
                continue
            delta=np.diff(z,axis=0)[pair]
            p12=np.linalg.norm(delta[:,:2],axis=1)
            p3=np.abs(delta[:,2])
            values.append([np.median(p12),np.quantile(p12,.9),np.median(p3),np.quantile(p3,.9)])
        names=[key+'_'+suffix for suffix in ['pc12_median','pc12_p90','pc3_median','pc3_p90']]
        groups[key]=names
        for col,v in zip(names,np.asarray(values).T):
            columns[col]=v
        print('CSI recipe:',key,flush=True)
    return pd.DataFrame(columns),groups


def radar_one(row):
    key=dict(split=row.split,minute=row.minute)
    path=C.CACHE_DIR/row.split/(row.minute+'.npz')
    try:
        with np.load(path) as d:
            radar=d['radar'].astype(np.float32)
        if not np.isfinite(radar).all():
            return dict(**key,radar_status='nonfinite')
        rd=radar[:,:,0]
        # Cached maps are log1p amplitude, not phase or calibrated power.
        mask=np.ones(rd.shape[-2:],dtype=bool)
        mask[10:14]=False
        mask[:,:2]=False
        amp=np.expm1(rd).reshape(-1,*rd.shape[-2:])[:,mask]
        floor=np.maximum(np.median(amp,axis=1),1e-9)
        robust_snr=10*np.log10(np.maximum(np.quantile(amp,.99,axis=1),1e-9)/floor)
        peak_snr=10*np.log10(np.maximum(amp.max(axis=1),1e-9)/floor)
        key['radar_status']='ok'
        for name,series in [('rad_robust_snr',robust_snr),('rad_peak_snr',peak_snr)]:
            for q in [.5,.9,.99]:
                key[name+f'_q{int(q*100)}']=float(np.quantile(series,q))
        for filt in ['raw','median3']:
            data=radar if filt=='raw' else median_filter(radar,size=(1,3,1,1,1),mode='nearest')
            for ch,channel in [(0,'rd'),(1,'ra')]:
                a=data[:,:,ch]
                if ch==0:
                    a=a[:,:,mask]
                else:
                    a=a.reshape(a.shape[0],a.shape[1],-1)
                # Temporal clutter removal is confined to one cached window.
                centered=a-np.median(a,axis=1,keepdims=True)
                std_map=np.std(centered,axis=1)
                std_energy=np.quantile(std_map,.9,axis=1)
                step=np.median(np.abs(np.diff(a,axis=1)),axis=(1,2))
                for statistic,series in [('std',std_energy),('step',step)]:
                    for q in [.5,.9]:
                        key[f'rad_{filt}_{channel}_{statistic}_q{int(q*100)}']=float(np.quantile(series,q))
        return key
    except Exception as e:
        return dict(**key,radar_status='error',radar_error=str(e))


def extract(reuse):
    configpath=OUT/'fusion_sweep_feature_groups.json'
    if reuse:
        return pd.read_csv(OUT/'fusion_sweep_features.csv'),json.loads(configpath.read_text())
    meta=pd.read_csv(OUT/'csi_pca_coverage.csv')
    csi,groups=csi_features(meta)
    df=pd.concat([meta[['split','minute','activity','label']],csi],axis=1)
    existing=pd.read_csv(OUT/'improved_fusion_features.csv')[['split','minute','snr_db','snr_mean_db','snr_peak_minus_mean']]
    df=df.merge(existing,on=['split','minute'],how='inner',validate='one_to_one')
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(radar_one,df.itertuples(index=False))):
            rows.append(row)
            if (i+1)%100==0:
                print('Radar cache:',i+1,'/',len(df),flush=True)
    df=df.merge(pd.DataFrame(rows),on=['split','minute'],how='left',validate='one_to_one')
    old=['snr_db','snr_mean_db','snr_peak_minus_mean']
    robust=[c for c in df if c.startswith(('rad_robust','rad_peak'))]
    temporal_raw=[c for c in df if c.startswith('rad_raw_')]
    temporal_median=[c for c in df if c.startswith('rad_median3_')]
    config=dict(csi=groups,radar={'snr':old,'robust_snr':old+robust,
                                 'clutter_motion':old+temporal_raw,
                                 'median_motion':old+temporal_median})
    df=df.sort_values(['split','minute']).reset_index(drop=True)
    df.to_csv(OUT/'fusion_sweep_features.csv',index=False)
    configpath.write_text(json.dumps(config,indent=2),encoding='utf-8')
    return df,config


def temporal(score,starts,method):
    out=np.empty(len(score))
    history=[]
    for i,p in enumerate(score):
        if starts[i]:
            history=[]
        history.append(float(p))
        history=history[-5:]
        if method=='none': out[i]=p
        elif method=='mean3': out[i]=np.mean(history[-3:])
        elif method=='median3': out[i]=np.median(history[-3:])
        elif method=='mean5': out[i]=np.mean(history)
        elif method=='ema05': out[i]=p if starts[i] else .5*p+.5*out[i-1]
    return out


def resets(df):
    t=pd.to_datetime(df.minute,format='%Y%m%d_%H%M')
    start=(df.split!=df.split.shift()) | (t.diff().dt.total_seconds()>90)
    return start.to_numpy()


def threshold(y,p):
    """Exact fast threshold search; BA, specificity, middle-of-tie."""
    u=np.unique(p)
    ts=np.r_[-np.inf,(u[:-1]+u[1:])/2,np.inf]
    pred=p[:,None]>=ts
    tpr=pred[y==1].mean(axis=0)
    fpr=pred[y==0].mean(axis=0)
    ba=(tpr+1-fpr)/2
    good=np.isclose(ba,ba.max(),atol=1e-12,rtol=0)
    good &= np.isclose(fpr,np.min(fpr[good]),atol=1e-12,rtol=0)
    ids=np.flatnonzero(good)
    return float(ts[ids[len(ids)//2]])


def train_weights(df,policy):
    if policy in ['target_train','group_balanced']:
        return weights(df,policy)
    mask=(df.split=='train').to_numpy()
    y=df.loc[mask,'label'].to_numpy()
    return mask,np.array([len(y)/(2*(y==v).sum()) for v in y])


def search(df,groups):
    featurecols=set(sum(groups['csi'].values(),[])+sum(groups['radar'].values(),[]))
    data=df.copy()
    logcols=[c for c in featurecols if not ('snr' in c)]
    data[logcols]=np.log10(data[logcols].clip(lower=1e-9))
    val=(df.split=='validation_minutes').to_numpy()
    vdf=df[val].reset_index(drop=True)
    y=df.label.to_numpy()
    yv=y[val]
    start=resets(vdf)
    time=pd.to_datetime(vdf.minute,format='%Y%m%d_%H%M')
    blocks=((time-time.min()).dt.total_seconds()//900).astype(int)
    folds=[]
    for seed in [42,17]:
        folds.append(list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed).split(vdf,yv,blocks)))
    records=[]
    best=None
    count=0
    for policy in ['target_train','target_binary']:
        train,w=train_weights(df,policy)
        for csi,ccols in groups['csi'].items():
            for radar,rcols in groups['radar'].items():
                cols=rcols+ccols
                X=data[cols].to_numpy()
                for regularization in [.03,.3,3.]:
                    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),
                                        LogisticRegression(C=regularization,max_iter=3000,random_state=42))
                    model.fit(X[train],y[train],logisticregression__sample_weight=w)
                    p=model.predict_proba(X[val])[:,1]
                    for post in ['none','mean3','median3','mean5','ema05']:
                        score=temporal(p,start,post)
                        ba=[]
                        spec=[]
                        for partition in folds:
                            pred=np.zeros(len(yv),dtype=bool)
                            for a,b in partition:
                                cut=threshold(yv[a],score[a])
                                pred[b]=score[b]>=cut
                            ba.append(balanced_accuracy_score(yv,pred))
                            spec.append((~pred[yv==0]).mean())
                        name=f'{policy}__{csi}__{radar}__C{regularization}__{post}'
                        record=dict(candidate=name,policy=policy,csi=csi,radar=radar,C=regularization,post=post,
                                    validation_group_ba=float(np.mean(ba)),validation_worst_partition_ba=min(ba),
                                    validation_specificity=float(np.mean(spec)),validation_auc=roc_auc_score(yv,score))
                        records.append(record)
                        delay={'none':0,'mean3':2,'median3':2,'mean5':4,'ema05':2}[post]
                        rank=(record['validation_group_ba'],record['validation_worst_partition_ba'],
                              record['validation_specificity'],-delay,-len(cols),-regularization)
                        if best is None or rank>best['rank']:
                            best=dict(rank=rank,model=model,cols=cols,logcols=[c for c in cols if c in logcols],
                                      record=record,threshold=threshold(yv,score),post=post)
                    count+=1
                if count%12==0:
                    print('Fitted',count,'models;',len(records),'configurations',flush=True)
    table=pd.DataFrame(records).sort_values('validation_group_ba',ascending=False)
    table.to_csv(OUT/'fusion_sweep_validation.csv',index=False)
    config={k:v for k,v in best.items() if k not in ['rank','model']}
    config['candidate_count']=len(records)
    config['test_used_for_selection']=False
    (OUT/'fusion_sweep_selected.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    with open(OUT/'fusion_sweep_model.pkl','wb') as fh:
        pickle.dump(best['model'],fh)
    # The selected config is now persisted; evaluate only this new selection.
    raw=best['model'].predict_proba(data[best['cols']].to_numpy())[:,1]
    scores=temporal(raw,resets(df),best['post'])
    pred=df[['split','minute','activity','label']].copy()
    pred['raw_score']=raw
    pred['score']=scores
    pred['prediction']=(scores>=best['threshold']).astype(int)
    pred.to_csv(OUT/'fusion_sweep_predictions.csv',index=False)
    results=[]
    for split in ['validation_minutes','test_minutes']:
        mask=(df.split==split).to_numpy()
        results.append(dict(model='processing_sweep_selected',split=split,
                            **measure(y[mask],scores[mask],scores[mask]>=best['threshold'])))
    previous=pd.read_csv(OUT/'improved_fusion_predictions.csv')
    merged=pred.merge(previous[['split','minute','score','prediction']],on=['split','minute'],suffixes=('','_previous'),validate='one_to_one')
    assert len(merged)==len(pred)
    for split in ['validation_minutes','test_minutes']:
        g=merged[merged.split==split]
        results.append(dict(model='previous_77_6',split=split,**measure(g.label,g.score_previous,g.prediction_previous)))
    metrics=pd.DataFrame(results)
    metrics.to_csv(OUT/'fusion_sweep_metrics.csv',index=False)
    report(df,config,table,metrics,pred)
    print(json.dumps(config,indent=2),flush=True)
    print(metrics.to_string(index=False),flush=True)


def report(df,config,table,metrics,pred):
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,split in zip(axes,['validation_minutes','test_minutes']):
        g=pred[pred.split==split]
        for act,color in [('empty','#252525'),('sleep','#cca600'),('present','#d62728')]:
            h=g[g.activity==act]
            if len(h):
                ax.scatter(pd.to_datetime(h.minute,format='%Y%m%d_%H%M'),h.score,c=color,s=14,label=act,alpha=.7)
        ax.axhline(config['threshold'],ls='--',c='#2378a3',label='Validation threshold')
        ax.set(title=split,ylabel='Occupancy score',ylim=(-.02,1.02))
        ax.tick_params(axis='x',rotation=25)
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle('Radar / CSI processing sweep — frozen selection')
    fig.tight_layout()
    fig.savefig(FIG/'fusion_sweep_scores.png',dpi=160)
    plt.close(fig)
    text='''# Radar and CSI processing/configuration sweep

Target: >80% balanced accuracy on the same 249 test minutes. This is an
exploratory benchmark: validation and test were examined in earlier turns.
No test labels are used for selection in this run, but repeated dataset reuse
means the result is not an independent generalization estimate.

## Processing

CSI input is the existing 1 Hz mean/std snapshot cache from raw CSI. Compare
original pooled-training PCA and a new PCA fitted on train-directory snapshots
only; also fit a gain-normalized PCA after dividing all 104 mean/std features
by mean subcarrier amplitude per dot. All PCA fits exclude validation/test.
Filters on PCA coordinates: none, causal trailing median of 3, EMA alpha .5,
and trailing Hampel-5 with 3 * 1.4826 MAD cutoff. Reset across missing seconds
and minute boundaries. Extract median/p90 adjacent PC1/PC2 vector distance and
absolute PC3 change; require 5 valid seconds and 3 valid adjacent pairs.

Radar input is the existing compressed amplitude-map cache, not raw complex
ADC. Compare existing peak/mean SNR with robust SNR proxies (99th-percentile
or maximum masked amplitude / masked median, 10 log10, summarized by minute
quantiles 50/90/99). This preserves the existing SNR proxy convention, not a
calibrated physical power SNR. Mask Doppler rows 10:14 and first two range bins.
Additional variants remove per-pixel median background inside each 50-frame
window and summarize temporal std and adjacent-frame absolute differences in
range-Doppler and range-azimuth. Temporal median-3 map filtering is confined
to each window; centered filtering is available by the completed-minute output.
No filter connects different cached windows. Cache timestamps do not allow a
full raw packet/frame-gap audit, and FFT resolution is limited by the cache.

Classifier: train-only imputer/scaler and regularized logistic regression,
C .03/.3/3. Compare activity-balanced and binary-balanced weighting of the
main train directory. Prior work found pooled sleep-only calibration hurt
transfer; original PCA still uses its previous pooled training basis.
Postprocessing: none, causal mean-3, median-3, mean-5, or EMA alpha .5.
Reset minute state across splits or gaps >90 s. Mean-5 can delay transitions
up to four minutes; EMA has a decaying tail without a strict maximum delay.
No timestamps, split IDs, or labels enter model features.

## Selection

840 configurations (7 CSI recipes x 4 radar recipes x 3 regularization settings
x 2 weighting policies x 5 temporal filters). Rank by mean held-fold balanced
accuracy across two fixed stratified 3-fold partitions of 15-minute validation
groups. Each fold's threshold is fitted on the other two folds. Tie breakers:
worst partition, specificity, shorter nominal history, fewer features, stronger
regularization. Large searches can overfit this small validation set despite
grouping. Refit threshold on all validation, save the selected config, then
evaluate only that new candidate on test. Previous results are compared on
identical minute keys. Missing sensor features remain NaN and are imputed
using training medians with indicators; they are never zero-filled.

## Selected configuration

'''
    text+='```json\n'+json.dumps(config,indent=2)+'\n```\n\n'
    text+='## Metrics\n\n```text\n'+metrics.to_string(index=False,float_format=lambda x:f'{x:.3f}')+'\n```\n\n'
    text+='## Top validation configurations\n\n```text\n'+table.head(15).to_string(index=False)+'\n```\n\n'
    text+='## Radar extraction status\n\n```text\n'+df.groupby(['split','radar_status']).size().to_string()+'\n```\n\n'
    text+='Run `python E2/fusion_processing_sweep.py`; use `--reuse-features` to reuse extracted features. Artifacts use fusion_sweep prefixes. The saved classifier also requires saved feature processing and temporal state. No deployment artifacts are changed.\n'
    (OUT/'FUSION_PROCESSING_SWEEP_REPORT.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--reuse-features',action='store_true')
    a=ap.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        data,groups=extract(a.reuse_features)
    search(data,groups)
