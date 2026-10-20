"""Reproducible, isolated experiment for the requested 20-second 2D CSI feature."""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.paper_packages'))
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import common as C
OUT = ROOT / 'E2/outputs/paper_multimodal_20s'
OUT.mkdir(parents=True, exist_ok=True)

def extract(rec):
    """Same one-second eligibility and mean/std calculation as csi_pca_path.extract."""
    row=dict(split=rec.source,minute=rec.folder.name,placement=rec.placement,label=rec.label,activity='',status='ok',valid_seconds=0)
    acts={s.rsplit('_',1)[-1] for s in rec.orig_labels if s in C.LABEL_TABLES[rec.source]}
    dots=np.full((60,104),np.nan,dtype=np.float32);counts=np.zeros(60,dtype=np.int32)
    if rec.status!='ok' or len(acts)!=1:
        row['status']='invalid_explicit_label';return row,dots,counts
    row['activity']=acts.pop()
    try:
        csi,ts=C.recording_csi(rec)
        if csi is None:
            row['status']='no_csi';return row,dots,counts
        rel=ts-rec.start_ts;keep=(rel>=0)&(rel<60)&np.isfinite(csi).all(axis=1)
        rel,amp=rel[keep],np.abs(csi[keep]).astype(np.float64)
        rel,unique=np.unique(rel,return_index=True);amp=amp[unique];bins=np.floor(rel).astype(int)
        for second in range(60):
            sel=bins==second;t,a=rel[sel],amp[sel];counts[second]=len(t)
            if len(t)<5:continue
            gaps=np.diff(np.r_[second,t,second+1])
            if t[-1]-t[0]<.5 or gaps.max()>.5:continue
            dots[second]=np.r_[a.mean(axis=0),a.std(axis=0)]
        valid=np.isfinite(dots).all(axis=1)
        row.update(valid_seconds=int(valid.sum()),packets=int(counts.sum()),median_packets_per_valid_second=float(np.median(counts[valid])) if valid.any() else np.nan,status='ok' if valid.any() else 'no_valid_seconds')
    except Exception as exc:row.update(status='read_error',error=str(exc))
    return row,dots,counts

def checkpoint_extract(rec):
    cache=OUT/'csi_records'/f'{rec.source}_{rec.folder.name}.npz'
    if cache.exists():
        try:
            with np.load(cache) as data:
                row=json.loads(str(data['row']))
                if row['status']!='read_error': return row,data['dots']
        except (OSError,ValueError,EOFError): pass
    # Vectorized numeric conversion preserves the original parser for unusual records.
    original=C.parse_csi_line
    def fast_parse(line):
        match=C._CSI_RE.search(line)
        if match is None: return None
        try: values=np.fromstring(match.group(1),sep=',',dtype=np.float64)
        except ValueError: return original(line)
        if len(values)!=128: return original(line)
        imag=values[0::2][C.CSI_SUBCARRIER_MASK]; real=values[1::2][C.CSI_SUBCARRIER_MASK]
        return (real+1j*imag).astype(np.complex64)
    C.parse_csi_line=fast_parse
    try: row,dots,counts=extract(rec)
    finally: C.parse_csi_line=original
    np.savez_compressed(cache,row=json.dumps(row),dots=dots)
    return row,dots

def csi_features(recs):
    import pandas as pd
    from sklearn.decomposition import PCA
    cached=pd.read_csv(OUT/'coverage.csv') if (OUT/'coverage.csv').exists() else None
    if (OUT/'snapshots.npz').exists() and cached is not None and not cached.status.eq('read_error').any():
        meta=cached; X=np.load(OUT/'snapshots.npz')['dots']
    else:
        rows=[]; dots=[]
        (OUT/'csi_records').mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,(row,x) in enumerate(pool.map(checkpoint_extract,recs)):
                rows.append(row); dots.append(x)
                if i%50==0: print('CSI',i,len(recs),flush=True)
        meta=pd.DataFrame(rows); X=np.array(dots)
        meta.to_csv(OUT/'coverage.csv',index=False)
        np.savez_compressed(OUT/'snapshots.npz',dots=X)
    valid=np.isfinite(X).all(axis=2)
    tr=meta.split.isin(['calibration_minutes','train']).to_numpy()
    pca=PCA(n_components=3,svd_solver='full',whiten=False).fit(X[tr][valid[tr]])
    Z=np.full((len(X),60,3),np.nan); Z[valid]=pca.transform(X[valid])
    meta['csi']=np.nan; meta['n_paths']=0
    for i in range(len(X)):
        paths=[np.linalg.norm(np.diff(Z[i,s:s+21,:2],axis=0),axis=1).sum() for s in range(40) if valid[i,s:s+21].all()]
        if paths: meta.loc[i,'csi']=np.median(paths); meta.loc[i,'n_paths']=len(paths)
    np.savez_compressed(OUT/'pca.npz',mean=pca.mean_,components=pca.components_,variance=pca.explained_variance_ratio_,fit_dots=valid[tr].sum())
    meta.to_csv(OUT/'csi_minutes.csv',index=False)
    return meta

def map_features(rec):
    """Power maps, range/angles in normalized sensor coordinates; no room calibration assumed."""
    from scipy import fft
    from preprocess import read_uint12_batch
    path=OUT/'radar_features'/f'{rec.source}_{rec.folder.name}.json'
    if path.exists(): return json.loads(path.read_text())
    result={'split':rec.source,'minute':rec.folder.name}
    try:
        payloads,ts=C.recording_radar_frames(rec)
        payloads=[p for p,t in zip(payloads,ts) if len(p)==36864 and 0<=t-rec.start_ts<60]
        if len(payloads)<20: return result
        theta=np.deg2rad(np.linspace(-60,60,25))
        phase=np.exp(1j*np.pi*np.sin(theta)).astype(np.complex64)
        radius=np.arange(64)/63
        ix=np.clip(((radius[:,None]*np.sin(theta)+1)/2*23).round().astype(int),0,23)
        iy=np.clip((radius[:,None]*np.cos(theta)*23).round().astype(int),0,23)
        maps={k:[] for k in ['rd','ra','re','xy']}
        for offset in range(0,len(payloads),16):
            batch=payloads[offset:offset+16]
            a=read_uint12_batch(b''.join(batch)).reshape(-1,64,128,3)
            a-=a.mean(axis=2,keepdims=True)
            a*=np.hanning(128)[None,None,:,None]
            R=fft.fft(a,axis=2)[:,:,:64,:]
            # Hann slow-time FFT for RD. Static energy is retained; only ADC offset removed.
            D=fft.fftshift(fft.fft(R*np.hanning(64)[None,:,None,None],axis=1),axes=1)
            rd=np.mean(np.abs(D)**2,axis=3)
            # Analytic two-element beam power: RX1/RX3 azimuth, RX2/RX3 elevation.
            for key,left in [('ra',0),('re',1)]:
                p=(np.abs(R[:,:,:,left])**2+np.abs(R[:,:,:,2])**2).mean(axis=1)
                cross=(R[:,:,:,left]*R[:,:,:,2].conj()).mean(axis=1)
                energy=np.maximum(p[:,:,None]+2*np.real(cross[:,:,None]*phase),0)
                energy[:,:2,:]=0
                maps[key].extend(np.log1p(energy).astype(np.float32))
                if key=='ra':
                    xy=np.zeros((len(batch),24,24),np.float32)
                    for j in range(len(batch)):
                        np.maximum.at(xy[j],(iy.ravel(),ix.ravel()),energy[j].ravel())
                    maps['xy'].extend(np.log1p(xy))
            rd[:,:,:2]=0
            maps['rd'].extend(np.log1p(rd).astype(np.float32))
        result['radar_frames']=len(payloads)
        for key,seq in maps.items():
            m=np.asarray(seq); mean=m.mean(axis=0); std=m.std(axis=0)
            result.update({f'{key}_mean':float(mean.mean()),f'{key}_p90':float(np.quantile(mean,.9)),f'{key}_std':float(np.quantile(std,.9)),f'{key}_delta':float(np.mean(np.abs(np.diff(m,axis=0)))),f'{key}_peak':float(mean.max())})
        # Save one representative map per recording for traceable plotting, no test selection.
        if rec.source=='train': np.savez_compressed(OUT/'radar_features'/f'{rec.source}_{rec.folder.name}_maps.npz',**{k:np.asarray(v).mean(axis=0) for k,v in maps.items()})
    except Exception as exc: result['error']=str(exc)
    path.write_text(json.dumps(result),encoding='utf8')
    return result

def threshold(y,p):
    from sklearn.metrics import balanced_accuracy_score
    candidates=np.unique(np.r_[0,.5,p,1.000001])
    scores=np.array([balanced_accuracy_score(y,p>=t) for t in candidates])
    best=np.flatnonzero(scores==scores.max())
    return float(candidates[best[np.argmin(abs(candidates[best]-.5))]])

def evaluate(df):
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, accuracy_score, f1_score
    groups={'CSI':['csi'],'SNR':['snr_db']}
    for key in ['rd','ra','re','xy']: groups[key.upper()]=[c for c in df if c.startswith(key+'_')]
    groups['CSI+SNR']=groups['CSI']+groups['SNR']
    for key in ['RD','RA','RE','XY']: groups['CSI+SNR+'+key]=groups['CSI+SNR']+groups[key]
    groups['Radar all']=sum([groups[k] for k in ['SNR','RD','RA','RE','XY']],[])
    groups['CSI+radar all']=groups['CSI']+groups['Radar all']
    # Primary ablations use exactly the same fully paired minutes.
    need=groups['CSI+radar all']
    paired=df[df.split.isin(['train','validation_minutes','test_minutes'])].dropna(subset=need).copy()
    paired.to_csv(OUT/'paired_minutes.csv',index=False)
    train=paired[paired.split=='train']; val=paired[paired.split=='validation_minutes']; test=paired[paired.split=='test_minutes']
    for name,part in [('train',train),('validation',val),('test',test)]:
        if part.label.nunique()!=2: raise ValueError(f'{name} matched cohort must contain both classes; inspect coverage before modeling')
    records=[]; predictions=[]; fitted={}
    for name,cols in groups.items():
        # Two prespecified lightweight classifiers; no hyperparameter sweep on test.
        for modelname,clf in [('LR',LogisticRegression(C=.3,class_weight='balanced',max_iter=3000,random_state=42)),('RF',RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=2))]:
            model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),clf)
            def features(frame):
                x=frame[cols].copy()
                if 'csi' in x: x['csi']=np.log1p(x.csi)
                return x
            model.fit(features(train),train.label.astype(int))
            pv=model.predict_proba(features(val))[:,1]; t=threshold(val.label,pv)
            fitted[(name,modelname)]=(model,cols,t)
            for split,data,p in [('validation_minutes',val,pv),('test_minutes',test,model.predict_proba(features(test))[:,1])]:
                y=data.label.astype(int); pred=p>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
                records.append(dict(features=name,model=modelname,split=split,n=len(y),empty=int((y==0).sum()),occupied=int((y==1).sum()),auc=roc_auc_score(y,p),balanced_accuracy=balanced_accuracy_score(y,pred),accuracy=accuracy_score(y,pred),f1=f1_score(y,pred),threshold=t,tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp)))
                for (_,r),prob,pr in zip(data.iterrows(),p,pred): predictions.append(dict(features=name,model=modelname,split=split,minute=r.minute,label=int(r.label),probability=prob,prediction=int(pr)))
    res=pd.DataFrame(records); res.to_csv(OUT/'results.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'predictions.csv',index=False)
    # Selection explicitly uses only validation balanced accuracy, then AUC, then lexical key.
    ranked=res[res.split=='validation_minutes'].sort_values(['balanced_accuracy','auc','features','model'],ascending=[False,False,True,True])
    win=ranked.iloc[0]
    (OUT/'selection.json').write_text(json.dumps(win.to_dict(),indent=2))
    import joblib
    selected_model,selected_columns,selected_threshold=fitted[(win['features'],win['model'])]
    joblib.dump(dict(model=selected_model,columns=selected_columns,threshold=selected_threshold,csi_transform='log1p'),OUT/'selected_model.joblib')
    raw=[]
    for split in ['validation_minutes','test_minutes']:
        d=df[(df.split==split)&df.csi.notna()]
        raw.append({'split':split,'n':len(d),'empty':int((d.label==0).sum()),'occupied':int((d.label==1).sum()),'auc':roc_auc_score(d.label,d.csi)})
    (OUT/'raw_csi_auc.json').write_text(json.dumps(raw,indent=2))
    print(res.to_string(index=False),flush=True)
    print('SELECTED ON VALIDATION',win.to_dict(),flush=True)

def plots(df):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paired=pd.read_csv(OUT/'paired_minutes.csv')
    fig,axs=plt.subplots(1,3,figsize=(11,3.2))
    for ax,split in zip(axs,['train','validation_minutes','test_minutes']):
        for act,col in [('empty','#444444'),('sleep','#c49a00'),('present','#c23838')]:
            d=paired[(paired.split==split)&(paired.activity==act)]
            ax.scatter(d.snr_db,d.csi,c=col,s=18,alpha=.7,label=f'{act} (n={len(d)})')
        ax.set(xlabel='Radar SNR proxy (dB)',ylabel='20 s, 2D CSI path length',yscale='log',title=split.replace('_minutes','')); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT/'csi_snr.png',dpi=200); plt.close(fig)
    result=pd.read_csv(OUT/'results.csv')
    chosen=result[result.split=='validation_minutes'].sort_values(['balanced_accuracy','auc','model'],ascending=[False,False,True]).drop_duplicates('features')
    fig,ax=plt.subplots(figsize=(10,4))
    labels=[]; va=[]; te=[]
    for _,r in chosen.iterrows():
        labels.append(r.features+' ('+r.model+')'); va.append(r.balanced_accuracy)
        te.append(result[(result.features==r.features)&(result.model==r.model)&(result.split=='test_minutes')].iloc[0].balanced_accuracy)
    x=np.arange(len(labels)); ax.bar(x-.18,va,.36,label='Validation'); ax.bar(x+.18,te,.36,label='Test')
    ax.set_xticks(x,labels,rotation=35,ha='right',fontsize=8);ax.set(ylabel='Balanced accuracy',ylim=(0,1));ax.axhline(.5,color='gray',ls=':');ax.legend();fig.tight_layout();fig.savefig(OUT/'ablation.png',dpi=200);plt.close(fig)
    representative=paired[(paired.split=='train')&(paired.activity=='present')].sort_values('minute').iloc[0]
    maps=np.load(OUT/'radar_features'/f'train_{representative.minute}_maps.npz')
    fig,axs=plt.subplots(1,4,figsize=(10,2.6))
    for ax,key in zip(axs,['rd','ra','re','xy']):
        ax.imshow(maps[key],origin='lower',aspect='auto');ax.set_title(key.upper());ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Minute-mean log-power maps: '+representative.minute,fontsize=10);fig.tight_layout();fig.savefig(OUT/'maps.png',dpi=200);plt.close(fig)

if __name__=='__main__':
    import pandas as pd
    recs,_=C.discover_recordings()
    meta=csi_features(recs)
    if meta.status.eq('read_error').mean()>.1: raise ValueError('Excessive CSI read errors: inspect parsing before modeling')
    (OUT/'radar_features').mkdir(exist_ok=True)
    # Restrict expensive map processing to the exact CSI-eligible matched cohort.
    keys=set(zip(meta.loc[meta.csi.notna(),'split'],meta.loc[meta.csi.notna(),'minute']))
    eligible=[r for r in recs if r.source!='calibration_minutes' and (r.source,r.folder.name) in keys]
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,r in enumerate(pool.map(map_features,eligible)):
            rows.append(r)
            if i%20==0: print('Radar maps',i,len(eligible),flush=True)
    radar=pd.DataFrame(rows); radar.to_csv(OUT/'radar_minutes.csv',index=False)
    snr=pd.read_csv(C.OUTPUT_DIR/'snr_minutes.csv')[['split','minute','snr_db']]
    df=meta.merge(radar,on=['split','minute'],how='left',validate='one_to_one').merge(snr,on=['split','minute'],how='left',validate='one_to_one')
    df.to_csv(OUT/'all_minutes.csv',index=False)
    evaluate(df); plots(df)
