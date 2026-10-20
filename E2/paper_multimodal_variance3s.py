"""Three one-second PCA dots: population variance trace, with no path distances."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.paper_packages'))
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import paper_multimodal_20s as base
OLD=base.OUT
OUT=ROOT/'E2/outputs/paper_multimodal_variance3s'

def compute():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(OLD/'coverage.csv')
    X=np.load(OLD/'snapshots.npz')['dots']
    pca=np.load(OLD/'pca.npz')
    valid=np.isfinite(X).all(axis=2)
    Z=np.full((len(X),60,2),np.nan)
    Z[valid]=(X[valid]-pca['mean'])@pca['components'][:2].T
    meta['csi']=np.nan;meta['n_windows_3s']=0;windows=[]
    for i,row in meta.iterrows():
        values=[]
        for start in range(58):
            pts=Z[i,start:start+3]
            if not np.isfinite(pts).all():continue
            coordinate_variance=np.var(pts,axis=0,ddof=0)
            value=float(coordinate_variance.sum())
            values.append(value)
            windows.append(dict(split=row['split'],minute=row['minute'],start_second=start,variance_pc1=float(coordinate_variance[0]),variance_pc2=float(coordinate_variance[1]),variance_trace=value))
        if values:
            meta.loc[i,'csi']=np.median(values);meta.loc[i,'n_windows_3s']=len(values)
    meta.to_csv(OUT/'csi_minutes.csv',index=False)
    pd.DataFrame(windows).to_csv(OUT/'csi_windows_3s.csv',index=False)
    shutil.copy2(OLD/'pca.npz',OUT/'pca.npz')
    print('CSI coverage',meta[meta.csi.notna()].groupby(['split','activity']).size().to_string(),flush=True)
    # Sanity: coordinate variance equals mean squared radius around centroid.
    example=np.array([[0.,0.],[3.,4.],[6.,8.]])
    assert np.isclose(example.var(axis=0).sum(),50/3)
    assert np.isclose(example.var(axis=0).sum(),np.mean(np.sum((example-example.mean(axis=0))**2,axis=1)))
    if '--coverage-only' in sys.argv:return
    (OUT/'radar_features').mkdir(exist_ok=True)
    for f in (OLD/'radar_features').iterdir():
        dest=OUT/'radar_features'/f.name
        if not dest.exists():shutil.copy2(f,dest)
    base.OUT=OUT
    recs,_=base.C.discover_recordings()
    keys=set(zip(meta.loc[meta.csi.notna(),'split'],meta.loc[meta.csi.notna(),'minute']))
    eligible=[r for r in recs if r.source!='calibration_minutes' and (r.source,r.folder.name) in keys]
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(base.map_features,eligible)):
            rows.append(result)
            if i%25==0:print('Radar',i,'/',len(eligible),flush=True)
    radar=pd.DataFrame(rows);radar.to_csv(OUT/'radar_minutes.csv',index=False)
    snr=pd.read_csv(base.C.OUTPUT_DIR/'snr_minutes.csv')[['split','minute','snr_db']]
    data=meta.merge(radar,on=['split','minute'],how='left',validate='one_to_one').merge(snr,on=['split','minute'],how='left',validate='one_to_one')
    data.to_csv(OUT/'all_minutes.csv',index=False)
    base.evaluate(data)
    paired=pd.read_csv(OUT/'paired_minutes.csv')
    audit=[]
    for split in ['calibration_minutes','train','validation_minutes','test_minutes']:
        for activity in ['empty','sleep','present']:
            raw=meta[(meta.split==split)&(meta.activity==activity)]
            joined=paired[(paired.split==split)&(paired.activity==activity)]
            audit.append(dict(split=split,activity=activity,raw=len(raw),csi_qualified=int(raw.csi.notna().sum()),paired=len(joined)))
    pd.DataFrame(audit).to_csv(OUT/'coverage_audit.csv',index=False)
    (OUT/'method.json').write_text(json.dumps(dict(window_seconds=3,dots_per_window=3,hop_seconds=1,dimensions=2,variance_ddof=0,feature='sum of coordinate variances (covariance trace)',minute_aggregation='median of valid windows',pca='fixed calibration+train basis, centered, unwhitened',one_second_quality='unchanged: >=5 packets, >=0.5s span, max gap <=0.5s',decision_unit='minute',selection='validation only; test already explored historically'),indent=2))

if __name__=='__main__':compute()
