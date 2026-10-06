"""Run from radar's top level: WiFiSensing guaranteed 100 Hz -> PCA -> 3 s variance.

The upstream resampler is loaded verbatim, not approximated. Existing balanced
minute IDs and original signal-quality rules are preserved for comparison.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'.paper_packages'),str(ROOT/'E2')]
import numpy as np
import pandas as pd
import common as C

UPSTREAM=ROOT.parent/'WifiSensingESP32HAR/src/train/utils.py'
OUT=ROOT/'E2/outputs/paper_100hz_variance3s'
PREVIOUS=ROOT/'E2/outputs/paper_balanced_variance3s'

def load_resampler():
    source=UPSTREAM.read_text(encoding='utf-8-sig')
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CSI_Loader')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_resample_equal_intervals')
    namespace={'np':np}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(UPSTREAM),'exec'),namespace)
    return namespace[method.name],ast.get_source_segment(source,method)

RESAMPLE,METHOD_SOURCE=load_resampler()
CONTEXT=SimpleNamespace(guaranteed_sr=100,verbose=False,_log=lambda *args:None)

def resample(amplitude,times_seconds):
    # Input is relative host timing in seconds, converted to upstream microseconds.
    # Only magnitude is used downstream; other channels do not affect its result.
    zeros=np.zeros_like(amplitude)
    result=RESAMPLE(CONTEXT,amplitude,zeros,zeros,zeros,np.zeros(len(amplitude)),times_seconds*1e6)
    values,_,_,_,_,times_us,counts,stats=result
    return values,times_us/1e6,counts,stats

def verify_resampler():
    t=np.array([0.,.001,.02,.03])
    a=np.array([[1.,2.],[3.,4.],[8.,10.],[10.,12.]])
    values,times,counts,stats=resample(a,t)
    # Last endpoint is clipped into the final bin by the upstream algorithm.
    assert counts.tolist()==[2,0,2]
    np.testing.assert_allclose(values,[[2,3],[5.5,7],[9,11]])
    np.testing.assert_allclose(times,[0,.01,.02],atol=1e-6)
    assert stats['empty_slots']==1
    assert np.isclose(np.array([[0,0],[3,4],[6,8]]).var(axis=0).sum(),50/3)

def extract(rec):
    cache=OUT/'csi_records'/f'{rec.source}_{rec.folder.name}.npz'
    if cache.exists():
        with np.load(cache) as saved:
            row=json.loads(str(saved['row']))
            if row['status']!='read_error':return row,saved['dots']
    row=dict(split=rec.source,minute=rec.folder.name,placement=rec.placement,label=rec.label,activity='',status='ok',valid_seconds=0)
    dots=np.full((60,104),np.nan,dtype=np.float64)
    acts={s.rsplit('_',1)[-1] for s in rec.orig_labels if s in C.LABEL_TABLES[rec.source]}
    try:
        if rec.status!='ok' or len(acts)!=1:raise ValueError('Invalid explicit activity label')
        row['activity']=acts.pop()
        # Equivalent vectorized parsing of standard 128-component CSI payloads.
        original=C.parse_csi_line
        def fast_parse(line):
            match=C._CSI_RE.search(line)
            if match is None:return None
            try:values=np.fromstring(match.group(1),sep=',',dtype=np.float64)
            except ValueError:return original(line)
            if len(values)!=128:return original(line)
            return (values[1::2][C.CSI_SUBCARRIER_MASK]+1j*values[0::2][C.CSI_SUBCARRIER_MASK]).astype(np.complex64)
        C.parse_csi_line=fast_parse
        try:csi,ts=C.recording_csi(rec)
        finally:C.parse_csi_line=original
        if csi is None or len(csi)<2:raise ValueError('Insufficient CSI samples')
        rel=ts-rec.start_ts;keep=(rel>=0)&(rel<60)&np.isfinite(csi).all(axis=1)
        rel,amp=rel[keep],np.abs(csi[keep]).astype(np.float64)
        rel,unique=np.unique(rel,return_index=True);amp=amp[unique]
        if len(rel)<2:raise ValueError('Insufficient finite, distinct CSI timestamps')
        uniform,t,spb,stats=resample(amp,rel)
        # Timestamp integer conversion in upstream can introduce <=1 us rounding.
        if len(t)>1:assert np.max(np.abs(np.diff(t)-.01))<=1.01e-6
        raw_bins=np.floor(rel).astype(int);bins=np.floor(t).astype(int)
        used=[];raw_valid=0;sample_counts=[]
        for sec in range(60):
            observed=rel[raw_bins==sec]
            if len(observed)<5 or observed[-1]-observed[0]<.5 or np.diff(np.r_[sec,observed,sec+1]).max()>.5:continue
            raw_valid+=1
            idx=np.flatnonzero(bins==sec)
            # Summaries use exactly 100 resampled values; partial boundary seconds are omitted.
            if len(idx)!=100:continue
            a=uniform[idx]
            dots[sec]=np.r_[a.mean(axis=0),a.std(axis=0,ddof=0)]
            used.extend(idx.tolist());sample_counts.append(len(observed))
        used=np.asarray(used,dtype=int)
        row.update(valid_seconds=int(np.isfinite(dots).all(axis=1).sum()),raw_valid_seconds=raw_valid,packets=len(rel),median_packets_per_valid_second=float(np.median(sample_counts)) if sample_counts else np.nan,resampled_samples=len(t),interpolated_samples=int((spb==0).sum()),interpolated_fraction=float((spb==0).mean()),used_resampled_samples=len(used),used_interpolated_samples=int((spb[used]==0).sum()),original_rate_hz=stats['actual_sampling_rate'],max_raw_gap_seconds=float(np.diff(rel).max()),target_rate_hz=100,status='ok' if len(used) else 'no_valid_seconds')
    except Exception as exc:
        row.update(status='read_error',error=str(exc))
    np.savez_compressed(cache,row=json.dumps(row),dots=dots)
    return row,dots

def main():
    verify_resampler()
    if '--self-test' in sys.argv:
        print('Upstream bin averaging, gap interpolation, 100 Hz timing, and variance checks passed.');return
    OUT.mkdir(exist_ok=True);(OUT/'csi_records').mkdir(exist_ok=True)
    digest=hashlib.sha256(METHOD_SOURCE.encode()).hexdigest()
    if (OUT/'resampler_provenance.json').exists():
        prior=json.loads((OUT/'resampler_provenance.json').read_text())
        if prior['sha256']!=digest or prior['target_hz']!=100:
            raise RuntimeError('Resampler changed; use a fresh experiment output directory to avoid stale caches')
    (OUT/'resampler_source.py.txt').write_text(METHOD_SOURCE,encoding='utf8')
    (OUT/'resampler_provenance.json').write_text(json.dumps({'path':str(UPSTREAM),'method':'CSI_Loader._resample_equal_intervals','sha256':digest,'target_hz':100,'timestamp_source':'relative host timestamps; converted to microseconds','amplitude':'upstream magnitude bin averages and interpolation used directly'},indent=2))
    recs,_=C.discover_recordings()
    expected=pd.read_csv(PREVIOUS/'all_minutes.csv')
    keys=set(zip(expected.split,expected.minute))
    recs=[r for r in recs if (r.source,r.folder.name) in keys]
    assert len(recs)==len(expected)
    rows=[];dots=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,(row,x) in enumerate(pool.map(extract,recs)):
            rows.append(row);dots.append(x)
            if (i+1)%25==0:print(f'100 Hz CSI: {i+1}/{len(recs)} minutes',flush=True)
    meta=pd.DataFrame(rows);X=np.asarray(dots)
    meta.to_csv(OUT/'resampling_audit.csv',index=False)
    if meta.status.eq('read_error').any():raise RuntimeError('CSI read errors recorded; resolve and rerun cached extraction')
    from sklearn.decomposition import PCA
    valid=np.isfinite(X).all(axis=2);fit=meta.split.isin(['calibration_minutes','train']).to_numpy()
    pca=PCA(n_components=2,svd_solver='full',whiten=False).fit(X[fit][valid[fit]])
    Z=np.full((len(X),60,2),np.nan);Z[valid]=pca.transform(X[valid])
    np.savez_compressed(OUT/'snapshots.npz',dots=X,projections=Z)
    np.savez_compressed(OUT/'pca.npz',mean=pca.mean_,components=pca.components_,variance=pca.explained_variance_ratio_,fit_dots=int(valid[fit].sum()))
    meta['csi']=np.nan;meta['n_windows_3s']=0;windows=[]
    for i,row in meta.iterrows():
        values=[]
        for start in range(58):
            pts=Z[i,start:start+3]
            if not np.isfinite(pts).all():continue
            v=pts.var(axis=0,ddof=0);values.append(float(v.sum()))
            windows.append(dict(split=row['split'],minute=row['minute'],start_second=start,variance_pc1=v[0],variance_pc2=v[1],variance_trace=v.sum()))
        if values:meta.loc[i,'csi']=np.median(values);meta.loc[i,'n_windows_3s']=len(values)
    meta.to_csv(OUT/'csi_minutes.csv',index=False);pd.DataFrame(windows).to_csv(OUT/'csi_windows_3s.csv',index=False)
    radarcols=['split','minute','radar_frames','snr_db']+[c for c in expected if c.startswith(('rd_','ra_','re_','xy_'))]
    data=meta.merge(expected[radarcols],on=['split','minute'],validate='one_to_one')
    data.to_csv(OUT/'all_minutes.csv',index=False)
    selected_ids=pd.read_csv(PREVIOUS/'paired_minutes.csv')[['split','minute']]
    selected=selected_ids.merge(data,on=['split','minute'],validate='one_to_one')
    assert selected.csi.notna().all(),'A frozen cohort minute lost CSI coverage; do not silently substitute samples'
    import paper_multimodal_20s as base
    base.OUT=OUT;base.evaluate(selected)
    import shutil
    for name in ['cohort_manifest.csv','balancing.json']:
        shutil.copy2(PREVIOUS/name,OUT/name)
    # Original collection coverage remains separate from the balanced cohort.
    audit=[]
    for split in ['calibration_minutes','train','validation_minutes','test_minutes']:
        for act in ['empty','sleep','present']:
            g=data[(data.split==split)&(data.activity==act)]
            complete=g.dropna(subset=['csi']+radarcols[2:])
            audit.append(dict(split=split,activity=act,raw=len(g),csi_qualified=int(g.csi.notna().sum()),paired=len(complete)))
    pd.DataFrame(audit).to_csv(OUT/'coverage_audit.csv',index=False)
    method=json.loads((PREVIOUS/'method.json').read_text())
    method.update(target_sampling_rate_hz=100,resampling='upstream bin averaging and linear interpolation before one-second mean/std and PCA',one_second_quality='original packet quality rules plus exactly 100 resampled values',pca_fit_dots=int(valid[fit].sum()),balanced_cohort='same minute IDs as previous balanced experiment')
    (OUT/'method.json').write_text(json.dumps(method,indent=2))
    print('COMPLETE: 100 Hz processing, new training-only PCA, fixed balanced cohort, and fusion evaluation.',flush=True)

if __name__=='__main__':main()
