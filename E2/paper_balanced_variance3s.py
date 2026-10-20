"""Freeze class-balanced evaluation cohorts before fitting/scoring candidates."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.paper_packages'))
import pandas as pd
import paper_multimodal_20s as base
SOURCE=ROOT/'E2/outputs/paper_multimodal_variance3s'
OUT=ROOT/'E2/outputs/paper_balanced_variance3s'
OUT.mkdir(exist_ok=True)
raw=pd.read_csv(SOURCE/'all_minutes.csv',float_precision='round_trip')
paired=pd.read_csv(SOURCE/'paired_minutes.csv',float_precision='round_trip')
parts=[paired[paired.split=='train']]
for split in ['validation_minutes','test_minutes']:
    d=paired[paired.split==split].sort_values('minute')
    n=int(d.groupby('label').size().min())
    for label in [0,1]:
        group=d[d.label==label]
        parts.append(group.sample(n=n,random_state=42) if len(group)>n else group)
selected=pd.concat(parts).sort_values(['split','minute'])
assert selected.groupby(['split','label']).size().loc['validation_minutes'].tolist()==[80,80]
assert selected.groupby(['split','label']).size().loc['test_minutes'].tolist()==[96,96]
manifest=raw[['split','minute','activity','label']].merge(selected[['split','minute']].assign(selected=True),on=['split','minute'],how='left')
manifest['selected']=manifest.selected.eq(True)
manifest.to_csv(OUT/'cohort_manifest.csv',index=False)
raw.to_csv(OUT/'all_minutes.csv',index=False)
for name in ['coverage_audit.csv','pca.npz','method.json']:
    shutil.copy2(SOURCE/name,OUT/name)
(OUT/'balancing.json').write_text(json.dumps({'seed':42,'method':'uniform majority-class undersampling without replacement after sorting minute IDs; minority retained in full','validation':[80,80],'test':[96,96],'training':'unchanged','selection_uses':'labels and IDs only; no features, predictions or scores'},indent=2))
# Persist the cohort before any model/threshold selection; no seed search.
selected.to_csv(OUT/'selected_minutes.csv',index=False)
base.OUT=OUT
base.evaluate(selected)
