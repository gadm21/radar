"""Dataset descriptions, balanced cohorts, boxplots, and modality coverage."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def revise_balanced(blocks,exp,out):
    blocks=list(blocks)
    raw=pd.read_csv(exp/'all_minutes.csv'); d=pd.read_csv(exp/'paired_minutes.csv')
    def replace(prefix,text):
        ids=[i for i,(k,v) in enumerate(blocks) if isinstance(v,str) and v.startswith(prefix)]
        assert len(ids)==1,(prefix,ids)
        i=ids[0];blocks[i]=(blocks[i][0],text)
    def insert_before(prefix,items):
        i=next(i for i,(k,v) in enumerate(blocks) if isinstance(v,str) and v.startswith(prefix))
        blocks[i:i]=items
    # Each row is a minute, so the boxes match the unit used for evaluation.
    fields=[('csi','CSI variance','log scale'),('snr_db','SNR proxy','dB'),('rd_mean','RD mean','log power'),('ra_mean','RA mean','log power'),('re_mean','RE mean','log power'),('xy_mean','XY mean','log power')]
    colors=['#555555','#b52936']
    fig,axs=plt.subplots(3,2,figsize=(3.5,4.6))
    for ax,(col,title,units) in zip(axs.flat,fields):
        vals=[d[(d.split==s)&(d.label==y)][col].values for s in ['validation_minutes','test_minutes'] for y in [0,1]]
        bp=ax.boxplot(vals,positions=[1,2,4,5],widths=.65,patch_artist=True,showfliers=True,flierprops={'markersize':2,'alpha':.55},medianprops={'color':'#ffffff','linewidth':1})
        for i,box in enumerate(bp['boxes']):box.set_facecolor(colors[i%2])
        ax.set_xticks([1.5,4.5],['Val.','Test']);ax.set_title(title,fontsize=8);ax.set_ylabel(units,fontsize=7);ax.tick_params(labelsize=7)
        if col=='csi':ax.set_yscale('log')
        ax.grid(axis='y',alpha=.15)
    fig.legend(handles=[Patch(color=colors[0],label='Empty'),Patch(color=colors[1],label='Occupied')],loc='upper center',ncol=2,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.95),h_pad=.9,w_pad=.6);fig.savefig(out/'feature_boxplots.png',dpi=300);plt.close(fig)
    # Availability is evaluated on the original collection, before undersampling.
    summaries=[]
    fig,axs=plt.subplots(2,1,figsize=(3.5,3.0))
    for ax,split,title in zip(axs,['validation_minutes','test_minutes'],['Validation','Test']):
        r=raw[raw.split==split];n=len(r)
        masks=[r.csi.notna(),r.snr_db.notna()]
        masks.extend(r[[c for c in r if c.startswith(key+'_')]].notna().all(axis=1) for key in ['rd','ra','re','xy'])
        masks.append(np.logical_and.reduce(masks))
        names=['CSI','SNR','RD','RA','RE','XY','Joint']
        counts=[int(m.sum()) for m in masks]
        ax.bar(names,counts,color=['#7570b3']+['#1b9e77']*5+['#2676a8'])
        ax.bar(names,[n-c for c in counts],bottom=counts,color='#cc5555')
        for i,c in enumerate(counts):ax.text(i,c+3,f'{c}',ha='center',fontsize=7)
        ax.set_ylim(0,n*1.24);ax.set_title(f'{title}: {n} collected minutes',fontsize=9);ax.set_ylabel('Usable minutes',fontsize=8);ax.tick_params(labelsize=7)
        for name,c in zip(names,counts):summaries.append(dict(split=split,modality=name,available=c,total=n))
    fig.tight_layout();fig.savefig(out/'modality_coverage.png',dpi=300);plt.close(fig)
    pd.DataFrame(summaries).to_csv(exp/'modality_coverage_summary.csv',index=False)
    # Within-minute quality is separate from availability of a minute feature.
    fig,axs=plt.subplots(1,2,figsize=(3.5,2.1))
    for ax,col,title in zip(axs,['valid_seconds','n_windows_3s'],['Valid CSI seconds','Valid 3 s windows']):
        vals=[raw[raw.split==s][col].values for s in ['validation_minutes','test_minutes']]
        ax.boxplot(vals,tick_labels=['Val.','Test'],widths=.5,flierprops={'markersize':2});ax.set_title(title,fontsize=8);ax.set_ylim(0,62);ax.tick_params(labelsize=7);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(out/'temporal_coverage.png',dpi=300);plt.close(fig)
    # Dataset overview follows the reference paper's collection summary style.
    rows=[['Dataset','Dates (2026)','Raw E/P','Eval. E/P'],['Validation','Sep. 15','112 / 80','80 / 80'],['Test','Sep. 16-17','96 / 153','96 / 96']]
    dataset=[('sub','B. Validation dataset'),('p','The validation dataset was recorded on 15 September 2026 for binary room occupancy detection. It contains 192 nominal one-minute recordings: 112 empty and 80 present. Present indicates a person in the collection room and may include sitting, standing, or walking; no sleep class is available. This later acquisition session is reserved for classifier and threshold selection, and does not fit the PCA basis or feature scaling. The balanced evaluation subset retains all 80 present minutes and samples 80 of the 112 empty minutes.'),('sub','C. Test dataset'),('p','The test dataset was recorded overnight on 16-17 September 2026 and contains 249 nominal one-minute recordings: 96 empty and 153 present. It represents a later acquisition period than validation. Repository metadata identifies the test acquisition group as pi, whereas validation uses t; those identifiers alone do not prove a change of room or sensor geometry. Test annotations follow the acquisition schedule, with empty beginning at 01:10 on 17 September. The balanced subset retains all 96 empty minutes and samples 96 of the 153 present minutes. No sleep discrimination is evaluated.'),('sub','D. Balanced evaluation protocol'),('p','Within each evaluation partition, fully paired minute IDs are sorted chronologically. The majority class is then sampled uniformly without replacement using random seed 42, while every minority-class minute is retained. Sampling uses only recording IDs and class labels, before candidate evaluation; it does not select easy examples, search seeds, duplicate recordings, or move minutes between partitions. The resulting validation and test datasets contain 160 and 192 minutes, respectively, each with equal empty and occupied counts. The excluded 32 validation-empty and 57 test-present minutes remain in the collection and coverage audit. Training remains unchanged at 268 fully paired minutes; class weights address its imbalance.')]
    insert_before('B. Annotation and synchronization',dataset)
    replace('B. Annotation and synchronization','E. Annotation and synchronization')
    replace('C. Quality criteria and coverage','F. Quality criteria and coverage')
    replace('Calibration recordings span','Calibration recordings span 17-18 August and training spans 6-8 September 2026. Calibration contains 808 sleep-labeled minutes; training contains 188 empty, 16 sleep, and 70 present minutes before availability filtering. Table I summarizes the two evaluation datasets in terms of recording period, task classes, and collection versus evaluation size. These are nominal minute counts rather than verified uninterrupted recording durations. Both datasets target occupancy in the room containing the sensors; room dimensions and physical sensor positions are not verified by the available metadata.')
    for split,title in [('validation_minutes','Validation'),('test_minutes','Test')]:
        r=raw[raw.split==split];s=d[d.split==split]
        start=pd.to_datetime(r.minute.min(),format='%Y%m%d_%H%M');end=pd.to_datetime(r.minute.max(),format='%Y%m%d_%H%M')
        text=(f'{title} recordings run from {start:%d September %H:%M} to {end:%d September %H:%M} by local folder timestamps, with possible gaps. Across the original collection, the selected CSI stream contributes {int(r.packets.sum()):,} decoded packets and the radar extraction uses {int(r.radar_frames.sum()):,} frames. The balanced subset contributes {int(s.packets.sum()):,} CSI packets and {int(s.radar_frames.sum()):,} radar frames over {len(s)} nominal minutes. These counts describe the loader-selected CSI stream and radar frames used by this pipeline, rather than all receiver traffic. Median valid CSI support is {r.valid_seconds.median():.1f} one-second bins and {r.n_windows_3s.median():.1f} three-second windows per collected minute.')
        insert_before('C. Test dataset' if title=='Validation' else 'D. Balanced evaluation protocol',[('p',text)])
    replace('A CSI second is valid only','A CSI second is valid only when it contains at least five packets, spans at least 0.5 seconds, and has no packet or boundary gap greater than 0.5 seconds. Three consecutive valid one-second bins define a variance window; gaps are not interpolated. All 192 validation and 249 test minutes provide CSI variance, SNR, and every radar-map feature. Figure 4 reports these modality-specific counts before balancing. Thus undersampling, rather than a missing modality, explains the smaller evaluation sets. Six of 274 training minutes lack a required radar/SNR feature, leaving 268 fully paired training minutes.')
    replace('The original chronological partitions','Chronological partition membership is preserved. Three-second windows from a minute cannot enter different partitions. Models are fitted on the 268 unchanged training minutes; thresholds and candidate selection use the balanced validation set of 160 minutes; frozen decisions are scored on the balanced test set of 192 minutes. All 26 candidates use exactly the same selected minute IDs, isolating feature and model differences from sample-composition differences.')
    replace('The raw three-second variance feature',next(v for k,v in blocks if isinstance(v,str) and v.startswith('The raw three-second variance feature')).split('Figure 2')[0]+'Figure 2 compares class-conditional boxplots in both balanced datasets. CSI uses the median three-second variance; SNR uses its minute proxy; each map panel displays its spatial mean of minute-mean log power. The classifiers retain all five descriptors per map. Separation and overlap differ across sessions, so descriptive boxes are not treated as a classifier evaluation.')
    replace('The revised matched cohort retains','All 192 collected validation minutes and all 249 collected test minutes meet the required modality-feature availability criteria (Fig. 4). Balancing then retains 80/112 empty and 80/80 present validation minutes, and 96/96 empty and 96/153 present test minutes. This is intentional class balancing, not a sensor dropout. Figure 5 separately reports within-minute CSI support: one usable window is sufficient for a feature, so 100% minute availability does not imply 60 seconds of valid observations.')
    replace('Test minutes contain between',next(v for k,v in blocks if isinstance(v,str) and v.startswith('Test minutes contain between')).replace('Test minutes','Collected test minutes')+' Balanced accuracy equals ordinary accuracy on each balanced evaluation set. The natural occupancy prevalence is not preserved, and temporal correlation remains after undersampling.')
    # Correct statements generated for the earlier unbalanced full-cohort revision.
    result=[]
    for kind,v in blocks:
        if isinstance(v,str):
            v=v.replace('On the matched test cohort of 192 minutes','On the class-balanced test cohort of 192 minutes (96 empty and 96 occupied)')
            v=v.replace('On the matched cohort, the validation-selected','On the balanced test subset of 96 empty and 96 occupied minutes, the validation-selected')
            v=v.replace('the enlarged cohort used in this revision','the balanced cohort used in this revision')
        if kind=='table' and v[2].startswith('TABLE I.'):
            v=(rows,[57,70,63,62],'TABLE I. Evaluation dataset overview. Both datasets address binary occupancy in the collection room. E/P: empty/present minute counts; eval. denotes the fixed balanced subset. Dates, nominal durations, and decoded sample counts are detailed in the text.')
        if kind=='fig':
            name,height,cap=v
            if name=='test_scatter.png':v=('feature_boxplots.png',331,'Fig. 2. Feature boxplots for balanced validation (80 per class) and test (96 per class). Boxes: interquartile range; line: median; whiskers: 1.5 IQR; dots: outliers. CSI uses a logarithmic axis; map panels show mean log-power descriptors.')
            if name=='coverage.png':v=('modality_coverage.png',216,'Fig. 4. Modality-feature coverage in the original validation and test collections before balancing. CSI, SNR, each radar view, and their joint availability cover all minutes. Radar views share one sensor and are not independent modalities.')
        result.append((kind,v))
        if kind=='fig' and v[0]=='modality_coverage.png':
            result.append(('fig',('temporal_coverage.png',151,'Fig. 5. Within-minute CSI coverage over the original collections. Left: valid one-second bins; right: valid three-second windows. Window overlap means these counts are not independent observations.')))
    return result
