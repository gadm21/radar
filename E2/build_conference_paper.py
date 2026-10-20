"""Standalone IEEE-style paper; measured results are read from the frozen experiment."""
import sys,re,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.paper_packages'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
for name,file in [('PaperSerif','times.ttf'),('PaperBold','timesbd.ttf'),('PaperItalic','timesi.ttf'),('PaperBoldItalic','timesbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('PaperSerif',normal='PaperSerif',bold='PaperBold',italic='PaperItalic',boldItalic='PaperBoldItalic')
HZ100='--100hz' in sys.argv
BALANCED='--balanced' in sys.argv or HZ100
VARIANCE3='--variance3s' in sys.argv or BALANCED
OUT=ROOT/('output/pdf/multimodal_conference_100hz_source' if HZ100 else 'output/pdf/multimodal_conference_balanced_source' if BALANCED else 'output/pdf/multimodal_conference_3s_variance_source' if VARIANCE3 else 'output/pdf/multimodal_conference_source');OUT.mkdir(parents=True,exist_ok=True)
EXP=ROOT/('E2/outputs/paper_100hz_variance3s' if HZ100 else 'E2/outputs/paper_balanced_variance3s' if BALANCED else 'E2/outputs/paper_multimodal_variance3s' if VARIANCE3 else 'E2/outputs/paper_multimodal_20s')
df=pd.read_csv(EXP/'paired_minutes.csv');res=pd.read_csv(EXP/'results.csv')
TITLE='Multimodal Wireless Sensing for Room Occupancy Detection: A CSI and Radar Fusion Study'
AUTHORS='Gad Gad*, Iqra Batool*, Mostafa Fouda<super>†</super>, Shikhar Verma<super>‡</super>, Zubair Md Fadlullah*'
AFFILS='*Department of Computer Science, Western University, London, ON, Canada<br/><super>†</super>Department of Electrical and Computer Engineering, Idaho State University, Pocatello, ID, USA<br/><super>‡</super>School of Informatics, Kochi University of Technology, Kochi, Japan'
plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'legend.fontsize':8,'font.family':'DejaVu Sans'})
# Column-sized scientific figures, regenerated from saved measurements.
fig,ax=plt.subplots(figsize=(3.5,2.5))
for act,c in [('empty','#333333'),('present','#b52936')]:
    d=df[(df.split=='test_minutes')&(df.activity==act)]
    ax.scatter(d.snr_db,d.csi,s=20,color=c,label=f'{act} (n={len(d)})',alpha=.8)
ax.set(xlabel='Radar SNR proxy (dB)',ylabel='3 s PCA variance trace' if VARIANCE3 else '20 s CSI path length',yscale='log');ax.legend();ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(OUT/'test_scatter.png',dpi=260);plt.close(fig)
names=['CSI','CSI+SNR','CSI+SNR+RD','CSI+SNR+RA','CSI+SNR+RE','CSI+SNR+XY','CSI+radar all']
best=res[res.split=='validation_minutes'].sort_values(['balanced_accuracy','auc','model'],ascending=[False,False,True]).drop_duplicates('features').set_index('features')
fig,ax=plt.subplots(figsize=(3.5,2.7))
y=np.arange(len(names));va=[];te=[]
for name in names:
    row=best.loc[name];va.append(row.balanced_accuracy)
    te.append(res[(res.features==name)&(res.model==row.model)&(res.split=='test_minutes')].iloc[0].balanced_accuracy)
ax.barh(y-.17,va,.34,label='Validation',color='#2676a8');ax.barh(y+.17,te,.34,label='Test',color='#d87927')
ax.set_yticks(y,['CSI','+ SNR','+ SNR + RD','+ SNR + RA','+ SNR + RE','+ SNR + XY','+ all radar']);ax.invert_yaxis();ax.set(xlim=(0,1),xlabel='Balanced accuracy');ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,frameon=False,fontsize=7.5);fig.tight_layout();fig.savefig(OUT/'fusion_comparison.png',dpi=260);plt.close(fig)
maps=np.load((ROOT/'E2/outputs/paper_multimodal_variance3s' if BALANCED else EXP)/'radar_features/train_20260906_1910_maps.npz')
fig,axs=plt.subplots(2,2,figsize=(3.5,3.0))
for ax,key in zip(axs.flat,['rd','ra','re','xy']):
    ax.imshow(maps[key],origin='lower',aspect='auto');ax.set_title(key.upper());ax.set_xticks([]);ax.set_yticks([])
fig.tight_layout(pad=.6);fig.savefig(OUT/'radar_maps.png',dpi=260);plt.close(fig)
# A data-driven coverage figure emphasizes the complete-case selection effect.
ct=df[df.split=='test_minutes'];coverage_counts=[(int((ct.label==0).sum()),96),(int((ct.label==1).sum()),153)]
fig,ax=plt.subplots(figsize=(3.5,2.1));ax.bar(['Empty','Occupied'],[a/b for a,b in coverage_counts],color=['#555555','#b52936']);ax.set(ylim=(0,1.15 if VARIANCE3 else .4),ylabel='Test minutes retained')
ticks=[0,.25,.5,.75,1] if VARIANCE3 else [0,.1,.2,.3,.4]
ax.set_yticks(ticks,[f'{100*x:.0f}%' for x in ticks])
for i,(a,b) in enumerate(coverage_counts):ax.text(i,a/b+.015,f'{a}/{b}',ha='center')
fig.tight_layout();fig.savefig(OUT/'coverage.png',dpi=260);plt.close(fig)

blocks=[]
def p(s):blocks.append(('p',s))
def h(s):blocks.append(('h',s))
def sub(s):blocks.append(('sub',s))
def eq(s):blocks.append(('eq',s))
def fig(name,height,caption):blocks.append(('fig',(name,height,caption)))
def table(rows,widths,caption):blocks.append(('table',(rows,widths,caption)))

p('<b>Abstract—</b>Room occupancy detection requires reliable recognition of both moving and stationary occupants. WiFi channel state information (CSI) and radar provide different observations of the same indoor environment, but combining them does not necessarily improve detection under changing acquisition conditions. This paper presents an empirical study of lightweight CSI–radar fusion using 1,523 labeled minute recordings. A 20-second trajectory-length feature is computed from consecutive two-dimensional principal-component projections of CSI amplitude statistics. It is fused with a radar signal-to-background proxy and descriptors of range-Doppler, range-azimuth, range-elevation, and projected XY energy maps. Thirteen feature configurations and two conventional classifiers are evaluated using chronological partitions and validation-selected decision thresholds. On the common test cohort, CSI alone achieves 0.693 area under the ROC curve (AUC) and 77.1% balanced accuracy. Adding radar range-Doppler information raises AUC to 0.784 but gives 76.0% balanced accuracy; full fusion gives 70.8%. Only 52 of 249 test minutes meet the feature-availability requirements, including four empty minutes. The study identifies representation shift and unequal coverage as central obstacles to robust fusion, and provides a reproducible baseline for subsequent collection and deployment evaluation.')
p('<b>Index Terms—</b>Wireless sensing, room occupancy, WiFi CSI, radar, multimodal fusion, principal component analysis.')
h('I. Introduction')
p('Occupancy information supports room-level automation and monitoring without requiring occupants to carry devices. The sensing objective is presence in the room containing the collection devices, including intervals with little visible movement. Wireless observations are attractive for this task because they can be collected without retaining camera imagery. However, an activity-sensitive signal is not automatically a reliable occupancy indicator: an occupied room can contain a seated, standing, or sleeping person whose signal changes only weakly.')
p('The motivating observations in this collection are that a radar signal-to-background feature separates many empty and occupied minutes, yet its distributions overlap. A person outside the radar field of view may generate weak radar evidence while remaining in the room. Small body movements may also resemble an empty-room baseline. These are plausible explanations rather than measured failure categories: the current annotations do not include occupant angle, visibility, or physiological ground truth. Work on stationary-person sensing illustrates why small-motion evidence deserves separate treatment [1].')
p('CSI measures the propagation channel across WiFi subcarriers. Changes in multipath conditions can produce temporal variation that complements a directional radar observation. Conversely, CSI can change because of environmental disturbances or activity outside the room of interest. Fusion therefore requires more than concatenating informative-looking features. The two inputs must have adequate temporal coverage, their distributions must transfer across recording periods, and their errors must differ in a way that a classifier can exploit.')
p('This paper investigates these issues using an interpretable representation: the cumulative Euclidean distance traversed by consecutive CSI projections in a fixed two-dimensional PCA space. The chosen 20-second configuration is combined with a radar SNR proxy and four radar-map representations. The contribution is an audited multimodal dataset description, a reproducible low-dimensional feature pipeline, and a matched-cohort comparison of 26 classifier/configuration candidates. We distinguish improved ranking from improved thresholded detection, and quantify how missing CSI affects the evaluated population.')
p('The study is exploratory. The best validation-selected candidate remains CSI alone, and no evaluated candidate exceeds 80% test balanced accuracy. These negative and mixed findings are useful for designing a more reliable sensing system: they show that additional radar representations can improve AUC while failing to improve the operating point selected on a different acquisition period.')
h('II. Related Work')
p('WiFi presence sensing must address both large movements and nearly stationary occupants. Wu et al. introduced DeMan, which uses CSI amplitude and phase for motion detection and breathing-related patterns for stationary human presence [1]. Their physiological processing motivates careful treatment of stationary occupancy; it does not imply that an undifferentiated trajectory-length statistic directly measures breathing.')
p('SHARP extracts Doppler information from phase-processed channel measurements and evaluates activity recognition across people, days, and environments [2]. Its generalization objective is relevant to the present chronological evaluation, although the task and classifier differ. LiteHAR uses random convolutional features with a ridge classifier as a lightweight alternative to training large feature-extraction networks [3]. It provides prior evidence that efficient wireless-sensing models are a research direction in their own right. Our experiment does not reproduce either method or make a measured runtime comparison with them.')
p('Radar-related occupancy sensing is also established. Chetty et al. use passive WiFi radar, range-Doppler processing, Doppler spectrograms, and CNN classification for occupancy and counting [4]. Passive WiFi radar is different from the separate raw-ADC radar and CSI receiver studied here. Chen et al. report radar–WiFi activity fusion with learned cross-modal alignment and adaptive weighting [5]. Thus, combining radar and WiFi is not itself a novelty claim; our focus is the behavior of compact, explicitly defined representations under incomplete coverage and temporal distribution shift.')
p('Room specificity presents a further challenge. Jung et al. examine multi-zone CSI occupancy estimation and identify signal variability and cross-room interference as practical difficulties [6]. Accordingly, scores from other datasets are not treated as directly comparable baselines. Our same-minute ablations isolate representation choices within one collection, while a broader multi-room and multi-participant study remains necessary.')
h('III. Signal Model and Sensing Task')
sub('A. CSI observations')
p('Let H<sub>k</sub>(t) denote complex CSI on retained subcarrier k at packet time t. A conceptual multipath model expresses the channel as a sum of path contributions with time-varying complex amplitudes and delays. The implementation uses A<sub>k</sub>(t) = |H<sub>k</sub>(t)| for 52 retained subcarriers. No phase sanitization or explicit path-delay estimation is performed. The resulting feature represents temporal amplitude change, not a calibrated physical displacement.')
eq('H<sub>k</sub>(t) = sum<sub>l</sub> a<sub>l</sub>(t) exp[-j 2π f<sub>k</sub> τ<sub>l</sub>(t)].  (1)')
p('Changes in a person’s position or motion can perturb these path contributions, but their effect depends on geometry, channel conditions, and receiver behavior. Therefore, low CSI variation is consistent with both an empty room and a sufficiently stationary occupied room. High variation is likewise not sufficient evidence of in-room motion without additional controls.')
sub('B. Radar observations and room-level labels')
p('Each decoded radar frame is a cube of 64 chirps, 128 fast-time ADC samples, and three receiver channels. Fast-time transformation yields range-bin responses; slow-time transformation yields Doppler-bin responses. Receiver-pair phase differences are used to form azimuth and elevation energy views. Since this analysis does not verify per-record RF settings or antenna calibration, axes are interpreted as bins and normalized sensor coordinates rather than calibrated meters, velocity, or occupant location.')
p('The binary target is y = 0 for empty and y = 1 for occupied, with sleep and present both mapped to occupied. The descriptive present label includes standing, sitting, and walking, but those activities are not separately evaluated. Sleep examples occur in calibration and training only. Consequently, the experiment does not establish sleep recognition, physiological monitoring, or three-class generalization.')
h('IV. Multimodal Dataset')
sub('A. Acquisition organization')
p('The collection contains 1,523 labeled recording folders spanning August and September 2026. Each folder represents a nominal minute and contains a manifest plus CSI and radar data in chunked files or a synchronized-capture container. Manifests identify ESP32 CSI and DreamHAT radar acquisition. CSI may contain multiple receiver streams; the existing loader selects the receiver with the greatest packet coverage rather than fusing receivers. Radar payloads contain packed 12-bit samples, decoded into the three-channel cube described above.')
p('Calibration recordings span 17–18 August; training spans 6–8 September; validation is from 15 September; and test spans 16–17 September 2026. Calibration has 808 sleep-labeled minutes. Training has 188 empty, 16 sleep, and 70 present minutes. Validation has 112 empty and 80 present minutes; test has 96 empty and 153 present minutes. Folder counts describe recording units, not verified uninterrupted hours of synchronized data.')
table([['Partition','Raw E/S/P','CSI E/S/P','Paired E/S/P'],['Calibration','0/808/0','0/441/0','—'],['Train','188/16/70','39/4/20','36/4/20'],['Validation','112/0/80','68/0/43','68/0/43'],['Test','96/0/153','4/0/48','4/0/48']],[52,67,66,67],'TABLE I. Recording counts. E: empty; S: sleep; P: present. CSI requires at least one valid 20-second path. Paired additionally requires radar descriptors and the SNR proxy. Calibration is used only for PCA fitting.')
sub('B. Annotation and synchronization')
p('Only recognized activity labels in the manifests are retained; automatic occupancy annotations are excluded. The repository records that test labels were assigned using the acquisition schedule, with the first empty minute at 01:10 on 17 September. Those schedule-derived labels are not independently verified frame by frame. Placement identifiers distinguish acquisition groups but do not establish verified room geometry or a controlled placement experiment.')
p('CSI and reconstructed radar maps are restricted to timestamps in the scheduled interval [0, 60) seconds. CSI timestamps are sorted and duplicates removed. Radar frame order is preserved; where burst-flushed frame timestamps are unreliable, the existing loader reconstructs timestamps within recorded second buckets. Pairing is by recording identity and host timing, not verified hardware-trigger synchronization. The historical SNR feature retains its original cached frame coverage, which can differ from the newly reconstructed maps.')
sub('C. Quality criteria and coverage')
p('A CSI second is valid only when it contains at least five packets, spans at least 0.5 seconds, and has no packet or boundary gap greater than 0.5 seconds. Twenty-one consecutive valid dots are required for a 20-second path. Missing intervals are not interpolated. These criteria leave 441 calibration, 63 training, 111 validation, and 52 test minutes with usable CSI paths. Requiring all radar inputs reduces supervised training to 60 minutes, while validation and test remain unchanged (Table I).')
p('Participant count, complete antenna calibration, exact room dimensions, and per-record RF configuration have not been verified in this dataset audit. The present evidence should therefore be interpreted as a chronological collection study rather than proof of cross-participant or cross-room performance. Public-release status and a stable dataset identifier are also not established.')
h('V. Feature Extraction and Fusion Methodology')
sub('A. Fixed PCA representation of CSI')
p('For valid second s, the mean and population standard deviation of each subcarrier amplitude are concatenated into a 104-dimensional vector x<sub>s</sub>. PCA is fitted once using 52,876 valid one-second vectors from calibration and training. The data are centered without feature standardization or whitening. The first two principal coordinates define z<sub>s</sub>. Validation and test neither fit nor update the basis.')
eq('x<sub>s</sub> = [μ<sub>1,s</sub>, …, μ<sub>52,s</sub>, σ<sub>1,s</sub>, …, σ<sub>52,s</sub>]<super>T</super>.  (2)')
eq('z<sub>s</sub> = W<sub>2</sub><super>T</super>(x<sub>s</sub> − x̄).  (3)')
p('Using a fixed basis is essential: if coordinates were recomputed independently for each dot, inter-dot distance would combine signal change with a change of axes. PCA reduces dimensionality but does not guarantee that occupancy information is retained; its directions maximize variance in the fit population, which is strongly influenced by calibration recordings.')
sub('B. Twenty-second trajectory length')
p('For start second s, cumulative path length sums 20 Euclidean displacements across 21 consecutive dots. The minute feature is the median of all valid paths starting at seconds 0 through 39. If no complete path exists, the minute is unavailable for the primary experiment. The classifier receives log(1 + L<sub>m</sub>) to reduce feature skew.')
eq('L(s) = sum<sub>j=0</sub><super>19</super> ||z<sub>s+j+1</sub> − z<sub>s+j</sub>||<sub>2</sub>.  (4)')
eq('L<sub>m</sub> = median {L(s): all 21 dots are valid}.  (5)')
p('Path length measures volatility rather than net displacement: a trajectory that returns to its starting point can still have a large value. It also depends on signal scale, residual noise, packet quality, and the selected PCA dimensions. These dependencies motivate using a shared basis and explicit packet-coverage criteria rather than interpreting path length as physical motion distance.')
sub('C. Radar maps')
p('For each chirp, the fast-time sample mean is removed, a Hann window is applied, and a 128-point FFT is computed. The first 64 range bins are retained. RD additionally applies a Hann window and 64-point FFT along chirps, centers Doppler, and averages power across receiver channels. The first two range bins are suppressed. Static slow-time components are retained, so the map contains both background and dynamic responses.')
p('The local radar implementation documents an L-shaped receiver array. Two-element conventional beamforming uses RX1/RX3 for RA and RX2/RX3 for RE. Twenty-five steering angles cover −60° to +60° with assumed half-wavelength spacing. Beam power is averaged over chirps. This reconstructs distinct horizontal and vertical responses rather than treating all three receivers as a linear azimuth array. Limited aperture and unverified phase calibration restrict the physical interpretation of these maps.')
eq('P<sub>ab</sub>(r,θ) = mean<sub>c</sub> |R<sub>a</sub>(c,r)e<super>jπ sin θ</super> + R<sub>b</sub>(c,r)|².  (6)')
p('The XY map is derived from RA using normalized range r: x = r sin θ and y = r cos θ. Energy is assigned to a 24 × 24 grid, using the maximum when multiple polar cells collide. XY is therefore a different representation of the same radar information, not an independent sensing modality. No room-coordinate registration is applied.')
fig('radar_maps.png',222,'Fig. 1. Mean log-power views for training minute 20260906_1910. RD rows are Doppler and columns range; RA/RE rows are range and columns angle; XY is a normalized sensor-coordinate projection. Colors are independently scaled; panels illustrate structure, not calibrated intensity comparisons.')
sub('D. Minute descriptors and the SNR proxy')
p('Each power map is transformed by log(1 + power). Five scalar descriptors are computed: spatial mean, 90th percentile, and maximum of the temporal mean map; spatial 90th percentile of temporal standard deviation; and mean absolute difference between consecutive frames. Frame differences are not divided by elapsed time, so timing irregularity is a possible nuisance factor. Pooling reduces map dimensionality but discards detailed target shape and spatial relationships.')
p('The existing SNR proxy is retained for comparability with the original radar analysis. For each cached RD frame, the strongest amplitude outside near-range and central Doppler masks is divided by the median background amplitude. Ten times the base-10 logarithm is taken, and the minute maximum is used. Because the cached quantity is amplitude rather than calibrated power, this is a historical signal-to-background discrimination proxy, not a physical SNR measurement.')
p('CSI + SNR produces two inputs. Adding one map produces seven; combining SNR and all maps gives 21; adding CSI gives 22. Radar descriptors summarize available frames within the minute, while CSI summarizes rolling 20-second paths. The evaluated decision unit is consequently a minute, even though the CSI subwindow spans 20 seconds.')
sub('E. Classifiers')
p('Early feature fusion concatenates the chosen scalar inputs. A balanced logistic regression (LR; C = 0.3, maximum 3,000 iterations) and a balanced random forest (RF; 200 trees, maximum depth 4, minimum leaf size 5, seed 42) are evaluated. Median imputation and standardization are fitted on training minutes only. All primary comparisons use complete cases, so no missing feature is actually imputed in those comparisons. Calibration contributes to PCA only, not classifier labels.')
h('VI. Dataset Characterization and Evaluation')
sub('A. Partition and selection protocol')
p('The original chronological partitions are preserved. Neither minute windows nor overlapping CSI paths are randomly distributed across training and evaluation. The primary comparison comprises 60 training minutes, 111 validation minutes, and 52 test minutes. Every feature configuration is evaluated on this identical cohort, preventing availability differences from masquerading as a fusion gain.')
p('The 13 configurations are CSI, SNR, each of RD/RA/RE/XY alone, CSI + SNR, CSI + SNR with each map, all radar features, and CSI + all radar features. Together with LR and RF, this gives 26 candidates. For each candidate, validation selects the probability threshold maximizing balanced accuracy. Ties prefer the threshold closest to 0.5. The overall candidate is selected by validation balanced accuracy, then AUC, then deterministic name ordering. Test does not select features, model, or threshold within this run.')
sub('B. Metrics and interpretation')
eq('BA = ½ [TN/(TN + FP) + TP/(TP + FN)].  (7)')
p('Balanced accuracy (BA) weights empty specificity and occupied sensitivity equally. ROC AUC measures ranking across possible thresholds, whereas BA describes the frozen operating point. Ordinary accuracy and occupied-class F1 are reported for the selected candidate, but cannot by themselves establish effective empty-room detection under strong imbalance. The paired test contains 48 occupied and only four empty minutes: predicting occupied throughout would yield 92.3% accuracy and 50% BA.')
p('Threshold tuning and comparison of 26 candidates make validation scores optimistic. In addition, the same test partition appeared in earlier project analyses, although it is not used for selection here. The reported test is therefore a fixed exploratory comparison, not a newly untouched confirmatory experiment. A new recording session is needed before strong generalization claims can be made.')
sub('C. Feature distributions')
p('The requested CSI configuration reproduces raw trajectory-length AUC of 0.934 on validation and 0.693 on test. Figure 2 illustrates overlap in the paired test distribution. A high CSI value or high radar proxy can identify some occupied minutes, but many occupied minutes occupy the same low-value region as empty examples. A two-feature boundary must therefore trade missed occupancy against false occupied decisions.')
fig('test_scatter.png',181,'Fig. 2. Paired test-minute CSI path length versus the radar SNR proxy. The vertical axis is logarithmic. The four empty examples are too few to characterize the full empty-room distribution.')
p('The radar maps also shift between partitions. Median minute-mean RA log power is 6.167 in training, 9.880 in validation, and 9.884 in test. These are observed feature differences; the present analysis does not isolate whether hardware, environment, acquisition settings, or their combination causes them. Such differences are consistent with difficulty transferring a training-fitted scale and classifier.')
h('VII. Results and Discussion')
rows=[['Inputs','Model','Val. BA','Test BA','Test AUC']]
for name in ['CSI','SNR','RD','RA','RE','XY','CSI+SNR','CSI+SNR+RD','CSI+SNR+RA','CSI+SNR+RE','CSI+SNR+XY','Radar all','CSI+radar all']:
    v=best.loc[name];t=res[(res.features==name)&(res.model==v.model)&(res.split=='test_minutes')].iloc[0]
    rows.append([name,v.model,f'{v.balanced_accuracy:.3f}',f'{t.balanced_accuracy:.3f}',f'{t.auc:.3f}'])
table(rows,[84,29,46,46,47],'TABLE II. Matched-cohort ablation. For each input set, the displayed classifier is chosen using validation only. All thresholds are frozen before test scoring. Full results retain both classifiers, validation AUC, and confusion counts.')
sub('A. CSI and simple fusion')
p('CSI alone with LR is the overall validation-selected candidate, with 0.886 validation BA and 0.934 AUC. At the selected threshold of 0.5972, test BA is 0.771 and AUC is 0.693. The confusion matrix is TN = 4, FP = 0, FN = 22, and TP = 26, corresponding to 100% specificity and 54.2% sensitivity. Ordinary test accuracy is 57.7%, and occupied-class F1 is 0.703. The high specificity should be interpreted in light of its four-example denominator.')
p('Adding SNR to CSI with LR produces the same test confusion counts and 0.771 BA, but AUC falls to 0.682. Thus the simple fusion does not improve the selected operating point. SNR alone gives 0.708 test BA and 0.641 AUC for its validation-selected LR. In this matched cohort, the CSI trajectory is the stronger individual discriminator, despite overlap in both distributions.')
sub('B. Contributions of the radar maps')
p('CSI + SNR + RD with RF yields test AUC of 0.784, an absolute increase of 0.091 over CSI alone. Its validation-selected threshold nevertheless yields 0.760 BA, with TN = 4, FP = 0, FN = 23, and TP = 25. Better ranking therefore does not translate into an improved fixed decision rule. Retuning the threshold on test to increase BA would invalidate the stated selection protocol.')
p('The validation-selected RA, RE, and XY fusion candidates give test BA of 0.656, 0.646, and 0.708, respectively. Full CSI–radar fusion reaches 0.755 AUC and 0.708 BA. No evaluated candidate exceeds 0.80 test BA. These outcomes do not support the claim that adding more map representations improves occupancy detection in this collection. They also do not establish that raw radar lacks complementary information: compact pooling, limited supervised data, redundancy, and distribution shift all constrain the tested approach.')
fig('fusion_comparison.png',195,'Fig. 3. Validation and test balanced accuracy for CSI and fusion variants. Each input set uses its validation-selected classifier. The radar-map variants improve neither all operating points nor all metrics.')
sub('C. Redundancy, shift, and failure mechanisms')
p('RA and XY are deterministically related, while SNR, RD, RA, and RE all originate from the same radar. Additional feature count should not be equated with additional independent sensing evidence. The full vector has 22 inputs for only 60 supervised training minutes. Even with regularization or shallow trees, nuisance variation can dominate a decision boundary when it changes substantially between acquisition periods.')
p('Some map-only LR candidates require extreme validation thresholds and transfer poorly to test. This is consistent with the observed map-distribution shift, but the experiment does not establish a causal explanation. A controlled follow-up should first verify decoder assumptions and acquisition settings, then compare background removal and relative-energy normalization using development data. Any resulting improvement must be assessed on a new session rather than repeatedly tuning to the current test.')
sub('D. Coverage is part of system performance')
fig('coverage.png',151,'Fig. 4. Fraction of labeled test minutes retained by the complete-case protocol. Coverage differs sharply by class, so matched-cohort results do not describe full-time deployment performance.')
p('The 52 paired test minutes represent only 20.9% of the 249 labeled test minutes. Retention is 4/96 = 4.2% for empty and 48/153 = 31.4% for occupied minutes (Fig. 4). The resulting sample is not representative of the raw class balance. Availability should therefore be treated as a system outcome, alongside sensitivity and specificity, rather than as an invisible preprocessing detail.')
p('One additional error among the four empty test minutes changes specificity by 25 percentage points and BA by 12.5 points. Adjacent minutes are correlated, so an independent-minute confidence interval would also exaggerate effective sample size. These constraints limit conclusions about small performance differences between methods. A missing-CSI radar fallback is a logical deployment option, but its full-dataset performance has not been measured in this experiment.')
sub('E. Computational considerations')
p('The representation is designed for conventional classifiers rather than a learned high-dimensional feature extractor. Once the per-second statistics are available, projection to two components requires a fixed small matrix multiplication, and each trajectory uses 20 two-dimensional distances. LR inference is linear in the number of scalar features. A depth-4, 200-tree RF requires at most about 800 internal decision tests per example. These structural observations do not quantify end-to-end processing speed.')
p('Radar decoding, FFTs, beamforming, and temporal aggregation may dominate inference cost. Moreover, a minute-aggregated decision incurs an observation interval distinct from computation time. No matched deep-learning baseline, wall-clock latency, device energy measurement, or edge deployment benchmark was performed. A speedup claim would require the same hardware, input coverage, observation window, and evaluation protocol, including preprocessing rather than classifier time alone. LiteHAR [3] is relevant prior work, not a substitute for that comparison.')
h('VIII. Reproducibility and Data Availability')
p('The accompanying experiment package contains per-minute labels and features, coverage metadata, the fitted PCA basis, all candidate scores, per-minute probabilities and predictions, the validation-selected configuration, and the selected classifier. The analysis scripts specify deterministic settings and preserve recording identifiers. Saved predictions allow confusion matrices and thresholded metrics to be checked independently of model fitting.')
p('The raw multimodal collection is not asserted to be publicly released. A stable repository or archival identifier, licensing, participant-consent documentation, and verified hardware/configuration metadata are required before a public dataset release. The existing CSI-only manuscript and its timing results concern a different experiment and are not used as numerical evidence for the fusion system.')
h('IX. Limitations and Future Work')
p('The main limitations are the small complete-case training set, four empty test minutes, unequal class-dependent coverage, temporal dependence, schedule-derived test annotations, and previous inspection of the evaluation partition. These make the results suitable for exploratory comparison, not a definitive performance claim. Calibration is dominated by sleep, and sleep is absent from validation and test; no conclusion about sleep discrimination follows from the binary scores.')
p('Hardware-level synchronization, angular calibration, per-record RF settings, participant count, and exact room geometry remain unverified. Field-of-view failure and breathing are motivating hypotheses rather than annotated mechanisms. Scalar map descriptors also discard spatial structure that a different lightweight model might use. These limitations should be addressed through controlled acquisition and measurement before increasing classifier complexity.')
p('Future collection should include prolonged empty intervals, stationary and sleeping occupants, movement inside and outside the radar view, and motion in adjacent rooms. Development should evaluate time-normalized changes, background-relative radar features, and a missing-modality policy. A prospectively held-out session should then report sensitivity, specificity, false occupied time, coverage, and computational cost together. This design would test whether fusion improves practical room occupancy rather than only ranking selected minutes.')
h('X. Conclusion')
p('This paper presented a reproducible CSI–radar occupancy study using a fixed 20-second, two-dimensional PCA trajectory feature and multiple radar representations. The matched-cohort evaluation shows that range-Doppler fusion improves test AUC from 0.693 to 0.784, while CSI alone remains the validation-selected classifier and achieves 77.1% test balanced accuracy. Full fusion does not improve this operating point. Unequal CSI coverage and radar feature shift are central findings, not secondary caveats. The resulting dataset audit and baseline provide a concrete foundation for improved acquisition, normalization, missing-modality handling, and prospective evaluation.')
h('References')
refs=[
('C. Wu, Z. Yang, Z. Zhou, X. Liu, Y. Liu, and J. Cao, “Non-Invasive Detection of Moving and Stationary Human With WiFi,” IEEE Journal on Selected Areas in Communications, vol. 33, no. 11, pp. 2329–2342, 2015.','10.1109/JSAC.2015.2430294'),
('F. Meneghello, D. Garlisi, N. Dal Fabbro, I. Tinnirello, and M. Rossi, “SHARP: Environment and Person Independent Activity Recognition With Commodity IEEE 802.11 Access Points,” IEEE Transactions on Mobile Computing, 2023.','10.1109/TMC.2022.3185681'),
('H. Salehinejad and S. Valaee, “LiteHAR: Lightweight Human Activity Recognition from WiFi Signals with Random Convolution Kernels,” Proc. IEEE ICASSP, 2022.','10.1109/ICASSP43922.2022.9746803'),
('K. Chetty, C. Tang, W. Li, S. Julier, and K. Woodbridge, “Occupancy Detection and People Counting Using WiFi Passive Radar,” Proc. IEEE Radar Conference, 2020.','10.1109/RadarConf2043947.2020.9266493'),
('Z. Chen, Y. Sun, and L. Qu, “Research on Cross-Scene Human Activity Recognition Based on Radar and Wi-Fi Multimodal Fusion,” Electronics, vol. 14, no. 8, art. 1518, 2025.','10.3390/electronics14081518'),
('M. Jung, J. Lee, D. Kim, D. Park, and T. Kim, “Implementing Wi-Fi CSI-based room-level occupancy Estimation: an experimental study in multi-zone residential environments,” Journal of Building Engineering, vol. 111, art. 113155, 2025.','10.1016/j.jobe.2025.113155')]
for i,(text,doi) in enumerate(refs,1):blocks.append(('ref',f'[{i}] {text} DOI: <link href="https://doi.org/{doi}" color="#174c80">{doi}</link>.'))

if VARIANCE3:
    from conference_variance_revision import revise
    blocks=revise(blocks,EXP)
if BALANCED:
    from conference_balanced_revision import revise_balanced
    blocks=revise_balanced(blocks,EXP,OUT)
if HZ100:
    from conference_100hz_revision import revise_100hz
    blocks=revise_100hz(blocks,EXP,OUT)
# Serif two-column conference layout. Authorship is retained from the supplied manuscript.
styles={
'p':ParagraphStyle('body',fontName='Times-Roman',fontSize=10,leading=11.6,alignment=TA_JUSTIFY,spaceAfter=5),
'h':ParagraphStyle('heading',fontName='Times-Roman',fontSize=10,leading=12,alignment=TA_CENTER,spaceBefore=10,spaceAfter=6,keepWithNext=True),
'sub':ParagraphStyle('sub',fontName='Times-Italic',fontSize=10,leading=11.6,spaceBefore=5,spaceAfter=4,keepWithNext=True),
'eq':ParagraphStyle('equation',fontName='Times-Roman',fontSize=9.5,leading=13,alignment=TA_CENTER,spaceBefore=5,spaceAfter=7),
'cap':ParagraphStyle('caption',fontName='Times-Roman',fontSize=8.5,leading=10,spaceBefore=4,spaceAfter=8),
'ref':ParagraphStyle('reference',fontName='Times-Roman',fontSize=8.5,leading=10,spaceAfter=5,wordWrap='CJK')}
for style in styles.values():
    style.fontName={'Times-Roman':'PaperSerif','Times-Italic':'PaperItalic'}[style.fontName]
    style.allowWidows=0;style.allowOrphans=0
def clean(s):return s.replace('–','-').replace('—','- ').replace('−','-').replace('’',"'").replace('“','"').replace('”','"').replace('x̄','x<sub>mean</sub>')
def header(canvas,doc):
    canvas.setFont('Times-Roman',8);canvas.drawCentredString(306,23,str(doc.page))
    if doc.page==1:
        title=Paragraph(TITLE,ParagraphStyle('title',fontName='Times-Roman',fontSize=21,leading=23,alignment=TA_CENTER))
        _,hh=title.wrap(528,90);title.drawOn(canvas,42,755-hh)
        author=Paragraph(AUTHORS,ParagraphStyle('authors',fontName='Times-Roman',fontSize=10.5,leading=13,alignment=TA_CENTER))
        _,ah=author.wrap(528,40);author.drawOn(canvas,42,755-hh-15-ah)
        aff=Paragraph(AFFILS,ParagraphStyle('affiliations',fontName='Times-Roman',fontSize=8.5,leading=10.5,alignment=TA_CENTER))
        _,fh=aff.wrap(540,50);aff.drawOn(canvas,36,755-hh-15-ah-6-fh)
def frames(top):return [Frame(36,40,261,top-40,id='left',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),Frame(315,40,261,top-40,id='right',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]
pdf=ROOT/('output/pdf/multimodal_wireless_sensing_100hz_paper.pdf' if HZ100 else 'output/pdf/multimodal_wireless_sensing_balanced_paper.pdf' if BALANCED else 'output/pdf/multimodal_wireless_sensing_3s_variance_paper.pdf' if VARIANCE3 else 'output/pdf/multimodal_wireless_sensing_conference_paper.pdf')
doc=BaseDocTemplate(str(pdf),pagesize=(612,792),title=TITLE,author='Gad Gad et al.',allowSplitting=True)
doc.addPageTemplates([PageTemplate(id='first',frames=frames(617),onPage=header,autoNextPageTemplate='body'),PageTemplate(id='body',frames=frames(754),onPage=header)])
story=[];md=['# '+TITLE,'\nGad Gad, Iqra Batool, Mostafa Fouda, Shikhar Verma, Zubair Md Fadlullah\n']
for kind,value in blocks:
    if kind in styles:
        story.append(Paragraph(clean(value),styles[kind]));md.append(('## ' if kind=='h' else '### ' if kind=='sub' else '')+value)
    elif kind=='fig':
        name,height,caption=value
        figure_parts=[Image(str(OUT/name),width=252,height=height),Paragraph(clean(caption),styles['cap'])]
        if story and isinstance(story[-1],Paragraph) and story[-1].style in (styles['h'],styles['sub']):
            figure_parts.insert(0,story.pop())
        story.append(KeepTogether(figure_parts));md.append(f'![{caption}]({name})')
    elif kind=='table':
        rows,widths,caption=value
        tab=Table([[clean(str(x)) for x in row] for row in rows],colWidths=widths,repeatRows=1)
        tab.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Times-Roman'),('FONTNAME',(0,0),(-1,0),'Times-Bold'),('FONTSIZE',(0,0),(-1,-1),7.6),('LEADING',(0,0),(-1,-1),9),('ALIGN',(1,0),(-1,-1),'CENTER'),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEABOVE',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.5,colors.black)]))
        table_parts=[Paragraph(clean(caption),styles['cap']),tab,Spacer(1,8)]
        if story and isinstance(story[-1],Paragraph) and story[-1].style is styles['h']:
            table_parts.insert(0,story.pop())
        story.append(KeepTogether(table_parts))
        md.append(caption+'\n\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in [rows[0],['---']*len(rows[0])]+rows[1:]))
doc.build(story)
(OUT/'paper.md').write_text('\n\n'.join(md),encoding='utf8')
(OUT/'content.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2),encoding='utf8')

# Editable IEEEtran source, with the same content and figures as the PDF.
def tex(s):
    s=clean(s)
    s=re.sub(r'<link[^>]*>(.*?)</link>',r'\1',s)
    s=re.sub(r'<sub>(.*?)</sub>',r'@@SUB{\1}',s);s=re.sub(r'<super>(.*?)</super>',r'@@SUP{\1}',s)
    s=s.replace('<b>','@@B{').replace('</b>','}').replace('<br/>',' ')
    s=s.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
    for a,b in {'μ':r'\ensuremath{\mu}','σ':r'\ensuremath{\sigma}','π':r'\ensuremath{\pi}','τ':r'\ensuremath{\tau}','θ':r'\ensuremath{\theta}','×':r'\ensuremath{\times}','½':r'\ensuremath{\frac12}','²':r'\textsuperscript{2}','°':r'\ensuremath{^\circ}','…':r'\ldots{}','x̄':r'\ensuremath{\bar{x}}'}.items():s=s.replace(a,b)
    return s.replace('@@SUB',r'\textsubscript').replace('@@SUP',r'\textsuperscript').replace('@@B',r'\textbf')
lines=[r'\documentclass[conference]{IEEEtran}',r'\usepackage[T1]{fontenc}',r'\usepackage[utf8]{inputenc}',r'\usepackage{amsmath,graphicx,booktabs,hyperref}',r'\title{'+TITLE+'}',r'\author{\IEEEauthorblockN{Gad Gad, Iqra Batool, Mostafa Fouda, Shikhar Verma, Zubair Md Fadlullah}\IEEEauthorblockA{Western University; Idaho State University; Kochi University of Technology\\Author affiliations retained from the supplied manuscript}}',r'\begin{document}',r'\maketitle']
lines[5]=r'\author{\IEEEauthorblockN{Gad Gad\textsuperscript{*}, Iqra Batool\textsuperscript{*}, Mostafa Fouda\textsuperscript{\dag}, Shikhar Verma\textsuperscript{\ddag}, Zubair Md Fadlullah\textsuperscript{*}}\IEEEauthorblockA{\textsuperscript{*}Department of Computer Science, Western University, London, ON, Canada\\\textsuperscript{\dag}Department of Electrical and Computer Engineering, Idaho State University, Pocatello, ID, USA\\\textsuperscript{\ddag}School of Informatics, Kochi University of Technology, Kochi, Japan}}'
for idx,(kind,v) in enumerate(blocks):
    if idx==0:lines.extend([r'\begin{abstract}',tex(v.replace('<b>Abstract—</b>','')),r'\end{abstract}']);continue
    if idx==1:lines.extend([r'\begin{IEEEkeywords}',tex(v.replace('<b>Index Terms—</b>','')),r'\end{IEEEkeywords}']);continue
    if kind=='h':
        if v=='References':lines.append(r'\section*{References}')
        else:lines.append(r'\section{'+tex(v.split('. ',1)[1])+'}')
    elif kind=='sub':lines.append(r'\subsection{'+tex(v.split('. ',1)[1])+'}')
    elif kind=='fig':
        name,height,caption=v;lines.extend([r'\begin{figure}[t]',r'\centering\includegraphics[width=\columnwidth]{'+name+'}',r'\caption{'+tex(caption.split('. ',1)[1])+'}',r'\end{figure}'])
    elif kind=='table':
        rows,widths,cap=v;lines.extend([r'\begin{table}[t]',r'\caption{'+tex(cap.split('. ',1)[1])+'}',r'\centering\scriptsize',r'\begin{tabular}{'+'l'+'c'*(len(rows[0])-1)+'}',r'\toprule'])
        for i,row in enumerate(rows):lines.append(' & '.join(tex(str(c)) for c in row)+r'\\');lines.extend([r'\midrule'] if i==0 else [])
        lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    elif kind=='eq':lines.extend([r'\begin{center}',tex(v),r'\end{center}'])
    else:lines.append(tex(v)+'\n')
lines.append(r'\end{document}');(OUT/'paper.tex').write_text('\n'.join(lines),encoding='utf8')
(OUT/'README.md').write_text('Standalone conference paper source. Authorship and affiliations are retained from the supplied manuscript and should be confirmed before submission. The delivered PDF was rendered with ReportLab and visually checked. paper.tex provides editable IEEEtran source; it has not been compiled in this environment, so pagination may differ. All figures derive from the saved experiment. DOI links in the references were checked against publisher/author records. No new experimental scores were introduced.\n',encoding='utf8')
print(json.dumps({'pdf':str(pdf),'pages':len(PdfReader(pdf).pages),'body_words':sum(len(re.sub('<[^>]+>','',v).split()) for k,v in blocks if isinstance(v,str)),'source':str(OUT)},indent=2))
