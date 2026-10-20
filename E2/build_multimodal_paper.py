"""Append the measured multimodal experiment to the supplied paper; preserve source."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.paper_packages'))
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY,TA_CENTER
from pypdf import PdfReader,PdfWriter
import numpy as np
OUT=ROOT/'E2/outputs/paper_multimodal_20s'
DEST=ROOT/'output/pdf'; DEST.mkdir(parents=True,exist_ok=True)
TMP=ROOT/'tmp/pdfs'; TMP.mkdir(parents=True,exist_ok=True)
SOURCE=Path('C:/Users/ggad/Downloads/IEEE_2026__wifi_sensing_.pdf')
df=pd.read_csv(OUT/'all_minutes.csv');paired=pd.read_csv(OUT/'paired_minutes.csv');results=pd.read_csv(OUT/'results.csv')
winner=json.loads((OUT/'selection.json').read_text());raw=json.loads((OUT/'raw_csi_auc.json').read_text())
test=results[(results.features==winner['features'])&(results.model==winner['model'])&(results.split=='test_minutes')].iloc[0]
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyPaper',fontName='Times-Roman',fontSize=10,leading=13,alignment=TA_JUSTIFY,spaceAfter=7))
styles.add(ParagraphStyle(name='TitlePaper',fontName='Times-Bold',fontSize=17,leading=20,alignment=TA_CENTER,spaceAfter=12))
styles.add(ParagraphStyle(name='HeadingPaper',fontName='Times-Bold',fontSize=11,leading=14,spaceBefore=10,spaceAfter=6))
styles.add(ParagraphStyle(name='CaptionPaper',fontName='Times-Roman',fontSize=9,leading=11,spaceAfter=9))
story=[];md=[]
def p(text,style='BodyPaper'):
    story.append(Paragraph(text,styles[style])); md.append(text.replace('<b>','').replace('</b>',''))
def heading(text):p(text,'HeadingPaper')
def table(rows,widths):
    tab=Table(rows,colWidths=widths,repeatRows=1,hAlign='CENTER')
    tab.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Times-Roman'),('FONTNAME',(0,0),(-1,0),'Times-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),('LEADING',(0,0),(-1,-1),11),('LINEABOVE',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('ALIGN',(1,1),(-1,-1),'CENTER')]))
    story.append(tab);story.append(Spacer(1,8))
    md.append('\n'.join(' | '.join(map(str,row)) for row in rows))
def fig(file,w,h,caption):
    story.append(Image(str(OUT/file),width=w,height=h));p(caption,'CaptionPaper');md.append(f'Figure: {file}')
def page():story.append(PageBreak())

p('Appendix A. Multimodal Room Occupancy Detection Using CSI and Radar','TitlePaper')
p('Experimental extension to the supplied WiFi sensing manuscript. Analysis date: 28 September 2026. The six original manuscript pages are preserved; this appendix reports a separate August-September 2026 room-occupancy experiment.','CaptionPaper')
heading('A. Motivation and task')
p('The task is to detect occupancy in the room containing the collection devices. Radar can respond weakly when a person is outside its field of view or remains nearly motionless. WiFi channel state information (CSI) measures changes in propagation paths and may provide complementary evidence. These explanations motivate fusion, but the current labels do not identify field-of-view failures or verify breathing; neither mechanism is established by this experiment.')
p('We evaluate the user-selected 20-second, two-dimensional PCA trajectory-length feature with radar signal-to-background ratio (SNR proxy), range-Doppler (RD), range-azimuth (RA), range-elevation (RE), and an XY energy projection. Sleep and present are mapped to occupied. Validation and test contain no sleep labels, so the evaluated endpoint is binary occupancy, not three-class activity recognition.')
heading('B. Dataset, labels, and usable coverage')
p('The recordings are organized into nominal one-minute folders with scheduled timestamps, manifests, and raw sensor captures. CSI contains 52 retained complex subcarriers after null/guard/DC exclusion. Radar payloads decode to 64 chirps by 128 ADC samples by three receiver channels, with 12-bit samples. Some captures are incomplete or extend past a nominal boundary; this analysis uses timestamps within the scheduled interval [0, 60) seconds. Manifests identify ESP32 CSI and DreamHAT radar acquisition. Exact RF configuration, antenna calibration, participant count, and room dimensions are not verified here.')
rows=[['Partition','Raw labeled E / S / P','20 s CSI E / S / P','Paired E / S / P']]
for split in ['calibration_minutes','train','validation_minutes','test_minutes']:
    def counts(d):return ' / '.join(str(int((d.activity==a).sum())) for a in ['empty','sleep','present'])
    a=df[df.split==split];b=a[a.csi.notna()];c=paired[paired.split==split]
    rows.append([split.replace('_minutes',''),counts(a),counts(b),counts(c) if split!='calibration_minutes' else 'not evaluated'])
table(rows,[85,139,139,139])
p('Table A1. E = empty, S = sleep, P = present. Raw labeled counts describe discovered recordings with a single recognized activity; usable CSI requires a valid 20-second path. Paired counts additionally require radar features and the existing SNR proxy. Calibration contributes to the unsupervised PCA basis only.','CaptionPaper')
periods=[]
for split in ['calibration_minutes','train','validation_minutes','test_minutes']:
    names=df.loc[df.split==split,'minute'].astype(str)
    periods.append(split.replace('_minutes','')+': '+pd.to_datetime(names.min(),format='%Y%m%d_%H%M').strftime('%d %b')+' to '+pd.to_datetime(names.max(),format='%Y%m%d_%H%M').strftime('%d %b %Y'))
p('Observed recording dates are '+ '; '.join(periods)+'. Modalities are paired by recording identity and host timestamps, not by verified hardware-level synchronization.','CaptionPaper')
p('Recognized manifest labels are used; automatic occupancy annotations are ignored. The repository documents that Pi test labels were assigned from the recording schedule, with 01:10 on 17 September 2026 as the empty transition. These labels are not independently frame-verified. Chronological train, validation, and test partitions are preserved without random window splitting. Adjacent minutes remain correlated, and this collection does not establish cross-room or cross-participant generalization.')
p('Reproducibility: the accompanying experiment script fixes preprocessing and model settings, saves per-minute coverage and features, preserves PCA parameters, exports all predictions and confusion counts, and records the validation-selected configuration. The supplied PDF is preserved as the first six pages; this appendix is an experimental extension rather than an alteration of the original manuscript\'s claims.','CaptionPaper')
page()
heading('C. CSI trajectory representation')
nfit=int(np.load(OUT/'pca.npz')['fit_dots'])
p(f'Where multiple CSI receivers exist, the existing loader selects the receiver with most packet coverage; receivers are not fused. For each second, concatenate the mean and population standard deviation of the 52 CSI amplitudes into a 104-dimensional vector. Duplicate timestamps are removed. A second is usable only if it contains at least five packets, spans at least 0.5 seconds, and has no inter-packet or boundary gap exceeding 0.5 seconds. A fixed, centered PCA basis is fitted to {nfit:,} valid calibration and training vectors, without standardization or whitening; only its first two coordinates are used for distance.')
p('For projected dots z(t) in two dimensions, a 20-second path contains 21 consecutive valid dots and 20 Euclidean steps: <b>L(s) = sum from j=0 to 19 of ||z(s+j+1) - z(s+j)||<sub>2</sub>.</b> The minute feature is the median of all valid paths with start seconds 0 through 39. No gap interpolation is used. The classifier receives log(1 + L). Euclidean distance assumes a fixed shared PCA basis; recomputing a separate basis for each dot would make these displacements incomparable.')
heading('D. Radar representations and fusion')
p('Raw radar frames within the same minute are decoded in acquisition order. Each chirp has its sample mean removed, receives a Hann range window, and undergoes a 128-point FFT, retaining 64 range bins. RD applies a Hann window and 64-point FFT along chirps, followed by Doppler centering and mean power across receivers. The first two range bins are suppressed. Static slow-time components are retained, so these features are not purely movement detectors.')
p('Following the local radar implementation\'s documented L-shaped receiver layout, two-element conventional beamforming uses RX1/RX3 for azimuth and RX2/RX3 for elevation. Twenty-five steering angles cover -60 to +60 degrees, assuming half-wavelength spacing. Beam power is averaged over chirps. These are reconstructed RA and RE energy maps; two-element angular resolution is limited, and unverified antenna phase calibration can bias the angular responses. The older cached three-channel angle FFT is not substituted for RE.')
p('XY is a deterministic projection of RA, not an additional sensing modality. Normalized range r maps to x = r sin(angle), y = r cos(angle), then to a 24 by 24 sensor-coordinate grid using maximum energy for collisions. No meter-scale room registration is claimed. All maps use log(1 + power). Each map contributes five minute descriptors: spatial mean, spatial 90th percentile, and maximum of the temporal mean map; the spatial 90th percentile of temporal standard deviation; and mean absolute frame-to-frame change. Irregular frame timing can affect the latter descriptor.')
p('The existing SNR feature is reused unchanged: the maximum framewise 10 log10 ratio of peak noncentral cached RD amplitude to median background amplitude. It is a historical discrimination proxy rather than a calibrated physical power SNR. Its original cached frame coverage is retained and can differ from the [0, 60) support used for the newly reconstructed maps; fusion is matched by minute identity, not by an identical frame subset. CSI + SNR has two features; adding one map gives seven; all-radar fusion has 21; CSI + all radar has 22. Radar descriptors aggregate the full available minute, whereas CSI uses the median of rolling 20-second paths. The evaluated decision interval is therefore a minute, not a demonstrated 20-second online detector.')
fig('maps.png',495,129,'Figure A1. Representative training-minute mean maps, shown in native array orientation. RD rows are Doppler bins and columns range; RA/RE rows are range and columns steering angle; XY rows are forward and columns lateral position. Color shows log power independently per panel; no calibrated physical axes are implied.')
page()
heading('E. Evaluation protocol and measured results')
p('All ablations use the same fully paired cohort. Classifiers are a balanced logistic regression (LR; C = 0.3) and a balanced random forest (RF; 200 trees, maximum depth 4, minimum leaf size 5, seed 42). Median imputation and standardization are fitted on training minutes only; the complete-case primary cohort requires no imputed values. Calibration does not supply supervised examples. For each candidate, validation selects the probability threshold maximizing balanced accuracy; ties prefer the threshold closest to 0.5. The overall candidate is selected by validation balanced accuracy, then AUC, then a deterministic name tie-break. Test data do not choose features, model, or threshold.')
rawv=next(x for x in raw if x['split']=='validation_minutes');rawt=next(x for x in raw if x['split']=='test_minutes')
p(f'The requested CSI configuration is reproduced: raw path-length AUC is {rawv["auc"]:.3f} on validation (n = {rawv["n"]}) and {rawt["auc"]:.3f} on test (n = {rawt["n"]}). These are ranking AUCs, not classification accuracies. Balanced accuracy is the mean of empty specificity and occupied sensitivity; it is emphasized because the paired test set is strongly imbalanced.')
chosen=results[results.split=='validation_minutes'].sort_values(['balanced_accuracy','auc','model'],ascending=[False,False,True]).drop_duplicates('features')
order=['CSI','SNR','RD','RA','RE','XY','CSI+SNR','CSI+SNR+RD','CSI+SNR+RA','CSI+SNR+RE','CSI+SNR+XY','Radar all','CSI+radar all']
rows=[['Features','Model','Val. BA','Val. AUC','Test BA','Test AUC']]
for name in order:
    v=chosen[chosen.features==name].iloc[0];t=results[(results.features==name)&(results.model==v.model)&(results.split=='test_minutes')].iloc[0]
    rows.append([name,v.model,f'{v.balanced_accuracy:.3f}',f'{v.auc:.3f}',f'{t.balanced_accuracy:.3f}',f'{t.auc:.3f}'])
table(rows,[157,45,75,75,75,75])
p('Table A2. Matched-cohort ablation. Each row displays the classifier selected for that feature set using validation only. All 26 candidates and their frozen-threshold confusion counts are available in the accompanying results CSV. BA denotes balanced accuracy.','CaptionPaper')
p(f'The validation-selected candidate is <b>{winner["features"]} ({winner["model"]})</b>, with validation balanced accuracy {winner["balanced_accuracy"]:.3f} and AUC {winner["auc"]:.3f}. Its frozen threshold is {winner["threshold"]:.4f}. On test, balanced accuracy is <b>{test.balanced_accuracy:.3f}</b>, AUC {test.auc:.3f}, ordinary accuracy {test.accuracy:.3f}, and occupied-class F1 {test.f1:.3f}. The confusion counts are TN = {int(test.tn)}, FP = {int(test.fp)}, FN = {int(test.fn)}, TP = {int(test.tp)}.')
p('CSI + SNR (LR) gives 0.771 test balanced accuracy and 0.682 AUC, matching CSI alone in thresholded decisions without improving ranking. CSI + SNR + RD (RF) improves test AUC to 0.784 but gives 0.760 balanced accuracy at its validation-selected threshold. CSI + all radar (RF) gives 0.708 balanced accuracy and 0.755 AUC. Thus none of the evaluated candidates establishes a test balanced accuracy above 80%, and adding every map does not improve the selected classifier. RA and XY are dependent representations, while all radar features originate from one sensor.')
p('Map distributions shift substantially between partitions: median minute-mean RA log power is 6.167 in training, 9.880 in validation, and 9.884 in test. This observed shift helps explain why training-based scaling and some LR outputs transfer poorly; its hardware/environmental cause is not established. It should be investigated before interpreting the maps as invariant occupancy signatures.','CaptionPaper')
page()
heading('F. Distributions, limitations, and interpretation')
fig('csi_snr.png',500,146,'Figure A2. CSI path length versus radar SNR proxy on the matched training, validation, and test cohorts. Sleep is mapped to occupied for modeling. The vertical axis is logarithmic.')
fig('ablation.png',500,200,'Figure A3. Validation and test balanced accuracy for each feature set, using its validation-selected classifier and validation-selected threshold. Ordering is based on validation only.')
p(f'The paired test contains only {int(test["empty"])} empty and {int(test.occupied)} occupied minutes. One additional empty error changes specificity by {100/int(test["empty"]):.1f} percentage points and balanced accuracy by {50/int(test["empty"]):.1f} points. Minutes are temporally dependent, so an ordinary independent-minute confidence interval would overstate certainty. Strict CSI coverage excludes many recordings, and these complete-case results must not be presented as accuracy over the entire dataset. Missing-modality fallback has not been evaluated in this experiment.')
alltest=df[df.split=='test_minutes']; ne=int((alltest.activity=='empty').sum()); no=int(alltest.activity.isin(['sleep','present']).sum())
p(f'Test retention is {100*test.n/len(alltest):.1f}% overall: {int(test["empty"])}/{ne} empty minutes ({100*test["empty"]/ne:.1f}%) versus {int(test.occupied)}/{no} occupied minutes ({100*test.occupied/no:.1f}%). This unequal coverage creates a selection-bias concern; a deployment evaluation must report both prediction quality and the fraction of time for which the system can produce a decision.')
p(f'An always-occupied rule would score {100*test.occupied/test.n:.1f}% ordinary accuracy on this paired test, but only 50% balanced accuracy. Therefore, an 80% ordinary-accuracy target would not demonstrate useful detection of empty-room periods.','CaptionPaper')
p('Validation has been used for method exploration in earlier work and for selection among the 26 candidates here; its apparent performance is optimistic. The current test partition also appeared in earlier project analyses, although it was not used for selection in this run. A new untouched acquisition session is required for confirmatory evaluation. Recommended collection factors are empty-room duration, stationary and sleeping occupants, motion inside and outside the radar view, and motion in adjacent rooms; the last condition is necessary to establish room-specific occupancy rather than nearby motion sensing.')
p('The classifiers are lightweight, but no end-to-end latency, energy, or matched deep-learning baseline was benchmarked here. Consequently, this appendix does not claim a measured speedup over deep learning. A fair comparison must include CSI/PCA/map preprocessing, model inference, hardware, batch size, observation-window latency, and accuracy at matched coverage. The original manuscript\'s CSI preprocessing timing results concern a different experiment and do not establish a fusion speed advantage.')

def footer(canvas,doc):
    canvas.setFont('Times-Roman',8);canvas.drawString(42,25,'Multimodal occupancy experimental extension | 28 September 2026');canvas.drawRightString(570,25,str(doc.page+6))
appendix=TMP/'multimodal_appendix.pdf'
SimpleDocTemplate(str(appendix),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=34,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
writer=PdfWriter();writer.clone_document_from_reader(PdfReader(SOURCE));writer.append(str(appendix))
final=DEST/'IEEE_2026_wifi_sensing_multimodal_report.pdf'
with final.open('wb') as f:writer.write(f)
(DEST/'multimodal_report.md').write_text('\n\n'.join(md),encoding='utf8')
audit={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'source_pages':len(PdfReader(SOURCE).pages),'appendix_pages':len(PdfReader(appendix).pages),'output_pages':len(PdfReader(final).pages),'selected_features':winner['features'],'selected_model':winner['model']}
(OUT/'document_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2));print(final)
