"""Describe guaranteed-rate preprocessing and its measured interpolation burden."""
import json
import pandas as pd
import matplotlib.pyplot as plt

def revise_100hz(blocks,exp,out):
    blocks=list(blocks);meta=pd.read_csv(exp/'all_minutes.csv')
    method=json.loads((exp/'method.json').read_text())
    nfit=method['pca_fit_dots']
    def replace(prefix,text):
        ids=[i for i,(k,v) in enumerate(blocks) if isinstance(v,str) and v.startswith(prefix)]
        assert len(ids)==1,(prefix,ids)
        i=ids[0];blocks[i]=(blocks[i][0],text)
    replace('For valid second s,','Before any one-second statistics or PCA, the selected CSI amplitude stream is resampled to 100 Hz using the guaranteed-sampling-rate routine from the companion WiFiSensing project. The original routine is executed unchanged: target times begin at the first observed packet with 10 ms spacing; samples in each centered time bin are averaged, and empty bins are linearly interpolated from populated bins. Endpoint assignment follows the source routine. Host timestamps are expressed in relative microseconds for its input. Magnitude is resampled directly; the algorithm does not reconstruct magnitude from separately averaged real and imaginary components.')
    i=next(i for i,(k,v) in enumerate(blocks) if k=='eq' and v.startswith('x<sub>s</sub>'))
    blocks[i:i]=[('p',f'Each accepted one-second bin contains exactly 100 resampled amplitude vectors. Subcarrier-wise mean and population standard deviation are concatenated into the 104-dimensional x<sub>s</sub>. A new centered, unwhitened two-component PCA is fitted on {nfit:,} accepted calibration-and-training vectors; the earlier PCA basis is not reused. Validation and test do not fit or update this basis. The downstream three-dot variance and minute median remain unchanged. Thus 100 Hz describes the input amplitude grid, while PCA dots remain one second apart.'),('p','Uniform resampling does not guarantee that 100 original packets were received each second. Missing bins are reconstructed values, and interpolation can smooth motion or bridge gaps. Original packet-quality checks are retained for accepting one-second summaries; no summary is admitted solely because interpolation supplied values. Partial edge seconds with fewer than 100 resampled values are omitted. Figure 6 reports the fraction of interpolated samples actually used by accepted summaries.')]
    replace('Using a fixed basis makes','Using a fixed basis makes coordinate variances comparable across windows and partitions. Refitting axes independently per window would change the representation along with the signal. PCA maximizes variance in its calibration-and-training fit population and does not guarantee occupancy relevance. The newly fitted basis is frozen throughout validation and test.')
    replace('A CSI second is valid only','Original observations in an accepted CSI second must contain at least five packets, span at least 0.5 seconds, and have no packet or boundary gap greater than 0.5 seconds. After 100 Hz resampling, the second must also contain exactly 100 amplitude vectors. Three consecutive accepted seconds define a variance window. All 192 validation and 249 test minutes retain the required CSI and radar features. Figure 4 reports pre-balancing availability; six of 274 training minutes lack a required radar/SNR feature, leaving 268 fully paired training minutes.')
    fig,ax=plt.subplots(figsize=(3.5,2.0))
    fractions=[]
    for split in ['validation_minutes','test_minutes']:
        r=meta[meta.split==split]
        fractions.append(100*r.used_interpolated_samples/r.used_resampled_samples)
    ax.boxplot(fractions,tick_labels=['Validation','Test'],widths=.45,flierprops={'markersize':2})
    ax.set_ylabel('Interpolated samples (%)',fontsize=8);ax.tick_params(labelsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'interpolation_coverage.png',dpi=300);plt.close(fig)
    stats=[]
    for split,title in [('validation_minutes','validation'),('test_minutes','test')]:
        r=meta[meta.split==split]
        pct=100*r.used_interpolated_samples/r.used_resampled_samples
        stats.append(f'{title}: median {pct.median():.1f}% (range {pct.min():.1f}-{pct.max():.1f}%)')
    result=[]
    for kind,value in blocks:
        if isinstance(value,str):
            value=value.replace('The CSI feature is the sum','CSI amplitudes are first resampled to a uniform 100 Hz grid by bin averaging and linear interpolation. The CSI feature is the sum')
            value=value.replace('All test minutes are now retained by the CSI feature','All test minutes retain a CSI feature after resampling and quality checks')
        result.append((kind,value))
        if kind=='fig' and value[0]=='temporal_coverage.png':
            result.extend([('fig',('interpolation_coverage.png',144,'Fig. 6. Interpolated fraction of the 100 Hz samples used in accepted one-second summaries, per minute in the original evaluation collections. Reconstructed samples are not independent observations.')),('p','Interpolation among samples used by the PCA summaries is '+ '; '.join(stats)+'. These fractions measure empty target bins filled by the upstream routine. They are reported alongside original packet counts to distinguish uniform processing rate from acquisition quality.')])
    return result
