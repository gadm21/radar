"""Exploratory minute CSI features from existing radar-aligned caches.

Run: python E2/csi_minutes_plot.py
No cache, labels, trained models, or radar figures are modified.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import common as C
from snr_minutes_plot import SPLITS, SPLIT_XLABELS

OUT = C.OUTPUT_DIR
FIG = OUT / 'figs'
COLORS = dict(empty='#252525', sleep='#d5aa00', present='#d62728')
FEATURES = {
    'amplitude_variance': 'Amplitude variance',
    'rolling_variance': '2 s rolling amplitude variance',
    'relative_variance': '2 s rolling relative variance',
    'motion_power': 'STFT motion power (0.5–3 Hz)',
}


def window_features(x, duration, count):
    """Amplitude input; no per-minute z-score (would erase variance).

    Conservative effective rate cap uses both native count and cached grid.
    A >=6 Hz average count rate is required; raw packet gaps are unavailable.
    """
    if duration < 3 or count / duration < 6 or not np.isfinite(x).all():
        return None
    fs = (len(x) - 1) / duration
    if fs < 6:
        return None
    width = min(len(x), max(2, int(round(2 * fs))))
    rolling = np.lib.stride_tricks.sliding_window_view(x, width, axis=0).var(axis=-1)
    scale = np.maximum(np.mean(x, axis=0), 1.0) ** 2
    # Each STFT segment lasts approximately 3 s; no invented continuity
    # between cached windows. This is a motion proxy, not a breathing estimate.
    nseg = min(len(x), max(4, int(round(3 * fs))))
    f, _, psd = spectrogram(x, fs=fs, window='hann', nperseg=nseg,
                            noverlap=nseg // 2, detrend='constant', axis=0,
                            scaling='density', mode='psd')
    band = (f >= 0.5) & (f <= 3.0)
    power = psd[band].sum(axis=0) * (fs / nseg)
    return [np.median(x.var(axis=0)), np.quantile(np.median(rolling, axis=1), .9),
            np.quantile(np.median(rolling / scale, axis=1), .9),
            np.quantile(np.median(power, axis=0), .9)]


def collect():
    recs, _ = C.discover_recordings()
    rows = []
    for rec in recs:
        if rec.status != 'ok':
            continue
        # Require explicit recognized labels; do not silently use time fallback.
        acts = {s.rsplit('_', 1)[-1] for s in rec.orig_labels if s in C.LABEL_TABLES[rec.source]}
        row = dict(split=rec.source, minute=rec.folder.name, placement=rec.placement,
                   activity=next(iter(acts)) if len(acts) == 1 else 'ambiguous', label=rec.label,
                   valid_windows=0, total_windows=0, status='missing_cache')
        path = C.CACHE_DIR / rec.source / (rec.folder.name + '.npz')
        if len(acts) != 1:
            row['status'] = 'missing_or_ambiguous_explicit_label'
        elif path.exists():
            with np.load(path) as d:
                x, durations, counts = d['csi'], d['win_dur'], d['csi_count']
                row['total_windows'] = len(x)
                vals = [window_features(a.astype(float), float(t), int(n))
                        for a, t, n, valid in zip(x, durations, counts, d['csi_valid']) if valid]
                vals = [v for v in vals if v is not None]
                row['valid_windows'] = len(vals)
                row['status'] = 'ok' if len(vals) >= 3 else 'insufficient_csi'
                if row['status'] == 'ok':
                    row.update(zip(FEATURES, np.quantile(vals, .9, axis=0)))
        rows.append(row)
        if len(rows) % 250 == 0:
            print(f'Processed {len(rows)} minutes', flush=True)
    df = pd.DataFrame(rows)
    radar = pd.read_csv(OUT / 'snr_minutes.csv')[['split', 'minute', 'snr_db']]
    return df.merge(radar, on=['split', 'minute'], how='left', validate='one_to_one')


def plots(df):
    rng = np.random.default_rng(42)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, label=a) for a, c in COLORS.items()]
    for kind in ['scatter', 'boxplot']:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (feature, title) in zip(axes.flat, FEATURES.items()):
            for si, split in enumerate(SPLITS):
                for ai, (act, color) in enumerate(COLORS.items()):
                    v = df.loc[(df.split == split) & (df.activity == act), feature].dropna()
                    if not len(v):
                        continue
                    if kind == 'scatter':
                        ax.scatter(si + rng.uniform(-.28, .28, len(v)), np.maximum(v, 1e-12), s=10, alpha=.4, c=color)
                    else:
                        bp = ax.boxplot([np.maximum(v, 1e-12)], positions=[si + (ai - 1)*.22], widths=.19,
                                        patch_artist=True, manage_ticks=False, showfliers=False,
                                        medianprops={'color': 'white'})
                        bp['boxes'][0].set_facecolor(color)
                        ax.text(si + (ai-1)*.22, .02, str(len(v)), transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
            ax.set_yscale('log')
            ax.set_xticks(range(4), [SPLIT_XLABELS[s] for s in SPLITS])
            ax.set_title(title)
            ax.grid(axis='y', alpha=.2)
            ax.set_ylabel('Minute 90th percentile (log axis)')
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .955), ncol=3)
        fig.suptitle('CSI per minute' + (' — boxes: quartiles, labels: counts; outliers hidden' if kind == 'boxplot' else ''), y=.995)
        fig.tight_layout(rect=(0, 0, 1, .90))
        fig.savefig(FIG / f'csi_minutes_{kind}.png', dpi=160)
        plt.close(fig)
    for name, x, y in [('csi_minutes_2d', 'relative_variance', 'motion_power'),
                        ('radar_csi_minutes_2d', 'snr_db', 'relative_variance')]:
        fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharex=True, sharey=True)
        for ax, split in zip(axes, SPLITS):
            for act, color in COLORS.items():
                g = df[(df.split == split) & (df.activity == act)].dropna(subset=[x, y])
                ax.scatter(g[x].clip(lower=1e-12), g[y].clip(lower=1e-12), c=color, s=12, alpha=.5)
            ax.set_title(SPLIT_XLABELS[split])
            ax.set_xlabel(FEATURES.get(x, 'Radar SNR (dB)'))
            if x != 'snr_db':
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(alpha=.15)
        axes[0].set_ylabel(FEATURES[y])
        fig.legend(handles=handles, loc='upper center', ncol=3)
        fig.tight_layout(rect=(0, 0, 1, .90))
        fig.savefig(FIG / (name + '.png'), dpi=160)
        plt.close(fig)


def evaluate(df):
    auc = []
    for (split, placement), g in df.groupby(['split', 'placement']):
        for feature in FEATURES:
            for neg, pos in [('empty', 'occupied'), ('empty', 'sleep'), ('empty', 'present'), ('sleep', 'present')]:
                h = g if pos == 'occupied' else g[g.activity.isin([neg, pos])]
                y = h.label if pos == 'occupied' else (h.activity == pos).astype(int)
                if y.nunique() == 2:
                    auc.append(dict(split=split, placement=placement, feature=feature, comparison=f'{neg}/{pos}',
                                    n=len(h), auc=roc_auc_score(y, h[feature])))
    pd.DataFrame(auc).to_csv(OUT / 'csi_feature_auc.csv', index=False)
    paired = df.dropna(subset=['snr_db']).copy()
    models = {'radar': ['snr_db'], 'csi': list(FEATURES), 'fusion': ['snr_db'] + list(FEATURES)}
    metrics = []
    for name, features in models.items():
        X = paired[features].copy()
        for f in features:
            if f != 'snr_db':
                X[f] = np.log10(X[f].clip(lower=1e-12))
        train = paired.split.isin(['calibration_minutes', 'train'])
        val = paired.split == 'validation_minutes'
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', C=1, max_iter=2000))
        model.fit(X[train], paired.loc[train, 'label'])
        probs = model.predict_proba(X)[:, 1]
        pval = probs[val]
        candidates = np.r_[0, (np.unique(pval)[:-1] + np.unique(pval)[1:])/2, 1]
        scores = [balanced_accuracy_score(paired.loc[val, 'label'], pval >= t) for t in candidates]
        threshold = float(np.median(candidates[np.isclose(scores, max(scores))]))
        for split in ['validation_minutes', 'test_minutes']:
            mask = paired.split == split
            y = paired.loc[mask, 'label']
            pred = probs[mask] >= threshold
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            metrics.append(dict(model=name, split=split, n=len(y), threshold=threshold,
                                auc=roc_auc_score(y, probs[mask]), balanced_accuracy=balanced_accuracy_score(y, pred),
                                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
    pd.DataFrame(metrics).to_csv(OUT / 'csi_fusion_baseline.csv', index=False)
    return metrics


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / 'csi_minutes.csv', index=False)
    coverage = df.groupby(['split', 'activity', 'status']).size().rename('minutes')
    coverage.to_csv(OUT / 'csi_minutes_coverage.csv')
    usable = df[df.status == 'ok'].copy()
    plots(usable)
    metrics = evaluate(usable)
    report = '''# CSI minute feature exploration

The existing radar figures contain substantial overlap around 5 dB. High radar
values distinguish some occupied minutes, but a low value does not imply empty.
The plots alone cannot establish whether overlap comes from field of view,
stillness, multipath, or another cause. The saved radar images use old split
display names (`train`/`train2`); these CSI plots use current directory names.

## Reproduction and outputs

Run `python E2/csi_minutes_plot.py` with E2/requirements.txt installed.
`csi_minutes.csv` includes features, exclusions, and an exact split/minute join
to the existing SNR CSV. `csi_minutes_coverage.csv` lists exclusions.
Figures: `figs/csi_minutes_scatter.png`, `figs/csi_minutes_boxplot.png`,
`figs/csi_minutes_2d.png`, and `figs/radar_csi_minutes_2d.png`.
Boxplot numbers are sample counts; outliers are hidden only in the boxplot.

## Feature definitions

Input: cached **linear CSI amplitude**, 52 subcarriers, 128 interpolated time
samples per radar-aligned window. This cache contains amplitude, despite the
README describing log1p in one place. One receiver is selected by the existing
preprocessor; receiver selection is not optimized using labels here.

- Amplitude variance: median across subcarriers of temporal window variance.
- Rolling variance: full two-second windows, variance per subcarrier, median
  across subcarriers, then 90th percentile over rolling windows.
- Relative variance: rolling variance divided by squared mean amplitude of
  each subcarrier (mean floored at 1), then the same aggregation.
- STFT motion power: approximately three-second Hann segments, 50% overlap,
  mean removed per segment, PSD summed over 0.5–3 Hz times frequency spacing;
  median across subcarriers, then 90th percentile across segments.

Each minute uses the 90th percentile across its eligible cached windows.
This emphasizes occasional activity without relying on a single maximum.
No windows cross minute boundaries; no spectra bridge gaps between windows.
Require >=3 eligible windows, each >=3 s, finite CSI, native average sample
rate >=6 Hz, cached grid rate >=6 Hz, and csi_valid. Excluded data is missing,
never assigned a zero-activity score. Native packet gaps are not available in
the cache: average-rate screening does not guarantee uniformly sampled data.
Interpolation and varying packet rates can still influence these features.

## Evaluation interpretation

`csi_feature_auc.csv` reports pairwise AUC separately by split and placement.
AUC assumes larger feature values indicate the second class; below 0.5 means
the direction is reversed, not automatically a useless feature. Missing class
pairs have no AUC. Calibration contains only sleep in the available plotted
data, so it cannot establish sleep-versus-empty separation by itself.

The baseline below uses identical paired minutes for radar, CSI, and fusion.
Log-transform CSI features; standardization and balanced logistic regression
(fixed C=1) are fitted on calibration + train. Each threshold maximizes
validation balanced accuracy; test does not select features, model settings,
or thresholds. These are exploratory diagnostics on previously examined test
data, not a fresh untouched test or proof of deployable performance. Adjacent
minutes and class/time/placement confounding limit independence. Test labels
were assigned by a time boundary in the existing pipeline; this script requires
explicit recognized manifest labels, but that does not independently verify them.

Sleep is occupied in binary metrics. Sleep/present cannot be evaluated on
validation or test because those splits have no sleep labels. These short
STFT windows are motion descriptors, **not breathing measurements**. Longer
contiguous raw CSI segments with packet-gap checks would be needed to study
slow periodic structure. The median across subcarriers may suppress a weak
signal confined to only a few subcarriers.

## Paired binary baseline results

| Model | Split | Minutes | ROC AUC | Balanced accuracy | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for r in metrics:
        report += f"| {r['model']} | {r['split']} | {r['n']} | {r['auc']:.3f} | {r['balanced_accuracy']:.3f} | {r['tn']} | {r['fp']} | {r['fn']} | {r['tp']} |\n"
    report += '''
## Findings in this run

Relative rolling variance is the clearest validation descriptor: the AUC for
*lower* variance indicating occupancy is 0.857, falling to 0.688 on test.
That orientation is inconsistent with train's weak positive association
(empty/present AUC 0.558 for higher variance). Raw motion-band power also
changes direction between validation and test. Thus a universal "more CSI
variation means occupied" rule is not supported here.

The fitted fusion model raises test ranking AUC slightly (0.636 to 0.669),
but its validation-selected threshold reduces test balanced accuracy
(0.598 to 0.591) and misses 109/136 occupied minutes versus radar's 88/136.
This baseline does not demonstrate a reliable occupancy improvement.
These conclusions concern these four summarized features and this simple
baseline; they do not establish that CSI cannot help.

There are only 15 usable sleep minutes in train, and calibration is sleep-only.
Three-class separation is not established. For the next data collection,
alternate empty, still occupied, and moving occupied periods within each
placement/session, and annotate radar-field-of-view status. Analyze contiguous
raw CSI and receiver-specific quality before interpreting spectral peaks as
breathing. Compare matched low-radar-SNR minutes across those conditions.
'''
    report += '\n## Coverage\n\n```text\n' + coverage.to_string() + '\n```\n'
    (OUT / 'CSI_FEATURE_REPORT.md').write_text(report, encoding='utf-8')
    print(coverage.to_string())
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
