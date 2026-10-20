"""CSI Studio-style one-second mean/std dots, fixed PCA, temporal path length.

python E2/csi_pca_path.py [--reuse-snapshots] [--workers 4]
Reads raw CSI; does not concatenate interpolated radar-window caches.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
import common as C
from snr_minutes_plot import SPLITS, SPLIT_XLABELS

OUT = C.OUTPUT_DIR
FIG = OUT / 'figs'
COLORS = {'empty': '#252525', 'sleep': '#cca600', 'present': '#d62728'}
SPANS = (10, 20, 30)  # elapsed seconds, hence span+1 dots and span edges
PRIMARY = 'path3d_20s_median'


def extract(rec):
    row = dict(split=rec.source, minute=rec.folder.name, placement=rec.placement,
               label=rec.label, activity='', status='ok', valid_seconds=0)
    acts = {s.rsplit('_', 1)[-1] for s in rec.orig_labels if s in C.LABEL_TABLES[rec.source]}
    dots = np.full((60, 104), np.nan, dtype=np.float32)
    counts = np.zeros(60, dtype=np.int32)
    if rec.status != 'ok' or len(acts) != 1:
        row['status'] = 'invalid_explicit_label'
        return row, dots, counts
    row['activity'] = acts.pop()
    try:
        csi, ts = C.recording_csi(rec)
        if csi is None:
            row['status'] = 'no_csi'
            return row, dots, counts
        # Fixed minute boundaries; finite timestamps and samples only.
        rel = ts - rec.start_ts
        keep = (rel >= 0) & (rel < 60) & np.isfinite(csi).all(axis=1)
        rel, amp = rel[keep], np.abs(csi[keep]).astype(np.float64)
        # Repeated host timestamps must not inflate sample counts.
        rel, unique = np.unique(rel, return_index=True)
        amp = amp[unique]
        bins = np.floor(rel).astype(int)
        for second in range(60):
            sel = bins == second
            t, a = rel[sel], amp[sel]
            counts[second] = len(t)
            if len(t) < 5:
                continue
            gaps = np.diff(np.r_[second, t, second + 1])
            if t[-1] - t[0] < .5 or gaps.max() > .5:
                continue
            dots[second] = np.r_[a.mean(axis=0), a.std(axis=0)]
        valid = np.isfinite(dots).all(axis=1)
        row['valid_seconds'] = int(valid.sum())
        row['packets'] = int(counts.sum())
        row['median_packets_per_valid_second'] = float(np.median(counts[valid])) if valid.any() else np.nan
        row['status'] = 'ok' if valid.any() else 'no_valid_seconds'
    except Exception as exc:
        row['status'] = 'read_error'
        row['error'] = str(exc)
    return row, dots, counts


def load_snapshots(reuse, workers):
    cache = OUT / 'csi_pca_snapshots.npz'
    meta = OUT / 'csi_pca_coverage.csv'
    if reuse:
        with np.load(cache) as d:
            return pd.read_csv(meta), d['snapshots'], d['counts']
    recs, _ = C.discover_recordings()
    rows, dots, counts = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (row, x, n) in enumerate(pool.map(extract, recs)):
            rows.append(row)
            dots.append(x)
            counts.append(n)
            if (i + 1) % 100 == 0:
                print(f'Raw CSI: {i+1}/{len(recs)} minutes', flush=True)
    df = pd.DataFrame(rows)
    X, counts = np.array(dots), np.array(counts)
    df.to_csv(meta, index=False)
    np.savez_compressed(cache, snapshots=X, counts=counts)
    return df, X, counts


def path_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def project_and_measure(df, X):
    train = df.split.isin(['calibration_minutes', 'train']).to_numpy()
    valid = np.isfinite(X).all(axis=2)
    fit = X[train][valid[train]]
    pca = PCA(n_components=3, svd_solver='full', whiten=False).fit(fit)
    Z = np.full((len(X), 60, 3), np.nan)
    Z[valid] = pca.transform(X[valid])
    np.savez_compressed(OUT / 'csi_pca_model.npz', mean=pca.mean_, components=pca.components_,
                        explained_variance_ratio=pca.explained_variance_ratio_)
    details = []
    for i, row in df.iterrows():
        for span in SPANS:
            paths = {2: [], 3: []}
            for start in range(60 - span):
                pts = Z[i, start:start + span + 1]
                if not np.isfinite(pts).all():
                    continue
                record = dict(split=row['split'], minute=row['minute'], activity=row['activity'],
                              start_second=start, elapsed_seconds=span)
                for dim in [2, 3]:
                    length = path_length(pts[:, :dim])
                    paths[dim].append(length)
                    record[f'path{dim}d'] = length
                details.append(record)
            df.loc[i, f'n_paths_{span}s'] = len(paths[3])
            for dim in [2, 3]:
                for suffix, q in [('median', .5), ('p90', .9)]:
                    df.loc[i, f'path{dim}d_{span}s_{suffix}'] = np.quantile(paths[dim], q) if paths[dim] else np.nan
    pd.DataFrame(details).to_csv(OUT / 'csi_pca_path_windows.csv', index=False)
    df = df.merge(pd.read_csv(OUT / 'snr_minutes.csv')[['split', 'minute', 'snr_db']],
                  on=['split', 'minute'], how='left', validate='one_to_one')
    df.to_csv(OUT / 'csi_pca_path_minutes.csv', index=False)
    np.savez_compressed(OUT / 'csi_pca_projections.npz', projections=Z)
    return df, Z, pca, len(fit)


def distributions(df):
    handles = [plt.Line2D([], [], color=c, marker='o', linestyle='', label=a) for a,c in COLORS.items()]
    rng = np.random.default_rng(42)
    for kind in ['scatter', 'boxplot']:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for di, dim in enumerate([2, 3]):
            for j, span in enumerate(SPANS):
                ax = axes[di, j]
                col = f'path{dim}d_{span}s_median'
                for si, split in enumerate(SPLITS):
                    for ai, (activity, color) in enumerate(COLORS.items()):
                        v = df.loc[(df.split == split) & (df.activity == activity), col].dropna()
                        if not len(v):
                            continue
                        if kind == 'scatter':
                            ax.scatter(si + rng.uniform(-.28, .28, len(v)), np.maximum(v, 1e-9), c=color, s=10, alpha=.45)
                        else:
                            pos = si + (ai-1)*.22
                            b = ax.boxplot([np.maximum(v, 1e-9)], positions=[pos], widths=.19,
                                           patch_artist=True, manage_ticks=False, showfliers=False,
                                           medianprops={'color': 'white'})
                            b['boxes'][0].set_facecolor(color)
                            ax.text(pos, .015, str(len(v)), transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
                ax.set_xticks(range(4), [SPLIT_XLABELS[s] for s in SPLITS])
                ax.set_yscale('log')
                ax.set_title(f'{dim}D PCA | {span} s path ({span+1} dots)')
                ax.set_ylabel('Median path length per minute (PCA units)')
                ax.grid(axis='y', alpha=.2)
        fig.suptitle('Consecutive CSI dots: total Euclidean distance' + (' | box outliers hidden; numbers = minutes' if kind == 'boxplot' else ''))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .955), ncol=3)
        fig.tight_layout(rect=(0,0,1,.91))
        fig.savefig(FIG / f'csi_pca_path_{kind}.png', dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharex=True, sharey=True)
    for ax, split in zip(axes, SPLITS):
        for act, color in COLORS.items():
            g = df[(df.split == split) & (df.activity == act)].dropna(subset=['snr_db', PRIMARY])
            ax.scatter(g.snr_db, g[PRIMARY].clip(lower=1e-9), c=color, s=12, alpha=.5)
        ax.set_title(SPLIT_XLABELS[split])
        ax.set_xlabel('Radar SNR (dB)')
        ax.set_yscale('log')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('20 s 3D PCA path length (minute median)')
    fig.legend(handles=handles, loc='upper center', ncol=3)
    fig.tight_layout(rect=(0,0,1,.9))
    fig.savefig(FIG / 'radar_csi_pca_path.png', dpi=160)
    plt.close(fig)


def examples(df, Z):
    # Deterministic central examples, not best-separated trajectories.
    fig = plt.figure(figsize=(14, 14))
    example_rows = []
    for si, split in enumerate(SPLITS):
        for ai, (act, color) in enumerate(COLORS.items()):
            ax = fig.add_subplot(4, 3, si*3+ai+1, projection='3d')
            g = df[(df.split == split) & (df.activity == act)].dropna(subset=[PRIMARY])
            ax.set_title(f'{SPLIT_XLABELS[split]} / {act}')
            if g.empty:
                ax.text2D(.15, .5, 'No eligible labeled minutes', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            i = (g[PRIMARY] - g[PRIMARY].median()).abs().idxmin()
            choices = [(s, Z[i,s:s+21]) for s in range(40) if np.isfinite(Z[i,s:s+21]).all()]
            start, pts = min(choices, key=lambda item: abs(path_length(item[1])-df.loc[i,PRIMARY]))
            # Translation preserves distances; common extent below preserves scale.
            pts = pts - pts[0]
            ax.plot(*pts.T, color=color, linewidth=1, alpha=.8)
            ax.scatter(*pts.T, c=np.arange(21), cmap='viridis', s=18)
            ax.scatter(*pts[0], marker='s', s=45, c=color)
            ax.set_xlabel('PC1 offset')
            ax.set_ylabel('PC2 offset')
            ax.set_zlabel('PC3 offset')
            ax.set_title(f'{SPLIT_XLABELS[split]} / {act}\n{df.loc[i,"minute"]} | L={path_length(pts):.1f}')
            example_rows.append((ax, pts))
    extent = max(np.abs(pts).max() for _,pts in example_rows)*1.08
    for ax, _ in example_rows:
        ax.set(xlim=(-extent,extent), ylim=(-extent,extent), zlim=(-extent,extent))
        ax.set_box_aspect((1,1,1))
    fig.suptitle('Representative 20 s trajectories | 21 one-second dots | same axis scale\nStart translated to origin; square = start; dark-to-yellow = time', y=.995)
    fig.tight_layout(rect=(0,0,1,.955))
    fig.savefig(FIG/'csi_pca_trajectories.png', dpi=150)
    plt.close(fig)


def evaluate(df):
    rows = []
    cols = [c for c in df if c.startswith('path')]
    for (split, placement), g in df.groupby(['split','placement']):
        for col in cols:
            for neg,pos in [('empty','occupied'),('empty','sleep'),('empty','present'),('sleep','present')]:
                h = g.dropna(subset=[col])
                if pos != 'occupied':
                    h = h[h.activity.isin([neg,pos])]
                y = h.label if pos == 'occupied' else (h.activity == pos).astype(int)
                if y.nunique() == 2:
                    rows.append(dict(split=split, placement=placement, feature=col, comparison=f'{neg}/{pos}',
                                     n=len(h), auc=roc_auc_score(y,h[col])))
    auc = pd.DataFrame(rows)
    auc.to_csv(OUT/'csi_pca_path_auc.csv', index=False)
    # Primary hypothesis fixed before results: more 20 s 3D distance = occupied.
    val = df[df.split == 'validation_minutes'].dropna(subset=[PRIMARY])
    unique = np.unique(val[PRIMARY])
    candidates = np.r_[-np.inf, (unique[:-1]+unique[1:])/2, np.inf]
    scores = np.array([balanced_accuracy_score(val.label, val[PRIMARY] >= t) for t in candidates])
    winners = np.flatnonzero(np.isclose(scores, scores.max()))
    threshold = candidates[winners[len(winners)//2]]
    metrics = []
    for split in ['validation_minutes','test_minutes']:
        g = df[df.split == split].dropna(subset=[PRIMARY])
        tn,fp,fn,tp = confusion_matrix(g.label, g[PRIMARY] >= threshold, labels=[0,1]).ravel()
        metrics.append(dict(split=split, n=len(g), threshold=float(threshold),
                            auc=roc_auc_score(g.label,g[PRIMARY]),
                            balanced_accuracy=balanced_accuracy_score(g.label,g[PRIMARY] >= threshold),
                            tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp)))
    pd.DataFrame(metrics).to_csv(OUT/'csi_pca_path_threshold.csv', index=False)
    summaries = df.groupby(['split','activity'])[cols].median()
    summaries.to_csv(OUT/'csi_pca_path_class_medians.csv')
    return auc, metrics, summaries


def report(df, pca, fit_count, auc, metrics, summaries):
    primary_auc = auc[(auc.feature == PRIMARY) & (auc.comparison == 'empty/occupied')]
    text = '''# CSI PCA trajectory path length

Source inspected: `C:/Users/ggad/Desktop/WifiSensingESP32HAR/src/live/csi_studio.py`,
`PCATracker.maybe_update`, `get_embedding`, and sequential trail plotting.
That live plot uses one-second CSI amplitude mean + standard deviation per
subcarrier (104 values), centered PCA, three components, without feature
standardization or whitening. It differs from the project's pca_train.py,
which flattens rolling-variance windows. This analysis follows the connected
live plot's mean/std representation.

## Method

Read raw CSI through E2/common.py, with its best-sample-count single-receiver
selection per minute. Use 60 fixed one-second bins from scheduled_start;
no interpolation and no bridging recordings. Deduplicate timestamps; require
at least five samples per second, observed span >=0.5 s, and maximum gap
including bin boundaries <=0.5 s. Save packet counts and every excluded minute.
These use host timestamps and cannot fully rule out host buffering artifacts.

Fit PCA once on all eligible calibration + train dots. Freeze this basis for
validation and test. Unlike the live display, do not repeatedly refit axes.
Do not normalize each minute, which would erase distance differences.

For z(t) in the first d components, L = sum(||z(t+1)-z(t)||_2).
Each 10/20/30 s path has 11/21/31 consecutive one-second dots and 10/20/30
edges. Reject the entire path if any dot is missing. Slide by one second;
save every path and summarize each minute by median and 90th percentile.
All comparison plots use the median. A minute needs at least one complete
path; path counts expose differing coverage. Overlapping paths are correlated;
metrics use one row per minute, not paths as independent samples.

Primary hypothesis chosen before results: higher median 20 s 3D path length
indicates occupied (sleep + present). Other spans/dimensions and p90 are
sensitivity analyses, not a test-selected winning model. AUC 0.5 is chance,
1 is perfect ordering in the proposed direction; AUC below 0.5 reverses it.
Threshold is chosen on validation balanced accuracy and applied unchanged
to test. Existing test labels were written from a time boundary; these are
exploratory results on previously examined data, not a new untouched test.
Calibration is sleep-only; no calibration pairwise AUC can be computed.

PCA distance measures changes in CSI window summaries, not physical travel
distance. Noise, packet density, receiver changes, and environment changes
can also contribute. PCA retains high-variance directions, which need not be
the most occupancy-sensitive ones. Empty rooms are not required to have zero
path length, and still/sleeping occupants may have small path length.

## Outputs and reproduction

Run `python E2/csi_pca_path.py`; `--reuse-snapshots` skips raw extraction.
Dependencies are E2/requirements.txt. Source data and reference project unchanged.
CSV outputs include csi_pca_path_minutes, csi_pca_path_windows, csi_pca_path_auc,
csi_pca_path_threshold, csi_pca_coverage, and csi_pca_path_class_medians.
NPZ files preserve snapshots, projections, and fitted PCA parameters.
Figures include csi_pca_path_scatter, csi_pca_path_boxplot,
csi_pca_trajectories, and radar_csi_pca_path.

## Results

'''
    text += f'PCA fitted on {fit_count:,} training dots. Explained variance ratios: ' + ', '.join(f'{v:.2%}' for v in pca.explained_variance_ratio_) + '.\n\n'
    text += 'Primary feature AUC by split:\n\n```text\n' + primary_auc.to_string(index=False) + '\n```\n\n'
    text += 'Validation-selected threshold evaluation:\n\n```text\n' + pd.DataFrame(metrics).to_string(index=False) + '\n```\n\n'
    text += 'Median primary path length by activity:\n\n```text\n' + summaries[[PRIMARY]].to_string() + '\n```\n\n'
    coverage = df.groupby(['split','activity']).agg(minutes=('minute','size'), valid_20s_minutes=(PRIMARY,'count'), valid_seconds=('valid_seconds','sum'))
    text += 'Coverage:\n\n```text\n' + coverage.to_string() + '\n```\n'
    text += '\n## Sensitivity to dimension and duration\n\n'
    text += '| Split | Dimensions | Seconds | Empty minutes | Occupied minutes | AUC |\n|---|---:|---:|---:|---:|---:|\n'
    for split in ['train','validation_minutes','test_minutes']:
        for dim in [2,3]:
            for span in SPANS:
                col = f'path{dim}d_{span}s_median'
                g = df[df.split == split].dropna(subset=[col])
                if g.label.nunique() == 2:
                    text += f'| {split} | {dim} | {span} | {(g.label==0).sum()} | {(g.label==1).sum()} | {roc_auc_score(g.label,g[col]):.3f} |\n'
    text += '''
## Interpretation and limitations

Compare class medians within each split, not pooled class distributions.
Present has a longer median 20 s 3D path than empty in all three comparable
splits. Thus this dataset supports the proposed direction descriptively.
However, test discrimination is weaker than validation. In particular, only
four empty test minutes qualify for 20 s, and one for 30 s. Those estimates
are fragile and subject to selection bias; the 10 s analysis has broader
coverage (36 empty and 93 present test minutes). Different durations use
different eligible subsets, so their AUC differences do not isolate duration.
No settings were chosen by optimizing test results.

Sleep and empty have overlapping path lengths in train, with very few eligible
sleep minutes. Sleep/present association is not equivalent to sleep/empty
occupancy discrimination. Calibration contains only sleep and contributes most
training dots to unsupervised PCA; its placement and class are confounded.
Receiver choice can vary across minutes in the existing reader. Fixed dot count
removes the direct effect of more trajectory steps, but differing native packet
counts can still affect each dot's mean/std precision.

The 20 s feature separates low-SNR validation minutes, but is near chance on
low-SNR test minutes (also with only four empty examples). Therefore these
plots do not yet establish reliable recovery of radar-missed occupancy.
The original live display can connect same-label dots across intervening other
labels; this analysis only connects chronologically adjacent valid seconds
inside a single labeled recording.

'''
    # Examine radar-ambiguous minutes without picking a threshold using labels.
    text += '\nRadar-low subset (SNR <=6 dB, a descriptive fixed cutoff):\n\n'
    for split in SPLITS:
        g = df[(df.split == split) & (df.snr_db <=6)].dropna(subset=[PRIMARY])
        if g.label.nunique()==2:
            text += f'- {split}: n={len(g)}, path AUC={roc_auc_score(g.label,g[PRIMARY]):.3f}.\n'
    text += '\nSpearman association of primary path length with median native packet count per valid second (diagnostic, not a causal adjustment):\n\n'
    for split in SPLITS:
        g = df[df.split == split].dropna(subset=[PRIMARY,'median_packets_per_valid_second'])
        if len(g) > 2:
            rho = g[[PRIMARY,'median_packets_per_valid_second']].corr(method='spearman').iloc[0,1]
            text += f'- {split}: n={len(g)}, rho={rho:.3f}.\n'
    (OUT/'CSI_PCA_PATH_REPORT.md').write_text(text, encoding='utf-8')
    print(text[text.index('## Results'):], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse-snapshots', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    df, X, counts = load_snapshots(args.reuse_snapshots, args.workers)
    df, Z, pca, fit_count = project_and_measure(df, X)
    distributions(df)
    examples(df,Z)
    auc, metrics, summaries = evaluate(df)
    report(df,pca,fit_count,auc,metrics,summaries)


if __name__ == '__main__':
    main()
