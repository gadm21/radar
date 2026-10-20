"""Render representative feature samples per modality and class.

Picks the train minute whose 10 s windowed features are closest to the
class median (in rolling-variance space), then plots, for empty and
occupied rows:
  CSI:  amplitude heatmap | causal rolling variance | 1-s PCA projections
  radar: window-mean RD map | window-mean XY map | per-frame SNR trace

Output: icc_assets/feats.png
"""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import occupancy_pipeline as P          # noqa: E402
import common as C                      # noqa: E402

OUT = P.OUT
ASSETS = Path(__file__).resolve().parents[1] / 'icc_assets'
SEC_HZ, N_SUB = P.SEC_HZ, P.N_SUB
W_SEC = 10                              # window shown for heatmaps/maps

BLUE, ORANGE = '#4C72B0', '#DD8452'


def _pick_minute(index, feat, ds, label):
    """Minute closest to the class median rolling variance in set `ds`,
    restricted to well-covered minutes (>=55 valid CSI seconds)."""
    good = index[(index.n_secs >= 55) & (index.radar_hz >= 8)
                 & (index.dataset == ds)]
    sub = feat[(feat.dataset == ds) & (feat.label == label)
               & feat.rec_id.isin(good.rec_id)]
    if 'aug' in sub.columns:
        sub = sub[~sub.aug.astype(bool)]
    med = sub.csi_rv_mean.median()
    rid = (sub.groupby('rec_id').csi_rv_mean.mean()
           .sub(med).abs().idxmin())
    return index[index.rec_id == rid].iloc[0]


def _valid_block(sec_valid, need=W_SEC):
    """First 'need' consecutive valid seconds; fallback to argmax run."""
    v = sec_valid.astype(bool)
    run = np.convolve(v.astype(int), np.ones(need, int), 'valid')
    s0 = int(np.argmax(run))
    return s0, s0 + need


def _rollvar(amp):
    """Causal 0.2 s rolling variance, mean over subcarriers."""
    s = pd.DataFrame(amp)
    return s.rolling(20, min_periods=2).var().mean(axis=1).values


def main():
    index = pd.read_csv(OUT / 'minutes_index.csv')
    index = index[index.usable.astype(bool)]
    feat = pd.read_csv(OUT / 'features_10s.csv')
    pcas = joblib.load(OUT / 'pcas.joblib')
    if isinstance(pcas, dict):
        pca_csi = pcas['csi']
        pca_rad = pcas.get('radar', pcas.get('rad'))
    else:
        pca_csi, pca_rad = pcas

    rows = []
    for ds, lab in (('train', 0), ('test', 0),
                    ('train', 1), ('test', 1)):
        r = _pick_minute(index, feat, ds, lab)
        z = np.load(r.npz)
        rows.append((ds, lab, z, r))
        print(f"{ds} label={lab} rec={r.rec_id} secs={int(r.n_secs)} "
              f"radar_hz={r.radar_hz:.1f}")

    fig, axes = plt.subplots(4, 6, figsize=(10.5, 5.6),
                             constrained_layout=True)
    titles = ['CSI amplitude\n(52 subcarriers, 100 Hz)',
              'CSI rolling var.\n(0.2 s window)',
              'CSI PCA proj.\n(per second)',
              'radar RD map\n(window mean)',
              'radar XY map\n(window mean)',
              'radar SNR\n(per frame)']

    for row, (ds, lab, z, r) in enumerate(rows):
        amp = z['amp60'].astype(np.float32).reshape(60 * SEC_HZ, N_SUB)
        sec_valid = z['sec_valid']
        dots = z['dots'].astype(np.float32)
        maps = z['maps'].astype(np.float32)
        snr = z['snr_f'].astype(np.float32)
        rts = z['rts'].astype(np.float32)
        rd_sec = z['rd_sec'].astype(np.float32)
        sec_has_rad = z['sec_has_radar']

        # representative 10 s block: first fully-valid stretch
        b0, b1 = _valid_block(sec_valid)
        sl = slice(b0 * SEC_HZ, b1 * SEC_HZ)
        fsel = (rts >= b0) & (rts < b1)

        cname = BLUE if lab == 0 else ORANGE
        ctag = f'{ds}, ' + ('empty' if lab == 0 else 'occupied')

        ax = axes[row, 0]
        ax.imshow(amp[sl].T, aspect='auto', origin='lower',
                  cmap='viridis', interpolation='nearest')
        ax.set_ylabel(f'{ctag}\nsubcarrier', fontsize=7)
        ax.set_xlabel('time (s)', fontsize=7)
        ax.set_xticks([0, 500, 1000])
        ax.set_xticklabels(['0', '5', '10'], fontsize=7)
        ax.tick_params(labelsize=6, length=2)

        ax = axes[row, 1]
        t = np.arange(W_SEC * SEC_HZ) / SEC_HZ
        ax.plot(t, _rollvar(amp[sl]), lw=0.7, color=cname)
        ax.set_ylabel('variance', fontsize=7)
        ax.set_xlabel('time (s)', fontsize=7)
        ax.tick_params(labelsize=6, length=2)

        ax = axes[row, 2]
        m = sec_valid.astype(bool)
        if m.any():
            proj = pca_csi.transform(dots[m])
            ts = np.flatnonzero(m)
            for k in range(P.PCA_DIM):
                ax.plot(ts, proj[:, k], lw=0.7,
                        label=f'PC{k + 1}' if row == 0 else None)
        if row == 0:
            ax.legend(fontsize=5, frameon=False, loc='upper right')
        ax.set_ylabel('projection', fontsize=7)
        ax.set_xlabel('second', fontsize=7)
        ax.tick_params(labelsize=6, length=2)

        ax = axes[row, 3]
        rd = maps[fsel, 0].mean(axis=0) if fsel.any() \
            else maps[:, 0].mean(axis=0)
        im = ax.imshow(rd, origin='lower', cmap='magma',
                       interpolation='nearest')
        ax.set_ylabel('doppler', fontsize=7)
        ax.set_xlabel('range', fontsize=7)
        ax.tick_params(labelsize=6, length=2)

        ax = axes[row, 4]
        xy = maps[fsel, 3].mean(axis=0) if fsel.any() \
            else maps[:, 3].mean(axis=0)
        ax.imshow(xy, origin='lower', cmap='magma',
                  interpolation='nearest')
        ax.set_ylabel('y (bins)', fontsize=7)
        ax.set_xlabel('x (bins)', fontsize=7)
        ax.tick_params(labelsize=6, length=2)

        ax = axes[row, 5]
        ax.plot(rts, snr, lw=0.5, color=cname)
        ax.set_ylabel('SNR (dB)', fontsize=7)
        ax.set_xlabel('time (s)', fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        # 1-s RD PCA projections as small secondary axis annotation
        m2 = sec_has_rad.astype(bool)
        if m2.any():
            rp = pca_rad.transform(rd_sec[m2])
            ax2 = ax.twinx()
            ax2.plot(np.flatnonzero(m2), rp[:, 0], lw=0.7,
                     color='gray', ls='--')
            ax2.set_yticks([])
            ax2.set_ylabel('RD PC1', fontsize=6, color='gray')

    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=7)

    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / 'feats.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
