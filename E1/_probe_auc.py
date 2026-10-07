import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

tr = pd.read_csv('E1/outputs/pca_deep/desc_train_5s.csv')
va = pd.read_csv('E1/outputs/pca_deep/desc_val_5s.csv')
te = pd.read_csv('E1/outputs/pca_deep/desc_testml_5s.csv')
cols = ['ra_mean', 'snr_mean', 'rd_dca', 'rd_energy', 'rd_range_cm',
        're_mean', 'xy_mean', 'snr_max', 'csi_amp_mean', 'csi_rv_mean',
        'c_std0', 'r_std0', 'j_std0', 'j_mean0', 'clu_kmdist0',
        'fft_snr_b0', 'fftc_b0', 'csi_dop_frac']
print('%-16s %6s %6s %6s' % ('col', 'tr', 'val', 'test'))
for c in cols:
    r = []
    for d in (tr, va, te):
        v = d[c].values
        m = np.isfinite(v)
        try:
            a = roc_auc_score(d.label.values[m], v[m])
        except Exception:
            a = float('nan')
        r.append(max(a, 1 - a) if a == a else float('nan'))
    print('%-16s %6.3f %6.3f %6.3f' % (c, r[0], r[1], r[2]))
