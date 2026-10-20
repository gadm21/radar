"""3-second window experiment on the combined train_minutes cohort.

Protocol (per user request):
  * calibration_minutes + train are combined as the training set
    (train_minutes/). present and sleep both map to occupied.
  * Non-overlapping 3 s windows (20 per minute, aligned to the minute).
  * CSI branch: guaranteed 100 Hz resampling (upstream WiFiSensing
    routine, verbatim) -> per-second mean/std dots -> PCA fit on real
    train_minutes dots -> variance of the 3 consecutive PCA projections
    per window (the established pipeline). Plus amplitude stats, causal
    rolling variance and STFT band powers on the 300 resampled samples.
  * Radar branch: RD/RA/RE/XY log-power maps per frame (same
    beamforming as the paper) -> per-window pooled descriptors +
    clutter-masked SNR proxy + variance of consecutive PCA projections
    of the per-frame RD maps (the CSI variance recipe applied to radar).
  * Class balance: empty train minutes are upsampled with different
    noise types (Gaussian, gain drift, mixup, temporal resampling) until
    empty minute count equals occupied minute count.
  * Models: RandomForest, XGBoost and an MLP (DL) on individual feature
    groups and the full fusion vector. Thresholds are chosen on
    validation windows only; metrics are reported per window and per
    minute (mean window probability) on the frozen balanced cohorts of
    paper_100hz_variance3s/paired_minutes.csv.

Usage:
    python E2/win3s_experiment.py extract --workers 6
    python E2/win3s_experiment.py features
    python E2/win3s_experiment.py models
    python E2/win3s_experiment.py all --workers 6
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft
from scipy import signal as sps
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.OUTPUT_DIR / 'win3s'
CACHE = C.CACHE_DIR / 'cache3s'
SEED = 42

WIN = 3.0              # seconds
NW = 20                # windows per minute
N_SUB = 52
ROLL_W = 20            # causal rolling-variance samples on the 100 Hz grid
STFT_BANDS = [(0.0, 0.5), (0.5, 2.0), (2.0, 8.0), (8.0, 25.0)]
MIN_RADAR_FRAMES = 5   # per 3 s window
_DC_ROWS = (10, 14)    # masked doppler rows on the 24x24 RD map
_NEAR_RANGE = 2

VIEW_NAMES = ['rd', 'ra', 're', 'xy']
DESC_STATS = ['mean', 'p90', 'std90', 'delta', 'peak']

CSI_AMP = ['csi_amp_mean', 'csi_amp_std', 'csi_amp_q90']
CSI_RV = ['csi_rv_mean', 'csi_rv_q90', 'csi_rv_max']
CSI_STFT = [f'csi_stft_b{i}' for i in range(len(STFT_BANDS))]
CSI_PCVAR = ['csi_pcvar1', 'csi_pcvar2', 'csi_pcvar']
SNR = ['snr_max', 'snr_mean']
RAD_PCVAR = ['rad_pcvar1', 'rad_pcvar2', 'rad_pcvar']
VIEW_COLS = {v: [f'{v}_{s}' for s in DESC_STATS] for v in VIEW_NAMES}

FEATURE_SETS = {
    'csi_amp': CSI_AMP,
    'csi_rollvar': CSI_RV,
    'csi_stft': CSI_STFT,
    'csi_pcvar': CSI_PCVAR,
    'csi_all': CSI_AMP + CSI_RV + CSI_STFT + CSI_PCVAR + ['csi_ok'],
    'snr': SNR,
    'rd': VIEW_COLS['rd'],
    'ra': VIEW_COLS['ra'],
    're': VIEW_COLS['re'],
    'xy': VIEW_COLS['xy'],
    'radar_pcvar': RAD_PCVAR,
    'radar_all': SNR + RAD_PCVAR
    + sum((VIEW_COLS[v] for v in VIEW_NAMES), []) + ['rad_ok'],
    'fusion': CSI_AMP + CSI_RV + CSI_STFT + CSI_PCVAR + ['csi_ok']
    + SNR + RAD_PCVAR + sum((VIEW_COLS[v] for v in VIEW_NAMES), [])
    + ['rad_ok'],
}

# ---------------------------------------------------------------------------
# Pass A: per-minute extraction
# ---------------------------------------------------------------------------
_HANN_R = np.hanning(128).astype(np.float32)
_HANN_D = np.hanning(64).astype(np.float32)
_THETA = np.deg2rad(np.linspace(-60, 60, 25))
_PHASE = np.exp(1j * np.pi * np.sin(_THETA)).astype(np.complex64)
_RADIUS = np.arange(64) / 63
_IX = np.clip(((_RADIUS[:, None] * np.sin(_THETA) + 1) / 2 * 23)
              .round().astype(int), 0, 23)
_IY = np.clip((_RADIUS[:, None] * np.cos(_THETA) * 23)
              .round().astype(int), 0, 23)


def _resample_100hz(amp, rel):
    """Upstream guaranteed-rate resampler (loaded verbatim).

    run_csi_100hz prepends E2/.deps (an incompatible bundled
    numpy/scipy); strip it so later imports resolve against this env."""
    import run_csi_100hz as R100
    bad = (ROOT / 'E2/.deps').resolve()
    sys.path[:] = [p for p in sys.path
                   if Path(p).resolve() != bad]
    return R100.resample(amp, rel)


def _csi_minute(rec):
    """Return (dots[60,104] f32, sec_valid[60], csi_w[NW,10])."""
    csi, ts = C.recording_csi(rec)
    dots = np.full((60, 104), np.nan, dtype=np.float32)
    sec_valid = np.zeros(60, dtype=bool)
    amp60 = np.full((60, 100, N_SUB), np.nan, dtype=np.float16)
    csi_w = np.full((NW, 10), np.nan, dtype=np.float32)
    if csi is None or len(csi) < 2:
        return dots, sec_valid, csi_w
    rel = ts - rec.start_ts
    keep = (rel >= 0) & (rel < 60) & np.isfinite(csi).all(axis=1)
    rel, amp = rel[keep], np.abs(csi[keep]).astype(np.float64)
    rel, uniq = np.unique(rel, return_index=True)
    amp = amp[uniq]
    if len(rel) < 2:
        return dots, sec_valid, csi_w
    uniform, t, spb, _ = _resample_100hz(amp, rel)
    raw_bins = np.floor(rel).astype(int)
    bins = np.floor(t).astype(int)
    for sec in range(60):
        observed = rel[raw_bins == sec]
        if (len(observed) < 5 or observed[-1] - observed[0] < .5
                or np.diff(np.r_[sec, observed, sec + 1]).max() > .5):
            continue
        idx = np.flatnonzero(bins == sec)
        if len(idx) != 100:
            continue
        a = uniform[idx]
        amp60[sec] = a.astype(np.float16)
        dots[sec] = np.r_[a.mean(axis=0), a.std(axis=0)]
        sec_valid[sec] = True
    return dots, sec_valid, _csi_window_stats(amp60, sec_valid)


def _rollvar(x, w):
    """Causal trailing variance along axis 1 of x: (n,T,S) -> (n,T,S)."""
    z = np.zeros((x.shape[0], 1, x.shape[2]), dtype=np.float64)
    cs = np.concatenate([z, np.cumsum(x, axis=1)], axis=1)
    cs2 = np.concatenate([z, np.cumsum(x * x, axis=1)], axis=1)
    hi = np.arange(1, x.shape[1] + 1)
    lo = np.clip(hi - w, 0, None)
    cnt = (hi - lo).astype(np.float64)[None, :, None]
    mean = (cs[:, hi] - cs[:, lo]) / cnt
    msq = (cs2[:, hi] - cs2[:, lo]) / cnt
    return np.clip(msq - mean * mean, 0, None)


def _csi_window_stats(amp60, sec_valid):
    """Per-window amplitude / rolling-variance / STFT features.

    amp60: (60,100,52) float16 with NaN on invalid seconds.
    Returns (NW, 10) float32: amp3, rv3, stft4 — NaN where the window
    is not fully valid (all three seconds eligible)."""
    out = np.full((NW, 10), np.nan, dtype=np.float32)
    for w in range(NW):
        if not sec_valid[3 * w:3 * w + 3].all():
            continue
        a = amp60[3 * w:3 * w + 3].astype(np.float32).reshape(300, N_SUB)
        la = np.log1p(a)
        out[w, 0] = a.mean()
        out[w, 1] = a.std()
        out[w, 2] = np.quantile(a, 0.9)
        rv = _rollvar(la[None], ROLL_W)[0]
        out[w, 3] = rv.mean()
        out[w, 4] = np.quantile(rv, 0.9)
        out[w, 5] = rv.max()
        xc = a - a.mean(axis=0, keepdims=True)
        f, _, Z = sps.stft(xc.T, fs=100.0, nperseg=128, noverlap=64,
                           padded=False, boundary=None)
        pw = (np.abs(Z) ** 2).mean(axis=(0, 2))          # (freq,)
        for b, (lo, hi) in enumerate(STFT_BANDS):
            band = (f >= lo) & (f < hi)
            out[w, 6 + b] = np.log1p(pw[band].mean()) if band.any() else 0.0
    return out


def _radar_minute(rec):
    """Decode frames -> (rd24[nF,24,24] f16, frame_win[nF], desc[NW,22])."""
    from preprocess import read_uint12_batch
    desc = np.full((NW, 22), np.nan, dtype=np.float32)
    payloads, rts = C.recording_radar_frames(rec)
    rel = np.asarray(rts) - rec.start_ts
    keep = [i for i, p in enumerate(payloads)
            if 0 <= rel[i] < 60 and len(p) == 36864]
    if not keep:
        return np.zeros((0, 24, 24), np.float16), np.zeros(0, np.int8), desc
    payloads = [payloads[i] for i in keep]
    frame_win = np.clip((rel[keep] / WIN).astype(int), 0, NW - 1)
    rd_list = {v: [] for v in VIEW_NAMES}
    for off in range(0, len(payloads), 16):
        batch = payloads[off:off + 16]
        a = read_uint12_batch(b''.join(batch)).reshape(-1, 64, 128, 3)
        a -= a.mean(axis=2, keepdims=True)
        a *= _HANN_R[None, None, :, None]
        R = fft.fft(a, axis=2)[:, :, :64, :]
        D = fft.fftshift(fft.fft(R * _HANN_D[None, :, None, None],
                                 axis=1, workers=-1), axes=1)
        rd = np.mean(np.abs(D) ** 2, axis=3)          # (n,64,64)
        rd[:, :, :2] = 0
        rd24 = zoom(np.log1p(rd), (1, 24 / 64, 24 / 64),
                    order=1).astype(np.float32)
        for key, left in (('ra', 0), ('re', 1)):
            p = (np.abs(R[:, :, :, left]) ** 2
                 + np.abs(R[:, :, :, 2]) ** 2).mean(axis=1)
            cross = (R[:, :, :, left] * R[:, :, :, 2].conj()).mean(axis=1)
            energy = np.maximum(
                p[:, :, None] + 2 * np.real(cross[:, :, None] * _PHASE), 0)
            energy[:, :2, :] = 0
            rd_list[key].extend(np.log1p(energy).astype(np.float32))
            if key == 'ra':
                xy = np.zeros((len(batch), 24, 24), np.float32)
                for j in range(len(batch)):
                    np.maximum.at(xy[j], (_IY.ravel(), _IX.ravel()),
                                  energy[j].ravel())
                rd_list['xy'].extend(np.log1p(xy))
        rd_list['rd'].extend(rd24)
    rd24 = np.asarray(rd_list['rd'], dtype=np.float16)  # (nF,24,24)

    # clutter-masked SNR proxy per frame on the 24x24 RD power map
    m = np.zeros((24, 24), bool)
    m[_DC_ROWS[0]:_DC_ROWS[1], :] = True
    m[:, :_NEAR_RANGE] = True
    pwr = np.expm1(rd24.astype(np.float32)).reshape(len(rd24), -1)
    med = np.median(pwr, axis=1) + 1e-9
    peak = np.where(m.reshape(-1)[None], 0.0, pwr).max(axis=1)
    snr_f = 10.0 * np.log10(peak / med)

    for w in range(NW):
        sel = np.flatnonzero(frame_win == w)
        if len(sel) < MIN_RADAR_FRAMES:
            continue
        row = [float(np.nanmax(snr_f[sel])), float(np.nanmean(snr_f[sel]))]
        for v in VIEW_NAMES:
            maps = np.asarray([rd_list[v][i] for i in sel], np.float32)
            mean_img = maps.mean(axis=0)
            std_img = maps.std(axis=0)
            delta = np.abs(np.diff(maps, axis=0)).mean()
            row += [float(mean_img.mean()),
                    float(np.quantile(mean_img, 0.9)),
                    float(np.quantile(std_img, 0.9)),
                    float(delta), float(mean_img.max())]
        desc[w] = row
    return rd24, frame_win.astype(np.int8), desc


def extract_minute(rec):
    """Pass A cache for one minute. Returns (rec_id, status)."""
    path = CACHE / rec.source / f'{rec.folder.name}.npz'
    if path.exists():
        return rec.rec_id, 'cached'
    try:
        dots, sec_valid, csi_w = _csi_minute(rec)
        rd24, frame_win, desc = _radar_minute(rec)
    except Exception as e:
        return rec.rec_id, f'error: {e}'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, dots=dots, sec_valid=sec_valid,
        csi_w=csi_w, rd24=rd24, frame_win=frame_win, desc=desc,
        meta=json.dumps({'rec_id': rec.rec_id, 'label': rec.label,
                         'placement': rec.placement,
                         'session_id': rec.session_id}))
    return rec.rec_id, 'ok'


def _extract_worker(rec):
    try:
        return extract_minute(rec)
    except Exception as e:
        return rec.rec_id, f'error: {e}'


def run_extract(workers=4, limit=None):
    recs, problems = C.discover_recordings()
    recs = C.assign_sessions(recs)
    recs = [r for r in recs if r.status == 'ok']
    if limit:
        recs = recs[:limit]
    todo = [r for r in recs
            if not (CACHE / r.source / f'{r.folder.name}.npz').exists()]
    print(f'{len(recs)} usable, {len(todo)} to extract', flush=True)
    stats = {'ok': 0, 'cached': len(recs) - len(todo)}
    errs = []
    t0 = time.time()
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, (rid, res) in enumerate(pool.map(
                    _extract_worker, todo, chunksize=2)):
                if res.startswith('error'):
                    errs.append(f'{rid}: {res}')
                else:
                    stats[res] = stats.get(res, 0) + 1
                if (i + 1) % 50 == 0:
                    print(f'[{i+1}/{len(todo)}] {time.time()-t0:.0f}s '
                          f'{stats} errs={len(errs)}', flush=True)
    else:
        for i, r in enumerate(todo):
            rid, res = _extract_worker(r)
            if res.startswith('error'):
                errs.append(f'{rid}: {res}')
            else:
                stats[res] = stats.get(res, 0) + 1
            if (i + 1) % 50 == 0:
                print(f'[{i+1}/{len(todo)}] {time.time()-t0:.0f}s '
                      f'{stats} errs={len(errs)}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'extract_errors.json').write_text(json.dumps(errs, indent=2))
    print('EXTRACT DONE', stats, f'errors={len(errs)}', flush=True)


# ---------------------------------------------------------------------------
# Pass B: PCA fits + full window feature table
# ---------------------------------------------------------------------------
TRAIN_SOURCES = ('calibration_minutes', 'train')
EVAL_SOURCES = ('validation_minutes', 'test_minutes')


def _iter_caches():
    for source in ('calibration_minutes', 'train',
                   'validation_minutes', 'test_minutes'):
        d = CACHE / source
        if not d.exists():
            continue
        for f in sorted(d.glob('*.npz')):
            yield source, f


def run_features():
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(SEED)
    # --- fit CSI PCA on real train_minutes dots -------------------------
    dots_tr = []
    for source, f in _iter_caches():
        if source not in TRAIN_SOURCES:
            continue
        d = np.load(f)
        v = d['sec_valid'].astype(bool)
        if v.any():
            dots_tr.append(d['dots'][v])
    D = np.concatenate(dots_tr)
    csi_pca = PCA(n_components=2, svd_solver='full').fit(D)
    print(f'CSI PCA on {len(D)} train seconds; '
          f'explained={csi_pca.explained_variance_ratio_.round(3)}',
          flush=True)
    # --- fit radar PCA on per-frame RD maps of train_minutes ------------
    samp = []
    for source, f in _iter_caches():
        if source not in TRAIN_SOURCES:
            continue
        d = np.load(f)
        rd = d['rd24']
        if len(rd):
            samp.append(rd.astype(np.float32).reshape(len(rd), -1))
    R = np.concatenate(samp)
    if len(R) > 200_000:
        R = R[rng.choice(len(R), 200_000, replace=False)]
    rad_pca = PCA(n_components=2, svd_solver='randomized',
                  random_state=SEED).fit(R)
    print(f'Radar PCA on {len(R)} train frames; '
          f'explained={rad_pca.explained_variance_ratio_.round(3)}',
          flush=True)
    np.savez_compressed(OUT / 'pca.npz',
                        csi_mean=csi_pca.mean_, csi_comp=csi_pca.components_,
                        rad_mean=rad_pca.mean_, rad_comp=rad_pca.components_)
    # --- per-window feature table ---------------------------------------
    cols = (CSI_AMP + CSI_RV + CSI_STFT + CSI_PCVAR + ['csi_ok']
            + SNR + VIEW_COLS['rd'] + VIEW_COLS['ra'] + VIEW_COLS['re']
            + VIEW_COLS['xy'] + RAD_PCVAR + ['rad_ok'])
    rows = []
    feats = []
    for source, f in _iter_caches():
        d = np.load(f)
        meta = json.loads(str(d['meta']))
        sv = d['sec_valid'].astype(bool)
        z = np.full((60, 2), np.nan, dtype=np.float32)
        if sv.any():
            z[sv] = csi_pca.transform(d['dots'][sv].astype(np.float64))
        csi_w = d['csi_w']                                  # (20,10)
        desc = d['desc']                                    # (20,22)
        rd = d['rd24'].astype(np.float32)
        fw = d['frame_win']
        pcvar = np.full((NW, 3), np.nan, dtype=np.float32)
        if len(rd):
            P = ((rd.reshape(len(rd), -1) - rad_pca.mean_)
                 @ rad_pca.components_.T)
            for w in range(NW):
                sel = fw == w
                if sel.sum() >= MIN_RADAR_FRAMES:
                    v = P[sel].var(axis=0, ddof=0)
                    pcvar[w] = [v[0], v[1], v.sum()]
        block = np.full((NW, len(cols)), np.nan, dtype=np.float32)
        block[:, :10] = csi_w
        block[:, 10:13] = np.nan
        csi_ok = np.isfinite(csi_w).all(axis=1)
        for w in range(NW):
            if csi_ok[w]:
                zz = z[3 * w:3 * w + 3]
                v = zz.var(axis=0, ddof=0)
                block[w, 10] = v[0]
                block[w, 11] = v[1]
                block[w, 12] = v.sum()
        block[:, 13] = csi_ok.astype(np.float32)
        block[:, 14:36] = desc
        rad_ok = np.isfinite(desc).all(axis=1)
        block[:, 36:39] = pcvar
        block[:, 39] = rad_ok.astype(np.float32)
        keep = csi_ok | rad_ok
        for w in np.flatnonzero(keep):
            rows.append({'split': source, 'minute': f.stem, 'win': int(w),
                         'label': int(meta['label']),
                         'placement': meta['placement'], 'aug': 0,
                         'aug_type': ''})
            feats.append(block[w])
    X = np.asarray(feats, dtype=np.float32)
    df = pd.DataFrame(rows)
    feat = pd.DataFrame(X, columns=cols)
    out = pd.concat([df, feat], axis=1)
    out.to_csv(OUT / 'windows.csv', index=False)
    print(f'wrote {OUT}/windows.csv: {len(out)} windows x {len(cols)} '
          f'features', flush=True)
    return out


# ---------------------------------------------------------------------------
# Pass C: augmentation + models
# ---------------------------------------------------------------------------
NOISE_TYPES = ['gauss', 'gain', 'mixup', 'temporal']


def _augment_minute(x, pool, sig, rng, kind):
    """One synthetic minute from an empty-minute window matrix x (W,F).

    pool: list of empty-minute feature matrices for mixup.
    sig:  per-feature std of real train windows.
    """
    if kind == 'gauss':
        return x + rng.normal(0, 0.15, x.shape) * sig
    if kind == 'gain':
        drift = rng.normal(0, 0.08, (1, x.shape[1]))
        jitter = rng.normal(0, 0.03, x.shape)
        return x * (1 + drift + jitter)
    if kind == 'mixup':
        other = pool[rng.integers(len(pool))]
        lam = rng.uniform(0.25, 0.75)
        w = min(len(x), len(other))
        out = x.copy()
        out[:w] = lam * x[:w] + (1 - lam) * other[:w]
        return out
    if kind == 'temporal':
        idx = rng.integers(0, len(x), len(x))
        return x[np.sort(idx)] + rng.normal(0, 0.05, x.shape) * sig
    raise ValueError(kind)


def run_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, roc_auc_score)
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    df = pd.read_csv(OUT / 'windows.csv')
    cohort = pd.read_csv(
        C.OUTPUT_DIR / 'paper_100hz_variance3s' / 'paired_minutes.csv')
    cohort = cohort[cohort.split.isin(EVAL_SOURCES)]
    keep_ids = set(zip(cohort.split, cohort.minute))
    df['_eval'] = [(s, m) in keep_ids for s, m in
                   zip(df.split, df.minute)]
    is_tr = df.split.isin(TRAIN_SOURCES).to_numpy()
    is_va = df.split.eq('validation_minutes').to_numpy() & df._eval
    is_te = df.split.eq('test_minutes').to_numpy() & df._eval
    y = df.label.to_numpy()

    # ---- augment empty train minutes -----------------------------------
    rng = np.random.default_rng(SEED)
    feat_cols = [c for c in df.columns
                 if c not in ('split', 'minute', 'win', 'label',
                              'placement', 'aug', 'aug_type', '_eval')]
    trm = df[is_tr]
    key = trm.split + '/' + trm.minute
    empty_min = key[trm.label == 0].unique()
    occ_min = key[trm.label == 1].unique()
    n_aug = len(occ_min) - len(empty_min)
    sig = np.nanstd(
        trm[feat_cols].to_numpy(np.float64), axis=0) + 1e-9
    pool = [g[feat_cols].to_numpy(np.float32)
            for _, g in trm[trm.label == 0].groupby([trm.split,
                                                   trm.minute])]
    aug_rows = []
    for i in range(max(n_aug, 0)):
        src = empty_min[i % len(empty_min)]
        g = trm[(trm.split + '/' + trm.minute) == src].sort_values('win')
        x = g[feat_cols].to_numpy(np.float32)
        kind = NOISE_TYPES[i % len(NOISE_TYPES)]
        xa = _augment_minute(x, pool, sig, rng, kind)
        for k, (_, r) in enumerate(g.iterrows()):
            row = {c: r[c] for c in
                   ('split', 'minute', 'win', 'label', 'placement')}
            row['minute'] = f'{src}__aug{i:04d}'
            row['aug'] = 1
            row['aug_type'] = kind
            row['_eval'] = False
            aug_rows.append((row, xa[k]))
    if aug_rows:
        aug_meta = pd.DataFrame([r for r, _ in aug_rows])
        aug_feat = np.asarray([x for _, x in aug_rows],
                              dtype=np.float32)
        aug_df = pd.concat([aug_meta,
                            pd.DataFrame(aug_feat, columns=feat_cols)],
                           axis=1)
    else:
        aug_df = df.iloc[:0].copy()
    full = pd.concat([df.drop(columns=['_eval']), aug_df[df.columns]],
                     ignore_index=True)
    full.to_csv(OUT / 'windows_augmented.csv', index=False)
    n_empty_aug = len(empty_min) + n_aug
    print(f'train minutes: {len(empty_min)} empty + {len(occ_min)} occupied'
          f' -> +{n_aug} augmented empty = {n_empty_aug} vs '
          f'{len(occ_min)}', flush=True)
    (OUT / 'augmentation.json').write_text(json.dumps({
        'empty_real': int(len(empty_min)),
        'occupied': int(len(occ_min)),
        'empty_augmented': int(n_aug),
        'noise_types': NOISE_TYPES,
        'balanced': bool(n_empty_aug == len(occ_min)),
    }, indent=2))

    masks = {'train': np.r_[is_tr, np.ones(len(aug_df), bool)],
             'validation_minutes': np.r_[is_va, np.zeros(len(aug_df), bool)],
             'test_minutes': np.r_[is_te, np.zeros(len(aug_df), bool)]}

    Xall = full[feat_cols].to_numpy(np.float32)
    yall = np.r_[y, np.zeros(len(aug_df), np.int64)]
    min_id = pd.factorize(
        full.split + '/' + full.minute.astype(str))[0]

    def thr_select(yy, p):
        grid = np.arange(0.05, 0.95, 0.01)
        s = np.array([balanced_accuracy_score(yy, p >= t) for t in grid])
        return float(np.median(grid[s >= s.max() - 1e-9]))

    def minute_agg(mask, p):
        yt, yp = [], []
        mm = min_id[mask]
        for r in np.unique(mm):
            m = mm == r
            yt.append(int(np.round(yall[mask][m].mean())))
            yp.append(float(p[m].mean()))
        return np.asarray(yt), np.asarray(yp)

    def metrics(yy, p, t):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
        return {'n': int(len(yy)),
                'auc': float(roc_auc_score(yy, p))
                if len(np.unique(yy)) == 2 else np.nan,
                'balanced_acc': float(balanced_accuracy_score(yy, pred)),
                'accuracy': float(accuracy_score(yy, pred)),
                'macro_f1': float(f1_score(yy, pred, average='macro')),
                'threshold': t, 'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)}

    def train_mlp(Xtr, ytr, Xva_sub, yva_sub):
        import torch
        from torch import nn
        torch.manual_seed(SEED)
        dev = 'cpu'
        net = nn.Sequential(
            nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3,
                               weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        Xtr_t = torch.tensor(Xtr)
        ytr_t = torch.tensor(ytr.astype(np.float32))
        Xva_t = torch.tensor(Xva_sub)
        best = {'ba': -1, 'state': None}
        bad = 0
        n = len(Xtr_t)
        for ep in range(60):
            net.train()
            perm = torch.randperm(n)
            for i in range(0, n, 1024):
                idx = perm[i:i + 1024]
                opt.zero_grad()
                loss = crit(net(Xtr_t[idx]).squeeze(1), ytr_t[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(net(Xva_t).squeeze(1)).numpy()
            ba = balanced_accuracy_score(yva_sub, pv >= 0.5)
            if ba > best['ba']:
                best = {'ba': ba, 'state': {k: v.clone()
                                            for k, v in
                                            net.state_dict().items()}}
                bad = 0
            else:
                bad += 1
                if bad >= 8:
                    break
        if best['state'] is not None:
            net.load_state_dict(best['state'])
        return net

    results = []
    rng2 = np.random.RandomState(SEED)
    for fset, cols in FEATURE_SETS.items():
        fi = [feat_cols.index(c) for c in cols]
        X = Xall[:, fi]
        tr_mask = masks['train']
        imp = SimpleImputer(strategy='median').fit(
            X[np.r_[is_tr, np.zeros(len(aug_df), bool)]][:, :])
        Xi = imp.transform(X)
        Xtr, ytr = Xi[tr_mask], yall[tr_mask]
        # RF / XGB -------------------------------------------------------
        learners = {
            'rf': RandomForestClassifier(
                n_estimators=300, class_weight='balanced_subsample',
                n_jobs=-1, random_state=SEED),
            'xgb': XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                random_state=SEED, eval_metric='logloss'),
        }
        sc = StandardScaler().fit(Xtr)
        Xz = sc.transform(Xi)
        # MLP trained on scaled features; early stop on val BA subsample
        vsub = np.flatnonzero(masks['validation_minutes'])
        if len(vsub) > 20000:
            vsub = np.sort(rng2.choice(vsub, 20000, replace=False))
        net = train_mlp(sc.transform(Xtr).astype(np.float32), ytr,
                        Xz[vsub].astype(np.float32), yall[vsub])
        learners['mlp'] = net
        for mname, model in learners.items():
            src = Xz if mname == 'mlp' else Xi
            if mname == 'mlp':
                import torch
                net.eval()
                with torch.no_grad():
                    p_all = torch.sigmoid(
                        net(torch.tensor(src.astype(np.float32)))
                        .squeeze(1)).numpy()
            else:
                model.fit(Xtr, ytr)
                p_all = model.predict_proba(src)[:, 1]
            va = masks['validation_minutes']
            te = masks['test_minutes']
            t_win = thr_select(yall[va], p_all[va])
            ytm, ypm = minute_agg(va, p_all[va])
            t_min = thr_select(ytm, ypm)
            for sname, msk in (('validation_minutes', va),
                               ('test_minutes', te)):
                rec = metrics(yall[msk], p_all[msk], t_win)
                rec.update(model=mname, feature_set=fset, split=sname,
                           level='window')
                results.append(rec)
                yt, yp = minute_agg(msk, p_all[msk])
                rec = metrics(yt, yp, t_min)
                rec.update(model=mname, feature_set=fset, split=sname,
                           level='minute')
                results.append(rec)
        print(f'  done: {fset}', flush=True)
    res = pd.DataFrame(results)
    res.to_csv(OUT / 'metrics.csv', index=False)
    print(res[res.level == 'minute'].sort_values(
        ['split', 'balanced_acc'], ascending=False)
        .head(20).to_string(index=False), flush=True)
    return res


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['extract', 'features', 'models',
                                      'all'])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase in ('extract', 'all'):
        run_extract(workers=a.workers, limit=a.limit)
    if a.phase in ('features', 'all'):
        run_features()
    if a.phase in ('models', 'all'):
        run_models()


if __name__ == '__main__':
    main()
