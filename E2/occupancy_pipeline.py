"""Occupancy pipeline for the reframed multi-modal paper.

Protocol (per user request):
  * Two partitions: train = train_minutes/ (calibration_minutes at wall1
    Aug 17-18 + train at wall2 Sept 6-8), test = test_minutes/ (wall3,
    Sept 15). Labels: empty vs occupied (sleep folds into occupied).
  * Cleaning: CSI reduced 64 -> 52 subcarriers (common.CSI_SUBCARRIER_MASK),
    CSI resampled to a guaranteed 100 Hz grid with the upstream
    WiFiSensing routine verbatim; minutes missing either modality are
    dropped.
  * Processing: non-overlapping 5 s and 10 s windows. Radar features:
    clutter-masked SNR + RD/RA/RE/XY log-power map descriptors + variance
    of per-second 3-D PCA projections of the mean RD map. CSI features:
    raw amplitude stats, causal rolling variance (0.2 s), variance of
    per-second 3-D PCA projections of the mean/std "dots".
  * Balancing: minority-class (empty) upsampling on train minutes via
    Gaussian noise / gain drift / mixup / temporal warp; majority-class
    downsampling on test minutes. Both at minute level.
  * Training: RF / XGBoost on csi / radar / fusion feature groups plus a
    mixture-of-experts combiner; DL encoders (1-D CNN, LSTM) per modality
    and a dual-encoder + decoder-classifier fusion (see occ_dl.py).
    A stratified 85/15 train/validation split on the training minutes is
    used for early stopping and threshold selection; test is only
    evaluated once.

Usage:
    python E2/occupancy_pipeline.py extract --workers 6
    python E2/occupancy_pipeline.py features
    python E2/occupancy_pipeline.py models
    python E2/occupancy_pipeline.py all --workers 6
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.OUTPUT_DIR / 'occupancy'
CACHE = C.E2_DIR / 'cache' / 'win'
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SUB = 52
SEC_HZ = 100                    # guaranteed CSI grid
WIN_LIST = (5, 10)              # window sizes under test
ROLL_W = 20                     # 0.2 s causal rolling-variance window
MIN_RADAR_FRAMES = {5: 5, 10: 8}  # min frames per window
VAL_FRAC = 0.15
PCA_DIM = 3                     # 3-D projection space, both modalities

_DC_ROWS = (10, 14)             # masked doppler rows on the 24x24 RD map
_NEAR_RANGE = 2

VIEW_NAMES = ['rd', 'ra', 're', 'xy']
DESC_STATS = ['mean', 'p90', 'std90', 'delta', 'peak']

CSI_AMP = ['csi_amp_mean', 'csi_amp_std', 'csi_amp_q90']
CSI_RV = ['csi_rv_mean', 'csi_rv_q90', 'csi_rv_max']
CSI_PCVAR = [f'csi_pcv{i}' for i in range(1, PCA_DIM + 1)]
SNR = ['snr_max', 'snr_mean']
RAD_PCVAR = [f'rad_pcv{i}' for i in range(1, PCA_DIM + 1)]
VIEW_COLS = {v: [f'{v}_{s}' for s in DESC_STATS] for v in VIEW_NAMES}

CSI_COLS = CSI_AMP + CSI_RV + CSI_PCVAR + ['csi_ok']
RADAR_COLS = (SNR + RAD_PCVAR
              + sum((VIEW_COLS[v] for v in VIEW_NAMES), []) + ['rad_ok'])
FUSION_COLS = CSI_COLS + RADAR_COLS

FEATURE_SETS = {
    'csi': CSI_COLS,
    'radar': RADAR_COLS,
    'fusion': FUSION_COLS,
}

AUG_KINDS = ('gauss', 'gain', 'mixup', 'warp')

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
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != bad]
    return R100.resample(amp, rel)


def _csi_minute(rec):
    """Return dict with amp60[60,100,52] f16, sec_valid[60],
    dots[60,104] f32 and packet-rate stats."""
    out = {
        'amp60': np.full((60, SEC_HZ, N_SUB), np.nan, np.float16),
        'sec_valid': np.zeros(60, bool),
        'dots': np.full((60, 104), np.nan, np.float32),
        'csi_hz': 0.0, 'csi_npkts': 0,
    }
    csi, ts = C.recording_csi(rec)
    if csi is None or len(csi) < 2:
        return out
    rel = ts - rec.start_ts
    keep = (rel >= 0) & (rel < 60) & np.isfinite(csi).all(axis=1)
    rel, amp = rel[keep], np.abs(csi[keep]).astype(np.float64)
    rel, uniq = np.unique(rel, return_index=True)
    amp = amp[uniq]
    out['csi_npkts'] = int(len(rel))
    out['csi_hz'] = float(len(rel) / 60.0)
    if len(rel) < 2:
        return out
    uniform, t, spb, _ = _resample_100hz(amp, rel)
    raw_bins = np.floor(rel).astype(int)
    bins = np.floor(t).astype(int)
    for sec in range(60):
        observed = rel[raw_bins == sec]
        if (len(observed) < 5 or observed[-1] - observed[0] < .5
                or np.diff(np.r_[sec, observed, sec + 1]).max() > .5):
            continue
        s = uniform[bins == sec]
        a = s[:, :N_SUB] if s.ndim == 2 else s[:N_SUB]
        if len(s) != SEC_HZ:
            continue
        out['amp60'][sec] = a.astype(np.float16)
        out['dots'][sec] = np.r_[a.mean(axis=0), a.std(axis=0)]
        out['sec_valid'][sec] = True
    return out


def _radar_minute(rec):
    """Decode frames -> per-frame maps (n,4,24,24) f16 [rd,ra,re,xy],
    snr per frame, relative frame times, per-second mean RD maps."""
    from preprocess import read_uint12_batch
    out = {
        'maps': np.zeros((0, 4, 24, 24), np.float16),
        'rd12': np.zeros((0, 12, 12), np.float16),
        'snr_f': np.zeros(0, np.float32),
        'rts': np.zeros(0, np.float64),
        'rd_sec': np.zeros((60, 24 * 24), np.float16),
        'sec_has_radar': np.zeros(60, bool),
        'radar_hz': 0.0,
    }
    payloads, rts = C.recording_radar_frames(rec)
    rel = np.asarray(rts) - rec.start_ts
    keep = [i for i, p in enumerate(payloads)
            if 0 <= rel[i] < 60 and len(p) == 36864]
    if not keep:
        return out
    payloads = [payloads[i] for i in keep]
    rel = rel[keep]
    out['radar_hz'] = float(len(keep) / 60.0)
    out['rts'] = rel
    views = {v: [] for v in VIEW_NAMES}
    rd12 = []                               # low-res RD for DL input
    for off in range(0, len(payloads), 16):
        batch = payloads[off:off + 16]
        a = read_uint12_batch(b''.join(batch)).reshape(-1, 64, 128, 3)
        a -= a.mean(axis=2, keepdims=True)
        a *= _HANN_R[None, None, :, None]
        R = fft.fft(a, axis=2)[:, :, :64, :]
        D = fft.fftshift(fft.fft(R * _HANN_D[None, :, None, None],
                                 axis=1, workers=-1), axes=1)
        rd = np.mean(np.abs(D) ** 2, axis=3)              # (n,64,64)
        rd[:, :, :_NEAR_RANGE] = 0
        views['rd'].extend(zoom(np.log1p(rd), (1, 24 / 64, 24 / 64),
                                order=1).astype(np.float32))
        rd12.extend(zoom(np.log1p(rd), (1, 12 / 64, 12 / 64),
                         order=1).astype(np.float16))
        xy = np.zeros((len(batch), 24, 24), np.float32)
        for key, left in (('ra', 0), ('re', 1)):
            p = (np.abs(R[:, :, :, left]) ** 2
                 + np.abs(R[:, :, :, 2]) ** 2).mean(axis=1)
            cross = (R[:, :, :, left] * R[:, :, :, 2].conj()).mean(axis=1)
            energy = np.maximum(
                p[:, :, None] + 2 * np.real(cross[:, :, None] * _PHASE), 0)
            energy[:, :2, :] = 0
            views[key].extend(
                zoom(np.log1p(energy), (1, 24 / 64, 24 / 25),
                     order=1).astype(np.float32))
            if key == 'ra':
                for j in range(len(batch)):
                    np.maximum.at(xy[j], (_IY.ravel(), _IX.ravel()),
                                  energy[j].ravel())
        views['xy'].extend(np.log1p(xy))
    maps = np.stack([np.asarray(views[v], np.float16) for v in VIEW_NAMES],
                    axis=1)                                  # (n,4,24,24)
    out['maps'] = maps
    out['rd12'] = np.asarray(rd12, np.float16)               # (n,12,12)

    # clutter-masked SNR proxy per frame on the 24x24 RD power map
    m = np.zeros((24, 24), bool)
    m[_DC_ROWS[0]:_DC_ROWS[1], :] = True
    m[:, :_NEAR_RANGE] = True
    pwr = np.expm1(maps[:, 0].astype(np.float32)).reshape(len(maps), -1)
    med = np.median(pwr, axis=1) + 1e-9
    peak = np.where(m.reshape(-1)[None], 0.0, pwr).max(axis=1)
    out['snr_f'] = (10.0 * np.log10(peak / med)).astype(np.float32)

    # per-second mean RD map for the 1-second PCA projections
    sec_idx = np.clip(np.floor(rel).astype(int), 0, 59)
    for sec in range(60):
        sel = sec_idx == sec
        if sel.any():
            out['rd_sec'][sec] = maps[sel, 0].mean(axis=0).reshape(-1)
            out['sec_has_radar'][sec] = True
    return out


def extract_minute(rec):
    """Pass-A cache for one minute. Returns (rec_id, status)."""
    path = CACHE / rec.source / f'{rec.folder.name}.npz'
    if path.exists():
        return rec.rec_id, 'cached'
    try:
        csid = _csi_minute(rec)
        radd = _radar_minute(rec)
    except Exception as e:
        return rec.rec_id, f'error: {e}'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        amp60=csid['amp60'], sec_valid=csid['sec_valid'], dots=csid['dots'],
        csi_hz=np.float32(csid['csi_hz']),
        csi_npkts=np.int32(csid['csi_npkts']),
        maps=radd['maps'], rd12=radd['rd12'],
        snr_f=radd['snr_f'], rts=radd['rts'],
        rd_sec=radd['rd_sec'], sec_has_radar=radd['sec_has_radar'],
        radar_hz=np.float32(radd['radar_hz']),
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
                    errs.append((rid, res))
                else:
                    stats['ok'] += 1
                if (i + 1) % 50 == 0:
                    dt = time.time() - t0
                    print(f'extract {i + 1}/{len(todo)} '
                          f'({dt / 60:.1f} min)', flush=True)
    else:
        for i, r in enumerate(todo):
            rid, res = _extract_worker(r)
            if res.startswith('error'):
                errs.append((rid, res))
            else:
                stats['ok'] += 1
            if (i + 1) % 50 == 0:
                print(f'extract {i + 1}/{len(todo)}', flush=True)
    print(f"done: {stats}, errors={len(errs)}", flush=True)
    for rid, res in errs[:20]:
        print(' ', rid, res, flush=True)


# ---------------------------------------------------------------------------
# Pass B: index, PCAs, per-window features
# ---------------------------------------------------------------------------
def load_index():
    """Minute-level index from pass-A caches (all sources)."""
    rows = []
    for src_dir in sorted(CACHE.iterdir()):
        if not src_dir.is_dir():
            continue
        for p in sorted(src_dir.glob('*.npz')):
            z = np.load(p)
            meta = json.loads(str(z['meta']))
            rows.append({
                'rec_id': meta['rec_id'], 'source': src_dir.name,
                'label': meta['label'], 'placement': meta['placement'],
                'session_id': meta['session_id'],
                'has_csi': bool(z['sec_valid'].sum() > 0),
                'has_radar': bool(len(z['snr_f']) >= 5),
                'csi_hz': float(z['csi_hz']),
                'radar_hz': float(z['radar_hz']),
                'n_secs': int(z['sec_valid'].sum()),
                'n_frames': int(len(z['snr_f'])),
                'npz': str(p),
            })
    df = pd.DataFrame(rows)
    df['dataset'] = np.where(df.source.isin(C.TRAIN_SOURCES),
                             'train', 'test')
    return df


def split_train_val(df):
    """Stratified 85/15 split of train minutes by (placement,label)."""
    rng = np.random.default_rng(SEED)
    tr = df[df.dataset == 'train']
    val_idx = []
    for _, g in tr.groupby(['placement', 'label']):
        n_val = max(1, int(round(len(g) * VAL_FRAC)))
        val_idx.extend(rng.choice(g.index, size=n_val, replace=False))
    df = df.copy()
    df['is_val'] = df.index.isin(val_idx) & (df.dataset == 'train')
    return df


def _fit_pcas(df):
    """Fit per-second 3-D PCAs on real training minutes."""
    from sklearn.decomposition import PCA
    dots, rds = [], []
    for p in df[(df.dataset == 'train') & ~df.is_val].npz:
        z = np.load(p)
        m = z['sec_valid']
        if m.any():
            dots.append(z['dots'][m])
        m2 = z['sec_has_radar']
        if m2.any():
            rds.append(z['rd_sec'][m2].astype(np.float32))
    dots = np.concatenate(dots)
    rds = np.concatenate(rds)
    pca_csi = PCA(n_components=PCA_DIM, random_state=SEED).fit(dots)
    pca_rad = PCA(n_components=PCA_DIM, random_state=SEED).fit(rds)
    ev_csi = pca_csi.explained_variance_ratio_
    ev_rad = pca_rad.explained_variance_ratio_
    print(f'csi PCA ev={ev_csi.round(3)}  radar PCA ev={ev_rad.round(3)}')
    return pca_csi, pca_rad


def _rollvar(x, w):
    """Causal trailing variance along axis 0 of x: (T,S) -> (T,S)."""
    z = np.zeros((1, x.shape[1]), dtype=np.float64)
    cs = np.concatenate([z, np.cumsum(x, axis=0)], axis=0)
    cs2 = np.concatenate([z, np.cumsum(x * x, axis=0)], axis=0)
    hi = np.arange(1, x.shape[0] + 1)
    lo = np.clip(hi - w, 0, None)
    cnt = (hi - lo).astype(np.float64)[:, None]
    mean = (cs[hi] - cs[lo]) / cnt
    msq = (cs2[hi] - cs2[lo]) / cnt
    return np.clip(msq - mean * mean, 0, None)


def _interp_seconds(mat, ok):
    """Per-column linear interpolation across invalid seconds of a
    (60, D) per-second matrix; edges take the nearest valid value.
    Returns None when no second is valid."""
    xs = np.flatnonzero(ok)
    if len(xs) == 0:
        return None
    out = np.empty(mat.shape, np.float32)
    grid = np.arange(len(mat))
    for c in range(mat.shape[1]):
        out[:, c] = np.interp(grid, xs, mat[xs, c])
    return out


def _ffill_rows(rows):
    """Fill all-NaN rows with the next finite row; trailing gaps use the last.

    This historical batch operation looks ahead and is not causal.
    """
    valid = np.isfinite(rows).all(axis=1)
    if not valid.any():
        return None
    vi = np.flatnonzero(valid)
    fill = np.clip(np.searchsorted(vi, np.arange(len(rows))),
                   0, len(vi) - 1)
    return rows[vi[fill]]


MIN_COV = 0.8      # minimum fraction of valid seconds per window


def minute_windows(z, meta, W, pca_csi, pca_rad):
    """Feature rows for one minute at window size W.
    A window is usable when >=80% of its seconds have native coverage;
    per-second products are interpolated across gaps and amplitude
    rows are forward-filled for the rolling-variance features.
    Returns (X[nw, len(FUSION_COLS)] f32, ) including *_ok flags."""
    nw = 60 // W
    X = np.full((nw, len(FUSION_COLS)), np.nan, np.float32)

    amp60, sec_valid, dots = z['amp60'], z['sec_valid'], z['dots']
    maps, snr_f, rts = z['maps'], z['snr_f'], z['rts']
    rd_sec, sec_has_radar = z['rd_sec'], z['sec_has_radar']

    dots_f = _interp_seconds(dots, sec_valid)
    csi_proj = (pca_csi.transform(dots_f) if dots_f is not None
                else np.full((60, PCA_DIM), np.nan, np.float32))
    rds_f = _interp_seconds(rd_sec.astype(np.float32), sec_has_radar)
    rad_proj = (pca_rad.transform(rds_f) if rds_f is not None
                else np.full((60, PCA_DIM), np.nan, np.float32))
    rows_f = _ffill_rows(amp60.reshape(-1, N_SUB).astype(np.float32))

    for w in range(nw):
        s0, s1 = w * W, (w + 1) * W
        t0, t1 = float(w * W), float((w + 1) * W)
        row = 0

        # --- CSI ---
        csi_ok = sec_valid[s0:s1].mean() >= MIN_COV
        X[w, FUSION_COLS.index('csi_ok')] = float(csi_ok)
        if csi_ok:
            a = amp60[s0:s1].astype(np.float32).reshape(-1, N_SUB)
            with np.errstate(all='ignore'):
                X[w, 0] = np.nanmean(a)
                X[w, 1] = np.nanstd(a)
                X[w, 2] = np.nanquantile(a, 0.9)
            rv = np.log1p(_rollvar(
                np.log1p(rows_f[s0 * SEC_HZ:s1 * SEC_HZ]), ROLL_W))
            X[w, 3] = rv.mean()
            X[w, 4] = np.quantile(rv, 0.9)
            X[w, 5] = rv.max()
            X[w, 6:6 + PCA_DIM] = np.var(csi_proj[s0:s1], axis=0)
        row = len(CSI_COLS)

        # --- radar ---
        sel = np.flatnonzero((rts >= t0) & (rts < t1))
        rad_ok = len(sel) >= MIN_RADAR_FRAMES[W]
        X[w, row + len(RADAR_COLS) - 1] = float(rad_ok)
        if rad_ok:
            X[w, row] = float(np.nanmax(snr_f[sel]))
            X[w, row + 1] = float(np.nanmean(snr_f[sel]))
            off = row + len(SNR) + PCA_DIM
            for vi, v in enumerate(VIEW_NAMES):
                mp = maps[sel, vi].astype(np.float32)
                mean_img = mp.mean(axis=0)
                std_img = mp.std(axis=0)
                delta = np.abs(np.diff(mp, axis=0)).mean() if len(mp) > 1 else 0.
                X[w, off:off + 5] = [mean_img.mean(),
                                     np.quantile(mean_img, 0.9),
                                     np.quantile(std_img, 0.9),
                                     delta, mean_img.max()]
                off += 5
            if sec_has_radar[s0:s1].mean() >= MIN_COV \
                    and rds_f is not None:
                X[w, row + len(SNR):row + len(SNR) + PCA_DIM] = \
                    np.var(rad_proj[s0:s1], axis=0)
    return X


def _augment_minute(z, kind, rng):
    """Return augmented copies of the cached arrays (empty-minute
    upsampling). Operates on amp60/maps/snr_f; sec_valid kept."""
    amp = z['amp60'].astype(np.float32).copy()
    maps = z['maps'].astype(np.float32).copy()
    rd12 = z['rd12'].astype(np.float32).copy() if len(z['rd12']) \
        else np.zeros((0, 12, 12), np.float32)
    snr = z['snr_f'].copy()
    rts = z['rts'].copy()
    if kind == 'gauss':
        amp += rng.normal(0, 0.05 * np.nanstd(amp), amp.shape)
        mstd = 0.05 * maps.std()
        maps += rng.normal(0, mstd, maps.shape)
        if len(rd12):
            rd12 += rng.normal(0, 0.05 * rd12.std(), rd12.shape)
    elif kind == 'gain':
        g = rng.uniform(0.8, 1.25, (1, 1, N_SUB)).astype(np.float32)
        amp *= g
        rg = rng.uniform(0.8, 1.25, (1, 4, 24, 1)).astype(np.float32)
        maps = np.log1p(np.expm1(maps) * rg)
        if len(rd12):
            rd12 = np.log1p(
                np.expm1(rd12)
                * rng.uniform(0.8, 1.25, (1, 12, 1)).astype(np.float32))
    elif kind == 'mixup':
        # expects rng.partner dict of another minute's arrays
        p = rng.partner
        lam = rng.uniform(0.25, 0.75)
        mix = lam * amp + (1 - lam) * p['amp60'].astype(np.float32)
        amp = np.where(np.isfinite(mix), mix, amp)
        n = min(len(maps), len(p['maps']))
        maps = maps[:n].copy()
        pm = p['maps'].astype(np.float32)
        idx = np.linspace(0, len(pm) - 1, n).astype(int)
        maps = lam * maps + (1 - lam) * pm[idx]
        snr = lam * snr[:n] + (1 - lam) * p['snr_f'][idx]
        rts = rts[:n]
        if len(rd12) and len(p['rd12']):
            pr = p['rd12'].astype(np.float32)
            rd12 = rd12[:n]
            rd12 = lam * rd12 + (1 - lam) * pr[
                np.linspace(0, len(pr) - 1, n).astype(int)]
    elif kind == 'warp':
        scale = rng.uniform(0.85, 1.15)
        flat = amp.reshape(60 * SEC_HZ, N_SUB)
        src_t = np.clip(np.arange(60 * SEC_HZ) * scale,
                        0, 60 * SEC_HZ - 1.0)
        grid = np.arange(60 * SEC_HZ)
        for ch in range(N_SUB):
            flat[:, ch] = np.interp(grid, src_t, flat[:, ch])
        amp = flat.reshape(60, SEC_HZ, N_SUB)
        nsel = np.clip((np.arange(len(maps)) * scale).astype(int),
                       0, len(maps) - 1)
        maps, snr, rts = maps[nsel], snr[nsel], rts[nsel]
        if len(rd12):
            rd12 = rd12[np.clip((np.arange(len(rd12)) * scale).astype(int),
                                0, len(rd12) - 1)]
    # recompute per-second products from the augmented arrays so the
    # PCA-projection features stay consistent
    with np.errstate(all='ignore'):
        dots = np.concatenate([np.nanmean(amp, axis=1),
                               np.nanstd(amp, axis=1)], axis=1)
    dots = dots.astype(np.float32)
    # seconds whose per-second stats are not fully finite are invalid
    sec_valid = np.isfinite(dots).all(axis=1)
    rd_sec = np.zeros((60, 24 * 24), np.float32)
    has = np.zeros(60, bool)
    if len(rts):
        sec_idx = np.clip(np.floor(rts).astype(int), 0, 59)
        for sec in range(60):
            mm = sec_idx == sec
            if mm.any():
                rd_sec[sec] = maps[mm, 0].mean(axis=0).reshape(-1)
                has[sec] = True
    z2 = dict(z)
    z2.update(amp60=amp.astype(np.float16), maps=maps.astype(np.float16),
              rd12=rd12.astype(np.float16), snr_f=snr, rts=rts,
              dots=dots, rd_sec=rd_sec, sec_valid=sec_valid,
              sec_has_radar=has)
    return z2


class _Rng(dict):
    """numpy Generator that also carries an optional partner cache."""
    def __init__(self, seed):
        super().__init__()
        self._g = np.random.default_rng(seed)
        self.partner = None

    def __getattr__(self, k):
        return getattr(self._g, k)


def plan_augment(clean):
    """Deterministic minute-level upsampling plan for the train-fit
    minority (empty) class. Returns a list of spec dicts; the same plan
    is reused by the feature and DL stages."""
    tr = clean[(clean.dataset == 'train') & ~clean.is_val]
    n_min = int((tr.label == 0).sum())
    n_maj = int((tr.label == 1).sum())
    need = max(0, n_maj - n_min)
    empties = tr[tr.label == 0].reset_index(drop=True)
    specs = []
    for k in range(need):
        src = empties.iloc[k % len(empties)]
        seed = SEED + 7919 * k
        kind = AUG_KINDS[k % len(AUG_KINDS)]
        partner_npz = None
        if kind == 'mixup':
            j = int(np.random.default_rng(seed).integers(0, len(empties)))
            partner_npz = empties.iloc[j].npz
        specs.append({'aug_id': k + 1, 'rec_id': src.rec_id,
                      'npz': src.npz, 'kind': kind, 'seed': seed,
                      'n_empties': len(empties), 'partner_npz': partner_npz})
    return specs


def realize_aug(spec, z=None):
    """Materialize one augmented minute dict from a plan spec."""
    if z is None:
        z = np.load(spec['npz'])
    rng = _Rng(spec['seed'])
    if spec['kind'] == 'mixup':
        rng.integers(0, spec['n_empties'])          # consume, keep parity
        rng.partner = dict(np.load(spec['partner_npz']))
    return _augment_minute(z, spec['kind'], rng)


def run_features():
    df = load_index()
    df = split_train_val(df)
    df['usable'] = df.has_csi & df.has_radar
    df.to_csv(OUT / 'minutes_index.csv', index=False)
    print(df.groupby(['dataset', 'label', 'usable']).size())

    clean = df[df.usable].copy()
    pca_csi, pca_rad = _fit_pcas(clean)
    import joblib
    joblib.dump({'csi': pca_csi, 'rad': pca_rad}, OUT / 'pcas.joblib')

    for W in WIN_LIST:
        frames = []
        for _, r in clean.iterrows():
            z = np.load(r.npz)
            X = minute_windows(z, r.to_dict(), W, pca_csi, pca_rad)
            n = X.shape[0]
            frames.append(pd.DataFrame(
                X, columns=FUSION_COLS).assign(
                rec_id=r.rec_id, label=r.label, dataset=r.dataset,
                is_val=r.is_val, aug=False, aug_id=0,
                win=np.arange(n)))
        # minute-level minority upsampling on the *train fit* split
        for spec in plan_augment(clean):
            z2 = realize_aug(spec)
            X = minute_windows(z2, {}, W, pca_csi, pca_rad)
            frames.append(pd.DataFrame(
                X, columns=FUSION_COLS).assign(
                rec_id=spec['rec_id'] + f"#aug{spec['aug_id'] - 1}",
                label=0, dataset='train', is_val=False, aug=True,
                aug_id=spec['aug_id'], win=np.arange(X.shape[0])))
        feat = pd.concat(frames, ignore_index=True)
        feat.to_csv(OUT / f'features_{W}s.csv', index=False)
        print(f'W={W}s: {len(feat)} windows '
              f'(aug {int(feat.aug.sum())})', flush=True)
    return df


# ---------------------------------------------------------------------------
# Pass C: models
# ---------------------------------------------------------------------------
def _metrics(y, prob, thr=0.5):
    """Metrics at the val-tuned threshold plus two unsupervised
    self-calibration variants (no labels: mimics deployment-time
    calibration to a new placement):
      - acc_cal : threshold at the median test probability
                  (assumes roughly balanced occupancy);
      - acc_km  : threshold from 2-means clustering of the test
                  probabilities (midpoint of centroids). It does not
                  require a 50/50 mix, but can fail under imbalance,
                  multimodal scores, or reversed score orientation."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    pred = (prob >= thr).astype(int)
    pred_c = (prob >= np.median(prob)).astype(int)
    if np.unique(prob).size > 2:
        ctr = KMeans(n_clusters=2, n_init=10,
                     random_state=SEED).fit(prob.reshape(-1, 1)) \
                    .cluster_centers_.ravel()
        thr_k = float(np.sort(ctr).mean())
    else:
        thr_k = float(np.median(prob))
    pred_k = (prob >= thr_k).astype(int)
    return {'acc': accuracy_score(y, pred),
            'f1': f1_score(y, pred),
            'auc': roc_auc_score(y, prob),
            'acc_cal': accuracy_score(y, pred_c),
            'f1_cal': f1_score(y, pred_c),
            'acc_km': accuracy_score(y, pred_k),
            'f1_km': f1_score(y, pred_k)}


def _best_thr(y, prob):
    """Balanced-accuracy optimal threshold; ties take the median
    candidate (robust to saturated probabilities)."""
    from sklearn.metrics import balanced_accuracy_score
    c = np.linspace(0.05, 0.95, 91)
    ba = np.array([balanced_accuracy_score(y, prob >= t) for t in c])
    best = np.flatnonzero(ba == ba.max())
    return float(np.median(c[best]))


def _minute_table(frame, prob_col):
    return (frame.groupby('rec_id')
            .agg(label=('label', 'first'), prob=(prob_col, 'mean')))


def run_models(dl=True, dl_epochs=40, n_seeds=5, dl_seeds=3,
               dump_probs=False):
    """When dump_probs=True, collect minute-level test probabilities
    per (W, model, features, seed) into test_probs.npz and skip writing
    results (used for the ROC figure)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from xgboost import XGBClassifier

    all_rows = []
    probs = {}
    for W in WIN_LIST:
        feat = pd.read_csv(OUT / f'features_{W}s.csv')
        train_real = feat[(feat.dataset == 'train')]
        fit_tr = train_real[~train_real.is_val]
        fit_va = train_real[train_real.is_val]

        # balanced test set: fixed cohort (same minutes for every seed)
        test = feat[feat.dataset == 'test']
        rng = np.random.default_rng(SEED)
        tmins = test.groupby('rec_id').label.first()
        cnt = tmins.value_counts()
        keep_min = cnt.min()
        keep = []
        for lab, g in tmins.groupby(tmins):
            keep.extend(rng.choice(g.index, size=min(keep_min, len(g)),
                                   replace=False))
        test_bal = test[test.rec_id.isin(keep)]
        yte = test_bal.label.values
        min_id = test_bal.rec_id.values
        print(f'W={W}s test minutes kept: {len(keep)} of {len(tmins)}')

        sets = {k: [c for c in v] for k, v in FEATURE_SETS.items()}
        for seed in range(SEED, SEED + n_seeds):
            proba = {}    # (model, split) -> prob on windows
            for setname, cols in sets.items():
                Xtr = fit_tr[cols].values
                ytr = fit_tr.label.values
                Xva = fit_va[cols].values
                Xte = test_bal[cols].values
                learners = {
                    'rf': RandomForestClassifier(
                        n_estimators=300,
                        class_weight='balanced_subsample',
                        n_jobs=-1, random_state=seed),
                    'xgb': XGBClassifier(
                        n_estimators=400, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                        random_state=seed, eval_metric='logloss'),
                }
                for mname, clf in learners.items():
                    model = make_pipeline(
                        SimpleImputer(strategy='median'),
                        StandardScaler(), clf)
                    model.fit(Xtr, ytr)
                    proba[(setname, mname, 'val')] = (
                        model.predict_proba(Xva)[:, 1])
                    proba[(setname, mname, 'test')] = (
                        model.predict_proba(Xte)[:, 1])

            # mixture of experts: modality experts + val-fitted gate
            for mname in ('rf', 'xgb'):
                pc = proba[('csi', mname, 'val')]
                pr = proba[('radar', mname, 'val')]
                yva = fit_va.label.values
                best_w, best_f1 = 0.5, -1
                for w in np.linspace(0, 1, 21):
                    p = w * pc + (1 - w) * pr
                    f1 = f1_score(yva, p >= 0.5)
                    if f1 > best_f1:
                        best_f1, best_w = f1, w
                proba[('moe', mname, 'test')] = (
                    best_w * proba[('csi', mname, 'test')]
                    + (1 - best_w) * proba[('radar', mname, 'test')])
                proba[('moe', mname, 'val')] = (
                    best_w * pc + (1 - best_w) * pr)
                print(f'  seed={seed} moe/{mname} w_csi={best_w:.2f}',
                      flush=True)

            # ---- window + minute metrics on the balanced test set ----
            for (setname, mname, split), p in list(proba.items()):
                if split != 'test':
                    continue
                thr = _best_thr(fit_va.label.values,
                                proba[(setname, mname, 'val')])
                met = _metrics(yte, p, thr)
                met.update(window=W, model=mname, features=setname,
                           thr=round(thr, 3), level='window', seed=seed)
                all_rows.append(met)
                # minute aggregation (threshold tuned on val minutes)
                dva = pd.DataFrame({'rec_id': fit_va.rec_id.values,
                                    'y': fit_va.label.values,
                                    'p': proba[(setname, mname, 'val')]})
                gva = dva.groupby('rec_id').agg(y=('y', 'first'),
                                              p=('p', 'mean'))
                thr_m = _best_thr(gva.y.values, gva.p.values)
                d = pd.DataFrame({'rec_id': min_id, 'y': yte, 'p': p})
                g = d.groupby('rec_id').agg(y=('y', 'first'),
                                            p=('p', 'mean'))
                metm = _metrics(g.y.values, g.p.values, thr_m)
                metm.update(window=W, model=mname, features=setname,
                            thr=round(thr_m, 3), level='minute', seed=seed)
                all_rows.append(metm)
                if dump_probs:
                    probs[(W, mname, setname, seed)] = (
                        g.y.values.astype(np.int8),
                        g.p.values.astype(np.float32))
            print(f'W={W}s seed={seed} done', flush=True)

        pd.DataFrame(all_rows).to_csv(OUT / 'results_seeds.csv',
                                     index=False)

        # ---- DL branch ----
        if dl:
            import occ_dl
            for dseed in range(SEED, SEED + dl_seeds):
                rows = occ_dl.run(
                    W, feat, OUT, epochs=dl_epochs, seed=dseed,
                    probs=probs if dump_probs else None)
                for r in rows:
                    r['seed'] = dseed
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(OUT / 'results_seeds.csv',
                                             index=False)

    if dump_probs:
        np.savez_compressed(
            OUT / 'test_probs.npz',
            **{f'{w}|{m}|{f}|{s}': np.stack([y, p])
               for (w, m, f, s), (y, p) in probs.items()})
        print(f'wrote test_probs.npz ({len(probs)} prob sets)')
        return

    # ---- aggregate mean +- std over seeds ---------------------------------
    seeds_df = pd.DataFrame(all_rows)
    seeds_df.to_csv(OUT / 'results_seeds.csv', index=False)
    mets = ['acc', 'f1', 'auc', 'acc_cal', 'f1_cal',
            'acc_km', 'f1_km']
    grp = seeds_df.groupby(['window', 'model', 'features', 'level'],
                           as_index=False)
    agg = grp[mets].mean()
    std = grp[mets].std(ddof=1).rename(columns={
        m: m + '_std' for m in mets})
    res = agg.merge(std, on=['window', 'model', 'features', 'level'])
    res['thr'] = grp['thr'].mean()['thr'].values
    res['n_seeds'] = grp.size()['size'].values
    res.to_csv(OUT / 'results.csv', index=False)
    print(res.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['extract', 'features', 'models', 'all'])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-dl', action='store_true')
    ap.add_argument('--probs', action='store_true',
                    help='dump minute-level test probabilities for ROC')
    args = ap.parse_args()
    if args.phase in ('extract', 'all'):
        run_extract(workers=args.workers, limit=args.limit)
    if args.phase in ('features', 'all'):
        run_features()
    if args.phase in ('models', 'all'):
        run_models(dl=not args.no_dl, dump_probs=args.probs,
                   n_seeds=3 if args.probs else 5,
                   dl_seeds=1 if args.probs else 3)


if __name__ == '__main__':
    main()
