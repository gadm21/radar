"""Shared constants and helpers for the E1 class-separability stage.

E1 (redefined): physical descriptors + features that separate the two
classes (empty vs occupied), applied uniformly to

  * ``multilink_train/``  — the Oct 3-5 captures (2 CSI links on
    thoth-chen/thoth-toronto + BGT60TR13C radar, JSONL streams), and
  * the legacy minutes (``train_minutes/train`` + ``test_minutes``) —
    one CSI link + radar, via the E1 pass-A cache produced by
    ``_extract_old.py`` (E2's per-minute products) and E2's published
    feature tables (``E2/outputs/occupancy/features_{5,10}s.csv``).

Per-second products (the "dots"):
  CSI   : 104-d [mean_52 | std_52] amplitude summary per second.
  radar : 576-d log1p mean range-Doppler map per second.
Both are projected with a 3-D PCA and the within-window variance of
the projected trajectory is the compressed-PCA descriptor (pcv1-3).
"""
import base64
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # .../radar
E1_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = E1_DIR / "outputs"
FIGS_DIR = OUTPUT_DIR / "figs"
CACHE = E1_DIR / "cache"
ML_DIR = ROOT / "multilink_train"
ML_CACHE = CACHE / "multilink"
OLD_CACHE = CACHE / "win"                               # from _extract_old.py
OLD_FEATS = ROOT / "E2" / "outputs" / "occupancy"      # features_*.csv
OLD_PCAS = OLD_FEATS / "pcas.joblib"

N_SUB = 52
CSI_SUBCARRIER_MASK = np.array(
    [False] * 6 + [True] * 26 + [False] + [True] * 26 + [False] * 5,
    dtype=bool)

WIN_LIST = (5, 10)
PCA_DIM = 3
MIN_COV = 0.8                    # fraction of valid seconds per window
VIEW_NAMES = ("rd", "ra", "re", "xy")
DESC_STATS = ("mean", "p90", "std90", "delta", "peak")

# ---------------------------------------------------------------------------
# feature-name registry (shared vocabulary; per-dataset prefixes differ)
# ---------------------------------------------------------------------------
RADAR_VIEW_COLS = [f"{v}_{s}" for v in VIEW_NAMES for s in DESC_STATS]
RADAR_PHYS = ["rd_range_cm", "rd_dop_cm", "rd_dop_spread", "rd_dca",
              "rd_energy"]
RADAR_COLS = (["snr_max", "snr_mean"]
              + [f"rad_pcv{i}" for i in range(1, PCA_DIM + 1)]
              + RADAR_VIEW_COLS + RADAR_PHYS)

# per-link CSI descriptors (multilink: csi1_* / csi2_*; legacy: csi_*)
CSI_LINK_COLS = (["amp_mean", "amp_std", "amp_q90",
                  "rv_mean", "rv_q90", "rv_max",
                  "tv", "dop_frac"]
                 + [f"pcv{i}" for i in range(1, PCA_DIM + 1)])


def csi_cols(prefix):
    """CSI feature names for one link, e.g. csi_cols('csi1_')."""
    return [prefix + c for c in CSI_LINK_COLS]


# ---------------------------------------------------------------------------
# JSONL decoding
# ---------------------------------------------------------------------------
def decode_csi_iq(payload):
    """payload dict -> complex64 [52] via the int8 'data' field."""
    b = base64.b64decode(payload["data"])
    v = np.frombuffer(b, np.int8).astype(np.float32)
    imag = v[0::2][CSI_SUBCARRIER_MASK]
    real = v[1::2][CSI_SUBCARRIER_MASK]
    return real + 1j * imag


def decode_radar_views(payload):
    """payload dict -> f32 views [4,24,24] (log1p power maps)."""
    b = base64.b64decode(payload["views_b64"])
    return np.frombuffer(b, np.float16).reshape(4, 24, 24)


def iter_jsonl(path):
    """Yield (ts, seq, payload_dict) rows from a sensor JSONL file."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            yield float(d["ts"]), int(d.get("seq", 0)), d["payload"]


def load_manifest(cap_dir):
    return json.loads((cap_dir / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# window-level radar view descriptors (shared with E2 formulas)
# ---------------------------------------------------------------------------
def view_descriptors(maps_window):
    """maps_window: (n,4,24,24) log1p maps -> dict of 20 descriptors."""
    out = {}
    for vi, v in enumerate(VIEW_NAMES):
        mp = maps_window[:, vi].astype(np.float32)
        mean_img = mp.mean(axis=0)
        std_img = mp.std(axis=0)
        delta = (np.abs(np.diff(mp, axis=0)).mean()
                 if len(mp) > 1 else 0.0)
        out[f"{v}_mean"] = float(mean_img.mean())
        out[f"{v}_p90"] = float(np.quantile(mean_img, 0.9))
        out[f"{v}_std90"] = float(np.quantile(std_img, 0.9))
        out[f"{v}_delta"] = float(delta)
        out[f"{v}_peak"] = float(mean_img.max())
    return out


# physical descriptors on the mean RD power map (power, not log)
_DC_ROWS = (10, 14)            # masked clutter rows (E2 convention)
_NEAR = 2                      # masked near-range bins


def rd_phys_descriptors(mean_rd_log):
    """Physical RD descriptors from a 24x24 log1p mean map.

    rd_range_cm  power-weighted range-bin centroid (row, 0..23)
    rd_dop_cm    power-weighted doppler-bin centroid (col, 0..23,
                 12 = DC/clutter centre)
    rd_dop_spread  power-weighted std of the doppler coordinate
    rd_dca       density of changed area — fraction of unmasked bins
                 above the 90th percentile (hot-pixel area)
    rd_energy    total exmp1 power of the unmasked map
    """
    p = np.expm1(mean_rd_log.astype(np.float64))
    mask = np.ones_like(p, bool)
    mask[_DC_ROWS[0]:_DC_ROWS[1], :] = False
    mask[:, :_NEAR] = False
    pm = np.where(mask, p, 0.0)
    tot = pm.sum() + 1e-12
    rr = np.arange(24)[:, None] * np.ones((1, 24))
    dd = np.ones((24, 1)) * np.arange(24)[None, :]
    rcm = float((pm * rr).sum() / tot)
    dcm = float((pm * dd).sum() / tot)
    dspread = float(np.sqrt((pm * (dd - dcm) ** 2).sum() / tot))
    nz = p[mask]
    dca = float((p[mask] > np.quantile(nz, 0.9)).mean()) if nz.size else 0.
    return {"rd_range_cm": rcm, "rd_dop_cm": dcm,
            "rd_dop_spread": dspread, "rd_dca": dca,
            "rd_energy": float(tot)}


def rollvar(x, w):
    """Causal trailing variance along axis 0 (E2 _rollvar)."""
    z = np.zeros((1, x.shape[1]), dtype=np.float64)
    cs = np.concatenate([z, np.cumsum(x, axis=0)], axis=0)
    cs2 = np.concatenate([z, np.cumsum(x * x, axis=0)], axis=0)
    hi = np.arange(1, x.shape[0] + 1)
    lo = np.clip(hi - w, 0, None)
    cnt = (hi - lo).astype(np.float64)[:, None]
    mean = (cs[hi] - cs[lo]) / cnt
    msq = (cs2[hi] - cs2[lo]) / cnt
    return np.clip(msq - mean * mean, 0, None)


def interp_seconds(mat, ok):
    """Per-column linear interp across invalid rows (E2 _interp_seconds)."""
    xs = np.flatnonzero(ok)
    if len(xs) == 0:
        return None
    out = np.empty(mat.shape, np.float32)
    grid = np.arange(len(mat))
    for c in range(mat.shape[1]):
        out[:, c] = np.interp(grid, xs, mat[xs, c])
    return out


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    sp = np.sqrt(((a.var(ddof=1) * (len(a) - 1))
                  + (b.var(ddof=1) * (len(b) - 1)))
                 / max(len(a) + len(b) - 2, 1))
    return float((b.mean() - a.mean()) / (sp + 1e-12))
