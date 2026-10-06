"""Pass-B: per-window feature tables + PCA dot projections.

Multilink captures -> window rows (W=5/10) with
  csi1_* / csi2_* link descriptors, radar descriptors (E2 formulas +
  physical RD descriptors), ok flags.
Legacy minutes    -> the published E2 feature table
  (features_{5,10}s.csv, real rows only) extended with the same
  physical descriptors computed from the E1 pass-A cache.

PCAs: multilink fits its own 3-D PCAs on pooled per-second dots
(unsupervised, both classes — same convention as E2's fit on train
minutes). Legacy data reuses E2's fitted pcas.joblib so its pcv
columns and the dot projections share one subspace.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.OUTPUT_DIR
SEED = 42


# ---------------------------------------------------------------------------
# PCA fits
# ---------------------------------------------------------------------------
def fit_pcas_multilink():
    from sklearn.decomposition import PCA
    dots1, dots2, rds = [], [], []
    for p in sorted(C.ML_CACHE.glob("*.npz")):
        z = np.load(p)
        for key, acc in (("csi1", dots1), ("csi2", dots2)):
            ok = z[f"{key}_ok"]
            if ok.any():
                acc.append(z[f"{key}_dots"][ok])
        rs = z["radar_rsec"]
        if rs.any():
            rds.append(z["radar_vsec"][rs, 0].reshape(-1, 576)
                     .astype(np.float32))
    fit = {}
    for name, arr in (("csi1", dots1), ("csi2", dots2), ("rad", rds)):
        arr = np.concatenate(arr)
        p = PCA(n_components=C.PCA_DIM, random_state=SEED).fit(arr)
        fit[name] = p
        print(f"  pca {name}: ev={p.explained_variance_ratio_.round(3)} "
              f"(n={len(arr)})", flush=True)
    return fit


def load_pcas_old():
    import joblib
    d = joblib.load(C.OLD_PCAS)
    if isinstance(d, dict):
        return d.get("csi"), d.get("rad")
    return d[0], d[1]


# ---------------------------------------------------------------------------
# multilink window rows
# ---------------------------------------------------------------------------
def _csi_window_feats(z, link, W, nw, pca):
    """Return (cols dict -> array[nw]) for one CSI link."""
    ok = z[f"{link}_ok"]
    dots = z[f"{link}_dots"]
    a90 = z[f"{link}_a90"]
    rv = z[f"{link}_rv"]
    tv = z[f"{link}_tv"]
    dopf = z[f"{link}_dopf"]
    n = len(ok)
    dots_f = C.interp_seconds(dots, ok)
    proj = (pca.transform(dots_f) if dots_f is not None
            else np.full((n, C.PCA_DIM), np.nan, np.float32))
    p = f"{link}_"
    out = {f"{p}amp_mean": np.full(nw, np.nan, np.float32),
           f"{p}amp_std": np.full(nw, np.nan, np.float32),
           f"{p}amp_q90": np.full(nw, np.nan, np.float32),
           f"{p}rv_mean": np.full(nw, np.nan, np.float32),
           f"{p}rv_q90": np.full(nw, np.nan, np.float32),
           f"{p}rv_max": np.full(nw, np.nan, np.float32),
           f"{p}tv": np.full(nw, np.nan, np.float32),
           f"{p}dop_frac": np.full(nw, np.nan, np.float32),
           f"{p}ok": np.zeros(nw, np.float32)}
    for i in range(C.PCA_DIM):
        out[f"{p}pcv{i + 1}"] = np.full(nw, np.nan, np.float32)
    for w in range(nw):
        s0, s1 = w * W, min((w + 1) * W, n)
        if s0 >= n:
            break
        seg_ok = ok[s0:s1]
        if seg_ok.mean() < C.MIN_COV:
            continue
        out[f"{p}ok"][w] = 1.0
        d = dots[s0:s1][seg_ok]
        m2 = d[:, :52] ** 2 + d[:, 52:] ** 2     # E[a^2] per second
        out[f"{p}amp_mean"][w] = d[:, :52].mean()
        out[f"{p}amp_std"][w] = np.sqrt(
            max(m2.mean() - d[:, :52].mean() ** 2, 0))
        out[f"{p}amp_q90"][w] = a90[s0:s1][seg_ok].mean()
        out[f"{p}rv_mean"][w] = rv[s0:s1][seg_ok, 0].mean()
        out[f"{p}rv_q90"][w] = rv[s0:s1][seg_ok, 1].mean()
        out[f"{p}rv_max"][w] = rv[s0:s1][seg_ok, 2].max()
        out[f"{p}tv"][w] = tv[s0:s1][seg_ok].sum()
        out[f"{p}dop_frac"][w] = dopf[s0:s1][seg_ok].mean()
        for i in range(C.PCA_DIM):
            out[f"{p}pcv{i + 1}"][w] = np.var(proj[s0:s1][:, i])
    return out


def _radar_window_feats(z, W):
    """Return (cols dict -> array[nw]) from per-window aggregates."""
    S, Q, D, N = (z[f"radar_w{W}_s"], z[f"radar_w{W}_q"],
                  z[f"radar_w{W}_d"], z[f"radar_w{W}_n"])
    nw = len(N)
    rts, snr_f = z["radar_rts"], z["radar_snr_f"]
    rsec = z["radar_rsec"]
    rd_sec = z["radar_vsec"][:, 0].reshape(len(rsec), -1).astype(np.float32)
    rd_f = C.interp_seconds(rd_sec, rsec)
    out = {k: np.full(nw, np.nan, np.float32)
           for k in C.RADAR_COLS + ["rad_ok"]}
    min_frames = {5: 5, 10: 8}
    for w in range(nw):
        sel = (rts >= w * W) & (rts < (w + 1) * W)
        n_fr = int(sel.sum())
        if n_fr < min_frames.get(W, 5):
            continue
        out["rad_ok"][w] = 1.0
        out["snr_max"][w] = np.nanmax(snr_f[sel])
        out["snr_mean"][w] = np.nanmean(snr_f[sel])
        s0, s1 = w * W, min((w + 1) * W, len(rsec))
        if rsec[s0:s1].mean() >= C.MIN_COV and rd_f is not None:
            proj = _RAD_PCA.transform(rd_f[s0:s1])
            for i in range(C.PCA_DIM):
                out[f"rad_pcv{i + 1}"][w] = np.var(proj[:, i])
        n_ = max(int(N[w]), 1)
        mean_img = S[w] / n_
        std_img = np.sqrt(np.clip(Q[w] / n_ - mean_img ** 2, 0, None))
        ndiff = max(n_fr - 1, 1)
        for vi, v in enumerate(C.VIEW_NAMES):
            mi, si, di = mean_img[vi], std_img[vi], D[w][vi] / ndiff
            out[f"{v}_mean"][w] = mi.mean()
            out[f"{v}_p90"][w] = np.quantile(mi, 0.9)
            out[f"{v}_std90"][w] = np.quantile(si, 0.9)
            out[f"{v}_delta"][w] = di.mean()
            out[f"{v}_peak"][w] = mi.max()
        for k, val in C.rd_phys_descriptors(mean_img[0]).items():
            out[k][w] = val
    return out


_RAD_PCA = None     # set by build_multilink_table before use


def build_multilink_table(W, pcas):
    global _RAD_PCA
    _RAD_PCA = pcas["rad"]
    rows = []
    for p in sorted(C.ML_CACHE.glob("*.npz")):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        label = 1 if meta["label"] == "occupied" else 0
        dur = float(meta.get("duration_s") or 0)
        nw = int(np.ceil(min(dur, z["radar_rts"].max() + 1) / W)) \
            if "radar_rts" in z.files and len(z["radar_rts"]) else 0
        rad = _radar_window_feats(z, W)
        nw = max(nw, len(rad["rad_ok"]))
        feats = {"cap": np.array([meta["cap"]] * nw),
                 "win": np.arange(nw),
                 "t_rel": np.arange(nw) * W,
                 "label": np.full(nw, label, np.int64),
                 **rad}
        for link, pca in (("csi1", pcas["csi1"]), ("csi2", pcas["csi2"])):
            if f"{link}_ok" in z.files:
                feats.update(_csi_window_feats(z, link, W, nw, pca))
            else:
                for c in C.csi_cols(f"{link}_"):
                    feats[c] = np.full(nw, np.nan, np.float32)
                feats[f"{link}_ok"] = np.zeros(nw, np.float32)
        rows.append(pd.DataFrame(feats))
        print(f"  {meta['cap']}: {nw} windows (label={meta['label']})",
              flush=True)
    df = pd.concat(rows, ignore_index=True)
    df["dataset"] = "multilink"
    return df


# ---------------------------------------------------------------------------
# legacy table: published E2 features + physical extras from E1 cache
# ---------------------------------------------------------------------------
def _old_phys_minute(npz_path, W):
    """Physical descriptors per window for one legacy minute cache."""
    z = np.load(npz_path)
    maps, rts = z["maps"], z["rts"]
    nw = 60 // W
    out = np.full((nw, len(C.RADAR_PHYS) + 2), np.nan, np.float32)
    for w in range(nw):
        sel = (rts >= w * W) & (rts < (w + 1) * W)
        if sel.sum() < 5:
            continue
        mi = maps[sel, 0].astype(np.float32).mean(axis=0)
        vals = C.rd_phys_descriptors(mi)
        for i, k in enumerate(C.RADAR_PHYS):
            out[w, i] = vals[k]
        # CSI tv + dop_frac from the 100 Hz per-second amp rows
        amp = z["amp60"][w * W:(w + 1) * W].astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr = np.nanmean(amp, axis=2)         # (W,100)
        tvs, dps = [], []
        for srow in tr:
            v = srow[np.isfinite(srow)]
            if len(v) < 20:
                continue
            tvs.append(np.abs(np.diff(v)).sum())
            v = v - v.mean()
            ps = np.abs(np.fft.rfft(v)) ** 2
            dps.append(ps[len(ps) // 3:].sum() / (ps[1:].sum() + 1e-12))
        out[w, -2] = np.mean(tvs) if tvs else np.nan
        out[w, -1] = np.mean(dps) if dps else np.nan
    return out


def build_old_table(W):
    csv = C.OLD_FEATS / f"features_{W}s.csv"
    df = pd.read_csv(csv)
    df = df[df.aug.fillna(False).astype(bool) == False]  # noqa: E712
    df = df[df.dataset.isin(["train", "test"])].copy()
    print(f"  old features_{W}s: {len(df)} real window rows", flush=True)
    # physical extras per minute (only minutes in the E1 cache)
    extra = {}
    idx_rows = []
    for npz in sorted(C.OLD_CACHE.rglob("*.npz")):
        rec = f"{npz.parent.name}/{npz.stem}"
        arr = _old_phys_minute(npz, W)
        extra[rec] = arr
        idx_rows.append(rec)
    phys_names = C.RADAR_PHYS + ["csi_tv", "csi_dop_frac"]
    for k in phys_names:
        df[k] = np.nan
    matched = 0
    for rec, arr in extra.items():
        m = df.rec_id == rec
        if not m.any():
            continue
        matched += 1
        wins = df.loc[m, "win"].astype(int).values
        for i, k in enumerate(phys_names):
            df.loc[m, k] = arr[wins, i]
    print(f"  physical extras merged for {matched} legacy minutes",
          flush=True)
    # unified column names -> csi1_* style handled at analysis time
    return df


# ---------------------------------------------------------------------------
# per-second PCA dot dumps (for trajectory/scatter figures)
# ---------------------------------------------------------------------------
def dump_pc_dots_multilink(pcas):
    """arrays: [label, sec, src, pc1..3]; src = capture index."""
    rows_c, rows_r = [], []
    for src, p in enumerate(sorted(C.ML_CACHE.glob("*.npz"))):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        lab = 1 if meta["label"] == "occupied" else 0
        for link in ("csi1", "csi2"):
            ok = z[f"{link}_ok"]
            if ok.any():
                pr = pcas[link].transform(z[f"{link}_dots"][ok])
                rows_c.append(np.c_[np.full(ok.sum(), lab),
                                    np.flatnonzero(ok),
                                    np.full(ok.sum(), src), pr])
        ok = z["radar_rsec"]
        if ok.any():
            rds = z["radar_vsec"][ok, 0].reshape(-1, 576).astype(np.float32)
            pr = pcas["rad"].transform(rds)
            rows_r.append(np.c_[np.full(ok.sum(), lab),
                                np.flatnonzero(ok),
                                np.full(ok.sum(), src), pr])
    np.savez_compressed(
        OUT / "pc_dots_multilink.npz",
        csi=np.concatenate(rows_c), rad=np.concatenate(rows_r))


def dump_pc_dots_old(pca_csi, pca_rad):
    """arrays: [label, sec, src, pc1..3]; src = minute index."""
    rows_c, rows_r = [], []
    for src, npz in enumerate(sorted(C.OLD_CACHE.rglob("*.npz"))):
        z = np.load(npz)
        meta = json.loads(str(z["meta"]))
        lab = int(meta["label"])
        ok = z["sec_valid"]
        if ok.any():
            pr = pca_csi.transform(z["dots"][ok])
            rows_c.append(np.c_[np.full(ok.sum(), lab),
                                np.flatnonzero(ok),
                                np.full(ok.sum(), src), pr])
        ok = z["sec_has_radar"]
        if ok.any():
            pr = pca_rad.transform(z["rd_sec"][ok].astype(np.float32))
            rows_r.append(np.c_[np.full(ok.sum(), lab),
                                np.flatnonzero(ok),
                                np.full(ok.sum(), src), pr])
    np.savez_compressed(
        OUT / "pc_dots_old.npz",
        csi=np.concatenate(rows_c), rad=np.concatenate(rows_r))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("[pca] multilink fits", flush=True)
    pcas_ml = fit_pcas_multilink()
    for W in C.WIN_LIST:
        print(f"[multilink] W={W}", flush=True)
        df = build_multilink_table(W, pcas_ml)
        df.to_csv(OUT / f"features_multilink_{W}s.csv", index=False)
    print("[pca] load legacy pcas", flush=True)
    pca_csi, pca_rad = load_pcas_old()
    for W in C.WIN_LIST:
        df = build_old_table(W)
        df.to_csv(OUT / f"features_old_{W}s_ext.csv", index=False)
    print("[dump] pc dots", flush=True)
    dump_pc_dots_multilink(pcas_ml)
    import joblib
    joblib.dump(pcas_ml, OUT / "pcas_multilink.joblib")
    dump_pc_dots_old(pca_csi, pca_rad)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
