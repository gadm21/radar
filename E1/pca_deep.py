"""E1 deep-dive: PCA-based physical descriptors + separability.

Pipeline
--------
preprocessing
  * per-second validity masks (sec_valid / csi*_ok / sec_has_radar /
    radar_rsec) - invalid seconds are excluded, never interpolated
    (interpolation fabricates motion energy);
  * StandardScaler fit on the TRAIN split of each dataset (multilink
    has no labelled train - its scaler/PCA are fit unsupervised on all
    its seconds; documented caveat);
  * PCA fit on train seconds, n_pc up to 8, projections kept to PC3.

per-second views
  * csi   : compressed CSI 'dots' (104 dims; multilink = concat of the
            two links -> 208 dims)
  * radar : flattened per-second radar view (rd_sec 576 dims legacy;
            vsec 4x24x24 -> 2304 dims multilink)
  * joint : concat of the two PC projections (6 dims) - used for
            clustering + trajectories

per-window (W=5 s, hop W/2) descriptors  [same schema all datasets]
  * pc{view}{i}_mean/std/iqr/min/max         - position + compressed var
  * pcv_{view}{i}                            - std(PC_i) (alias, kept)
  * traj_{view}_{speed,speedmax,path,disp,tort,revers,ac1,slope}
                                             - temporal-trajectory stats
  * clu_km{dist0,dist1,margin} clu_gm_occ    - cluster geometry
    (kmeans/gmm fit on train joint-PC space, per dataset)
  * fft_snr_*                                - 8-10 Hz radar SNR spectral
    bands + domfreq + entropy (all datasets)
  * fftc_*                                   - 100 Hz CSI amplitude FFT
    bands (legacy only; multilink streams expose no raw CSI)

outputs: E1/outputs/pca_deep/
  desc_<split>_5s.csv   window descriptor tables
  pcas_deep.joblib      scalers+pcas per dataset
  pc_dots_<split>.npz   per-second PC sequences (joint 6ch) for seqcnn
  analysis_pca.json     variance explained / purity / top AUCs
  REPORT_PCA.md + figs_pca/
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (adjusted_rand_score, roc_auc_score,
                             silhouette_score)

E1 = Path(__file__).resolve().parent
ROOT = E1.parent
CACHE = E1 / "cache"
OUT = E1 / "outputs" / "pca_deep"
OUT.mkdir(parents=True, exist_ok=True)

W = 5
HOP = 2
N_PC = 8          # fit this many; keep 3 for descriptors/sequences
BANDS = ((0.10, 0.50), (0.50, 1.50), (1.50, 4.00))
BANDS_CSI = ((0.10, 0.50), (0.50, 2.0), (2.0, 8.0), (8.0, 25.0))

SPLITS = {
    "train":  CACHE / "win" / "train",
    "val":    CACHE / "win" / "test_minutes",   # validation_minutes
    "calib":  CACHE / "win" / "calibration_minutes",
    "testml": CACHE / "multilink",
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_win(d: Path):
    """legacy minute caches -> list of per-minute dicts."""
    out = []
    for f in sorted(d.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        out.append({
            "id": meta["rec_id"], "label": int(meta["label"]),
            "session": meta.get("session_id", "?"),
            "amp60": z["amp60"], "sec_valid": z["sec_valid"],
            "dots": z["dots"].astype(np.float32),
            "rd_sec": z["rd_sec"].astype(np.float32),
            "sec_has_radar": z["sec_has_radar"],
            "snr_f": z["snr_f"], "rts": z["rts"],
            "snr_hz": float(z["radar_hz"]),
            "vsec": z["maps"].astype(np.float32),
            "a90": None, "rv": None, "tv": None, "dopf": None,
        })
    return out


def load_ml(d: Path):
    out = []
    for f in sorted(d.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        lab = 0 if "empty" in meta.get("cap", "") else 1
        n = min(len(z["csi1_ok"]), len(z["csi2_ok"]),
                len(z["radar_rsec"]))
        ok1, ok2 = z["csi1_ok"][:n], z["csi2_ok"][:n]
        ok = ok1 | ok2
        d1 = z["csi1_dots"][:n]
        d2 = z["csi2_dots"][:n]
        denom = (ok1.astype(np.float32) + ok2.astype(np.float32))
        dots = np.where(
            denom[:, None] > 0,
            (d1 * ok1[:, None] + d2 * ok2[:, None])
            / np.maximum(denom[:, None], 1), np.nan)
        out.append({
            "id": meta["cap"], "label": lab, "session": meta["cap"],
            "amp60": None, "sec_valid": ok,
            "dots": dots.astype(np.float32),
            "rd_sec": z["radar_vsec"][:n]
                      .reshape(n, -1).astype(np.float32),
            "sec_has_radar": z["radar_rsec"][:n],
            "snr_f": z["radar_snr_f"], "rts": z["radar_rts"],
            "snr_hz": len(z["radar_snr_f"]) / max(z["radar_rts"][-1]
                                                - z["radar_rts"][0], 1),
            "vsec": z["radar_vsec"][:n].astype(np.float32),
            "a90": np.where(
                ok1 & ok2,
                (z["csi1_a90"][:n] + z["csi2_a90"][:n]) / 2,
                np.where(ok1, z["csi1_a90"][:n],
                         z["csi2_a90"][:n])),
            "rv": (z["csi1_rv"][:n] * ok1[:, None]
                   + z["csi2_rv"][:n] * ok2[:, None])
                  / np.maximum(denom[:, None], 1),
            "tv": (z["csi1_tv"][:n] * ok1
                   + z["csi2_tv"][:n] * ok2)
                  / np.maximum(denom, 1),
            "dopf": (z["csi1_dopf"][:n] * ok1
                     + z["csi2_dopf"][:n] * ok2)
                    / np.maximum(denom, 1),
        })
    return out


# ---------------------------------------------------------------------------
# spectral descriptors
# ---------------------------------------------------------------------------
def _bands(sig, fs, bands):
    """mean-normalised band powers + domfreq + entropy of a 1-D signal."""
    if len(sig) < 16 or np.nanstd(sig) < 1e-9:
        return [np.nan] * (len(bands) + 2)
    sig = np.nan_to_num(sig - np.nanmean(sig))
    P = np.abs(np.fft.rfft(sig)) ** 2
    f = np.fft.rfftfreq(len(sig), 1 / fs)
    tot = P.sum() + 1e-12
    out = [float(P[(f >= lo) & (f < hi)].sum() / tot)
           for lo, hi in bands]
    out.append(float(f[np.argmax(P)]))                     # domfreq
    q = P / tot + 1e-12
    out.append(float(-(q * np.log(q)).sum()))              # spectral entropy
    return out


def fft_snr_feats(rts, snr, t0, t1, fs):
    m = (rts >= t0) & (rts < t1)
    return _bands(snr[m], fs, BANDS)


def fft_csi_feats(amp60, s0):
    """100-Hz CSI amplitude FFT over a W-second slice (legacy only).

    Vectorised: one rfft over all subcarriers at once."""
    seg = amp60[s0:s0 + W]                       # (W,100,52)
    if seg.shape[0] < W:
        return [np.nan] * (len(BANDS_CSI) + 2)
    x = seg.reshape(-1, seg.shape[-1]).astype(np.float32)  # (W*100,52)
    x = x[:, np.isfinite(x).all(0)]
    if x.shape[1] == 0:
        return [np.nan] * (len(BANDS_CSI) + 2)
    x = x - x.mean(0)
    P = np.abs(np.fft.rfft(x, axis=0)) ** 2      # (nF, nsub)
    f = np.fft.rfftfreq(len(x), 1 / 100.0)
    tot = P.sum(0) + 1e-12
    out = [float((P[(f >= lo) & (f < hi)].sum(0) / tot).mean())
           for lo, hi in BANDS_CSI]
    out.append(float(f[P.argmax(0)].mean()))     # mean domfreq
    q = P / tot + 1e-12
    out.append(float((-(q * np.log(q)).sum(0)).mean()))
    return out


def phys_feats(r, s0, t0, t1):
    """shared physical radar/csi descriptors for one window."""
    d = {}
    v = r["vsec"]
    if v is not None and len(v) == 0:
        v = None
    # radar: win caches keep per-frame maps (use rts slice); ml keeps
    # per-second vsec (use s0 slice)
    if r["vsec"].shape[0] == len(r["sec_has_radar"]):      # per-second
        m = r["vsec"][s0:s0 + W][r["sec_has_radar"][s0:s0 + W]]
    else:                                                # per-frame
        fidx = (r["rts"] >= t0) & (r["rts"] < t1)
        m = r["vsec"][fidx]
    if m.size:
        views = {"rd": 0, "ra": 1, "re": 2, "xy": 3}
        for vn, vi in views.items():
            a = m[:, vi]
            fr = a.reshape(len(a), -1)
            d[f"{vn}_mean"] = fr.mean()
            d[f"{vn}_p90"] = np.percentile(fr, 90)
            d[f"{vn}_std90"] = fr.std(1).mean()
            d[f"{vn}_delta"] = fr[-1].mean() - fr[0].mean()
            d[f"{vn}_peak"] = fr.max()
        rd = m[:, 0]
        tot = rd.sum((1, 2)) + 1e-9
        rng_cm = (rd.sum(2) @ np.arange(rd.shape[2])) / tot
        dop_cm = (rd.sum(1) @ np.arange(rd.shape[1])) / tot
        d["rd_range_cm"] = rng_cm.mean()
        d["rd_dop_cm"] = dop_cm.mean()
        d["rd_dop_spread"] = np.sqrt(
            (((np.arange(rd.shape[1])[None, :] - dop_cm[:, None]) ** 2)
             * rd.sum(1)).sum(1) / tot).mean()
        thr = np.percentile(rd, 75)
        d["rd_dca"] = (rd > thr).mean()
        d["rd_energy"] = rd.reshape(len(rd), -1).sum(1).mean()
        d["rad_cov"] = float(len(m)) / max(W, 1)
    else:
        for vn in ("rd", "ra", "re", "xy"):
            for st in ("mean", "p90", "std90", "delta", "peak"):
                d[f"{vn}_{st}"] = np.nan
        for k in ("rd_range_cm", "rd_dop_cm", "rd_dop_spread",
                  "rd_dca", "rd_energy"):
            d[k] = np.nan
        d["rad_cov"] = 0.0
    # snr over window
    mm = (r["rts"] >= t0) & (r["rts"] < t1) if len(r["rts"]) \
        else np.zeros(0, bool)
    snr = r["snr_f"][mm] if len(mm) else np.array([])
    d["snr_mean"] = snr.mean() if len(snr) else np.nan
    d["snr_max"] = snr.max() if len(snr) else np.nan
    # csi physical: legacy amp60 or ml per-second aggregates
    if r["amp60"] is not None:
        a = r["amp60"][s0:s0 + W]                # (W,100,52)
        a2 = a.reshape(-1, a.shape[-1]).astype(np.float32)
        a2 = a2[:, np.isfinite(a2).all(0)]
        if a2.size:
            d["csi_amp_mean"] = a2.mean()
            d["csi_amp_std"] = a2.std()
            d["csi_amp_q90"] = np.percentile(a2, 90)
            rv = a2.var(0)                        # per-subcarrier var
            d["csi_rv_mean"] = rv.mean()
            d["csi_rv_q90"] = np.percentile(rv, 90)
            d["csi_rv_max"] = rv.max()
            dif = np.abs(np.diff(a2, axis=0)).mean(0)
            d["csi_tv"] = dif.mean()
            # fraction of window dominated by >0.5 Hz content
            x = a2 - a2.mean(0)
            P = np.abs(np.fft.rfft(x, axis=0)) ** 2
            f = np.fft.rfftfreq(len(x), 1 / 100.0)
            d["csi_dop_frac"] = float(
                (P[f >= 0.5].sum(0) / (P.sum(0) + 1e-9) > .5).mean())
        else:
            for k in ("csi_amp_mean", "csi_amp_std", "csi_amp_q90",
                      "csi_rv_mean", "csi_rv_q90", "csi_rv_max",
                      "csi_tv", "csi_dop_frac"):
                d[k] = np.nan
        d["csi_cov"] = r["sec_valid"][s0:s0 + W].mean()
    elif r["a90"] is not None:
        ok = r["sec_valid"][s0:s0 + W]
        if ok.sum() >= 3:
            a = r["a90"][s0:s0 + W][ok]
            rv = r["rv"][s0:s0 + W][ok].mean(1)
            d["csi_amp_mean"] = a.mean()
            d["csi_amp_std"] = a.std()
            d["csi_amp_q90"] = np.percentile(a, 90)
            d["csi_rv_mean"] = rv.mean()
            d["csi_rv_q90"] = np.percentile(rv, 90)
            d["csi_rv_max"] = rv.max()
            d["csi_tv"] = r["tv"][s0:s0 + W][ok].mean()
            d["csi_dop_frac"] = r["dopf"][s0:s0 + W][ok].mean()
        else:
            for k in ("csi_amp_mean", "csi_amp_std", "csi_amp_q90",
                      "csi_rv_mean", "csi_rv_q90", "csi_rv_max",
                      "csi_tv", "csi_dop_frac"):
                d[k] = np.nan
        d["csi_cov"] = float(ok.mean())
    return d


# ---------------------------------------------------------------------------
# PCA / cluster models
# ---------------------------------------------------------------------------
def fit_view(recs, key, mask_key, n_pc=N_PC, rng=0):
    """scaler + PCA fit on stacked valid seconds of the given recs."""
    X = np.concatenate([r[key][r[mask_key]] for r in recs], axis=0)
    X = X[np.isfinite(X).all(1)]
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=min(n_pc, X.shape[1]),
              random_state=rng).fit(sc.transform(X))
    return sc, pca


def project(sc, pca, X):
    if len(X) == 0:
        return np.zeros((0, pca.n_components_), np.float32)
    X = np.asarray(X, dtype=np.float32)
    X = np.where(np.isfinite(X), X, sc.mean_[None, :])
    return pca.transform(sc.transform(X))


def win_desc(pc, feats_extra):
    """window descriptors from a (n,3) pc sequence within one window."""
    d = {}
    if len(pc) < 3:
        return None
    nd = pc.shape[1]
    for i in range(nd):
        v = pc[:, i]
        d[f"mean{i}"] = v.mean()
        d[f"std{i}"] = v.std()
        d[f"iqr{i}"] = np.percentile(v, 75) - np.percentile(v, 25)
        d[f"min{i}"] = v.min()
        d[f"max{i}"] = v.max()
    step = np.linalg.norm(np.diff(pc, axis=0), axis=1)
    path = step.sum()
    disp = np.linalg.norm(pc[-1] - pc[0])
    rev = np.sum(step[1:] * step[:-1] < 0) if len(step) > 1 else 0
    sgn = np.sign(np.diff(pc[:, 0]))
    d.update({
        "speed": step.mean(), "speedmax": step.max() if len(step) else 0,
        "path": path, "disp": disp,
        "tort": path / (disp + 1e-6),
        "revers": int((sgn[1:] * sgn[:-1] < 0).sum()),
        "ac1": (np.corrcoef(pc[:-1, 0], pc[1:, 0])[0, 1]
                if len(pc) > 3 and pc[:, 0].std() > 1e-9 else 0.0),
        "slope": np.polyfit(np.arange(len(pc)), pc[:, 0], 1)[0]
                 if len(pc) > 2 else 0.0,
        **feats_extra,
    })
    return d


# ---------------------------------------------------------------------------
# per-dataset pipeline
# ---------------------------------------------------------------------------
def build(recs, fit_recs, tag, fit_pool=None, fit_ids=None):
    """fit scalers/pcas on fit_recs, project every rec, build windows.

    fit_pool: joint-PC array (n,6) of the FIT split for kmeans/gmm/
    dbscan; if None, computed from this split's own sequences.
    fit_ids: rec ids whose windows define the purity/ARI reference
    (defaults to fit_recs ids)."""
    sc_c, pca_c = fit_view(fit_recs, "dots", "sec_valid")
    sc_r, pca_r = fit_view(fit_recs, "rd_sec", "sec_has_radar")
    ev = {"csi": pca_c.explained_variance_ratio_.tolist(),
          "radar": pca_r.explained_variance_ratio_.tolist()}

    seqs = []                                   # (T,6) joint pc + meta
    rows = []
    dots_store = {}
    for r in recs:
        pc_c = project(sc_c, pca_c, r["dots"])[:, :3]
        pc_r = project(sc_r, pca_r, r["rd_sec"])[:, :3]
        joint = np.concatenate([pc_c, pc_r], axis=1)   # (T,6)
        okc = r["sec_valid"]
        okr = r["sec_has_radar"]
        ok = okc & okr
        dots_store[r["id"]] = np.c_[ok.astype(np.float32), joint]
        seqs.append((r["id"], joint, ok))
        T = len(joint)
        fs = r["snr_hz"]
        for s0 in range(0, T - W + 1, HOP):
            t0, t1 = (r["rts"][0] + s0, r["rts"][0] + s0 + W) \
                if len(r["rts"]) else (s0, s0 + W)
            wok = ok[s0:s0 + W]
            if wok.sum() < 3:
                continue
            jc = np.c_[pc_c[s0:s0 + W][wok]]
            jr = np.c_[pc_r[s0:s0 + W][wok]]
            jj = np.c_[joint[s0:s0 + W][wok]]
            d = {"id": r["id"], "s0": s0, "label": r["label"],
                 "session": r["session"], "cov": wok.mean()}
            dc = win_desc(jc, {})
            dr = win_desc(jr, {})
            dj = win_desc(jj, {})
            if dc is None or dr is None or dj is None:
                continue
            d.update({f"c_{k}": v for k, v in dc.items()})
            d.update({f"r_{k}": v for k, v in dr.items()})
            d.update({f"j_{k}": v for k, v in dj.items()})
            d.update(phys_feats(r, s0, t0, t1))
            snr = fft_snr_feats(r["rts"], r["snr_f"], t0, t1, fs)
            for b, (lo, hi) in enumerate(BANDS):
                d[f"fft_snr_b{b}"] = snr[b]
            d["fft_snr_domf"] = snr[-2]
            d["fft_snr_entr"] = snr[-1]
            if r["amp60"] is not None:
                fc = fft_csi_feats(r["amp60"], s0)
                for b in range(len(BANDS_CSI)):
                    d[f"fftc_b{b}"] = fc[b]
                d["fftc_domf"] = fc[-2]
                d["fftc_entr"] = fc[-1]
            rows.append(d)

    df = pd.DataFrame(rows)

    # ---- cluster geometry on joint PC space --------------------------
    train_ids = set(fit_ids) if fit_ids is not None \
        else {r["id"] for r in fit_recs}
    if fit_pool is not None:
        pool = fit_pool
    else:
        sel = [j[ok] for i, j, ok in seqs if i in train_ids]
        if not sel:
            sel = [j[ok] for i, j, ok in seqs]
        pool = np.concatenate(sel)
    rng = np.random.default_rng(0)
    sub = pool[rng.choice(len(pool), min(20000, len(pool)),
                          replace=False)]
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(sub)
    gm = GaussianMixture(n_components=2, random_state=0).fit(sub)
    db = DBSCAN(eps=np.percentile(
            np.linalg.norm(sub - sub.mean(0), axis=1), 60) or 1.0,
            min_samples=10).fit(sub)
    # cluster <-> label purity on labelled training windows
    jcols = [f"j_mean{i}" for i in range(6)]
    jv = df if len(df) else df
    cents = km.cluster_centers_
    lab = jv.label.values
    km_pred = km.predict(jv[jcols].values) if len(jv) else np.array([])
    if len(km_pred):
        mapping = {}
        for c in np.unique(km_pred):
            m = km_pred == c
            mapping[c] = int(np.bincount(
                lab[m].astype(int)).argmax()) if m.sum() else 0
        pred = np.array([mapping[c] for c in km_pred])
        purity = float((pred == lab).mean())
    else:
        purity = np.nan
    ari = float(adjusted_rand_score(lab, km_pred)) if len(km_pred) \
        else np.nan
    sil = float(silhouette_score(jv[jcols].values, lab)) \
        if len(jv) > 20 and len(np.unique(lab)) > 1 else np.nan

    # add cluster-distance descriptors
    X = df[jcols].values
    dd = np.linalg.norm(X[:, None, :] - cents[None, :, :], axis=2)
    df["clu_kmdist0"] = dd[:, 0]
    df["clu_kmdist1"] = dd[:, 1]
    df["clu_kmmargin"] = np.abs(dd[:, 1] - dd[:, 0])
    df["clu_kmlabel"] = km.predict(X)
    if len(df):
        df["clu_gm_occ"] = gm.predict_proba(X).max(1)
    noise = float((db.labels_ == -1).mean()) if len(db.labels_) else np.nan

    return df, {"sc_c": sc_c, "pca_c": pca_c, "sc_r": sc_r,
                "pca_r": pca_r, "km": km, "gm": gm,
                "ev": ev, "purity": purity, "ari": ari,
                "silhouette": sil, "dbscan_noise": noise,
                "dbscan_clusters": int(len(set(db.labels_) - {-1}))}, \
        seqs, dots_store


# ---------------------------------------------------------------------------
# separability report
# ---------------------------------------------------------------------------
def sep_stats(df):
    cols = [c for c in df.columns
            if c not in ("id", "s0", "label", "session", "cov")]
    y = df.label.values
    out = []
    for c in cols:
        v = df[c].values
        m = np.isfinite(v)
        if m.sum() < 20 or np.nanstd(v[m]) < 1e-9:
            continue
        try:
            auc = roc_auc_score(y[m], v[m])
        except Exception:
            auc = np.nan
        a, b = v[m & (y == 0)], v[m & (y == 1)]
        d = ((a.mean() - b.mean()) /
             np.sqrt((a.std() ** 2 + b.std() ** 2) / 2 + 1e-12))
        out.append({"col": c, "auc": auc,
                    "auc_eff": max(auc, 1 - auc), "cohend": d})
    return pd.DataFrame(out).sort_values("auc_eff", ascending=False)


def main():
    t0 = time.time()
    datasets = {}
    print("[load] win splits", flush=True)
    for name in ("train", "val", "calib"):
        if SPLITS[name].is_dir() and any(SPLITS[name].glob("*.npz")):
            datasets[name] = load_win(SPLITS[name])
            print(f"  {name}: {len(datasets[name])} recs", flush=True)
    datasets["testml"] = load_ml(SPLITS["testml"])
    print(f"  testml: {len(datasets['testml'])} caps", flush=True)

    job = {}
    res = {}
    seqs_all = {}
    dots_all = {}
    order = [n for n in ("train", "val", "calib", "testml")
             if n in datasets]
    train_pool, train_ids = None, None
    for name in order:
        recs = datasets[name]
        if name == "testml":
            fit_recs = recs                  # unsupervised fit (no labels)
            fp, fi = None, None
        else:
            fit_recs = datasets["train"]     # legacy: always train stats
            fp, fi = train_pool, train_ids
        df, art, seqs, dots = build(recs, fit_recs, name,
                                    fit_pool=fp, fit_ids=fi)
        if name == "train":
            train_ids = {r["id"] for r in recs}
            train_pool = np.concatenate(
                [j[ok] for i, j, ok in seqs if i in train_ids])
            # re-fit clusters on train pool (build used it already
            # via fit_pool=None fallback -> same thing)
        df.to_csv(OUT / f"desc_{name}_{W}s.csv", index=False)
        job[name] = art
        res[name] = art
        seqs_all[name] = seqs
        dots_all[name] = {k: v for k, v in dots.items()}
        np.savez_compressed(OUT / f"pc_dots_{name}.npz", **dots)
        s = sep_stats(df)
        s.to_csv(OUT / f"sep_{name}.csv", index=False)
        res[name]["top"] = s.head(15).to_dict("records")
        print(f"[{name}] rows={len(df)} ev_csi="
              f"{np.sum(art['ev']['csi'][:3]):.2f} ev_rad="
              f"{np.sum(art['ev']['radar'][:3]):.2f} "
              f"pur={art['purity']:.2f} ari={art['ari']:.2f} "
              f"sil={art['silhouette']:.2f}", flush=True)

    slim = {n: {k: v for k, v in a.items()
                if k in ("ev", "purity", "ari", "silhouette",
                         "dbscan_noise", "dbscan_clusters", "top")}
            for n, a in res.items()}
    (OUT / "analysis_pca.json").write_text(json.dumps(slim, indent=1))
    joblib.dump({n: {"sc_c": a["sc_c"], "pca_c": a["pca_c"],
                     "sc_r": a["sc_r"], "pca_r": a["pca_r"],
                     "km": a["km"], "gm": a["gm"]}
                 for n, a in job.items()},
                OUT / "pcas_deep.joblib")
    print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
