"""Raw-signal + descriptor-stat analysis for the E1/E2 occupancy task.

Reads the raw per-second products from the E1 caches
(``E1/cache/win/{train,test_minutes,calibration_minutes}`` legacy minutes,
``E1/cache/multilink`` JSONL-derived captures), computes an improved set
of *physical* window descriptors (W=5 s, hop=2 s — row-compatible with
``E1/outputs/pca_deep/desc_<split>_5s.csv``), and reports which
descriptors carry *decisive* class signal that survives the
train -> wall3(val) -> multilink(testml) domain shifts.

New descriptors vs pca_deep (physical meaning, LLM-friendly):
  radar (per-second log1p RD mean maps rd_sec / vsec[:,0] + snr_f)
    rd_motion   mean |delta| of consecutive per-second RD maps (change rate)
    rd_still    fraction of seconds with delta < session median (stillness)
    rd_burst    fraction of seconds with delta > session 80th pct (burstiness)
    rd_hotn     mean count of RD bins above median+2*MAD (changed-area extent)
    rd_rshift   |linear slope| of power-weighted range centroid (m/s-ish)
    rd_dopbw    power-weighted doppler bandwidth (clutter rows masked)
    snr_breath  FFT power fraction of snr trace in 0.15-0.45 Hz (breathing)
    snr_ac1     lag-1 autocorrelation of snr trace (activity persistence)
  csi   (legacy: amp60 100 Hz x 52; multilink: per-second aggregates)
    csi_tv_s    per-second total variation of the mean-amplitude trace
    csi_ac1     lag-1 autocorrelation of mean-amp trace (1 Hz envelope)
    csi_xcorr   mean |off-diag corr| across subcarriers (spatial coherence)
    csi_envbw   spectral centroid of mean-amp trace (motion rate, Hz)
    csi_imp     fraction of |d amp| above session q95 (impulsiveness)
    csi_madc    MAD of per-subcarrier variance (channel-selective motion)
  joint
    agree_mr    fraction of seconds where csi & radar activity agree
                on above/below-median (cross-modal confirmation)

Outputs:
  E2/outputs/three_pipelines/desc_v2_<split>.csv   new-descriptor table
  E2/outputs/three_pipelines/decisive_report.md    stats + patterns
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

E2 = Path(__file__).resolve().parent
ROOT = E2.parent
E1C = ROOT / "E1" / "cache"
PCA_DIR = ROOT / "E1" / "outputs" / "pca_deep"
OUT = E2 / "outputs" / "three_pipelines"
OUT.mkdir(parents=True, exist_ok=True)

W, HOP = 5, 2
N_TRAIN, N_VAL, N_CAL = 80, 80, 40          # legacy minute samples
_DC = (10, 14)                              # clutter doppler rows
_NEAR = 2

SPLITS = {
    "train":  (E1C / "win" / "train", N_TRAIN),
    "val":    (E1C / "win" / "test_minutes", N_VAL),
    "calib":  (E1C / "win" / "calibration_minutes", N_CAL),
    "testml": (E1C / "multilink", None),
}

NEW_COLS = [
    "rd_motion", "rd_still", "rd_burst", "rd_hotn", "rd_rshift",
    "rd_dopbw", "snr_breath", "snr_ac1",
    "csi_tv_s", "csi_ac1", "csi_xcorr", "csi_envbw", "csi_imp",
    "csi_madc", "agree_mr",
]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_win_files(d: Path, n: int | None):
    fs = sorted(d.glob("*.npz"))
    if n and len(fs) > n:
        # deterministic stratified-ish sample by label
        metas = []
        for f in fs:
            m = json.loads(str(np.load(f)["meta"]))
            metas.append((f, int(m["label"])))
        rng = np.random.default_rng(0)
        keep = []
        for lab in (0, 1):
            pool = [f for f, l in metas if l == lab]
            k = min(len(pool), n // 2)
            keep.extend(rng.choice(pool, size=k, replace=False))
        fs = sorted(set(keep))
    return fs


def _sec_rd(z):
    """per-second mean log1p RD map -> (ns,24,24) f32."""
    if "rd_sec" in z.files:                     # legacy: (60,576)
        return z["rd_sec"].astype(np.float32).reshape(-1, 24, 24)
    return z["radar_vsec"][:, 0].astype(np.float32)  # ml: (ns,24,24)


# ---------------------------------------------------------------------------
# descriptor math
# ---------------------------------------------------------------------------
def _pwr(m):                                   # log1p -> power
    return np.expm1(np.clip(m.astype(np.float64), 0, 20))


def rd_phys(rd_sec, rsec):
    """per-window scalar stats from per-second RD maps."""
    out = {}
    valid = rd_sec[rsec]
    if len(valid) < 2:
        for k in ("motion", "still", "burst", "hotn", "rshift", "dopbw"):
            out[k] = np.nan
        return out
    p = _pwr(valid)
    p[:, _DC[0]:_DC[1], :] = 0.0
    p[:, :, :_NEAR] = 0.0
    delta = np.abs(np.diff(p, axis=0)).mean(axis=(1, 2))
    med = np.median(delta)
    out["motion"] = delta.mean()
    out["still"] = float((delta < med).mean())
    out["burst"] = float((delta > np.percentile(delta, 80)).mean())
    flat = p.reshape(len(p), -1)
    med2 = np.median(flat, axis=1, keepdims=True)
    mad = np.median(np.abs(flat - med2), axis=1, keepdims=True)
    out["hotn"] = float((flat > med2 + 2 * mad).sum(1).mean())
    tot = p.sum((1, 2)) + 1e-12
    rr = np.arange(24)[None, :, None] * np.ones((1, 1, 24))
    rcm = (p * rr).sum((1, 2)) / tot
    out["rshift"] = float(abs(np.polyfit(np.arange(len(rcm)), rcm, 1)[0])) \
        if len(rcm) > 2 else 0.0
    dop_marg = p.sum(1)                                    # (ns,24)
    dcm = (dop_marg * np.arange(24)[None, :]).sum(1) / (dop_marg.sum(1) + 1e-12)
    out["dopbw"] = float(np.sqrt(
        ((np.arange(24)[None, :] - dcm[:, None]) ** 2 * dop_marg).sum(1)
        / (dop_marg.sum(1) + 1e-12)).mean())
    return out


def snr_feats(rts, snr, t0, t1):
    m = (rts >= t0) & (rts < t1)
    s = snr[m]
    s = s[np.isfinite(s)]
    if len(s) < 16:
        return {"breath": np.nan, "ac1": np.nan}
    x = s - s.mean()
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / 10.0)       # radar ~10 Hz
    breath = float(P[(f >= 0.15) & (f <= 0.45)].sum() / (P.sum() + 1e-12))
    ac1 = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if s.std() > 1e-9 else 0.0
    return {"breath": breath, "ac1": ac1}


def csi_legacy_feats(amp60, s0):
    """W-second slice of 100 Hz x 52 amplitude -> descriptors."""
    a = amp60[s0:s0 + W].astype(np.float32).reshape(-1, 52)
    a = a[:, np.isfinite(a).all(0)]
    if a.shape[0] < 100 or a.shape[1] < 4:
        return None
    tr = a.mean(1)
    dif = np.abs(np.diff(tr))
    x = tr - tr.mean()
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / 100.0)
    envbw = float((f * P).sum() / (P.sum() + 1e-12))
    var = a.var(0)
    C = np.corrcoef(a.T)
    off = C[np.triu_indices_from(C, 1)]
    return {
        "tv_s": float(dif.mean()),
        "ac1": float(np.corrcoef(tr[:-1], tr[1:])[0, 1])
        if tr.std() > 1e-9 else 0.0,
        "xcorr": float(np.abs(off).mean()),
        "envbw": envbw,
        "imp": float((dif > np.percentile(dif, 95)).mean()),
        "madc": float(np.median(np.abs(var - np.median(var)))),
    }


def csi_ml_feats(r, s0):
    """multilink per-second aggregates -> the computable subset."""
    ok = r["sec_valid"][s0:s0 + W]
    if ok.sum() < 3:
        return None
    amp = r["a90"][s0:s0 + W][ok]
    tv = r["tv"][s0:s0 + W][ok]
    dop = r["dopf"][s0:s0 + W][ok]
    rv = r["rv"][s0:s0 + W][ok]
    ac1 = float(np.corrcoef(amp[:-1], amp[1:])[0, 1]) \
        if len(amp) > 2 and amp.std() > 1e-9 else 0.0
    return {"tv_s": float(tv.mean()), "ac1": ac1,
            "xcorr": np.nan, "envbw": float(dop.mean()),
            "imp": float((rv[:, 2] > np.percentile(rv[:, 2], 80)).mean()),
            "madc": float(np.median(rv[:, 0]))}


# ---------------------------------------------------------------------------
# per-recording window rows
# ---------------------------------------------------------------------------
def rows_legacy(z, meta):
    """one legacy minute -> list of window dicts."""
    out = []
    rid, lab = meta["rec_id"], int(meta["label"])
    rd_sec = _sec_rd(z)
    rsec = z["sec_has_radar"]
    sv = z["sec_valid"]
    amp60 = z["amp60"]
    rts, snr = z["rts"], z["snr_f"]
    # per-second activity traces for cross-modal agreement
    dots = z["dots"].astype(np.float32)
    cact = np.r_[0, np.linalg.norm(np.diff(dots, axis=0), axis=1)]
    psec = _pwr(rd_sec)
    psec[:, _DC[0]:_DC[1], :] = 0.0
    ract = np.r_[0, np.abs(np.diff(psec, axis=0)).mean(axis=(1, 2))]
    med_c = np.median(cact[sv]) if sv.any() else np.nan
    med_r = np.median(ract[rsec]) if rsec.any() else np.nan
    agree = ((cact > med_c) == (ract > med_r)).astype(np.float32)
    # session baselines for agree/still (full-minute scale)
    for s0 in range(0, 60 - W + 1, HOP):
        cov = float((sv[s0:s0 + W] & rsec[s0:s0 + W]).mean())
        if cov < 0.4:
            continue
        t0, t1 = (rts[0] + s0, rts[0] + s0 + W) if len(rts) else (s0, s0 + W)
        rp = rd_phys(rd_sec[s0:s0 + W], rsec[s0:s0 + W])
        sf = snr_feats(rts, snr, t0, t1)
        cf = csi_legacy_feats(amp60, s0)
        d = {"id": rid, "s0": s0, "label": lab, "cov": cov,
             "rd_motion": rp["motion"], "rd_still": rp["still"],
             "rd_burst": rp["burst"], "rd_hotn": rp["hotn"],
             "rd_rshift": rp["rshift"], "rd_dopbw": rp["dopbw"],
             "snr_breath": sf["breath"], "snr_ac1": sf["ac1"]}
        if cf:
            d.update({"csi_tv_s": cf["tv_s"], "csi_ac1": cf["ac1"],
                      "csi_xcorr": cf["xcorr"], "csi_envbw": cf["envbw"],
                      "csi_imp": cf["imp"], "csi_madc": cf["madc"]})
        else:
            d.update({k: np.nan for k in
                      ("csi_tv_s", "csi_ac1", "csi_xcorr",
                       "csi_envbw", "csi_imp", "csi_madc")})
        d["agree_mr"] = float(agree[s0:s0 + W].mean())
        out.append(d)
    return out


def rows_ml(z, meta):
    """one multilink capture -> list of window dicts."""
    out = []
    rid = meta["cap"]
    lab = 0 if "empty" in rid else 1
    n = min(len(z["csi1_ok"]), len(z["csi2_ok"]), len(z["radar_rsec"]))
    ok1, ok2 = z["csi1_ok"][:n], z["csi2_ok"][:n]
    sv = ok1 | ok2
    denom = ok1.astype(np.float32) + ok2.astype(np.float32)
    a90 = np.where(denom > 0,
                   (z["csi1_a90"][:n] * ok1 + z["csi2_a90"][:n] * ok2)
                   / np.maximum(denom, 1), np.nan)
    tv = np.where(denom > 0,
                  (z["csi1_tv"][:n] * ok1 + z["csi2_tv"][:n] * ok2)
                  / np.maximum(denom, 1), np.nan)
    dopf = np.where(denom > 0,
                    (z["csi1_dopf"][:n] * ok1 + z["csi2_dopf"][:n] * ok2)
                    / np.maximum(denom, 1), np.nan)
    rv = (z["csi1_rv"][:n] * ok1[:, None]
          + z["csi2_rv"][:n] * ok2[:, None]) / np.maximum(denom[:, None], 1)
    rd_sec = z["radar_vsec"][:n, 0].astype(np.float32)
    rsec = z["radar_rsec"][:n]
    rts, snr = z["radar_rts"], z["radar_snr_f"]
    # cross-modal activity agreement (CSI tv vs radar RD delta)
    psec = _pwr(rd_sec)
    psec[:, _DC[0]:_DC[1], :] = 0.0
    ract = np.r_[0, np.abs(np.diff(psec, axis=0)).mean(axis=(1, 2))]
    cact = np.where(np.isfinite(tv), tv, np.nan)
    med_c = np.nanmedian(cact[sv]) if sv.any() else np.nan
    med_r = np.median(ract[rsec]) if rsec.any() else np.nan
    agree = ((cact > med_c) == (ract > med_r)).astype(np.float32)
    agree[~np.isfinite(cact)] = np.nan
    T = n
    for s0 in range(0, T - W + 1, HOP):
        cov = float((sv[s0:s0 + W] & rsec[s0:s0 + W]).mean())
        if cov < 0.4:
            continue
        t0, t1 = rts[0] + s0, rts[0] + s0 + W
        rp = rd_phys(rd_sec[s0:s0 + W], rsec[s0:s0 + W])
        sf = snr_feats(rts, snr, t0, t1)
        rec = {"sec_valid": sv, "a90": a90, "tv": tv, "dopf": dopf,
               "rv": rv}
        cf = csi_ml_feats(rec, s0)
        d = {"id": rid, "s0": s0, "label": lab, "cov": cov,
             "rd_motion": rp["motion"], "rd_still": rp["still"],
             "rd_burst": rp["burst"], "rd_hotn": rp["hotn"],
             "rd_rshift": rp["rshift"], "rd_dopbw": rp["dopbw"],
             "snr_breath": sf["breath"], "snr_ac1": sf["ac1"]}
        if cf:
            d.update({"csi_tv_s": cf["tv_s"], "csi_ac1": cf["ac1"],
                      "csi_xcorr": cf["xcorr"], "csi_envbw": cf["envbw"],
                      "csi_imp": cf["imp"], "csi_madc": cf["madc"]})
        else:
            d.update({k: np.nan for k in
                      ("csi_tv_s", "csi_ac1", "csi_xcorr",
                       "csi_envbw", "csi_imp", "csi_madc")})
        d["agree_mr"] = float(np.nanmean(agree[s0:s0 + W]))
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------
def sep_table(df, cols):
    y = df.label.values
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        v = df[c].values
        m = np.isfinite(v)
        if m.sum() < 20 or np.nanstd(v[m]) < 1e-9:
            continue
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y[m], v[m])
        a, b = v[m & (y == 0)], v[m & (y == 1)]
        d = ((a.mean() - b.mean())
             / np.sqrt((a.std() ** 2 + b.std() ** 2) / 2 + 1e-12))
        rows.append({"col": c, "auc": auc, "auc_eff": max(auc, 1 - auc),
                     "dir": "+" if auc >= .5 else "-", "cohend": d,
                     "e_mean": a.mean(), "o_mean": b.mean(),
                     "e_med": np.median(a), "o_med": np.median(b)})
    return pd.DataFrame(rows).sort_values("auc_eff", ascending=False)


def main():
    t0 = time.time()
    all_new = {}
    for split, (d, n) in SPLITS.items():
        if not d.is_dir():
            continue
        fs = load_win_files(d, n)
        rows = []
        for f in fs:
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            try:
                if split == "testml":
                    rows.extend(rows_ml(z, meta))
                else:
                    rows.extend(rows_legacy(z, meta))
            except Exception as e:
                print(f"  {f.name}: {e}", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"desc_v2_{split}.csv", index=False)
        all_new[split] = df
        print(f"[{split}] {len(fs)} recs -> {len(df)} windows "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- decisive-pattern report ------------------------------------
    lines = ["# Decisive patterns: new physical descriptors (W=5s, hop=2s)",
             "",
             "Columns: univariate AUC per split + sign-consistency flag.",
             "`*` = decisive: |AUC|>=0.75 on >=2 splits AND same direction "
             "on all available splits.",
             ""]
    cols = NEW_COLS
    tabs = {s: sep_table(all_new[s], cols) for s in all_new}
    piv = {}
    for c in cols:
        piv[c] = {s: (tabs[s].set_index("col").loc[c]
                      if c in set(tabs[s]["col"]) else None)
                  for s in tabs}
    head = "| descriptor | " + " | ".join(tabs) + " | stable |"
    lines.append(head)
    lines.append("|" + "---|" * (len(tabs) + 2))
    for c in cols:
        aucs = {s: (float(piv[c][s]["auc"]) if piv[c][s] is not None
                    else np.nan) for s in tabs}
        dirs = {s: ("+" if (aucs[s] >= .5) else "-")
                for s in tabs if np.isfinite(aucs[s])}
        effs = {s: max(a, 1 - a) for s, a in aucs.items()
                if np.isfinite(a)}
        stable = (len(set(dirs.values())) == 1 and len(dirs) >= 2)
        decisive = stable and sum(v >= .75 for v in effs.values()) >= 2
        mark = " *" if decisive else ""
        cells = " | ".join(
            f"{aucs[s]:.3f}({dirs.get(s, '?')})"
            if np.isfinite(aucs[s]) else "-" for s in tabs)
        lines.append(f"| {c} | {cells} | {'Y' if stable else 'n'}{mark} |")
    lines.append("")
    lines.append("## Class profiles (median per class, train split)")
    if "train" in tabs:
        t = tabs["train"].set_index("col")
        lines.append("| descriptor | empty med | occupied med | cohen d |")
        lines.append("|---|---|---|---|")
        for c in cols:
            if c in t.index:
                r = t.loc[c]
                lines.append(f"| {c} | {r['e_med']:.4g} | {r['o_med']:.4g}"
                             f" | {r['cohend']:+.2f} |")
    (OUT / "decisive_report.md").write_text("\n".join(lines))
    print(f"done {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
