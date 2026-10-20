"""Three-pipeline occupancy evaluation: rule / ml-dl / llm.

Unified eval protocol over the joined E1 pca_deep descriptors + the
desc_v2 physical descriptors (E2/desc_stats.py output):

    train   legacy train minutes (wall2)          -- fit/calibrate
    val     legacy test_minutes (wall3, labelled) -- validation
    calib   calibration_minutes (t1_sleep)        -- supervised cal block
    testml  multilink captures (2 empty + 2 occ)  -- test

Pipelines
---------
rule    physvote: evidence-weighted vote on session-percentile physical
        descriptors (weights = train |cohen d|, direction = train sign);
        plus stump / knn / threshcal baselines.
mldl    svm/rf/gbm on descriptor views with unsupervised per-set
        re-standardisation (_norm) + pct_mlp (MLP on session-percentile
        features) + seqcnn (torch 1-D CNN on joint PC-dot sequences).
llm     improved JEV judge: per-session percentile encoding (fixes the
        global-rank val collapse), physics glossary + direction hints +
        class percentile profiles + few-shot train exemplars, batched
        JSON calls.  Runs when JEV_API_KEY is set; `llm_off` is a
        deterministic surrogate implementing the same evidence rule so
        the pipeline is measurable without an API.

Metrics per pipeline x set: acc, bal, auc, rec_e, rec_o, n,
  acc_med   unsup-cal accuracy (threshold = median of eval probs)
  acc_km    unsup-cal accuracy (2-means midpoint threshold)
  acc_sup   sup-cal accuracy (threshold tuned on labelled reference:
            val probs for testml, train probs for val/calib)
Confusion matrices for the best model per pipeline at the sup-cal thr.

Usage:
    python E2/three_pipelines.py            # rule + mldl + llm_off
    python E2/three_pipelines.py --llm      # + real LLM calls (needs key)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

E2 = Path(__file__).resolve().parent
ROOT = E2.parent
PCA_DIR = ROOT / "E1" / "outputs" / "pca_deep"
OUT = E2 / "outputs" / "three_pipelines"
OUT.mkdir(parents=True, exist_ok=True)

from sklearn.cluster import KMeans                              # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,   # noqa: E402
                              RandomForestClassifier)
from sklearn.metrics import (accuracy_score,                    # noqa: E402
                             balanced_accuracy_score,
                             confusion_matrix, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier              # noqa: E402
from sklearn.neural_network import MLPClassifier                # noqa: E402
from sklearn.preprocessing import StandardScaler                # noqa: E402
from sklearn.svm import SVC                                     # noqa: E402

RNG = np.random.default_rng(0)
META = {"id", "s0", "label", "session", "cov", "clu_kmlabel"}
RAD_PREFIX = ("snr_", "rd_", "ra_", "re_", "xy_", "r_", "fft_snr_",
              "rad_")
CSI_PREFIX = ("csi_", "c_")
JNT_PREFIX = ("j_", "clu_")
V2_COLS = ["rd_motion", "rd_still", "rd_burst", "rd_hotn", "rd_rshift",
           "rd_dopbw", "snr_breath", "snr_ac1", "csi_tv_s", "csi_ac1",
           "csi_xcorr", "csi_envbw", "csi_imp", "csi_madc", "agree_mr"]

# physical glossary for the LLM prompt (name-prefix -> meaning+direction)
GLOSSARY = {
    "ra_mean": "radar range-azimuth map mean energy; occupied -> higher",
    "re_mean": "radar range-elevation mean energy; occupied -> higher",
    "xy_mean": "radar top-view energy; occupied -> higher",
    "xy_std90": "radar top-view spatial spread; occupied -> higher",
    "xy_peak": "radar top-view peak energy; occupied -> higher",
    "ra_peak": "radar range-azimuth peak energy; occupied -> higher",
    "re_std90": "range-elevation spatial spread; occupied -> higher",
    "rd_std90": "range-Doppler spatial spread; occupied -> higher",
    "rd_mean": "range-Doppler mean energy; occupied -> higher",
    "rd_delta": "range-Doppler frame-change rate; occupied -> higher",
    "rd_motion": "per-second RD-map change rate; occupied -> higher",
    "rd_burst": "fraction of high-activity seconds; occupied -> higher",
    "rd_still": "fraction of still seconds; occupied -> lower",
    "rd_hotn": "hot-bin count on RD map; occupied -> higher",
    "rd_dopbw": "doppler bandwidth (limb/torso speed range); occ -> higher",
    "snr_mean": "radar clutter-masked SNR mean; occupied -> higher",
    "snr_max": "radar clutter-masked SNR peak; occupied -> higher",
    "snr_breath": "0.15-0.45Hz SNR band fraction (breathing); occ -> higher",
    "snr_ac1": "SNR persistence (lag-1 autocorr); occupied -> higher",
    "csi_amp_mean": "CSI mean amplitude level; site-dependent",
    "csi_amp_std": "CSI amplitude std; occupied -> higher",
    "csi_amp_q90": "CSI amplitude 90th pct; occupied -> higher",
    "csi_rv_mean": "CSI 0.2s rolling-variance mean; occupied -> higher",
    "csi_rv_q90": "CSI rolling-variance 90th pct; occupied -> higher",
    "csi_rv_max": "CSI rolling-variance max; occupied -> higher",
    "csi_tv": "CSI temporal total-variation; occupied -> higher",
    "csi_tv_s": "CSI per-second mean-amp variation; occupied -> higher",
    "csi_dop_frac": "CSI high-band PSD fraction (micro-doppler)",
    "csi_ac1": "CSI envelope persistence; occupied -> higher",
    "csi_xcorr": "CSI subcarrier coherence; empty(still) -> higher",
    "csi_envbw": "CSI envelope bandwidth; empty(electronic jitter) "
                 "-> higher, occupied(slow body motion) -> lower",
    "csi_imp": "CSI impulsiveness; occupied -> higher",
    "csi_madc": "CSI cross-subcarrier variance spread; weak",
    "agree_mr": "csi+radar activity agreement; occupied -> higher",
    "c_": "csi PCA position/variance stats (mean,iqr,min,max,std)",
    "r_": "radar PCA position/variance stats",
    "j_": "joint PCA position/variance stats",
    "traj": "PC-space trajectory: path/speed/tortuosity; occ -> higher",
    "clu_": "distance to unsupervised cluster centroids; weak",
    "fft_snr_": "radar SNR spectral bands + entropy",
    "fftc_": "100Hz CSI spectral bands (b0=low, b3=high) + entropy",
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_joined() -> dict[str, pd.DataFrame]:
    dfs = {}
    for s in ("train", "val", "calib", "testml"):
        a = pd.read_csv(PCA_DIR / f"desc_{s}_5s.csv")
        v2f = OUT / f"desc_v2_{s}.csv"
        if v2f.exists():
            b = pd.read_csv(v2f)
            keep = ["id", "s0"] + [c for c in V2_COLS if c in b.columns]
            a = a.merge(b[keep], on=["id", "s0"], how="inner")
        dfs[s] = a
    # full (unsampled) train table for ML fitting; v2 cols left-joined
    # (NaN outside the sampled recs -> median-imputed downstream)
    tf = pd.read_csv(PCA_DIR / "desc_train_5s.csv")
    v2f = OUT / "desc_v2_train.csv"
    if v2f.exists():
        b = pd.read_csv(v2f)
        keep = ["id", "s0"] + [c for c in V2_COLS if c in b.columns]
        tf = tf.merge(b[keep], on=["id", "s0"], how="left")
    dfs["train_full"] = tf
    return dfs


def views(dfs):
    cols = [c for c in dfs["train"].columns if c not in META]
    shared = [c for c in cols
              if all(c in d.columns and d[c].notna().mean() > 0.3
                     for d in dfs.values())]
    v = {
        "radar": [c for c in shared if c.startswith(RAD_PREFIX)],
        "csi": [c for c in shared if c.startswith(CSI_PREFIX)],
        "joint": [c for c in shared
                  if c.startswith(JNT_PREFIX)],
        "phys": [c for c in V2_COLS if c in shared],
    }
    v["all"] = shared
    # portable = physically-grounded descriptors only. PC coordinates
    # (c_*, r_*, j_*), PC trajectories (traj_*), cluster geometry in PC
    # space (clu_*) and PC-domain FFT bands (fftc_*, fftd_*) are NOT
    # portable: each split's PCA is fit independently, so axis signs /
    # subspaces differ across domains.
    npc = ("c_", "r_", "j_", "traj_", "clu_", "fftc_", "fftd_")
    v["port"] = [c for c in shared if not c.startswith(npc)]
    return v


def xy(df, cols):
    X = df[cols].values.astype(np.float32)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0)
    X = np.where(np.isfinite(X), X, med)
    return X, df.label.values.astype(int)


def session_pct(df, cols):
    """within-SESSION percentile rank (per id) -- weak: meaningless on
    single-label sessions.  Kept for ablation only."""
    out = df.copy()
    for c in cols:
        out[c] = df.groupby("id")[c].rank(pct=True)
    return out


class CdfEnc:
    """Class-conditional percentile encoding.

    For descriptor c:  pct_e(v) = ECDF_train_empty(v),
                       pct_o(v) = ECDF_train_occupied(v).
    A value at the 90th pct of empty but 20th of occupied votes EMPTY.
    Tail-extrapolation keeps evidence for out-of-train-range values:
    pct clipped to [1e-3, 1-1e-3] then logit-space so 0.999 vs 0.9
    still separates."""

    def __init__(self):
        self.cols = []
        self.se, self.so = {}, {}
        self.mid, self.iqr = {}, {}
        # per-class tail anchors for z extrapolation
        self.edge = {}          # c -> {cls: (lo, hi, iqr)}

    def fit(self, df, cols):
        self.cols = list(cols)
        y = df.label.values
        for c in self.cols:
            v = df[c].values.astype(np.float64)
            e = np.sort(v[y == 0])
            o = np.sort(v[y == 1])
            self.se[c] = e[np.isfinite(e)]
            self.so[c] = o[np.isfinite(o)]
            allv = np.concatenate([self.se[c], self.so[c]])
            self.edge[c] = {}
            for cls, arr in ((0, self.se[c]), (1, self.so[c])):
                if len(arr) < 10:
                    self.edge[c][cls] = (0.0, 0.0, 1.0)
                    continue
                lo, hi = np.percentile(arr, [2.5, 97.5])
                q = np.percentile(arr, [25, 75])
                self.edge[c][cls] = (float(lo), float(hi),
                                     float(max(q[1] - q[0], 1e-9)))
            if len(allv) == 0:
                self.mid[c], self.iqr[c] = 0.0, 1.0
                continue
            self.mid[c] = float(np.median(allv))
            q = np.percentile(allv, [25, 75])
            self.iqr[c] = float(max(q[1] - q[0], 1e-9))
        return self

    def _cdf(self, arr_sorted, v):
        if len(arr_sorted) == 0:
            return np.full_like(v, 0.5, dtype=np.float64)
        return np.searchsorted(arr_sorted, v) / len(arr_sorted)

    def transform(self, df):
        """returns dict c -> (pct_e, pct_o) float64 arrays in [0,1]."""
        out = {}
        for c in self.cols:
            v = df[c].values.astype(np.float64)
            v = np.where(np.isfinite(v), v, self.mid[c])
            pe = np.clip(self._cdf(self.se[c], v), 1e-3, 1 - 1e-3)
            po = np.clip(self._cdf(self.so[c], v), 1e-3, 1 - 1e-3)
            out[c] = (pe, po)
        return out

    @staticmethod
    def _probit(p):
        from scipy.special import ndtri
        return ndtri(p)

    def zscores(self, df):
        """per-descriptor probit z of the value under each class pool:
        z_e = sigmas from the empty pool's centre. PIT makes each class
        pool exactly N(0,1) in its own z-space. Out-of-range values
        saturate at z=±3.09 -> 'atypical of this pool'; the remaining
        in-range descriptors then carry the vote (empirically the most
        domain-robust choice)."""
        tt = self.transform(df)
        return {c: (self._probit(pe), self._probit(po))
                for c, (pe, po) in tt.items()}

    def evidence(self, df):
        """Gaussian LLR proxy in PIT space: (z_e^2 - z_o^2)/2.
        >0 => v closer to the occupied pool's centre => occupied."""
        zz = self.zscores(df)
        return {c: np.clip((ze ** 2 - zo ** 2) / 2, -8, 8)
                for c, (ze, zo) in zz.items()}


def metrics(y, p, proba):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"acc": float(accuracy_score(y, p)),
            "bal": float(balanced_accuracy_score(y, p)),
            "auc": float(roc_auc_score(y, proba))
            if len(np.unique(y)) > 1 else float("nan"),
            "rec_e": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "rec_o": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "n": int(len(y))}


def cal_metrics(y, proba, thr_sup):
    """raw acc @0.5 + unsup-cal (median, 2-means) + sup-cal thr."""
    proba = np.asarray(proba, float)
    out = {}
    out["acc"] = float(accuracy_score(y, proba >= 0.5))
    out["bal"] = float(balanced_accuracy_score(y, proba >= 0.5))
    out["auc"] = float(roc_auc_score(y, proba)) \
        if len(np.unique(y)) > 1 else float("nan")
    p_med = (proba >= np.median(proba)).astype(int)
    out["acc_med"] = float(accuracy_score(y, p_med))
    if np.unique(proba).size > 2:
        ctr = KMeans(n_clusters=2, n_init=10, random_state=0) \
            .fit(proba.reshape(-1, 1)).cluster_centers_.ravel()
        thr_k = float(np.sort(ctr).mean())
    else:
        thr_k = float(np.median(proba))
    out["acc_km"] = float(accuracy_score(y, proba >= thr_k))
    out["acc_sup"] = float(accuracy_score(y, proba >= thr_sup))
    tn, fp, fn, tp = confusion_matrix(
        y, (proba >= thr_sup).astype(int), labels=[0, 1]).ravel()
    out.update(rec_e=float(tn / max(tn + fp, 1)),
               rec_o=float(tp / max(tp + fn, 1)), n=int(len(y)),
               thr_sup=float(thr_sup),
               cm=[[int(tn), int(fp)], [int(fn), int(tp)]])
    return out


def best_thr(y, prob):
    c = np.linspace(0.05, 0.95, 91)
    ba = [balanced_accuracy_score(y, prob >= t) for t in c]
    return float(np.median(c[np.flatnonzero(np.array(ba) == max(ba))]))


# ---------------------------------------------------------------------------
# PIPELINE 1: rule-based
# ---------------------------------------------------------------------------
class PhysVote:
    """Evidence-weighted vote on CLASS-CONDITIONAL percentiles.

    Per descriptor i: evidence_i = logit(pct_occ(v)) - logit(pct_empty(v))
    (>0 => occupied).  score = sum_i w_i*evidence_i / sum_i w_i with
    w = train |cohen d|.  Ranks w.r.t. class pools are domain-robust:
    amplitude drift shifts both CDFs similarly."""

    def __init__(self, topk=16):
        self.topk = topk

    def fit(self, df, cols, extra=()):
        """cols ranked by train |cohen d|; `extra` cols are always
        appended (curated physics descriptors, weight = their |d|)."""
        y = df.label.values
        sc = {}
        for c in cols:
            v = df[c].values
            a, b = v[y == 0], v[y == 1]
            if np.isfinite(a).sum() < 20 or np.isfinite(b).sum() < 20:
                continue
            d = ((np.nanmean(b) - np.nanmean(a))
                 / np.sqrt((np.nanstd(a) ** 2 + np.nanstd(b) ** 2) / 2
                           + 1e-12))
            if np.isfinite(d) and abs(d) > 0.05:
                sc[c] = abs(d)
        order = sorted(sc, key=sc.get, reverse=True)
        pick = order[:self.topk]
        for c in extra:                     # curated phys set, capped
            if c in sc and c not in pick:
                pick.append(c)
        pick = pick[:self.topk + len(extra)]
        self.feats = [(c, sc[c]) for c in pick]
        self.cols = pick
        self.w = np.array([sc[c] for c in pick])
        self.dir = np.array([
            np.sign(np.nanmean(df[c][y == 1]) - np.nanmean(df[c][y == 0]))
            or 1.0 for c in self.cols])
        self.enc = CdfEnc().fit(df, self.cols)
        return self

    @staticmethod
    def session_evidence(df, cols, dirs):
        """deviation from the session's own quiet baseline:
        (v - q25_session) / IQR_session, signed by train direction.
        Label-free: every session has low-activity moments."""
        g = df.groupby("id")
        out = {}
        for c, d in zip(cols, dirs):
            q25 = g[c].transform(lambda x: x.quantile(.25))
            iqr = (g[c].transform(lambda x: x.quantile(.75))
                   - q25).clip(lower=1e-9)
            z = ((df[c] - q25) / iqr).fillna(0).values
            out[c] = d * np.clip(z, 0, 4)
        return out

    def score(self, df):
        ev = self.enc.evidence(df)
        X = np.stack([ev[c] for c in self.cols], 1)
        return (X * self.w[None, :]).sum(1) / (self.w.sum() + 1e-12)

    def predict_proba(self, df):
        s = self.score(df)
        return 1 / (1 + np.exp(-s))


class Stump:
    def fit(self, X, y):
        best = (None, -1, 0, .5)
        for j in range(X.shape[1]):
            v = X[:, j]
            if np.nanstd(v) < 1e-9:
                continue
            for t in np.percentile(v, np.linspace(5, 95, 19)):
                for sgn in (1, -1):
                    p = ((v * sgn) > (t * sgn)).astype(int)
                    b = balanced_accuracy_score(y, p)
                    if b > best[1]:
                        best = (j, b, t, sgn)
        self.j, self.t, self.sgn = best[0], best[2], best[3]
        return self

    def proba(self, X):
        d = (X[:, self.j] - self.t) * self.sgn
        return 1 / (1 + np.exp(-d / (np.std(d) + 1e-9)))


class ThreshCal:
    """mid-point thresholds on top-K AUC descriptors (vote rule)."""
    def __init__(self, k=8):
        self.k = k

    def fit(self, X, y):
        aucs = []
        for j in range(X.shape[1]):
            v = X[:, j]
            if np.nanstd(v) < 1e-9:
                aucs.append(0.5)
                continue
            a = roc_auc_score(y, v)
            aucs.append(max(a, 1 - a))
        self.idx = np.argsort(aucs)[-self.k:]
        self.sign = {j: (1 if roc_auc_score(y, X[:, j]) > .5 else -1)
                     for j in self.idx}
        self.thr = {j: np.percentile(X[:, j], 50) for j in self.idx}
        return self

    def calibrate(self, X_cal):
        for j in self.idx:
            self.thr[j] = (self.thr[j] + np.median(X_cal[:, j])) / 2
        return self

    def proba(self, X):
        v = np.zeros(len(X))
        for j in self.idx:
            v += ((X[:, j] - self.thr[j]) * self.sign[j] > 0)
        return v / len(self.idx)


def run_rule(dfs, V, ref_probs):
    probs = {}

    # --- physvote on class-conditional percentiles (portable only) ---
    pv = PhysVote(topk=16).fit(dfs["train"], V["port"],
                               extra=V["phys"])
    for s in ("train", "val", "calib", "testml"):
        probs.setdefault("physvote", {})[s] = pv.predict_proba(dfs[s])
    print("  physvote feats:", pv.cols[:8], flush=True)

    # --- stump + knn + threshcal on raw 'all' (full train fit) -------
    Xtr, ytr = xy(dfs["train_full"], V["all"])
    sc = StandardScaler().fit(Xtr)
    st = Stump().fit(sc.transform(Xtr), ytr)
    knn = KNeighborsClassifier(5, weights="distance", n_jobs=-1) \
        .fit(sc.transform(Xtr), ytr)
    tc = ThreshCal(k=8).fit(sc.transform(Xtr), ytr)
    if "calib" in dfs:
        Xc, _ = xy(dfs["calib"], V["all"])
        tc.calibrate(sc.transform(Xc))
    Xtrj, _ = xy(dfs["train"], V["all"])
    probs["stump"] = {"train": st.proba(sc.transform(Xtrj))}
    probs["knn"] = {"train": knn.predict_proba(sc.transform(Xtrj))[:, 1]}
    probs["threshcal"] = {"train": tc.proba(sc.transform(Xtrj))}
    for s in ("val", "calib", "testml"):
        Xs, _ = xy(dfs[s], V["all"])
        probs["stump"][s] = st.proba(sc.transform(Xs))
        probs["knn"][s] = knn.predict_proba(sc.transform(Xs))[:, 1]
        probs["threshcal"][s] = tc.proba(sc.transform(Xs))
        # unsupervised _norm variant for stump/knn
        sc_e = StandardScaler().fit(Xs)
        Xn = sc.inverse_transform(sc_e.transform(Xs))
        probs["stump"][s + "_norm"] = st.proba(sc.transform(Xn))
        probs["knn"][s + "_norm"] = \
            knn.predict_proba(sc.transform(Xn))[:, 1]
    return probs, pv


# ---------------------------------------------------------------------------
# PIPELINE 2: ml / dl
# ---------------------------------------------------------------------------
def ml_zoo():
    return {
        "svm": SVC(kernel="rbf", C=4, probability=True,
                   class_weight="balanced", random_state=0),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=12,
                                     n_jobs=-1, random_state=0,
                                     class_weight="balanced"),
        "gbm": HistGradientBoostingClassifier(max_iter=300,
                                              learning_rate=.07,
                                              random_state=0),
    }


def run_mldl(dfs, V):
    probs = {}
    # 'sel' = top-24 |train AUC| of 'all' (per model_compare)
    Xa, ya = xy(dfs["train_full"], V["all"])
    aucs = {}
    for i, c in enumerate(V["all"]):
        v = Xa[:, i]
        m = np.isfinite(v)
        if m.sum() > 50 and np.nanstd(v[m]) > 1e-9:
            a = roc_auc_score(ya[m], v[m])
            aucs[c] = max(a, 1 - a)
    V["sel"] = sorted(aucs, key=aucs.get, reverse=True)[:24]

    for v in ("csi", "joint", "sel", "phys", "all"):
        cols = V[v]
        if not cols:
            continue
        Xtr, ytr = xy(dfs["train_full"], cols)
        for mn, clf in ml_zoo().items():
            key = f"{mn}/{v}"
            sc = StandardScaler().fit(Xtr)
            try:
                clf.fit(sc.transform(Xtr), ytr)
            except Exception as e:
                print(f"  {key} fit fail {e}", flush=True)
                continue
            Xtrj, _ = xy(dfs["train"], cols)
            probs[key] = {"train": clf.predict_proba(
                sc.transform(Xtrj))[:, 1]}
            for s in ("val", "calib", "testml"):
                Xs, _ = xy(dfs[s], cols)
                probs[key][s] = clf.predict_proba(
                    sc.transform(Xs))[:, 1]
                sc_e = StandardScaler().fit(Xs)
                Xn = sc.inverse_transform(sc_e.transform(Xs))
                Xn = np.where(np.isfinite(Xn), Xn, 0)
                probs[key][s + "_norm"] = clf.predict_proba(
                    sc.transform(Xn))[:, 1]
            print(f"  {key} done", flush=True)

    # --- pct_mlp: MLP on class-conditional percentile features
    #     (2*k dims: pct_empty + pct_occ) -- domain-invariant inputs
    pcols = [c for c in V["sel"]]
    enc = CdfEnc().fit(dfs["train_full"], pcols)

    def px(split):
        tt = enc.transform(dfs[split])
        return np.concatenate(
            [np.stack([tt[c][0] for c in pcols], 1),
             np.stack([tt[c][1] for c in pcols], 1)], axis=1)

    mlp = MLPClassifier((64, 32), alpha=1e-3, max_iter=500,
                        random_state=0)
    mlp.fit(px("train_full"), dfs["train_full"].label.values)
    probs["pct_mlp"] = {"train": mlp.predict_proba(px("train"))[:, 1]}
    for s in ("val", "calib", "testml"):
        probs["pct_mlp"][s] = mlp.predict_proba(px(s))[:, 1]
    print("  pct_mlp done", flush=True)

    # --- seqcnn on joint PC-dot sequences (joined rows only) --------
    try:
        probs["seqcnn"] = seqcnn_probs(dfs)
        print("  seqcnn done", flush=True)
    except Exception as e:
        print(f"  seqcnn failed: {e}", flush=True)
    return probs


def seqcnn_probs(dfs):
    import torch
    from torch import nn
    torch.manual_seed(0)
    base = {"val": "val", "calib": "calib", "testml": "testml",
            "train": "train"}
    zs = {s: np.load(PCA_DIR / f"pc_dots_{s}.npz") for s in base}

    def seqs(split):
        arr, idx = [], []
        df = dfs[split]
        for j, (rid, s0) in enumerate(zip(df.id, df.s0.astype(int))):
            if rid not in zs[split]:
                continue
            a = zs[split][rid]
            seg = a[s0:s0 + 5, 1:7].T.astype(np.float32)
            if seg.shape[1] == 5:
                arr.append(seg)
                idx.append(j)
        return arr, idx

    Xtr_l, itr = seqs("train")
    Xtr = torch.tensor(np.stack(Xtr_l))
    ytr = torch.tensor(dfs["train"].label.values[itr], dtype=torch.float32)

    net = nn.Sequential(
        nn.Conv1d(6, 32, 3, padding=1), nn.ReLU(),
        nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveMaxPool1d(1), nn.Flatten(),
        nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_w = torch.tensor([float((1 - ytr.mean())
                                / (ytr.mean() + 1e-9))]).clamp(.5, 4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(25):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            lossf(net(Xtr[b]).squeeze(-1), ytr[b]).backward()
            opt.step()
    out = {}
    for s in ("train", "val", "calib", "testml"):
        arrs, idx = seqs(s)
        pr = np.full(len(dfs[s]), np.nan, np.float32)
        if arrs:
            with torch.no_grad():
                pr[idx] = torch.sigmoid(
                    net(torch.tensor(np.stack(arrs))).squeeze(-1)
                ).numpy()
        out[s] = pr
    return out


# ---------------------------------------------------------------------------
# PIPELINE 3: llm (improved jev) + offline surrogate
# ---------------------------------------------------------------------------
LLM_SYS = """You are a strict binary occupancy judge for a room monitored by a
60 GHz mmWave radar + Wi-Fi CSI sensor. Each 5 s window is described by
physical descriptors encoded as CLASS-CONDITIONAL PERCENTILES against
the labelled train pool:
  pct_empty = ECDF rank of the window's value inside train-EMPTY
              windows   (0..1; .9 = typical of empty, .1 = atypical)
  pct_occ   = ECDF rank inside train-OCCUPIED windows.
So pct_occ - pct_empty > 0 votes OCCUPIED. This encoding is
domain-invariant: it asks "which class does this value look like",
not "is the value big".

DESCRIPTOR GLOSSARY (name -> physical meaning):
{glossary}

DECISION POLICY:
1. Radar map-energy / spread descriptors (ra_, re_, xy_, rd_, snr_) and
   CSI variance descriptors (csi_rv, csi_amp_std, csi_tv) high => motion
   energy => occupied.
2. Quiet-baseline descriptors vote the OPPOSITE way: csi_envbw
   (electronic-jitter bandwidth) and csi_xcorr (channel coherence) are
   HIGH in empty rooms and LOW when a body moves.
3. PC-trajectory descriptors (traj/c_/r_/j_ std,iqr,path,speed) high =>
   persistent motion => occupied.
4. Weigh the pre-computed votes by descriptor reliability: radar map
   energies and csi_envbw are the most cross-site stable; amplitude
   levels (csi_amp_*, snr_mean) drift with site and should only break
   ties.
5. A sleeper is still OCCUPIED: rely on snr_breath (0.15-0.45 Hz band)
   and residual micro-motion rather than large motion alone.
6. If votes are near-balanced or coverage < 0.5, output ~0.5.

Reply with JSON only:
{{"occupied": <0..1>, "evidence": ["<top1>","<top2>","<top3>"],
 "reason": "<=15 words>"}}"""


def _glossary_text(cols):
    lines = []
    for c in cols:
        hit = next((f"{k}: {v}" for k, v in GLOSSARY.items()
                    if c == k or c.startswith(k)), None)
        lines.append(f"- {hit or c}")
    return "\n".join(lines)


def llm_class_profiles(df_train, cols):
    """raw-value class reference quantiles for the prompt."""
    s = {}
    for lab, name in ((0, "empty"), (1, "occupied")):
        d = df_train[df_train.label == lab]
        s[name] = {c: {"q10": round(float(np.nanquantile(d[c], .1)), 3),
                       "med": round(float(np.nanmedian(d[c])), 3),
                       "q90": round(float(np.nanquantile(d[c], .9)), 3)}
                   for c in cols}
    return s


def llm_exemplars(df_train, enc, cols, k=3):
    """few-shot: k empty + k occupied train windows with the most
    decisive evidence in each direction (clear archetypes)."""
    ev = enc.evidence(df_train)
    E = np.stack([np.clip(ev[c], -4, 4) for c in cols], 1).sum(1)
    ex = []
    zz = enc.zscores(df_train)
    for lab, name in ((0, "empty"), (1, "occupied")):
        sub = np.where(df_train.label.values == lab)[0]
        order = sub[np.argsort(E[sub])]
        pick = order[:k] if lab == 0 else order[-k:]
        for i in pick:
            ex.append({"label": name,
                       "z_empty": {c: round(float(zz[c][0][i]), 1)
                                   for c in cols},
                       "z_occ": {c: round(float(zz[c][1][i]), 1)
                                 for c in cols}})
    return ex


def pick_cols_stable(dfs, k=12, pool=None):
    """top-k by train |cohen d|, restricted to descriptors whose
    class-separation direction is consistent between train AND val
    (direction-stable = cross-domain)."""
    def dvec(df):
        y = df.label.values
        out = {}
        for c in df.columns:
            if c in META or (pool is not None and c not in pool):
                continue
            v = df[c].values
            a, b = v[y == 0], v[y == 1]
            if np.isfinite(a).sum() < 20 or np.isfinite(b).sum() < 20:
                continue
            sp = np.sqrt((np.nanstd(a) ** 2 + np.nanstd(b) ** 2) / 2
                         + 1e-12)
            out[c] = (np.nanmean(b) - np.nanmean(a)) / sp
        return out
    dt, dv = dvec(dfs["train"]), dvec(dfs["val"])
    cand = sorted(dt, key=lambda c: -abs(dt[c]))[:k * 3]
    stable = [c for c in cand
              if c in dv and np.sign(dt[c]) == np.sign(dv[c])
              and dt[c] != 0][:k]
    if len(stable) < 5:
        stable = sorted(dt, key=lambda c: -abs(dt[c]))[:k]
    return stable


def _post(url, headers, payload, retries=3, timeout=45):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
    return {}


def llm_call(prompt, cfg):
    if cfg["provider"] == "anthropic":
        out = _post(f"{cfg['base']}/messages",
                    {"x-api-key": cfg["key"],
                     "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"},
                    {"model": cfg["model"], "max_tokens": 250,
                     "temperature": 0, "system": cfg["sys"],
                     "messages": [{"role": "user",
                                  "content": prompt}]})
        return "".join(b.get("text", "") for b in out.get("content", []))
    out = _post(f"{cfg['base']}/chat/completions",
                {"Authorization": f"Bearer {cfg['key']}",
                 "Content-Type": "application/json"},
                {"model": cfg["model"], "temperature": 0,
                 "messages": [{"role": "system", "content": cfg["sys"]},
                              {"role": "user", "content": prompt}],
                 "response_format": {"type": "json_object"}})
    return out["choices"][0]["message"]["content"]


def llm_prompt(i, tt, zz, cols, profs, exemplars):
    enc = {}
    votes = []
    for c in cols:
        pe, po = tt[c][0][i], tt[c][1][i]
        ze, zo = zz[c][0][i], zz[c][1][i]
        enc[c] = {"pct_empty": round(float(pe), 2),
                  "pct_occ": round(float(po), 2),
                  "z_empty": round(float(ze), 1),
                  "z_occ": round(float(zo), 1)}
        llr = (ze * ze - zo * zo) / 2
        if abs(llr) > 1.0:
            votes.append(f"{c}->{'occupied' if llr > 0 else 'empty'}")
    return ("Window descriptor encoding. For each descriptor: "
            "pct_empty/pct_occ = ECDF rank of this value inside the "
            "train empty/occupied pools; z_empty/z_occ = equivalent "
            "sigmas from that pool's centre (|z|>2 = atypical).\n"
            + json.dumps(enc)
            + "\nPre-computed votes (|LLR|>1): " + json.dumps(votes)
            + "\nClass raw-value profiles (train q10/med/q90): "
            + json.dumps(profs)
            + "\nLabeled archetype windows (train): "
            + json.dumps(exemplars)
            + "\nReturn JSON only.")


def _parse_llm(txt):
    try:
        j = json.loads(txt)
        return float(np.clip(float(j["occupied"]), 0, 1))
    except Exception:
        import re
        m = re.search(r'"occupied"\s*:\s*([0-9.]+)', txt)
        return float(np.clip(float(m.group(1)), 0, 1)) if m else 0.5


def run_llm(dfs, cols, sample_n=1200):
    """returns {set: proba} for llm (API) and llm_off (surrogate)."""
    enc = CdfEnc().fit(dfs["train"], cols)
    profs = llm_class_profiles(dfs["train"], cols)
    exemplars = llm_exemplars(dfs["train"], enc, cols)
    out = {"llm_off": {}}
    # deterministic surrogate: weighted evidence vote on the same
    # (pct_e,pct_o) pairs the prompt hands to the LLM - measurable
    # without an API key.
    ytr = dfs["train"].label.values
    w = np.array([
        abs(np.nanmean(dfs["train"][c][ytr == 1])
            - np.nanmean(dfs["train"][c][ytr == 0]))
        / np.sqrt((np.nanstd(dfs["train"][c][ytr == 1]) ** 2
                   + np.nanstd(dfs["train"][c][ytr == 0]) ** 2) / 2
                  + 1e-12)
        for c in cols])
    dirs = np.array([
        np.sign(np.nanmean(dfs["train"][c][ytr == 1])
                - np.nanmean(dfs["train"][c][ytr == 0])) or 1.0
        for c in cols])
    w = np.where(np.isfinite(w), w, 0)
    w /= w.sum() + 1e-12

    def surrogate(df):
        # what a competent LLM does with the prompt: |LLR|>1 votes vs
        # the train pools, weighted by descriptor reliability w.
        ev = enc.evidence(df)
        X = np.stack([ev[c] for c in cols], 1)
        s = (X * w[None, :] * (np.abs(X) > 1.0)).sum(1)
        return 1 / (1 + np.exp(-s))

    key = os.environ.get("JEV_API_KEY")
    cfg = None
    if key:
        cfg = {"key": key,
               "base": os.environ.get("JEV_BASE_URL",
                                      "https://api.openai.com/v1"),
               "model": os.environ.get("JEV_MODEL", "gpt-4o-mini"),
               "provider": os.environ.get("JEV_PROVIDER", "openai"),
               "sys": LLM_SYS.format(glossary=_glossary_text(cols))}
        out["llm"] = {}
    # surrogate is free -> full tables on every split (gives a labelled
    # train reference for supervised calibration)
    for s in ("train", "val", "calib", "testml"):
        if s not in dfs:
            continue
        out["llm_off"][s] = surrogate(dfs[s])
        out["llm_off"][s + "_idx"] = dfs[s].index.values
    for s in ("val", "testml"):
        df = dfs[s]
        if sample_n and len(df) > sample_n:
            df = df.groupby(["id", "label"], group_keys=False) \
                   .apply(lambda g: g.sample(
                       max(1, int(sample_n * len(g) / len(dfs[s]))),
                       random_state=0))
        if cfg:
            tt = enc.transform(df)
            zz = enc.zscores(df)
            pr = np.empty(len(df), np.float32)
            for i in range(len(df)):
                pr[i] = _parse_llm(
                    llm_call(llm_prompt(i, tt, zz, cols, profs,
                                        exemplars), cfg))
                if (i + 1) % 50 == 0:
                    print(f"    llm/{s} {i + 1}/{len(df)}", flush=True)
            out["llm"][s] = pr
            out["llm"][s + "_idx"] = df.index.values
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true",
                    help="also call the real LLM (JEV_API_KEY)")
    ap.add_argument("--skip-ml", action="store_true",
                    help="skip the ML/DL zoo (fast iteration)")
    ap.add_argument("--llm-n", type=int, default=1200,
                    help="max windows per eval set for llm pipelines")
    args = ap.parse_args()
    t0 = time.time()

    dfs = load_joined()
    V = views(dfs)
    print("rows:", {s: len(d) for s, d in dfs.items()}, flush=True)
    print("views:", {k: len(v) for k, v in V.items()}, flush=True)

    probs = {}
    print("[rule]", flush=True)
    rule_probs, pv = run_rule(dfs, V, None)
    probs.update(rule_probs)
    if not args.skip_ml:
        print("[mldl]", flush=True)
        probs.update(run_mldl(dfs, V))

    # llm reads the same portable descriptor set physvote selected
    # (empirically cross-domain: top-|d| on train, portable pool)
    llm_cols = pv.cols[:12]
    print(f"[llm] cols ({len(llm_cols)}): {llm_cols}", flush=True)
    lres = run_llm(dfs, llm_cols, sample_n=args.llm_n)
    for name, sets in lres.items():
        if name == "llm" and not args.llm:
            continue
        probs[name] = {s: v for s, v in sets.items()
                       if not s.endswith("_idx")}
        probs[name + "__idx"] = {s[:-4]: v for s, v in sets.items()
                                 if s.endswith("_idx")}

    # ---- metrics: thr_sup tuned on a labelled reference --------------
    # val probs for testml; train probs for val/calib.
    rows = []
    for name, sets in probs.items():
        if name.endswith("__idx"):
            continue
        for s in ("val", "testml", "calib"):
            if s not in sets or s.endswith("_norm"):
                continue
            pr = np.asarray(sets[s], float)
            idx = probs.get(name + "__idx", {}).get(s)
            ysrc = dfs[s] if idx is None else dfs[s].loc[idx]
            y = ysrc.label.values.astype(int)
            m = np.isfinite(pr)
            if m.sum() < 20:
                continue
            pr, y = pr[m], y[m]
            ref = "train" if s == "val" else "val"
            if s == "calib":
                ref = "val"
            pr_ref = np.asarray(sets.get(ref, sets.get("train", pr)),
                                float)
            idx_ref = probs.get(name + "__idx", {}).get(ref)
            y_ref = (dfs[ref].loc[idx_ref].label.values.astype(int)
                     if idx_ref is not None
                     else dfs[ref].label.values.astype(int)
                     [:len(pr_ref)])
            mr = np.isfinite(pr_ref)
            thr = best_thr(y_ref[mr], pr_ref[mr]) \
                if mr.sum() > 20 else 0.5
            mm = cal_metrics(y, pr, thr)
            mm.update(model=name, set=s)
            rows.append(mm)
            # *_norm variants (unsupervised re-standardisation)
            sn = s + "_norm"
            if sn in sets:
                pr2 = np.asarray(sets[sn], float)[m]
                thr2 = best_thr(y_ref[mr], pr_ref[mr]) \
                    if mr.sum() > 20 else 0.5
                mm2 = cal_metrics(y, pr2, thr2)
                mm2.update(model=name + "_norm", set=s)
                rows.append(mm2)

    res = pd.DataFrame(rows)
    res = res.sort_values(["set", "bal"], ascending=[True, False])
    res.to_csv(OUT / "results.csv", index=False)

    # ---- report -----------------------------------------------------
    lines = ["# Three-pipeline occupancy results (W=5s)",
             "",
             "Metrics: acc @0.5 | bal | auc | acc_med/acc_km (unsup-cal) "
             "| acc_sup (thr tuned on labelled reference set) | "
             "rec_e/rec_o | cm = sup-cal confusion matrix [[tn,fp],"
             "[fn,tp]]", ""]
    for s in ("val", "testml"):
        lines.append(f"## {s}")
        lines.append("| model | acc | bal | auc | acc_med | acc_km | "
                     "acc_sup | rec_e | rec_o | n |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        sub = res[res.set == s]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r.model} | {r.acc:.3f} | {r.bal:.3f} | {r.auc:.3f} "
                f"| {r.acc_med:.3f} | {r.acc_km:.3f} | {r.acc_sup:.3f} "
                f"| {r.rec_e:.3f} | {r.rec_o:.3f} | {int(r.n)} |")
        lines.append("")
    lines.append("## Confusion matrices (best per pipeline, sup-cal thr)")
    for pipe in ("physvote", "pct_mlp", "seqcnn", "llm_off", "llm"):
        sub = res[(res.model.str.startswith(pipe))
                  & res.set.isin(["val", "testml"])]
        if not len(sub):
            continue
        for _, r in sub.iterrows():
            lines.append(f"- `{r.model}` {r.set}: {r.cm}")
    (OUT / "REPORT.md").write_text("\n".join(lines))

    np.savez_compressed(
        OUT / "probs.npz",
        **{f"{n}|{s}": v for n, sets in probs.items()
           for s, v in sets.items() if not s.endswith("_idx")})
    print(f"done {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
