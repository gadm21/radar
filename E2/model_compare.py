"""E2 model comparison — ml / dl / rule / jev over descriptor views.

Splits (E1 pca_deep outputs, W=5 s windows):
    train   legacy train_minutes/train (274 min, wall2)
    val     legacy validation_minutes    (192 min, wall3)
    calib   40 calibration_minutes       (all t1_sleep -> occupied)
    testml  multilink_test_minutes       (4 x 1 h, 2 empty + 2 occ)

Descriptor views (row-aligned window tables):
    radar   snr_/rd_/ra_/re_/xy_/radar-PC descriptors/fft_snr_*
    csi     csi_* physical + csi-PC descriptors
    joint   j_* (6-dim joint-PC window stats) + clu_*
    all     union (shared cols only -> cross-domain safe)

Workflows:
    A  descriptor fusion      per-view models + 'all' model
    B  shared embedding       PCA (fit on train shared descriptors) ->
                              k-dim embedding -> models
    C  probability fusion     per-view model probs -> mean/max/weighted
                              (weights learned on val)

Metrics per eval: acc, bal_acc, f1, auc, recall_e, recall_o.
Calibration checks:
    cal_sup    first 10 min of t1_sleep calib (occupied-only recall)
    cal_unsup  5 empty + 5 occupied minutes sampled from val (acc)
ROC + confusion matrices for best model per class.  jev runs only if
E2/jev.py is configured (JEV_API_KEY).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

E2 = Path(__file__).resolve().parent
ROOT = E2.parent
PCA_DIR = ROOT / "E1" / "outputs" / "pca_deep"
OUT = E2 / "outputs" / "model_compare"
OUT.mkdir(parents=True, exist_ok=True)
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(E2))

from sklearn.decomposition import PCA                     # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,
                             RandomForestClassifier)      # noqa: E402
from sklearn.linear_model import LogisticRegression       # noqa: E402
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)    # noqa: E402
from sklearn.neighbors import KNeighborsClassifier        # noqa: E402
from sklearn.neural_network import MLPClassifier          # noqa: E402
from sklearn.preprocessing import StandardScaler          # noqa: E402
from sklearn.svm import SVC                               # noqa: E402

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)

META = {"id", "s0", "label", "session", "cov"}
RAD_PREFIX = ("snr_", "rd_", "ra_", "re_", "xy_", "r_",
              "fft_snr_", "rad_")
CSI_PREFIX = ("csi_", "c_")
JNT_PREFIX = ("j_", "clu_")
DROP = META | {"clu_kmlabel"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load():
    dfs = {}
    for s in ("train", "val", "calib", "testml"):
        f = PCA_DIR / f"desc_{s}_5s.csv"
        if f.exists():
            dfs[s] = pd.read_csv(f)
    return dfs


def views(dfs):
    cols = [c for c in dfs["train"].columns if c not in DROP]
    shared = [c for c in cols
              if all(c in d.columns for d in dfs.values())]
    v = {
        "radar": [c for c in shared if c.startswith(RAD_PREFIX)],
        "csi": [c for c in shared if c.startswith(CSI_PREFIX)],
        "joint": [c for c in shared if c.startswith(JNT_PREFIX)],
    }
    v["all"] = sorted(set(v["radar"] + v["csi"] + v["joint"]))
    return v


def xy(df, cols):
    X = df[cols].values.astype(np.float32)
    X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    return X, df.label.values.astype(int)


def metrics(y, p, proba):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"acc": float(accuracy_score(y, p)),
            "bal": float(balanced_accuracy_score(y, p)),
            "f1": float(f1_score(y, p)),
            "auc": (float(roc_auc_score(y, proba))
                    if len(np.unique(y)) > 1 else float("nan")),
            "re": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "ro": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "n": int(len(y))}


# ---------------------------------------------------------------------------
# model zoo
# ---------------------------------------------------------------------------
def ml_zoo():
    return {
        "logreg": LogisticRegression(C=1.0, max_iter=2000,
                                     class_weight="balanced"),
        "svm": SVC(kernel="rbf", C=4, probability=True,
                   class_weight="balanced", random_state=0),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=12,
                                     n_jobs=-1, random_state=0,
                                     class_weight="balanced"),
        "gbm": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=.07, random_state=0),
        "mlp": MLPClassifier((64, 32), alpha=1e-3, max_iter=400,
                             random_state=0),
    }


def fit(clf, X, y):
    sc = StandardScaler().fit(X)
    clf.fit(sc.transform(X), y)
    return sc


def predict(clf, sc, X):
    p = clf.predict(sc.transform(X))
    try:
        pr = clf.predict_proba(sc.transform(X))[:, 1]
    except Exception:
        pr = p.astype(float)
    return p, pr


class Stump:
    """rule: best single-feature threshold on train (max bal-acc)."""
    def fit(self, X, y):
        best = (None, -1, 0, .5)
        for j in range(X.shape[1]):
            v = X[:, j]
            for t in np.percentile(v, np.linspace(5, 95, 19)):
                for sgn in (1, -1):
                    p = ((v * sgn) > (t * sgn)).astype(int)
                    b = balanced_accuracy_score(y, p)
                    if b > best[1]:
                        best = (j, b, t, sgn)
        self.j, self.bal, self.t, self.sgn = best
        return self

    def predict(self, X):
        return ((X[:, self.j] * self.sgn) >
                (self.t * self.sgn)).astype(int)

    def predict_proba(self, X):
        d = (X[:, self.j] - self.t) * self.sgn
        pr = 1 / (1 + np.exp(-d / (np.std(d) + 1e-9)))
        return np.c_[1 - pr, pr]


class CentroidRule:
    """rule: nearest class centroid in scaled space; 'occ' centroid may
    come from a calibration block instead of train."""

    def __init__(self, occ_centroid=None):
        self.occ_centroid = occ_centroid

    def fit(self, X, y):
        self.c0 = X[y == 0].mean(0)
        self.c1 = (self.occ_centroid
                   if self.occ_centroid is not None
                   else X[y == 1].mean(0))
        return self

    def predict(self, X):
        return (np.linalg.norm(X - self.c1, axis=1)
                < np.linalg.norm(X - self.c0, axis=1)).astype(int)

    def predict_proba(self, X):
        d0 = np.linalg.norm(X - self.c0, axis=1)
        d1 = np.linalg.norm(X - self.c1, axis=1)
        pr = 1 / (1 + np.exp(d0 - d1))
        return np.c_[1 - pr, pr]


class ClusterVote:
    """rule: vote of the pca_deep kmeans label -> class via train
    majority."""

    def fit(self, X, y, km_labels):
        self.map_ = {}
        for c in np.unique(km_labels):
            m = km_labels == c
            self.map_[c] = int(np.bincount(y[m]).argmax())
        return self

    def predict(self, X, km_labels):
        return np.array([self.map_.get(c, 0) for c in km_labels])


class ThreshCal:
    """rule: mid-point thresholds on top-K descriptors, threshold set
    between train-empty mean and CALIBRATION occupied mean — the
    'thresholds learned during calibration' rule."""

    def __init__(self, k=8):
        self.k = k

    def fit(self, X, y):
        # rank features by |auc| on train
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
        """move each threshold to midpoint between stored empty-train
        median and the calibration (occupied) median."""
        for j in self.idx:
            e, o = self.thr[j], np.median(X_cal[:, j])
            self.thr[j] = (e + o) / 2
        return self

    def predict(self, X):
        v = np.zeros(len(X))
        for j in self.idx:
            v += ((X[:, j] - self.thr[j]) * self.sign[j] > 0)
        return (v > len(self.idx) / 2).astype(int)

    def predict_proba(self, X):
        v = np.zeros(len(X))
        for j in self.idx:
            v += ((X[:, j] - self.thr[j]) * self.sign[j] > 0)
        pr = v / len(self.idx)
        return np.c_[1 - pr, pr]


def rule_zoo():
    return {"stump": Stump(),
            "knn": KNeighborsClassifier(5, weights="distance",
                                        n_jobs=-1)}


# ---------------------------------------------------------------------------
# torch seq-CNN on joint PC sequences
# ---------------------------------------------------------------------------
def seqcnn_eval(dfs):
    import torch
    from torch import nn
    torch.manual_seed(0)

    def seqs_for(dfs_map):
        base_of = {"cal_sup": "calib", "cal_unsup": "val"}
        z = {base_of.get(s, s): np.load(
            PCA_DIR / f"pc_dots_{base_of.get(s, s)}.npz")
            for s in dfs_map}
        out = {}
        for s, df in dfs_map.items():
            base = base_of.get(s, s)
            arr, ys = [], []
            for rid, g in df.groupby("id"):
                a = z[base][rid]
                for s0 in g.s0.astype(int):
                    seg = a[s0:s0 + 5, 1:7].T.astype(np.float32)
                    if seg.shape[1] == 5:
                        arr.append(seg)
                        ys.append(int(g.label.iloc[0]))
            out[s] = (arr, ys)
        return out

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.f = nn.Sequential(
                nn.Conv1d(6, 32, 3, padding=1), nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(s, x):
            return s.f(x).squeeze(-1)

    smap = {s: df for s, df in dfs.items()}
    cal = dfs.get("calib")
    if cal is not None:
        ids = list(dict.fromkeys(cal.id))[:10]
        smap["cal_sup"] = cal[cal.id.isin(ids)]
    val = dfs.get("val")
    if val is not None:
        e = val[val.label == 0].id.unique()[:5]
        o = val[val.label == 1].id.unique()[:5]
        smap["cal_unsup"] = val[val.id.isin(list(e) + list(o))]
    sets = seqs_for(smap)
    Xtr = torch.tensor(np.stack(sets["train"][0]))
    ytr = torch.tensor(np.array(sets["train"][1], np.float32))
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3,
                           weight_decay=1e-4)
    pos_w = torch.tensor(
        [float((1 - ytr.mean()) / (ytr.mean() + 1e-9))]).clamp(.5, 4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    n = len(Xtr)
    for ep in range(25):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            lossf(net(Xtr[b]), ytr[b]).backward()
            opt.step()
    res = {}
    for s in sets:
        if not len(sets[s][0]):
            continue
        Xs = torch.tensor(np.stack(sets[s][0]))
        ys = np.array(sets[s][1])
        with torch.no_grad():
            pr = torch.sigmoid(net(Xs)).numpy()
        res[s] = metrics(ys, (pr > .5).astype(int), pr)
        res[s]["proba"] = pr
    return res


# ---------------------------------------------------------------------------
def eval_all(clf, sc, dfcols, sets, name):
    """evaluate fitted clf on all sets -> {set: metrics(+proba)}.

    '<set>_norm' = unsupervised-calibration variant: the eval set's
    features are re-standardised with their OWN (unlabelled) stats
    before prediction — counters cross-session amplitude drift."""
    out = {}
    for s, (X, y) in sets.items():
        if len(y) == 0:
            continue
        p, pr = predict(clf, sc, X)
        m = metrics(y, p, pr)
        m["proba"] = pr
        out[s] = m
        if s != "train" and len(y) >= 30:
            sc_e = StandardScaler().fit(X)
            # model input is sc-transformed train space; emulate
            # 'recalibrate scaler on unlabelled eval data' by
            # predicting on eval-set-scaled data through sc.inverse
            Xn = sc.inverse_transform(sc_e.transform(X))
            Xn = np.where(np.isfinite(Xn), Xn, 0)
            p2, pr2 = predict(clf, sc, Xn)
            m2 = metrics(y, p2, pr2)
            m2["proba"] = pr2
            out[s + "_norm"] = m2
    return out


def calib_sets(dfs, cols):
    """supervised: first 10 min of calib; unsupervised: 5+5 from val."""
    sets = {}
    cal = dfs.get("calib")
    if cal is not None:
        ids = list(dict.fromkeys(cal.id))[:10]
        d = cal[cal.id.isin(ids)]
        sets["cal_sup"] = xy(d, cols)
    val = dfs.get("val")
    if val is not None:
        e = val[val.label == 0].id.unique()[:5]
        o = val[val.label == 1].id.unique()[:5]
        d = val[val.id.isin(list(e) + list(o))]
        sets["cal_unsup"] = xy(d, cols)
    return sets


def main():
    t0 = time.time()
    dfs = load()
    V = views(dfs)
    # 'sel': descriptor-fusion view = top-24 of 'all' by train |auc|
    Xa, ya = xy(dfs["train"], V["all"])
    aucs = {}
    for i, c in enumerate(V["all"]):
        v = Xa[:, i]
        m = np.isfinite(v)
        if m.sum() > 50 and np.nanstd(v[m]) > 1e-9:
            a = roc_auc_score(ya[m], v[m])
            aucs[c] = max(a, 1 - a)
    V["sel"] = sorted(aucs, key=aucs.get, reverse=True)[:24]
    print({k: len(v) for k, v in V.items()}, flush=True)
    print("sel cols:", V["sel"][:12], flush=True)

    # per-set matrices per view
    sets = {v: {} for v in V}
    for v, cols in V.items():
        for s, df in dfs.items():
            sets[v][s] = xy(df, cols)
        sets[v].update(calib_sets(dfs, cols))

    results, probs = {}, {}

    # ---------------- workflow A: descriptor fusion -------------------
    for v, cols in V.items():
        Xtr, ytr = sets[v]["train"]
        for mn, clf in {**ml_zoo(), **rule_zoo()}.items():
            try:
                sc = fit(clf, Xtr, ytr)
            except Exception as e:
                print(f"  {mn}/{v} fit fail: {e}", flush=True)
                continue
            r = eval_all(clf, sc, cols, sets[v], mn)
            results[f"{mn}/{v}"] = {
                k: {m: val for m, val in vv.items() if m != "proba"}
                for k, vv in r.items()}
            probs[f"{mn}/{v}"] = {k: vv["proba"] for k, vv in r.items()}
            print(f"{mn}/{v}: val={r.get('val', {}).get('bal', 0):.3f} "
                  f"test={r.get('testml', {}).get('bal', 0):.3f}",
                  flush=True)

    # centroid rule w/ calib-derived occupied centroid (view='all')
    cols = V["all"]
    Xtr, ytr = sets["all"]["train"]
    if "cal_sup" in sets["all"]:
        Xcs, _ = sets["all"]["cal_sup"]
        sc_c = StandardScaler().fit(Xtr)
        Xtr_s = sc_c.transform(Xtr)
        cr = CentroidRule(
            occ_centroid=sc_c.transform(Xcs).mean(0)).fit(Xtr_s, ytr)
        for s, (X, y) in sets["all"].items():
            Xs = sc_c.transform(X)
            p, pr = cr.predict(Xs), cr.predict_proba(Xs)[:, 1]
            m = metrics(y, p, pr)
            results.setdefault("centroid_cal/all", {})[s] = {
                k2: v2 for k2, v2 in m.items()}
            probs.setdefault("centroid_cal/all", {})[s] = pr

        # calibrated-threshold rule on the same scaled space
        tc = ThreshCal(k=8).fit(Xtr_s, ytr)
        tc.calibrate(sc_c.transform(Xcs))
        for s, (X, y) in sets["all"].items():
            Xs = sc_c.transform(X)
            p, pr = tc.predict(Xs), tc.predict_proba(Xs)[:, 1]
            m = metrics(y, p, pr)
            results.setdefault("threshcal/all", {})[s] = {
                k2: v2 for k2, v2 in m.items()}
            probs.setdefault("threshcal/all", {})[s] = pr

    # cluster-vote rule on joint view (km labels stored per row)
    X_j, y_j = sets["joint"]["train"]
    tr_km = dfs["train"].clu_kmlabel.values
    cv = ClusterVote().fit(X_j, y_j, tr_km)
    cal = dfs.get("calib")
    km_eval_sets = {s: (dfs[s].clu_kmlabel.values,
                        dfs[s].label.values.astype(int))
                    for s in ("train", "val", "testml") if s in dfs}
    if cal is not None:
        ids = list(dict.fromkeys(cal.id))[:10]
        d = cal[cal.id.isin(ids)]
        km_eval_sets["cal_sup"] = (d.clu_kmlabel.values,
                                   d.label.values.astype(int))
    val = dfs.get("val")
    if val is not None:
        e = val[val.label == 0].id.unique()[:5]
        o = val[val.label == 1].id.unique()[:5]
        d = val[val.id.isin(list(e) + list(o))]
        km_eval_sets["cal_unsup"] = (d.clu_kmlabel.values,
                                     d.label.values.astype(int))
    for s, (kml, y) in km_eval_sets.items():
        p = cv.predict(None, kml)
        results.setdefault("clustervote/joint", {})[s] = metrics(
            y, p, p.astype(float))

    # ---------------- workflow B: shared embedding --------------------
    emb_rows = {}
    for v in ("all",):
        Xtr, ytr = sets[v]["train"]
        sc_e = StandardScaler().fit(Xtr)
        pca = PCA(n_components=min(12, Xtr.shape[1]),
                  random_state=0).fit(sc_e.transform(Xtr))
        for mn, clf in ml_zoo().items():
            try:
                clf.fit(pca.transform(sc_e.transform(Xtr)), ytr)
            except Exception as e:
                print(f"  emb {mn} fail {e}", flush=True)
                continue
            for s, (X, y) in sets[v].items():
                if len(y) == 0:
                    continue
                Xe = pca.transform(sc_e.transform(X))
                p = clf.predict(Xe)
                pr = (clf.predict_proba(Xe)[:, 1]
                      if hasattr(clf, "predict_proba")
                      else p.astype(float))
                m = metrics(y, p, pr)
                results.setdefault(f"emb_{mn}/{v}", {})[s] = {
                    k2: v2 for k2, v2 in m.items() if k2 != "proba"}
                probs.setdefault(f"emb_{mn}/{v}", {})[s] = pr
        emb_rows["ev"] = pca.explained_variance_ratio_.tolist()

    # ---------------- dl: seqcnn on PC-dot sequences ------------------
    try:
        sq = seqcnn_eval(dfs)
        for s, r in sq.items():
            results.setdefault("seqcnn/jointseq", {})[s] = {
                k: v for k, v in r.items() if k != "proba"}
            probs.setdefault("seqcnn/jointseq", {})[s] = r["proba"]
        print(f"seqcnn: val={sq.get('val', {}).get('bal', 0):.3f} "
              f"test={sq.get('testml', {}).get('bal', 0):.3f}",
              flush=True)
    except Exception as e:
        print(f"seqcnn failed: {e}", flush=True)

    # ---------------- workflow C: probability fusion ------------------
    base = {}
    for mn in ("rf", "gbm", "mlp"):
        for v in ("radar", "csi", "joint"):
            k = f"{mn}/{v}"
            if k in probs and "testml" in probs[k] \
                    and "val" in probs[k]:
                base[k] = probs[k]
    # pair: same model across views
    for mn in ("rf", "gbm", "mlp"):
        keys = [f"{mn}/{v}" for v in ("radar", "csi", "joint")
                if f"{mn}/{v}" in base]
        if len(keys) < 2:
            continue
        eval_sets = [s for s in ("val", "val_norm", "testml",
                                 "testml_norm", "cal_sup", "cal_unsup")
                     if all(s in base[k] for k in keys)]
        for s in eval_sets:
            base_s = s.replace("_norm", "")
            if base_s not in sets["all"]:
                continue
            P = np.stack([base[k][s] for k in keys])
            y = sets["all"][base_s][1]
            for fuse in ("mean", "max"):
                pr = getattr(np, fuse)(P, axis=0)
                results.setdefault(f"fuse{fuse}_{mn}/prob", {})[s] = {
                    k2: v2 for k2, v2 in
                    metrics(y, (pr > .5).astype(int), pr).items()}
        # weighted: weights = val bal-acc of each view
        if all("val" in base[k] for k in keys):
            w = np.array([results[k]["val"]["bal"] for k in keys])
            w = np.clip(w, .01, None) ** 2
            w /= w.sum()
            for s in eval_sets:
                base_s = s.replace("_norm", "")
                if base_s not in sets["all"]:
                    continue
                P = np.stack([base[k][s] for k in keys])
                pr = (P * w[:, None]).sum(0)
                y = sets["all"][base_s][1]
                results.setdefault(f"fusw_{mn}/prob", {})[s] = {
                    k2: v2 for k2, v2 in
                    metrics(y, (pr > .5).astype(int), pr).items()}

    # ---------------- jev (skipped without key) -----------------------
    if os.environ.get("JEV_API_KEY"):
        try:
            import jev
            tr = dfs["train"]
            cols = jev.pick_cols(tr)
            for s in ("val", "testml", "cal_sup", "cal_unsup"):
                if s in ("cal_sup", "cal_unsup"):
                    _, y = sets["all"][s]
                    df = (dfs["calib"]
                          [dfs.calib.id.isin(
                              list(dict.fromkeys(
                                  dfs.calib.id))[:10])]
                          if s == "cal_sup" else None)
                    if s == "cal_unsup":
                        val = dfs["val"]
                        e = val[val.label == 0].id.unique()[:5]
                        o = val[val.label == 1].id.unique()[:5]
                        df = val[val.id.isin(list(e) + list(o))]
                else:
                    df = dfs[s]
                if df is None or not len(df):
                    continue
                df = df.sample(min(300, len(df)), random_state=0)
                pr = jev.jev_predict(
                    (row for _, row in df.iterrows()), tr, cols=cols)
                m = metrics(df.label.values.astype(int),
                            (pr > .5).astype(int), pr)
                results.setdefault("jev", {})[s] = {
                    k2: v2 for k2, v2 in m.items()}
            results["jev"]["note"] = (
                f"model={os.environ.get('JEV_MODEL', 'gpt-4o-mini')} "
                f"cols={len(cols)}")
        except Exception as e:
            results["jev"] = {"skipped": str(e)}
    else:
        results["jev"] = {"skipped": "JEV_API_KEY not set"}

    # ---------------- persist ------------------------------------------
    flat = []
    for k, sets_r in results.items():
        for s, m in sets_r.items():
            if isinstance(m, dict) and "bal" in m:
                flat.append({"model": k, "set": s,
                             **{kk: vv for kk, vv in m.items()
                                if kk != "proba"}})
    pd.DataFrame(flat).to_csv(OUT / "results.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(
        {k: {s: m for s, m in v.items()} for k, v in results.items()},
        indent=1, default=str))
    (OUT / "probs.pkl").write_bytes(pickle.dumps(probs))
    print(f"done {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
