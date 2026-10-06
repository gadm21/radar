"""E2 — multilink evaluation: ML/DL models, train/val -> multilink test.

Protocols
---------
within_old : train on legacy train_minutes, test on legacy test_minutes
             (in-domain reference, same room/session stats as E1).
cross      : train on legacy train_minutes, validate on legacy
             test_minutes, TEST on the 4 multilink captures
             (cross-domain generalisation: 1-link CSI model applied to
             the 2-link mean; radar identical hardware).
loco_ml    : leave-one-capture-out on the 4 multilink captures
             (in-domain, honest cross-session estimate).

Feature space (shared across datasets): radar view descriptors +
physical RD descriptors + rad_pcv + snr, and CSI amplitude/rolling-
variance/tv/dop_frac/pcv — multilink side uses the 2-link mean
(csiM_* renamed csi_*).

Models: sklearn zoo (logreg, svm-rbf, rf, histgb, mlp) + a small
torch Conv1d trained on per-second compressed-PCA dot sequences
(6ch: csi pc1-3 + rad pc1-3; W=5s).

Outputs: E2/outputs/multilink_eval/{results.json,REPORT.md,figs/,models/}
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent          # radar/
E1_OUT = ROOT / "E1" / "outputs"
OUT = ROOT / "E2" / "outputs" / "multilink_eval"
FIGS = OUT / "figs"
MODELS = OUT / "models"

RADAR_FEATS = [
    "snr_mean", "snr_max", "rad_pcv1", "rad_pcv2", "rad_pcv3",
    "rd_mean", "rd_p90", "rd_std90", "rd_delta", "rd_peak",
    "ra_mean", "ra_p90", "ra_std90", "ra_delta", "ra_peak",
    "re_mean", "re_p90", "re_std90", "re_delta", "re_peak",
    "xy_mean", "xy_p90", "xy_std90", "xy_delta", "xy_peak",
    "rd_range_cm", "rd_dop_cm", "rd_dop_spread", "rd_dca", "rd_energy",
]
CSI_FEATS = [
    "csi_amp_mean", "csi_amp_std", "csi_amp_q90",
    "csi_rv_mean", "csi_rv_q90", "csi_rv_max",
    "csi_tv", "csi_dop_frac",
    "csi_pcv1", "csi_pcv2", "csi_pcv3",
]
SETS = {"radar": RADAR_FEATS, "csi": CSI_FEATS,
        "fusion": RADAR_FEATS + CSI_FEATS}

W = 5                                                  # window seconds


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_tables():
    ml = pd.read_csv(E1_OUT / "features_multilink_5s.csv")
    for s in ("amp_mean", "amp_std", "amp_q90", "rv_mean", "rv_q90",
              "rv_max", "tv", "dop_frac", "pcv1", "pcv2", "pcv3"):
        ml[f"csi_{s}"] = ml[[f"csi1_{s}", f"csi2_{s}"]].mean(axis=1)
    ml["grp"] = ml.cap
    old = pd.read_csv(E1_OUT / "features_old_5s_ext.csv")
    old["grp"] = old.rec_id
    return ml, old


def xy(df, feats):
    X = df[feats].astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median()), df.label.values.astype(int), df.grp.values


def zoo():
    return {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0)),
        "svm": make_pipeline(StandardScaler(),
                             SVC(C=4, gamma="scale", probability=True)),
        "rf": RandomForestClassifier(400, min_samples_leaf=2,
                                     n_jobs=-1, random_state=0),
        "gbm": HistGradientBoostingClassifier(max_iter=400,
                                              learning_rate=.06,
                                              random_state=0),
        "mlp": make_pipeline(StandardScaler(),
                             MLPClassifier((96, 48), max_iter=500,
                                           early_stopping=True,
                                           random_state=0)),
    }


def metrics(y, p, proba):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"acc": accuracy_score(y, p),
            "bal_acc": balanced_accuracy_score(y, p),
            "f1": f1_score(y, p),
            "auc": (roc_auc_score(y, proba)
                    if len(np.unique(y)) > 1 else float("nan")),
            "recall_empty": (tn / (tn + fp)) if (tn + fp) else np.nan,
            "recall_occ": (tp / (tp + fn)) if (tp + fn) else np.nan}


def best_thr(y, proba):
    ths = np.linspace(.05, .95, 91)
    return float(ths[np.argmax([balanced_accuracy_score(y, proba >= t)
                                for t in ths])])


# ---------------------------------------------------------------------------
# DL: Conv1d over per-second PC-dot sequences
# ---------------------------------------------------------------------------
def build_seqs_6ch(npz_path):
    """[csi pc1-3 | rad pc1-3] channels, per (src,sec); csi rows may hold
    several links (multilink) -> mean over links naturally handled by
    averaging all csi rows at that second."""
    z = np.load(npz_path)
    per = {}
    for name, off in (("csi", 0), ("rad", 3)):
        for lab, sec, src, p1, p2, p3 in z[name]:
            key = (int(src), int(sec), int(lab))
            per.setdefault(key, []).append([p1, p2, p3, off])
    seqs = []
    by_src = {}
    for (src, sec, lab), rows in per.items():
        vec = np.zeros(6, np.float32)
        cnt = np.zeros(2, np.float32)
        for p1, p2, p3, off in rows:
            vec[off:off + 3] += [p1, p2, p3]
            cnt[off // 3] += 1
        vec[:3] /= max(cnt[0], 1)
        vec[3:] /= max(cnt[1], 1)
        by_src.setdefault(src, []).append((sec, lab, vec))
    for src, rows in by_src.items():
        rows.sort()
        secs = np.array([r[0] for r in rows])
        labs = np.array([r[1] for r in rows])
        for i in range(len(rows) - W + 1):
            if secs[i + W - 1] - secs[i] != W - 1:
                continue
            X = np.stack([rows[i + j][2] for j in range(W)], axis=0).T
            seqs.append((X.astype(np.float32), int(labs[i]), src))
    return seqs


def train_seqcnn(tr, va, seed=0, epochs=25):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.f = nn.Sequential(
                nn.Conv1d(6, 32, 3, padding=1), nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64),
                nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(.2),
                nn.Linear(32, 1))

        def forward(s, x):
            return s.f(x).squeeze(-1)

    def batches(seqs, bs=256):
        idx = np.random.permutation(len(seqs))
        for i in range(0, len(seqs), bs):
            b = [seqs[j] for j in idx[i:i + bs]]
            yield (torch.from_numpy(np.stack([s[0] for s in b])).to(dev),
                   torch.tensor([s[1] for s in b],
                                dtype=torch.float32).to(dev))

    mu = np.stack([s[0] for s in tr]).mean(axis=(0, 2))
    sd = np.stack([s[0] for s in tr]).std(axis=(0, 2)) + 1e-6
    tr = [( (X - mu[:, None]) / sd[:, None], y, s) for X, y, s in tr]
    va = [( (X - mu[:, None]) / sd[:, None], y, s) for X, y, s in va]

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best, wait = 0.0, 0
    for ep in range(epochs):
        net.train()
        for Xb, yb in batches(tr):
            opt.zero_grad()
            lossf(net(Xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(torch.from_numpy(
                np.stack([s[0] for s in va])).to(dev))).cpu().numpy()
        a = balanced_accuracy_score([s[1] for s in va], pv >= .5)
        if a > best:
            best, wait, state = a, 0, {k: v.clone()
                                      for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait > 6:
                break
    net.load_state_dict(state)
    return net, mu, sd


def seqcnn_predict(net, mu, sd, seqs):
    import torch
    dev = next(net.parameters()).device
    Xs = np.stack([(X - mu[:, None]) / sd[:, None] for X, _, _ in seqs])
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(Xs).to(dev))).cpu().numpy()
    return p, np.array([s[1] for s in seqs])


# ---------------------------------------------------------------------------
# protocols
# ---------------------------------------------------------------------------
def run_sklearn(dtrain, dtest, dval=None):
    """Feature-table model zoo. Returns {model/set: metrics}."""
    out = {}
    for setname, feats in SETS.items():
        Xtr, ytr, _ = xy(dtrain, feats)
        Xte, yte, _ = xy(dtest, feats)
        Xva = yva = None
        if dval is not None:
            Xva, yva, _ = xy(dval, feats)
        for mname, model in zoo().items():
            model.fit(Xtr, ytr)
            ptr = model.predict_proba(Xte)[:, 1] \
                if hasattr(model, "predict_proba") \
                else model.decision_function(Xte)
            thr = .5
            if Xva is not None:
                pva = model.predict_proba(Xva)[:, 1] \
                    if hasattr(model, "predict_proba") \
                    else model.decision_function(Xva)
                thr = best_thr(yva, pva)
            m = metrics(yte, ptr >= thr, ptr)
            m["thr"] = thr
            out[f"{mname}/{setname}"] = m
    return out


def run_seqcnn(tr_seqs, te_seqs, va_seqs, tag):
    import torch
    va = va_seqs if va_seqs else tr_seqs
    net, mu, sd = train_seqcnn(tr_seqs, va)
    p, y = seqcnn_predict(net, mu, sd, te_seqs)
    thr = .5
    if va_seqs:
        pv, yv = seqcnn_predict(net, mu, sd, va_seqs)
        thr = best_thr(yv, pv)
    out = metrics(y, p >= thr, p) | {"thr": thr}
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd},
               MODELS / f"seqcnn_{tag}.pt")
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    t0 = time.time()

    ml, old = load_tables()
    old_tr = old[old.dataset == "train"].copy()
    old_te = old[old.dataset == "test"].copy()
    print(f"old train {len(old_tr)}  old test {len(old_te)}  "
          f"multilink {len(ml)}", flush=True)

    # sequences for DL; pc_dots_old src index follows
    # sorted(E1/cache/win rglob) -> recover dataset per src
    seqs_old = build_seqs_6ch(E1_OUT / "pc_dots_old.npz")
    seqs_ml = build_seqs_6ch(E1_OUT / "pc_dots_multilink.npz")
    files = sorted((ROOT / "E1" / "cache" / "win").rglob("*.npz"))
    src_ds = {i: f.parent.name for i, f in enumerate(files)}
    seqs_old_tr = [s for s in seqs_old
                   if src_ds.get(s[2]) == "train"]
    seqs_old_te = [s for s in seqs_old
                   if src_ds.get(s[2]) == "test_minutes"]
    print(f"seqs: old_tr {len(seqs_old_tr)}  old_te {len(seqs_old_te)} "
          f" ml {len(seqs_ml)}", flush=True)
    # grouped val split for the CNN on train minutes (last 20% srcs)
    srcs = sorted({s[2] for s in seqs_old_tr})
    va_srcs = set(srcs[int(.8 * len(srcs)):])
    seqs_fit = [s for s in seqs_old_tr if s[2] not in va_srcs]
    seqs_val = [s for s in seqs_old_tr if s[2] in va_srcs]

    results = {}
    # --- within legacy (reference)
    print("\n===== within_old =====", flush=True)
    results["within_old"] = run_sklearn(old_tr, old_te)
    results["within_old"]["seqcnn/fusion"] = run_seqcnn(
        seqs_fit, seqs_old_te, seqs_val, "within_old")
    # --- cross-domain: legacy -> multilink (val on legacy test)
    print("\n===== cross_legacy_to_multilink =====", flush=True)
    results["cross_legacy_to_multilink"] = run_sklearn(
        old_tr, ml, dval=old_te)
    results["cross_legacy_to_multilink"]["seqcnn/fusion"] = run_seqcnn(
        seqs_old_tr, seqs_ml, seqs_old_te, "cross")
    # --- multilink LOCO (in-domain, honest)
    cap_order = sorted(ml.cap.unique())
    loco = {}
    for cap in cap_order:
        tr_df = ml[ml.cap != cap]
        te_df = ml[ml.cap == cap]
        src_i = cap_order.index(cap)
        te_seq = [s for s in seqs_ml if s[2] == src_i]
        tr_seq = [s for s in seqs_ml if s[2] != src_i]
        out = run_sklearn(tr_df, te_df)
        out["seqcnn/fusion"] = run_seqcnn(tr_seq, te_seq, tr_seq,
                                          f"loco_{src_i}")
        loco[cap] = out
        print(f"  loco {cap}: "
              f"rf/fusion={out['rf/fusion']['bal_acc']:.3f}", flush=True)
    results["loco_ml"] = loco

    # aggregate
    results["_meta"] = {"window": W, "n_old_train": len(old_tr),
                        "n_old_test": len(old_te), "n_ml": len(ml),
                        "elapsed_s": time.time() - t0}
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    write_report(results)

    # ---- summary tables + figures ----
    def tbl(res, tag):
        rows = []
        for k, v in res.items():
            m, s = k.split("/")
            rows.append({"model": m, "set": s,
                         "acc": v["acc"], "bal": v["bal_acc"],
                         "auc": v["auc"], "rec_e": v["recall_empty"],
                         "rec_o": v["recall_occ"],
                         "thr": v.get("thr", .5)})
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"results_{tag}.csv", index=False)
        return df

    d1 = tbl(results["within_old"], "within_old")
    d2 = tbl(results["cross_legacy_to_multilink"], "cross")
    # loco aggregate: average metrics per model/set across caps
    agg = {}
    for cap, res in loco.items():
        for k, v in res.items():
            agg.setdefault(k, []).append(v)
    rows = []
    for k, vs in agg.items():
        m, s = k.split("/")
        rows.append({"model": m, "set": s,
                     "bal_mean": np.nanmean([v["bal_acc"] for v in vs]),
                     "bal_min": np.nanmin([v["bal_acc"] for v in vs]),
                     "rec_e": np.nanmean([v["recall_empty"] for v in vs]),
                     "rec_o": np.nanmean([v["recall_occ"] for v in vs])})
    d3 = pd.DataFrame(rows)
    d3.to_csv(OUT / "results_loco.csv", index=False)

    # bar figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (df, ttl, col) in zip(
            axes, ((d1, "within legacy (train→test)", "bal"),
                   (d2, "cross-domain (legacy→multilink)", "bal"),
                   (d3, "multilink LOCO", "bal_mean"))):
        df = df.sort_values(col)
        lbl = df.model + "/" + df.set
        ax.barh(lbl, df[col], color="#3a6ea5")
        ax.set_title(ttl); ax.set_xlim(0, 1.05)
        ax.tick_params(labelsize=7)
        ax.axvline(.5, c="r", ls="--", lw=.7)
    fig.suptitle("Balanced accuracy — multilink_eval (W=5s)")
    fig.tight_layout()
    fig.savefig(FIGS / "results_bars.png", dpi=130)
    plt.close(fig)

    print(f"\n[{time.time() - t0:.0f}s] -> {OUT}")
    for tag, df in (("within_old", d1), ("cross", d2)):
        print(f"\n== {tag} (top by bal_acc) ==")
        print(df.sort_values("bal", ascending=False).head(8)
              .to_string(index=False))
    print("\n== multilink LOCO (mean/min bal-acc) ==")
    print(d3.sort_values("bal_mean", ascending=False).head(8)
          .to_string(index=False))


def write_report(results):
    def top(res, n=6):
        rows = sorted(res.items(),
                      key=lambda kv: -kv[1]["bal_acc"])[:n]
        return [{"model/set": k, "acc": f"{v['acc']:.3f}",
                 "bal_acc": f"{v['bal_acc']:.3f}",
                 "auc": f"{v['auc']:.3f}",
                 "rec_e": f"{v['recall_empty']:.2f}",
                 "rec_o": f"{v['recall_occ']:.2f}",
                 "thr": f"{v.get('thr', .5):.2f}"}
                for k, v in rows]

    def md(rows):
        ks = list(rows[0])
        h = ("| " + " | ".join(ks) + " |\n"
             + "|" + "|".join(["---"] * len(ks)) + "|\n")
        b = "\n".join("| " + " | ".join(str(r[k]) for k in ks)
                      + " |" for r in rows)
        return h + b + "\n"

    L = ["# E2 multilink_eval — ML/DL results (W=5s)",
         "Train/validation on legacy train_minutes (+test_minutes as "
         "validation for thresholding), test on the 4 multilink "
         "captures (2-CSI-link + radar). Multilink CSI = 2-link mean.\n"]
    L.append("## within-legacy reference (train_minutes → test_minutes)")
    L.append(md(top(results["within_old"])))
    L.append("## cross-domain: legacy train → multilink test")
    L.append(md(top(results["cross_legacy_to_multilink"])))
    L.append("## multilink LOCO — mean/min balanced accuracy")
    agg = {}
    for cap, res in results["loco_ml"].items():
        for k, v in res.items():
            agg.setdefault(k, []).append(v)
    rows = []
    for k, vs in agg.items():
        rows.append({"model/set": k,
                     "bal_mean": f"{np.nanmean([v['bal_acc'] for v in vs]):.3f}",
                     "bal_min": f"{np.nanmin([v['bal_acc'] for v in vs]):.3f}",
                     "rec_e": f"{np.nanmean([v['recall_empty'] for v in vs]):.2f}",
                     "rec_o": f"{np.nanmean([v['recall_occ'] for v in vs]):.2f}"})
    rows.sort(key=lambda r: -float(r["bal_mean"]))
    L.append(md(rows[:10]))
    # per-cap detail
    L.append("\n## multilink LOCO — per-capture detail (fusion set)")
    det = []
    for cap, res in results["loco_ml"].items():
        row = {"cap": cap}
        for k in ("rf/fusion", "gbm/fusion", "mlp/fusion",
                  "seqcnn/fusion", "rf/radar", "logreg/radar"):
            if k in res:
                row[k] = f"{res[k]['bal_acc']:.2f}"
        det.append(row)
    if det:
        L.append(md(det))
    L.append("Note: single-class test captures leave the absent-class "
             "recall undefined; aggregates use nan-aware means. The "
             "empty-Oct-4 capture is the hard fold — most models score "
             "~0 on it (predicted occupied). Its `ra_mean`/`snr_mean` "
             "sit near empty-Oct-3, but `csi2_amp_mean` shifted "
             "22.7 vs 41.2 — a cross-session CSI amplitude drift the "
             "trained models don't tolerate (cf. E1 LOCO recall).\n")
    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
