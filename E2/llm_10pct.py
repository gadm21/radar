"""Three independent LLM evaluations on a fixed 10% benchmark holdout.

Uses the same 5 s feature table and held-out 160-minute test cohort as the
RF MoE comparison. Calibration examples and class statistics come only from
the train-fit / validation minutes. Requires JEV_API_KEY for live calls.
"""
from __future__ import annotations

import json
import os
import sys
import argparse
import getpass
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
                     "axes.edgecolor":"#4C5963", "axes.linewidth":.7,
                     "figure.facecolor":"white", "savefig.facecolor":"white"})
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from jev import _post, _parse
from physical_comparison import cal

ROOT = Path(__file__).resolve().parent
OCC = ROOT / "outputs/occupancy"
OUT = ROOT / "outputs/revision"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figs"
FIG.mkdir(exist_ok=True)
DESC = ["ra_std90", "re_std90", "xy_std90", "ra_delta", "re_delta",
        "csi_amp_mean", "snr_mean", "rad_pcv1"]
SYSTEM = ("You classify room occupancy from radar and WiFi CSI descriptors. "
          "The provided labeled calibration examples and class statistics "
          "are the only labeled reference. The MoE probability is a machine "
          "model suggestion, not ground truth. Return JSON only with "
          "occupied_probability (number from 0 to 1) and brief_reason. "
          "Use the same rule for every recording. Do not assume a balanced "
          "class distribution.")


def make_inputs():
    f = pd.read_csv(OCC / "features_5s.csv")
    tr = f[(f.dataset == "train") & (~f.is_val.astype(bool)) &
           (f.aug.astype(str).isin(["False", "0", "none", "nan"]))]
    if tr.empty: tr = f[(f.dataset == "train") & (~f.is_val.astype(bool))]
    va = f[(f.dataset == "train") & f.is_val.astype(bool)]
    test = f[f.dataset == "test"]
    rng = np.random.default_rng(42)
    tmins = test.groupby("rec_id").label.first()
    keep = []
    size = tmins.value_counts().min()
    for _, g in tmins.groupby(tmins):
        keep.extend(rng.choice(g.index, size=size, replace=False))
    cohort = test[test.rec_id.isin(keep)].groupby("rec_id", sort=True).agg(
        {"label": "first", **{c: "mean" for c in DESC}})
    if len(cohort) != 160:
        raise ValueError(f"Expected 160 test minutes, got {len(cohort)}")
    # Eight minutes per class = 16 of 160 = exactly 10%.
    picks = []
    for _, g in cohort.groupby("label"):
        picks.extend(rng.choice(g.index, size=8, replace=False))
    selected = cohort.loc[sorted(picks)].copy()
    val_minutes = va.groupby("rec_id", sort=True).agg(
        {"label": "first", **{c: "mean" for c in DESC}})
    cal = pd.concat([g.head(4) for _, g in val_minutes.groupby("label")])
    stats = {}
    for lab, g in tr.groupby("label"):
        stats[("empty", "occupied")[int(lab)]] = {
            c: {"mean": round(float(g[c].mean()), 3),
                "std": round(float(g[c].std()), 3)} for c in DESC}
    examples = [{"label": ("empty", "occupied")[int(r.label)],
                 "descriptors": {c: round(float(getattr(r, c)), 3) for c in DESC}}
                for r in cal.itertuples()]
    z = np.load(OCC / "test_probs.npz")
    def saved(model, feature, seed=42):
        key = f"5|{model}|{feature}|{seed}"
        if key not in z: return None
        y, p = z[key]
        if not np.array_equal(y.astype(int), cohort.label.to_numpy().astype(int)):
            raise ValueError(f"Saved {key} labels do not match test IDs")
        return pd.Series(p, index=cohort.index).loc[selected.index].to_numpy()
    moe_seeds = [int(k.rsplit("|", 1)[1]) for k in z.files
                 if k.startswith("5|rf|moe|")]
    if not moe_seeds:
        raise ValueError("No saved 5 s RF MoE probabilities")
    moe_full = np.stack([z[f"5|rf|moe|{s}"][1]
                         for s in sorted(moe_seeds)]).mean(0)
    moe = pd.Series(moe_full, index=cohort.index).loc[selected.index].to_numpy()
    selected["moe_probability"] = moe
    context = {"training_class_statistics": stats,
               "labeled_validation_calibration_examples": examples,
               "sample_ids": selected.index.tolist(), "n_test_cohort": len(cohort),
               "n_sample": len(selected), "descriptors": DESC,
               "moe_saved_seeds": sorted(moe_seeds)}
    (OUT / "llm_10pct_context.json").write_text(json.dumps(context, indent=2))
    selected.to_csv(OUT / "llm_10pct_sample.csv")
    comparators = {"RF MoE": (moe >= np.median(moe)).astype(int)}
    for title, model, feature in (("RF CSI", "rf", "csi"),
                                  ("Radar LSTM", "radar-lstm", "dl")):
        p = saved(model, feature)
        if p is not None:
            comparators[title] = (p >= np.median(p)).astype(int)
    return selected, context, comparators


def prompt_for(row, context):
    values = {c: round(float(row[c]), 3) for c in DESC}
    return json.dumps({
        "task": "Estimate P(occupied) for this held-out minute.",
        "training_class_statistics": context["training_class_statistics"],
        "labeled_calibration_examples": context[
            "labeled_validation_calibration_examples"],
        "minute_descriptors": values,
        "moe_probability": round(float(row.moe_probability), 4)},
        separators=(",", ":"))


def live_predict(prompt, run, model, key):
    cfg = {"base": os.environ.get("JEV_BASE_URL", "https://api.openai.com/v1"),
           "key": key, "model": model}
    payload = {"model": cfg["model"],
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}]}
    if model == "gpt-4o-mini":
        payload.update({"temperature": .2, "seed": 1300 + run})
    else:
        payload["max_completion_tokens"] = 1024
    data = _post(cfg["base"] + "/chat/completions",
                 {"Authorization": "Bearer " + cfg["key"],
                  "Content-Type": "application/json"},
                 payload)
    txt = data["choices"][0]["message"]["content"]
    obj = json.loads(txt)
    return float(np.clip(obj["occupied_probability"], 0, 1))


def _cm(ax, y, pred, title):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    ax.imshow(cm, cmap="YlGnBu", vmin=0, vmax=max(1, cm.max()))
    for i in (0,1):
        for j in (0,1): ax.text(j, i, cm[i,j], ha="center", va="center")
    ax.set_xticks([0,1], ["Empty", "Occupied"], fontsize=8)
    ax.set_yticks([0,1], ["Empty", "Occupied"], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Predicted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-5.4"])
    parser.add_argument("--key-stdin", action="store_true")
    args = parser.parse_args()
    key = (getpass.getpass("API key: ") if sys.stdin.isatty() else sys.stdin.readline().strip()) if args.key_stdin else os.environ.get("JEV_API_KEY", "")
    selected, context, comparators = make_inputs()
    prompts = [prompt_for(r, context) for _, r in selected.iterrows()]
    (OUT / "llm_10pct_prompt_preview.json").write_text(prompts[0])
    y = selected.label.to_numpy().astype(int)
    baseline = {name: {"accuracy": float(accuracy_score(y, pred)),
                       "confusion_matrix": confusion_matrix(y, pred,
                           labels=[0,1]).tolist()}
                for name, pred in comparators.items()}
    (OUT / "llm_10pct_baselines.json").write_text(json.dumps(baseline, indent=2))
    fig, axes = plt.subplots(1, len(comparators), figsize=(3*len(comparators), 3.2))
    for ax, (name, pred) in zip(np.ravel(axes), comparators.items()):
        _cm(ax, y, pred, name)
    fig.suptitle("Same 16 held-out minutes; median calibrated baselines")
    fig.tight_layout(); fig.savefig(FIG / "llm_sample_ml_dl_cm.png", dpi=190)
    plt.close(fig)
    if not key and not all((OUT / f"llm_10pct_predictions_{m}.csv").exists() for m in args.models):
        print("JEV_API_KEY absent: 16-minute sample and prompt prepared; no LLM scores claimed")
        return
    summaries = {}
    llm_first = {}
    for model in args.models:
        records = []
        path = OUT / f"llm_10pct_predictions_{model}.csv"
        if path.exists():
            records = pd.read_csv(path).to_dict("records")
        done = {(int(r["run"]), str(r["rec_id"])) for r in records}
        for run in range(3):
            for rec_id, prompt in zip(selected.index, prompts):
                if (run+1, str(rec_id)) in done:
                    continue
                if not key:
                    raise RuntimeError(f"Missing saved prediction for {model}, run {run+1}, {rec_id}")
                p = live_predict(prompt, run, model, key)
                records.append({"run": run+1, "rec_id": rec_id,
                                "label": int(selected.loc[rec_id, "label"]),
                                "probability": p, "prediction": int(p >= .5)})
                pd.DataFrame(records).to_csv(path, index=False)
                print(f"{model}: {len(records)}/48 predictions", flush=True)
                time.sleep(.05)
        df = pd.DataFrame(records)
        by_run = [g.set_index("rec_id").loc[selected.index] for _,g in df.groupby("run")]
        scores = [accuracy_score(g.label, g.prediction) for g in by_run]
        med_scores = [cal(g.label.to_numpy().astype(int),g.probability.to_numpy())["acc_med"] for g in by_run]
        km_scores = [cal(g.label.to_numpy().astype(int),g.probability.to_numpy())["acc_km"] for g in by_run]
        summary = {"model": model, "n_minutes": len(selected), "runs": 3,
                   "run_accuracy": scores, "mean_accuracy": float(np.mean(scores)),
                   "std_accuracy": float(np.std(scores, ddof=1)),
                   "median_calibrated_run_accuracy": med_scores,
                   "median_calibrated_mean_accuracy": float(np.mean(med_scores)),
                   "median_calibrated_std_accuracy": float(np.std(med_scores,ddof=1)),
                   "two_means_run_accuracy": km_scores,
                   "two_means_mean_accuracy": float(np.mean(km_scores)),
                   "two_means_std_accuracy": float(np.std(km_scores,ddof=1)),
                   "sample_seed": 42}
        summaries[model] = summary
        first = by_run[0]
        llm_first[model] = (first.probability.to_numpy() >= np.median(first.probability.to_numpy())).astype(int)
        (OUT / "llm_10pct_results.json").write_text(json.dumps(summaries, indent=2))
    panels = {**{f"{m} run 1": p for m,p in llm_first.items()}, **comparators}
    fig, axes = plt.subplots(1, len(panels), figsize=(3*len(panels), 3.2))
    for ax, (name, pred) in zip(np.ravel(axes), panels.items()):
        _cm(ax, y, pred, name)
    fig.suptitle("Same 16 held-out minutes; median-calibrated probabilities")
    fig.tight_layout(); fig.savefig(FIG / "llm_vs_ml_dl_cm.png", dpi=190)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 8.0))
    for ax, (name, pred) in zip(axes.ravel(), panels.items()):
        _cm(ax, y, pred, name)
    for ax in axes.ravel()[len(panels):]: ax.axis("off")
    fig.suptitle("Fixed 10% benchmark sample (n=16); median calibration")
    fig.tight_layout(); fig.savefig(FIG / "llm_vs_ml_dl_cm_paper.png", dpi=360,
                                   bbox_inches="tight")
    fig.savefig(FIG / "llm_vs_ml_dl_cm_paper.pdf", dpi=360,
                bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
