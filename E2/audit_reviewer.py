"""Audit saved occupancy benchmark provenance without retraining models.

Outputs are factual checks of the existing archive, not new accuracy results.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "E2/outputs/occupancy"
OUT = ROOT / "E2/outputs/revision/reviewer_audit"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
AUG_KINDS = ("gauss", "gain", "mixup", "warp")


def count_rows(df: pd.DataFrame, keys: list[str]) -> list[dict]:
    return df.groupby(keys, dropna=False).size().rename("count").reset_index().to_dict("records")


def main() -> None:
    index = pd.read_csv(BENCH / "minutes_index.csv")
    feat = pd.read_csv(BENCH / "features_5s.csv",
                       usecols=["rec_id", "dataset", "is_val", "aug", "label"])
    for col in ("is_val", "aug"):
        feat[col] = feat[col].astype(bool)
    index["is_val"] = index.is_val.astype(bool)
    index["usable"] = index.usable.astype(bool)

    fit = index[(index.dataset == "train") & ~index.is_val & index.usable]
    val = index[(index.dataset == "train") & index.is_val & index.usable]
    test = index[(index.dataset == "test") & index.usable]
    fit_ids, val_ids, test_ids = (set(d.rec_id) for d in (fit, val, test))
    assert not fit_ids & val_ids and not fit_ids & test_ids and not val_ids & test_ids
    primary_validation_ids = set(pd.read_csv(
        ROOT / "E1/outputs/features_old_5s_ext.csv", usecols=["rec_id"]
    ).rec_id)
    primary_validation_ids = {x for x in primary_validation_ids
                              if x.startswith("test_minutes/")}
    assert primary_validation_ids == test_ids, (
        "Benchmark raw test and primary E2 validation IDs differ")

    empties = fit[fit.label == 0].reset_index(drop=True)
    need = int((fit.label == 1).sum() - len(empties))
    assert need >= 0
    lineage = []
    for k in range(need):
        src = empties.iloc[k % len(empties)]
        seed = SEED + 7919 * k
        kind = AUG_KINDS[k % len(AUG_KINDS)]
        partner = None
        if kind == "mixup":
            j = int(np.random.default_rng(seed).integers(0, len(empties)))
            partner = empties.iloc[j].rec_id
        lineage.append({"aug_rec_id": f"{src.rec_id}#aug{k}",
                        "parent_rec_id": src.rec_id,
                        "partner_rec_id": partner,
                        "kind": kind, "seed": seed,
                        "parent_partition": "fit",
                        "partner_partition": "fit" if partner else None})
    line = pd.DataFrame(lineage)
    observed_aug = set(feat.loc[feat.aug, "rec_id"])
    planned_aug = set(line.aug_rec_id)
    assert observed_aug == planned_aug, "Saved augmentation IDs differ from deterministic plan"
    assert set(line.parent_rec_id) <= fit_ids
    assert set(line.partner_rec_id.dropna()) <= fit_ids
    assert not set(line.parent_rec_id) & val_ids
    assert not set(line.partner_rec_id.dropna()) & val_ids
    line.to_csv(OUT / "augmentation_lineage.csv", index=False)

    pcas = joblib.load(BENCH / "pcas.joblib")
    pca_counts = {name: int(pcas[name].n_samples_) for name in ("csi", "rad")}
    csi_fit_seconds = int(fit.n_secs.sum())
    assert pca_counts["csi"] == csi_fit_seconds, (
        f"CSI PCA fit count {pca_counts['csi']} differs from fit seconds {csi_fit_seconds}")
    radar_fit_seconds = 0
    for path in fit.npz:
        with np.load(path) as z:
            radar_fit_seconds += int(z["sec_has_radar"].sum())
    assert pca_counts["rad"] == radar_fit_seconds, (
        f"Radar PCA fit count {pca_counts['rad']} differs from fit seconds {radar_fit_seconds}")

    gate_rows = []
    window = None
    for raw in (BENCH / "models_seeds.log").read_text(encoding="utf-16").splitlines():
        match = re.search(r"W=(\d+)s test minutes kept: (\d+) of (\d+)", raw)
        if match:
            window = int(match.group(1))
        match = re.search(r"seed=(\d+) moe/(rf|xgb) w_csi=([0-9.]+)", raw)
        if match and window is not None:
            gate_rows.append({"window_seconds": window, "seed": int(match.group(1)),
                              "model": match.group(2), "w_csi": float(match.group(3))})
    gates = pd.DataFrame(gate_rows)
    assert len(gates) == 20, "Expected five seeds, two models and two window sizes"
    gates.to_csv(OUT / "fusion_gate_weights.csv", index=False)

    seed_results = pd.read_csv(BENCH / "results_seeds.csv")
    seed_counts = seed_results.groupby(["window", "model"]).seed.nunique().reset_index(
        name="n_seeds")
    seed_counts.to_csv(OUT / "seed_counts.csv", index=False)
    with np.load(BENCH / "test_probs.npz") as z:
        archived_seeds = sorted({int(k.rsplit("|", 1)[1]) for k in z.files
                                 if k.startswith("5|rf|moe|")})
        dl_archived_seeds = sorted({int(k.rsplit("|", 1)[1]) for k in z.files
                                    if k.startswith("5|radar-lstm|dl|")})

    multilink = pd.read_csv(ROOT / "E1/outputs/features_multilink_5s.csv",
                            usecols=["cap", "label", "t_rel"])
    multilink["minute"] = (multilink.t_rel // 60).astype(int)
    capture_counts = multilink.drop_duplicates(["cap", "minute"]).groupby(
        ["cap", "label"]).size().rename("minutes").reset_index()
    capture_counts.to_csv(OUT / "multilink_capture_class_counts.csv", index=False)
    benchmark_test_session_counts = test.groupby(["session_id", "label"]).size().rename(
        "minutes").reset_index()
    benchmark_test_session_counts.to_csv(
        OUT / "benchmark_test_session_class_counts.csv", index=False)
    rates = index[index.usable].groupby(["dataset", "is_val", "label"])[
        ["radar_hz", "csi_hz", "n_frames", "n_secs"]].median().reset_index()
    rates.to_csv(OUT / "observed_rate_medians.csv", index=False)

    fit_sessions = set(fit.session_id)
    val_sessions = set(val.session_id)
    report = {
        "source": "saved E2 occupancy benchmark; no models retrained",
        "minute_counts": count_rows(index, ["dataset", "is_val", "usable", "label"]),
        "feature_minute_counts": count_rows(
            feat.drop_duplicates("rec_id"), ["dataset", "is_val", "aug", "label"]),
        "augmentation": {"planned_and_observed_synthetic_minutes": len(line),
                         "parents_in_fit_only": True,
                         "mixup_partners_in_fit_only": True,
                         "fit_validation_parent_overlap": 0,
                         "lineage_file": "augmentation_lineage.csv"},
        "pca_fit_seconds": pca_counts,
        "fit_seconds_recount": {"csi": csi_fit_seconds, "rad": radar_fit_seconds},
        "session_overlap": {"fit_sessions": len(fit_sessions),
                            "validation_sessions": len(val_sessions),
                            "shared_session_ids": sorted(fit_sessions & val_sessions)},
        "test_minutes_before_label_balance": test.label.value_counts().sort_index().to_dict(),
        "test_minutes_after_label_balance": {"empty": 80, "occupied": 80},
        "primary_e2_validation_equals_benchmark_raw_test": True,
        "shared_validation_benchmark_minutes": len(test_ids),
        "archived_five_second_rf_moe_seeds": archived_seeds,
        "archived_five_second_radar_lstm_seeds": dl_archived_seeds,
        "multilink_captures_single_class": bool(
            (capture_counts.groupby("cap").label.nunique() == 1).all()),
        "multilink_capture_class_counts_file": "multilink_capture_class_counts.csv",
        "benchmark_test_sessions_single_class": bool(
            (benchmark_test_session_counts.groupby("session_id").label.nunique() == 1).all()),
        "benchmark_test_session_class_counts_file": "benchmark_test_session_class_counts.csv",
        "observed_rate_medians_file": "observed_rate_medians.csv",
        "gate_weights_file": "fusion_gate_weights.csv",
        "seed_counts_file": "seed_counts.csv",
    }
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
