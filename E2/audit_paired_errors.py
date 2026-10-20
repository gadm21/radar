"""Count paired RF/CSI versus RF/MoE minute errors on the fixed holdout."""
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent / "outputs/revision/reviewer_audit"


def main() -> None:
    for condition in ("with_augmentation", "no_augmentation"):
        d = pd.read_csv(OUT / f"{condition}_predictions.csv")
        med = d.groupby(["seed", "view"]).probability.transform("median")
        d["pred"] = (d.probability >= med).astype(int)
        labels = d.drop_duplicates(["seed", "rec_id"]).set_index(
            ["seed", "rec_id"]).label
        q = d.pivot(index=["seed", "rec_id"], columns="view", values="pred").join(labels)
        rows = []
        for seed, g in q.groupby(level="seed"):
            csi_ok = g.csi == g.label
            moe_ok = g.moe == g.label
            rows.append({"seed": seed,
                         "both_correct": int((csi_ok & moe_ok).sum()),
                         "csi_only_correct": int((csi_ok & ~moe_ok).sum()),
                         "moe_only_correct": int((~csi_ok & moe_ok).sum()),
                         "both_wrong": int((~csi_ok & ~moe_ok).sum())})
        result = pd.DataFrame(rows)
        result.to_csv(OUT / f"{condition}_rf_paired_errors.csv", index=False)
        print(condition, "\n", result.to_string(index=False))


if __name__ == "__main__":
    main()
