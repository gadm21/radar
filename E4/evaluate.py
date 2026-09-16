"""End-to-end evaluation of `predict_capture` on test_minutes/.

Runs the full inference path (capture.npz -> combined label) on every
test_minutes recording that has a capture.npz, and reports accuracy for:
  * the combined label  (empty / sleep / present(l|r))
  * the 3-class activity head (empty / sleep / present)
  * the binary position head  (left / right) — only t_present recordings
    carry a left/right ground-truth label, so position is validated
    exclusively on those recordings

Note: the position model was trained on ALL t recordings (deployment
model), so its accuracy here is a sanity check, not a held-out estimate —
the honest held-out number is the E3 5-fold CV result (0.989 window /
1.000 minute). The activity model, by contrast, was never trained on t,
so its accuracy here IS a true held-out estimate.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from predict import predict_capture


def _combined_label(activity, position):
    if activity == 0 or position is None:
        return C.ACTIVITY_NAMES[activity]
    return f"{C.ACTIVITY_NAMES[activity]}({C.POSITION_NAMES[position]})"


def main():
    label_map = C.scan_all_labels()
    rows = []
    skipped = []
    for rec_id, info in sorted(label_map.items()):
        if info["source"] != "test_minutes" or info["activity"] is None:
            continue
        npz = Path(info["folder"]) / "capture.npz"
        if not npz.exists():
            skipped.append(rec_id)
            continue
        try:
            label, details = predict_capture(npz, return_probs=True)
        except Exception as e:
            skipped.append(f"{rec_id}: {e}")
            continue
        true_combined = _combined_label(info["activity"], info["position"])
        rows.append({
            "rec_id": rec_id,
            "true_activity": info["activity"],
            "true_position": info["position"],
            "true_combined": true_combined,
            "pred_combined": label,
            "pred_activity": details["activity"],
            "pred_position": details["position"],
            "occupancy_prob": details["occupancy_prob"],
            "activity_probs": details["activity_probs"],
            "position_probs": details["position_probs"],
        })
        print(f"{rec_id}: true={true_combined:16s} pred={label:16s}", flush=True)

    n = len(rows)
    act_correct = sum(1 for r in rows
                      if C.ACTIVITY_NAMES[r["true_activity"]] == r["pred_activity"])
    # combined label is correct iff activity matches AND (the recording
    # has no position label to check against OR the position matches)
    comb_correct = sum(
        1 for r in rows
        if C.ACTIVITY_NAMES[r["true_activity"]] == r["pred_activity"]
        and (r["true_position"] is None
             or C.POSITION_NAMES[r["true_position"]] == r["pred_position"]))

    # position can only be validated on t_present recordings — they are
    # the only ones with a left/right ground-truth label. Two numbers:
    #   * coverage: fraction where the pipeline emits a position at all
    #     (requires the activity head to predict "present")
    #   * conditional accuracy: of those, fraction with correct left/right
    pos_rows = [r for r in rows if r["true_position"] is not None]
    pos_emitted = [r for r in pos_rows if r["pred_position"] is not None]
    pos_correct = sum(1 for r in pos_emitted
                      if C.POSITION_NAMES[r["true_position"]] == r["pred_position"])

    metrics = {
        "n_evaluated": n,
        "n_skipped": len(skipped),
        "skipped": skipped,
        "combined_accuracy": comb_correct / n if n else None,
        "activity_accuracy": act_correct / n if n else None,
        "n_position_labeled": len(pos_rows),
        "position_coverage": len(pos_emitted) / len(pos_rows) if pos_rows else None,
        "position_accuracy_when_emitted": pos_correct / len(pos_emitted) if pos_emitted else None,
        "position_end_to_end_accuracy": pos_correct / len(pos_rows) if pos_rows else None,
        "confusion_combined": {
            f"{t} -> {p}": c for (t, p), c in Counter(
                (r["true_combined"], r["pred_combined"]) for r in rows).items()},
    }
    print(f"\n=== E4 end-to-end on test_minutes (n={n}) ===")
    print(f"combined 4-way accuracy : {metrics['combined_accuracy']:.3f}")
    print(f"activity 3-way accuracy : {metrics['activity_accuracy']:.3f}  (true held-out)")
    print(f"position (n={len(pos_rows)} t_present recs with left/right labels):")
    print(f"  coverage (position emitted)      : {metrics['position_coverage']:.3f}")
    print(f"  accuracy when emitted            : {metrics['position_accuracy_when_emitted']:.3f}")
    print(f"  end-to-end (emitted AND correct) : {metrics['position_end_to_end_accuracy']:.3f}")
    print("  (sanity check only — the position model was trained on all t; "
          "the held-out estimate is the E3 5-fold CV: see E3/outputs/REPORT.md)")

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.OUTPUT_DIR / "predict_capture_eval.json").write_text(
        json.dumps({"metrics": metrics, "rows": rows}, indent=2, default=str))
    print(f"saved {C.OUTPUT_DIR / 'predict_capture_eval.json'}")


if __name__ == "__main__":
    main()
