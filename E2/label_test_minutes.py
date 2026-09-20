"""Write ground-truth labels into test_minutes/ manifests.

The Pi test night (Sept 16-17) was recorded without task labels. Ground
truth is the 1:10 AM boundary: minutes whose folder name is before
`20260917_0110` are `present` (occupied), at/after are `empty`.

For each folder: manifest.json is loaded tolerantly (clean JSON in
practice for this split), `labels` is set to exactly the ground-truth
label, and the file is rewritten. A folder with no manifest gets a
minimal one (`{"labels": [...], "labeled_by": "boundary"}`).

Usage:
    python E2/label_test_minutes.py --dry-run   # report only
    python E2/label_test_minutes.py             # apply
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = C.TEST_MINUTES_DIR
    n_empty = n_present = n_created = n_changed = 0
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        label = "empty" if folder.name >= C.TEST_BOUNDARY_FOLDER else "present"
        mpath = folder / "manifest.json"
        if mpath.exists():
            man, _mode = C.load_manifest(mpath)
            if not isinstance(man, dict):
                man = {}
        else:
            man = {"labeled_by": "boundary"}
            n_created += 1
        old = man.get("labels") or []
        if old == [label]:
            continue
        man["labels"] = [label]
        n_changed += 1
        if label == "empty":
            n_empty += 1
        else:
            n_present += 1
        if not a.dry_run:
            mpath.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"test_minutes: {n_changed} manifests labeled "
          f"({n_present} present / {n_empty} empty), "
          f"{n_created} manifest(s) created"
          + ("  [dry run]" if a.dry_run else ""))


if __name__ == "__main__":
    main()
