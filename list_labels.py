import argparse
import json
from collections import Counter
from pathlib import Path


def _parse_rename(rename_str):
    if not rename_str:
        return None, None
    if ":" not in rename_str:
        raise SystemExit("Rename must be specified as 'old_label:new_label'")
    old, new = rename_str.split(":", 1)
    old = old.strip()
    new = new.strip()
    if not old or not new:
        raise SystemExit("Both old and new labels must be non-empty")
    return old, new


def list_labels(root="minutes", exclude=None, remove=None, rename=None, rename_before=None, apply=False):
    exclude = set(exclude or [])
    remove = set(remove or [])
    old_label, new_label = _parse_rename(rename) if rename else (None, None)

    if old_label and not rename_before:
        raise SystemExit("--rename requires --before")

    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    minutes = sorted(p for p in root.iterdir() if p.is_dir())
    if not minutes:
        print(f"No minute folders found in {root}")
        return

    label_counter = Counter()
    removed_minutes = 0
    renamed_minutes = 0

    print("Labels per minute (in chronological order):\n")
    for minute_dir in minutes:
        manifest = minute_dir / "manifest.json"
        data = {}
        labels = []
        loaded = False

        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    labels = data.get("labels", []) or []
                    loaded = True
            except (json.JSONDecodeError, OSError):
                pass

        changed = False
        before_cutoff = rename_before and minute_dir.name < rename_before

        if remove:
            new_labels = [label for label in labels if label not in remove]
            if new_labels != labels:
                labels = new_labels
                changed = True
                removed_minutes += 1

        if old_label and new_label and before_cutoff:
            new_labels = [new_label if label == old_label else label for label in labels]
            if new_labels != labels:
                labels = new_labels
                changed = True
                renamed_minutes += 1

        report_labels = [label for label in labels if label not in exclude]

        if apply and loaded and changed:
            data["labels"] = labels
            try:
                manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                print(f"Error writing {manifest}: {exc}")

        print(f"{minute_dir.name}: {report_labels if report_labels else '(none)'}")
        for label in report_labels:
            label_counter[label] += 1

    if remove or old_label:
        mode = "APPLY" if apply else "DRY RUN"
        print(f"\nMode: {mode}")
        print(f"  Minute folders inspected: {len(minutes)}")
        if remove:
            print(f"  Manifests with labels removed: {removed_minutes}")
        if old_label:
            print(f"  Manifests with '{old_label}' renamed to '{new_label}' before '{rename_before}': {renamed_minutes}")

        if remove:
            print(f"\nRemoved label(s): {', '.join(sorted(remove))}")
        if old_label:
            print(f"Renamed '{old_label}' -> '{new_label}' for minutes before '{rename_before}'")

    print("\nNumber of minutes associated with each label:\n")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List, remove, or rename labels in minute manifests.")
    parser.add_argument(
        "--exc",
        type=str,
        default="",
        help="Comma-separated list of labels to exclude from the report.",
    )
    parser.add_argument(
        "--remove",
        type=str,
        default="",
        help="Comma-separated list of labels to remove from every manifest.",
    )
    parser.add_argument(
        "--rename",
        type=str,
        default="",
        metavar="OLD:NEW",
        help="Rename a label for all minutes before the --before cutoff.",
    )
    parser.add_argument(
        "--before",
        type=str,
        default="",
        help="Timestamp cutoff (same naming convention as minute folders) for renaming.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the changes without modifying manifest.json.",
    )
    args = parser.parse_args()
    excluded = [label.strip() for label in args.exc.split(",") if label.strip()]
    remove = [label.strip() for label in args.remove.split(",") if label.strip()]
    list_labels(
        exclude=excluded,
        remove=remove,
        rename=args.rename,
        rename_before=args.before,
        apply=(not args.dry_run and (remove or args.rename)),
    )
