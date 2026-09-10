import argparse
import json
import shutil
from pathlib import Path


def remove_empty_minutes(root="minutes", apply=False):
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    minutes = sorted(p for p in root.iterdir() if p.is_dir())
    to_delete = []

    for minute_dir in minutes:
        manifest = minute_dir / "manifest.json"
        labels = None

        if not manifest.exists():
            labels = []
        else:
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    labels = data.get("labels", []) or []
                else:
                    labels = []
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                # Treat unparseable manifests as having no labels.
                labels = []

        if labels is None:
            labels = []

        if not labels:
            to_delete.append(minute_dir.name)
            if apply:
                shutil.rmtree(minute_dir)

    mode = "APPLY" if apply else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"  Minute folders inspected: {len(minutes)}")
    print(f"  Empty folders to delete:  {len(to_delete)}")

    if to_delete:
        print("\nFolders to delete:")
        for name in to_delete:
            print(f"  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove minute folders that have no labels.")
    parser.add_argument(
        "--root",
        type=str,
        default="minutes",
        help="Root directory containing minute folders.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the empty minute folders.",
    )
    args = parser.parse_args()
    remove_empty_minutes(root=args.root, apply=args.apply)
