"""Peek at the first rows of each multilink JSONL stream."""
import itertools
import json
from pathlib import Path

BASE = Path(r"c:/Users/ggad/Desktop/radar/multilink_train")
CAP = BASE / "capture_occupied_20261005_124135"

for f in ["csi-bb8b.jsonl", "radar-a316.jsonl",
          "thoth-toronto_csi-8b45.jsonl"]:
    print("===", f)
    p = CAP / f
    if not p.exists():
        print("   MISSING")
        continue
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in itertools.islice(fh, 2):
            d = json.loads(line)
            print({k: str(v)[:100] for k, v in d.items()})
            pay = d.get("payload")
            if isinstance(pay, dict):
                print("   payload:",
                      {k: str(v)[:60] for k, v in pay.items()})
    # manifest
print("=== manifest")
print((CAP / "manifest.json").read_text()[:800])
