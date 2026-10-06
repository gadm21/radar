"""Run the E1 class-separability analysis end to end.

    python E1/run_all.py [--skip-extract]

Steps:
  _extract_old.py       legacy train/test minutes -> E1/cache/win
                        (skippable once cached; needs E2 on path)
  extract_multilink.py  multilink_train/*.jsonl -> E1/cache/multilink
  features.py           window feature tables + PCA fits + pc-dot dumps
  analyze.py            iterative separability screen -> analysis.json
  plots.py              figures -> outputs/figs
  report.py             REPORT.md
"""
import subprocess
import sys
from pathlib import Path

E1 = Path(__file__).resolve().parent
PY = sys.executable


def step(args):
    print(f"\n>>> {' '.join(str(a) for a in args)}", flush=True)
    r = subprocess.run([PY] + [str(a) for a in args],
                       cwd=str(E1.parent))
    if r.returncode != 0:
        print(f"step failed: {args} (exit {r.returncode})", flush=True)
        sys.exit(r.returncode)


def main():
    skip_extract = "--skip-extract" in sys.argv
    if not skip_extract:
        step([E1 / "_extract_old.py", "--workers", "6"])
        step([E1 / "extract_multilink.py"])
    step([E1 / "features.py"])
    step([E1 / "analyze.py"])
    step([E1 / "plots.py"])
    step([E1 / "report.py"])
    print("\nAll done. See E1/outputs/ (REPORT.md, analysis.json, "
          "features_*.csv, rankings_*.csv, figs/).")


if __name__ == "__main__":
    main()
