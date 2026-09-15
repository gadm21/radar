"""Run the whole E3 pipeline end to end.

    python E3/run_all.py                     # full pipeline
    python E3/run_all.py --skip-preprocess    # reuse existing cache

Steps: inspect -> preprocess (cache) -> k-fold experiment (config search
+ cross-validation) -> figures -> report.
"""
import subprocess
import sys
from pathlib import Path

E3 = Path(__file__).resolve().parent
PY = sys.executable


def step(args):
    print(f"\n>>> {' '.join(args)}", flush=True)
    r = subprocess.run([PY] + args, cwd=str(E3.parent))
    if r.returncode != 0:
        print(f"step failed: {args} (exit {r.returncode})", flush=True)
        sys.exit(r.returncode)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-folds", type=int, default=5)
    a = ap.parse_args()

    step([str(E3 / "inspect_dataset.py")])
    if not a.skip_preprocess:
        step([str(E3 / "preprocess.py"), "--workers", str(a.workers)])
    step([str(E3 / "experiments.py"), "--n-folds", str(a.n_folds)])
    step([str(E3 / "plots.py")])
    step([str(E3 / "report.py")])
    print("\nAll done. See E3/outputs/ (REPORT.md, figs/, *.json).")


if __name__ == "__main__":
    main()
