"""Run the E1 dataset audit end to end.

    python E1/run_all.py

Steps: inspect -> figures -> report. Outputs land in E1/outputs/.
"""
import subprocess
import sys
from pathlib import Path

E1 = Path(__file__).resolve().parent
PY = sys.executable


def step(args):
    print(f"\n>>> {' '.join(args)}", flush=True)
    r = subprocess.run([PY] + args, cwd=str(E1.parent))
    if r.returncode != 0:
        print(f"step failed: {args} (exit {r.returncode})", flush=True)
        sys.exit(r.returncode)


def main():
    step([str(E1 / "inspect_dataset.py")])
    step([str(E1 / "plots.py")])
    step([str(E1 / "report.py")])
    print("\nAll done. See E1/outputs/ (REPORT.md, inspection.json, "
          "per_minute.csv, problems.csv, figs/).")


if __name__ == "__main__":
    main()
