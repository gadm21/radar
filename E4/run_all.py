"""Run the E4 combined pipeline end to end.

    python E4/run_all.py

Prerequisites: E2 and E3 caches must already exist
(`python E2/run_all.py`, `python E3/run_all.py`).

Steps: train activity (t1+t2 -> t) + position (t, deployment model on all
t) -> end-to-end predict_capture evaluation on test_minutes.
"""
import subprocess
import sys
from pathlib import Path

E4 = Path(__file__).resolve().parent
PY = sys.executable


def step(args):
    print(f"\n>>> {' '.join(args)}", flush=True)
    r = subprocess.run([PY] + args, cwd=str(E4.parent))
    if r.returncode != 0:
        print(f"step failed: {args} (exit {r.returncode})", flush=True)
        sys.exit(r.returncode)


def main():
    step([str(E4 / "train.py")])
    step([str(E4 / "evaluate.py")])
    step([str(E4 / "plots.py")])
    print("\nDone. See E4/outputs/ and E4/saved_models/.")


if __name__ == "__main__":
    main()
