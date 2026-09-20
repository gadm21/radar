"""Run the whole E2 pipeline end to end.

    python E2/run_all.py            # full pipeline
    python E2/run_all.py --skip-preprocess   # reuse existing cache

Steps: inspect -> preprocess (cache) -> E2 (train+train2 / validation /
test, balanced, config search) -> scene model (deployment export) ->
deploy eval -> figures -> report.
"""
import subprocess
import sys
from pathlib import Path

E2 = Path(__file__).resolve().parent
PY = sys.executable


def step(args):
    print(f"\n>>> {' '.join(args)}", flush=True)
    r = subprocess.run([PY] + args, cwd=str(E2.parent))
    if r.returncode != 0:
        print(f"step failed: {args} (exit {r.returncode})", flush=True)
        sys.exit(r.returncode)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    step([str(E2 / "inspect_dataset.py")])
    if not a.skip_preprocess:
        step([str(E2 / "preprocess.py"), "--workers", str(a.workers)])
    step([str(E2 / "experiments.py"), "--steps", "e2"])
    step([str(E2 / "scene_model.py")])
    step([str(E2 / "export_models.py")])
    step([str(E2 / "eval_deploy.py")])
    step([str(E2 / "plots.py")])
    step([str(E2 / "report.py")])
    print("\nAll done. See E2/outputs/ (REPORT.md, figs/, *.json).")


if __name__ == "__main__":
    main()
