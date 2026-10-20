"""Rebuild the current E1 figures from retained feature tables/PCA dots."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
script = Path(__file__).resolve().parent / "revision_figures.py"
raise SystemExit(subprocess.call([sys.executable, str(script)], cwd=root))
