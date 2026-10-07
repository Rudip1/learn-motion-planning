#!/usr/bin/env python3
"""Regenerate every figure in 1_theory/figures/ by running tools/figures/fig_*.py.

Usage:  make_figures.py [NN ...]      (NN selects scripts named fig_NN_*.py)
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    scripts = sorted((ROOT / "tools" / "figures").glob("fig_[0-9][0-9]_*.py"))
    if len(sys.argv) > 1:
        scripts = [s for s in scripts if s.name[4:6] in sys.argv[1:]]
    for script in scripts:
        print(f"running {script.relative_to(ROOT)}")
        runpy.run_path(str(script), run_name="__main__")
