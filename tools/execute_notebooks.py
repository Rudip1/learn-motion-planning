#!/usr/bin/env python3
"""Execute solution notebooks top to bottom and store their outputs.

Usage:  execute_notebooks.py [--no-write] [NN ...]
Exits non-zero if any cell raises.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS = ROOT / "2_notebooks" / "solutions"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true", help="execute only, do not save outputs")
    ap.add_argument("--timeout", type=int, default=600, help="per-cell timeout in seconds")
    ap.add_argument("chapters", nargs="*")
    args = ap.parse_args()
    notebooks = sorted(SOLUTIONS.glob("[0-9][0-9]_*.ipynb"))
    if args.chapters:
        notebooks = [n for n in notebooks if n.name[:2] in args.chapters]
    failed = []
    for path in notebooks:
        nb = nbformat.read(path, as_version=4)
        t0 = time.time()
        try:
            NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                           resources={"metadata": {"path": str(path.parent)}}).execute()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {path.name}: {exc}")
            failed.append(path.name)
            continue
        print(f"ok   {path.name} ({time.time() - t0:.1f} s)")
        if not args.no_write:
            nb.metadata.pop("widgets", None)
            nbformat.write(nb, path)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
