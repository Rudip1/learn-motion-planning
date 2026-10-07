#!/usr/bin/env python3
"""Generate 2_notebooks/{exercises,solutions}/NN_<chapter>.ipynb from tools/notebooks/NN_<chapter>.py.

Source format (a small subset of the "percent" format):

    # %% [markdown]          markdown cell; each line starts with "# " (or is a bare "#")
    # %%                     code cell, in both notebooks
    # %% [solution]          code cell only in the solution notebook
    # %% [exercise]          code cell only in the exercise notebook (the reader's version of the cell above)

Usage:  build_notebooks.py [--check] [NN ...]
  --check   fail if a generated notebook's cells differ from the committed one (outputs are ignored)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tools" / "notebooks"
OUT = ROOT / "2_notebooks"
HEADER = re.compile(r"^# %%(?: \[(markdown|solution|exercise)\])?\s*$")


def parse(path: Path) -> list[tuple[str, str]]:
    cells: list[tuple[str, list[str]]] = []
    for line in path.read_text().splitlines():
        m = HEADER.match(line)
        if m:
            cells.append((m.group(1) or "code", []))
        elif cells:
            cells[-1][1].append(line)
        elif line.strip():
            raise ValueError(f"{path}: text before the first cell marker")
    out = []
    for kind, lines in cells:
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if kind == "markdown":
            lines = [re.sub(r"^# ?", "", ln) for ln in lines]
        out.append((kind, "\n".join(lines)))
    return out


def make(cells: list[tuple[str, str]], variant: str) -> nbformat.NotebookNode:
    skip = "exercise" if variant == "solutions" else "solution"
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    for kind, text in cells:
        if kind == skip:
            continue
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(text))
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:02d}"
    return nb


def same_cells(a: nbformat.NotebookNode, b: nbformat.NotebookNode) -> bool:
    key = lambda nb: [(c.cell_type, c.source) for c in nb.cells]  # noqa: E731
    return key(a) == key(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("chapters", nargs="*")
    args = ap.parse_args()
    sources = sorted(SRC.glob("[0-9][0-9]_*.py"))
    if args.chapters:
        sources = [s for s in sources if s.name[:2] in args.chapters]
    bad = 0
    for src in sources:
        cells = parse(src)
        for variant in ("exercises", "solutions"):
            target = OUT / variant / (src.stem + ".ipynb")
            nb = make(cells, variant)
            if args.check:
                if not target.exists() or not same_cells(nb, nbformat.read(target, as_version=4)):
                    print(f"out of date: {target.relative_to(ROOT)}")
                    bad += 1
                continue
            if variant == "solutions" and target.exists():
                old = nbformat.read(target, as_version=4)
                if same_cells(nb, old):
                    continue  # keep executed outputs
            target.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, target)
            print(f"wrote {target.relative_to(ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
