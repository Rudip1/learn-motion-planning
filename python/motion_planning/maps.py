"""Example maps used by the notebooks and figure scripts.

Maps are built from rectangles in metres so they need no data files (and work on Colab). Every map is
deterministic. This module only constructs data; the algorithms that use it live in C++.
"""

from __future__ import annotations

import numpy as np

from ._core import OccupancyGrid


def _grid_from_boxes(width_m, height_m, resolution, boxes, border=True):
    """Rasterise axis-aligned boxes (xmin, ymin, xmax, ymax) in metres: a cell is occupied if its centre is inside."""
    w, h = int(round(width_m / resolution)), int(round(height_m / resolution))
    xs = (np.arange(w) + 0.5) * resolution
    ys = (np.arange(h) + 0.5) * resolution
    X, Y = np.meshgrid(xs, ys)
    occ = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        occ[(X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)] = 1
    if border:
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 1
    return OccupancyGrid(occ, resolution)


def rooms(resolution=0.1):
    """10 m x 8 m floor with four rooms, doors of 1 m, and a table."""
    t = 0.2  # wall thickness
    boxes = [
        (5.0 - t / 2, 0.0, 5.0 + t / 2, 2.0), (5.0 - t / 2, 3.0, 5.0 + t / 2, 5.5), (5.0 - t / 2, 6.5, 5.0 + t / 2, 8.0),
        (0.0, 4.0 - t / 2, 1.5, 4.0 + t / 2), (2.5, 4.0 - t / 2, 6.5, 4.0 + t / 2), (7.5, 4.0 - t / 2, 10.0, 4.0 + t / 2),
        (1.5, 1.2, 3.0, 2.2),  # table
        (7.0, 5.6, 8.6, 6.6),  # table
    ]
    return _grid_from_boxes(10.0, 8.0, resolution, boxes)


def narrow_passage(resolution=0.05, gap=0.6, center=2.0):
    """6 m x 4 m map split by a wall with one gap of the given width [m], centred at y = `center`."""
    boxes = [(2.9, 0.0, 3.1, center - gap / 2), (2.9, center + gap / 2, 3.1, 4.0)]
    return _grid_from_boxes(6.0, 4.0, resolution, boxes)


def trap(resolution=0.1):
    """8 m x 6 m map with a U-shaped obstacle open towards the left (a local-minimum trap for potential fields)."""
    boxes = [(4.0, 1.8, 4.3, 4.2), (2.6, 1.8, 4.3, 2.1), (2.6, 3.9, 4.3, 4.2)]
    return _grid_from_boxes(8.0, 6.0, resolution, boxes)


def clutter(resolution=0.1, n=25, seed=4, size=(10.0, 8.0)):
    """Random axis-aligned boxes; deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    boxes = []
    for _ in range(n):
        w, h = rng.uniform(0.3, 1.2, 2)
        x, y = rng.uniform(0, size[0] - w), rng.uniform(0, size[1] - h)
        boxes.append((x, y, x + w, y + h))
    return _grid_from_boxes(size[0], size[1], resolution, boxes)


def maze(cells=(8, 6), cell_size=1.0, resolution=0.1, seed=2):
    """A perfect maze (one path between any two cells) of `cells` corridors, carved by a seeded random DFS."""
    rng = np.random.default_rng(seed)
    nx, ny = cells
    k = int(round(cell_size / resolution))
    occ = np.ones((ny * k + 1, nx * k + 1), dtype=np.uint8)
    wall = max(1, k // 5)
    def carve(x0, y0, x1, y1):
        occ[y0:y1, x0:x1] = 0
    seen = np.zeros((ny, nx), bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    carve(wall, wall, k, k)
    while stack:
        cx, cy = stack[-1]
        nbrs = [(cx + dx, cy + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= cx + dx < nx and 0 <= cy + dy < ny and not seen[cy + dy, cx + dx]]
        if not nbrs:
            stack.pop()
            continue
        nx_, ny_ = nbrs[rng.integers(len(nbrs))]
        seen[ny_, nx_] = True
        carve(nx_ * k + wall, ny_ * k + wall, nx_ * k + k, ny_ * k + k)
        # open the wall between the two cells
        carve(min(cx, nx_) * k + wall, min(cy, ny_) * k + wall, max(cx, nx_) * k + k, max(cy, ny_) * k + k)
        stack.append((nx_, ny_))
    return OccupancyGrid(occ, resolution)


EXAMPLES = {"rooms": rooms, "narrow_passage": narrow_passage, "trap": trap, "clutter": clutter, "maze": maze}


def example_map(name: str, **kwargs) -> OccupancyGrid:
    """One of: rooms, narrow_passage, trap, clutter, maze."""
    return EXAMPLES[name](**kwargs)
