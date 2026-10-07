"""Figures for 1_theory/04_graph_search.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import motion_planning as mp
from motion_planning import maps
from motion_planning.plotting import INK, INK_MUTED, OBSTACLE, SERIES, show_grid, use_style

OUT = Path(__file__).resolve().parents[2] / "1_theory" / "figures"

# Polygonal obstacles shared by the figure and the notebook (metres).
POLYGONS = [
    np.array([[1.5, 1.0], [3.5, 0.8], [3.8, 2.6], [2.0, 3.0]]),
    np.array([[5.0, 2.0], [6.5, 2.2], [6.0, 4.5], [4.6, 4.0]]),
    np.array([[2.5, 4.0], [3.8, 4.3], [3.2, 5.6]]),
    np.array([[7.0, 0.5], [8.4, 0.9], [8.0, 2.6], [7.4, 2.0]]),
]
START, GOAL = np.array([0.5, 0.5]), np.array([9.0, 5.0])


def expansions():
    g = maps.rooms()
    s, t = g.world_to_cell([2.0, 2.8]), g.world_to_cell([8.0, 7.0])
    cases = [
        ("Dijkstra", mp.GridSearchOptions(mp.SearchAlgorithm.Dijkstra)),
        ("A*, octile heuristic", mp.GridSearchOptions(mp.SearchAlgorithm.AStar)),
        ("Weighted A*, w = 2", mp.GridSearchOptions(mp.SearchAlgorithm.WeightedAStar, weight=2.0)),
        ("Greedy best-first", mp.GridSearchOptions(mp.SearchAlgorithm.GreedyBestFirst)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    optimal = None
    for ax, (name, opt) in zip(axes, cases):
        r = mp.grid_search(g, s, t, opt)
        optimal = optimal or r.cost
        order = np.full((g.height, g.width), np.nan)
        for i, (x, y) in enumerate(r.expanded):
            order[y, x] = i
        show_grid(ax, g, field=order, cmap="Blues")
        xy = np.array([g.cell_center(tuple(c)) for c in r.path])
        ax.plot(xy[:, 0], xy[:, 1], color=SERIES[1], lw=2)
        ax.set_title(f"{name}\n{len(r.expanded)} expanded, cost {r.cost:.2f} m ({r.cost / optimal:.2f}×)")
    fig.tight_layout()
    fig.savefig(OUT / "04_expansions.png")
    plt.close(fig)


def visibility():
    vg = mp.visibility_graph(POLYGONS, [START, GOAL])
    r = mp.graph_search(vg, 0, 1)
    P = vg.points
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i in range(len(vg)):
        for j, _ in vg.edges(i):
            if j > i:
                ax.plot(*P[[i, j]].T, color=INK_MUTED, lw=0.6, alpha=0.6)
    for poly in POLYGONS:
        ax.fill(*poly.T, color=OBSTACLE)
    ax.plot(*P[r.path].T, color=SERIES[1], lw=2.5, label=f"shortest path, {r.cost:.2f} m")
    ax.plot(*START, "o", color=INK, ms=7)
    ax.plot(*GOAL, "*", color=INK, ms=12)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(f"Visibility graph: {len(vg)} nodes, {vg.num_edges // 2} edges")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "04_visibility_graph.png")
    plt.close(fig)


if __name__ == "__main__":
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    expansions()
    visibility()
