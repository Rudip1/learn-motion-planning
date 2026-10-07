"""Figures for 1_theory/05_sampling_planning.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

import motion_planning as mp
from motion_planning.plotting import INK, INK_MUTED, OBSTACLE, SERIES, use_style

OUT = Path(__file__).resolve().parents[2] / "1_theory" / "figures"

POLYGONS = [
    np.array([[1.5, 1.0], [3.5, 0.8], [3.8, 2.6], [2.0, 3.0]]),
    np.array([[5.0, 2.0], [6.5, 2.2], [6.0, 4.5], [4.6, 4.0]]),
    np.array([[2.5, 4.0], [3.8, 4.3], [3.2, 5.6]]),
    np.array([[7.0, 0.5], [8.4, 0.9], [8.0, 2.6], [7.4, 2.0]]),
]
LOWER, UPPER = np.array([0.0, 0.0]), np.array([9.6, 6.0])
START, GOAL = np.array([0.5, 0.5]), np.array([9.0, 5.0])


def problem():
    return mp.problem_from_polygons(POLYGONS, LOWER, UPPER, START, GOAL)


def optimum():
    return mp.graph_search(mp.visibility_graph(POLYGONS, [START, GOAL]), 0, 1).cost


def draw_world(ax):
    for poly in POLYGONS:
        ax.fill(*poly.T, color=OBSTACLE)
    ax.plot(*START, "o", color=INK, ms=7, zorder=5)
    ax.plot(*GOAL, "*", color=INK, ms=12, zorder=5)
    ax.set_xlim(LOWER[0], UPPER[0])
    ax.set_ylim(LOWER[1], UPPER[1])
    ax.set_aspect("equal")
    ax.grid(False)


def draw_tree(ax, t, color):
    N = t.nodes
    for i, p in enumerate(t.parent):
        if p >= 0:
            ax.plot(*N[[p, i]].T, color=color, lw=0.6, alpha=0.7)


def trees():
    opt = mp.RrtOptions(max_iterations=3000, step=0.5, stop_at_first_solution=False, seed=2)
    c_star = optimum()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (name, planner) in zip(axes, [("RRT", mp.rrt), ("RRT*", mp.rrt_star)]):
        t = planner(problem(), opt)
        draw_world(ax)
        draw_tree(ax, t, INK_MUTED)
        ax.plot(*t.path.T, color=SERIES[1], lw=2.5)
        ax.set_title(f"{name}, {opt.max_iterations} iterations: path {t.path_cost:.2f} m "
                     f"({t.path_cost / c_star:.3f}× optimum)")
    fig.tight_layout()
    fig.savefig(OUT / "05_rrt_vs_rrt_star.png")
    plt.close(fig)


def convergence():
    c_star = optimum()
    iters = 6000
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, planner, color in [("RRT (keeps growing)", mp.rrt, SERIES[0]), ("RRT*", mp.rrt_star, SERIES[1])]:
        H = []
        for seed in range(1, 21):
            opt = mp.RrtOptions(max_iterations=iters, step=0.5, stop_at_first_solution=False, seed=seed)
            H.append(np.array(planner(problem(), opt).best_cost_history) / c_star)
        H = np.minimum(np.array(H), 10.0)  # "no solution yet" ranks above every finite cost
        k = np.arange(1, iters + 1)
        med = np.median(H, axis=0)
        lo, hi = np.percentile(H, 10, axis=0), np.percentile(H, 90, axis=0)
        ax.fill_between(k, lo, hi, color=color, alpha=0.2, lw=0)
        ax.plot(k, med, color=color, label=f"{name}, median of 20 seeds")
    ax.axhline(1.0, color=INK, lw=1, ls="--", label="optimum (visibility graph)")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.98, 1.4)
    ax.set_xlabel("iterations")
    ax.set_ylabel("best path cost / optimum")
    ax.set_title("Asymptotic optimality: RRT* converges, RRT does not (band: 10–90 %)")
    ax.legend()
    fig.savefig(OUT / "05_convergence.png")
    plt.close(fig)


def roadmap():
    r = mp.prm(problem(), mp.PrmOptions(num_samples=300, seed=3))
    P = r.roadmap.points
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    draw_world(ax)
    for i in range(len(r.roadmap)):
        for j, _ in r.roadmap.edges(i):
            if j > i:
                ax.plot(*P[[i, j]].T, color=INK_MUTED, lw=0.5, alpha=0.7)
    ax.plot(P[2:, 0], P[2:, 1], ".", color=SERIES[0], ms=3)
    ax.plot(*P[r.query.path].T, color=SERIES[1], lw=2.5)
    ax.set_title(f"PRM*: 300 samples, {r.roadmap.num_edges // 2} edges, path {r.query.cost:.2f} m")
    fig.savefig(OUT / "05_prm.png")
    plt.close(fig)


if __name__ == "__main__":
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    trees()
    convergence()
    roadmap()
