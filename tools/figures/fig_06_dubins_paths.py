"""Figures for 1_theory/06_dubins_paths.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import motion_planning as mp
from motion_planning import maps
from motion_planning.plotting import INK, SERIES, mark_pose, show_grid, use_style

OUT = Path(__file__).resolve().parents[2] / "1_theory" / "figures"
WORDS = ["LSL", "RSR", "LSR", "RSL", "RLR", "LRL"]


def six_words():
    q0, q1 = np.zeros(3), np.array([-2.4, 1.6, 2.6])
    paths = mp.all_dubins_paths(q0, q1, 1.0)
    assert len(paths) == 6
    best = min(p.length() for p in paths)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, p, c in zip(axes.flat, paths, SERIES):
        s = p.sample(0.02)
        ax.plot(s[:, 0], s[:, 1], color=c, lw=2.5)
        mark_pose(ax, q0, size=0.6)
        mark_pose(ax, q1, size=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(-4.5, 2.5)
        ax.set_ylim(-2.5, 3.5)
        tag = "  ← shortest" if np.isclose(p.length(), best) else ""
        ax.set_title(f"{p.word.name}: {p.length():.2f}{tag}")
    fig.suptitle("The six Dubins words between the same two poses (radius 1)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "06_six_words.png")
    plt.close(fig)


def word_map():
    xs = np.linspace(-5, 5, 201)
    word = np.zeros((xs.size, xs.size))
    dist = np.zeros_like(word)
    for i, y in enumerate(xs):
        for j, x in enumerate(xs):
            p = mp.shortest_dubins_path(np.zeros(3), np.array([x, y, 0.0]), 1.0)
            word[i, j] = WORDS.index(p.word.name)
            dist[i, j] = p.length()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax1.imshow(dist, origin="lower", extent=(-5, 5, -5, 5), cmap="Blues_r")
    ax1.contour(xs, xs, dist, levels=np.arange(1, 16), colors="white", linewidths=0.5)
    fig.colorbar(im, ax=ax1, shrink=0.8, label="Dubins distance")
    ax1.set_title("Distance from the origin (heading east)\nto goals heading east")
    ax2.imshow(word, origin="lower", extent=(-5, 5, -5, 5), cmap=ListedColormap(SERIES[:6]), vmin=-0.5, vmax=5.5,
               interpolation="nearest")
    for k, w in enumerate(WORDS):
        ax2.plot([], [], "s", color=SERIES[k], ms=10, label=w)
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax2.set_title("Shortest word")
    for ax in (ax1, ax2):
        mark_pose(ax, np.zeros(3), color=INK, size=0.8)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT / "06_word_map.png")
    plt.close(fig)


def dubins_rrt():
    g = maps.rooms()
    infl = mp.inflate(g, 0.25)
    problem = mp.problem_from_grid(infl, [2.0, 2.8], [8.0, 7.0])
    start, goal = np.array([2.0, 2.8, np.pi / 2]), np.array([8.0, 7.0, 0.0])
    t = mp.dubins_rrt_star(problem, start, goal,
                           mp.DubinsPlannerOptions(max_iterations=2500, radius=0.5, step=2.0, seed=3))
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    show_grid(ax, g)
    for i, p in enumerate(t.parent):
        if p >= 0:
            a, b = t.nodes[p], t.nodes[i]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=SERIES[0], lw=0.4, alpha=0.4)
    path = t.sample_path(0.02)
    ax.plot(path[:, 0], path[:, 1], color=SERIES[1], lw=2.5)
    mark_pose(ax, start, size=0.6)
    mark_pose(ax, goal, size=0.6)
    ax.set_title(f"Dubins-RRT*, turning radius 0.5 m: path {t.path_cost:.2f} m\n(tree edges drawn as chords)")
    fig.savefig(OUT / "06_dubins_rrt_star.png")
    plt.close(fig)


if __name__ == "__main__":
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    six_words()
    word_map()
    dubins_rrt()
