"""Figures for 1_theory/08_local_planning.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import motion_planning as mp
from motion_planning import maps
from motion_planning.plotting import INK, INK_MUTED, SERIES, draw_vehicle, show_grid, use_style

OUT = Path(__file__).resolve().parents[2] / "1_theory" / "figures"


def global_path(grid, start, goal, margin):
    infl = mp.inflate(grid, margin)
    plan = mp.grid_search(infl, infl.world_to_cell(start[:2]), infl.world_to_cell(goal))
    return np.array([grid.cell_center(tuple(c)) for c in plan.path])


def window_figure():
    g = maps.trap(resolution=0.05)
    D = mp.distance_transform(g)
    cfg = mp.DwaConfig()
    q, u = np.array([3.45, 3.0, 0.0]), np.array([0.5, 0.3])
    goal = np.array([6.5, 3.0])
    d = mp.dwa_step(g, D, q, u, goal, cfg)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    show_grid(ax1, g)
    for c in d.candidates:
        t = mp.rollout(q, c.v, c.w, cfg)
        ax1.plot(t[:, 0], t[:, 1], color=SERIES[0] if c.admissible else SERIES[7], lw=0.6, alpha=0.6)
    ax1.plot(d.best_rollout[:, 0], d.best_rollout[:, 1], color=SERIES[1], lw=2.5)
    draw_vehicle(ax1, q, 0.4, 0.4, color=INK, filled=True)
    ax1.plot(*goal, "*", color=INK, ms=12)
    ax1.set_xlim(1.5, 7.5)
    ax1.set_ylim(1.0, 5.0)
    ax1.set_title("2 s rollouts: admissible = can still brake before the wall (blue),\nnot admissible (red), chosen (orange)")

    ax2.add_patch(Rectangle((cfg.min_speed, -cfg.max_yaw_rate), cfg.max_speed - cfg.min_speed, 2 * cfg.max_yaw_rate,
                            fill=False, edgecolor=INK_MUTED, lw=1.5, ls="--"))
    ax2.annotate("$V_s$: robot limits", (0.02, cfg.max_yaw_rate - 0.15), color=INK_MUTED)
    w = mp.dynamic_window(u, cfg)
    ax2.add_patch(Rectangle((w.v_min, w.w_min), w.v_max - w.v_min, w.w_max - w.w_min, fill=False, edgecolor=INK, lw=1.5))
    ax2.annotate("$V_d$: dynamic window", (w.v_min, w.w_max + 0.05), color=INK)
    V = np.array([c.v for c in d.candidates])
    W = np.array([c.w for c in d.candidates])
    T = np.array([c.total for c in d.candidates])
    A = np.array([c.admissible for c in d.candidates])
    sc = ax2.scatter(V[A], W[A], c=T[A], cmap="viridis", s=18, label="admissible, coloured by score")
    ax2.scatter(V[~A], W[~A], marker="x", color=SERIES[7], s=18, label="not admissible (8.3)")
    ax2.plot(*d.command, "o", ms=12, mfc="none", mec=SERIES[1], mew=2, label="chosen")
    ax2.plot(*u, "+", color=INK, ms=10, label="current velocity")
    fig.colorbar(sc, ax=ax2, label="G(v, ω), eq. (8.4)")
    ax2.set_xlim(-0.05, 0.7)
    ax2.set_ylim(-1.7, 1.7)
    ax2.set_xlabel("v [m/s]")
    ax2.set_ylabel("ω [rad/s]")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_title("Velocity space")
    fig.tight_layout()
    fig.savefig(OUT / "08_dynamic_window.png")
    plt.close(fig)


def trap_figure():
    g = maps.trap(resolution=0.05)
    cfg = mp.DwaConfig()
    start, goal = np.array([1.0, 3.0, 0.0]), np.array([6.5, 3.0])
    path = global_path(g, start, goal, cfg.robot_radius + 0.15)
    alone = mp.run_dwa(g, start, goal, cfg, max_time=40.0)
    guided = mp.run_dwa(g, start, goal, cfg, path, 0.6)
    fig, ax = plt.subplots(figsize=(7, 5))
    show_grid(ax, g)
    ax.plot(path[:, 0], path[:, 1], "--", color=INK, lw=1, label="global A* path (inflated map)")
    ax.plot(alone.states[:, 0], alone.states[:, 1], color=SERIES[7], lw=2,
            label=f"DWA alone: reached = {alone.reached_goal}, stuck = {alone.stuck}")
    ax.plot(guided.states[:, 0], guided.states[:, 1], color=SERIES[0], lw=2,
            label=f"DWA chasing a carrot on the path: reached = {guided.reached_goal}")
    ax.plot(*start[:2], "o", color=INK, ms=7)
    ax.plot(*goal, "*", color=INK, ms=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_title("Why a local planner needs a global one")
    fig.savefig(OUT / "08_trap.png")
    plt.close(fig)


if __name__ == "__main__":
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    window_figure()
    trap_figure()
