"""Figures for 1_theory/03_potential_fields.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import motion_planning as mp
from motion_planning import maps
from motion_planning.plotting import INK, SERIES, show_grid, use_style

OUT = Path(__file__).resolve().parents[2] / "1_theory" / "figures"


def cell_path_to_xy(grid, path):
    return np.array([grid.cell_center(tuple(c)) for c in path])


def trap_potentials():
    g = maps.trap()
    k = mp.PotentialParams(zeta=1.0, d_star=2.0, eta=1.0, q_star=1.0)
    start_xy, goal_xy = np.array([1.0, 3.0]), np.array([6.5, 3.0])
    start, goal = g.world_to_cell(start_xy), g.world_to_cell(goal_xy)
    U_att = mp.attractive_field(g, g.cell_center(goal), k)
    U_rep = mp.repulsive_field(g, k)
    U = U_att + U_rep
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    for ax, f, title in [(axes[0], U_att, "Attractive, eq. (3.1)"), (axes[1], U_rep, "Repulsive, eq. (3.3)"),
                         (axes[2], U, "Total, eq. (3.4), and steepest descent")]:
        show_grid(ax, g, field=np.minimum(f, 12.0), cmap="viridis")
        ax.set_title(title)
    r = mp.descend(U, start, goal)
    xy = cell_path_to_xy(g, r.path)
    axes[2].contour(np.minimum(U, 12.0), levels=20, origin="lower", extent=g.extent, colors="white",
                    linewidths=0.5, alpha=0.6)
    axes[2].plot(xy[:, 0], xy[:, 1], color=SERIES[1], lw=2.5)
    axes[2].plot(*start_xy, "o", color="white", ms=7)
    axes[2].plot(*goal_xy, "*", color="white", ms=12)
    axes[2].annotate("local minimum", xy[-1], xytext=(-20, -55), textcoords="offset points", color="white",
                     arrowprops=dict(arrowstyle="->", color="white"))
    fig.tight_layout()
    fig.savefig(OUT / "03_trap_potentials.png")
    plt.close(fig)


def wavefront_paths():
    g = maps.rooms()
    start_xy, goal_xy = np.array([2.0, 2.8]), np.array([8.0, 7.0])
    start, goal = g.world_to_cell(start_xy), g.world_to_cell(goal_xy)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, conn, color in [(axes[0], mp.Connectivity.Four, SERIES[1]), (axes[1], mp.Connectivity.Eight, SERIES[1])]:
        w = mp.wavefront(g, goal, conn)
        r = mp.descend(w, start, goal, conn)
        show_grid(ax, g, field=w * g.resolution, cmap="Blues_r", colorbar="steps × h [m]")
        xy = cell_path_to_xy(g, r.path)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2)
        ax.plot(*start_xy, "o", color=INK, ms=7)
        ax.plot(*goal_xy, "*", color=INK, ms=12)
        ax.set_title(f"Wave-front, {conn.name.lower()}-connected: {len(r.path) - 1} steps to the goal")
    fig.tight_layout()
    fig.savefig(OUT / "03_wavefront.png")
    plt.close(fig)


if __name__ == "__main__":
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    trap_potentials()
    wavefront_paths()
