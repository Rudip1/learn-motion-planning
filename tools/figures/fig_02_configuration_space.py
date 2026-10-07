"""Figures for 1_theory/02_configuration_space.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

import motion_planning as mp
from motion_planning import maps
from motion_planning.plotting import INK, INK_SECONDARY, OBSTACLE, SERIES, show_grid, use_style

OUT = Path(__file__).resolve().parents[2] / "1_theory" / "figures"


def distance_fields():
    g = maps.rooms()
    fields = [
        ("Brushfire, 4-connected (L1)", mp.brushfire(g, mp.Connectivity.Four) * g.resolution),
        ("Brushfire, 8-connected (L∞)", mp.brushfire(g, mp.Connectivity.Eight) * g.resolution),
        ("Exact Euclidean transform", mp.distance_transform(g)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmax = max(f.max() for _, f in fields)
    for ax, (title, f) in zip(axes, fields):
        show_grid(ax, g, field=f, cmap="Blues_r", vmin=0, vmax=vmax)
        ax.contour(f, levels=[0.5, 1.0, 1.5], origin="lower", extent=g.extent, colors=INK, linewidths=0.8)
        ax.set_title(title)
    fig.colorbar(axes[-1].images[0], ax=axes, shrink=0.8, label="distance to nearest obstacle [m]")
    fig.suptitle("Iso-distance lines at 0.5, 1.0 and 1.5 m: diamonds, squares and circles", y=1.0)
    fig.savefig(OUT / "02_distance_fields.png")
    plt.close(fig)


def inflation():
    g = maps.rooms()
    r = 0.35
    inflated = mp.inflate(g, r)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    show_grid(ax1, g)
    for p in [(2.0, 3.0), (6.2, 2.6), (8.0, 7.2)]:
        ax1.add_patch(plt.Circle(p, r, color=SERIES[0], alpha=0.8))
    ax1.set_title(f"Workspace: a disc robot of radius {r} m")
    added = inflated.to_array().astype(float) - g.to_array()
    show_grid(ax2, g)
    ax2.imshow(np.ma.masked_where(added == 0, added), origin="lower", extent=g.extent,
               cmap=plt.matplotlib.colors.ListedColormap([SERIES[1]]), alpha=0.6, interpolation="nearest")
    for p in [(2.0, 3.0), (6.2, 2.6), (8.0, 7.2)]:
        ax2.plot(*p, "o", color=SERIES[0], ms=6)
    ax2.set_title("Configuration space: obstacles grown by the radius,\nthe robot shrinks to a point")
    fig.tight_layout()
    fig.savefig(OUT / "02_inflation.png")
    plt.close(fig)


def segment_traversal():
    g = mp.OccupancyGrid(9, 4)
    a, b = g.cell_center((0, 0)), g.cell_center((7, 2))
    exact = mp.traverse_segment(g, a, b)
    bres = mp.bresenham((0, 0), (7, 2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, cells, title, color in [
        (axes[0], bres, f"Bresenham: {len(bres)} cells, one per column", SERIES[1]),
        (axes[1], exact, f"Exact traversal (Amanatides–Woo): {len(exact)} cells", SERIES[0]),
    ]:
        for x, y in cells:
            ax.add_patch(Rectangle((x, y), 1, 1, color=color, alpha=0.35))
        ax.add_patch(Rectangle((2, 0), 1, 1, fill=False, hatch="///", edgecolor=INK, lw=1.2))
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=1.5)
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 4)
        ax.set_xticks(range(10))
        ax.set_yticks(range(5))
        ax.grid(True, color=INK_SECONDARY, lw=0.6)
        ax.set_aspect("equal")
        ax.set_title(title)
    axes[0].annotate("clipped cell (2, 0)\nnot visited", (2.5, 0.5), xytext=(3.4, 0.25), color=INK, fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=INK))
    fig.tight_layout()
    fig.savefig(OUT / "02_segment_traversal.png")
    plt.close(fig)


def cspace_slices():
    g = maps.narrow_passage(resolution=0.05, gap=0.9)
    fp = mp.RectangleFootprint(rear=0.15, front=0.75, half_width=0.2)
    thetas = [0.0, np.pi / 4, np.pi / 2]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, th, color in zip(axes, thetas, SERIES):
        c = mp.configuration_space_slice(g, fp, th)
        added = c.to_array().astype(float) - g.to_array()
        show_grid(ax, g)
        ax.imshow(np.ma.masked_where(added == 0, added), origin="lower", extent=g.extent,
                  cmap=plt.matplotlib.colors.ListedColormap([color]), alpha=0.5, interpolation="nearest")
        q = np.array([1.5, 2.0, th])
        ax.add_patch(Polygon(fp.corners(q), closed=True, facecolor=color, edgecolor=INK, lw=1))
        ax.plot(*q[:2], "o", color=INK, ms=4)
        ax.set_title(f"θ = {np.degrees(th):.0f}°: is the passage open? "
                     f"{'yes' if not c.occupied(c.world_to_cell([3.0, 2.0])) else 'no'}")
    fig.suptitle("Slices of the configuration space of a 0.9 m × 0.4 m robot (reference point = black dot)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "02_cspace_slices.png")
    plt.close(fig)


if __name__ == "__main__":
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    distance_fields()
    inflation()
    segment_traversal()
    cspace_slices()
