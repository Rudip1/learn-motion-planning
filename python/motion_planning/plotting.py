"""Plotting helpers shared by the notebooks and the figure scripts. No algorithms live here."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# Categorical colours, always assigned in this order (validated for colour-vision deficiency).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
OBSTACLE = "#3b3a37"


def use_style() -> None:
    """Apply the module's matplotlib style: thin marks, recessive axes and grid, fixed series order."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "text.color": INK,
            "figure.dpi": 100,
            "savefig.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def draw_vehicle(ax, pose, length=1.0, width=0.5, color=INK_SECONDARY, alpha=1.0, filled=False):
    """Draw a vehicle footprint as an arrow-shaped polygon whose tip points along the heading.

    The rear edge is centred on the reference point (rear axle for the bicycle model).
    """
    x, y, th = pose
    body = np.array(
        [[0.0, -width / 2], [0.75 * length, -width / 2], [length, 0.0], [0.75 * length, width / 2], [0.0, width / 2]]
    )
    c, s = np.cos(th), np.sin(th)
    world = body @ np.array([[c, s], [-s, c]]) + np.array([x, y])
    patch = Polygon(world, closed=True, fill=filled, edgecolor=color, facecolor=color, lw=1.5, alpha=alpha)
    ax.add_patch(patch)
    return patch


def plot_path(ax, states, label=None, color=None, every=None, vehicle_length=None, **kwargs):
    """Plot the (x, y) columns of an (N, >=2) array; optionally draw the vehicle every `every` samples."""
    states = np.asarray(states)
    (line,) = ax.plot(states[:, 0], states[:, 1], label=label, color=color, **kwargs)
    if every and vehicle_length:
        for q in states[::every]:
            draw_vehicle(ax, q, vehicle_length, vehicle_length / 2, color=line.get_color(), alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    return line


def mark_pose(ax, pose, color=INK, size=1.0, label=None):
    """Mark a pose with a dot and a heading arrow."""
    x, y, th = pose
    ax.plot([x], [y], "o", color=color, label=label, ms=6)
    ax.annotate(
        "", xy=(x + size * np.cos(th), y + size * np.sin(th)), xytext=(x, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
    )


def new_axes(ncols=1, nrows=1, width=4.5, height=4.0, **kwargs):
    """Create a figure with the module style applied."""
    use_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), **kwargs)
    return fig, axes


def show_grid(ax, grid, field=None, cmap="viridis", obstacle_alpha=1.0, colorbar=None, **kwargs):
    """Draw an OccupancyGrid in world coordinates (obstacles dark), optionally under a scalar field.

    `field` is an array shaped like the grid (rows = y); infinite values are left blank.
    """
    occ = grid.to_array().astype(float)
    extent = grid.extent
    if field is not None:
        f = np.where(np.isfinite(field), field, np.nan)
        im = ax.imshow(f, origin="lower", extent=extent, cmap=cmap, interpolation="nearest", **kwargs)
        if colorbar:
            ax.figure.colorbar(im, ax=ax, label=colorbar, shrink=0.8)
    masked = np.ma.masked_where(occ == 0, occ)
    ax.imshow(masked, origin="lower", extent=extent, cmap=mpl.colors.ListedColormap([OBSTACLE]),
              interpolation="nearest", alpha=obstacle_alpha, vmin=0, vmax=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.grid(False)
    return ax


STATUS_COLORS = {"Success": "#008300", "Failure": "#e34948", "Running": "#eda100"}
_KIND_LABEL = {
    "Sequence": "→", "SequenceWithMemory": "→*", "Fallback": "?", "FallbackWithMemory": "?*", "Parallel": "⇉",
    "Inverter": "¬", "Retry": "retry", "Repeat": "repeat", "Timeout": "timeout", "ForceSuccess": "✓",
    "ForceFailure": "✗",
}


def draw_tree(ax, root, x_gap=1.0, y_gap=1.0, fontsize=8):
    """Draw a behaviour tree (motion_planning.bt node) top-down, coloured by each node's last status.

    Composites and decorators are squares with their symbol; conditions are ellipses, actions rectangles, both with
    their name. Nodes not ticked since the last halt are drawn white.
    """
    from matplotlib.patches import Ellipse, FancyBboxPatch

    positions = {}
    next_leaf = [0.0]

    def layout(node, depth):
        kids = list(node.children)
        if not kids:
            x = next_leaf[0]
            next_leaf[0] += x_gap
        else:
            xs = [layout(c, depth + 1) for c in kids]
            x = 0.5 * (xs[0] + xs[-1])
        positions[id(node)] = (x, -depth * y_gap, node)
        return x

    layout(root, 0)

    def draw(node):
        x, y, _ = positions[id(node)]
        for c in node.children:
            cx, cy, _ = positions[id(c)]
            ax.plot([x, cx], [y - 0.22 * y_gap, cy + 0.22 * y_gap], color=INK_MUTED, lw=1, zorder=1)
            draw(c)
        face = STATUS_COLORS[node.last_status.name] if node.ticked else SURFACE
        text_color = "white" if node.ticked else INK
        if node.kind in ("Condition", "Action"):
            w, h = 0.9 * x_gap, 0.42 * y_gap
            if node.kind == "Condition":
                ax.add_patch(Ellipse((x, y), w, h, facecolor=face, edgecolor=INK, lw=1, zorder=2))
            else:
                ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02",
                                            facecolor=face, edgecolor=INK, lw=1, zorder=2))
            ax.text(x, y, node.name, ha="center", va="center", fontsize=fontsize, color=text_color, zorder=3)
        else:
            s = 0.36 * y_gap
            ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s, boxstyle="square,pad=0.02",
                                        facecolor=face, edgecolor=INK, lw=1.2, zorder=2))
            label = _KIND_LABEL.get(node.kind, node.kind)
            ax.text(x, y, label, ha="center", va="center", fontsize=fontsize + 2 if len(label) <= 2 else fontsize - 2,
                    color=text_color, zorder=3)
            ax.text(x + s / 2 + 0.05, y + s / 2, node.name, ha="left", va="bottom", fontsize=fontsize - 1,
                    color=INK_SECONDARY, zorder=3)

    draw(root)
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - x_gap, max(xs) + x_gap)
    ax.set_ylim(min(ys) - 0.6 * y_gap, max(ys) + 0.6 * y_gap)
    ax.set_aspect("equal")
    ax.axis("off")
    for status, color in STATUS_COLORS.items():
        ax.plot([], [], "s", color=color, ms=9, label=status.upper())
    ax.plot([], [], "s", mfc=SURFACE, mec=INK, ms=9, label="not ticked")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=fontsize)
