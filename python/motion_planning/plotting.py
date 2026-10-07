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
