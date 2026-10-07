"""Figures for 1_theory/07_smoothing_tracking.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import motion_planning as mp
from motion_planning import maps
from motion_planning.plotting import INK, INK_MUTED, INK_SECONDARY, SERIES, draw_vehicle, show_grid, use_style

OUT = Path(__file__).resolve().parents[2] / "1_theory" / "figures"


def planned():
    g = maps.rooms()
    infl = mp.inflate(g, 0.3)
    p = mp.problem_from_grid(infl, [2.0, 2.8], [8.0, 7.0])
    raw = mp.rrt(p, mp.RrtOptions(step=0.3, seed=2)).path
    greedy = mp.shortcut_greedy(raw, p.motion_valid)
    rand = mp.shortcut_greedy(mp.shortcut_random(raw, p.motion_valid, 300), p.motion_valid)
    return g, raw, greedy, rand


def shortcut_figure():
    g, raw, greedy, rand = planned()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    show_grid(ax, g)
    for path, name, c, lw in [(raw, "RRT", SERIES[0], 1.2), (greedy, "greedy shortcut", SERIES[2], 2.0),
                              (rand, "random shortcuts, then greedy", SERIES[1], 2.5)]:
        ax.plot(path[:, 0], path[:, 1], "o-" if len(path) < 30 else "-", color=c, lw=lw, ms=4,
                label=f"{name}: {mp.path_length(path):.2f} m, {len(path)} points")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_title("Shortcutting an RRT path (obstacles inflated by 0.3 m for planning)")
    fig.savefig(OUT / "07_shortcutting.png")
    plt.close(fig)


def pure_pursuit_geometry():
    fig, ax = plt.subplots(figsize=(6, 5))
    target = np.array([3.0, 1.0])
    R = 1 / mp.pure_pursuit_curvature(np.zeros(3), target)
    a = np.linspace(-np.pi / 2, -np.pi / 2 + 0.75, 100)
    ax.plot(R * np.cos(a), R + R * np.sin(a), color=SERIES[1], lw=2, label=f"commanded arc, R = {R:.1f}")
    path_x = np.linspace(-1, 6, 50)
    ax.plot(path_x, 1.2 - 0.04 * (path_x - 3) ** 2, color=SERIES[0], lw=2, label="path")
    ax.plot([0, target[0]], [0, target[1]], color=INK_SECONDARY, lw=1, ls="--")
    ax.plot([0, 0], [0, R], color=INK_MUTED, lw=1, ls=":")
    ax.plot([target[0], 0], [target[1], R], color=INK_MUTED, lw=1, ls=":")
    ax.plot(0, R, "o", color=INK, ms=4)
    ax.annotate("centre", (0, R), xytext=(6, 0), textcoords="offset points", color=INK_SECONDARY)
    ax.plot(*target, "o", color=SERIES[1], ms=7)
    ax.annotate("target, $L_d$ ahead", target, xytext=(8, -18), textcoords="offset points", color=INK)
    ax.annotate("$L_d$", target / 2, xytext=(4, -14), textcoords="offset points", color=INK)
    ang = np.linspace(0, np.arctan2(1, 3), 20)
    ax.plot(0.9 * np.cos(ang), 0.9 * np.sin(ang), color=INK, lw=1)
    ax.annotate("$\\alpha$", (0.95, 0.12), color=INK)
    ax.annotate("$y_v$", (3.05, 0.4), color=INK)
    ax.plot([3, 3], [0, 1], color=INK_MUTED, lw=1)
    ax.plot([0, 3], [0, 0], color=INK_MUTED, lw=1)
    draw_vehicle(ax, (0, 0, 0), 0.8, 0.4, color=INK, filled=True)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 5.6)
    ax.grid(False)
    ax.legend(loc="upper right")
    ax.set_title("Pure pursuit: the arc through the rear axle and the target")
    fig.savefig(OUT / "07_pure_pursuit_geometry.png")
    plt.close(fig)


def tracking_figure():
    g, _, _, path = planned()
    bike = mp.KinematicBicycle(0.5, 0.6)
    start = np.r_[path[0], np.arctan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0])]
    cases = [
        ("pure pursuit, $L_d$ = 0.3 m", mp.TrackingOptions(mp.TrackingController.PurePursuit, speed=1.0, lookahead=0.3)),
        ("pure pursuit, $L_d$ = 1.0 m", mp.TrackingOptions(mp.TrackingController.PurePursuit, speed=1.0, lookahead=1.0)),
        ("Stanley, k = 2", mp.TrackingOptions(mp.TrackingController.Stanley, speed=1.0, stanley_gain=2.0)),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    show_grid(ax, g)
    ax.plot(path[:, 0], path[:, 1], color=INK, lw=1, ls="--", label="path")
    for (name, opt), c in zip(cases, [SERIES[0], SERIES[1], SERIES[2]]):
        r = mp.track_path(bike, path, start, opt)
        ax.plot(r.states[:, 0], r.states[:, 1], color=c, lw=2, label=name)
        t = np.arange(len(r.cross_track)) * opt.dt
        ax2.plot(t, r.cross_track, color=c, lw=1.5, label=name)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title("Rear-axle trajectories, wheelbase 0.5 m, 1 m/s")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("cross-track error of the rear axle [m]")
    ax2.legend()
    ax2.set_title("Errors spike at the corners of the polyline")
    fig.tight_layout()
    fig.savefig(OUT / "07_tracking.png")
    plt.close(fig)


if __name__ == "__main__":
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    shortcut_figure()
    pure_pursuit_geometry()
    tracking_figure()
