"""Figures for 1_theory/01_vehicle_kinematics.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

import motion_planning as mp
from motion_planning.plotting import INK, INK_MUTED, INK_SECONDARY, SERIES, draw_vehicle, mark_pose, use_style

OUT = Path(__file__).resolve().parents[2] / "1_theory" / "figures"


def wheel(ax, centre, angle, length=0.55, width=0.16, color=INK):
    c, s = np.cos(angle), np.sin(angle)
    corners = np.array([[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2],
                        [-length / 2, width / 2]])
    world = corners @ np.array([[c, s], [-s, c]]) + centre
    ax.add_patch(plt.Polygon(world, closed=True, facecolor=color, edgecolor=color))


def bicycle_geometry():
    L, gamma, theta, l_r = 2.0, 0.42, 0.30, 0.9
    rear = np.array([0.0, 0.0])
    fwd = np.array([np.cos(theta), np.sin(theta)])
    left = np.array([-np.sin(theta), np.cos(theta)])
    front = rear + L * fwd
    R = L / np.tan(gamma)
    icr = rear + R * left
    ref = rear + l_r * fwd
    beta = np.arctan(l_r / L * np.tan(gamma))

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.set_aspect("equal")
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.plot(*np.c_[rear, front], color=INK, lw=2.5)
    wheel(ax, rear, theta)
    wheel(ax, front, theta + gamma)
    # radii to the instantaneous centre of rotation
    ax.plot(*np.c_[rear, icr], color=INK_MUTED, lw=1, ls="--")
    ax.plot(*np.c_[front, icr], color=INK_MUTED, lw=1, ls="--")
    ax.plot(*np.c_[ref, icr], color=SERIES[1], lw=1, ls=":")
    ax.plot(*icr, "o", color=INK, ms=6)
    ax.annotate("ICR", icr, xytext=(8, -4), textcoords="offset points", color=INK)
    ax.annotate("$R = L / \\tan\\gamma$", (rear + icr) / 2, xytext=(-92, 0), textcoords="offset points",
                color=INK_SECONDARY)
    ax.annotate("$\\sqrt{R^2 + l_r^2}$", (ref + icr) / 2, xytext=(6, -6), textcoords="offset points",
                color=SERIES[1])

    # heading and steering directions
    ax.plot(*np.c_[front, front + 1.0 * fwd], color=INK_MUTED, lw=1)
    steer_dir = np.array([np.cos(theta + gamma), np.sin(theta + gamma)])
    ax.plot(*np.c_[front, front + 1.0 * steer_dir], color=INK_MUTED, lw=1)
    a = np.linspace(theta, theta + gamma, 20)
    ax.plot(front[0] + 0.75 * np.cos(a), front[1] + 0.75 * np.sin(a), color=INK, lw=1)
    ax.annotate("$\\gamma$", front + 0.85 * np.array([np.cos(theta + gamma / 2), np.sin(theta + gamma / 2)]),
                color=INK)
    ax.plot([rear[0], rear[0] + 1.1], [rear[1], rear[1]], color=INK_MUTED, lw=1)
    a = np.linspace(0, theta, 20)
    ax.plot(rear[0] + 0.55 * np.cos(a), rear[1] + 0.55 * np.sin(a), color=INK, lw=1)
    ax.annotate("$\\theta$", rear + 0.6 * np.array([np.cos(theta / 2), np.sin(theta / 2)]), color=INK,
                xytext=(3, -6), textcoords="offset points")

    # wheelbase label, offset to the right of the body
    for end in (rear, front):
        ax.plot(*np.c_[end - 0.25 * left, end - 0.6 * left], color=INK_MUTED, lw=0.8)
    ax.annotate("", xy=front - 0.5 * left, xytext=rear - 0.5 * left,
                arrowprops=dict(arrowstyle="<->", color=INK_SECONDARY, lw=0.8))
    ax.annotate("$L$", (rear + front) / 2 - 0.5 * left, color=INK, ha="center", va="top",
                xytext=(0, -4), textcoords="offset points")
    ax.annotate("$l_r$", (rear + ref) / 2 + 0.12 * left, color=SERIES[1], ha="center", va="bottom")

    # reference point and its velocity, at slip angle beta from the body axis
    ax.plot(*ref, "o", color=SERIES[1], ms=6)
    ax.annotate("reference point", ref, xytext=(40, 50), textcoords="offset points", color=SERIES[1],
                arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.8))
    vdir = np.array([np.cos(theta + beta), np.sin(theta + beta)])
    ax.annotate("", xy=ref + 1.2 * vdir, xytext=ref, arrowprops=dict(arrowstyle="-|>", color=SERIES[1], lw=1.5))
    ax.annotate("$v$", ref + 1.25 * vdir, color=SERIES[1])
    ax.annotate("$\\beta$", ref + 0.9 * np.array([np.cos(theta + beta * 0.4), np.sin(theta + beta * 0.4)]),
                color=SERIES[1], xytext=(4, -2), textcoords="offset points")
    ax.annotate("rear axle", rear, xytext=(-62, -6), textcoords="offset points", color=INK_SECONDARY)
    ax.annotate("front axle", front, xytext=(14, -18), textcoords="offset points", color=INK_SECONDARY)
    ax.set_xlim(-1.4, 3.6)
    ax.set_ylim(-1.2, R + 0.8)
    ax.set_title("Kinematic bicycle: geometry of a steady turn")
    fig.savefig(OUT / "01_bicycle_geometry.png")
    plt.close(fig)


def integrators():
    uni = mp.Unicycle()
    q0 = np.zeros(3)
    u = np.array([1.0, 1.0])
    T = 4.5  # not a whole turn: over a full circle every scheme closes its polygon by symmetry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3))
    t = np.linspace(0, T, 200)
    exact = np.array([mp.unicycle_exact_step(q0, u, ti) for ti in t])
    ax1.plot(exact[:, 0], exact[:, 1], color=INK_MUTED, lw=4, alpha=0.4, label="exact arc, eq. (1.16)")
    n = 9
    for method, color in zip([mp.Integrator.Euler, mp.Integrator.Midpoint, mp.Integrator.RK4], SERIES):
        traj = uni.simulate(q0, np.tile(u, (n, 1)), T / n, method)
        ax1.plot(traj[:, 0], traj[:, 1], "o-", color=color, ms=4, lw=1.5, label=method.name)
    ax1.set_aspect("equal")
    ax1.set_title(f"A {T:g} s constant turn in {n} steps")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.legend(loc="lower left")

    steps = np.array([8, 16, 32, 64, 128, 256])
    for method, color in zip([mp.Integrator.Euler, mp.Integrator.Midpoint, mp.Integrator.RK4], SERIES):
        err = []
        for n in steps:
            traj = uni.simulate(q0, np.tile(u, (n, 1)), T / n, method)
            err.append(np.linalg.norm(traj[-1] - mp.unicycle_exact_step(q0, u, T)))
        ax2.loglog(T / steps, err, "o-", color=color, ms=5, label=method.name)
        ax2.annotate(f"slope {np.polyfit(np.log(T / steps), np.log(err), 1)[0]:.1f}", (T / steps[0], err[0]),
                     xytext=(6, -4), textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("time step Δt [s]")
    ax2.set_ylabel("final pose error")
    ax2.set_title(f"Global error at t = {T:g} s")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "01_integrators.png")
    plt.close(fig)


def pose_regulation():
    goal = np.array([0.0, 0.0, np.pi / 2])
    opts = mp.RegulationOptions(position_tolerance=0.02, heading_tolerance=0.05)
    fig, ax = plt.subplots(figsize=(6, 6))
    n = 12
    for i in range(n):
        a = 2 * np.pi * i / n
        q0 = np.array([3 * np.cos(a), 3 * np.sin(a), 0.0])
        r = mp.regulate_pose(mp.Unicycle(), q0, goal, mp.PoseGains(), opts)
        reverse = r.inputs[0, 0] < 0
        color = SERIES[1] if reverse else SERIES[0]
        ax.plot(r.states[:, 0], r.states[:, 1], color=color, lw=1.5)
        draw_vehicle(ax, q0, 0.45, 0.25, color=color, filled=True)
    ax.plot([], [], color=SERIES[0], label="drives forwards")
    ax.plot([], [], color=SERIES[1], label="backs in (goal starts behind)")
    mark_pose(ax, goal, size=0.6, label="goal pose")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Polar pose law, eq. (1.21), from 12 start poses heading east")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.savefig(OUT / "01_pose_regulation.png")
    plt.close(fig)


if __name__ == "__main__":
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    bicycle_geometry()
    integrators()
    pose_regulation()
