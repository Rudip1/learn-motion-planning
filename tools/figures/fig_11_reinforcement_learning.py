"""Figures for 1_theory/11_reinforcement_learning.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from motion_planning import rl
from motion_planning.plotting import INK, SERIES, use_style

OUT = Path(__file__).resolve().parents[2] / "1_theory" / "figures"
ARROW = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}  # action -> (dx, dy) on screen (row 0 at the top)


def draw_grid(ax, grid, V=None, policy=None, path=None, vmin=None, vmax=None, title=""):
    rows, cols = grid.rows, grid.cols
    img = np.full((rows, cols), np.nan)
    if V is not None:
        for s in range(rows * cols):
            if grid.cell(s) not in "#GP":
                img[grid.row(s), grid.col(s)] = V[s]
    if V is None:  # paths only: a light background with grid lines
        ax.imshow(np.full((rows, cols), 1.0), cmap="Greys", vmin=0, vmax=10)
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1)
    else:
        ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
    for s in range(rows * cols):
        r, c, ch = grid.row(s), grid.col(s), grid.cell(s)
        if ch == "#":
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="0.25"))
        elif ch == "P":
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#e34948", alpha=0.8))
        elif ch == "G":
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#1baf7a"))
            ax.text(c, r, "G", ha="center", va="center", color="white", fontsize=9, weight="bold")
        elif ch == "S":
            ax.text(c, r + 0.33, "S", ha="center", va="center", color="white" if V is not None else INK, fontsize=7)
        if policy is not None and policy[s] >= 0 and ch not in "#GP":
            dx, dy = ARROW[policy[s]]
            ax.annotate("", (c + 0.32 * dx, r - 0.32 * dy), (c - 0.32 * dx, r + 0.32 * dy),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.0))
    if path is not None:
        ax.plot([grid.col(s) for s in path], [grid.row(s) for s in path], "-", color="#eda100", lw=2.5, alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=9)


def policies_figure():
    cliff = rl.cliff_world()
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.2))
    for ax, slip in zip(axes, [0.0, 0.1, 0.2]):
        g = rl.GridWorld.from_layout(cliff.layout, -1.0, 0.0, -100.0, slip)
        m = g.to_mdp()
        vi = rl.value_iteration(m, 1.0, 1e-10)
        path = rl.rollout(m, vi.policy, g.start()) if slip < 0.2 else None
        draw_grid(ax, g, vi.V, vi.policy, path, vmin=-45, vmax=0,
                  title=f"slip {slip:.1f}: $V^*$(start) = {vi.V[g.start()]:.1f}, value iteration {vi.iterations} sweeps")
    fig.tight_layout()
    fig.savefig(OUT / "11_cliff_policies.png", dpi=120)
    plt.close(fig)


def learning_figure():
    g = rl.cliff_world()
    m = g.to_mdp()
    fig = plt.figure(figsize=(11, 4.2))
    ax0 = fig.add_subplot(1, 2, 1)
    runs = {}
    for k, (name, method) in enumerate([("Q-learning", rl.TdMethod.QLearning), ("SARSA", rl.TdMethod.Sarsa)]):
        curves = []
        for seed in range(1, 11):
            r = rl.td_control(m, g.start(), rl.TdConfig(method, 1.0, 0.5, 0.1, 500, 1000, seed))
            curves.append(r.episode_returns)
            if seed == 7:
                runs[name] = r
        mean = np.mean(curves, axis=0)
        smooth = np.convolve(mean, np.ones(10) / 10, mode="valid")
        ax0.plot(np.arange(len(smooth)) + 10, smooth, color=SERIES[k], label=name)
    ax0.axhline(-13, color=INK, ls="--", lw=1, label="optimal return (-13)")
    ax0.set_ylim(-110, 0)
    ax0.set_xlabel("episode")
    ax0.set_ylabel("return during training (10 seeds, smoothed)")
    ax0.legend(fontsize=8)
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 2, 4)
    for ax, name in [(ax1, "Q-learning"), (ax2, "SARSA")]:
        Q = runs[name].Q
        pol = rl.greedy_from_q(m, Q)
        path = rl.rollout(m, pol, g.start())
        draw_grid(ax, g, None, None, path, title=f"{name}, greedy path after 500 episodes: {len(path) - 1} moves")
    fig.tight_layout()
    fig.savefig(OUT / "11_cliff_learning.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    policies_figure()
    learning_figure()
