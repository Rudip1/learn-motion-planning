# %% [markdown]
# # 8 · Local planning: the Dynamic Window Approach
#
# Search velocity space within what the motors can reach, keep only commands the robot can still brake from,
# score them — and see why this needs a global plan.

# %%
# Installs the module when it is missing (e.g. on Colab); a local `pip install -e .` is used as is.
import importlib.util

if importlib.util.find_spec("motion_planning") is None:
    %pip install -q git+https://github.com/Rudip1/learn-motion-planning

# %% [markdown]
# ## Recap
#
# Theory: [`1_theory/08_local_planning.md`](../../1_theory/08_local_planning.md).
#
# - Candidates are commands $(v, \omega)$; each is an exact arc over the horizon (1.16).
# - Search only the dynamic window $V_s \cap V_d$ (8.1)–(8.2), keep admissible commands
#   $v^2 \le 2\dot v_b\, d(v, \omega)$ (8.3), maximise heading + free distance + velocity (8.4).
# - Alone, DWA falls into local minima; chasing a carrot on a global path (8.5) it does not.

# %%
import matplotlib.pyplot as plt
import numpy as np

import motion_planning as mp
from motion_planning import maps
from motion_planning.plotting import INK, SERIES, show_grid, use_style

use_style()
cfg = mp.DwaConfig()
print({k: getattr(cfg, k) for k in ["max_speed", "max_yaw_rate", "max_accel", "max_yaw_accel", "control_period",
                                     "horizon", "robot_radius"]})

# %% [markdown]
# ### ✏️ Exercise 1 — the dynamic window
#
# Implement eqs. (8.1)–(8.2) (use `cfg.max_accel` for both acceleration and braking). Return
# `(v_min, v_max, w_min, w_max)`.

# %% [solution]
def window(v, w, cfg):
    dv, dw = cfg.max_accel * cfg.control_period, cfg.max_yaw_accel * cfg.control_period
    return (max(cfg.min_speed, v - dv), min(cfg.max_speed, v + dv),
            max(-cfg.max_yaw_rate, w - dw), min(cfg.max_yaw_rate, w + dw))

# %% [exercise]
def window(v, w, cfg):
    ...  # ✏️ eqs. (8.1)-(8.2)

# %%
assert np.allclose(window(0.2, 0.0, cfg), (0.14, 0.26, -0.3, 0.3))
rng = np.random.default_rng(8)
for _ in range(200):
    v, w = rng.uniform(0, 0.6), rng.uniform(-1.5, 1.5)
    ref = mp.dynamic_window([v, w], cfg)
    assert np.allclose(window(v, w, cfg), (ref.v_min, ref.v_max, ref.w_min, ref.w_max))
print("✓ dynamic window")

# %% [markdown]
# ### ✏️ Exercise 2 — admissibility
#
# Implement eq. (8.3) with the margin used by the library: a candidate with free distance `d` (possibly `inf`) and
# speed `v` is admissible iff $v^2 \le 2 \dot v_b \max(0, d - m)$ with $m = 2 v \Delta t_\text{sim} + h$
# ($h$ = grid resolution). Check it against every candidate of a real decision near a wall.

# %% [solution]
def admissible(v, d, cfg, h):
    if np.isinf(d):
        return True
    m = 2 * abs(v) * cfg.sim_step + h
    return v * v <= 2 * cfg.max_accel * max(0.0, d - m)

# %% [exercise]
def admissible(v, d, cfg, h):
    ...  # ✏️ eq. (8.3) with the margin

# %%
trap = maps.trap(resolution=0.05)
D = mp.distance_transform(trap)
decision = mp.dwa_step(trap, D, np.array([3.45, 3.0, 0.0]), np.array([0.5, 0.3]), np.array([6.5, 3.0]), cfg)
assert all(admissible(c.v, c.free_distance, cfg, trap.resolution) == c.admissible for c in decision.candidates)
print(f"✓ admissibility: {sum(c.admissible for c in decision.candidates)} of {len(decision.candidates)} admissible; "
      f"chosen (v, ω) = {np.round(decision.command, 2)}")

# %% [markdown]
# ## DWA in the rooms map
#
# Open space and a goal in the same room: DWA alone is enough. The commands respect the window step by step.

# %%
rooms = maps.rooms(resolution=0.05)
run = mp.run_dwa(rooms, np.array([1.0, 5.0, 0.0]), np.array([4.0, 7.2]), cfg)
print("reached:", run.reached_goal, f"in {len(run.commands) * cfg.control_period:.1f} s")
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4))
show_grid(ax, rooms)
ax.plot(run.states[:, 0], run.states[:, 1], color=SERIES[0], lw=2)
ax.set_xlim(0, 5.2)
ax.set_ylim(3.8, 8)
t = np.arange(len(run.commands)) * cfg.control_period
ax2.plot(t, run.commands[:, 0], label="v [m/s]")
ax2.plot(t, run.commands[:, 1], label="ω [rad/s]")
ax2.set_xlabel("t [s]")
ax2.legend()
plt.show()
dv = np.abs(np.diff(np.r_[[[0, 0]], run.commands], axis=0))
assert dv[:, 0].max() <= cfg.max_accel * cfg.control_period + 1e-9
assert dv[:, 1].max() <= cfg.max_yaw_accel * cfg.control_period + 1e-9

# %% [markdown]
# ### 🔨 Break it — DWA alone in front of a trap
#
# The goal is behind a U. DWA heads straight in and stays.

# %%
start, goal = np.array([1.0, 3.0, 0.0]), np.array([6.5, 3.0])
alone = mp.run_dwa(trap, start, goal, cfg, max_time=40.0)
print(f"reached: {alone.reached_goal}, stuck: {alone.stuck}, final position {np.round(alone.states[-1, :2], 2)}")

# %% [markdown]
# ### ✏️ Exercise 3 — give it a global plan
#
# Plan with A* (chapter 4) on the trap map inflated by the robot radius plus 0.15 m (chapter 2), convert the cells
# to an (N, 2) array of points, and run DWA with that `global_path` and a carrot distance of 0.6 m.

# %% [solution]
inflated = mp.inflate(trap, cfg.robot_radius + 0.15)
plan = mp.grid_search(inflated, inflated.world_to_cell(start[:2]), inflated.world_to_cell(goal))
global_path = np.array([trap.cell_center(tuple(c)) for c in plan.path])
guided = mp.run_dwa(trap, start, goal, cfg, global_path, 0.6)

# %% [exercise]
global_path = ...  # ✏️ (N, 2) points of an A* path on the inflated map
guided = ...       # ✏️ mp.run_dwa(..., global_path, 0.6)

# %%
assert guided.reached_goal
print(f"✓ reached the goal in {len(guided.commands) * cfg.control_period:.1f} s")
fig, ax = plt.subplots(figsize=(6.5, 4.8))
show_grid(ax, trap)
ax.plot(global_path[:, 0], global_path[:, 1], "--", color=INK, lw=1, label="A* path")
ax.plot(alone.states[:, 0], alone.states[:, 1], color=SERIES[7], lw=2, label="DWA alone")
ax.plot(guided.states[:, 0], guided.states[:, 1], color=SERIES[0], lw=2, label="DWA + carrot")
ax.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### 🔨 Break it — the carrot too far ahead
#
# A long carrot jumps round the corners of the global path; the straight bearing to it cuts the corner and pins
# the robot against the arm of the U.

# %%
for carrot in [0.3, 0.6, 1.0, 1.5, 2.0]:
    r = mp.run_dwa(trap, start, goal, cfg, global_path, carrot, max_time=40.0)
    print(f"carrot {carrot:3.1f} m: reached {r.reached_goal!s:5}  stuck {r.stuck!s:5}  "
          f"time {len(r.commands) * cfg.control_period:5.1f} s")

# %% [markdown]
# ## The local planner's real job: obstacles the map did not show
#
# Plan the global path on the map as it was, then put a box near that path in the *real* world, which is what DWA
# sees. A box that only clips the path is driven round. A box squarely on it is another local minimum: the
# carrot sits behind it. Then the global planner must replan on the updated map — the division of labour of
# section 8.4.

# %%
s2, g2 = np.array([1.0, 2.8, 0.0]), np.array([8.0, 2.8])


def global_plan(grid):
    infl = mp.inflate(grid, cfg.robot_radius + 0.15)
    plan = mp.grid_search(infl, infl.world_to_cell(s2[:2]), infl.world_to_cell(g2))
    return np.array([grid.cell_center(tuple(c)) for c in plan.path])


def with_box(x0, x1, y0, y1):
    w = rooms.to_array().copy()
    X, Y = np.meshgrid((np.arange(rooms.width) + 0.5) * 0.05, (np.arange(rooms.height) + 0.5) * 0.05)
    w[(X > x0) & (X < x1) & (Y > y0) & (Y < y1)] = 1
    return mp.OccupancyGrid(w, 0.05)


old_path = global_plan(rooms)
clipping, blocking = with_box(6.5, 6.8, 2.9, 3.4), with_box(6.5, 6.8, 2.6, 3.0)
runs = {
    "box clips the path, old plan": (clipping, old_path),
    "box blocks the path, old plan": (blocking, old_path),
    "box blocks the path, replanned": (blocking, global_plan(blocking)),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, (name, (world, path)) in zip(axes, runs.items()):
    r = mp.run_dwa(world, s2, g2, cfg, path, 0.6, max_time=60.0)
    show_grid(ax, world)
    ax.plot(path[:, 0], path[:, 1], "--", color=INK, lw=1)
    ax.plot(r.states[:, 0], r.states[:, 1], color=SERIES[0] if r.reached_goal else SERIES[7], lw=2)
    ax.set_xlim(4.5, 8.5)
    ax.set_ylim(1.0, 4.0)
    ax.set_title(f"{name}: reached = {r.reached_goal}")
    print(f"{name:>32}: reached {r.reached_goal}")
plt.show()

# %% [markdown]
# ## What to remember
#
# - DWA searches commands, not paths: the window (8.2) holds what the motors can reach in one period.
# - Admissible = can still brake before the first collision on the arc (8.3); quantised distances need a margin.
# - Score the heading where the rollout comes closest to the goal, and the obstacle term as free distance along
#   the arc.
# - Local planners fall into local minima; give them a global path and a short carrot (8.5).
# - Their job is reacting to what the map did not contain, within the robot's dynamics; what blocks the global
#   path needs a replan.
