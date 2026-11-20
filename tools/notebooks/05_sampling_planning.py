# %% [markdown]
# # 5 · Sampling-based planning
#
# PRM, RRT and RRT*: planners that only ask a collision checker two questions, and what their probabilistic
# guarantees mean in practice.

# %%
# Installs the module when it is missing (e.g. on Colab); a local `pip install -e .` is used as is.
import importlib.util

if importlib.util.find_spec("motion_planning") is None:
    %pip install -q git+https://github.com/Rudip1/learn-motion-planning

# %% [markdown]
# ## Recap
#
# Theory: [`1_theory/05_sampling_planning.md`](../../1_theory/05_sampling_planning.md).
#
# - Two oracles: `state_valid(q)` and `motion_valid(a, b)`; trees grow by `steer` (5.1), at most $\eta$ per step.
# - PRM: sample, connect neighbours within a radius, search the roadmap — multi-query.
# - RRT: extend the nearest node towards a random sample; the Voronoi bias pulls it into unexplored space.
#   Probabilistically complete (5.2), not optimal.
# - RRT*: choose the cheapest parent (5.5) and rewire neighbours (5.6) inside the radius
#   $\gamma(\log n/n)^{1/d}$ (5.3); asymptotically optimal for $\gamma$ above (5.4).

# %%
import time

import matplotlib.pyplot as plt
import numpy as np

import motion_planning as mp
from motion_planning import maps
from motion_planning.plotting import INK, INK_MUTED, OBSTACLE, SERIES, show_grid, use_style

use_style()

polygons = [
    np.array([[1.5, 1.0], [3.5, 0.8], [3.8, 2.6], [2.0, 3.0]]),
    np.array([[5.0, 2.0], [6.5, 2.2], [6.0, 4.5], [4.6, 4.0]]),
    np.array([[2.5, 4.0], [3.8, 4.3], [3.2, 5.6]]),
    np.array([[7.0, 0.5], [8.4, 0.9], [8.0, 2.6], [7.4, 2.0]]),
]
lower, upper = np.array([0.0, 0.0]), np.array([9.6, 6.0])
start, goal = np.array([0.5, 0.5]), np.array([9.0, 5.0])
problem = mp.problem_from_polygons(polygons, lower, upper, start, goal)
optimum = mp.graph_search(mp.visibility_graph(polygons, [start, goal]), 0, 1).cost
print(f"exact optimum (visibility graph, chapter 4): {optimum:.3f} m")


def draw_world(ax):
    for poly in polygons:
        ax.fill(*poly.T, color=OBSTACLE)
    ax.plot(*start, "o", color=INK, ms=7, zorder=5)
    ax.plot(*goal, "*", color=INK, ms=12, zorder=5)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_aspect("equal")
    ax.grid(False)


def draw_tree(ax, nodes, parent, color=INK_MUTED):
    nodes = np.asarray(nodes)
    for i, p in enumerate(parent):
        if p >= 0:
            ax.plot(*nodes[[p, i]].T, color=color, lw=0.6, alpha=0.7)


# %% [markdown]
# ## RRT
#
# ### ✏️ Exercise 1 — steer
#
# Implement eq. (5.1).

# %% [solution]
def steer(q_near, q_rand, eta):
    d = np.asarray(q_rand, float) - q_near
    n = np.linalg.norm(d)
    return np.asarray(q_rand, float) if n <= eta else q_near + eta * d / n

# %% [exercise]
def steer(q_near, q_rand, eta):
    ...  # ✏️ eq. (5.1)

# %%
assert np.allclose(steer(np.zeros(2), [3.0, 4.0], 1.0), [0.6, 0.8])
assert np.allclose(steer(np.zeros(2), [0.3, 0.4], 1.0), [0.3, 0.4])
print("✓ steer")

# %% [markdown]
# ### ✏️ Exercise 2 — RRT in Python
#
# Implement the algorithm of section 5.3 using only `problem.state_valid` and `problem.motion_valid`. Return
# `(nodes, parent, path)`, where `path` is the list of points from start to goal (empty if none was found within
# `iterations`).

# %% [solution]
def rrt(problem, iterations=3000, eta=0.5, goal_bias=0.05, seed=0):
    rng = np.random.default_rng(seed)
    nodes, parent = [np.asarray(problem.start, float)], [-1]
    for _ in range(iterations):
        q_rand = problem.goal if rng.random() < goal_bias else rng.uniform(problem.lower, problem.upper)
        i_near = int(np.argmin([np.sum((q - q_rand) ** 2) for q in nodes]))
        q_new = steer(nodes[i_near], q_rand, eta)
        if not (problem.state_valid(q_new) and problem.motion_valid(nodes[i_near], q_new)):
            continue
        nodes.append(q_new)
        parent.append(i_near)
        if np.linalg.norm(q_new - problem.goal) <= eta and problem.motion_valid(q_new, problem.goal):
            path, i = [problem.goal], len(nodes) - 1
            while i >= 0:
                path.append(nodes[i])
                i = parent[i]
            return nodes, parent, path[::-1]
    return nodes, parent, []

# %% [exercise]
def rrt(problem, iterations=3000, eta=0.5, goal_bias=0.05, seed=0):
    ...  # ✏️ section 5.3

# %%
nodes, parent, path = rrt(problem)
assert len(path) > 1 and np.allclose(path[0], start) and np.allclose(path[-1], goal)
assert all(problem.motion_valid(a, b) for a, b in zip(path[:-1], path[1:]))
length = sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:]))
assert length >= optimum - 1e-9
print(f"✓ RRT: {len(nodes)} nodes, path {length:.2f} m = {length / optimum:.3f} × optimum")

fig, ax = plt.subplots(figsize=(7, 4.4))
draw_world(ax)
draw_tree(ax, nodes, parent)
ax.plot(*np.array(path).T, color=SERIES[1], lw=2.5)
ax.set_title("Your RRT")
plt.show()

# %% [markdown]
# **Voronoi bias.** In an empty square, the first nodes of an RRT shoot outwards: frontier nodes own the large
# Voronoi cells, so they are the ones the samples pick.

# %%
empty = mp.PlanningProblem2D()
empty.lower, empty.upper = np.zeros(2), np.full(2, 10.0)
empty.state_valid = lambda q: True
empty.motion_valid = lambda a, b: True
empty.start, empty.goal = np.full(2, 5.0), np.full(2, 100.0)  # unreachable goal: pure exploration
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, n in zip(axes, [30, 150, 1000]):
    t = mp.rrt(empty, mp.RrtOptions(max_iterations=n, step=0.6, goal_bias=0.0, seed=1))
    draw_tree(ax, t.nodes, t.parent, SERIES[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title(f"{n} iterations")
plt.show()

# %% [markdown]
# ## RRT*
#
# ### ✏️ Exercise 3 — the radius
#
# Write the radius of eq. (5.3) (without the cap at $\eta$) and the PRM* threshold of eq. (5.4) for $d = 2$.

# %% [solution]
def radius(n, gamma):
    return gamma * np.sqrt(np.log(n) / n)


def gamma_star(free_area):
    return 2.0 * np.sqrt(1.5 * free_area / np.pi)

# %% [exercise]
def radius(n, gamma):
    ...  # ✏️ eq. (5.3), d = 2


def gamma_star(free_area):
    ...  # ✏️ eq. (5.4), PRM* constant, d = 2

# %%
for n in [10, 100, 5000]:
    assert np.isclose(radius(n, 3.0), mp.rewiring_radius(n, 3.0))
assert np.isclose(gamma_star(57.6), mp.optimal_gamma(57.6))
g = gamma_star(np.prod(upper - lower))
print(f"✓ gamma* = {g:.2f}; radius at n = 100, 1000, 10000: "
      + ", ".join(f"{radius(n, g):.2f}" for n in [100, 1000, 10000]) + " m")

# %% [markdown]
# RRT against RRT* over 20 seeds and a growing budget. RRT* approaches the optimum; RRT stalls.

# %%
budgets = [500, 1000, 2000, 4000, 8000]
fig, ax = plt.subplots(figsize=(7, 4))
for name, planner, c in [("RRT", mp.rrt, SERIES[0]), ("RRT*", mp.rrt_star, SERIES[1])]:
    med = []
    for b in budgets:
        costs = [planner(problem, mp.RrtOptions(max_iterations=b, stop_at_first_solution=False, seed=s)).path_cost
                 for s in range(20)]
        med.append(np.median(costs) / optimum)
    ax.plot(budgets, med, "o-", color=c, label=name)
    print(name, " ".join(f"{m:.3f}" for m in med))
ax.axhline(1.0, color=INK, ls="--", lw=1)
ax.set_xscale("log")
ax.set_xlabel("iterations")
ax.set_ylabel("median cost / optimum")
ax.legend()
plt.show()

# %% [markdown]
# ### 🔨 Break it — the wrong radius
#
# Fix $\gamma$ far below the threshold: RRT* can no longer rewire and is no better than RRT. Far above it, every
# iteration checks a large part of the tree: same quality, much slower.

# %%
for name, gamma in [("tiny γ", 0.05), ("γ* (default)", 0.0), ("huge γ", 200.0)]:
    t0 = time.perf_counter()
    costs = [mp.rrt_star(problem, mp.RrtOptions(max_iterations=3000, step=10.0 if gamma > 100 else 0.5,
                                                gamma=gamma, seed=s)).path_cost for s in range(5)]
    dt = (time.perf_counter() - t0) / 5
    print(f"{name:>13}: median cost {np.median(costs) / optimum:.3f} × optimum, {1e3 * dt:6.1f} ms per run")

# %% [markdown]
# ## PRM and connectivity
#
# ### ✏️ Exercise 4 — connected components
#
# A roadmap answers a query only if start and goal lie in the same connected component. Count the components of
# a `mp.Graph` (use `graph.edges(i)`, a list of `(j, cost)`), with a breadth-first search.

# %% [solution]
def components(graph):
    label = [-1] * len(graph)
    count = 0
    for s in range(len(graph)):
        if label[s] >= 0:
            continue
        label[s] = count
        queue = [s]
        while queue:
            u = queue.pop()
            for v, _ in graph.edges(u):
                if label[v] < 0:
                    label[v] = count
                    queue.append(v)
        count += 1
    return count, label

# %% [exercise]
def components(graph):
    ...  # ✏️ return (count, label per node)

# %%
for r in [0.2, 0.4, 0.8, -1.0]:
    res = mp.prm(problem, mp.PrmOptions(num_samples=300, connection_radius=r, seed=2))
    k, label = components(res.roadmap)
    name = "PRM* radius" if r < 0 else f"r = {r} m"
    print(f"{name:>12}: {k:3d} components, start and goal connected: {label[0] == label[1]}, "
          f"query found: {res.query.found}, collision checks {res.collision_checks}")
    assert (label[0] == label[1]) == res.query.found

# %% [markdown]
# ### 🔨 Break it — narrow passages
#
# A wall with one gap, start and goal on either side and off its axis. With the same budget, the success rate
# of RRT collapses as the gap narrows: a uniform sample lands in the gap with probability proportional to its
# width (5.2).

# %%
widths = [0.8, 0.4, 0.2, 0.1, 0.06]
s_xy, g_xy = [1.0, 0.5], [5.0, 3.5]
for w in widths:
    g = maps.narrow_passage(resolution=0.02, gap=w)
    p = mp.problem_from_grid(g, s_xy, g_xy)
    found = [mp.rrt(p, mp.RrtOptions(max_iterations=600, step=0.3, seed=s)).found for s in range(40)]
    print(f"gap {w:4.2f} m: RRT success in 600 iterations {100 * np.mean(found):5.1f} %")

g = maps.narrow_passage(resolution=0.02, gap=0.1)
t = mp.rrt(mp.problem_from_grid(g, s_xy, g_xy), mp.RrtOptions(max_iterations=600, step=0.3, seed=0))
fig, ax = plt.subplots(figsize=(6, 3.5))
show_grid(ax, g)
draw_tree(ax, t.nodes, t.parent, SERIES[0])
ax.plot(*s_xy, "o", color=INK)
ax.plot(*g_xy, "*", color=INK, ms=12)
ax.set_title(f"0.1 m gap, 600 iterations: found = {t.found}")
plt.show()

# %% [markdown]
# ### Reference: OMPL
#
# If OMPL is installed (`pip install ompl`), run its RRT* on the same problem for one second. The test
# `python/tests/test_ompl_reference.py` does this automatically.

# %%
try:
    from ompl import base as ob, geometric as og, util as ou

    ou.setLogLevel(ou.LOG_ERROR)
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    for i in range(2):
        bounds.setLow(i, lower[i])
        bounds.setHigh(i, upper[i])
    space.setBounds(bounds)
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(lambda s: problem.state_valid(np.array([s[0], s[1]])))
    si = ss.getSpaceInformation()
    si.setStateValidityCheckingResolution(0.001)
    a, b = space.allocState(), space.allocState()
    a[0], a[1] = start
    b[0], b[1] = goal
    ss.setStartAndGoalStates(a, b, 0.05)
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    ss.setPlanner(og.RRTstar(si))
    ss.solve(1.0)
    print(f"OMPL RRT*: {ss.getSolutionPath().length() / optimum:.3f} × optimum (goal tolerance 0.05 m)")
except ImportError:
    print("OMPL not installed; skipping the reference run")

# %% [markdown]
# ## What to remember
#
# - Sampling planners need only `state_valid` and `motion_valid`; collision checks are the cost to count.
# - "Not found" is not "does not exist": completeness is probabilistic and narrow passages make it slow.
# - RRT explores fast and returns poor paths; RRT* rewires within $\gamma(\log n / n)^{1/d}$ and converges to the
#   optimum.
# - PRM builds a reusable roadmap; whether a query succeeds is a question of connected components.
# - Validate against something exact (the visibility graph) or an established library (OMPL), over many seeds.
