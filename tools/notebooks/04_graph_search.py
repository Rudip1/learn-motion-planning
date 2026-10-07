# %% [markdown]
# # 4 · Graph search
#
# One template — best-first search — and the order in which it expands nodes: breadth-first, Dijkstra, A*,
# weighted A*, greedy. On grids and on visibility graphs.

# %%
# Installs the module when it is missing (e.g. on Colab); a local `pip install -e .` is used as is.
import importlib.util

if importlib.util.find_spec("motion_planning") is None:
    %pip install -q git+https://github.com/Rudip1/learn-motion-planning

# %% [markdown]
# ## Recap
#
# Theory: [`1_theory/04_graph_search.md`](../../1_theory/04_graph_search.md).
#
# - Relax edges, $g(v) \leftarrow \min(g(v), g(u) + c(u, v))$ (4.1); expand the open node with the smallest key.
# - Keys: BFS = discovery order (4.2), Dijkstra $f = g$ (4.3), A* $f = g + h$ (4.4), weighted A* $f = g + w h$ and
#   greedy $f = h$ (4.7).
# - A consistent heuristic (4.5) makes A* optimal while closing each node once (4.6); weighted A* costs at most
#   $w C^*$ (4.8).
# - On 8-connected grids the octile distance is the exact empty-grid cost; Manhattan overestimates diagonals.
# - The shortest path among polygons runs along the visibility graph.

# %%
import heapq

import matplotlib.pyplot as plt
import numpy as np

import motion_planning as mp
from motion_planning import maps
from motion_planning.plotting import INK, INK_MUTED, OBSTACLE, SERIES, show_grid, use_style

use_style()
rng = np.random.default_rng(4)
A = mp.SearchAlgorithm

# %% [markdown]
# ## A* on a map

# %%
grid = maps.rooms()
s, t = grid.world_to_cell([2.0, 2.8]), grid.world_to_cell([8.0, 7.0])
r = mp.grid_search(grid, s, t)
print(f"found={r.found}, cost={r.cost:.3f} m, expanded {len(r.expanded)} of {grid.width * grid.height} cells")

# %% [markdown]
# ### ✏️ Exercise 1 — Dijkstra with a binary heap
#
# Implement the template of section 4.3 for $f = g$ on a graph given as `adj[u] = [(v, cost), ...]`. Use
# `heapq` with lazy deletion. Return `(cost, path)`, or `(inf, [])` if `t` is unreachable.

# %% [solution]
def dijkstra(adj, s, t):
    g = {s: 0.0}
    parent = {s: None}
    closed = set()
    heap = [(0.0, s)]
    while heap:
        gu, u = heapq.heappop(heap)
        if u in closed or gu > g[u]:
            continue
        closed.add(u)
        if u == t:
            path = [t]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return gu, path[::-1]
        for v, c in adj[u]:
            if v not in closed and gu + c < g.get(v, np.inf):
                g[v] = gu + c
                parent[v] = u
                heapq.heappush(heap, (g[v], v))
    return np.inf, []

# %% [exercise]
def dijkstra(adj, s, t):
    ...  # ✏️ section 4.3 with f = g

# %%
for _ in range(50):
    n = 15
    adj = [[(int(v), float(rng.uniform(0, 5))) for v in rng.choice(n, 3, replace=False) if v != u] for u in range(n)]
    ref = mp.best_first_search(n, 0, n - 1, lambda u: adj[u], lambda u: 0.0, A.Dijkstra)
    cost, path = dijkstra(adj, 0, n - 1)
    assert np.isclose(cost, ref.cost) if ref.found else cost == np.inf
print("✓ Dijkstra")

# %% [markdown]
# ### 🔨 Break it — stop when the goal is discovered
#
# A tempting shortcut: return as soon as the goal is *pushed*. On this graph the direct edge $0 \to 3$ (cost 10)
# discovers the goal first, but the path $0 \to 1 \to 2 \to 3$ costs 3.

# %%
adj_trap = [[(3, 10.0), (1, 1.0)], [(2, 1.0)], [(3, 1.0)], []]


def dijkstra_stop_on_discovery(adj, s, t):
    g, heap = {s: 0.0}, [(0.0, s)]
    while heap:
        gu, u = heapq.heappop(heap)
        for v, c in adj[u]:
            if gu + c < g.get(v, np.inf):
                g[v] = gu + c
                if v == t:
                    return g[v]  # wrong: the goal is not closed yet
                heapq.heappush(heap, (g[v], v))
    return np.inf


print("stop on discovery:", dijkstra_stop_on_discovery(adj_trap, 0, 3), "  stop on expansion:", dijkstra(adj_trap, 0, 3)[0])

# %% [markdown]
# ## Heuristics
#
# ### ✏️ Exercise 2 — the octile distance and consistency
#
# Implement the octile heuristic in cell units, then check consistency (4.5), $h(u) \le c(u, v) + h(v)$, on every
# edge of an empty 8-connected grid towards the goal. Count the violations for octile and Manhattan.

# %% [solution]
def octile(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def consistency_violations(h, goal, size=12):
    bad = 0
    for x in range(size):
        for y in range(size):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (dx, dy) != (0, 0) and 0 <= x + dx < size and 0 <= y + dy < size:
                        c = np.hypot(dx, dy)
                        bad += h((x, y), goal) > c + h((x + dx, y + dy), goal) + 1e-12
    return bad

# %% [exercise]
def octile(a, b):
    ...  # ✏️


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def consistency_violations(h, goal, size=12):
    ...  # ✏️ count edges with h(u) > c(u, v) + h(v)

# %%
assert np.isclose(octile((0, 0), (3, 5)), mp.grid_heuristic(mp.Heuristic.Octile, (0, 0), (3, 5)))
assert consistency_violations(octile, (7, 3)) == 0
print("✓ octile is consistent; Manhattan violations:", consistency_violations(manhattan, (7, 3)))

# %% [markdown]
# ### 🔨 Break it — Manhattan on an 8-connected grid
#
# Over random maps, how often does A* with the inadmissible Manhattan heuristic return a longer path than
# Dijkstra, and by how much? It does expand fewer nodes — that is the temptation.

# %%
worse, ratios, saved = 0, [], []
for _ in range(60):
    occ = (rng.random((30, 40)) < 0.3).astype(np.uint8)
    occ[0, 0] = occ[-1, -1] = 0
    g = mp.OccupancyGrid(occ)
    opt = mp.grid_search(g, (0, 0), (39, 29), mp.GridSearchOptions(A.Dijkstra))
    if not opt.found:
        continue
    man = mp.grid_search(g, (0, 0), (39, 29), mp.GridSearchOptions(A.AStar, heuristic=mp.Heuristic.Manhattan))
    ratios.append(man.cost / opt.cost)
    saved.append(len(man.expanded) / len(opt.expanded))
    worse += man.cost > opt.cost + 1e-9
print(f"{worse} of {len(ratios)} solvable maps give a longer path; worst ratio {max(ratios):.3f}; "
      f"expansions {100 * np.mean(saved):.0f}% of Dijkstra's")

# %% [markdown]
# ## Weighted A*
#
# Sweep $w$ on the rooms map: the cost stays below the bound $w C^*$ (4.8) and the expansions drop.

# %%
ws = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
res = [mp.grid_search(grid, s, t, mp.GridSearchOptions(A.WeightedAStar, weight=w)) for w in ws]
best = res[0].cost
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.plot(ws, [r.cost / best for r in res], "o-", label="cost / optimal")
ax1.plot(ws, ws, "--", color=INK_MUTED, label="bound w, eq. (4.8)")
ax1.set_xlabel("w")
ax1.legend()
ax2.plot(ws, [len(r.expanded) for r in res], "o-", color=SERIES[1])
ax2.set_xlabel("w")
ax2.set_ylabel("expanded cells")
ax2.set_yscale("log")
plt.show()
assert all(r.cost <= w * best + 1e-9 for r, w in zip(res, ws))

# %% [markdown]
# ### ✏️ Exercise 3 — a clearance cost
#
# Shortest paths graze walls. Build a cell cost (4.9) $\kappa(c) = k \max(0, 1 - D(c)/d_0)$ from the distance
# transform $D$, with $k = 5$ and $d_0 = 0.6$ m, and plan with it. The minimum clearance along the path should rise
# above 0.3 m while the length grows only a little.

# %% [solution]
D = mp.distance_transform(grid)
kappa = 5.0 * np.maximum(0.0, 1.0 - D / 0.6)

# %% [exercise]
D = mp.distance_transform(grid)
kappa = ...  # ✏️ eq. (4.9) cell cost

# %%
safe = mp.grid_search(grid, s, t, mp.GridSearchOptions(), kappa)
length = lambda p: sum(np.hypot(*(p[i + 1] - p[i])) for i in range(len(p) - 1)) * grid.resolution  # noqa: E731
clear = lambda p: min(D[y, x] for x, y in p)  # noqa: E731
print(f"plain: length {length(r.path):.2f} m, clearance {clear(r.path):.2f} m")
print(f"safe : length {length(safe.path):.2f} m, clearance {clear(safe.path):.2f} m")
assert clear(safe.path) > 0.3 and length(safe.path) < 1.2 * length(r.path)

fig, ax = plt.subplots(figsize=(6, 5))
show_grid(ax, grid, field=kappa, cmap="Oranges", vmin=0, vmax=6)
for p, c, lab in [(r.path, SERIES[0], "shortest"), (safe.path, SERIES[2], "with clearance cost")]:
    xy = np.array([grid.cell_center(tuple(q)) for q in p])
    ax.plot(xy[:, 0], xy[:, 1], color=c, lw=2, label=lab)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
plt.show()

# %% [markdown]
# ## Visibility graphs
#
# Among polygons the shortest path bends only at obstacle vertices. Build the visibility graph and search it.

# %%
polygons = [
    np.array([[1.5, 1.0], [3.5, 0.8], [3.8, 2.6], [2.0, 3.0]]),
    np.array([[5.0, 2.0], [6.5, 2.2], [6.0, 4.5], [4.6, 4.0]]),
    np.array([[2.5, 4.0], [3.8, 4.3], [3.2, 5.6]]),
    np.array([[7.0, 0.5], [8.4, 0.9], [8.0, 2.6], [7.4, 2.0]]),
]
start, goal = np.array([0.5, 0.5]), np.array([9.0, 5.0])
vg = mp.visibility_graph(polygons, [start, goal])
vr = mp.graph_search(vg, 0, 1)
print(f"{len(vg)} nodes, {vg.num_edges // 2} edges; shortest path {vr.cost:.3f} m through nodes {vr.path}")

# %% [markdown]
# ### ✏️ Exercise 4 — what the grid costs
#
# Rasterise the polygons on a 0.05 m grid (a cell is occupied if its centre is inside a polygon: use
# `mp.point_in_polygon`), run A* with 4- and 8-connectivity between the cells of `start` and `goal`, and return
# the two costs. Both are longer than the visibility-graph path: 4-connected paths by up to a factor $\sqrt2$,
# 8-connected ones by up to $\sqrt{4 - 2\sqrt2} \approx 1.082$ (the largest ratio of octile to Euclidean length,
# at $22.5^\circ$).

# %% [solution]
def grid_costs(polygons, start, goal, h=0.05, size=(9.6, 6.0)):
    w, hgt = int(size[0] / h), int(size[1] / h)
    occ = np.zeros((hgt, w), np.uint8)
    for y in range(hgt):
        for x in range(w):
            c = ((x + 0.5) * h, (y + 0.5) * h)
            occ[y, x] = any(mp.point_in_polygon(c, p) for p in polygons)
    g = mp.OccupancyGrid(occ, h)
    out = []
    for conn, heur in [(mp.Connectivity.Four, mp.Heuristic.Manhattan), (mp.Connectivity.Eight, mp.Heuristic.Octile)]:
        rr = mp.grid_search(g, g.world_to_cell(start), g.world_to_cell(goal),
                            mp.GridSearchOptions(A.AStar, conn, heur))
        out.append(rr.cost)
    return tuple(out)

# %% [exercise]
def grid_costs(polygons, start, goal, h=0.05, size=(9.6, 6.0)):
    ...  # ✏️ return (cost_4, cost_8)

# %%
c4, c8 = grid_costs(polygons, start, goal)
print(f"visibility {vr.cost:.3f} m, 8-connected grid {c8:.3f} m ({c8 / vr.cost:.3f}x), "
      f"4-connected {c4:.3f} m ({c4 / vr.cost:.3f}x)")
assert vr.cost - 0.1 < c8 < c4 <= np.sqrt(2) * vr.cost + 0.1

fig, ax = plt.subplots(figsize=(8, 4.8))
P = vg.points
for i in range(len(vg)):
    for j, _ in vg.edges(i):
        if j > i:
            ax.plot(*P[[i, j]].T, color=INK_MUTED, lw=0.5, alpha=0.6)
for poly in polygons:
    ax.fill(*poly.T, color=OBSTACLE)
ax.plot(*P[vr.path].T, color=SERIES[1], lw=2.5)
ax.grid(False)
ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## What to remember
#
# - Every search here is the same loop; only the key changes. Stop when the goal is *expanded*.
# - Use the largest consistent heuristic you have (octile on 8-connected grids); it saves most of Dijkstra's work
#   at no cost in optimality.
# - Weighted A* is a dial: $w$ times fewer guarantees, often orders of magnitude fewer expansions.
# - Clearance belongs in the cost (4.9), not in post-processing.
# - Grids lengthen paths ($\le \sqrt2$ for 4-connected); visibility graphs give exact shortest paths among polygons
#   — touching the obstacles.
