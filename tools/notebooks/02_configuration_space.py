# %% [markdown]
# # 2 · Configuration space and occupancy grids
#
# Grids, distance transforms, inflation and collision checks — the machinery every planner in the following
# chapters stands on.

# %%
# Installs the module when it is missing (e.g. on Colab); a local `pip install -e .` is used as is.
import importlib.util

if importlib.util.find_spec("motion_planning") is None:
    %pip install -q git+https://github.com/Rudip1/learn-motion-planning

# %% [markdown]
# ## Recap
#
# Theory: [`1_theory/02_configuration_space.md`](../../1_theory/02_configuration_space.md).
#
# - $\mathcal{C}_\text{obs}$ is the set of configurations where the robot touches an obstacle (2.2). For a
#   translating robot it is the Minkowski sum $\mathcal{O} \oplus (-\mathcal{A}_0)$ (2.3); for a disc, the
#   obstacles grown by the radius (2.4).
# - A grid cell is $(\lfloor (x - x_0)/h \rfloor, \lfloor (y - y_0)/h \rfloor)$ (2.5); arrays are indexed
#   `[y, x]`; outside the map is occupied.
# - Brushfire (2.6) gives $L_1$ / $L_\infty$ distances; the Felzenszwalb–Huttenlocher transform (2.7)–(2.9) gives
#   exact Euclidean distances, also in linear time.
# - Inflation is a threshold on the distance transform (2.10); the disc test (2.11) is conservative by $\sqrt2 h$.
# - Segments are checked by exact cell traversal, footprints with the separating-axis test (2.12).

# %%
from collections import deque
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import motion_planning as mp
from motion_planning import maps
from motion_planning.plotting import INK, SERIES, show_grid, use_style

use_style()
rng = np.random.default_rng(2)

# %% [markdown]
# ## A grid and its distance transform

# %%
grid = maps.rooms()
print(f"{grid.width} x {grid.height} cells of {grid.resolution} m, extent {grid.extent}")
edt = mp.distance_transform(grid)

fig, ax = plt.subplots(figsize=(6.5, 5))
show_grid(ax, grid, field=edt, cmap="Blues_r", colorbar="distance to nearest obstacle [m]")
ax.set_title("Exact Euclidean distance transform")
plt.show()

# %% [markdown]
# ### ✏️ Exercise 1 — brushfire
#
# Implement eq. (2.6) with a breadth-first queue: return an array shaped like `occ` (rows = $y$) holding the
# number of 4-connected steps from each cell to the nearest occupied cell. The assert compares with the C++
# implementation on random maps.

# %% [solution]
def brushfire4(occ):
    h, w = occ.shape
    d = np.full((h, w), np.inf)
    queue = deque()
    for y, x in zip(*np.nonzero(occ)):
        d[y, x] = 0
        queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and d[ny, nx] == np.inf:
                d[ny, nx] = d[y, x] + 1
                queue.append((ny, nx))
    return d

# %% [exercise]
def brushfire4(occ):
    ...  # ✏️ eq. (2.6), 4-connected

# %%
for _ in range(10):
    occ = (rng.random((15, 22)) < 0.08).astype(np.uint8)
    assert np.array_equal(brushfire4(occ), mp.brushfire(mp.OccupancyGrid(occ), mp.Connectivity.Four))
print("✓ brushfire")

# %% [markdown]
# ### ✏️ Exercise 2 — the one-dimensional transform by brute force
#
# Implement eq. (2.8) directly, $d(p) = \min_q \big((p - q)^2 + f(q)\big)$, in $O(n^2)$. Compare with the
# linear-time lower-envelope version in C++, then time both.

# %% [solution]
def sdt_1d_brute(f):
    p = np.arange(len(f))[:, None]
    q = np.arange(len(f))[None, :]
    return np.min((p - q) ** 2 + np.asarray(f)[None, :], axis=1)

# %% [exercise]
def sdt_1d_brute(f):
    ...  # ✏️ eq. (2.8), O(n^2)

# %%
for n in [1, 2, 7, 40]:
    f = np.where(rng.random(n) < 0.3, np.inf, rng.uniform(0, 20, n))
    assert np.allclose(sdt_1d_brute(f), mp.squared_distance_transform_1d(f))
print("✓ 1-D transform")

for n in [1000, 4000]:
    f = np.where(rng.random(n) < 0.99, np.inf, 0.0)
    t0 = time.perf_counter(); sdt_1d_brute(f); t1 = time.perf_counter()
    mp.squared_distance_transform_1d(f); t2 = time.perf_counter()
    print(f"n = {n}: brute force {1e3 * (t1 - t0):7.2f} ms, lower envelope {1e3 * (t2 - t1):5.3f} ms")

# %% [markdown]
# The lower envelope itself, for a few samples: $d$ (thick) is the minimum of the parabolas rooted at the finite
# samples.

# %%
f = np.array([np.inf, 3.0, np.inf, np.inf, 0.0, np.inf, np.inf, 5.0, np.inf, 1.0, np.inf, np.inf])
d = mp.squared_distance_transform_1d(f)
xs = np.linspace(0, len(f) - 1, 400)
fig, ax = plt.subplots(figsize=(7, 3.5))
for q in np.nonzero(np.isfinite(f))[0]:
    ax.plot(xs, (xs - q) ** 2 + f[q], color=SERIES[0], lw=1, alpha=0.5)
    ax.plot(q, f[q], "o", color=SERIES[0])
ax.plot(np.arange(len(f)), d, "s-", color=SERIES[1], lw=3, label="d(p), eq. (2.8)")
ax.set_ylim(-0.5, 12)
ax.set_xlabel("p")
ax.legend()
ax.set_title("Lower envelope of parabolas")
plt.show()

# %% [markdown]
# ## Inflation
#
# ### ✏️ Exercise 3 — inflate from the transform
#
# Using only `mp.distance_transform`, return the occupancy array of the grid inflated by `radius` metres
# (eq. 2.10). Compare with `mp.inflate`.

# %% [solution]
def inflate_array(grid, radius):
    return (mp.distance_transform(grid) <= radius).astype(np.uint8)

# %% [exercise]
def inflate_array(grid, radius):
    ...  # ✏️ eq. (2.10)

# %%
for r in [0.0, 0.15, 0.35, 0.8]:
    assert np.array_equal(inflate_array(grid, r), mp.inflate(grid, r).to_array())
print("✓ inflation")

# %% [markdown]
# ### ✏️ Exercise 4 — which disc covers a rectangle?
#
# A rectangular robot extends `rear` behind its reference point, `front` ahead of it and `half_width` to each
# side. Return the radius of the smallest disc *centred on the reference point* that contains the robot
# (circumscribed) and of the largest such disc contained in it (inscribed).

# %% [solution]
def disc_radii(fp):
    circumscribed = np.hypot(max(fp.rear, fp.front), fp.half_width)
    inscribed = min(fp.rear, fp.front, fp.half_width)
    return circumscribed, inscribed

# %% [exercise]
def disc_radii(fp):
    ...  # ✏️ return (circumscribed, inscribed)

# %%
fp = mp.RectangleFootprint(rear=0.15, front=0.75, half_width=0.2)
assert np.allclose(disc_radii(fp), (np.hypot(0.75, 0.2), 0.15))
print("radii:", np.round(disc_radii(fp), 3))

# %% [markdown]
# ### 🔨 Break it — inflate by the wrong radius
#
# Plan a point robot on a grid inflated by the inscribed radius, and test the real rectangle at random free
# positions of that grid: many poses collide. With the circumscribed radius almost none collide — the few that
# do sit within a cell of the inflated boundary: inflation (2.10) is exact at cell *centres*, and a position
# anywhere in a cell can be up to $\sqrt2 h$ closer to an obstacle, the margin of eq. (2.11). Adding that margin
# makes the test safe. Either safe radius closes the passage, although the robot fits through it lengthwise.

# %%
g = maps.narrow_passage(resolution=0.05, gap=0.9)
r_out, r_in = disc_radii(fp)
for name, r in [("inscribed", r_in), ("circumscribed", r_out), ("circ. + √2 h", r_out + np.sqrt(2) * g.resolution)]:
    infl = mp.inflate(g, r)
    poses = []
    while len(poses) < 2000:
        q = np.r_[rng.uniform(0, 6), rng.uniform(0, 4), rng.uniform(-np.pi, np.pi)]
        if infl.point_free(q[:2]):
            poses.append(q)
    hits = sum(not mp.footprint_free(g, q, fp) for q in poses)
    gap_open = infl.point_free([3.0, 2.0])
    print(f"{name:>13} radius {r:.3f} m: {hits:4d} of 2000 'free' poses collide; passage open: {gap_open}")

# %% [markdown]
# Planning in $(x, y, \theta)$ with exact footprint checks resolves both problems: the passage is open at
# $\theta = 0$ and closed at $\theta = 90^\circ$.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, th in zip(axes, [0.0, np.pi / 2]):
    c = mp.configuration_space_slice(g, fp, th)
    show_grid(ax, c)
    q = np.array([1.5, 2.0, th])
    ax.add_patch(Polygon(fp.corners(q), closed=True, facecolor=SERIES[0], edgecolor=INK))
    ax.set_title(f"C-space slice at θ = {np.degrees(th):.0f}°")
plt.show()

# %% [markdown]
# ## Collision checking along segments
#
# ### ✏️ Exercise 5 — a segment check by sampling
#
# Write the naive check: sample the segment every `step` metres (including both ends) and return `False` if any
# sample lies in an occupied cell. It agrees with the exact `mp.segment_free` on most random segments of the
# rooms map with a fine step.

# %% [solution]
def segment_free_sampled(grid, a, b, step):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
    return all(grid.point_free(a + (b - a) * k / n) for k in range(n + 1))

# %% [exercise]
def segment_free_sampled(grid, a, b, step):
    ...  # ✏️

# %%
segments = [(rng.uniform([0, 0], [10, 8]), rng.uniform([0, 0], [10, 8])) for _ in range(500)]
agree = np.mean([segment_free_sampled(grid, a, b, 0.01) == mp.segment_free(grid, a, b) for a, b in segments])
assert agree > 0.98
print(f"✓ fine sampling agrees with the exact traversal on {100 * agree:.1f}% of segments")

# %% [markdown]
# ### 🔨 Break it — a coarse step and Bresenham
#
# With a step of 0.3 m, sampling jumps over the 0.2 m walls. Bresenham, run on cells, skips clipped corners.
# Count how many truly blocked segments each method calls free.

# %%
blocked = [(a, b) for a, b in segments if not mp.segment_free(grid, a, b)]
missed_sampling = sum(segment_free_sampled(grid, a, b, 0.3) for a, b in blocked)
missed_bresenham = sum(
    all(not grid.occupied(tuple(c)) for c in mp.bresenham(grid.world_to_cell(a), grid.world_to_cell(b)))
    for a, b in blocked
)
print(f"{len(blocked)} blocked segments: sampling at 0.3 m calls {missed_sampling} free, "
      f"Bresenham calls {missed_bresenham} free")

a, b = None, None
for a, b in blocked:
    if segment_free_sampled(grid, a, b, 0.3):
        break
fig, ax = plt.subplots(figsize=(6, 4.8))
show_grid(ax, grid)
n = int(np.ceil(np.linalg.norm(b - a) / 0.3))
pts = np.array([a + (b - a) * k / n for k in range(n + 1)])
ax.plot(*np.c_[a, b], color=SERIES[1], lw=1.5, label="segment through a wall")
ax.plot(pts[:, 0], pts[:, 1], "o", color=SERIES[0], ms=4, label="samples every 0.3 m (all free)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.set_title("Sampling misses thin walls")
plt.show()

# %% [markdown]
# ### 🔨 Break it — resolution decides the answer
#
# A 0.62 m gap and a disc robot of radius 0.33 m: the robot does not fit ($2 \cdot 0.33 > 0.62$). Rasterising
# by cell centres shaves up to $h/2$ off each wall, so on some grids the inflated passage is *open*. Print the
# width of free space left in the passage after inflation, for several resolutions.

# %%
for h in [0.01, 0.02, 0.05, 0.1, 0.2]:
    gh = maps.narrow_passage(resolution=h, gap=0.62, center=2.03)
    inflated = mp.inflate(gh, 0.33).to_array()
    wall_columns = np.nonzero(gh.to_array()[5])[0][1:-1]
    width = min((inflated[:, c] == 0).sum() for c in wall_columns) * h
    print(f"h = {h:4} m: free width through the passage {width:.2f} m  ->  {'OPEN (wrong)' if width else 'closed'}")

# %% [markdown]
# ## What to remember
#
# - Plan for a point in $\mathcal{C}$: grow the obstacles (Minkowski sum), do not shrink the robot.
# - Compute the exact Euclidean distance transform once; inflation, clearance costs and disc tests all read it.
# - Index arrays `[y, x]`, treat the outside of the map as occupied.
# - Check motions with exact cell traversal, never with a fixed sampling step or Bresenham.
# - A disc approximation of a long robot is either unsafe (inscribed) or over-cautious (circumscribed); plan in
#   $(x, y, \theta)$ when the difference matters.
