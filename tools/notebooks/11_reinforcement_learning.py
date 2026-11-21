# %% [markdown]
# # 11 · Reinforcement learning basics
#
# MDPs on grid worlds: solve them with value and policy iteration when the model is known, learn them with
# Q-learning and SARSA when it is not — all in C++, inspected from Python.

# %%
# Installs the module when it is missing (e.g. on Colab); a local `pip install -e .` is used as is.
import importlib.util

if importlib.util.find_spec("motion_planning") is None:
    %pip install -q git+https://github.com/Rudip1/learn-motion-planning

# %% [markdown]
# ## Recap
#
# Theory: [`1_theory/11_reinforcement_learning.md`](../../1_theory/11_reinforcement_learning.md).
#
# - MDP $\langle S, A, P, R, \gamma \rangle$ (11.1); return $G_t = \sum_k \gamma^k r_{t+k+1}$ (11.2).
# - Policy evaluation is linear: $(I - \gamma P_\pi) V^\pi = r_\pi$ (11.3); $Q(s, a)$ from $V$ (11.4); greedy policy
#   (11.5).
# - Value iteration $V_{k+1} = \max_a Q_k$ (11.6) is a $\gamma$-contraction (11.7); policy iteration alternates exact
#   evaluation and greedy improvement.
# - TD control: $Q \leftarrow Q + \alpha(\text{target} - Q)$ (11.8); Q-learning bootstraps on $\max_{a'} Q(s', a')$,
#   SARSA on the action actually taken (11.9); ε-greedy exploration (11.10).

# %%
import matplotlib.pyplot as plt
import numpy as np

from motion_planning import rl
from motion_planning.plotting import INK, SERIES, use_style

use_style()
ROOM = ["S....#....",
        ".##..#.##.",
        ".#.......G",
        ".#..###...",
        "....#....."]
ARROWS = "↑→↓←"


def show(grid, V=None, policy=None, title=""):
    """Print a grid with optimal values or policy arrows."""
    print(title)
    for r in range(grid.rows):
        cells = []
        for c in range(grid.cols):
            s = grid.state(r, c)
            ch = grid.cell(s)
            if ch in "#GP":
                cells.append(f"{ch:>6}")
            elif policy is not None:
                cells.append(f"{ARROWS[policy[s]]:>6}")
            else:
                cells.append(f"{V[s]:6.1f}")
        print("".join(cells))

# %% [markdown]
# ## Value iteration on a room
#
# Every move costs 1; the goal is 11 moves from the start.

# %%
room = rl.GridWorld.from_layout(ROOM)
m = room.to_mdp()
vi = rl.value_iteration(m, 1.0)
show(room, V=vi.V, title=f"V* (gamma = 1), {vi.iterations} sweeps")
show(room, policy=vi.policy, title="\ngreedy policy")
plt.figure(figsize=(5, 3))
plt.semilogy(vi.residuals, "o-")
plt.xlabel("sweep"); plt.ylabel(r"max $|V_{k+1} - V_k|$"); plt.title("value iteration, gamma = 1"); plt.show()

# %% [markdown]
# ### ✏️ Exercise 1 — one Bellman backup
#
# Implement eq. (11.4) and one sweep of eq. (11.6) in NumPy, using `m.P[s][a]` (a list of transitions with
# `next`, `probability`, `reward`). Terminal states keep the value 0.

# %% [solution]
def q_value(m, V, s, a, gamma):
    return sum(t.probability * (t.reward + gamma * V[t.next]) for t in m.P[s][a])


def bellman_sweep(m, V, gamma):
    out = np.zeros_like(V)
    for s in range(m.num_states):
        if not m.is_terminal(s):
            out[s] = max(q_value(m, V, s, a, gamma) for a in range(m.num_actions))
    return out

# %% [exercise]
def q_value(m, V, s, a, gamma):
    ...  # ✏️ eq. (11.4)


def bellman_sweep(m, V, gamma):
    ...  # ✏️ eq. (11.6); terminal states stay 0

# %%
slippery = rl.GridWorld.from_layout(ROOM, -1.0, 10.0, -100.0, 0.2).to_mdp()
V = np.zeros(slippery.num_states)
for _ in range(400):
    V = bellman_sweep(slippery, V, 0.95)
assert np.allclose(V, rl.value_iteration(slippery, 0.95, 1e-12).V, atol=1e-8)
print("✓ bellman_sweep")

# %% [markdown]
# ### ✏️ Exercise 2 — values from distances
#
# In the deterministic room, compute $V^*$ for $\gamma = 0.9$ from breadth-first distances to the goal with eq.
# (11.11), without any MDP code, and compare.

# %% [solution]
from collections import deque


def distances(grid):
    d = {s: None for s in range(grid.rows * grid.cols)}
    q = deque(s for s in d if grid.cell(s) == "G")
    for s in q:
        d[s] = 0
    while q:
        s = q.popleft()
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            r, c = grid.row(s) + dr, grid.col(s) + dc
            if 0 <= r < grid.rows and 0 <= c < grid.cols and grid.layout[r][c] != "#":
                n = grid.state(r, c)
                if d[n] is None:
                    d[n] = d[s] + 1
                    q.append(n)
    return d


def v_from_distance(dist, gamma):
    return -(1 - gamma ** dist) / (1 - gamma)

# %% [exercise]
def distances(grid):
    ...  # ✏️ breadth-first search from the goal over free cells


def v_from_distance(dist, gamma):
    ...  # ✏️ eq. (11.11)

# %%
d = distances(room)
V09 = rl.value_iteration(m, 0.9, 1e-12).V
for s in range(m.num_states):
    if not m.is_terminal(s):
        assert abs(V09[s] - v_from_distance(d[s], 0.9)) < 1e-9
print("✓ V* = -(1 - gamma^d) / (1 - gamma), e.g. start:", round(V09[room.start()], 3))

# %% [markdown]
# ## Policy iteration
#
# Exact evaluation (11.3) and greedy improvement. Few iterations, each a linear solve.

# %%
pi = rl.policy_iteration(slippery, 0.95)
vi_s = rl.value_iteration(slippery, 0.95, 1e-12)
print(f"policy iteration: {pi.iterations} steps; value iteration: {vi_s.iterations} sweeps; "
      f"max |V_pi - V_vi| = {np.abs(pi.V - vi_s.V).max():.1e}")

# %% [markdown]
# ### ✏️ Exercise 3 — policy evaluation as a linear system
#
# Build $I - \gamma P_\pi$ and $r_\pi$ for a policy and solve (11.3) with `np.linalg.solve`. Terminal rows are the
# identity with right-hand side 0.

# %% [solution]
def evaluate(m, policy, gamma):
    n = m.num_states
    A, b = np.eye(n), np.zeros(n)
    for s in range(n):
        if m.is_terminal(s):
            continue
        for t in m.P[s][policy[s]]:
            A[s, t.next] -= gamma * t.probability
            b[s] += t.probability * t.reward
    return np.linalg.solve(A, b)

# %% [exercise]
def evaluate(m, policy, gamma):
    ...  # ✏️ eq. (11.3)

# %%
policy = [max(a, 0) for a in pi.policy]
assert np.allclose(evaluate(slippery, policy, 0.95), rl.evaluate_policy(slippery, policy, 0.95))
print("✓ evaluate")

# %% [markdown]
# ## How much caution does slip deserve?
#
# The cliff with slip 0, 0.1, 0.2: the optimal policy moves away from the edge, and at slip 0.2 the start action is
# to push into the wall (←): it waits for a safe drift upwards.

# %%
cliff = rl.cliff_world()
for slip in (0.0, 0.1, 0.2):
    g = rl.GridWorld.from_layout(cliff.layout, -1.0, 0.0, -100.0, slip)
    v = rl.value_iteration(g.to_mdp(), 1.0, 1e-10)
    show(g, policy=v.policy, title=f"\nslip {slip}: V*(start) = {v.V[g.start()]:.1f}")

# %% [markdown]
# ## Learning: Q-learning against SARSA
#
# The agent does not see $P$: it acts, observes, updates. Same cliff, ε = 0.1.

# %%
cm = cliff.to_mdp()
fig, ax = plt.subplots(figsize=(6.5, 3.5))
for k, (name, method) in enumerate([("Q-learning", rl.TdMethod.QLearning), ("SARSA", rl.TdMethod.Sarsa)]):
    r = rl.td_control(cm, cliff.start(), rl.TdConfig(method, 1.0, 0.5, 0.1, 500, 1000, 7))
    path = rl.rollout(cm, rl.greedy_from_q(cm, r.Q), cliff.start())
    smooth = np.convolve(r.episode_returns, np.ones(20) / 20, mode="valid")
    ax.plot(smooth, color=SERIES[k], label=f"{name}: greedy path {len(path) - 1} moves")
    print(f"{name:10s} mean return of the last 100 episodes {np.mean(r.episode_returns[-100:]):.1f}, "
          f"falls {sum(x <= -100 for x in r.episode_returns)}")
ax.axhline(-13, color=INK, ls="--", lw=1)
ax.set_ylim(-110, 0); ax.set_xlabel("episode"); ax.set_ylabel("return (smoothed)"); ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ### ✏️ Exercise 4 — the Q-learning update
#
# Write tabular Q-learning yourself (eqs. 11.8–11.10) for the room, using `rl.Simulator`-like sampling from
# `m.P` with NumPy's random generator. After 300 episodes its greedy policy must reach the goal in 11 moves.

# %% [solution]
def q_learning(m, start, episodes=300, alpha=0.5, epsilon=0.2, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((m.num_states, m.num_actions))

    def act(s):
        if rng.random() < epsilon:
            return int(rng.integers(m.num_actions))
        best = np.flatnonzero(Q[s] == Q[s].max())
        return int(rng.choice(best))

    for _ in range(episodes):
        s = start
        for _ in range(1000):
            a = act(s)
            outcomes = m.P[s][a]
            t = outcomes[rng.choice(len(outcomes), p=[o.probability for o in outcomes])]
            target = t.reward if m.is_terminal(t.next) else t.reward + gamma * Q[t.next].max()
            Q[s, a] += alpha * (target - Q[s, a])
            if m.is_terminal(t.next):
                break
            s = t.next
    return Q

# %% [exercise]
def q_learning(m, start, episodes=300, alpha=0.5, epsilon=0.2, gamma=1.0, seed=0):
    ...  # ✏️ eqs. (11.8)-(11.10)

# %%
Q = q_learning(m, room.start())
path = rl.rollout(m, rl.greedy_from_q(m, Q), room.start())
assert len(path) - 1 == 11 and room.cell(path[-1]) == "G"
print("✓ q_learning: greedy path", len(path) - 1, "moves")

# %% [markdown]
# ## 🔨 Break it — what exploration costs
#
# Train Q-learning on the cliff with ε = 0, 0.1 and 0.3. Surprise: ε = 0 still finds the optimal path. $Q$ starts
# at 0 and every reward is negative, so an action that has been tried looks *worse* than one that has not, and the
# greedy agent tries everything — **optimistic initial values** do the exploring. Raising ε only adds falls.

# %%
for eps in (0.0, 0.1, 0.3):
    r = rl.td_control(cm, cliff.start(), rl.TdConfig(rl.TdMethod.QLearning, 1.0, 0.5, eps, 500, 1000, 7))
    path = rl.rollout(cm, rl.greedy_from_q(cm, r.Q), cliff.start())
    print(f"epsilon {eps}: falls {sum(x <= -100 for x in r.episode_returns):3d} / 500, "
          f"greedy path {len(path) - 1} moves, mean return of the last 100 episodes {np.mean(r.episode_returns[-100:]):.1f}")

# %% [markdown]
# ## 🔨 Break it — gamma = 1 and a policy that never ends
#
# Evaluate "always push up" in the room with $\gamma = 1$: from the top row the agent bumps into the border forever,
# so its return is $-\infty$ and the linear system (11.3) is singular.

# %%
always_up = [0] * m.num_states
try:
    rl.evaluate_policy(m, always_up, 1.0)
except RuntimeError as e:
    print("RuntimeError:", e)
print("with gamma = 0.9 it is finite:", round(rl.evaluate_policy(m, always_up, 0.9)[room.start()], 2), "= -1 / (1 - 0.9)")

# %% [markdown]
# ## 🔨 Break it — a learning rate that never settles
#
# With slip, transitions are random and a constant α keeps $Q$ jumping around its target. Compare the spread of
# $\max_a Q(\text{start}, a)$ over the last 200 of 2000 episodes for α = 0.5 and α = 0.05 with the true $V^*$.

# %%
g02 = rl.GridWorld.from_layout(ROOM, -1.0, 0.0, -100.0, 0.2)
m02 = g02.to_mdp()
v_star = rl.value_iteration(m02, 1.0, 1e-10).V[g02.start()]
for alpha in (0.5, 0.05):
    vals = []
    for seed in range(1, 6):
        r = rl.td_control(m02, g02.start(), rl.TdConfig(rl.TdMethod.QLearning, 1.0, alpha, 0.2, 2000, 1000, seed))
        vals.append(r.Q[g02.start()].max())
    print(f"alpha {alpha}: max Q(start) over 5 seeds = {np.round(vals, 2)}  (V* = {v_star:.2f})")

# %% [markdown]
# ## What to remember
#
# - An MDP adds probabilities to planning; the optimal policy trades speed against risk (the cliff with slip).
# - Value iteration is a $\gamma$-contraction; policy iteration converges in few exact steps.
# - Q-learning learns the optimal policy off-policy; SARSA learns the value of what it actually does, exploration
#   included — safer while learning.
# - Exploration is necessary, and costs: learn in simulation before trusting a robot near a cliff.
