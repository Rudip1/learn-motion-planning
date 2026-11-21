"""Chapter 11: MDP solvers through the Python interface, against NumPy and closed forms."""

import numpy as np

from motion_planning import rl


def test_value_iteration_matches_numpy_bellman_backup():
    g = rl.GridWorld.from_layout(["S...", ".#..", "...G"], slip=0.2)
    m = g.to_mdp()
    vi = rl.value_iteration(m, 0.9, 1e-12)
    for s in range(m.num_states):
        if m.is_terminal(s):
            continue
        q = [sum(t.probability * (t.reward + 0.9 * vi.V[t.next]) for t in m.P[s][a]) for a in range(4)]
        assert abs(max(q) - vi.V[s]) < 1e-9


def test_policy_evaluation_is_the_linear_solve():
    g = rl.GridWorld.from_layout(["S..", "...", "..G"])
    m = g.to_mdp()
    policy = [1 if not m.is_terminal(s) else 0 for s in range(m.num_states)]  # always right
    policy = [2 if g.col(s) == 2 else a for s, a in enumerate(policy)]       # then down
    n = m.num_states
    A, b = np.eye(n), np.zeros(n)
    for s in range(n):
        if m.is_terminal(s):
            continue
        for t in m.P[s][policy[s]]:
            A[s, t.next] -= 0.9 * t.probability
            b[s] += t.probability * t.reward
    np.testing.assert_allclose(rl.evaluate_policy(m, policy, 0.9), np.linalg.solve(A, b), atol=1e-12)


def test_cliff_q_learning_path():
    c = rl.cliff_world()
    m = c.to_mdp()
    r = rl.td_control(m, c.start(), rl.TdConfig(rl.TdMethod.QLearning, 1.0, 0.5, 0.1, 500, 1000, 7))
    assert len(rl.rollout(m, rl.greedy_from_q(m, r.Q), c.start())) - 1 == 13
