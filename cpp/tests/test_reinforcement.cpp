// Chapter 11 tests: transition models that sum to one, a two-state MDP solved by hand, the closed-form optimal
// values of a deterministic grid (from breadth-first distances), value iteration against policy iteration and
// against exact policy evaluation, the gamma-contraction of the Bellman update, and Q-learning / SARSA on the cliff.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <deque>
#include <numeric>

#include "motion_planning/reinforcement.hpp"

using namespace motion_planning::rl;

namespace {

const std::vector<std::string> kRoom = {
    "S....#....",
    ".##..#.##.",
    ".#.......G",
    ".#..###...",
    "....#.....",
};

/// Shortest number of moves from every cell to the goal (independent of the MDP code).
std::vector<int> bfs_distance(const GridWorld& g) {
    std::vector<int> d(size_t(g.rows * g.cols), -1);
    std::deque<int> q;
    for (int s = 0; s < g.rows * g.cols; ++s)
        if (g.cell(s) == 'G') d[size_t(s)] = 0, q.push_back(s);
    const int dr[4] = {-1, 0, 1, 0}, dc[4] = {0, 1, 0, -1};
    while (!q.empty()) {
        const int s = q.front();
        q.pop_front();
        for (int k = 0; k < 4; ++k) {
            const int r = g.row(s) + dr[k], c = g.col(s) + dc[k];
            if (r < 0 || r >= g.rows || c < 0 || c >= g.cols || g.layout[size_t(r)][size_t(c)] == '#') continue;
            const int n = g.state(r, c);
            if (d[size_t(n)] < 0) d[size_t(n)] = d[size_t(s)] + 1, q.push_back(n);
        }
    }
    return d;
}

}  // namespace

TEST_CASE("grid transition probabilities sum to one", "[rl]") {
    for (double slip : {0.0, 0.2, 0.5}) {
        const Mdp m = GridWorld::from_layout(kRoom, -1.0, 0.0, -100.0, slip).to_mdp();
        for (int s = 0; s < m.num_states; ++s) {
            if (m.is_terminal(s)) continue;
            for (int a = 0; a < 4; ++a) {
                double p = 0.0;
                for (const auto& t : m.P[size_t(s)][size_t(a)]) p += t.probability;
                CHECK(std::abs(p - 1.0) < 1e-12);
            }
        }
    }
}

TEST_CASE("two-state MDP solved by hand", "[rl]") {
    // s0: "stay" earns 1, "go" moves to s1 earning 0; s1: both actions stay and earn 2. gamma = 0.9:
    // V(s1) = 2 / (1 - 0.9) = 20; V(s0) = max(1 + 0.9 V(s0), 0.9 V(s1)) = max(10, 18) = 18, policy "go".
    Mdp m;
    m.num_states = 2;
    m.num_actions = 2;
    m.terminal = {false, false};
    m.P = {{{{0, 1.0, 1.0}}, {{1, 1.0, 0.0}}}, {{{1, 1.0, 2.0}}, {{1, 1.0, 2.0}}}};
    const ValueIterationResult vi = value_iteration(m, 0.9, 1e-12);
    CHECK(std::abs(vi.V(0) - 18.0) < 1e-9);
    CHECK(std::abs(vi.V(1) - 20.0) < 1e-9);
    CHECK(vi.policy[0] == 1);
    CHECK(std::abs(evaluate_policy(m, {0, 0}, 0.9)(0) - 10.0) < 1e-12);  // always "stay": 1 / (1 - 0.9)
}

TEST_CASE("deterministic grid: optimal values from shortest distances", "[rl]") {
    // With -1 per move and no other reward, V*(s) = -(1 - gamma^d) / (1 - gamma), d = moves to the goal (eq. 11.11).
    const GridWorld g = GridWorld::from_layout(kRoom);
    const Mdp m = g.to_mdp();
    const auto d = bfs_distance(g);
    for (double gamma : {1.0, 0.9}) {
        const ValueIterationResult vi = value_iteration(m, gamma, 1e-12);
        for (int s = 0; s < m.num_states; ++s) {
            if (m.is_terminal(s)) continue;
            const double expected = gamma == 1.0 ? -d[size_t(s)] : -(1 - std::pow(gamma, d[size_t(s)])) / (1 - gamma);
            CHECK(std::abs(vi.V(s) - expected) < 1e-9);
        }
        const auto path = rollout(m, vi.policy, g.start());
        CHECK(static_cast<int>(path.size()) - 1 == d[size_t(g.start())]);
        CHECK(g.cell(path.back()) == 'G');
    }
}

TEST_CASE("value iteration and policy iteration agree, policy evaluation is exact", "[rl]") {
    const Mdp m = GridWorld::from_layout(kRoom, -1.0, 10.0, -100.0, 0.2).to_mdp();
    const double gamma = 0.95;
    const ValueIterationResult vi = value_iteration(m, gamma, 1e-12);
    const PolicyIterationResult pi = policy_iteration(m, gamma);
    CHECK((vi.V - pi.V).cwiseAbs().maxCoeff() < 1e-8);
    CHECK(pi.iterations < vi.iterations);
    // The value of the greedy policy of V* is V* (eq. 11.3 solved exactly) ...
    std::vector<int> policy = vi.policy;
    for (auto& a : policy) a = std::max(a, 0);
    const Eigen::VectorXd Vpi = evaluate_policy(m, policy, gamma);
    CHECK((Vpi - vi.V).cwiseAbs().maxCoeff() < 1e-8);
    // ... and the exact solution equals the fixed point of repeated policy backups.
    Eigen::VectorXd V = Eigen::VectorXd::Zero(m.num_states);
    for (int k = 0; k < 2000; ++k) {
        Eigen::VectorXd next = Eigen::VectorXd::Zero(m.num_states);
        for (int s = 0; s < m.num_states; ++s)
            if (!m.is_terminal(s)) next(s) = m.q_value(V, s, policy[size_t(s)], gamma);
        V = next;
    }
    CHECK((V - Vpi).cwiseAbs().maxCoeff() < 1e-8);
}

TEST_CASE("the Bellman update contracts by gamma", "[rl]") {
    // Eq. (11.7): |V_{k+1} - V_k| <= gamma |V_k - V_{k-1}| in the max norm.
    const Mdp m = GridWorld::from_layout(kRoom, -1.0, 10.0, -100.0, 0.3).to_mdp();
    const ValueIterationResult vi = value_iteration(m, 0.9, 1e-10);
    for (size_t k = 1; k < vi.residuals.size(); ++k) CHECK(vi.residuals[k] <= 0.9 * vi.residuals[k - 1] + 1e-12);
}

TEST_CASE("Q-learning finds the shortest path of a deterministic grid", "[rl]") {
    const GridWorld g = GridWorld::from_layout(kRoom);
    const Mdp m = g.to_mdp();
    TdConfig cfg;
    cfg.method = TdMethod::QLearning;
    cfg.gamma = 1.0;
    cfg.alpha = 0.5;
    cfg.epsilon = 0.2;
    cfg.episodes = 400;
    cfg.seed = 3;
    const TdResult r = td_control(m, g.start(), cfg);
    const auto path = rollout(m, greedy_from_q(m, r.Q), g.start());
    const int d = bfs_distance(g)[size_t(g.start())];
    CHECK(static_cast<int>(path.size()) - 1 == d);
    CHECK(g.cell(path.back()) == 'G');
    CHECK(std::abs(r.Q.row(g.start()).maxCoeff() - (-d)) < 1e-3);
}

TEST_CASE("cliff: Q-learning learns the edge, SARSA the safe path", "[rl]") {
    // Sutton & Barto, example 6.6, with epsilon = 0.1: the greedy policy of Q-learning walks along the cliff
    // (13 moves), SARSA's keeps a distance because its own exploration would push it off; during training SARSA
    // therefore earns more.
    const GridWorld g = cliff_world();
    const Mdp m = g.to_mdp();
    TdConfig cfg;
    cfg.gamma = 1.0;
    cfg.alpha = 0.5;
    cfg.epsilon = 0.1;
    cfg.episodes = 500;
    cfg.seed = 7;
    cfg.method = TdMethod::QLearning;
    const TdResult q = td_control(m, g.start(), cfg);
    cfg.method = TdMethod::Sarsa;
    const TdResult sarsa = td_control(m, g.start(), cfg);
    const auto q_path = rollout(m, greedy_from_q(m, q.Q), g.start());
    const auto s_path = rollout(m, greedy_from_q(m, sarsa.Q), g.start());
    CHECK(q_path.size() - 1 == 13);
    CHECK(s_path.size() - 1 > 13);
    CHECK(g.cell(q_path.back()) == 'G');
    CHECK(g.cell(s_path.back()) == 'G');
    auto mean_last = [](const std::vector<double>& v) {
        return std::accumulate(v.end() - 100, v.end(), 0.0) / 100.0;
    };
    CHECK(mean_last(sarsa.episode_returns) > mean_last(q.episode_returns));
}
