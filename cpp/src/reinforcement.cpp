#include "motion_planning/reinforcement.hpp"

#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace motion_planning::rl {

double Mdp::q_value(const Eigen::VectorXd& V, int s, int a, double gamma) const {
    double q = 0.0;
    for (const Transition& t : P[size_t(s)][size_t(a)]) q += t.probability * (t.reward + gamma * V(t.next));
    return q;  // eq. (11.4)
}

// ================================================================================================ grid worlds

GridWorld GridWorld::from_layout(const std::vector<std::string>& layout, double step_reward, double goal_reward,
                                 double pit_reward, double slip) {
    if (layout.empty()) throw std::invalid_argument("GridWorld: empty layout");
    GridWorld g;
    g.layout = layout;
    g.rows = static_cast<int>(layout.size());
    g.cols = static_cast<int>(layout[0].size());
    for (const auto& row : layout)
        if (static_cast<int>(row.size()) != g.cols) throw std::invalid_argument("GridWorld: ragged layout");
    g.step_reward = step_reward;
    g.goal_reward = goal_reward;
    g.pit_reward = pit_reward;
    g.slip = slip;
    return g;
}

int GridWorld::start() const {
    for (int s = 0; s < rows * cols; ++s)
        if (cell(s) == 'S') return s;
    return 0;
}

Mdp GridWorld::to_mdp() const {
    static const int dr[4] = {-1, 0, 1, 0}, dc[4] = {0, 1, 0, -1};
    Mdp m;
    m.num_states = rows * cols;
    m.num_actions = 4;
    m.P.assign(size_t(m.num_states), std::vector<std::vector<Transition>>(4));
    m.terminal.assign(size_t(m.num_states), false);
    for (int s = 0; s < m.num_states; ++s) {
        const char c = cell(s);
        if (c == '#' || c == 'G' || c == 'P') {
            m.terminal[size_t(s)] = true;
            continue;
        }
        for (int a = 0; a < 4; ++a) {
            // Intended move with 1 - slip, each perpendicular move with slip / 2 (section 11.1).
            std::vector<std::pair<int, double>> moves{{a, 1.0 - slip}};
            if (slip > 0.0) {
                moves.push_back({(a + 1) % 4, slip / 2});
                moves.push_back({(a + 3) % 4, slip / 2});
            }
            std::map<int, Transition> merged;
            for (auto [mv, p] : moves) {
                if (p <= 0.0) continue;
                int r = row(s) + dr[mv], k = col(s) + dc[mv];
                if (r < 0 || r >= rows || k < 0 || k >= cols || layout[size_t(r)][size_t(k)] == '#') r = row(s), k = col(s);
                const int next = state(r, k);
                double reward = step_reward;
                if (cell(next) == 'G') reward += goal_reward;
                if (cell(next) == 'P') reward += pit_reward;
                const auto it = merged.find(next);
                if (it == merged.end())
                    merged[next] = Transition{next, p, reward};
                else
                    it->second.probability += p;  // two moves bumped into the same wall
            }
            for (auto& [next, t] : merged) m.P[size_t(s)][size_t(a)].push_back(t);
        }
    }
    return m;
}

GridWorld cliff_world() {
    return GridWorld::from_layout({"............", "............", "............", "SPPPPPPPPPPG"}, -1.0, 0.0, -100.0, 0.0);
}

// ================================================================================================ planning

std::vector<int> greedy_policy(const Mdp& mdp, const Eigen::VectorXd& V, double gamma) {
    std::vector<int> pi(size_t(mdp.num_states), -1);
    for (int s = 0; s < mdp.num_states; ++s) {
        if (mdp.is_terminal(s)) continue;
        double best = -std::numeric_limits<double>::infinity();
        for (int a = 0; a < mdp.num_actions; ++a) {
            const double q = mdp.q_value(V, s, a, gamma);
            if (q > best + 1e-12) best = q, pi[size_t(s)] = a;  // eq. (11.5), lowest index on ties
        }
    }
    return pi;
}

ValueIterationResult value_iteration(const Mdp& mdp, double gamma, double tol, int max_iterations) {
    ValueIterationResult r;
    r.V = Eigen::VectorXd::Zero(mdp.num_states);
    for (r.iterations = 0; r.iterations < max_iterations;) {
        Eigen::VectorXd next = Eigen::VectorXd::Zero(mdp.num_states);
        for (int s = 0; s < mdp.num_states; ++s) {
            if (mdp.is_terminal(s)) continue;
            double best = -std::numeric_limits<double>::infinity();
            for (int a = 0; a < mdp.num_actions; ++a) best = std::max(best, mdp.q_value(r.V, s, a, gamma));
            next(s) = best;  // eq. (11.6)
        }
        const double residual = (next - r.V).cwiseAbs().maxCoeff();
        r.V = next;
        ++r.iterations;
        r.residuals.push_back(residual);
        if (residual < tol) break;
    }
    r.policy = greedy_policy(mdp, r.V, gamma);
    return r;
}

Eigen::VectorXd evaluate_policy(const Mdp& mdp, const std::vector<int>& policy, double gamma) {
    const int n = mdp.num_states;
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    for (int s = 0; s < n; ++s) {
        if (mdp.is_terminal(s)) continue;
        for (const Transition& t : mdp.P[size_t(s)][size_t(policy[size_t(s)])]) {
            A(s, t.next) -= gamma * t.probability;  // eq. (11.3): (I - gamma P_pi) V = r_pi
            b(s) += t.probability * t.reward;
        }
    }
    const Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
    if (!lu.isInvertible())
        throw std::runtime_error("evaluate_policy: the policy never terminates from some state (use gamma < 1)");
    return lu.solve(b);
}

PolicyIterationResult policy_iteration(const Mdp& mdp, double gamma, int max_iterations) {
    PolicyIterationResult r;
    r.policy.assign(size_t(mdp.num_states), 0);
    for (int s = 0; s < mdp.num_states; ++s)
        if (mdp.is_terminal(s)) r.policy[size_t(s)] = -1;
    for (r.iterations = 0; r.iterations < max_iterations;) {
        std::vector<int> eval_policy = r.policy;
        for (auto& a : eval_policy) a = std::max(a, 0);
        r.V = evaluate_policy(mdp, eval_policy, gamma);
        ++r.iterations;
        // Improvement: change an action only if another is strictly better (keeps the loop finite on ties).
        bool stable = true;
        for (int s = 0; s < mdp.num_states; ++s) {
            if (mdp.is_terminal(s)) continue;
            const int cur = r.policy[size_t(s)];
            double best = mdp.q_value(r.V, s, cur, gamma);
            for (int a = 0; a < mdp.num_actions; ++a) {
                const double q = mdp.q_value(r.V, s, a, gamma);
                if (q > best + 1e-9) best = q, r.policy[size_t(s)] = a, stable = false;
            }
        }
        if (stable) break;
    }
    return r;
}

// ================================================================================================ learning

Simulator::Simulator(const Mdp& mdp, std::uint64_t seed) : mdp_(mdp), state_(seed) {}

double Simulator::uniform() {
    // splitmix64: small, fast, and identical on every platform (tests rely on the seed).
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return double(z >> 11) * 0x1.0p-53;
}

int Simulator::random_action() {
    return std::min(mdp_.num_actions - 1, static_cast<int>(uniform() * mdp_.num_actions));
}

Transition Simulator::step(int s, int a) {
    const auto& outcomes = mdp_.P[size_t(s)][size_t(a)];
    if (outcomes.empty()) throw std::logic_error("Simulator::step from a terminal state");
    double u = uniform();
    for (const Transition& t : outcomes) {
        if (u < t.probability) return t;
        u -= t.probability;
    }
    return outcomes.back();
}

namespace {

/// Epsilon-greedy (eq. 11.10), with random tie-breaking among the greedy actions.
int epsilon_greedy(const Eigen::MatrixXd& Q, int s, double epsilon, Simulator& sim) {
    if (sim.uniform() < epsilon) return sim.random_action();
    const double best = Q.row(s).maxCoeff();
    std::vector<int> ties;
    for (int a = 0; a < Q.cols(); ++a)
        if (Q(s, a) >= best - 1e-12) ties.push_back(a);
    return ties[std::min(ties.size() - 1, size_t(sim.uniform() * double(ties.size())))];
}

}  // namespace

TdResult td_control(const Mdp& mdp, int start, const TdConfig& config) {
    TdResult r;
    r.Q = Eigen::MatrixXd::Zero(mdp.num_states, mdp.num_actions);
    Simulator sim(mdp, config.seed);
    for (int ep = 0; ep < config.episodes; ++ep) {
        int s = start, steps = 0;
        double ret = 0.0;
        int a = epsilon_greedy(r.Q, s, config.epsilon, sim);
        while (steps < config.max_steps) {
            const Transition t = sim.step(s, a);
            ret += t.reward;
            ++steps;
            double target = t.reward;
            int a_next = -1;
            if (!mdp.is_terminal(t.next)) {
                a_next = epsilon_greedy(r.Q, t.next, config.epsilon, sim);
                target += config.gamma * (config.method == TdMethod::QLearning ? r.Q.row(t.next).maxCoeff()  // (11.8)
                                                                                : r.Q(t.next, a_next));      // (11.9)
            }
            r.Q(s, a) += config.alpha * (target - r.Q(s, a));  // eqs. (11.8), (11.9)
            if (mdp.is_terminal(t.next)) break;
            s = t.next;
            a = a_next;
        }
        r.episode_returns.push_back(ret);
        r.episode_lengths.push_back(steps);
    }
    return r;
}

std::vector<int> greedy_from_q(const Mdp& mdp, const Eigen::MatrixXd& Q) {
    std::vector<int> pi(size_t(mdp.num_states), -1);
    for (int s = 0; s < mdp.num_states; ++s) {
        if (mdp.is_terminal(s)) continue;
        Eigen::Index best;
        Q.row(s).maxCoeff(&best);
        pi[size_t(s)] = static_cast<int>(best);
    }
    return pi;
}

std::vector<int> rollout(const Mdp& mdp, const std::vector<int>& policy, int start, int max_steps) {
    std::vector<int> path{start};
    int s = start;
    for (int k = 0; k < max_steps && !mdp.is_terminal(s); ++k) {
        const auto& outcomes = mdp.P[size_t(s)][size_t(policy[size_t(s)])];
        const auto it = std::max_element(outcomes.begin(), outcomes.end(),
                                         [](const Transition& x, const Transition& y) { return x.probability < y.probability; });
        s = it->next;
        if (std::find(path.begin(), path.end(), s) != path.end()) {
            path.push_back(s);
            break;
        }
        path.push_back(s);
    }
    return path;
}

}  // namespace motion_planning::rl
