#pragma once

/// Chapter 11 — reinforcement learning basics: finite MDPs, value iteration, policy iteration, and tabular
/// temporal-difference control (Q-learning, SARSA) on grid worlds. Theory: 1_theory/11_reinforcement_learning.md.
/// Equation numbers below refer to that file.

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

namespace motion_planning::rl {

/// One possible outcome of taking an action: next state, its probability and the reward received.
struct Transition {
    int next = 0;
    double probability = 1.0;
    double reward = 0.0;
};

/// A finite MDP: P[s][a] lists the outcomes of action a in state s. Terminal states have no outcomes (their value
/// is 0). Eq. (11.1).
struct Mdp {
    int num_states = 0;
    int num_actions = 0;
    std::vector<std::vector<std::vector<Transition>>> P;
    std::vector<bool> terminal;

    bool is_terminal(int s) const { return terminal[size_t(s)]; }
    /// Expected reward plus discounted value of the successors: Q(s, a) = sum_s' p (r + gamma V(s')). Eq. (11.4).
    double q_value(const Eigen::VectorXd& V, int s, int a, double gamma) const;
};

// ----------------------------------------------------------------------------------------------- grid worlds

/// Moves of a grid world, in this order (row 0 is the top row, so Up decreases the row).
enum class Move { Up = 0, Right = 1, Down = 2, Left = 3 };

/// A rectangular grid world. Cells are (row, col) with row 0 at the top; state = row * cols + col.
/// Every move earns `step_reward` (negative: a cost); entering a goal ends the episode and adds `goal_reward`,
/// entering a pit ends it and adds `pit_reward`. Walls and the border block a move (the agent stays). With `slip` > 0 the intended
/// move happens with probability 1 - slip and each of the two perpendicular moves with slip / 2.
struct GridWorld {
    int rows = 0, cols = 0;
    std::vector<std::string> layout;  ///< '.' free, '#' wall, 'G' goal, 'P' pit, 'S' start
    double step_reward = -1.0;
    double goal_reward = 0.0;
    double pit_reward = -100.0;
    double slip = 0.0;

    /// Parse a layout, one string per row.
    static GridWorld from_layout(const std::vector<std::string>& layout, double step_reward = -1.0,
                                 double goal_reward = 0.0, double pit_reward = -100.0, double slip = 0.0);
    int state(int row, int col) const { return row * cols + col; }
    int row(int s) const { return s / cols; }
    int col(int s) const { return s % cols; }
    char cell(int s) const { return layout[size_t(row(s))][size_t(col(s))]; }
    int start() const;  ///< the 'S' cell (0 if none)

    /// The MDP of this grid (walls are states that are never reached; they are marked terminal). Eq. (11.1).
    Mdp to_mdp() const;
};

/// The classic cliff (Sutton and Barto, example 6.6): 4 x 12, start bottom left, goal bottom right, the bottom
/// row between them a cliff. Step reward -1; falling off adds -100 and, unlike the book, ends the episode.
GridWorld cliff_world();

// ----------------------------------------------------------------------------------------------- planning

struct ValueIterationResult {
    Eigen::VectorXd V;
    std::vector<int> policy;          ///< greedy action per state (-1 for terminal states)
    int iterations = 0;
    std::vector<double> residuals;    ///< max |V_{k+1} - V_k| per sweep
};

/// Value iteration: V_{k+1}(s) = max_a Q_k(s, a) until the update is below tol. Eq. (11.6).
ValueIterationResult value_iteration(const Mdp& mdp, double gamma, double tol = 1e-10, int max_iterations = 100000);

/// Exact value of a deterministic policy: solve (I - gamma P_pi) V = r_pi. Eq. (11.3).
Eigen::VectorXd evaluate_policy(const Mdp& mdp, const std::vector<int>& policy, double gamma);

struct PolicyIterationResult {
    Eigen::VectorXd V;
    std::vector<int> policy;
    int iterations = 0;  ///< policy improvement steps until the policy is stable
};

/// Policy iteration: evaluate exactly, improve greedily, until stable. Section 11.4. The initial policy is action 0
/// everywhere.
PolicyIterationResult policy_iteration(const Mdp& mdp, double gamma, int max_iterations = 1000);

/// Greedy policy for V: argmax_a Q(s, a) (lowest index on ties; -1 for terminal states). Eq. (11.5).
std::vector<int> greedy_policy(const Mdp& mdp, const Eigen::VectorXd& V, double gamma);

// ----------------------------------------------------------------------------------------------- learning

/// Samples transitions of an MDP: the environment a learning agent interacts with (it never sees P).
class Simulator {
  public:
    Simulator(const Mdp& mdp, std::uint64_t seed);
    /// Take action a in state s: returns the outcome drawn with the MDP's probabilities.
    Transition step(int s, int a);
    double uniform();
    int random_action();

  private:
    const Mdp& mdp_;
    std::uint64_t state_;
};

enum class TdMethod {
    QLearning,  ///< off-policy: target r + gamma max_a' Q(s', a'). Eq. (11.8)
    Sarsa,      ///< on-policy: target r + gamma Q(s', a') with a' the action actually taken next. Eq. (11.9)
};

struct TdConfig {
    TdMethod method = TdMethod::QLearning;
    double gamma = 1.0;
    double alpha = 0.5;     ///< learning rate
    double epsilon = 0.1;   ///< exploration of the epsilon-greedy behaviour policy. Eq. (11.10)
    int episodes = 500;
    int max_steps = 1000;   ///< per episode
    std::uint64_t seed = 1;
};

struct TdResult {
    Eigen::MatrixXd Q;                    ///< num_states x num_actions
    std::vector<double> episode_returns;  ///< undiscounted return of each training episode
    std::vector<int> episode_lengths;
};

/// Tabular TD control from a fixed start state (Algorithm in section 11.6).
TdResult td_control(const Mdp& mdp, int start, const TdConfig& config);

/// Greedy policy of a Q table (-1 for terminal states).
std::vector<int> greedy_from_q(const Mdp& mdp, const Eigen::MatrixXd& Q);

/// Follow a deterministic policy from start (deterministic transitions: the most likely outcome) and return the
/// visited states; stops at a terminal state, a repeated state or after max_steps.
std::vector<int> rollout(const Mdp& mdp, const std::vector<int>& policy, int start, int max_steps = 1000);

}  // namespace motion_planning::rl
