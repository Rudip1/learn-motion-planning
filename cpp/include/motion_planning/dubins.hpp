#pragma once

/// Chapter 6 — non-holonomic paths: Dubins curves and Dubins-RRT*.
/// Theory: 1_theory/06_dubins_paths.md. Equation numbers below refer to that file.

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "motion_planning/kinematics.hpp"
#include "motion_planning/sampling.hpp"

namespace motion_planning {

/// The six candidate words of Dubins' theorem: three segments, each a Left turn, a Right turn or a Straight
/// line (section 6.2).
enum class DubinsWord { LSL, RSR, LSR, RSL, RLR, LRL };

/// "LSL", "RSR", ...
std::string to_string(DubinsWord word);

/// A Dubins path: three segments of the given lengths [m] driven forwards with turning radius `radius`.
struct DubinsPath {
    Pose start = Pose::Zero();
    DubinsWord word = DubinsWord::LSL;
    std::array<double, 3> lengths{0.0, 0.0, 0.0};
    double radius = 1.0;

    double length() const { return lengths[0] + lengths[1] + lengths[2]; }
    /// Pose after arc length s along the path (clamped to [0, length]). Each segment is an exact unicycle
    /// flow, eq. (1.16).
    Pose at(double s) const;
    /// Poses every `step` metres, including both ends.
    Trajectory sample(double step) const;
    /// The same path cut after arc length s.
    DubinsPath truncated(double s) const;
};

/// The path of one word from q0 to q1, if that word can connect them. Eqs. (6.3)–(6.8).
std::optional<DubinsPath> dubins_path(const Pose& q0, const Pose& q1, double radius, DubinsWord word);

/// All feasible words, in the order LSL, RSR, LSR, RSL, RLR, LRL.
std::vector<DubinsPath> all_dubins_paths(const Pose& q0, const Pose& q1, double radius);

/// The shortest of the six: by Dubins' theorem, the shortest forward path with curvature at most 1/radius.
DubinsPath shortest_dubins_path(const Pose& q0, const Pose& q1, double radius);

/// Length of the shortest Dubins path. Not symmetric: d(q0, q1) != d(q1, q0) in general.
double dubins_distance(const Pose& q0, const Pose& q1, double radius);

/// True if the path is collision-free: poses every `step` metres are valid and the chords between them are
/// valid motions. The chord of an arc of length step deviates from the arc by at most step^2 / (8 radius).
/// Eq. (6.9).
bool dubins_path_valid(const DubinsPath& path, const PlanningProblem2D& problem, double step);

/// Dubins-RRT(*) options.
struct DubinsPlannerOptions {
    int max_iterations = 3000;
    double radius = 1.0;  ///< minimum turning radius
    double step = 3.0;    ///< longest path length added per extension
    double goal_bias = 0.05;
    double gamma = 0.0;            ///< neighbourhood constant; 0 selects a default from the box size
    double collision_step = 0.05;  ///< spacing of collision samples along curves [m]
    bool star = true;              ///< false: plain Dubins-RRT (no choose-parent, no rewiring)
    std::uint32_t seed = 1;
};

/// Result of a Dubins tree search.
struct DubinsTreeResult {
    std::vector<Pose> nodes;  ///< node 0 is the start
    std::vector<int> parent;
    std::vector<double> cost;  ///< path length from the start along the tree
    bool found = false;
    std::vector<DubinsPath> segments;  ///< start ... goal
    double path_cost = std::numeric_limits<double>::infinity();
    std::vector<double> best_cost_history;
};

/// RRT* in (x, y, theta) with Dubins paths as the steering function and their length as the edge cost. Uses
/// the positions part of `problem` (bounds and validity checks); start and goal are poses. Section 6.4.
DubinsTreeResult dubins_rrt_star(const PlanningProblem2D& problem, const Pose& start, const Pose& goal,
                                 const DubinsPlannerOptions& options = {});

}  // namespace motion_planning
