#pragma once

/// Chapter 5 — sampling-based planning: PRM, RRT and RRT* for a point robot in the plane.
/// Theory: 1_theory/05_sampling_planning.md. Equation numbers below refer to that file.

#include <Eigen/Core>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include "motion_planning/graph_search.hpp"
#include "motion_planning/grid.hpp"

namespace motion_planning {

/// A planning problem for a point in a rectangle: the two collision queries every sampling planner needs.
struct PlanningProblem2D {
    Eigen::Vector2d lower = Eigen::Vector2d::Zero();
    Eigen::Vector2d upper = Eigen::Vector2d::Ones();
    std::function<bool(const Eigen::Vector2d&)> state_valid;
    std::function<bool(const Eigen::Vector2d&, const Eigen::Vector2d&)> motion_valid;
    Eigen::Vector2d start = Eigen::Vector2d::Zero();
    Eigen::Vector2d goal = Eigen::Vector2d::Ones();
    double goal_radius =
        0.0;  ///< the goal region is the disc of this radius; 0 means "connect to the goal point"
};

/// Problem on an occupancy grid: states in free cells, motions by exact cell traversal (chapter 2).
PlanningProblem2D problem_from_grid(const OccupancyGrid& grid, const Eigen::Vector2d& start,
                                    const Eigen::Vector2d& goal);

/// Problem among polygons inside a rectangle: states outside every polygon, motions that do not enter any
/// polygon (chapter 4).
PlanningProblem2D problem_from_polygons(const std::vector<Polygon>& polygons, const Eigen::Vector2d& lower,
                                        const Eigen::Vector2d& upper, const Eigen::Vector2d& start,
                                        const Eigen::Vector2d& goal);

/// The radius of eq. (5.3) in 2-D: gamma * sqrt(log(n) / n).
double rewiring_radius(int n, double gamma);

/// The smallest gamma for which RRT* and PRM* are asymptotically optimal in 2-D, for a free area mu_free:
/// gamma* = 2 sqrt((1 + 1/2) mu_free / pi). Eq. (5.4).
double optimal_gamma(double free_area);

/// PRM options.
struct PrmOptions {
    int num_samples = 500;
    double connection_radius =
        -1.0;            ///< fixed radius; <= 0 selects the PRM* radius of eq. (5.3) with `gamma`
    double gamma = 0.0;  ///< used when connection_radius <= 0; 0 selects optimal_gamma of the box
    std::uint32_t seed = 1;
};

/// PRM result: the roadmap (start = node 0, goal = node 1, then the samples) and the query answer.
struct PrmResult {
    Graph roadmap;
    SearchResult query;
    int collision_checks = 0;  ///< motion-validity calls
};

/// Probabilistic roadmap: sample valid states, connect pairs within the radius whose straight motion is
/// valid, add start and goal the same way, search with A*. Algorithm in section 5.2.
PrmResult prm(const PlanningProblem2D& problem, const PrmOptions& options = {});

/// RRT / RRT* options.
struct RrtOptions {
    int max_iterations = 5000;
    double step = 0.5;                   ///< eta: the longest edge the tree grows in one step
    double goal_bias = 0.05;             ///< probability of sampling the goal
    bool stop_at_first_solution = true;  ///< RRT: stop when the goal is reached; RRT* ignores it
    double gamma = 0.0;                  ///< RRT* radius constant; 0 selects optimal_gamma of the box
    std::uint32_t seed = 1;
};

/// A search tree and its best solution.
struct TreeResult {
    std::vector<Eigen::Vector2d> nodes;  ///< node 0 is the start
    std::vector<int> parent;             ///< -1 for the root
    std::vector<double> cost;            ///< cost-to-come along the tree
    bool found = false;
    std::vector<Eigen::Vector2d> path;  ///< start ... goal
    double path_cost = std::numeric_limits<double>::infinity();
    int first_solution_iteration = -1;
    std::vector<double>
        best_cost_history;  ///< best solution cost after each iteration (inf before the first)
};

/// Rapidly-exploring random tree. Algorithm in section 5.3.
TreeResult rrt(const PlanningProblem2D& problem, const RrtOptions& options = {});

/// RRT*: RRT with choose-parent and rewiring inside the shrinking radius of eq. (5.3). Algorithm in
/// section 5.4.
TreeResult rrt_star(const PlanningProblem2D& problem, const RrtOptions& options = {});

/// Length of a polyline.
double path_length(const std::vector<Eigen::Vector2d>& path);

}  // namespace motion_planning
