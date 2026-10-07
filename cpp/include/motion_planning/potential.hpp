#pragma once

/// Chapter 3 — potential fields and the wave-front planner.
/// Theory: 1_theory/03_potential_fields.md. Equation numbers below refer to that file.

#include <Eigen/Core>
#include <vector>

#include "motion_planning/grid.hpp"

namespace motion_planning {

/// Parameters of the attractive and repulsive potentials.
struct PotentialParams {
    double zeta = 1.0;    ///< attractive gain
    double d_star = 2.0;  ///< distance [m] at which the attractive potential turns from quadratic to conic
    double eta = 1.0;     ///< repulsive gain
    double q_star = 1.0;  ///< influence distance [m] of the obstacles
};

/// Attractive potential at a point: quadratic within d_star of the goal, conic beyond. Eq. (3.1).
double attractive_potential(const Eigen::Vector2d& p, const Eigen::Vector2d& goal,
                            const PotentialParams& params);
/// Gradient of eq. (3.1). Eq. (3.2).
Eigen::Vector2d attractive_gradient(const Eigen::Vector2d& p, const Eigen::Vector2d& goal,
                                    const PotentialParams& params);
/// Repulsive potential for clearance D [m] (distance to the nearest obstacle). +inf at D = 0. Eq. (3.3).
double repulsive_potential(double clearance, const PotentialParams& params);

/// Attractive potential evaluated at every cell centre.
FieldArray attractive_field(const OccupancyGrid& grid, const Eigen::Vector2d& goal,
                            const PotentialParams& params);
/// Repulsive potential of every cell from the Euclidean distance transform (chapter 2). Eq. (3.3).
FieldArray repulsive_field(const OccupancyGrid& grid, const PotentialParams& params);
/// Sum of the two. Eq. (3.4).
FieldArray total_field(const OccupancyGrid& grid, const Eigen::Vector2d& goal, const PotentialParams& params);

/// Wave-front potential: number of steps from every free cell to the goal cell through free cells
/// (breadth-first from the goal). +inf on occupied and unreachable cells. Eq. (3.6).
FieldArray wavefront(const OccupancyGrid& grid, const Cell& goal,
                     Connectivity connectivity = Connectivity::Four);

/// Outcome of a descent on a grid field.
struct DescentResult {
    std::vector<Cell> path;  ///< visited cells, start first
    bool reached_goal = false;
    bool local_minimum = false;  ///< stopped at a cell no neighbour of which is lower (and not the goal)
};

/// Discrete steepest descent: repeatedly move to the neighbour with the lowest value while it is strictly
/// lower than the current cell. Algorithm in section 3.3.
DescentResult descend(const FieldArray& field, const Cell& start, const Cell& goal,
                      Connectivity connectivity = Connectivity::Eight, int max_steps = 100000);

}  // namespace motion_planning
