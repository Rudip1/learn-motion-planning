#pragma once

/// Chapter 8 — local planning with the Dynamic Window Approach, alone and behind a global planner.
/// Theory: 1_theory/08_local_planning.md. Equation numbers below refer to that file.

#include <Eigen/Core>
#include <vector>

#include "motion_planning/grid.hpp"
#include "motion_planning/kinematics.hpp"
#include "motion_planning/tracking.hpp"

namespace motion_planning {

/// Parameters of a differential-drive robot (a unicycle, chapter 1) and of the DWA objective.
struct DwaConfig {
    double max_speed = 0.6;       ///< [m/s]
    double min_speed = 0.0;       ///< [m/s]; DWA here drives forwards only
    double max_yaw_rate = 1.5;    ///< [rad/s]
    double max_accel = 0.6;       ///< [m/s^2], also the braking deceleration
    double max_yaw_accel = 3.0;   ///< [rad/s^2]
    double control_period = 0.1;  ///< [s] between decisions; sets the window (eq. 8.2)
    double horizon = 2.0;         ///< [s] of simulated motion per candidate
    double sim_step = 0.1;        ///< [s] between collision checks along a rollout
    int v_samples = 11;
    int w_samples = 21;
    double robot_radius = 0.2;     ///< [m]
    double heading_weight = 1.0;   ///< alpha, eq. (8.4)
    double distance_weight = 0.3;  ///< beta
    double velocity_weight = 0.3;  ///< gamma
    double distance_cap = 1.0;     ///< [m] free distance beyond this earns no more reward
};

/// The dynamic window: the velocities reachable within one control period, intersected with the robot's
/// limits. Eqs. (8.1)–(8.2).
struct DynamicWindow {
    double v_min, v_max, w_min, w_max;
};
DynamicWindow dynamic_window(const Input& current, const DwaConfig& config);

/// Poses along a constant-(v, omega) arc, every sim_step up to the horizon, starting with q. Exact arcs, eq.
/// (1.16).
Trajectory rollout(const Pose& q, double v, double w, const DwaConfig& config);

/// Score of one candidate velocity.
struct DwaCandidate {
    double v = 0.0, w = 0.0;
    double heading = 0.0;   ///< in [0, 1]: 1 - |bearing error| / pi where the rollout is closest to the goal
    double distance = 0.0;  ///< in [0, 1]: free distance along the arc / distance_cap (1 if collision-free)
    double velocity = 0.0;  ///< in [0, 1]: v / max_speed
    double total = 0.0;     ///< weighted sum, eq. (8.4)
    double free_distance =
        0.0;                  ///< arc length driven before the first collision (inf if none in the horizon)
    bool admissible = false;  ///< can stop before that collision, eq. (8.3)
};

/// The decision of one DWA step.
struct DwaDecision {
    bool found = false;  ///< false if no admissible velocity exists: the robot should brake
    Input command = Input::Zero();
    std::vector<DwaCandidate> candidates;
    Trajectory best_rollout;
};

/// Evaluate a grid of velocities in the dynamic window and pick the admissible one with the highest score.
/// `distance` is the Euclidean distance transform of the grid (chapter 2). Algorithm in section 8.3.
DwaDecision dwa_step(const OccupancyGrid& grid, const FieldArray& distance, const Pose& q,
                     const Input& current, const Eigen::Vector2d& goal, const DwaConfig& config);

/// Outcome of a closed-loop DWA run.
struct DwaRun {
    Trajectory states;
    InputSequence commands;
    bool reached_goal = false;
    bool stuck = false;  ///< no progress towards the (local) goal for `stuck_time` seconds
};

/// Drive a unicycle with DWA to `goal`. If `global_path` has at least two points, the DWA goal at every step
/// is the point `carrot_distance` ahead of the robot's projection onto that path (section 8.4); otherwise the
/// final goal.
DwaRun run_dwa(const OccupancyGrid& grid, const Pose& start, const Eigen::Vector2d& goal,
               const DwaConfig& config, const Points2D& global_path = Points2D(),
               double carrot_distance = 1.0, double goal_tolerance = 0.15, double max_time = 120.0,
               double stuck_time = 10.0);

}  // namespace motion_planning
