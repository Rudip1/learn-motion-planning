#pragma once

/// Chapter 1 — driving a kinematic model to a point or to a pose.
/// Theory: 1_theory/01_vehicle_kinematics.md, section "Driving the model". Equation numbers refer to that
/// file.

#include <Eigen/Core>
#include <limits>

#include "motion_planning/kinematics.hpp"

namespace motion_planning {

/// Goal expressed in polar coordinates relative to the vehicle. Eq. (1.19).
struct PolarError {
    double rho = 0.0;    ///< distance to the goal position
    double alpha = 0.0;  ///< bearing of the goal in the vehicle frame, in [-pi, pi)
    double beta = 0.0;   ///< goal heading minus the heading of the line of sight, in [-pi, pi)
};

/// Polar error of goal pose (x_g, y_g, theta_g) seen from pose q. Eq. (1.19).
PolarError polar_error(const Pose& q, const Pose& goal);

/// Gains of the move-to-point law, eq. (1.18).
struct PointGains {
    double k_v = 0.5;  ///< speed per metre of distance, > 0
    double k_h = 2.0;  ///< yaw rate per radian of heading error, > 0
};

/// Gains of the pose law, eq. (1.21). Locally exponentially stable iff is_stable(), eq. (1.23).
struct PoseGains {
    double k_rho = 0.5;
    double k_alpha = 1.5;
    double k_beta = -0.6;
    bool is_stable() const { return k_rho > 0.0 && k_beta < 0.0 && k_alpha - k_rho > 0.0; }
};

/// Unicycle command (v, omega) that turns towards and drives at a goal point. Eq. (1.18).
Input move_to_point(const Pose& q, const Eigen::Vector2d& goal, const PointGains& gains = {});

/// True when the goal position lies behind the vehicle, |alpha| > pi/2. The pose law of eq. (1.21) is only
/// valid for |alpha| <= pi/2; such goals are reached in reverse.
bool goal_is_behind(const Pose& q, const Eigen::Vector2d& goal);

/// Unicycle command (v, omega) of the polar pose law, eq. (1.21). With `reverse` the law is applied to the
/// vehicle with heading flipped by pi and the speed is negated, so the vehicle backs into the pose.
Input move_to_pose(const Pose& q, const Pose& goal, const PoseGains& gains = {}, bool reverse = false);

/// Closed-loop rates (rho', alpha', beta') of the polar system under the pose law. Eq. (1.22).
Eigen::Vector3d polar_closed_loop_rates(const PolarError& e, const PoseGains& gains);

/// Options for a closed-loop regulation run.
struct RegulationOptions {
    double dt = 0.05;
    int max_steps = 4000;
    double position_tolerance = 0.05;                                    ///< stop when rho is below this ...
    double heading_tolerance = std::numeric_limits<double>::infinity();  ///< ... and |theta - theta_g| too
    RateLimits speed;           ///< bounds on v and on its rate (acceleration); default unbounded
    bool allow_reverse = true;  ///< pose runs: back into goals that start behind the vehicle
};

/// Result of a regulation run: poses (N+1 rows), applied inputs (N rows) and whether the goal was reached.
struct RegulationResult {
    Trajectory states;
    InputSequence inputs;
    bool converged = false;
};

/// Drive a unicycle to a goal pose with move_to_pose. Applied inputs are (v, omega).
RegulationResult regulate_pose(const Unicycle& model, const Pose& q0, const Pose& goal,
                               const PoseGains& gains, const RegulationOptions& options = {});

/// Drive a rear-axle kinematic bicycle (l_r = 0) to a goal pose. The unicycle command is mapped to a
/// steering angle with eq. (1.24) and saturated. Applied inputs are (v, gamma).
RegulationResult regulate_pose(const KinematicBicycle& model, const Pose& q0, const Pose& goal,
                               const PoseGains& gains, const RegulationOptions& options = {});

/// Drive a unicycle to a goal point with move_to_point (the final heading is whatever it ends up being).
RegulationResult regulate_point(const Unicycle& model, const Pose& q0, const Eigen::Vector2d& goal,
                                const PointGains& gains, const RegulationOptions& options = {});

/// Drive a rear-axle kinematic bicycle to a goal point. Applied inputs are (v, gamma).
RegulationResult regulate_point(const KinematicBicycle& model, const Pose& q0, const Eigen::Vector2d& goal,
                                const PointGains& gains, const RegulationOptions& options = {});

}  // namespace motion_planning
