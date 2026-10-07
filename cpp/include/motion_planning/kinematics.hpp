#pragma once

/// Chapter 1 — vehicle kinematic models and their integration.
/// Theory: 1_theory/01_vehicle_kinematics.md. Equation numbers below refer to that file.

#include <Eigen/Core>
#include <limits>
#include <stdexcept>

namespace motion_planning {

/// Planar configuration q = (x, y, theta) of the vehicle reference point, eq. (1.1).
using Pose = Eigen::Vector3d;
/// Two-dimensional control input; its meaning depends on the model (see each model).
using Input = Eigen::Vector2d;
/// A trajectory sampled at fixed time steps, one pose per row.
using Trajectory = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
/// A sequence of inputs, one per row.
using InputSequence = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

/// Unicycle: input u = (v, omega), forward speed and yaw rate. Eq. (1.2).
struct Unicycle {
    Pose derivative(const Pose& q, const Input& u) const;
};

/// Differential drive: input u = (omega_left, omega_right), wheel angular speeds in rad/s.
/// Reference point is the midpoint of the wheel axle. Eqs. (1.3)–(1.4).
struct DifferentialDrive {
    double wheel_radius = 0.033;  ///< r [m]
    double track_width = 0.160;   ///< b, distance between the two wheel contact points [m]

    DifferentialDrive() = default;
    DifferentialDrive(double r, double b);

    /// Wheel speeds (omega_l, omega_r) -> body velocity (v, omega). Eq. (1.3).
    Input wheel_to_body(const Input& wheels) const;
    /// Body velocity (v, omega) -> wheel speeds (omega_l, omega_r). Eq. (1.4).
    Input body_to_wheel(const Input& body) const;
    Pose derivative(const Pose& q, const Input& wheels) const;
};

/// Kinematic bicycle: input u = (v, gamma), speed of the reference point and front steering angle.
/// The reference point lies on the vehicle axis at distance `rear_to_reference` (l_r) ahead of the rear
/// axle; l_r = 0 gives the rear-axle model of eq. (1.7), l_r > 0 the slip-angle model of eqs. (1.8)–(1.9).
struct KinematicBicycle {
    double wheelbase = 1.0;          ///< L [m]
    double rear_to_reference = 0.0;  ///< l_r, 0 <= l_r <= L [m]
    double max_steering = 0.7;       ///< |gamma| <= max_steering [rad], 0 < max_steering < pi/2

    KinematicBicycle() = default;
    KinematicBicycle(double L, double max_steering_angle, double l_r = 0.0);

    /// Steering angle saturated to [-max_steering, max_steering].
    double clamp_steering(double gamma) const;
    /// Turning radius of the rear axle, R = L / tan(gamma). Eq. (1.5). Infinite for gamma = 0.
    double turning_radius(double gamma) const;
    /// Slip angle beta between the vehicle axis and the velocity of the reference point. Eq. (1.8).
    double slip_angle(double gamma) const;
    /// Yaw rate for speed v and steering gamma. Eqs. (1.6) and (1.9).
    double yaw_rate(double v, double gamma) const;
    /// Steering angle that produces yaw rate omega at speed v (inverse of yaw_rate, not saturated).
    /// Returns 0 when |v| is negligible. Eq. (1.24).
    double steering_for(double v, double omega) const;
    /// Eq. (1.7) for l_r = 0, eq. (1.9) otherwise. Steering is saturated first.
    Pose derivative(const Pose& q, const Input& u) const;
};

/// Numerical integration scheme for one time step. Eqs. (1.13)–(1.15).
enum class Integrator { Euler, Midpoint, RK4 };

/// One integration step of q' = model.derivative(q, u) with u held constant over dt.
template <class Model>
Pose step(const Model& model, const Pose& q, const Input& u, double dt, Integrator method = Integrator::RK4) {
    switch (method) {
        case Integrator::Euler:  // eq. (1.13)
            return q + dt * model.derivative(q, u);
        case Integrator::Midpoint: {  // eq. (1.14)
            const Pose k1 = model.derivative(q, u);
            return q + dt * model.derivative(q + 0.5 * dt * k1, u);
        }
        case Integrator::RK4: {  // eq. (1.15)
            const Pose k1 = model.derivative(q, u);
            const Pose k2 = model.derivative(q + 0.5 * dt * k1, u);
            const Pose k3 = model.derivative(q + 0.5 * dt * k2, u);
            const Pose k4 = model.derivative(q + dt * k3, u);
            return q + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
    }
    throw std::invalid_argument("unknown integrator");
}

/// Exact solution of the unicycle over dt for constant (v, omega): an arc of a circle. Eq. (1.16).
Pose unicycle_exact_step(const Pose& q, const Input& u, double dt);

/// Apply a sequence of inputs, each held for dt. Returns N+1 poses, the first being q0.
/// Headings are not wrapped, so the trajectory is continuous in theta.
template <class Model>
Trajectory simulate(const Model& model, const Pose& q0, const InputSequence& inputs, double dt,
                    Integrator method = Integrator::RK4) {
    Trajectory out(inputs.rows() + 1, 3);
    out.row(0) = q0.transpose();
    Pose q = q0;
    for (Eigen::Index k = 0; k < inputs.rows(); ++k) {
        q = step(model, q, inputs.row(k).transpose(), dt, method);
        out.row(k + 1) = q.transpose();
    }
    return out;
}

/// Residual of the rolling-without-slipping constraint x' sin(theta) - y' cos(theta) = 0. Eq. (1.10).
/// Zero for any admissible velocity of a point on the rear axle.
double nonholonomic_residual(const Pose& q, const Pose& q_dot);

/// Lie bracket [g_drive, g_turn] of the unicycle input vector fields at q. Eq. (1.11).
Pose unicycle_lie_bracket(const Pose& q);

/// Bounds on a scalar command and on its rate of change. Eq. (1.17).
struct RateLimits {
    double min_value = -std::numeric_limits<double>::infinity();
    double max_value = std::numeric_limits<double>::infinity();
    double min_rate = -std::numeric_limits<double>::infinity();  ///< most negative allowed d/dt (<= 0)
    double max_rate = std::numeric_limits<double>::infinity();   ///< most positive allowed d/dt (>= 0)
};

/// Move from `previous` towards `desired` without exceeding the rate bounds over dt, then saturate.
/// Eq. (1.17).
double rate_limit(double previous, double desired, const RateLimits& limits, double dt);

}  // namespace motion_planning
