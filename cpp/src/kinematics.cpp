#include "motion_planning/kinematics.hpp"

#include <algorithm>
#include <cmath>

#include "motion_planning/angles.hpp"

namespace motion_planning {

Pose Unicycle::derivative(const Pose& q, const Input& u) const {
    return Pose(u[0] * std::cos(q[2]), u[0] * std::sin(q[2]), u[1]);  // eq. (1.2)
}

DifferentialDrive::DifferentialDrive(double r, double b) : wheel_radius(r), track_width(b) {
    if (r <= 0.0 || b <= 0.0) throw std::invalid_argument("wheel radius and track width must be positive");
}

Input DifferentialDrive::wheel_to_body(const Input& wheels) const {
    const double wl = wheels[0], wr = wheels[1];
    return Input(wheel_radius * (wr + wl) / 2.0, wheel_radius * (wr - wl) / track_width);  // eq. (1.3)
}

Input DifferentialDrive::body_to_wheel(const Input& body) const {
    const double v = body[0], w = body[1];
    return Input((v - w * track_width / 2.0) / wheel_radius,  // eq. (1.4)
                 (v + w * track_width / 2.0) / wheel_radius);
}

Pose DifferentialDrive::derivative(const Pose& q, const Input& wheels) const {
    return Unicycle{}.derivative(q, wheel_to_body(wheels));
}

KinematicBicycle::KinematicBicycle(double L, double max_steering_angle, double l_r)
    : wheelbase(L), rear_to_reference(l_r), max_steering(max_steering_angle) {
    if (L <= 0.0) throw std::invalid_argument("wheelbase must be positive");
    if (l_r < 0.0 || l_r > L) throw std::invalid_argument("reference point must lie between the axles");
    if (max_steering_angle <= 0.0 || max_steering_angle >= 0.5 * kPi)
        throw std::invalid_argument("max steering must be in (0, pi/2)");
}

double KinematicBicycle::clamp_steering(double gamma) const {
    return std::clamp(gamma, -max_steering, max_steering);
}

double KinematicBicycle::turning_radius(double gamma) const {
    const double t = std::tan(gamma);
    if (t == 0.0) return std::numeric_limits<double>::infinity();
    return wheelbase / t;  // eq. (1.5)
}

double KinematicBicycle::slip_angle(double gamma) const {
    return std::atan(rear_to_reference / wheelbase * std::tan(gamma));  // eq. (1.8)
}

double KinematicBicycle::yaw_rate(double v, double gamma) const {
    return v * std::cos(slip_angle(gamma)) * std::tan(gamma) / wheelbase;  // eqs. (1.6), (1.9)
}

double KinematicBicycle::steering_for(double v, double omega) const {
    if (std::abs(v) < 1e-9) return 0.0;
    // eq. (1.24): k = omega L / v = t / sqrt(1 + (l_r t / L)^2) with t = tan(gamma)
    const double k = omega * wheelbase / v;
    const double c = rear_to_reference / wheelbase;
    const double denom = 1.0 - k * k * c * c;
    if (denom <= 0.0) return std::copysign(0.5 * kPi, k);  // yaw rate unreachable at this speed
    return std::atan(k / std::sqrt(denom));
}

Pose KinematicBicycle::derivative(const Pose& q, const Input& u) const {
    const double v = u[0];
    const double gamma = clamp_steering(u[1]);
    const double beta = slip_angle(gamma);
    return Pose(v * std::cos(q[2] + beta), v * std::sin(q[2] + beta), yaw_rate(v, gamma));  // eq. (1.9)
}

Pose unicycle_exact_step(const Pose& q, const Input& u, double dt) {
    const double v = u[0], w = u[1], th = q[2];
    const double th_next = th + w * dt;
    if (std::abs(w * dt) < 1e-9) {  // straight line; the limit of eq. (1.16) as omega -> 0
        const double th_mid = th + 0.5 * w * dt;
        return Pose(q[0] + v * dt * std::cos(th_mid), q[1] + v * dt * std::sin(th_mid), th_next);
    }
    const double R = v / w;  // eq. (1.16)
    return Pose(q[0] + R * (std::sin(th_next) - std::sin(th)), q[1] - R * (std::cos(th_next) - std::cos(th)),
                th_next);
}

double nonholonomic_residual(const Pose& q, const Pose& q_dot) {
    return q_dot[0] * std::sin(q[2]) - q_dot[1] * std::cos(q[2]);  // eq. (1.10)
}

Pose unicycle_lie_bracket(const Pose& q) {
    return Pose(std::sin(q[2]), -std::cos(q[2]), 0.0);  // eq. (1.11)
}

double rate_limit(double previous, double desired, const RateLimits& limits, double dt) {
    const double change = std::clamp(desired - previous, limits.min_rate * dt, limits.max_rate * dt);
    return std::clamp(previous + change, limits.min_value, limits.max_value);  // eq. (1.17)
}

}  // namespace motion_planning
