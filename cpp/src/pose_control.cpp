#include "motion_planning/pose_control.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "motion_planning/angles.hpp"

namespace motion_planning {

PolarError polar_error(const Pose& q, const Pose& goal) {
    const double dx = goal[0] - q[0], dy = goal[1] - q[1];
    PolarError e;
    e.rho = std::hypot(dx, dy);  // eq. (1.19)
    e.alpha = angle_difference(std::atan2(dy, dx), q[2]);
    e.beta = wrap_angle(goal[2] - q[2] - e.alpha);
    return e;
}

Input move_to_point(const Pose& q, const Eigen::Vector2d& goal, const PointGains& gains) {
    const double dx = goal[0] - q[0], dy = goal[1] - q[1];
    const double v = gains.k_v * std::hypot(dx, dy);  // eq. (1.18)
    const double w = gains.k_h * angle_difference(std::atan2(dy, dx), q[2]);
    return Input(v, w);
}

bool goal_is_behind(const Pose& q, const Eigen::Vector2d& goal) {
    const double bearing = angle_difference(std::atan2(goal[1] - q[1], goal[0] - q[0]), q[2]);
    return std::abs(bearing) > 0.5 * kPi;
}

Input move_to_pose(const Pose& q, const Pose& goal, const PoseGains& gains, bool reverse) {
    if (!reverse) {
        const PolarError e = polar_error(q, goal);
        return Input(gains.k_rho * e.rho, gains.k_alpha * e.alpha + gains.k_beta * e.beta);  // eq. (1.21)
    }
    // A vehicle driving backwards with heading theta is a vehicle driving forwards with heading theta + pi.
    const Pose q_flip(q[0], q[1], q[2] + kPi);
    const Pose goal_flip(goal[0], goal[1], goal[2] + kPi);
    const Input u = move_to_pose(q_flip, goal_flip, gains, false);
    return Input(-u[0], u[1]);
}

Eigen::Vector3d polar_closed_loop_rates(const PolarError& e, const PoseGains& g) {
    const double v = g.k_rho * e.rho;  // eqs. (1.20) with (1.21) substituted gives eq. (1.22)
    const double w = g.k_alpha * e.alpha + g.k_beta * e.beta;
    const double s = e.rho > 0.0 ? std::sin(e.alpha) / e.rho : 0.0;
    return Eigen::Vector3d(-v * std::cos(e.alpha), v * s - w, -v * s);
}

namespace {

// Map a unicycle command to the model's own input.
Input to_model_input(const Unicycle&, double v, double w) { return Input(v, w); }
Input to_model_input(const KinematicBicycle& m, double v, double w) {
    return Input(v, m.clamp_steering(m.steering_for(v, w)));  // eq. (1.24), then saturation
}

template <class Model, class Law, class Done>
RegulationResult run(const Model& model, const Pose& q0, const RegulationOptions& opt, Law law, Done done) {
    std::vector<Pose> poses{q0};
    std::vector<Input> inputs;
    Pose q = q0;
    double v_prev = 0.0;
    bool converged = done(q);
    for (int k = 0; k < opt.max_steps && !converged; ++k) {
        const Input cmd = law(q);
        const double v = rate_limit(v_prev, cmd[0], opt.speed, opt.dt);  // eq. (1.17)
        const Input u = to_model_input(model, v, cmd[1]);
        q = step(model, q, u, opt.dt, Integrator::RK4);
        v_prev = v;
        poses.push_back(q);
        inputs.push_back(u);
        converged = done(q);
    }
    RegulationResult r;
    r.states.resize(static_cast<Eigen::Index>(poses.size()), 3);
    for (std::size_t i = 0; i < poses.size(); ++i) r.states.row(static_cast<Eigen::Index>(i)) = poses[i];
    r.inputs.resize(static_cast<Eigen::Index>(inputs.size()), 2);
    for (std::size_t i = 0; i < inputs.size(); ++i) r.inputs.row(static_cast<Eigen::Index>(i)) = inputs[i];
    r.converged = converged;
    return r;
}

template <class Model>
RegulationResult regulate_pose_impl(const Model& model, const Pose& q0, const Pose& goal, const PoseGains& g,
                                    const RegulationOptions& opt) {
    // The direction is chosen once: switching mid-run at |alpha| = pi/2 would make the law chatter.
    const bool reverse = opt.allow_reverse && goal_is_behind(q0, goal.head<2>());
    auto law = [&](const Pose& q) { return move_to_pose(q, goal, g, reverse); };
    auto done = [&](const Pose& q) {
        return (q.head<2>() - goal.head<2>()).norm() < opt.position_tolerance &&
               std::abs(angle_difference(q[2], goal[2])) < opt.heading_tolerance;
    };
    return run(model, q0, opt, law, done);
}

template <class Model>
RegulationResult regulate_point_impl(const Model& model, const Pose& q0, const Eigen::Vector2d& goal,
                                     const PointGains& g, const RegulationOptions& opt) {
    auto law = [&](const Pose& q) { return move_to_point(q, goal, g); };
    auto done = [&](const Pose& q) { return (q.head<2>() - goal).norm() < opt.position_tolerance; };
    return run(model, q0, opt, law, done);
}

void require_rear_axle(const KinematicBicycle& m) {
    if (m.rear_to_reference != 0.0)
        throw std::invalid_argument(
            "regulation assumes the rear-axle reference point (rear_to_reference = 0)");
}

}  // namespace

RegulationResult regulate_pose(const Unicycle& model, const Pose& q0, const Pose& goal,
                               const PoseGains& gains, const RegulationOptions& options) {
    return regulate_pose_impl(model, q0, goal, gains, options);
}

RegulationResult regulate_pose(const KinematicBicycle& model, const Pose& q0, const Pose& goal,
                               const PoseGains& gains, const RegulationOptions& options) {
    require_rear_axle(model);
    return regulate_pose_impl(model, q0, goal, gains, options);
}

RegulationResult regulate_point(const Unicycle& model, const Pose& q0, const Eigen::Vector2d& goal,
                                const PointGains& gains, const RegulationOptions& options) {
    return regulate_point_impl(model, q0, goal, gains, options);
}

RegulationResult regulate_point(const KinematicBicycle& model, const Pose& q0, const Eigen::Vector2d& goal,
                                const PointGains& gains, const RegulationOptions& options) {
    require_rear_axle(model);
    return regulate_point_impl(model, q0, goal, gains, options);
}

}  // namespace motion_planning
