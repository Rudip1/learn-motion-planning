#include "motion_planning/dwa.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "motion_planning/angles.hpp"

namespace motion_planning {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

// Clearance of a point: distance to the nearest obstacle cell minus the robot radius (chapter 2, eq. 2.4).
double clearance_at(const OccupancyGrid& grid, const FieldArray& distance, const Eigen::Vector2d& p,
                    double radius) {
    const Cell c = grid.world_to_cell(p);
    if (grid.occupied(c)) return -radius;
    return distance(c.y, c.x) - radius;
}
}  // namespace

DynamicWindow dynamic_window(const Input& current, const DwaConfig& cfg) {
    const double dv = cfg.max_accel * cfg.control_period, dw = cfg.max_yaw_accel * cfg.control_period;
    DynamicWindow w;  // eq. (8.2): reachable velocities, clipped to the robot's limits (8.1)
    w.v_min = std::max(cfg.min_speed, current[0] - dv);
    w.v_max = std::min(cfg.max_speed, current[0] + dv);
    w.w_min = std::max(-cfg.max_yaw_rate, current[1] - dw);
    w.w_max = std::min(cfg.max_yaw_rate, current[1] + dw);
    return w;
}

Trajectory rollout(const Pose& q, double v, double w, const DwaConfig& cfg) {
    const int n = std::max(1, static_cast<int>(std::round(cfg.horizon / cfg.sim_step)));
    Trajectory t(n + 1, 3);
    for (int k = 0; k <= n; ++k) t.row(k) = unicycle_exact_step(q, Input(v, w), k * cfg.sim_step).transpose();
    return t;
}

DwaDecision dwa_step(const OccupancyGrid& grid, const FieldArray& distance, const Pose& q,
                     const Input& current, const Eigen::Vector2d& goal, const DwaConfig& cfg) {
    const DynamicWindow win = dynamic_window(current, cfg);
    DwaDecision d;
    double best = -kInf;
    for (int i = 0; i < cfg.v_samples; ++i) {
        const double v =
            cfg.v_samples > 1 ? win.v_min + (win.v_max - win.v_min) * i / (cfg.v_samples - 1) : win.v_max;
        for (int j = 0; j < cfg.w_samples; ++j) {
            const double w =
                cfg.w_samples > 1 ? win.w_min + (win.w_max - win.w_min) * j / (cfg.w_samples - 1) : 0.0;
            DwaCandidate c;
            c.v = v;
            c.w = w;
            const Trajectory t = rollout(q, v, w, cfg);
            c.free_distance = kInf;
            for (Eigen::Index k = 0; k < t.rows(); ++k) {
                if (clearance_at(grid, distance, t.row(k).head<2>().transpose(), cfg.robot_radius) <= 0.0) {
                    c.free_distance = std::abs(v) * k * cfg.sim_step;  // first collision along the arc
                    break;
                }
            }
            // eq. (8.3): admissible if the robot can brake to a stop before the collision. The free distance
            // is quantised by the rollout samples (which shift by up to one sample between decisions) and by
            // the grid cells, so a margin of two samples plus one cell is taken off it.
            const double margin = 2.0 * std::abs(v) * cfg.sim_step + grid.resolution();
            c.admissible = !std::isfinite(c.free_distance) ||
                           v * v <= 2.0 * cfg.max_accel * std::max(0.0, c.free_distance - margin);
            // eq. (8.4) terms. The heading is scored where the rollout comes closest to the goal, not at its
            // end: a rollout that reaches the goal and drives on would otherwise face away from it and lose
            // to standing still.
            Eigen::Index closest = 0;
            for (Eigen::Index k = 1; k < t.rows(); ++k)  // ties go to the later pose (turning on the spot)
                if ((t.row(k).head<2>().transpose() - goal).norm() <=
                    (t.row(closest).head<2>().transpose() - goal).norm() + 1e-12)
                    closest = k;
            const Pose qc = t.row(closest).transpose();
            const double reach = (qc.head<2>() - goal).norm();
            const double bearing = std::atan2(goal.y() - qc[1], goal.x() - qc[0]);
            c.heading = reach < 0.05 ? 1.0 : 1.0 - std::abs(angle_difference(bearing, qc[2])) / kPi;
            c.distance = std::min(c.free_distance, cfg.distance_cap) / cfg.distance_cap;
            c.velocity = cfg.max_speed > 0.0 ? v / cfg.max_speed : 0.0;
            c.total = cfg.heading_weight * c.heading + cfg.distance_weight * c.distance +
                      cfg.velocity_weight * c.velocity;
            if (c.admissible && c.total > best) {
                best = c.total;
                d.found = true;
                d.command = Input(v, w);
                d.best_rollout = t;
            }
            d.candidates.push_back(c);
        }
    }
    return d;
}

DwaRun run_dwa(const OccupancyGrid& grid, const Pose& start, const Eigen::Vector2d& goal,
               const DwaConfig& cfg, const Points2D& global_path, double carrot_distance,
               double goal_tolerance, double max_time, double stuck_time) {
    const FieldArray distance = distance_transform(grid);
    const bool guided = global_path.rows() >= 2;
    std::vector<Pose> states{start};
    std::vector<Input> commands;
    Pose q = start;
    Input u = Input::Zero();
    DwaRun run;
    const int steps = static_cast<int>(max_time / cfg.control_period);
    const int stuck_steps = static_cast<int>(stuck_time / cfg.control_period);
    double best_progress = kInf;  // smallest distance to the final goal so far
    int last_improvement = 0;
    for (int k = 0; k < steps; ++k) {
        if ((q.head<2>() - goal).norm() <= goal_tolerance) {
            run.reached_goal = true;
            break;
        }
        Eigen::Vector2d local_goal = goal;
        if (guided) {  // section 8.4: chase a carrot on the global path
            const Path2D path(global_path);
            local_goal = path.point_at(path.project(q.head<2>()).s + carrot_distance);
        }
        const DwaDecision d = dwa_step(grid, distance, q, u, local_goal, cfg);
        if (d.found) {
            u = d.command;
        } else {  // nothing admissible: brake as hard as allowed and turn in place
            u = Input(std::max(0.0, u[0] - cfg.max_accel * cfg.control_period), u[1]);
        }
        q = unicycle_exact_step(q, u, cfg.control_period);
        states.push_back(q);
        commands.push_back(u);
        const double dist = (q.head<2>() - goal).norm();
        if (dist < best_progress - 0.05) {
            best_progress = dist;
            last_improvement = k;
        } else if (k - last_improvement > stuck_steps) {
            run.stuck = true;
            break;
        }
    }
    run.states.resize(static_cast<Eigen::Index>(states.size()), 3);
    for (std::size_t i = 0; i < states.size(); ++i) run.states.row(static_cast<Eigen::Index>(i)) = states[i];
    run.commands.resize(static_cast<Eigen::Index>(commands.size()), 2);
    for (std::size_t i = 0; i < commands.size(); ++i)
        run.commands.row(static_cast<Eigen::Index>(i)) = commands[i];
    return run;
}

}  // namespace motion_planning
