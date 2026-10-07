#include "motion_planning/tracking.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include "motion_planning/angles.hpp"

namespace motion_planning {

namespace {

double cross2(const Eigen::Vector2d& a, const Eigen::Vector2d& b) { return a.x() * b.y() - a.y() * b.x(); }

Points2D from_vector(const std::vector<Eigen::Vector2d>& v) {
    Points2D out(static_cast<Eigen::Index>(v.size()), 2);
    for (std::size_t i = 0; i < v.size(); ++i) out.row(static_cast<Eigen::Index>(i)) = v[i].transpose();
    return out;
}

}  // namespace

Points2D shortcut_greedy(const Points2D& path, const MotionCheck& motion_valid) {
    const Eigen::Index n = path.rows();
    if (n <= 2) return path;
    std::vector<Eigen::Vector2d> out{path.row(0).transpose()};
    Eigen::Index i = 0;
    while (i < n - 1) {
        Eigen::Index j = n - 1;  // the farthest vertex visible from i; i + 1 is visible on a valid path
        while (j > i + 1 && !motion_valid(path.row(i).transpose(), path.row(j).transpose())) --j;
        out.push_back(path.row(j).transpose());
        i = j;
    }
    return from_vector(out);
}

Points2D shortcut_random(const Points2D& path, const MotionCheck& motion_valid, int iterations,
                         std::uint32_t seed) {
    if (path.rows() <= 2) return path;
    std::mt19937 rng(seed);
    Points2D current = path;
    for (int it = 0; it < iterations; ++it) {
        const Path2D p(current);
        std::uniform_real_distribution<double> u(0.0, p.length());
        double s1 = u(rng), s2 = u(rng);
        if (s1 > s2) std::swap(s1, s2);
        const Eigen::Vector2d a = p.point_at(s1), b = p.point_at(s2);
        if (s2 - s1 <= (b - a).norm() + 1e-12) continue;  // already straight between them
        if (!motion_valid(a, b)) continue;
        // splice: the vertices before s1, the shortcut a -> b, the vertices after s2
        std::vector<Eigen::Vector2d> out;
        double s = 0.0;
        for (Eigen::Index i = 0; i < current.rows(); ++i) {
            if (i > 0) s += (current.row(i) - current.row(i - 1)).norm();
            if (s < s1) out.push_back(current.row(i).transpose());
        }
        out.push_back(a);
        out.push_back(b);
        s = 0.0;
        for (Eigen::Index i = 0; i < current.rows(); ++i) {
            if (i > 0) s += (current.row(i) - current.row(i - 1)).norm();
            if (s > s2) out.push_back(current.row(i).transpose());
        }
        current = from_vector(out);
    }
    // drop points that repeat their predecessor
    std::vector<Eigen::Vector2d> clean{current.row(0).transpose()};
    for (Eigen::Index i = 1; i < current.rows(); ++i)
        if ((current.row(i).transpose() - clean.back()).norm() > 1e-12)
            clean.push_back(current.row(i).transpose());
    return from_vector(clean);
}

Eigen::VectorXd turning_angles(const Points2D& path) {
    const Eigen::Index n = path.rows();
    Eigen::VectorXd a = Eigen::VectorXd::Zero(std::max<Eigen::Index>(0, n - 2));
    for (Eigen::Index i = 1; i + 1 < n; ++i) {
        const Eigen::Vector2d d0 = path.row(i) - path.row(i - 1), d1 = path.row(i + 1) - path.row(i);
        a[i - 1] = std::abs(angle_difference(std::atan2(d1.y(), d1.x()), std::atan2(d0.y(), d0.x())));
    }
    return a;
}

Path2D::Path2D(const Points2D& points) {
    std::vector<Eigen::Vector2d> pts;
    for (Eigen::Index i = 0; i < points.rows(); ++i)
        if (pts.empty() || (points.row(i).transpose() - pts.back()).norm() > 1e-12)
            pts.push_back(points.row(i).transpose());
    if (pts.size() < 2) throw std::invalid_argument("a path needs two distinct points");
    points_ = from_vector(pts);
    s_.assign(pts.size(), 0.0);
    for (std::size_t i = 1; i < pts.size(); ++i) s_[i] = s_[i - 1] + (pts[i] - pts[i - 1]).norm();
}

int Path2D::segment_at(double s) const {
    const auto it = std::upper_bound(s_.begin(), s_.end(), s);
    const int i = static_cast<int>(it - s_.begin()) - 1;
    return std::clamp(i, 0, static_cast<int>(s_.size()) - 2);
}

Eigen::Vector2d Path2D::point_at(double s) const {
    s = std::clamp(s, 0.0, length());
    const int i = segment_at(s);
    const double t = (s - s_[i]) / (s_[i + 1] - s_[i]);
    return (1.0 - t) * points_.row(i).transpose() + t * points_.row(i + 1).transpose();
}

double Path2D::heading_at(double s) const {
    const int i = segment_at(std::clamp(s, 0.0, length()));
    const Eigen::Vector2d d = points_.row(i + 1) - points_.row(i);
    return std::atan2(d.y(), d.x());
}

PathProjection Path2D::project(const Eigen::Vector2d& p) const {
    PathProjection best;
    double best_d = std::numeric_limits<double>::infinity();
    for (int i = 0; i + 1 < static_cast<int>(s_.size()); ++i) {
        const Eigen::Vector2d a = points_.row(i), b = points_.row(i + 1), ab = b - a;
        const double t = std::clamp((p - a).dot(ab) / ab.squaredNorm(), 0.0, 1.0);
        const Eigen::Vector2d c = a + t * ab;
        const double d = (p - c).norm();
        if (d < best_d - 1e-12) {
            best_d = d;
            best.s = s_[i] + t * (s_[i + 1] - s_[i]);
            best.point = c;
            best.heading = std::atan2(ab.y(), ab.x());
            best.lateral = cross2(ab.normalized(), p - c);  // eq. (7.1): positive to the left
        }
    }
    return best;
}

double pure_pursuit_curvature(const Pose& q, const Eigen::Vector2d& target) {
    const Eigen::Vector2d d = target - q.head<2>();
    const double L2 = d.squaredNorm();
    if (L2 < 1e-12) return 0.0;
    const double y_v =
        -std::sin(q[2]) * d.x() + std::cos(q[2]) * d.y();  // lateral offset in the vehicle frame
    return 2.0 * y_v / L2;                                 // eq. (7.3)
}

double stanley_steering(double heading_error, double lateral_error, double speed, double gain,
                        double softening) {
    return heading_error - std::atan(gain * lateral_error / (softening + std::abs(speed)));  // eq. (7.5)
}

TrackingResult track_path(const KinematicBicycle& vehicle, const Points2D& points, const Pose& start,
                          const TrackingOptions& opt) {
    if (vehicle.rear_to_reference != 0.0)
        throw std::invalid_argument("tracking assumes the rear-axle reference");
    const Path2D path(points);
    const double L = vehicle.wheelbase;
    TrackingResult r;
    std::vector<Pose> states{start};
    Pose q = start;
    double gamma = 0.0;
    const double end_zone = std::max(0.05, 0.01 * path.length());
    for (int k = 0; k < opt.max_steps; ++k) {
        const PathProjection rear = path.project(q.head<2>());
        r.cross_track.push_back(rear.lateral);
        if (rear.s >= path.length() - end_zone) {
            r.reached_end = true;
            break;
        }
        double gamma_cmd;
        if (opt.controller == TrackingController::PurePursuit) {
            const double Ld = opt.lookahead + opt.lookahead_gain * opt.speed;  // eq. (7.4)
            const double kappa = pure_pursuit_curvature(q, path.point_at(rear.s + Ld));
            gamma_cmd = std::atan(L * kappa);  // eq. (1.6) inverted
        } else {
            const Eigen::Vector2d front = q.head<2>() + L * Eigen::Vector2d(std::cos(q[2]), std::sin(q[2]));
            const PathProjection f = path.project(front);
            gamma_cmd = stanley_steering(angle_difference(f.heading, q[2]), f.lateral, opt.speed,
                                         opt.stanley_gain, opt.softening);
        }
        RateLimits lim;  // eq. (1.17): steering rate and saturation
        lim.min_value = -vehicle.max_steering;
        lim.max_value = vehicle.max_steering;
        lim.min_rate = -opt.steering_rate;
        lim.max_rate = opt.steering_rate;
        gamma = rate_limit(gamma, gamma_cmd, lim, opt.dt);
        r.steering.push_back(gamma);
        q = step(vehicle, q, Input(opt.speed, gamma), opt.dt, Integrator::RK4);
        states.push_back(q);
    }
    r.states.resize(static_cast<Eigen::Index>(states.size()), 3);
    for (std::size_t i = 0; i < states.size(); ++i) r.states.row(static_cast<Eigen::Index>(i)) = states[i];
    return r;
}

}  // namespace motion_planning
