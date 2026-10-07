// Chapter 7 tests: path parametrisation and projection by hand, shortcutting invariants, the pure-pursuit and
// Stanley laws against the worked examples of 1_theory/07_smoothing_tracking.md and the exact circle case,
// and closed-loop convergence on the kinematic bicycle of chapter 1.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "motion_planning/angles.hpp"
#include "motion_planning/tracking.hpp"

using namespace motion_planning;
using Catch::Approx;

namespace {
Points2D pts(std::initializer_list<std::pair<double, double>> xy) {
    Points2D p(static_cast<Eigen::Index>(xy.size()), 2);
    Eigen::Index i = 0;
    for (const auto& [x, y] : xy) p.row(i++) << x, y;
    return p;
}
double polyline_length(const Points2D& p) {
    double L = 0.0;
    for (Eigen::Index i = 1; i < p.rows(); ++i) L += (p.row(i) - p.row(i - 1)).norm();
    return L;
}
// a wall on x = 2 between y = -1 and y = 1
const MotionCheck wall = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    if ((a.x() - 2.0) * (b.x() - 2.0) > 0.0) return true;
    if (a.x() == b.x())
        return std::abs(a.x() - 2.0) > 1e-12 || std::max(a.y(), b.y()) < -1 || std::min(a.y(), b.y()) > 1;
    const double t = (2.0 - a.x()) / (b.x() - a.x());
    const double y = a.y() + t * (b.y() - a.y());
    return y > 1.0 || y < -1.0;
};
}  // namespace

TEST_CASE("path parametrisation and projection") {
    const Path2D p(pts({{0, 0}, {4, 0}, {4, 3}}));
    CHECK(p.length() == Approx(7.0));
    CHECK(p.point_at(5.5).isApprox(Eigen::Vector2d(4, 1.5)));
    CHECK(p.point_at(-1.0).isApprox(Eigen::Vector2d(0, 0)));
    CHECK(p.heading_at(5.0) == Approx(kPi / 2));
    const PathProjection a = p.project({2, 1});
    CHECK(a.s == Approx(2.0));
    CHECK(a.lateral == Approx(1.0));  // left of the path
    const PathProjection b = p.project({5, 1});
    CHECK(b.s == Approx(5.0));
    CHECK(b.lateral == Approx(-1.0));  // right of the upward segment
    CHECK_THROWS(Path2D(pts({{1, 1}, {1, 1}})));
}

TEST_CASE("shortcutting keeps endpoints and validity and never lengthens") {
    const Points2D zigzag =
        pts({{0, 0}, {0.5, 1.5}, {1.0, 1.5}, {1.5, 1.4}, {2.5, 1.4}, {3.0, 0.5}, {4.0, 0.0}});
    for (int i = 1; i < zigzag.rows(); ++i) REQUIRE(wall(zigzag.row(i - 1), zigzag.row(i)));
    const Points2D g = shortcut_greedy(zigzag, wall);
    const Points2D r = shortcut_random(zigzag, wall, 300, 7);
    for (const Points2D* p : {&g, &r}) {
        CHECK(p->row(0).isApprox(zigzag.row(0)));
        CHECK(p->row(p->rows() - 1).isApprox(zigzag.row(zigzag.rows() - 1)));
        CHECK(polyline_length(*p) <= polyline_length(zigzag) + 1e-12);
        for (int i = 1; i < p->rows(); ++i) CHECK(wall(p->row(i - 1), p->row(i)));
    }
    // the best possible path goes over the wall end at (2, 1): 2 * sqrt(5)
    CHECK(polyline_length(r) >= 2.0 * std::sqrt(5.0) - 1e-9);
    CHECK(polyline_length(r) <= 1.05 * 2.0 * std::sqrt(5.0));
    // in free space everything collapses to the straight segment
    const MotionCheck free = [](const Eigen::Vector2d&, const Eigen::Vector2d&) { return true; };
    CHECK(shortcut_greedy(zigzag, free).rows() == 2);
    // random shortcuts only approach it (they cut between interior points); a greedy pass finishes the job
    const Points2D rough = shortcut_random(zigzag, free, 50);
    CHECK(polyline_length(rough) < 1.02 * 4.0);
    CHECK(polyline_length(shortcut_greedy(rough, free)) == Approx(4.0));
    CHECK(turning_angles(pts({{0, 0}, {1, 0}, {1, 1}}))[0] == Approx(kPi / 2));
}

TEST_CASE("pure pursuit worked example and the exact circle") {
    // target (3, 1) from the origin heading east: the circle through both has radius 5
    CHECK(pure_pursuit_curvature(Pose(0, 0, 0), {3, 1}) == Approx(0.2));
    // on a circle of radius R, heading along it, a target L_d further along the arc gives curvature 1/R
    const double R = 4.0, Ld = 1.7;
    const Pose q(R, 0, kPi / 2);
    const Eigen::Vector2d target(R * std::cos(Ld / R), R * std::sin(Ld / R));
    CHECK(pure_pursuit_curvature(q, target) == Approx(1.0 / R));
}

TEST_CASE("Stanley worked example") {
    CHECK(stanley_steering(0.0, 0.5, 2.0, 1.0, 0.0) == Approx(-std::atan(0.25)));
    CHECK(stanley_steering(0.3, 0.0, 2.0, 1.0, 0.0) == Approx(0.3));
}

TEST_CASE("both controllers converge to a straight path") {
    const KinematicBicycle bike(1.0, 0.6);
    const Points2D line = pts({{0, 0}, {30, 0}});
    for (TrackingController c : {TrackingController::PurePursuit, TrackingController::Stanley}) {
        TrackingOptions opt;
        opt.controller = c;
        opt.speed = 2.0;
        opt.lookahead = 2.0;
        opt.stanley_gain = 1.5;
        const TrackingResult r = track_path(bike, line, Pose(0, 1.0, 0.0), opt);
        CHECK(r.reached_end);
        CHECK(std::abs(r.cross_track.back()) < 0.02);
        for (double g : r.steering) CHECK(std::abs(g) <= 0.6 + 1e-12);
    }
}

TEST_CASE("a larger Stanley gain converges faster, a slow steering actuator overshoots") {
    const KinematicBicycle bike(1.0, 0.6);
    const Points2D line = pts({{0, 0}, {40, 0}});
    auto settle_time = [&](double k, double rate) {
        TrackingOptions opt;
        opt.controller = TrackingController::Stanley;
        opt.speed = 2.0;
        opt.stanley_gain = k;
        opt.steering_rate = rate;
        const TrackingResult r = track_path(bike, line, Pose(0, 1.0, 0.0), opt);
        int last_big = 0;
        double overshoot = 0.0;
        for (int i = 0; i < static_cast<int>(r.cross_track.size()); ++i) {
            if (std::abs(r.cross_track[i]) > 0.05) last_big = i;
            overshoot = std::max(overshoot, -r.cross_track[i]);
        }
        return std::make_pair(last_big * opt.dt, overshoot);
    };
    CHECK(settle_time(2.0, 1e9).first < settle_time(0.5, 1e9).first);
    CHECK(settle_time(2.0, 0.3).second > settle_time(2.0, 1e9).second + 0.05);
}

TEST_CASE("pure pursuit follows a circle of polyline vertices closely") {
    const KinematicBicycle bike(1.0, 0.7);
    const int n = 400;
    const double R = 6.0;
    Points2D circle(n + 1, 2);
    for (int i = 0; i <= n; ++i)
        circle.row(i) << R * std::cos(1.5 * kPi * i / n), R * std::sin(1.5 * kPi * i / n);
    TrackingOptions opt;
    opt.speed = 1.5;
    opt.lookahead = 1.0;
    const TrackingResult r = track_path(bike, circle, Pose(R, 0, kPi / 2), opt);
    CHECK(r.reached_end);
    double worst = 0.0;
    for (double e : r.cross_track) worst = std::max(worst, std::abs(e));
    CHECK(worst < 0.01);
}
