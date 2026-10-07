// Chapter 1 tests: point and pose regulation, checked against the hand-worked polar example and the
// closed-form linearisation of 1_theory/01_vehicle_kinematics.md.
#include <Eigen/Eigenvalues>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>

#include "motion_planning/angles.hpp"
#include "motion_planning/pose_control.hpp"

using namespace motion_planning;
using Catch::Approx;

TEST_CASE("polar error worked example") {
    const PolarError e = polar_error(Pose(0, 0, 0), Pose(1, 1, kPi / 2));
    CHECK(e.rho == Approx(std::sqrt(2.0)));
    CHECK(e.alpha == Approx(kPi / 4));
    CHECK(e.beta == Approx(kPi / 4));
    const Input u = move_to_pose(Pose(0, 0, 0), Pose(1, 1, kPi / 2), PoseGains{0.5, 1.5, -0.6});
    CHECK(u[0] == Approx(0.5 * std::sqrt(2.0)));
    CHECK(u[1] == Approx(1.5 * kPi / 4 - 0.6 * kPi / 4));
}

TEST_CASE("linearised polar loop has the closed-form characteristic polynomial") {
    const PoseGains g{0.7, 2.1, -0.9};
    // Jacobian of (rho', alpha', beta') at the origin by central differences (rho row at rho = 1)
    auto f = [&](const Eigen::Vector3d& x) {
        return polar_closed_loop_rates(PolarError{x[0], x[1], x[2]}, g);
    };
    const double h = 1e-6;
    Eigen::Matrix2d A;  // the (alpha, beta) block
    for (int j = 0; j < 2; ++j) {
        Eigen::Vector3d xp(1.0, 0.0, 0.0), xm(1.0, 0.0, 0.0);
        xp[j + 1] += h;
        xm[j + 1] -= h;
        A.col(j) = ((f(xp) - f(xm)) / (2 * h)).tail<2>();
    }
    // eq. (1.23): lambda^2 + (k_alpha - k_rho) lambda - k_beta k_rho = 0
    const double b = g.k_alpha - g.k_rho, c = -g.k_beta * g.k_rho;
    const std::complex<double> disc = std::sqrt(std::complex<double>(b * b - 4 * c));
    const std::complex<double> l1 = (-b + disc) / 2.0, l2 = (-b - disc) / 2.0;
    Eigen::EigenSolver<Eigen::Matrix2d> es(A);
    const auto ev = es.eigenvalues();
    const bool same_order = std::abs(ev[0] - l1) < std::abs(ev[0] - l2);
    CHECK(std::abs(ev[0] - (same_order ? l1 : l2)) < 1e-6);
    CHECK(std::abs(ev[1] - (same_order ? l2 : l1)) < 1e-6);
    CHECK(ev.real().maxCoeff() < 0.0);
    // rho decouples with rate -k_rho
    Eigen::Vector3d r1(1.0 + h, 0, 0), r0(1.0 - h, 0, 0);
    CHECK(((f(r1) - f(r0)) / (2 * h))[0] == Approx(-g.k_rho));
}

TEST_CASE("stability conditions") {
    CHECK(PoseGains{0.5, 1.5, -0.6}.is_stable());
    CHECK_FALSE(PoseGains{0.5, 0.4, -0.6}.is_stable());  // k_alpha < k_rho
    CHECK_FALSE(PoseGains{0.5, 1.5, 0.6}.is_stable());   // k_beta > 0
    CHECK_FALSE(PoseGains{-0.5, 1.5, -0.6}.is_stable());
}

TEST_CASE("unicycle reaches poses ahead of and behind it") {
    const Pose goal(2.0, 1.0, kPi / 3);
    RegulationOptions opt;
    opt.position_tolerance = 0.02;
    opt.heading_tolerance = 0.05;
    int behind = 0;
    for (int i = 0; i < 12; ++i) {
        const double a = 2.0 * kPi * i / 12.0;
        const Pose q0(goal[0] + 3.0 * std::cos(a), goal[1] + 3.0 * std::sin(a), 0.4 * i);
        behind += goal_is_behind(q0, goal.head<2>());
        const RegulationResult r = regulate_pose(Unicycle{}, q0, goal, PoseGains{}, opt);
        INFO("start " << i);
        CHECK(r.converged);
    }
    CHECK(behind > 0);  // the ring of starts exercises the reverse branch
}

TEST_CASE("reverse branch drives backwards") {
    const Pose q0(0, 0, 0);
    const Pose goal(-3.0, 0.5, 0.0);
    REQUIRE(goal_is_behind(q0, goal.head<2>()));
    const RegulationResult r = regulate_pose(Unicycle{}, q0, goal, PoseGains{}, RegulationOptions{});
    CHECK(r.converged);
    CHECK(r.inputs.col(0).maxCoeff() <= 0.0);
}

TEST_CASE("bicycle with steering limit reaches a pose in front") {
    const KinematicBicycle bike(1.2, 35.0 * kPi / 180.0);
    RegulationOptions opt;
    opt.position_tolerance = 0.1;
    opt.heading_tolerance = 0.1;
    opt.speed.min_value = -5.0;
    opt.speed.max_value = 5.0;
    opt.speed.min_rate = -3.0;
    opt.speed.max_rate = 1.5;
    const RegulationResult r =
        regulate_pose(bike, Pose(0, 0, 0), Pose(15.0, 10.0, -kPi / 2), PoseGains{}, opt);
    CHECK(r.converged);
    CHECK(r.inputs.col(1).cwiseAbs().maxCoeff() <= bike.max_steering + 1e-12);
    // applied speed changes no faster than the rate limits allow
    for (Eigen::Index k = 1; k < r.inputs.rows(); ++k) {
        const double dv = r.inputs(k, 0) - r.inputs(k - 1, 0);
        CHECK(dv <= 1.5 * opt.dt + 1e-12);
        CHECK(dv >= -3.0 * opt.dt - 1e-12);
    }
    CHECK_THROWS(regulate_pose(KinematicBicycle(1.0, 0.5, 0.5), Pose(0, 0, 0), Pose(1, 0, 0), PoseGains{}));
}

TEST_CASE("unstable gains miss the heading") {
    RegulationOptions opt;
    opt.heading_tolerance = 0.05;
    opt.max_steps = 2000;
    const RegulationResult r =
        regulate_pose(Unicycle{}, Pose(0, 0, 0), Pose(4, 2, kPi / 2), PoseGains{0.5, 1.5, 0.6}, opt);
    CHECK_FALSE(r.converged);
}

TEST_CASE("move to point reaches the goal with either model") {
    const Eigen::Vector2d goal(-25.0, 20.0);
    RegulationOptions opt;
    opt.position_tolerance = 0.5;
    opt.max_steps = 20000;
    CHECK(regulate_point(Unicycle{}, Pose(0, 0, 0), goal, PointGains{}, opt).converged);
    const KinematicBicycle bike(1.2, 0.6);
    CHECK(regulate_point(bike, Pose(0, 0, 0), goal, PointGains{0.5, 2.0}, opt).converged);
}
