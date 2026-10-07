// Chapter 1 tests: every model is checked against a closed form or a worked example of
// 1_theory/01_vehicle_kinematics.md.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>

#include "motion_planning/angles.hpp"
#include "motion_planning/kinematics.hpp"

using namespace motion_planning;
using Catch::Approx;

TEST_CASE("wrap_angle maps into the half-open interval from -pi to pi") {
    CHECK(wrap_angle(0.0) == Approx(0.0));
    CHECK(wrap_angle(kPi) == Approx(-kPi));
    CHECK(wrap_angle(-kPi) == Approx(-kPi));
    CHECK(wrap_angle(3.0 * kPi / 2.0) == Approx(-kPi / 2.0));
    CHECK(wrap_angle(-7.0 * kPi / 2.0) == Approx(kPi / 2.0));
    CHECK(wrap_angle(20.0 * kPi + 0.1) == Approx(0.1));
    CHECK(angle_difference(-3.0, 3.0) == Approx(2.0 * kPi - 6.0));
}

TEST_CASE("bicycle worked example with L 2.5 m and gamma 15 deg") {
    const KinematicBicycle bike(2.5, 0.7);
    const double gamma = 15.0 * kPi / 180.0;
    CHECK(bike.turning_radius(gamma) == Approx(9.3301).epsilon(1e-4));
    CHECK(bike.yaw_rate(10.0, gamma) == Approx(1.0718).epsilon(1e-4));
    CHECK(std::isinf(bike.turning_radius(0.0)));
}

TEST_CASE("differential drive worked example and inverse") {
    const DifferentialDrive dd(0.033, 0.160);
    const Input body = dd.wheel_to_body(Input(5.0, 10.0));
    CHECK(body[0] == Approx(0.2475));
    CHECK(body[1] == Approx(1.03125));
    const Input wheels = dd.body_to_wheel(body);
    CHECK(wheels[0] == Approx(5.0));
    CHECK(wheels[1] == Approx(10.0));
    // equal wheel speeds drive straight, opposite speeds turn on the spot
    CHECK(dd.wheel_to_body(Input(3.0, 3.0))[1] == Approx(0.0));
    CHECK(dd.wheel_to_body(Input(-3.0, 3.0))[0] == Approx(0.0));
}

TEST_CASE("unicycle exact step closes a full circle") {
    const double v = 1.3, w = 0.4;
    const double T = 2.0 * kPi / w;
    const Pose q0(1.0, -2.0, 0.3);
    const Pose q = unicycle_exact_step(q0, Input(v, w), T);
    CHECK(q[0] == Approx(q0[0]).margin(1e-12));
    CHECK(q[1] == Approx(q0[1]).margin(1e-12));
    // a quarter turn: the displacement is a chord of length sqrt(2) R
    const Pose q4 = unicycle_exact_step(q0, Input(v, w), T / 4.0);
    CHECK((q4.head<2>() - q0.head<2>()).norm() == Approx(std::sqrt(2.0) * v / w));
    // straight-line limit
    const Pose qs = unicycle_exact_step(Pose(0, 0, 0), Input(2.0, 0.0), 1.5);
    CHECK(qs[0] == Approx(3.0));
    CHECK(qs[1] == Approx(0.0).margin(1e-15));
}

TEST_CASE("integrators converge at their theoretical order") {
    // Constant input, exact solution known (eq. 1.16). Halving dt divides the global error by 2^p.
    const Unicycle uni;
    const Pose q0(0.0, 0.0, 0.0);
    const Input u(1.0, 0.8);
    const double T = 4.0;
    auto global_error = [&](Integrator method, int n) {
        InputSequence inputs(n, 2);
        inputs.rowwise() = u.transpose();
        const Trajectory traj = simulate(uni, q0, inputs, T / n, method);
        const Pose exact = unicycle_exact_step(q0, u, T);
        return (traj.row(n).transpose() - exact).norm();
    };
    struct Case {
        Integrator method;
        double order;
    };
    for (const Case c :
         {Case{Integrator::Euler, 1.0}, Case{Integrator::Midpoint, 2.0}, Case{Integrator::RK4, 4.0}}) {
        const double e1 = global_error(c.method, 40);
        const double e2 = global_error(c.method, 80);
        const double observed = std::log2(e1 / e2);
        CHECK(observed == Approx(c.order).margin(0.15));
    }
}

TEST_CASE("rear-axle bicycle is a unicycle with omega = v tan(gamma) / L") {
    const KinematicBicycle bike(1.2, 0.7);
    const double v = 2.0, gamma = 0.3;
    const Pose q(0.5, 0.1, -1.0);
    const Pose a = bike.derivative(q, Input(v, gamma));
    const Pose b = Unicycle{}.derivative(q, Input(v, v * std::tan(gamma) / 1.2));
    CHECK((a - b).norm() == Approx(0.0).margin(1e-14));
    // steering beyond the limit is saturated
    CHECK(bike.derivative(q, Input(v, 1.5))[2] == Approx(v * std::tan(0.7) / 1.2));
}

TEST_CASE("bicycle trajectories are circles of the predicted radius") {
    const double L = 2.0, gamma = 0.25;
    const double l_r = GENERATE(0.0, 0.8, 2.0);
    const KinematicBicycle bike(L, 0.7, l_r);
    InputSequence inputs(400, 2);
    inputs.rowwise() = Input(1.5, gamma).transpose();
    const Trajectory traj = simulate(bike, Pose(0, 0, 0), inputs, 0.02);
    // centre of rotation (ICR) for a start at the origin facing +x: R_rear to the left of the rear axle,
    // and the reference point sits l_r ahead of the rear axle
    const double R_rear = L / std::tan(gamma);
    const Eigen::Vector2d icr(-l_r, R_rear);
    const double expected = std::hypot(R_rear, l_r);
    for (Eigen::Index k = 0; k < traj.rows(); k += 37) {
        CHECK((traj.row(k).head<2>().transpose() - icr).norm() == Approx(expected).epsilon(1e-8));
    }
}

TEST_CASE("rolling constraint holds on the rear axle and fails by -v sin(beta) elsewhere") {
    const double v = 1.2, gamma = 0.4;
    const Pose q(0.0, 0.0, 0.9);
    CHECK(nonholonomic_residual(q, Unicycle{}.derivative(q, Input(v, 0.5))) == Approx(0.0).margin(1e-15));
    const KinematicBicycle rear(1.0, 0.7, 0.0);
    CHECK(nonholonomic_residual(q, rear.derivative(q, Input(v, gamma))) == Approx(0.0).margin(1e-15));
    const KinematicBicycle mid(1.0, 0.7, 0.5);
    CHECK(nonholonomic_residual(q, mid.derivative(q, Input(v, gamma))) ==
          Approx(-v * std::sin(mid.slip_angle(gamma))));
}

TEST_CASE("steering_for inverts yaw_rate") {
    const double l_r = GENERATE(0.0, 0.4, 1.0);
    const KinematicBicycle bike(1.0, 1.2, l_r);
    for (double gamma : {-1.0, -0.3, 0.0, 0.2, 0.9}) {
        for (double v : {-2.0, 0.5, 3.0}) {
            CHECK(bike.steering_for(v, bike.yaw_rate(v, gamma)) == Approx(gamma).margin(1e-12));
        }
    }
    CHECK(bike.steering_for(0.0, 1.0) == 0.0);
}

TEST_CASE("parallel-parking manoeuvre moves sideways along the Lie bracket") {
    // drive eps, turn eps, drive -eps, turn -eps; flows are exact (eq. 1.16 with v or omega zero)
    auto manoeuvre = [](double eps) {
        Pose q(0.0, 0.0, 0.0);
        q = unicycle_exact_step(q, Input(1.0, 0.0), eps);
        q = unicycle_exact_step(q, Input(0.0, 1.0), eps);
        q = unicycle_exact_step(q, Input(-1.0, 0.0), eps);
        q = unicycle_exact_step(q, Input(0.0, -1.0), eps);
        return q;
    };
    // worked example in the theory file, eps = 0.1
    const Pose q = manoeuvre(0.1);
    CHECK(q[0] == Approx(0.1 * (1.0 - std::cos(0.1))));
    CHECK(q[1] == Approx(-0.1 * std::sin(0.1)));
    CHECK(q[0] == Approx(0.000499583).epsilon(1e-6));
    CHECK(q[1] == Approx(-0.00998334).epsilon(1e-6));
    CHECK(q[2] == Approx(0.0).margin(1e-15));
    // displacement / eps^2 tends to the bracket
    const Pose bracket = unicycle_lie_bracket(Pose(0, 0, 0));
    const double eps = 1e-3;
    CHECK(((manoeuvre(eps) / (eps * eps)) - bracket).norm() < 1e-3);
}

TEST_CASE("rate_limit respects value and rate bounds") {
    RateLimits lim;
    lim.min_value = 0.0;
    lim.max_value = 6.0;
    lim.min_rate = -4.0;
    lim.max_rate = 2.0;
    const double dt = 0.1;
    CHECK(rate_limit(1.0, 5.0, lim, dt) == Approx(1.2));   // accelerate at most 2 m/s^2
    CHECK(rate_limit(3.0, 0.0, lim, dt) == Approx(2.6));   // brake at most 4 m/s^2
    CHECK(rate_limit(1.0, 1.1, lim, dt) == Approx(1.1));   // small change passes through
    CHECK(rate_limit(5.9, 9.0, lim, dt) == Approx(6.0));   // saturated
    CHECK(rate_limit(0.1, -5.0, lim, dt) == Approx(0.0));  // cannot go below the minimum
    CHECK(rate_limit(1.0, 5.0, RateLimits{}, dt) == Approx(5.0));
}

TEST_CASE("invalid parameters throw") {
    CHECK_THROWS(KinematicBicycle(0.0, 0.5));
    CHECK_THROWS(KinematicBicycle(1.0, 2.0));
    CHECK_THROWS(KinematicBicycle(1.0, 0.5, 1.5));
    CHECK_THROWS(DifferentialDrive(0.0, 0.1));
}
