// Chapter 8 tests: the dynamic window by hand, rollouts against the exact arc of chapter 1, the braking
// condition in front of a wall, actuator limits in closed loop, and the trap that a global plan resolves
// (1_theory/08_local_planning.md).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>

#include "motion_planning/angles.hpp"
#include "motion_planning/dwa.hpp"
#include "motion_planning/graph_search.hpp"

using namespace motion_planning;
using Catch::Approx;

namespace {

// 8 m x 6 m room, 0.05 m cells, with an optional U-shaped trap open towards -x around x = 3.5 .. 4.3
OccupancyGrid room(bool trap) {
    OccupancyGrid g(160, 120, 0.05);
    for (int y = 0; y < 120; ++y)
        for (int x = 0; x < 160; ++x) {
            const Eigen::Vector2d p = g.cell_center({x, y});
            bool occ = x == 0 || y == 0 || x == 159 || y == 119;
            if (trap) {
                occ |= p.x() > 4.0 && p.x() < 4.3 && p.y() > 1.5 && p.y() < 4.5;  // back wall
                occ |= p.x() > 2.5 && p.x() < 4.3 && p.y() > 1.5 && p.y() < 1.8;  // lower arm
                occ |= p.x() > 2.5 && p.x() < 4.3 && p.y() > 4.2 && p.y() < 4.5;  // upper arm
            }
            g.set({x, y}, occ);
        }
    return g;
}

}  // namespace

TEST_CASE("dynamic window worked example") {
    DwaConfig cfg;  // a = 0.6 m/s^2, yaw acceleration 3 rad/s^2, period 0.1 s
    const DynamicWindow w = dynamic_window(Input(0.2, 0.0), cfg);
    CHECK(w.v_min == Approx(0.14));
    CHECK(w.v_max == Approx(0.26));
    CHECK(w.w_min == Approx(-0.3));
    CHECK(w.w_max == Approx(0.3));
    const DynamicWindow rest = dynamic_window(Input(0.0, 1.4), cfg);
    CHECK(rest.v_min == 0.0);          // no reversing
    CHECK(rest.w_max == Approx(1.5));  // the yaw-rate limit cuts the window
}

TEST_CASE("rollouts are exact arcs") {
    DwaConfig cfg;
    const Pose q(1.0, 2.0, 0.4);
    const Trajectory t = rollout(q, 0.5, -0.7, cfg);
    CHECK(t.rows() == 21);
    CHECK((t.row(20).transpose() - unicycle_exact_step(q, Input(0.5, -0.7), 2.0)).norm() < 1e-12);
}

TEST_CASE("a fast candidate towards a near wall is not admissible") {
    OccupancyGrid g(80, 40, 0.05);
    for (int y = 0; y < 40; ++y) g.set({60, y}, true);  // wall at x = 3.0 .. 3.05
    const FieldArray D = distance_transform(g);
    DwaConfig cfg;
    cfg.w_samples = 1;  // straight ahead only
    cfg.v_samples = 1;  // the top of the window
    const DwaDecision d = dwa_step(g, D, Pose(2.55, 1.0, 0.0), Input(0.55, 0.0), Eigen::Vector2d(5, 1), cfg);
    REQUIRE(d.candidates.size() == 1);
    const DwaCandidate& c = d.candidates[0];
    CHECK(c.v == Approx(0.6));
    CHECK(c.free_distance > 0.2);
    CHECK(c.free_distance <= 0.3 + 1e-9);  // the 0.2 m disc reaches the wall cells 0.25-0.3 m ahead
    CHECK_FALSE(c.admissible);             // 0.6 > sqrt(2 * 0.3 * 0.6): cannot stop in time
    CHECK_FALSE(d.found);
    // far from the wall the same candidate is fine
    CHECK(dwa_step(g, D, Pose(0.5, 1.0, 0.0), Input(0.55, 0.0), Eigen::Vector2d(5, 1), cfg).found);
}

TEST_CASE("DWA reaches a goal in an open room within its actuator limits and without collisions") {
    const OccupancyGrid g = room(false);
    const FieldArray D = distance_transform(g);
    DwaConfig cfg;
    const DwaRun r = run_dwa(g, Pose(1.0, 1.0, 0.0), Eigen::Vector2d(7.0, 5.0), cfg);
    CHECK(r.reached_goal);
    for (Eigen::Index k = 0; k < r.states.rows(); ++k) {
        const Cell c = g.world_to_cell(r.states.row(k).head<2>().transpose());
        CHECK(D(c.y, c.x) > cfg.robot_radius);
    }
    Input prev = Input::Zero();
    for (Eigen::Index k = 0; k < r.commands.rows(); ++k) {
        const Input u = r.commands.row(k).transpose();
        CHECK(std::abs(u[0] - prev[0]) <= cfg.max_accel * cfg.control_period + 1e-9);
        CHECK(std::abs(u[1] - prev[1]) <= cfg.max_yaw_accel * cfg.control_period + 1e-9);
        CHECK(u[0] <= cfg.max_speed + 1e-12);
        prev = u;
    }
}

TEST_CASE("DWA alone is trapped by a U; behind a global plan it is not") {
    const OccupancyGrid g = room(true);
    const Pose start(1.0, 3.0, 0.0);
    const Eigen::Vector2d goal(6.5, 3.0);
    DwaConfig cfg;
    const DwaRun alone = run_dwa(g, start, goal, cfg);
    CHECK_FALSE(alone.reached_goal);
    CHECK(alone.stuck);

    // global plan: A* on the grid inflated by the robot radius plus a margin (chapters 2 and 4)
    const OccupancyGrid inflated = inflate(g, cfg.robot_radius + 0.15);
    const GridSearchResult plan =
        grid_search(inflated, inflated.world_to_cell(start.head<2>()), inflated.world_to_cell(goal));
    REQUIRE(plan.found);
    Points2D path(static_cast<Eigen::Index>(plan.path.size()), 2);
    for (std::size_t i = 0; i < plan.path.size(); ++i)
        path.row(static_cast<Eigen::Index>(i)) = g.cell_center(plan.path[i]).transpose();
    const DwaRun guided = run_dwa(g, start, goal, cfg, path, 0.6);
    CHECK(guided.reached_goal);
}

TEST_CASE("DWA never enters the collision band, over many random runs") {
    const OccupancyGrid g = room(true);
    const FieldArray D = distance_transform(g);
    DwaConfig cfg;
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> ux(0.3, 7.7), uy(0.3, 5.7), ut(-kPi, kPi);
    auto clear = [&](const Eigen::Vector2d& p) {
        const Cell c = g.world_to_cell(p);
        return D(c.y, c.x);
    };
    for (int trial = 0; trial < 12; ++trial) {
        Eigen::Vector2d s, t;
        do s = Eigen::Vector2d(ux(rng), uy(rng));
        while (clear(s) < 0.4);
        do t = Eigen::Vector2d(ux(rng), uy(rng));
        while (clear(t) < 0.4);
        const DwaRun r = run_dwa(g, Pose(s.x(), s.y(), ut(rng)), t, cfg, Points2D(), 1.0, 0.15, 40.0);
        for (Eigen::Index k = 0; k < r.states.rows(); ++k)
            CHECK(clear(r.states.row(k).head<2>().transpose()) > cfg.robot_radius);
    }
}
