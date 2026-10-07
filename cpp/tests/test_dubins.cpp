// Chapter 6 tests: every Dubins word is integrated with the exact unicycle flow of chapter 1 and must land on
// the goal pose; closed-form lengths from 1_theory/06_dubins_paths.md; Dubins-RRT* against the obstacle-free
// optimum. OMPL's Dubins distance is compared in python/tests/test_ompl_reference.py.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>

#include "motion_planning/angles.hpp"
#include "motion_planning/dubins.hpp"

using namespace motion_planning;
using Catch::Approx;

namespace {
double pose_error(const Pose& a, const Pose& b) {
    return (a.head<2>() - b.head<2>()).norm() + std::abs(angle_difference(a[2], b[2]));
}
}  // namespace

TEST_CASE("every feasible word ends exactly at the goal pose") {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> ux(-6.0, 6.0), ut(-kPi, kPi), ur(0.3, 2.5);
    int feasible[6] = {0, 0, 0, 0, 0, 0};
    for (int trial = 0; trial < 2000; ++trial) {
        const Pose q0(ux(rng), ux(rng), ut(rng)), q1(ux(rng), ux(rng), ut(rng));
        const double r = ur(rng);
        for (const DubinsPath& p : all_dubins_paths(q0, q1, r)) {
            ++feasible[static_cast<int>(p.word)];
            INFO(to_string(p.word));
            CHECK(pose_error(p.at(p.length()), q1) < 1e-8);
            for (double l : p.lengths) CHECK(l >= 0.0);
        }
    }
    for (int w = 0; w < 6; ++w) CHECK(feasible[w] > 0);  // every word occurs
    CHECK(feasible[0] == 2000);                          // LSL and RSR always exist
    CHECK(feasible[1] == 2000);
}

TEST_CASE("closed-form Dubins lengths") {
    const double r = 1.5;
    // straight ahead
    CHECK(dubins_distance(Pose(0, 0, 0), Pose(5, 0, 0), r) == Approx(5.0));
    // a quarter circle to the left
    CHECK(dubins_distance(Pose(0, 0, 0), Pose(r, r, kPi / 2), r) == Approx(kPi * r / 2));
    // a U-turn onto the parallel lane 2r to the left: a half circle
    CHECK(dubins_distance(Pose(0, 0, 0), Pose(0, 2 * r, kPi), r) == Approx(kPi * r));
    // turning around on the spot is impossible: going back to the start heading reversed costs a loop
    const DubinsPath back = shortest_dubins_path(Pose(0, 0, 0), Pose(0, 0, kPi), r);
    CHECK(back.length() > kPi * r);
}

TEST_CASE("shortest path properties") {
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> ux(-4.0, 4.0), ut(-kPi, kPi);
    int ccc_best = 0, asymmetric = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const Pose q0(ux(rng), ux(rng), ut(rng)), q1(ux(rng), ux(rng), ut(rng));
        const DubinsPath best = shortest_dubins_path(q0, q1, 1.0);
        for (const DubinsPath& p : all_dubins_paths(q0, q1, 1.0)) CHECK(best.length() <= p.length() + 1e-12);
        CHECK(best.length() >= (q1.head<2>() - q0.head<2>()).norm() - 1e-12);  // no shorter than straight
        ccc_best += best.word == DubinsWord::RLR || best.word == DubinsWord::LRL;
        asymmetric += std::abs(best.length() - dubins_distance(q1, q0, 1.0)) > 1e-6;
    }
    CHECK(ccc_best > 0);    // CCC words do win, for nearby poses
    CHECK(asymmetric > 0);  // d(q0, q1) != d(q1, q0)
}

TEST_CASE("sampling and truncation") {
    const DubinsPath p = shortest_dubins_path(Pose(0, 0, 0), Pose(3, 4, 1.0), 1.0);
    const Trajectory s = p.sample(0.1);
    CHECK(pose_error(s.row(0).transpose(), p.start) < 1e-12);
    CHECK(pose_error(s.row(s.rows() - 1).transpose(), Pose(3, 4, 1.0)) < 1e-8);
    for (Eigen::Index k = 1; k < s.rows(); ++k)
        CHECK((s.row(k) - s.row(k - 1)).head<2>().norm() <= 0.1 + 1e-12);
    const DubinsPath h = p.truncated(p.length() / 2);
    CHECK(h.length() == Approx(p.length() / 2));
    CHECK(pose_error(h.at(h.length()), p.at(p.length() / 2)) < 1e-12);
}

TEST_CASE("Dubins-RRT* in free space approaches the Dubins distance") {
    PlanningProblem2D free;
    free.lower = Eigen::Vector2d(0, 0);
    free.upper = Eigen::Vector2d(10, 10);
    free.state_valid = [&](const Eigen::Vector2d& x) {
        return (x.array() >= 0.0).all() && (x.array() <= 10.0).all();
    };
    free.motion_valid = [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return free.state_valid(a) && free.state_valid(b);
    };
    const Pose start(2, 2, 0), goal(8, 7, kPi / 2);
    DubinsPlannerOptions opt;
    opt.radius = 1.0;
    opt.max_iterations = 1500;
    const DubinsTreeResult t = dubins_rrt_star(free, start, goal, opt);
    REQUIRE(t.found);
    const double optimum = dubins_distance(start, goal, 1.0);
    CHECK(t.path_cost >= optimum - 1e-9);
    CHECK(t.path_cost <= 1.1 * optimum);
    // the segments chain up: each starts where the previous one ends, and the last ends at the goal
    CHECK(pose_error(t.segments.front().start, start) < 1e-12);
    double total = 0.0;
    for (std::size_t i = 0; i < t.segments.size(); ++i) {
        total += t.segments[i].length();
        if (i > 0)
            CHECK(pose_error(t.segments[i].start, t.segments[i - 1].at(t.segments[i - 1].length())) < 1e-8);
    }
    CHECK(pose_error(t.segments.back().at(t.segments.back().length()), goal) < 1e-8);
    CHECK(total == Approx(t.path_cost));
    for (std::size_t i = 1; i < t.nodes.size(); ++i)
        CHECK(t.cost[i] >= t.cost[t.parent[i]] + (t.nodes[i] - t.nodes[t.parent[i]]).head<2>().norm() - 1e-9);
}

TEST_CASE("Dubins-RRT* avoids obstacles on a grid") {
    OccupancyGrid g(100, 60, 0.1);
    for (int y = 0; y < 45; ++y)
        for (int x = 48; x < 52; ++x) g.set({x, y}, true);  // a wall from the bottom, open at the top
    const PlanningProblem2D p = problem_from_grid(g, Eigen::Vector2d(2, 1), Eigen::Vector2d(8, 1));
    DubinsPlannerOptions opt;
    opt.radius = 0.6;
    opt.max_iterations = 3000;
    opt.star = false;
    const DubinsTreeResult t = dubins_rrt_star(p, Pose(2, 1, kPi / 2), Pose(8, 1, -kPi / 2), opt);
    REQUIRE(t.found);
    for (const DubinsPath& s : t.segments) CHECK(dubins_path_valid(s, p, 0.02));
    CHECK(t.path_cost >= 2 * 3.5 + 6.0 - 4 * 0.6 - 1e-6);  // must climb over y = 4.5
}
