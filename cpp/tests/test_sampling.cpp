// Chapter 5 tests: sampling planners against the exact shortest path of the visibility graph (chapter 4) and
// the closed form around a square; tree bookkeeping after rewiring; collision-free outputs.
// The comparison with OMPL is in python/tests/test_ompl_reference.py.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "motion_planning/angles.hpp"
#include "motion_planning/sampling.hpp"

using namespace motion_planning;
using Catch::Approx;

namespace {

Polygon square() {
    Polygon sq(4, 2);
    sq << 1, -1, 3, -1, 3, 1, 1, 1;
    return sq;
}

PlanningProblem2D square_problem() {
    return problem_from_polygons({square()}, Eigen::Vector2d(-0.5, -2.5), Eigen::Vector2d(4.5, 2.5),
                                 Eigen::Vector2d(0, 0), Eigen::Vector2d(4, 0));
}

const double kOptimal = 2.0 + 2.0 * std::sqrt(2.0);  // around the square, see test_graph_search.cpp

bool path_valid(const PlanningProblem2D& p, const std::vector<Eigen::Vector2d>& path) {
    for (std::size_t i = 1; i < path.size(); ++i)
        if (!p.motion_valid(path[i - 1], path[i])) return false;
    return true;
}

}  // namespace

TEST_CASE("radius of eq. 5.3 and the optimal gamma") {
    CHECK(optimal_gamma(kPi) == Approx(std::sqrt(6.0)));
    CHECK(rewiring_radius(100, 2.0) == Approx(2.0 * std::sqrt(std::log(100.0) / 100.0)));
    CHECK(rewiring_radius(1000, 2.0) < rewiring_radius(100, 2.0));
    CHECK(std::isinf(rewiring_radius(1, 2.0)));
}

TEST_CASE("RRT finds a collision-free path that is never shorter than the optimum") {
    const PlanningProblem2D p = square_problem();
    for (std::uint32_t seed = 1; seed <= 20; ++seed) {
        RrtOptions opt;
        opt.seed = seed;
        opt.step = 0.4;
        const TreeResult r = rrt(p, opt);
        REQUIRE(r.found);
        CHECK(path_valid(p, r.path));
        CHECK(r.path.front().isApprox(p.start));
        CHECK(r.path.back().isApprox(p.goal));
        CHECK(r.path_cost >= kOptimal - 1e-9);
        CHECK(r.path_cost == Approx(path_length(r.path)));
        for (std::size_t i = 1; i < r.nodes.size(); ++i)  // every edge is at most one step long
            CHECK((r.nodes[i] - r.nodes[r.parent[i]]).norm() <= opt.step + 1e-12);
    }
}

TEST_CASE("RRT* converges towards the optimum and keeps its cost bookkeeping") {
    const PlanningProblem2D p = square_problem();
    RrtOptions opt;
    opt.max_iterations = 4000;
    opt.step = 0.5;
    opt.seed = 3;
    const TreeResult r = rrt_star(p, opt);
    REQUIRE(r.found);
    CHECK(path_valid(p, r.path));
    CHECK(r.path_cost >= kOptimal - 1e-9);
    CHECK(r.path_cost <= 1.03 * kOptimal);
    // cost-to-come of every node equals its parent's plus the edge, after all the rewiring
    for (std::size_t i = 1; i < r.nodes.size(); ++i)
        CHECK(r.cost[i] == Approx(r.cost[r.parent[i]] + (r.nodes[i] - r.nodes[r.parent[i]]).norm()));
    // the best cost never increases
    for (std::size_t k = 1; k < r.best_cost_history.size(); ++k)
        CHECK(r.best_cost_history[k] <= r.best_cost_history[k - 1]);
    // and RRT* beats plain RRT on average over seeds
    double rrt_sum = 0.0, star_sum = 0.0;
    for (std::uint32_t seed = 1; seed <= 10; ++seed) {
        RrtOptions o;
        o.seed = seed;
        o.max_iterations = 1500;
        o.stop_at_first_solution = false;
        rrt_sum += rrt(p, o).path_cost;
        star_sum += rrt_star(p, o).path_cost;
    }
    CHECK(star_sum < rrt_sum);
}

TEST_CASE("PRM paths are valid, above the optimum, and close to it with enough samples") {
    const PlanningProblem2D p = square_problem();
    PrmOptions opt;
    opt.num_samples = 1500;
    const PrmResult r = prm(p, opt);
    REQUIRE(r.query.found);
    std::vector<Eigen::Vector2d> path;
    for (int i : r.query.path) path.push_back(r.roadmap.point(i));
    CHECK(path_valid(p, path));
    CHECK(r.query.cost >= kOptimal - 1e-9);
    CHECK(r.query.cost <= 1.06 * kOptimal);
    // every roadmap edge is collision-free
    for (int i = 0; i < r.roadmap.size(); ++i)
        for (const auto& [j, c] : r.roadmap.edges(i))
            CHECK(p.motion_valid(r.roadmap.point(i), r.roadmap.point(j)));
    // a radius smaller than the gap between samples leaves start and goal disconnected
    opt.num_samples = 50;
    opt.connection_radius = 0.05;
    CHECK_FALSE(prm(p, opt).query.found);
}

TEST_CASE("planners on an occupancy grid, goal regions and determinism") {
    OccupancyGrid g(60, 40, 0.1);
    for (int y = 0; y < 30; ++y) g.set({30, y}, true);  // a wall with a gap at the top
    const PlanningProblem2D p = problem_from_grid(g, Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(5.0, 1.0));
    RrtOptions opt;
    opt.step = 0.3;
    opt.max_iterations = 20000;
    const TreeResult a = rrt(p, opt), b = rrt(p, opt);
    REQUIRE(a.found);
    CHECK(path_valid(p, a.path));
    CHECK(a.path_cost == b.path_cost);                      // same seed, same tree
    CHECK(a.path_cost >= 2 * std::hypot(2.0, 2.0) - 1e-9);  // must climb over the wall end at y = 3

    PlanningProblem2D region = p;
    region.goal_radius = 0.5;
    const TreeResult c = rrt(region, opt);
    REQUIRE(c.found);
    CHECK((c.path.back() - region.goal).norm() <= 0.5);

    PlanningProblem2D bad = p;
    bad.start = Eigen::Vector2d(3.05, 1.0);  // inside the wall
    CHECK_THROWS(rrt(bad, opt));
}
