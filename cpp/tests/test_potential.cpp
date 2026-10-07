// Chapter 3 tests: potentials against their closed forms and finite differences, the wave-front against the
// hand-worked example and closed-form distances of 1_theory/03_potential_fields.md, and the local-minimum
// failure of the summed potential.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>

#include "motion_planning/potential.hpp"

using namespace motion_planning;
using Catch::Approx;

namespace {

// A U-shaped obstacle open towards -x, between a start on the left and a goal on the right.
OccupancyGrid trap_grid() {
    OccupancyGrid g(40, 30, 0.2);
    for (int y = 8; y <= 22; ++y)
        for (int x = 20; x <= 21; ++x) g.set({x, y}, true);
    for (int x = 13; x <= 21; ++x)
        for (int y : {8, 9, 21, 22}) g.set({x, y}, true);
    return g;
}

}  // namespace

TEST_CASE("attractive potential is quadratic then conic, continuous with a continuous gradient") {
    const PotentialParams k{2.0, 1.5, 1.0, 1.0};
    const Eigen::Vector2d goal(1.0, -1.0);
    CHECK(attractive_potential(goal + Eigen::Vector2d(1.0, 0.0), goal, k) == Approx(0.5 * 2.0 * 1.0));
    CHECK(attractive_potential(goal + Eigen::Vector2d(0.0, 3.0), goal, k) ==
          Approx(1.5 * 2.0 * 3.0 - 0.5 * 2.0 * 2.25));
    const Eigen::Vector2d u(0.6, 0.8);
    const double eps = 1e-9;
    CHECK(attractive_potential(goal + (1.5 - eps) * u, goal, k) ==
          Approx(attractive_potential(goal + (1.5 + eps) * u, goal, k)));
    // gradient against central differences, on both sides of d_star
    for (double d : {0.4, 1.4, 1.6, 5.0}) {
        const Eigen::Vector2d p = goal + d * u;
        const double h = 1e-6;
        Eigen::Vector2d fd;
        for (int i = 0; i < 2; ++i) {
            Eigen::Vector2d e = Eigen::Vector2d::Zero();
            e[i] = h;
            fd[i] = (attractive_potential(p + e, goal, k) - attractive_potential(p - e, goal, k)) / (2 * h);
        }
        CHECK((attractive_gradient(p, goal, k) - fd).norm() < 1e-6);
    }
    // the conic part has constant gradient magnitude zeta d_star
    CHECK(attractive_gradient(goal + 7.0 * u, goal, k).norm() == Approx(2.0 * 1.5));
}

TEST_CASE("repulsive potential") {
    const PotentialParams k{1.0, 1.0, 3.0, 0.8};
    CHECK(repulsive_potential(0.9, k) == 0.0);
    CHECK(repulsive_potential(0.8, k) == Approx(0.0).margin(1e-15));
    CHECK(repulsive_potential(0.4, k) == Approx(0.5 * 3.0 * (2.5 - 1.25) * (2.5 - 1.25)));
    CHECK(std::isinf(repulsive_potential(0.0, k)));
    CHECK(repulsive_potential(0.1, k) > repulsive_potential(0.2, k));
}

TEST_CASE("wave-front worked example") {
    // y=2:  .  .  .  .
    // y=1:  .  #  #  .
    // y=0:  G  .  .  .
    OccupancyGrid g(4, 3);
    g.set({1, 1}, true);
    g.set({2, 1}, true);
    const FieldArray w = wavefront(g, {0, 0}, Connectivity::Four);
    const double expected[3][4] = {{0, 1, 2, 3}, {1, -1, -1, 4}, {2, 3, 4, 5}};
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x) {
            if (expected[y][x] < 0)
                CHECK(std::isinf(w(y, x)));
            else
                CHECK(w(y, x) == expected[y][x]);
        }
}

TEST_CASE("wave-front on an empty map is the L1 or L-infinity distance to the goal") {
    OccupancyGrid g(13, 9);
    const Cell goal{4, 6};
    const FieldArray w4 = wavefront(g, goal, Connectivity::Four),
                     w8 = wavefront(g, goal, Connectivity::Eight);
    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 13; ++x) {
            CHECK(w4(y, x) == std::abs(x - goal.x) + std::abs(y - goal.y));
            CHECK(w8(y, x) == std::max(std::abs(x - goal.x), std::abs(y - goal.y)));
        }
}

TEST_CASE("descent on the wave-front reaches the goal from every reachable cell") {
    std::mt19937 rng(21);
    std::bernoulli_distribution occ(0.25);
    for (int trial = 0; trial < 10; ++trial) {
        OccupancyGrid g(25, 18);
        for (int y = 0; y < 18; ++y)
            for (int x = 0; x < 25; ++x) g.set({x, y}, occ(rng));
        const Cell goal{12, 9};
        g.set(goal, false);
        for (Connectivity c : {Connectivity::Four, Connectivity::Eight}) {
            const FieldArray w = wavefront(g, goal, c);
            for (int y = 0; y < 18; ++y)
                for (int x = 0; x < 25; ++x) {
                    if (!std::isfinite(w(y, x))) continue;
                    const DescentResult r = descend(w, {x, y}, goal, c);
                    REQUIRE(r.reached_goal);
                    CHECK(r.path.size() == static_cast<std::size_t>(w(y, x)) + 1);  // one step per level
                }
        }
    }
    OccupancyGrid blocked(3, 3);
    blocked.set({1, 1}, true);
    CHECK_THROWS(wavefront(blocked, {1, 1}));
}

TEST_CASE("the summed potential traps the robot inside a U-shaped obstacle") {
    const OccupancyGrid g = trap_grid();
    const PotentialParams k{1.0, 2.0, 1.0, 1.0};
    const Cell start{5, 15}, goal{35, 15};
    const Eigen::Vector2d goal_xy = g.cell_center(goal);
    const FieldArray U = total_field(g, goal_xy, k);
    const DescentResult r = descend(U, start, goal, Connectivity::Eight);
    CHECK(r.local_minimum);
    CHECK_FALSE(r.reached_goal);
    // the final cell really is a local minimum, inside the U
    const Cell m = r.path.back();
    for (const Cell& d : neighbour_offsets(Connectivity::Eight))
        CHECK(U(m.y + d.y, m.x + d.x) >= U(m.y, m.x));
    CHECK(m.x > 13);
    CHECK(m.x < 20);
    // the wave-front has no such minimum
    const DescentResult w =
        descend(wavefront(g, goal, Connectivity::Eight), start, goal, Connectivity::Eight);
    CHECK(w.reached_goal);
    // and from a start that does not face the trap the summed potential works
    CHECK(descend(U, {5, 2}, goal, Connectivity::Eight).reached_goal);
}
