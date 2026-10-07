// Chapter 2 tests: every grid algorithm against brute force on random maps, plus the worked example of
// 1_theory/02_configuration_space.md.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <set>

#include "motion_planning/angles.hpp"
#include "motion_planning/grid.hpp"

using namespace motion_planning;
using Catch::Approx;

namespace {

OccupancyGrid random_grid(std::mt19937& rng, int w, int h, double density, double res = 1.0) {
    OccupancyGrid g(w, h, res, Eigen::Vector2d(-1.0, 2.0));
    std::bernoulli_distribution occ(density);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) g.set({x, y}, occ(rng));
    return g;
}

// distance from point p to the closed axis-aligned box [lo, hi]
double point_box_distance(const Eigen::Vector2d& p, const Eigen::Vector2d& lo, const Eigen::Vector2d& hi) {
    const Eigen::Vector2d d = (lo - p).cwiseMax(p - hi).cwiseMax(0.0);
    return d.norm();
}

// distance from segment [a, b] to the closed box, by fine sampling (only used as an upper bound)
double segment_box_distance(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& lo,
                            const Eigen::Vector2d& hi) {
    // the distance is convex along the segment: ternary search finds the minimum
    double l = 0.0, r = 1.0;
    for (int i = 0; i < 200; ++i) {
        const double m1 = l + (r - l) / 3, m2 = r - (r - l) / 3;
        if (point_box_distance(a + m1 * (b - a), lo, hi) < point_box_distance(a + m2 * (b - a), lo, hi))
            r = m2;
        else
            l = m1;
    }
    return point_box_distance(a + 0.5 * (l + r) * (b - a), lo, hi);
}

}  // namespace

TEST_CASE("cell and world coordinates") {
    OccupancyGrid g(4, 3, 0.5, Eigen::Vector2d(-1.0, 2.0));
    CHECK(g.world_to_cell({-1.0, 2.0}) == Cell{0, 0});
    CHECK(g.world_to_cell({-0.01, 2.6}) == Cell{1, 1});
    CHECK(g.world_to_cell({-1.01, 2.0}) == Cell{-1, 0});
    CHECK(g.cell_center({1, 1}).isApprox(Eigen::Vector2d(-0.25, 2.75)));
    CHECK(g.occupied({-1, 0}));  // outside counts as occupied
    CHECK_FALSE(g.occupied({0, 0}));
    g.set({3, 2}, true);
    CHECK(g.to_array()(2, 3) == 1);
    CHECK(OccupancyGrid(g.to_array()).occupied({3, 2}));
}

TEST_CASE("worked example: brushfire and Euclidean distance on a 5 by 3 grid") {
    OccupancyGrid g(5, 3);
    g.set({0, 0}, true);
    g.set({4, 2}, true);
    const FieldArray b4 = brushfire(g, Connectivity::Four);
    const FieldArray b8 = brushfire(g, Connectivity::Eight);
    const FieldArray e = distance_transform(g);
    CHECK(b4(1, 2) == 3.0);
    CHECK(b8(1, 2) == 2.0);
    CHECK(e(1, 2) == Approx(std::sqrt(5.0)));
    CHECK(e(0, 4) == Approx(2.0));
    CHECK(b4(0, 4) == 2.0);
    CHECK(e(2, 0) == Approx(2.0));
}

TEST_CASE("brushfire equals the L1 and L-infinity distance to the obstacle set") {
    std::mt19937 rng(7);
    for (int trial = 0; trial < 20; ++trial) {
        const OccupancyGrid g = random_grid(rng, 17, 11, trial % 2 ? 0.05 : 0.2);
        const FieldArray b4 = brushfire(g, Connectivity::Four), b8 = brushfire(g, Connectivity::Eight);
        for (int y = 0; y < g.height(); ++y)
            for (int x = 0; x < g.width(); ++x) {
                double l1 = std::numeric_limits<double>::infinity(), linf = l1;
                for (int oy = 0; oy < g.height(); ++oy)
                    for (int ox = 0; ox < g.width(); ++ox)
                        if (g.occupied({ox, oy})) {
                            l1 = std::min(l1, double(std::abs(ox - x) + std::abs(oy - y)));
                            linf = std::min(linf, double(std::max(std::abs(ox - x), std::abs(oy - y))));
                        }
                CHECK(b4(y, x) == l1);
                CHECK(b8(y, x) == linf);
            }
    }
}

TEST_CASE("one-dimensional squared distance transform equals brute force") {
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> val(0.0, 30.0);
    std::bernoulli_distribution inf(0.4);
    for (int trial = 0; trial < 200; ++trial) {
        const int n = 1 + trial % 25;
        Eigen::VectorXd f(n);
        for (int i = 0; i < n; ++i) f[i] = inf(rng) ? std::numeric_limits<double>::infinity() : val(rng);
        const Eigen::VectorXd d = squared_distance_transform_1d(f);
        for (int p = 0; p < n; ++p) {
            double best = std::numeric_limits<double>::infinity();
            for (int q = 0; q < n; ++q) best = std::min(best, (p - q) * double(p - q) + f[q]);
            if (std::isinf(best))
                CHECK(std::isinf(d[p]));
            else
                CHECK(d[p] == Approx(best));
        }
    }
}

TEST_CASE("Euclidean distance transform equals brute force") {
    std::mt19937 rng(11);
    const double res = GENERATE(1.0, 0.25);
    for (int trial = 0; trial < 20; ++trial) {
        const OccupancyGrid g = random_grid(rng, 23, 14, 0.02 + 0.03 * (trial % 5), res);
        const FieldArray e = distance_transform(g);
        for (int y = 0; y < g.height(); ++y)
            for (int x = 0; x < g.width(); ++x) {
                double best = std::numeric_limits<double>::infinity();
                for (int oy = 0; oy < g.height(); ++oy)
                    for (int ox = 0; ox < g.width(); ++ox)
                        if (g.occupied({ox, oy})) best = std::min(best, std::hypot(ox - x, oy - y) * res);
                if (std::isinf(best))
                    CHECK(std::isinf(e(y, x)));
                else
                    CHECK(e(y, x) == Approx(best).margin(1e-12));
            }
    }
    CHECK(std::isinf(distance_transform(OccupancyGrid(4, 4))(2, 2)));
}

TEST_CASE("inflation equals the brute-force dilation") {
    std::mt19937 rng(5);
    const OccupancyGrid g = random_grid(rng, 30, 20, 0.03, 0.1);
    const double radius = 0.35;
    const OccupancyGrid inf = inflate(g, radius);
    for (int y = 0; y < g.height(); ++y)
        for (int x = 0; x < g.width(); ++x) {
            bool expected = false;
            for (int oy = 0; oy < g.height() && !expected; ++oy)
                for (int ox = 0; ox < g.width() && !expected; ++ox)
                    expected = g.occupied({ox, oy}) && std::hypot(ox - x, oy - y) * 0.1 <= radius + 1e-12;
            CHECK(inf.occupied({x, y}) == expected);
        }
}

TEST_CASE("disc_free is conservative") {
    std::mt19937 rng(9);
    const OccupancyGrid g = random_grid(rng, 30, 30, 0.02, 0.2);
    const FieldArray d = distance_transform(g);
    std::uniform_real_distribution<double> ux(-1.0, 5.0), uy(2.0, 8.0), ur(0.05, 1.0);
    int free_count = 0;
    for (int i = 0; i < 3000; ++i) {
        const Eigen::Vector2d p(ux(rng), uy(rng));
        const double r = ur(rng);
        if (!disc_free(g, d, p, r)) continue;
        ++free_count;
        for (int y = 0; y < g.height(); ++y)
            for (int x = 0; x < g.width(); ++x)
                if (g.occupied({x, y})) {
                    const Eigen::Vector2d lo = g.origin() + 0.2 * Eigen::Vector2d(x, y);
                    REQUIRE(point_box_distance(p, lo, lo + Eigen::Vector2d(0.2, 0.2)) > r);
                }
    }
    CHECK(free_count > 100);
}

TEST_CASE("segment traversal visits exactly the cells the segment passes through") {
    std::mt19937 rng(13);
    OccupancyGrid g(40, 30, 0.5, Eigen::Vector2d(-3.0, 1.0));
    std::uniform_real_distribution<double> ux(-3.0, 17.0), uy(1.0, 16.0);
    for (int trial = 0; trial < 300; ++trial) {
        const Eigen::Vector2d a(ux(rng), uy(rng)), b(ux(rng), uy(rng));
        const std::vector<Cell> cells = traverse_segment(g, a, b);
        REQUIRE(cells.front() == g.world_to_cell(a));
        REQUIRE(cells.back() == g.world_to_cell(b));
        std::set<std::pair<int, int>> visited;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            visited.insert({cells[i].x, cells[i].y});
            if (i > 0)
                CHECK(std::abs(cells[i].x - cells[i - 1].x) + std::abs(cells[i].y - cells[i - 1].y) == 1);
            const Eigen::Vector2d lo = g.origin() + 0.5 * Eigen::Vector2d(cells[i].x, cells[i].y);
            CHECK(segment_box_distance(a, b, lo, lo + Eigen::Vector2d(0.5, 0.5)) < 1e-9);
        }
        for (int k = 0; k <= 2000; ++k) {  // every sampled point lies in a visited cell
            const Cell c = g.world_to_cell(a + (b - a) * (k / 2000.0));
            CHECK(visited.count({c.x, c.y}) == 1);
        }
    }
}

TEST_CASE("Bresenham misses cells that the segment clips") {
    // worked example: centre of (0, 0) to centre of (7, 2); the segment crosses y = 1 at x = 2.25
    const std::vector<Cell> line = bresenham({0, 0}, {7, 2});
    CHECK(line.front() == Cell{0, 0});
    CHECK(line.back() == Cell{7, 2});
    CHECK(line.size() == 8);  // one cell per column
    OccupancyGrid g(9, 4);
    const std::vector<Cell> exact = traverse_segment(g, g.cell_center({0, 0}), g.cell_center({7, 2}));
    CHECK(exact.size() == 10);  // eight columns plus the two row changes
    // cell (2, 0) is clipped by the segment but skipped by Bresenham
    g.set({2, 0}, true);
    bool bresenham_hits = false;
    for (const Cell& c : line) bresenham_hits |= g.occupied(c);
    CHECK_FALSE(bresenham_hits);
    CHECK_FALSE(segment_free(g, g.cell_center({0, 0}), g.cell_center({7, 2})));
}

TEST_CASE("footprint test agrees with dense sampling") {
    std::mt19937 rng(17);
    const OccupancyGrid g = random_grid(rng, 25, 25, 0.04, 0.4);
    const RectangleFootprint fp{0.3, 1.5, 0.45};
    std::uniform_real_distribution<double> ux(-1.0, 9.0), uy(2.0, 12.0), ut(-kPi, kPi);
    int collisions = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const Pose q(ux(rng), uy(rng), ut(rng));
        const bool free = footprint_free(g, q, fp);
        collisions += !free;
        auto sampled_hit = [&](double grow) {
            const RectangleFootprint big{fp.rear + grow, fp.front + grow, fp.half_width + grow};
            const Eigen::Matrix<double, 4, 2> c = big.corners(q);
            const Eigen::Vector2d o = c.row(0), e1 = c.row(1) - c.row(0), e2 = c.row(3) - c.row(0);
            for (int i = 0; i <= 120; ++i)
                for (int j = 0; j <= 60; ++j)
                    if (g.occupied(g.world_to_cell(o + e1 * (i / 120.0) + e2 * (j / 60.0)))) return true;
            return false;
        };
        if (free)
            CHECK_FALSE(sampled_hit(0.0));  // no missed collision
        else
            CHECK(sampled_hit(0.02));  // a reported collision is real (up to a sliver of 0.02 m)
    }
    CHECK(collisions > 50);
}

TEST_CASE("configuration-space slice of a square robot is the Minkowski sum") {
    OccupancyGrid g(15, 15);
    g.set({7, 7}, true);
    const RectangleFootprint square{1.2, 1.2, 1.2};
    const OccupancyGrid c0 = configuration_space_slice(g, square, 0.0);
    for (int y = 0; y < 15; ++y)
        for (int x = 0; x < 15; ++x) {
            // reference point at a cell centre collides iff |dx|, |dy| <= 1.2 + 0.5 — or the robot leaves the
            // map
            const bool hits_cell = std::abs(x - 7) <= 1 && std::abs(y - 7) <= 1;
            const bool hits_border = x < 1 || y < 1 || x > 13 || y > 13;  // spans [x - 0.7, x + 1.7]
            CHECK(c0.occupied({x, y}) == (hits_cell || hits_border));
        }
    // rotated by 45 degrees the square reaches sqrt(2) * 1.2 = 1.70 along the axes, less along the diagonals
    const OccupancyGrid c45 = configuration_space_slice(g, square, kPi / 4);
    CHECK(c45.occupied({9, 7}));
    CHECK_FALSE(c45.occupied({9, 9}));
}
