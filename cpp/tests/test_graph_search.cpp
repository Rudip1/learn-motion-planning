// Chapter 4 tests: searches against Floyd–Warshall and Bellman–Ford references, the hand-worked grid of
// 1_theory/04_graph_search.md, the weighted-A* bound, and a visibility graph against its closed form.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>

#include "motion_planning/graph_search.hpp"

using namespace motion_planning;
using Catch::Approx;

namespace {

// 6 x 4 grid of the worked example:   y=3  . . . . . .
//                                     y=2  . # # # . .
//                                     y=1  . . . # . .
//                                     y=0  S . . # . G
OccupancyGrid worked_example() {
    OccupancyGrid g(6, 4);
    for (const Cell c : {Cell{1, 2}, Cell{2, 2}, Cell{3, 2}, Cell{3, 1}, Cell{3, 0}}) g.set(c, true);
    return g;
}

OccupancyGrid random_grid(std::mt19937& rng, int w, int h, double density) {
    OccupancyGrid g(w, h, 0.5);
    std::bernoulli_distribution occ(density);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) g.set({x, y}, occ(rng));
    g.set({0, 0}, false);
    g.set({w - 1, h - 1}, false);
    return g;
}

// Bellman–Ford on the 8-connected grid graph (no corner cutting): an independent reference for Dijkstra.
FieldArray bellman_ford(const OccupancyGrid& g, const Cell& s) {
    FieldArray d = FieldArray::Constant(g.height(), g.width(), std::numeric_limits<double>::infinity());
    d(s.y, s.x) = 0.0;
    for (bool changed = true; changed;) {
        changed = false;
        for (int y = 0; y < g.height(); ++y)
            for (int x = 0; x < g.width(); ++x) {
                if (g.occupied({x, y}) || !std::isfinite(d(y, x))) continue;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if ((dx == 0 && dy == 0) || g.occupied({x + dx, y + dy})) continue;
                        if (dx != 0 && dy != 0 && (g.occupied({x + dx, y}) || g.occupied({x, y + dy})))
                            continue;
                        const double c = d(y, x) + g.resolution() * std::hypot(dx, dy);
                        if (c < d(y + dy, x + dx) - 1e-12) {
                            d(y + dy, x + dx) = c;
                            changed = true;
                        }
                    }
            }
    }
    return d;
}

}  // namespace

TEST_CASE("worked example: 4-connected grid with a wall") {
    const OccupancyGrid g = worked_example();
    GridSearchOptions opt;
    opt.connectivity = Connectivity::Four;
    opt.heuristic = Heuristic::Manhattan;
    for (SearchAlgorithm a :
         {SearchAlgorithm::BreadthFirst, SearchAlgorithm::Dijkstra, SearchAlgorithm::AStar}) {
        opt.algorithm = a;
        const GridSearchResult r = grid_search(g, {0, 0}, {5, 0}, opt);
        REQUIRE(r.found);
        CHECK(r.cost == Approx(11.0));
        CHECK(r.path.size() == 12);
        CHECK(r.path.front() == Cell{0, 0});
        CHECK(r.path.back() == Cell{5, 0});
    }
    opt.algorithm = SearchAlgorithm::Dijkstra;
    const auto dij = grid_search(g, {0, 0}, {5, 0}, opt);
    opt.algorithm = SearchAlgorithm::AStar;
    const auto ast = grid_search(g, {0, 0}, {5, 0}, opt);
    CHECK(ast.expanded.size() <= dij.expanded.size());
}

TEST_CASE("Dijkstra and A* with a zero heuristic match Floyd-Warshall on random graphs") {
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> w(0.0, 10.0);
    std::bernoulli_distribution edge(0.25);
    for (int trial = 0; trial < 30; ++trial) {
        const int n = 12;
        std::vector<std::vector<std::pair<int, double>>> adj(n);
        Eigen::MatrixXd D = Eigen::MatrixXd::Constant(n, n, std::numeric_limits<double>::infinity());
        Eigen::MatrixXd hops = D;
        for (int i = 0; i < n; ++i) D(i, i) = hops(i, i) = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j && edge(rng)) {
                    const double c = w(rng);
                    adj[i].emplace_back(j, c);
                    D(i, j) = std::min(D(i, j), c);
                    hops(i, j) = 1.0;
                }
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    D(i, j) = std::min(D(i, j), D(i, k) + D(k, j));
                    hops(i, j) = std::min(hops(i, j), hops(i, k) + hops(k, j));
                }
        auto nb = [&](int u, std::vector<std::pair<int, double>>& out) {
            for (const auto& e : adj[u]) out.push_back(e);
        };
        auto zero = [](int) { return 0.0; };
        for (int s = 0; s < n; ++s)
            for (int t = 0; t < n; ++t) {
                for (SearchAlgorithm a : {SearchAlgorithm::Dijkstra, SearchAlgorithm::AStar}) {
                    const SearchResult r = best_first_search(n, s, t, nb, zero, a);
                    CHECK(r.found == std::isfinite(D(s, t)));
                    if (r.found) CHECK(r.cost == Approx(D(s, t)).margin(1e-9));
                }
                const SearchResult b = best_first_search(n, s, t, nb, zero, SearchAlgorithm::BreadthFirst);
                if (b.found) CHECK(double(b.path.size() - 1) == hops(s, t));
            }
    }
}

TEST_CASE("grid search: optimal heuristics, the weighted-A* bound, inadmissible Manhattan") {
    std::mt19937 rng(4);
    int manhattan_worse = 0;
    for (int trial = 0; trial < 25; ++trial) {
        const OccupancyGrid g = random_grid(rng, 30, 20, 0.25);
        const Cell s{0, 0}, t{29, 19};
        const double reference = bellman_ford(g, s)(t.y, t.x);
        GridSearchOptions opt;
        opt.algorithm = SearchAlgorithm::Dijkstra;
        const GridSearchResult dij = grid_search(g, s, t, opt);
        REQUIRE(dij.found == std::isfinite(reference));
        if (!dij.found) continue;
        CHECK(dij.cost == Approx(reference));
        opt.algorithm = SearchAlgorithm::AStar;
        for (Heuristic h : {Heuristic::Zero, Heuristic::Euclidean, Heuristic::Octile, Heuristic::Chebyshev}) {
            opt.heuristic = h;
            const GridSearchResult r = grid_search(g, s, t, opt);
            CHECK(r.cost == Approx(reference));
            // an informed heuristic never expands more (h = 0 is Dijkstra up to tie-breaking)
            if (h != Heuristic::Zero) CHECK(r.expanded.size() <= dij.expanded.size());
        }
        opt.heuristic =
            Heuristic::Manhattan;  // overestimates diagonal moves: not admissible on 8-connected grids
        const GridSearchResult man = grid_search(g, s, t, opt);
        CHECK(man.cost >= reference - 1e-9);
        manhattan_worse += man.cost > reference + 1e-9;
        opt.heuristic = Heuristic::Octile;
        opt.algorithm = SearchAlgorithm::WeightedAStar;
        for (double w : {1.5, 3.0}) {
            opt.weight = w;
            const GridSearchResult r = grid_search(g, s, t, opt);
            CHECK(r.cost <= w * reference + 1e-9);  // eq. (4.8)
        }
    }
    CHECK(manhattan_worse > 0);
}

TEST_CASE("unreachable goal, start equals goal, corner cutting, cell costs") {
    OccupancyGrid g(5, 5);
    for (int y = 0; y < 5; ++y) g.set({2, y}, true);
    CHECK_FALSE(grid_search(g, {0, 0}, {4, 4}).found);
    const GridSearchResult same = grid_search(g, {1, 1}, {1, 1});
    CHECK(same.found);
    CHECK(same.cost == 0.0);
    CHECK(same.path.size() == 1);

    OccupancyGrid c(2, 2);
    c.set({1, 0}, true);
    GridSearchOptions opt;
    CHECK(grid_search(c, {0, 0}, {1, 1}, opt).cost == Approx(2.0));  // around the corner
    opt.allow_corner_cutting = true;
    CHECK(grid_search(c, {0, 0}, {1, 1}, opt).cost == Approx(std::sqrt(2.0)));

    // a penalty on a band of cells makes the search detour around it
    OccupancyGrid open(9, 5);
    FieldArray penalty = FieldArray::Zero(5, 9);
    penalty.block(0, 4, 4, 1).setConstant(100.0);  // column x = 4, rows y = 0..3
    GridSearchOptions o4;
    o4.connectivity = Connectivity::Four;
    o4.heuristic = Heuristic::Manhattan;
    const GridSearchResult r = grid_search(open, {0, 0}, {8, 0}, o4, penalty);
    CHECK(r.cost == Approx(16.0));  // up 4, across 8, down 4: no step enters the penalised cells
    CHECK_THROWS(grid_search(open, {0, 0}, {8, 0}, o4, FieldArray::Zero(2, 2)));
}

TEST_CASE("point in polygon and segment clearance") {
    Polygon sq(4, 2);
    sq << 0, 0, 2, 0, 2, 2, 0, 2;
    CHECK(point_in_polygon({1, 1}, sq));
    CHECK_FALSE(point_in_polygon({3, 1}, sq));
    CHECK_FALSE(point_in_polygon({2, 1}, sq));  // boundary counts as outside
    const std::vector<Polygon> obs{sq};
    CHECK(segment_clear_of_polygons({-1, 0}, {3, 0}, obs));         // slides along an edge
    CHECK(segment_clear_of_polygons({0, 0}, {2, 0}, obs));          // the edge itself
    CHECK_FALSE(segment_clear_of_polygons({0, 0}, {2, 2}, obs));    // a diagonal
    CHECK_FALSE(segment_clear_of_polygons({-1, -1}, {3, 3}, obs));  // enters and leaves through vertices
    CHECK_FALSE(segment_clear_of_polygons({-1, 1}, {3, 1}, obs));   // proper crossing
    CHECK(segment_clear_of_polygons({-1, 3}, {3, 3}, obs));
    CHECK(segment_clear_of_polygons({-1, -1}, {3, -1}, obs));
}

TEST_CASE("visibility graph shortest path around a square has the closed-form length") {
    Polygon sq(4, 2);
    sq << 1, -1, 3, -1, 3, 1, 1, 1;
    const Graph g = visibility_graph({sq}, {Eigen::Vector2d(0, 0), Eigen::Vector2d(4, 0)});
    CHECK(g.size() == 6);
    const SearchResult r = graph_search(g, 0, 1);
    REQUIRE(r.found);
    CHECK(r.cost == Approx(2.0 + 2.0 * std::sqrt(2.0)));
    CHECK(r.path.size() == 4);  // start, two corners, goal
    // Dijkstra agrees, and a fine grid path is only slightly longer (octile vs Euclidean)
    CHECK(graph_search(g, 0, 1, SearchAlgorithm::Dijkstra).cost == Approx(r.cost));
    OccupancyGrid grid(100, 60, 0.05, Eigen::Vector2d(-0.5, -1.5));
    for (int y = 0; y < 60; ++y)
        for (int x = 0; x < 100; ++x) {
            const Eigen::Vector2d p = grid.cell_center({x, y});
            grid.set({x, y}, p.x() > 0.95 && p.x() < 3.05 && p.y() > -1.05 && p.y() < 1.05);
        }
    const GridSearchResult gr = grid_search(grid, grid.world_to_cell({0, 0}), grid.world_to_cell({4, 0}));
    CHECK(gr.cost >= r.cost - 0.1);
    CHECK(gr.cost <= 1.1 * r.cost);
}
