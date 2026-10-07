#pragma once

/// Chapter 4 — graph search: breadth-first, Dijkstra, A*, weighted A* and greedy best-first, on implicit grid
/// graphs and on explicit graphs (visibility graphs here, roadmaps in chapter 5).
/// Theory: 1_theory/04_graph_search.md. Equation numbers below refer to that file.

#include <Eigen/Core>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "motion_planning/grid.hpp"

namespace motion_planning {

/// Which best-first rule orders the open list. Eqs. (4.2)–(4.4) and (4.7).
enum class SearchAlgorithm {
    BreadthFirst,    ///< first in, first out; fewest edges
    Dijkstra,        ///< f = g
    AStar,           ///< f = g + h
    WeightedAStar,   ///< f = g + w h, w >= 1
    GreedyBestFirst  ///< f = h
};

/// Result of a search between node ids.
struct SearchResult {
    bool found = false;
    std::vector<int> path;  ///< start ... goal; empty if not found
    double cost = std::numeric_limits<double>::infinity();
    std::vector<int> expanded;  ///< nodes in the order they were expanded (closed)
};

/// Appends (neighbour, edge cost) pairs of a node to the output vector. Edge costs must be non-negative.
using NeighbourFunction = std::function<void(int, std::vector<std::pair<int, double>>&)>;
/// Heuristic estimate of the cost from a node to the goal.
using HeuristicFunction = std::function<double(int)>;

/// Generic best-first search on nodes 0 .. num_nodes-1. Each node is expanded at most once, so A* is optimal
/// for consistent heuristics (eq. 4.6); weighted A* is within a factor w of optimal (eq. 4.8). Algorithm in
/// section 4.3.
SearchResult best_first_search(int num_nodes, int start, int goal, const NeighbourFunction& neighbours,
                               const HeuristicFunction& heuristic, SearchAlgorithm algorithm,
                               double weight = 1.0);

/// Grid heuristics. Eq. (4.5) and table in section 4.4.
enum class Heuristic { Zero, Euclidean, Manhattan, Octile, Chebyshev };

/// Heuristic value between two cells, in metres for resolution h.
double grid_heuristic(Heuristic kind, const Cell& a, const Cell& b, double resolution = 1.0);

/// Options for grid search.
struct GridSearchOptions {
    SearchAlgorithm algorithm = SearchAlgorithm::AStar;
    Connectivity connectivity = Connectivity::Eight;
    Heuristic heuristic = Heuristic::Octile;
    double weight = 1.0;
    bool allow_corner_cutting = false;  ///< allow a diagonal step past an occupied orthogonal neighbour
};

/// Result of a grid search.
struct GridSearchResult {
    bool found = false;
    std::vector<Cell> path;
    double cost = std::numeric_limits<double>::infinity();  ///< metres (times the cell-cost factor, if any)
    std::vector<Cell> expanded;
};

/// Search an occupancy grid. Steps cost their length (h or sqrt(2) h). If `cell_cost` is non-empty (same
/// shape as the grid, values >= 0), a step into cell c costs length * (1 + cell_cost(c)) — e.g. a clearance
/// penalty. Eq. (4.9).
GridSearchResult grid_search(const OccupancyGrid& grid, const Cell& start, const Cell& goal,
                             const GridSearchOptions& options = {},
                             const FieldArray& cell_cost = FieldArray());

/// An explicit graph with points in the plane, for roadmaps and visibility graphs.
class Graph {
  public:
    int add_node(const Eigen::Vector2d& p);
    /// Add an edge; a negative cost means the Euclidean distance between the endpoints.
    void add_edge(int a, int b, double cost = -1.0, bool bidirectional = true);
    int size() const { return static_cast<int>(points_.size()); }
    const Eigen::Vector2d& point(int i) const { return points_.at(i); }
    const std::vector<std::pair<int, double>>& edges(int i) const { return adjacency_.at(i); }
    int num_edges() const;

  private:
    std::vector<Eigen::Vector2d> points_;
    std::vector<std::vector<std::pair<int, double>>> adjacency_;
};

/// Search an explicit graph with the Euclidean distance to the goal as heuristic.
SearchResult graph_search(const Graph& graph, int start, int goal,
                          SearchAlgorithm algorithm = SearchAlgorithm::AStar, double weight = 1.0);

/// A polygon as an (N, 2) array of vertices in order (either orientation).
using Polygon = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

/// True if p lies strictly inside the polygon (points on the boundary are outside).
bool point_in_polygon(const Eigen::Vector2d& p, const Polygon& polygon);

/// True if the segment does not pass through the interior of any polygon (touching edges and vertices is
/// allowed).
bool segment_clear_of_polygons(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                               const std::vector<Polygon>& polygons);

/// Visibility graph: nodes are the given points followed by every polygon vertex; an edge joins two nodes iff
/// the segment between them does not enter any polygon. Edge cost = length. Section 4.6.
Graph visibility_graph(const std::vector<Polygon>& polygons, const std::vector<Eigen::Vector2d>& points);

}  // namespace motion_planning
