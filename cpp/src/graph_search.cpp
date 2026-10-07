#include "motion_planning/graph_search.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace motion_planning {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

std::vector<int> reconstruct(const std::vector<int>& parent, int goal) {
    std::vector<int> path;
    for (int v = goal; v != -1; v = parent[v]) path.push_back(v);
    std::reverse(path.begin(), path.end());
    return path;
}
}  // namespace

SearchResult best_first_search(int n, int start, int goal, const NeighbourFunction& neighbours,
                               const HeuristicFunction& heuristic, SearchAlgorithm algorithm, double weight) {
    if (start < 0 || start >= n || goal < 0 || goal >= n)
        throw std::out_of_range("start or goal out of range");
    if (algorithm == SearchAlgorithm::WeightedAStar && weight < 1.0)
        throw std::invalid_argument("weighted A* needs w >= 1");
    std::vector<double> g(n, kInf);
    std::vector<int> parent(n, -1);
    std::vector<char> closed(n, 0);
    std::vector<std::pair<int, double>> nbrs;
    SearchResult result;
    g[start] = 0.0;

    if (algorithm == SearchAlgorithm::BreadthFirst) {  // eq. (4.2): FIFO order, fewest edges
        std::deque<int> queue{start};
        std::vector<char> discovered(n, 0);
        discovered[start] = 1;
        while (!queue.empty()) {
            const int u = queue.front();
            queue.pop_front();
            result.expanded.push_back(u);
            if (u == goal) break;
            nbrs.clear();
            neighbours(u, nbrs);
            for (const auto& [v, c] : nbrs) {
                if (discovered[v]) continue;
                discovered[v] = 1;
                g[v] = g[u] + c;
                parent[v] = u;
                queue.push_back(v);
            }
        }
    } else {
        auto key = [&](int v) {  // eqs. (4.3), (4.4), (4.7)
            switch (algorithm) {
                case SearchAlgorithm::Dijkstra:
                    return g[v];
                case SearchAlgorithm::AStar:
                    return g[v] + heuristic(v);
                case SearchAlgorithm::WeightedAStar:
                    return g[v] + weight * heuristic(v);
                case SearchAlgorithm::GreedyBestFirst:
                    return heuristic(v);
                default:
                    return g[v];
            }
        };
        // (key, h, g at push time, node); ties on the key go to the smaller h, i.e. closer to the goal
        using Entry = std::tuple<double, double, double, int>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
        open.emplace(key(start), heuristic(start), 0.0, start);
        while (!open.empty()) {
            const auto [f, h, g_push, u] = open.top();
            open.pop();
            if (closed[u] || g_push > g[u]) continue;  // stale entry: u was reached more cheaply since
            closed[u] = 1;
            result.expanded.push_back(u);
            if (u == goal) break;
            nbrs.clear();
            neighbours(u, nbrs);
            for (const auto& [v, c] : nbrs) {
                if (closed[v]) continue;
                const double candidate = g[u] + c;  // eq. (4.1): relax the edge
                if (candidate < g[v]) {
                    g[v] = candidate;
                    parent[v] = u;
                    open.emplace(key(v), heuristic(v), candidate, v);
                }
            }
        }
    }
    if (g[goal] < kInf && (algorithm == SearchAlgorithm::BreadthFirst || closed[goal] || goal == start)) {
        result.found = true;
        result.cost = g[goal];
        result.path = reconstruct(parent, goal);
    }
    return result;
}

double grid_heuristic(Heuristic kind, const Cell& a, const Cell& b, double h) {
    const double dx = std::abs(a.x - b.x), dy = std::abs(a.y - b.y);
    switch (kind) {  // section 4.4
        case Heuristic::Zero:
            return 0.0;
        case Heuristic::Euclidean:
            return h * std::hypot(dx, dy);
        case Heuristic::Manhattan:
            return h * (dx + dy);
        case Heuristic::Octile:
            return h * (std::max(dx, dy) + (std::sqrt(2.0) - 1.0) * std::min(dx, dy));
        case Heuristic::Chebyshev:
            return h * std::max(dx, dy);
    }
    return 0.0;
}

GridSearchResult grid_search(const OccupancyGrid& grid, const Cell& start, const Cell& goal,
                             const GridSearchOptions& opt, const FieldArray& cell_cost) {
    if (grid.occupied(start) || grid.occupied(goal)) throw std::invalid_argument("start or goal is occupied");
    const bool has_cost = cell_cost.size() > 0;
    if (has_cost && (cell_cost.rows() != grid.height() || cell_cost.cols() != grid.width()))
        throw std::invalid_argument("cell_cost must have the grid's shape");
    const int W = grid.width();
    auto id = [W](const Cell& c) { return c.y * W + c.x; };
    auto cell = [W](int i) { return Cell{i % W, i / W}; };
    const double h = grid.resolution();
    const auto& offsets = neighbour_offsets(opt.connectivity);

    NeighbourFunction neighbours = [&](int u, std::vector<std::pair<int, double>>& out) {
        const Cell c = cell(u);
        for (const Cell& d : offsets) {
            const Cell n{c.x + d.x, c.y + d.y};
            if (grid.occupied(n)) continue;
            const bool diagonal = d.x != 0 && d.y != 0;
            if (diagonal && !opt.allow_corner_cutting &&
                (grid.occupied({c.x + d.x, c.y}) || grid.occupied({c.x, c.y + d.y})))
                continue;
            double cost = diagonal ? std::sqrt(2.0) * h : h;
            if (has_cost) cost *= 1.0 + cell_cost(n.y, n.x);
            out.emplace_back(id(n), cost);
        }
    };
    HeuristicFunction heuristic = [&](int u) { return grid_heuristic(opt.heuristic, cell(u), goal, h); };

    const SearchResult r = best_first_search(W * grid.height(), id(start), id(goal), neighbours, heuristic,
                                             opt.algorithm, opt.weight);
    GridSearchResult out;
    out.found = r.found;
    out.cost = r.cost;
    for (int i : r.path) out.path.push_back(cell(i));
    for (int i : r.expanded) out.expanded.push_back(cell(i));
    return out;
}

int Graph::add_node(const Eigen::Vector2d& p) {
    points_.push_back(p);
    adjacency_.emplace_back();
    return size() - 1;
}

void Graph::add_edge(int a, int b, double cost, bool bidirectional) {
    if (a < 0 || b < 0 || a >= size() || b >= size()) throw std::out_of_range("edge endpoint out of range");
    if (cost < 0.0) cost = (points_[a] - points_[b]).norm();
    adjacency_[a].emplace_back(b, cost);
    if (bidirectional) adjacency_[b].emplace_back(a, cost);
}

int Graph::num_edges() const {
    int n = 0;
    for (const auto& a : adjacency_) n += static_cast<int>(a.size());
    return n;
}

SearchResult graph_search(const Graph& graph, int start, int goal, SearchAlgorithm algorithm, double weight) {
    const Eigen::Vector2d target = graph.point(goal);
    return best_first_search(
        graph.size(), start, goal,
        [&](int u, std::vector<std::pair<int, double>>& out) {
            for (const auto& e : graph.edges(u)) out.push_back(e);
        },
        [&](int u) { return (graph.point(u) - target).norm(); }, algorithm, weight);
}

namespace {

double cross(const Eigen::Vector2d& o, const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return (a - o).x() * (b - o).y() - (a - o).y() * (b - o).x();
}

double point_segment_distance(const Eigen::Vector2d& p, const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    const Eigen::Vector2d ab = b - a;
    const double len2 = ab.squaredNorm();
    const double t = len2 > 0.0 ? std::clamp((p - a).dot(ab) / len2, 0.0, 1.0) : 0.0;
    return (a + t * ab - p).norm();
}

constexpr double kEps = 1e-9;

}  // namespace

bool point_in_polygon(const Eigen::Vector2d& p, const Polygon& poly) {
    const int n = static_cast<int>(poly.rows());
    for (int i = 0; i < n; ++i)
        if (point_segment_distance(p, poly.row(i), poly.row((i + 1) % n)) < kEps)
            return false;  // on the boundary
    bool inside = false;   // crossing number of a ray towards +x
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const double yi = poly(i, 1), yj = poly(j, 1);
        if ((yi > p.y()) != (yj > p.y())) {
            const double x = poly(j, 0) + (p.y() - yj) / (yi - yj) * (poly(i, 0) - poly(j, 0));
            if (p.x() < x) inside = !inside;
        }
    }
    return inside;
}

bool segment_clear_of_polygons(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                               const std::vector<Polygon>& polygons) {
    std::vector<double> ts{0.0, 1.0};
    const double len = (b - a).norm();
    for (const Polygon& poly : polygons) {
        const int n = static_cast<int>(poly.rows());
        for (int i = 0; i < n; ++i) {
            const Eigen::Vector2d p = poly.row(i), q = poly.row((i + 1) % n);
            // a proper crossing (each segment strictly separates the other's endpoints) enters the polygon
            const double o1 = cross(a, b, p), o2 = cross(a, b, q), o3 = cross(p, q, a), o4 = cross(p, q, b);
            const double tol = kEps * std::max(1.0, len * (q - p).norm());
            if (((o1 > tol && o2 < -tol) || (o1 < -tol && o2 > tol)) &&
                ((o3 > tol && o4 < -tol) || (o3 < -tol && o4 > tol)))
                return false;
            // vertices lying on the segment split it into pieces that touch no boundary in between
            if (len > 0.0 && point_segment_distance(p, a, b) < kEps)
                ts.push_back((p - a).dot(b - a) / (len * len));
        }
    }
    std::sort(ts.begin(), ts.end());
    for (std::size_t k = 0; k + 1 < ts.size(); ++k) {
        if (ts[k + 1] - ts[k] < 1e-12) continue;
        const Eigen::Vector2d mid = a + 0.5 * (ts[k] + ts[k + 1]) * (b - a);
        for (const Polygon& poly : polygons)
            if (point_in_polygon(mid, poly)) return false;
    }
    return true;
}

Graph visibility_graph(const std::vector<Polygon>& polygons, const std::vector<Eigen::Vector2d>& points) {
    Graph g;
    for (const auto& p : points) g.add_node(p);
    for (const Polygon& poly : polygons)
        for (int i = 0; i < poly.rows(); ++i) g.add_node(poly.row(i).transpose());
    for (int i = 0; i < g.size(); ++i)
        for (int j = i + 1; j < g.size(); ++j)
            if (segment_clear_of_polygons(g.point(i), g.point(j), polygons)) g.add_edge(i, j);
    return g;
}

}  // namespace motion_planning
