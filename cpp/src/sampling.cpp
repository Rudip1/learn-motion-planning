#include "motion_planning/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include "motion_planning/angles.hpp"

namespace motion_planning {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Eigen::Vector2d steer(const Eigen::Vector2d& from, const Eigen::Vector2d& to, double eta) {
    const Eigen::Vector2d d = to - from;
    const double n = d.norm();
    return n <= eta ? to : Eigen::Vector2d(from + eta / n * d);  // eq. (5.1)
}

class Sampler {
  public:
    Sampler(const PlanningProblem2D& p, std::uint32_t seed) : p_(p), rng_(seed), u_(0.0, 1.0) {}
    Eigen::Vector2d uniform() {
        return p_.lower + (p_.upper - p_.lower).cwiseProduct(Eigen::Vector2d(u_(rng_), u_(rng_)));
    }
    double unit() { return u_(rng_); }

  private:
    const PlanningProblem2D& p_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> u_;
};

int nearest(const std::vector<Eigen::Vector2d>& nodes, const Eigen::Vector2d& x) {
    int best = 0;
    double best_d = kInf;
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        const double d = (nodes[i] - x).squaredNorm();
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

void check(const PlanningProblem2D& p) {
    if (!p.state_valid || !p.motion_valid) throw std::invalid_argument("problem needs both validity checks");
    if (!p.state_valid(p.start)) throw std::invalid_argument("start state is invalid");
    if (p.goal_radius == 0.0 && !p.state_valid(p.goal)) throw std::invalid_argument("goal state is invalid");
}

double box_gamma(const PlanningProblem2D& p, double gamma) {
    if (gamma > 0.0) return gamma;
    const Eigen::Vector2d size = p.upper - p.lower;
    return optimal_gamma(size.x() * size.y());
}

// Shared tree builder: RRT when `star` is false, RRT* otherwise.
TreeResult grow_tree(const PlanningProblem2D& p, const RrtOptions& opt, bool star) {
    check(p);
    if (opt.step <= 0.0) throw std::invalid_argument("step must be positive");
    const double gamma = box_gamma(p, opt.gamma);
    Sampler sampler(p, opt.seed);
    TreeResult t;
    t.nodes = {p.start};
    t.parent = {-1};
    t.cost = {0.0};
    std::vector<std::vector<int>> children(1);
    std::vector<int> goal_candidates;  // nodes from which the goal is reached

    auto reaches_goal = [&](int i) {
        if (p.goal_radius > 0.0) return (t.nodes[i] - p.goal).norm() <= p.goal_radius;
        return (t.nodes[i] - p.goal).norm() <= opt.step && p.motion_valid(t.nodes[i], p.goal);
    };
    auto to_goal = [&](int i) { return p.goal_radius > 0.0 ? 0.0 : (t.nodes[i] - p.goal).norm(); };
    auto best_candidate = [&]() {
        int best = -1;
        double best_cost = kInf;
        for (int c : goal_candidates)
            if (t.cost[c] + to_goal(c) < best_cost) {
                best_cost = t.cost[c] + to_goal(c);
                best = c;
            }
        return std::make_pair(best, best_cost);
    };
    // add delta to the cost of every node below `root` (after a rewire changed the cost of root)
    auto propagate = [&](int root, double delta) {
        std::vector<int> stack{root};
        while (!stack.empty()) {
            const int v = stack.back();
            stack.pop_back();
            t.cost[v] += delta;
            for (int c : children[v]) stack.push_back(c);
        }
    };

    for (int it = 0; it < opt.max_iterations; ++it) {
        const Eigen::Vector2d x_rand = sampler.unit() < opt.goal_bias ? p.goal : sampler.uniform();
        const int i_near = nearest(t.nodes, x_rand);
        const Eigen::Vector2d x_new = steer(t.nodes[i_near], x_rand, opt.step);
        if ((x_new - t.nodes[i_near]).norm() > 0.0 && p.state_valid(x_new) &&
            p.motion_valid(t.nodes[i_near], x_new)) {
            int parent = i_near;
            double c_new = t.cost[i_near] + (x_new - t.nodes[i_near]).norm();
            std::vector<int> near;
            if (star) {
                const int n = static_cast<int>(t.nodes.size()) + 1;
                const double r = std::min(rewiring_radius(n, gamma), opt.step);  // eq. (5.3)
                for (int i = 0; i < static_cast<int>(t.nodes.size()); ++i)
                    if ((t.nodes[i] - x_new).norm() <= r) near.push_back(i);
                for (int i : near) {  // choose the parent that gives the cheapest cost-to-come, eq. (5.5)
                    const double c = t.cost[i] + (x_new - t.nodes[i]).norm();
                    if (c < c_new && p.motion_valid(t.nodes[i], x_new)) {
                        c_new = c;
                        parent = i;
                    }
                }
            }
            const int k = static_cast<int>(t.nodes.size());
            t.nodes.push_back(x_new);
            t.parent.push_back(parent);
            t.cost.push_back(c_new);
            children.emplace_back();
            children[parent].push_back(k);
            if (star) {
                for (int i : near) {  // rewire: route neighbours through the new node if cheaper, eq. (5.6)
                    if (i == parent) continue;
                    const double c = c_new + (t.nodes[i] - x_new).norm();
                    if (c < t.cost[i] - 1e-12 && p.motion_valid(x_new, t.nodes[i])) {
                        auto& siblings = children[t.parent[i]];
                        siblings.erase(std::find(siblings.begin(), siblings.end(), i));
                        t.parent[i] = k;
                        children[k].push_back(i);
                        propagate(i, c - t.cost[i]);
                    }
                }
            }
            if (reaches_goal(k)) {
                goal_candidates.push_back(k);
                if (t.first_solution_iteration < 0) t.first_solution_iteration = it;
            }
        }
        t.best_cost_history.push_back(best_candidate().second);
        if (!star && opt.stop_at_first_solution && !goal_candidates.empty()) break;
    }

    const auto [best, best_cost] = best_candidate();
    if (best >= 0) {
        t.found = true;
        t.path_cost = best_cost;
        for (int v = best; v != -1; v = t.parent[v]) t.path.push_back(t.nodes[v]);
        std::reverse(t.path.begin(), t.path.end());
        if (p.goal_radius == 0.0) t.path.push_back(p.goal);
    }
    return t;
}

}  // namespace

PlanningProblem2D problem_from_grid(const OccupancyGrid& grid, const Eigen::Vector2d& start,
                                    const Eigen::Vector2d& goal) {
    PlanningProblem2D p;
    p.lower = grid.origin();
    p.upper = grid.origin() + grid.resolution() * Eigen::Vector2d(grid.width(), grid.height());
    p.state_valid = [grid](const Eigen::Vector2d& x) { return grid.point_free(x); };
    p.motion_valid = [grid](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return segment_free(grid, a, b);
    };
    p.start = start;
    p.goal = goal;
    return p;
}

PlanningProblem2D problem_from_polygons(const std::vector<Polygon>& polygons, const Eigen::Vector2d& lower,
                                        const Eigen::Vector2d& upper, const Eigen::Vector2d& start,
                                        const Eigen::Vector2d& goal) {
    PlanningProblem2D p;
    p.lower = lower;
    p.upper = upper;
    auto inside_box = [lower, upper](const Eigen::Vector2d& x) {
        return (x.array() >= lower.array()).all() && (x.array() <= upper.array()).all();
    };
    p.state_valid = [polygons, inside_box](const Eigen::Vector2d& x) {
        if (!inside_box(x)) return false;
        for (const Polygon& poly : polygons)
            if (point_in_polygon(x, poly)) return false;
        return true;
    };
    p.motion_valid = [polygons, inside_box](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return inside_box(a) && inside_box(b) && segment_clear_of_polygons(a, b, polygons);  // box is convex
    };
    p.start = start;
    p.goal = goal;
    return p;
}

double rewiring_radius(int n, double gamma) {
    if (n < 2) return kInf;
    return gamma * std::sqrt(std::log(static_cast<double>(n)) / n);  // eq. (5.3), d = 2
}

double optimal_gamma(double free_area) {
    return 2.0 * std::sqrt(1.5 * free_area / kPi);  // eq. (5.4), d = 2
}

PrmResult prm(const PlanningProblem2D& p, const PrmOptions& opt) {
    check(p);
    Sampler sampler(p, opt.seed);
    PrmResult r;
    Graph& g = r.roadmap;
    g.add_node(p.start);
    g.add_node(p.goal);
    int attempts = 0;
    while (g.size() < opt.num_samples + 2 && attempts < 100 * (opt.num_samples + 1)) {
        ++attempts;
        const Eigen::Vector2d x = sampler.uniform();
        if (p.state_valid(x)) g.add_node(x);
    }
    const int n = g.size();
    const double radius =
        opt.connection_radius > 0.0 ? opt.connection_radius : rewiring_radius(n, box_gamma(p, opt.gamma));
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            if ((g.point(i) - g.point(j)).norm() > radius) continue;
            ++r.collision_checks;
            if (p.motion_valid(g.point(i), g.point(j))) g.add_edge(i, j);
        }
    r.query = graph_search(g, 0, 1, SearchAlgorithm::AStar);
    return r;
}

TreeResult rrt(const PlanningProblem2D& problem, const RrtOptions& options) {
    return grow_tree(problem, options, false);
}

TreeResult rrt_star(const PlanningProblem2D& problem, const RrtOptions& options) {
    return grow_tree(problem, options, true);
}

double path_length(const std::vector<Eigen::Vector2d>& path) {
    double L = 0.0;
    for (std::size_t i = 1; i < path.size(); ++i) L += (path[i] - path[i - 1]).norm();
    return L;
}

}  // namespace motion_planning
