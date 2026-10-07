#include "motion_planning/potential.hpp"

#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>

namespace motion_planning {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
}

double attractive_potential(const Eigen::Vector2d& p, const Eigen::Vector2d& goal, const PotentialParams& k) {
    const double d = (p - goal).norm();
    if (d <= k.d_star) return 0.5 * k.zeta * d * d;                     // eq. (3.1), quadratic part
    return k.d_star * k.zeta * d - 0.5 * k.zeta * k.d_star * k.d_star;  // conic part
}

Eigen::Vector2d attractive_gradient(const Eigen::Vector2d& p, const Eigen::Vector2d& goal,
                                    const PotentialParams& k) {
    const Eigen::Vector2d e = p - goal;
    const double d = e.norm();
    if (d <= k.d_star) return k.zeta * e;  // eq. (3.2)
    return k.d_star * k.zeta * e / d;
}

double repulsive_potential(double clearance, const PotentialParams& k) {
    if (clearance > k.q_star) return 0.0;
    if (clearance <= 0.0) return kInf;
    const double s = 1.0 / clearance - 1.0 / k.q_star;
    return 0.5 * k.eta * s * s;  // eq. (3.3)
}

FieldArray attractive_field(const OccupancyGrid& grid, const Eigen::Vector2d& goal,
                            const PotentialParams& params) {
    FieldArray f(grid.height(), grid.width());
    for (int y = 0; y < grid.height(); ++y)
        for (int x = 0; x < grid.width(); ++x)
            f(y, x) = attractive_potential(grid.cell_center({x, y}), goal, params);
    return f;
}

FieldArray repulsive_field(const OccupancyGrid& grid, const PotentialParams& params) {
    const FieldArray d = distance_transform(grid);
    return d.unaryExpr([&](double c) { return repulsive_potential(c, params); });
}

FieldArray total_field(const OccupancyGrid& grid, const Eigen::Vector2d& goal,
                       const PotentialParams& params) {
    return attractive_field(grid, goal, params) + repulsive_field(grid, params);  // eq. (3.4)
}

FieldArray wavefront(const OccupancyGrid& grid, const Cell& goal, Connectivity connectivity) {
    FieldArray w = FieldArray::Constant(grid.height(), grid.width(), kInf);
    if (grid.occupied(goal)) throw std::invalid_argument("goal cell is occupied");
    w(goal.y, goal.x) = 0.0;
    std::deque<Cell> queue{goal};
    while (!queue.empty()) {  // eq. (3.6): breadth-first wave from the goal through free cells
        const Cell c = queue.front();
        queue.pop_front();
        for (const Cell& d : neighbour_offsets(connectivity)) {
            const Cell n{c.x + d.x, c.y + d.y};
            if (!grid.occupied(n) && w(n.y, n.x) == kInf) {
                w(n.y, n.x) = w(c.y, c.x) + 1.0;
                queue.push_back(n);
            }
        }
    }
    return w;
}

DescentResult descend(const FieldArray& field, const Cell& start, const Cell& goal, Connectivity connectivity,
                      int max_steps) {
    const int H = static_cast<int>(field.rows()), W = static_cast<int>(field.cols());
    auto inside = [&](const Cell& c) { return c.x >= 0 && c.y >= 0 && c.x < W && c.y < H; };
    if (!inside(start)) throw std::invalid_argument("start outside the field");
    DescentResult r;
    Cell c = start;
    r.path.push_back(c);
    for (int k = 0; k < max_steps; ++k) {
        if (c == goal) {
            r.reached_goal = true;
            return r;
        }
        Cell best = c;
        for (const Cell& d : neighbour_offsets(connectivity)) {
            const Cell n{c.x + d.x, c.y + d.y};
            if (inside(n) && field(n.y, n.x) < field(best.y, best.x)) best = n;
        }
        if (best == c) {  // no neighbour is strictly lower
            r.local_minimum = true;
            return r;
        }
        c = best;
        r.path.push_back(c);
    }
    r.reached_goal = c == goal;
    return r;
}

}  // namespace motion_planning
