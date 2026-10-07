#include "motion_planning/grid.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>

namespace motion_planning {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
}

OccupancyGrid::OccupancyGrid(int width, int height, double resolution, const Eigen::Vector2d& origin)
    : width_(width), height_(height), resolution_(resolution), origin_(origin) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("grid dimensions must be positive");
    if (resolution <= 0.0) throw std::invalid_argument("resolution must be positive");
    data_.assign(static_cast<std::size_t>(width) * height, 0);
}

OccupancyGrid::OccupancyGrid(const GridArray& occupancy, double resolution, const Eigen::Vector2d& origin)
    : OccupancyGrid(static_cast<int>(occupancy.cols()), static_cast<int>(occupancy.rows()), resolution,
                    origin) {
    for (int y = 0; y < height_; ++y)
        for (int x = 0; x < width_; ++x) data_[index({x, y})] = occupancy(y, x) != 0;
}

void OccupancyGrid::set(const Cell& c, bool occupied) {
    if (!in_bounds(c)) throw std::out_of_range("cell outside the grid");
    data_[index(c)] = occupied;
}

Cell OccupancyGrid::world_to_cell(const Eigen::Vector2d& p) const {
    const Eigen::Vector2d u = (p - origin_) / resolution_;  // eq. (2.5)
    return {static_cast<int>(std::floor(u.x())), static_cast<int>(std::floor(u.y()))};
}

Eigen::Vector2d OccupancyGrid::cell_center(const Cell& c) const {
    return origin_ + resolution_ * Eigen::Vector2d(c.x + 0.5, c.y + 0.5);
}

GridArray OccupancyGrid::to_array() const {
    GridArray a(height_, width_);
    for (int y = 0; y < height_; ++y)
        for (int x = 0; x < width_; ++x) a(y, x) = data_[index({x, y})];
    return a;
}

const std::vector<Cell>& neighbour_offsets(Connectivity connectivity) {
    static const std::vector<Cell> four{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    static const std::vector<Cell> eight{{1, 0}, {0, 1},  {-1, 0},  {0, -1},
                                         {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
    return connectivity == Connectivity::Four ? four : eight;
}

FieldArray brushfire(const OccupancyGrid& grid, Connectivity connectivity) {
    FieldArray dist = FieldArray::Constant(grid.height(), grid.width(), kInf);
    std::deque<Cell> queue;
    for (int y = 0; y < grid.height(); ++y)
        for (int x = 0; x < grid.width(); ++x)
            if (grid.occupied({x, y})) {
                dist(y, x) = 0.0;
                queue.push_back({x, y});
            }
    // Breadth-first: every cell is labelled the first time the wave reaches it, which is the fewest steps.
    while (!queue.empty()) {
        const Cell c = queue.front();
        queue.pop_front();
        for (const Cell& d : neighbour_offsets(connectivity)) {
            const Cell n{c.x + d.x, c.y + d.y};
            if (grid.in_bounds(n) && dist(n.y, n.x) == kInf) {
                dist(n.y, n.x) = dist(c.y, c.x) + 1.0;
                queue.push_back(n);
            }
        }
    }
    return dist;
}

Eigen::VectorXd squared_distance_transform_1d(const Eigen::VectorXd& f) {
    const int n = static_cast<int>(f.size());
    Eigen::VectorXd d = Eigen::VectorXd::Constant(n, kInf);
    std::vector<int> v(n);         // abscissae of the parabolas in the lower envelope
    std::vector<double> z(n + 1);  // boundaries between consecutive envelope parabolas
    int k = -1;
    for (int q = 0; q < n; ++q) {
        if (!std::isfinite(f[q])) continue;  // a parabola at +inf never reaches the envelope
        if (k < 0) {
            k = 0;
            v[0] = q;
            z[0] = -kInf;
            z[1] = kInf;
            continue;
        }
        // eq. (2.9): where the parabola rooted at q overtakes the envelope parabola rooted at v[k]; pop
        // parabolas that the new one hides completely (z[0] = -inf stops the loop)
        auto intersect = [&](int r) {
            return ((f[q] + double(q) * q) - (f[r] + double(r) * r)) / (2.0 * q - 2.0 * r);
        };
        double s = intersect(v[k]);
        while (s <= z[k]) s = intersect(v[--k]);
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = kInf;
    }
    if (k < 0) return d;
    k = 0;
    for (int p = 0; p < n; ++p) {  // eq. (2.8): read the envelope
        while (z[k + 1] < p) ++k;
        d[p] = (p - v[k]) * double(p - v[k]) + f[v[k]];
    }
    return d;
}

FieldArray distance_transform(const OccupancyGrid& grid) {
    const int W = grid.width(), H = grid.height();
    FieldArray sq(H, W);
    // eq. (2.7): the 2-D squared transform separates into a pass along columns, then along rows
    Eigen::VectorXd col(H);
    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) col[y] = grid.occupied({x, y}) ? 0.0 : kInf;
        sq.col(x) = squared_distance_transform_1d(col);
    }
    Eigen::VectorXd row(W);
    for (int y = 0; y < H; ++y) {
        row = sq.row(y).transpose();
        sq.row(y) = squared_distance_transform_1d(row).transpose();
    }
    return sq.cwiseSqrt() * grid.resolution();
}

OccupancyGrid inflate(const OccupancyGrid& grid, double radius) {
    const FieldArray d = distance_transform(grid);
    OccupancyGrid out = grid;
    for (int y = 0; y < grid.height(); ++y)
        for (int x = 0; x < grid.width(); ++x)
            if (d(y, x) <= radius) out.set({x, y}, true);  // eq. (2.10)
    return out;
}

bool disc_free(const OccupancyGrid& grid, const FieldArray& distance, const Eigen::Vector2d& p,
               double radius) {
    const Cell c = grid.world_to_cell(p);
    if (grid.occupied(c)) return false;
    // the map border counts as an obstacle
    const Eigen::Vector2d lo = grid.origin();
    const Eigen::Vector2d hi = lo + grid.resolution() * Eigen::Vector2d(grid.width(), grid.height());
    const double border = std::min({p.x() - lo.x(), p.y() - lo.y(), hi.x() - p.x(), hi.y() - p.y()});
    if (border <= radius) return false;
    // eq. (2.11): the point may sit h/sqrt(2) from its cell centre, and the obstacle extends h/sqrt(2) from
    // its own centre
    return distance(c.y, c.x) - std::sqrt(2.0) * grid.resolution() > radius;
}

std::vector<Cell> traverse_segment(const OccupancyGrid& grid, const Eigen::Vector2d& p0,
                                   const Eigen::Vector2d& p1) {
    const double h = grid.resolution();
    const Eigen::Vector2d u0 = (p0 - grid.origin()) / h, u1 = (p1 - grid.origin()) / h;
    Cell c = grid.world_to_cell(p0);
    const Cell end = grid.world_to_cell(p1);
    const Eigen::Vector2d d = u1 - u0;
    const int step_x = d.x() > 0 ? 1 : -1, step_y = d.y() > 0 ? 1 : -1;
    // parameter t in [0, 1] at which the segment crosses the next vertical / horizontal grid line
    double t_max_x = d.x() != 0.0 ? ((c.x + (step_x > 0 ? 1 : 0)) - u0.x()) / d.x() : kInf;
    double t_max_y = d.y() != 0.0 ? ((c.y + (step_y > 0 ? 1 : 0)) - u0.y()) / d.y() : kInf;
    const double t_delta_x = d.x() != 0.0 ? 1.0 / std::abs(d.x()) : kInf;
    const double t_delta_y = d.y() != 0.0 ? 1.0 / std::abs(d.y()) : kInf;
    std::vector<Cell> cells{c};
    const int n = std::abs(end.x - c.x) + std::abs(end.y - c.y);
    for (int i = 0; i < n; ++i) {
        if (t_max_x < t_max_y) {
            c.x += step_x;
            t_max_x += t_delta_x;
        } else {
            c.y += step_y;
            t_max_y += t_delta_y;
        }
        cells.push_back(c);
    }
    return cells;
}

std::vector<Cell> bresenham(const Cell& a, const Cell& b) {
    std::vector<Cell> cells;
    int x = a.x, y = a.y;
    const int dx = std::abs(b.x - a.x), dy = -std::abs(b.y - a.y);
    const int sx = a.x < b.x ? 1 : -1, sy = a.y < b.y ? 1 : -1;
    int err = dx + dy;
    while (true) {
        cells.push_back({x, y});
        if (x == b.x && y == b.y) break;
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y += sy;
        }
    }
    return cells;
}

bool segment_free(const OccupancyGrid& grid, const Eigen::Vector2d& p0, const Eigen::Vector2d& p1) {
    for (const Cell& c : traverse_segment(grid, p0, p1))
        if (grid.occupied(c)) return false;
    return true;
}

Eigen::Matrix<double, 4, 2> RectangleFootprint::corners(const Pose& q) const {
    const Eigen::Vector2d f(std::cos(q[2]), std::sin(q[2])), l(-std::sin(q[2]), std::cos(q[2]));
    const Eigen::Vector2d p = q.head<2>();
    Eigen::Matrix<double, 4, 2> c;
    c.row(0) = (p - rear * f - half_width * l).transpose();
    c.row(1) = (p + front * f - half_width * l).transpose();
    c.row(2) = (p + front * f + half_width * l).transpose();
    c.row(3) = (p - rear * f + half_width * l).transpose();
    return c;
}

namespace {

// eq. (2.12): two convex polygons are disjoint iff their projections are disjoint on some edge normal.
// For a rectangle and an axis-aligned square the candidate axes are x, y and the two rectangle axes.
bool rectangle_overlaps_box(const Eigen::Matrix<double, 4, 2>& rect, const Eigen::Vector2d& f,
                            const Eigen::Vector2d& lo, const Eigen::Vector2d& hi) {
    const Eigen::Vector2d rmin = rect.colwise().minCoeff().transpose();
    const Eigen::Vector2d rmax = rect.colwise().maxCoeff().transpose();
    if (rmax.x() < lo.x() || hi.x() < rmin.x() || rmax.y() < lo.y() || hi.y() < rmin.y()) return false;
    const Eigen::Vector2d l(-f.y(), f.x());
    const Eigen::Matrix<double, 4, 2> box =
        (Eigen::Matrix<double, 4, 2>() << lo.x(), lo.y(), hi.x(), lo.y(), hi.x(), hi.y(), lo.x(), hi.y())
            .finished();
    for (const Eigen::Vector2d& axis : {f, l}) {
        const Eigen::Vector4d pr = rect * axis, pb = box * axis;
        if (pr.maxCoeff() < pb.minCoeff() || pb.maxCoeff() < pr.minCoeff()) return false;
    }
    return true;
}

}  // namespace

bool footprint_free(const OccupancyGrid& grid, const Pose& q, const RectangleFootprint& footprint) {
    const Eigen::Matrix<double, 4, 2> rect = footprint.corners(q);
    const Eigen::Vector2d f(std::cos(q[2]), std::sin(q[2]));
    const Cell c0 = grid.world_to_cell(rect.colwise().minCoeff().transpose());
    const Cell c1 = grid.world_to_cell(rect.colwise().maxCoeff().transpose());
    const double h = grid.resolution();
    for (int y = c0.y; y <= c1.y; ++y)
        for (int x = c0.x; x <= c1.x; ++x) {
            if (!grid.occupied({x, y})) continue;
            const Eigen::Vector2d lo = grid.origin() + h * Eigen::Vector2d(x, y);
            if (rectangle_overlaps_box(rect, f, lo, lo + Eigen::Vector2d(h, h))) return false;
        }
    return true;
}

OccupancyGrid configuration_space_slice(const OccupancyGrid& grid, const RectangleFootprint& footprint,
                                        double theta) {
    OccupancyGrid out(grid.width(), grid.height(), grid.resolution(), grid.origin());
    for (int y = 0; y < grid.height(); ++y)
        for (int x = 0; x < grid.width(); ++x) {
            const Eigen::Vector2d p = grid.cell_center({x, y});
            out.set({x, y}, !footprint_free(grid, Pose(p.x(), p.y(), theta), footprint));
        }
    return out;
}

}  // namespace motion_planning
