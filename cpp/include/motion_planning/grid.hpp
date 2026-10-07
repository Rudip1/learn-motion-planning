#pragma once

/// Chapter 2 — occupancy grids, distance transforms, inflation and collision checking.
/// Theory: 1_theory/02_configuration_space.md. Equation numbers below refer to that file.

#include <Eigen/Core>
#include <cstdint>
#include <vector>

#include "motion_planning/kinematics.hpp"

namespace motion_planning {

/// Integer cell index: x = column, y = row. Cell (0, 0) is the lower-left cell of the map.
struct Cell {
    int x = 0;
    int y = 0;
    bool operator==(const Cell& o) const { return x == o.x && y == o.y; }
    bool operator!=(const Cell& o) const { return !(*this == o); }
};

/// Row-major matrix types used to exchange grids with NumPy (row = y index, column = x index).
using GridArray = Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FieldArray = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// Binary occupancy grid. Cell (x, y) covers the square
/// [origin_x + x h, origin_x + (x+1) h) x [origin_y + y h, origin_y + (y+1) h), h = resolution. Eq. (2.5).
/// Everything outside the map counts as occupied.
class OccupancyGrid {
  public:
    OccupancyGrid() = default;
    OccupancyGrid(int width, int height, double resolution = 1.0,
                  const Eigen::Vector2d& origin = Eigen::Vector2d::Zero());
    /// From an array with one row per y index; non-zero entries are occupied.
    explicit OccupancyGrid(const GridArray& occupancy, double resolution = 1.0,
                           const Eigen::Vector2d& origin = Eigen::Vector2d::Zero());

    int width() const { return width_; }
    int height() const { return height_; }
    double resolution() const { return resolution_; }
    const Eigen::Vector2d& origin() const { return origin_; }

    bool in_bounds(const Cell& c) const { return c.x >= 0 && c.y >= 0 && c.x < width_ && c.y < height_; }
    /// Occupied, or outside the map.
    bool occupied(const Cell& c) const { return !in_bounds(c) || data_[index(c)] != 0; }
    void set(const Cell& c, bool occupied);

    /// Cell containing world point p (may be out of bounds). Eq. (2.5).
    Cell world_to_cell(const Eigen::Vector2d& p) const;
    /// World coordinates of the centre of cell c.
    Eigen::Vector2d cell_center(const Cell& c) const;
    /// True if p lies in a free cell of the map.
    bool point_free(const Eigen::Vector2d& p) const { return !occupied(world_to_cell(p)); }

    GridArray to_array() const;
    std::size_t index(const Cell& c) const { return static_cast<std::size_t>(c.y) * width_ + c.x; }

  private:
    int width_ = 0;
    int height_ = 0;
    double resolution_ = 1.0;
    Eigen::Vector2d origin_ = Eigen::Vector2d::Zero();
    std::vector<std::uint8_t> data_;
};

/// Neighbourhood used by grid propagation and grid search.
enum class Connectivity { Four, Eight };

/// The 4 or 8 neighbour offsets, axis-aligned first.
const std::vector<Cell>& neighbour_offsets(Connectivity connectivity);

/// Brushfire: breadth-first wave from all occupied cells. Returns, for every cell, the number of steps to the
/// nearest occupied cell — the L1 distance for 4-connectivity, the L-infinity distance for 8-connectivity
/// (eq. 2.6). In cells; +inf if the map has no occupied cell.
FieldArray brushfire(const OccupancyGrid& grid, Connectivity connectivity = Connectivity::Four);

/// Exact Euclidean distance transform: distance in metres from every cell centre to the nearest occupied
/// cell centre (0 on occupied cells). Felzenszwalb–Huttenlocher lower-envelope algorithm, eqs. (2.7)–(2.9).
FieldArray distance_transform(const OccupancyGrid& grid);

/// One-dimensional squared distance transform d(p) = min_q ((p - q)^2 + f(q)), eq. (2.8). Exposed for tests
/// and the notebook; f may contain +inf.
Eigen::VectorXd squared_distance_transform_1d(const Eigen::VectorXd& f);

/// Configuration-space obstacles of a disc robot of the given radius [m]: every cell whose centre is within
/// `radius` of an occupied cell centre becomes occupied. Eq. (2.10).
OccupancyGrid inflate(const OccupancyGrid& grid, double radius);

/// Conservative disc test from a precomputed distance transform: true only if the disc of `radius` around p
/// cannot touch an occupied cell. Eq. (2.11).
bool disc_free(const OccupancyGrid& grid, const FieldArray& distance, const Eigen::Vector2d& p,
               double radius);

/// All cells whose interior the segment p0 -> p1 passes through, in order (Amanatides–Woo traversal).
/// Section 2.5.
std::vector<Cell> traverse_segment(const OccupancyGrid& grid, const Eigen::Vector2d& p0,
                                   const Eigen::Vector2d& p1);

/// Bresenham's line between two cells: one cell per step of the major axis. Misses cells the true segment
/// clips; kept to show why it is not a collision checker.
std::vector<Cell> bresenham(const Cell& a, const Cell& b);

/// True if every cell the segment passes through is free.
bool segment_free(const OccupancyGrid& grid, const Eigen::Vector2d& p0, const Eigen::Vector2d& p1);

/// Rectangular footprint in the vehicle frame: from `rear` behind the reference point to `front` ahead of it,
/// and `half_width` to each side.
struct RectangleFootprint {
    double rear = 0.0;
    double front = 1.0;
    double half_width = 0.25;
    /// The four corners in world coordinates for a vehicle at pose q, counter-clockwise from rear-right.
    Eigen::Matrix<double, 4, 2> corners(const Pose& q) const;
};

/// Exact test of an oriented rectangle against the occupied cells of the grid (separating-axis theorem,
/// eq. 2.12). Cells outside the map count as occupied.
bool footprint_free(const OccupancyGrid& grid, const Pose& q, const RectangleFootprint& footprint);

/// One slice of the configuration space at heading theta: cell (x, y) is occupied iff the footprint with its
/// reference point at that cell's centre collides. Eq. (2.2).
OccupancyGrid configuration_space_slice(const OccupancyGrid& grid, const RectangleFootprint& footprint,
                                        double theta);

}  // namespace motion_planning
