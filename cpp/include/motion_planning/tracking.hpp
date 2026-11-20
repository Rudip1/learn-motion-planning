#pragma once

/// Chapter 7 — path smoothing and path tracking: shortcutting, pure pursuit and Stanley.
/// Theory: 1_theory/07_smoothing_tracking.md. Equation numbers below refer to that file.

#include <Eigen/Core>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include "motion_planning/kinematics.hpp"

namespace motion_planning {

using Points2D = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using MotionCheck = std::function<bool(const Eigen::Vector2d&, const Eigen::Vector2d&)>;

/// Greedy shortcutting ("string pulling"): from each kept vertex jump to the farthest later vertex that is
/// directly reachable. Section 7.1.
Points2D shortcut_greedy(const Points2D& path, const MotionCheck& motion_valid);

/// Random shortcutting: repeatedly pick two points on the path (anywhere along it, not only vertices) and
/// replace the stretch between them by a straight segment if it is collision-free. Section 7.1.
Points2D shortcut_random(const Points2D& path, const MotionCheck& motion_valid, int iterations = 200,
                         std::uint32_t seed = 1);

/// Turning angle at every interior vertex of a polyline [rad]; a measure of how far a path is from drivable.
Eigen::VectorXd turning_angles(const Points2D& path);

/// Result of projecting a point onto a path.
struct PathProjection {
    double s = 0.0;  ///< arc length of the closest point
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    double heading = 0.0;  ///< path heading at that point
    double lateral = 0.0;  ///< signed distance, positive to the left of the path. Eq. (7.1)
};

/// A polyline parametrised by arc length.
class Path2D {
  public:
    explicit Path2D(const Points2D& points);
    double length() const { return s_.back(); }
    const Points2D& points() const { return points_; }
    /// Point at arc length s (clamped to [0, length]).
    Eigen::Vector2d point_at(double s) const;
    /// Heading of the segment containing arc length s.
    double heading_at(double s) const;
    /// Closest point of the path to p. Eq. (7.1).
    PathProjection project(const Eigen::Vector2d& p) const;

  private:
    Points2D points_;
    std::vector<double> s_;  ///< cumulative arc length at each vertex
    int segment_at(double s) const;
};

/// Pure pursuit: curvature of the arc from the rear axle (pose q) through the target point. Eq. (7.3).
double pure_pursuit_curvature(const Pose& q, const Eigen::Vector2d& target);

/// Stanley steering angle from the heading error and the signed lateral error of the front axle. Eq. (7.5).
double stanley_steering(double heading_error, double lateral_error, double speed, double gain,
                        double softening);

enum class TrackingController { PurePursuit, Stanley };

/// Path-tracking simulation options.
struct TrackingOptions {
    TrackingController controller = TrackingController::PurePursuit;
    double speed = 1.0;           ///< constant forward speed of the rear axle [m/s]
    double lookahead = 1.0;       ///< pure pursuit: base lookahead distance L_0 [m]
    double lookahead_gain = 0.0;  ///< pure pursuit: L_d = L_0 + lookahead_gain * v. Eq. (7.4)
    double stanley_gain = 1.0;    ///< Stanley: k [1/s]
    double softening = 0.1;       ///< Stanley: k_s [m/s]
    double steering_rate = std::numeric_limits<double>::infinity();  ///< max |d gamma / dt| [rad/s]
    double dt = 0.02;
    int max_steps = 20000;
};

/// Outcome of a tracking run on a kinematic bicycle.
struct TrackingResult {
    Trajectory states;                ///< rear-axle poses, one per step
    std::vector<double> cross_track;  ///< signed lateral error of the rear axle, eq. (7.1)
    std::vector<double> steering;     ///< applied steering angle per step
    bool reached_end = false;
};

/// Drive a rear-axle kinematic bicycle (chapter 1) along the path with pure pursuit or Stanley at constant
/// speed, integrating with RK4. Stops when the rear axle projects onto the last 1 % of the path (or 5 cm of
/// it).
TrackingResult track_path(const KinematicBicycle& vehicle, const Points2D& path, const Pose& start,
                          const TrackingOptions& options = {});

}  // namespace motion_planning
