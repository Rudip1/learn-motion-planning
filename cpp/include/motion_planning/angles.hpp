#pragma once

#include <cmath>

namespace motion_planning {

constexpr double kPi = 3.14159265358979323846;

/// Wrap an angle to the half-open interval [-pi, pi).
///
/// See "Angles" in 1_theory/00_notation.md: every heading difference in the module goes through this
/// function, never through a raw subtraction.
inline double wrap_angle(double angle) {
    double wrapped = std::fmod(angle + kPi, 2.0 * kPi);
    if (wrapped < 0.0) wrapped += 2.0 * kPi;
    return wrapped - kPi;
}

/// Signed smallest rotation that takes heading `from` to heading `to`, in [-pi, pi).
inline double angle_difference(double to, double from) { return wrap_angle(to - from); }

}  // namespace motion_planning
