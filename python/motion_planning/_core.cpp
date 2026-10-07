// Python bindings for the motion_planning C++ library. One `bind_*` function per chapter.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "motion_planning/angles.hpp"
#include "motion_planning/kinematics.hpp"
#include "motion_planning/pose_control.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace motion_planning;

namespace {

template <class Model, class PyClass>
void add_integration(PyClass& cls) {
    cls.def(
           "step",
           [](const Model& m, const Pose& q, const Input& u, double dt, Integrator method) {
               return step(m, q, u, dt, method);
           },
           "q"_a, "u"_a, "dt"_a, "method"_a = Integrator::RK4, "One integration step with u held constant.")
        .def(
            "simulate",
            [](const Model& m, const Pose& q0, const InputSequence& inputs, double dt, Integrator method) {
                return simulate(m, q0, inputs, dt, method);
            },
            "q0"_a, "inputs"_a, "dt"_a, "method"_a = Integrator::RK4,
            "Apply an (N, 2) input sequence; returns (N + 1, 3) poses.");
}

void bind_kinematics(py::module_& m) {
    m.def("wrap_angle", py::vectorize(wrap_angle), "angle"_a, "Wrap to [-pi, pi).");
    m.def("angle_difference", py::vectorize(angle_difference), "to"_a, "from_"_a,
          "Smallest signed rotation from `from_` to `to`.");

    py::enum_<Integrator>(m, "Integrator")
        .value("Euler", Integrator::Euler)
        .value("Midpoint", Integrator::Midpoint)
        .value("RK4", Integrator::RK4);

    py::class_<Unicycle> uni(m, "Unicycle", "Unicycle model, input (v, omega). Eq. (1.2).");
    uni.def(py::init<>()).def("derivative", &Unicycle::derivative, "q"_a, "u"_a);
    add_integration<Unicycle>(uni);

    py::class_<DifferentialDrive> dd(
        m, "DifferentialDrive", "Differential drive, input (omega_left, omega_right). Eqs. (1.3)-(1.4).");
    dd.def(py::init<double, double>(), "wheel_radius"_a = 0.033, "track_width"_a = 0.160)
        .def_readonly("wheel_radius", &DifferentialDrive::wheel_radius)
        .def_readonly("track_width", &DifferentialDrive::track_width)
        .def("wheel_to_body", &DifferentialDrive::wheel_to_body, "wheels"_a)
        .def("body_to_wheel", &DifferentialDrive::body_to_wheel, "body"_a)
        .def("derivative", &DifferentialDrive::derivative, "q"_a, "wheels"_a);
    add_integration<DifferentialDrive>(dd);

    py::class_<KinematicBicycle> bike(m, "KinematicBicycle",
                                      "Kinematic bicycle, input (v, gamma). Eqs. (1.5)-(1.9).");
    bike.def(py::init<double, double, double>(), "wheelbase"_a = 1.0, "max_steering"_a = 0.7,
             "rear_to_reference"_a = 0.0)
        .def_readonly("wheelbase", &KinematicBicycle::wheelbase)
        .def_readonly("max_steering", &KinematicBicycle::max_steering)
        .def_readonly("rear_to_reference", &KinematicBicycle::rear_to_reference)
        .def("clamp_steering", &KinematicBicycle::clamp_steering, "gamma"_a)
        .def("turning_radius", &KinematicBicycle::turning_radius, "gamma"_a)
        .def("slip_angle", &KinematicBicycle::slip_angle, "gamma"_a)
        .def("yaw_rate", &KinematicBicycle::yaw_rate, "v"_a, "gamma"_a)
        .def("steering_for", &KinematicBicycle::steering_for, "v"_a, "omega"_a)
        .def("derivative", &KinematicBicycle::derivative, "q"_a, "u"_a);
    add_integration<KinematicBicycle>(bike);

    m.def("unicycle_exact_step", &unicycle_exact_step, "q"_a, "u"_a, "dt"_a, "Arc solution, eq. (1.16).");
    m.def("nonholonomic_residual", &nonholonomic_residual, "q"_a, "q_dot"_a, "Eq. (1.10).");
    m.def("unicycle_lie_bracket", &unicycle_lie_bracket, "q"_a, "Eq. (1.11).");

    py::class_<RateLimits>(m, "RateLimits", "Bounds on a command and its rate. Eq. (1.17).")
        .def(py::init([](double min_value, double max_value, double min_rate, double max_rate) {
                 return RateLimits{min_value, max_value, min_rate, max_rate};
             }),
             "min_value"_a = -INFINITY, "max_value"_a = INFINITY, "min_rate"_a = -INFINITY,
             "max_rate"_a = INFINITY)
        .def_readwrite("min_value", &RateLimits::min_value)
        .def_readwrite("max_value", &RateLimits::max_value)
        .def_readwrite("min_rate", &RateLimits::min_rate)
        .def_readwrite("max_rate", &RateLimits::max_rate);
    m.def("rate_limit", &rate_limit, "previous"_a, "desired"_a, "limits"_a, "dt"_a);
}

void bind_pose_control(py::module_& m) {
    py::class_<PolarError>(m, "PolarError")
        .def(py::init([](double rho, double alpha, double beta) { return PolarError{rho, alpha, beta}; }),
             "rho"_a, "alpha"_a, "beta"_a)
        .def_readwrite("rho", &PolarError::rho)
        .def_readwrite("alpha", &PolarError::alpha)
        .def_readwrite("beta", &PolarError::beta)
        .def("__repr__", [](const PolarError& e) {
            return "PolarError(rho=" + std::to_string(e.rho) + ", alpha=" + std::to_string(e.alpha) +
                   ", beta=" + std::to_string(e.beta) + ")";
        });
    m.def("polar_error", &polar_error, "q"_a, "goal"_a, "Eq. (1.19).");

    py::class_<PointGains>(m, "PointGains")
        .def(py::init([](double k_v, double k_h) { return PointGains{k_v, k_h}; }), "k_v"_a = 0.5,
             "k_h"_a = 2.0)
        .def_readwrite("k_v", &PointGains::k_v)
        .def_readwrite("k_h", &PointGains::k_h);
    py::class_<PoseGains>(m, "PoseGains")
        .def(py::init([](double kr, double ka, double kb) { return PoseGains{kr, ka, kb}; }), "k_rho"_a = 0.5,
             "k_alpha"_a = 1.5, "k_beta"_a = -0.6)
        .def_readwrite("k_rho", &PoseGains::k_rho)
        .def_readwrite("k_alpha", &PoseGains::k_alpha)
        .def_readwrite("k_beta", &PoseGains::k_beta)
        .def("is_stable", &PoseGains::is_stable, "Eq. (1.23).");

    m.def("move_to_point", &move_to_point, "q"_a, "goal"_a, "gains"_a = PointGains{}, "Eq. (1.18).");
    m.def("goal_is_behind", &goal_is_behind, "q"_a, "goal"_a);
    m.def("move_to_pose", &move_to_pose, "q"_a, "goal"_a, "gains"_a = PoseGains{}, "reverse"_a = false,
          "Eq. (1.21).");
    m.def("polar_closed_loop_rates", &polar_closed_loop_rates, "e"_a, "gains"_a, "Eq. (1.22).");

    py::class_<RegulationOptions>(m, "RegulationOptions")
        .def(py::init([](double dt, int max_steps, double pos_tol, double head_tol, RateLimits speed,
                         bool allow_reverse) {
                 return RegulationOptions{dt, max_steps, pos_tol, head_tol, speed, allow_reverse};
             }),
             "dt"_a = 0.05, "max_steps"_a = 4000, "position_tolerance"_a = 0.05,
             "heading_tolerance"_a = INFINITY, "speed"_a = RateLimits{}, "allow_reverse"_a = true)
        .def_readwrite("dt", &RegulationOptions::dt)
        .def_readwrite("max_steps", &RegulationOptions::max_steps)
        .def_readwrite("position_tolerance", &RegulationOptions::position_tolerance)
        .def_readwrite("heading_tolerance", &RegulationOptions::heading_tolerance)
        .def_readwrite("speed", &RegulationOptions::speed)
        .def_readwrite("allow_reverse", &RegulationOptions::allow_reverse);
    py::class_<RegulationResult>(m, "RegulationResult")
        .def_readonly("states", &RegulationResult::states)
        .def_readonly("inputs", &RegulationResult::inputs)
        .def_readonly("converged", &RegulationResult::converged);

    m.def("regulate_pose",
          py::overload_cast<const Unicycle&, const Pose&, const Pose&, const PoseGains&,
                            const RegulationOptions&>(&regulate_pose),
          "model"_a, "q0"_a, "goal"_a, "gains"_a = PoseGains{}, "options"_a = RegulationOptions{});
    m.def("regulate_pose",
          py::overload_cast<const KinematicBicycle&, const Pose&, const Pose&, const PoseGains&,
                            const RegulationOptions&>(&regulate_pose),
          "model"_a, "q0"_a, "goal"_a, "gains"_a = PoseGains{}, "options"_a = RegulationOptions{});
    m.def("regulate_point",
          py::overload_cast<const Unicycle&, const Pose&, const Eigen::Vector2d&, const PointGains&,
                            const RegulationOptions&>(&regulate_point),
          "model"_a, "q0"_a, "goal"_a, "gains"_a = PointGains{}, "options"_a = RegulationOptions{});
    m.def("regulate_point",
          py::overload_cast<const KinematicBicycle&, const Pose&, const Eigen::Vector2d&, const PointGains&,
                            const RegulationOptions&>(&regulate_point),
          "model"_a, "q0"_a, "goal"_a, "gains"_a = PointGains{}, "options"_a = RegulationOptions{});
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core of the motion_planning learning module.";
    bind_kinematics(m);
    bind_pose_control(m);
}
