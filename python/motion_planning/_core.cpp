// Python bindings for the motion_planning C++ library. One `bind_*` function per chapter.
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

#include "motion_planning/angles.hpp"
#include "motion_planning/dubins.hpp"
#include "motion_planning/dwa.hpp"
#include "motion_planning/graph_search.hpp"
#include "motion_planning/grid.hpp"
#include "motion_planning/kinematics.hpp"
#include "motion_planning/pose_control.hpp"
#include "motion_planning/potential.hpp"
#include "motion_planning/sampling.hpp"
#include "motion_planning/tracking.hpp"

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

Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> cells_to_array(const std::vector<Cell>& cells) {
    Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> a(static_cast<Eigen::Index>(cells.size()), 2);
    for (std::size_t i = 0; i < cells.size(); ++i)
        a.row(static_cast<Eigen::Index>(i)) << cells[i].x, cells[i].y;
    return a;
}

void bind_grid(py::module_& m) {
    py::class_<Cell>(m, "Cell", "Integer cell index (x = column, y = row).")
        .def(py::init<>())
        .def(py::init([](int x, int y) { return Cell{x, y}; }), "x"_a, "y"_a)
        .def(py::init([](py::sequence s) {
            if (py::len(s) != 2) throw py::value_error("a cell needs two indices");
            return Cell{s[0].cast<int>(), s[1].cast<int>()};
        }))
        .def_readwrite("x", &Cell::x)
        .def_readwrite("y", &Cell::y)
        .def("__eq__", &Cell::operator==)
        .def("__hash__", [](const Cell& c) { return py::hash(py::make_tuple(c.x, c.y)); })
        .def("__iter__", [](const Cell& c) { return py::iter(py::make_tuple(c.x, c.y)); })
        .def("__repr__",
             [](const Cell& c) { return "Cell(" + std::to_string(c.x) + ", " + std::to_string(c.y) + ")"; });
    py::implicitly_convertible<py::tuple, Cell>();
    py::implicitly_convertible<py::list, Cell>();

    py::enum_<Connectivity>(m, "Connectivity")
        .value("Four", Connectivity::Four)
        .value("Eight", Connectivity::Eight);

    py::class_<OccupancyGrid>(
        m, "OccupancyGrid",
        "Binary occupancy grid; array rows are y, columns are x; outside counts as occupied.")
        .def(py::init<int, int, double, const Eigen::Vector2d&>(), "width"_a, "height"_a,
             "resolution"_a = 1.0, "origin"_a = Eigen::Vector2d::Zero())
        .def(py::init([](py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> a, double res,
                         const Eigen::Vector2d& origin) {
                 if (a.ndim() != 2) throw py::value_error("occupancy must be a 2-D array");
                 GridArray g(a.shape(0), a.shape(1));
                 std::copy(a.data(), a.data() + a.size(), g.data());
                 return OccupancyGrid(g, res, origin);
             }),
             "occupancy"_a, "resolution"_a = 1.0, "origin"_a = Eigen::Vector2d::Zero())
        .def_property_readonly("width", &OccupancyGrid::width)
        .def_property_readonly("height", &OccupancyGrid::height)
        .def_property_readonly("resolution", &OccupancyGrid::resolution)
        .def_property_readonly("origin", &OccupancyGrid::origin)
        .def_property_readonly(
            "extent",
            [](const OccupancyGrid& g) {
                const double h = g.resolution();
                return py::make_tuple(g.origin().x(), g.origin().x() + h * g.width(), g.origin().y(),
                                      g.origin().y() + h * g.height());
            },
            "(xmin, xmax, ymin, ymax), for matplotlib's imshow(extent=...).")
        .def("in_bounds", &OccupancyGrid::in_bounds, "cell"_a)
        .def("occupied", &OccupancyGrid::occupied, "cell"_a)
        .def("set", &OccupancyGrid::set, "cell"_a, "occupied"_a)
        .def("world_to_cell", &OccupancyGrid::world_to_cell, "p"_a)
        .def("cell_center", &OccupancyGrid::cell_center, "cell"_a)
        .def("point_free", &OccupancyGrid::point_free, "p"_a)
        .def("to_array", &OccupancyGrid::to_array);

    m.def("brushfire", &brushfire, "grid"_a, "connectivity"_a = Connectivity::Four, "Eq. (2.6).");
    m.def("distance_transform", &distance_transform, "grid"_a,
          "Exact Euclidean transform, eqs. (2.7)-(2.9).");
    m.def("squared_distance_transform_1d", &squared_distance_transform_1d, "f"_a, "Eq. (2.8).");
    m.def("inflate", &inflate, "grid"_a, "radius"_a, "Eq. (2.10).");
    m.def("disc_free", &disc_free, "grid"_a, "distance"_a, "p"_a, "radius"_a, "Eq. (2.11).");
    m.def(
        "traverse_segment",
        [](const OccupancyGrid& g, const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
            return cells_to_array(traverse_segment(g, a, b));
        },
        "grid"_a, "p0"_a, "p1"_a, "Cells crossed by the segment, as an (N, 2) array of (x, y).");
    m.def(
        "bresenham", [](const Cell& a, const Cell& b) { return cells_to_array(bresenham(a, b)); }, "a"_a,
        "b"_a, "Bresenham cells from a to b, as an (N, 2) array of (x, y).");
    m.def("segment_free", &segment_free, "grid"_a, "p0"_a, "p1"_a);

    py::class_<RectangleFootprint>(m, "RectangleFootprint")
        .def(py::init([](double rear, double front, double half_width) {
                 return RectangleFootprint{rear, front, half_width};
             }),
             "rear"_a = 0.0, "front"_a = 1.0, "half_width"_a = 0.25)
        .def_readwrite("rear", &RectangleFootprint::rear)
        .def_readwrite("front", &RectangleFootprint::front)
        .def_readwrite("half_width", &RectangleFootprint::half_width)
        .def("corners", &RectangleFootprint::corners, "q"_a);
    m.def("footprint_free", &footprint_free, "grid"_a, "q"_a, "footprint"_a, "Eq. (2.12).");
    m.def("configuration_space_slice", &configuration_space_slice, "grid"_a, "footprint"_a, "theta"_a,
          "Eq. (2.2).");
}

void bind_potential(py::module_& m) {
    py::class_<PotentialParams>(m, "PotentialParams")
        .def(py::init([](double zeta, double d_star, double eta, double q_star) {
                 return PotentialParams{zeta, d_star, eta, q_star};
             }),
             "zeta"_a = 1.0, "d_star"_a = 2.0, "eta"_a = 1.0, "q_star"_a = 1.0)
        .def_readwrite("zeta", &PotentialParams::zeta)
        .def_readwrite("d_star", &PotentialParams::d_star)
        .def_readwrite("eta", &PotentialParams::eta)
        .def_readwrite("q_star", &PotentialParams::q_star);
    m.def("attractive_potential", &attractive_potential, "p"_a, "goal"_a, "params"_a, "Eq. (3.1).");
    m.def("attractive_gradient", &attractive_gradient, "p"_a, "goal"_a, "params"_a, "Eq. (3.2).");
    m.def("repulsive_potential",
          py::vectorize([](double c, PotentialParams k) { return repulsive_potential(c, k); }), "clearance"_a,
          "params"_a, "Eq. (3.3).");
    m.def("attractive_field", &attractive_field, "grid"_a, "goal"_a, "params"_a);
    m.def("repulsive_field", &repulsive_field, "grid"_a, "params"_a, "Eq. (3.3) on the distance transform.");
    m.def("total_field", &total_field, "grid"_a, "goal"_a, "params"_a, "Eq. (3.4).");
    m.def("wavefront", &wavefront, "grid"_a, "goal"_a, "connectivity"_a = Connectivity::Four, "Eq. (3.6).");

    py::class_<DescentResult>(m, "DescentResult")
        .def_property_readonly(
            "path", [](const DescentResult& r) { return cells_to_array(r.path); },
            "(N, 2) array of visited cells (x, y)")
        .def_readonly("reached_goal", &DescentResult::reached_goal)
        .def_readonly("local_minimum", &DescentResult::local_minimum);
    m.def("descend", &descend, "field"_a, "start"_a, "goal"_a, "connectivity"_a = Connectivity::Eight,
          "max_steps"_a = 100000, "Discrete steepest descent, section 3.3.");
}

void bind_graph_search(py::module_& m) {
    py::enum_<SearchAlgorithm>(m, "SearchAlgorithm")
        .value("BreadthFirst", SearchAlgorithm::BreadthFirst)
        .value("Dijkstra", SearchAlgorithm::Dijkstra)
        .value("AStar", SearchAlgorithm::AStar)
        .value("WeightedAStar", SearchAlgorithm::WeightedAStar)
        .value("GreedyBestFirst", SearchAlgorithm::GreedyBestFirst);
    py::enum_<Heuristic>(m, "Heuristic")
        .value("Zero", Heuristic::Zero)
        .value("Euclidean", Heuristic::Euclidean)
        .value("Manhattan", Heuristic::Manhattan)
        .value("Octile", Heuristic::Octile)
        .value("Chebyshev", Heuristic::Chebyshev);

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("found", &SearchResult::found)
        .def_readonly("path", &SearchResult::path)
        .def_readonly("cost", &SearchResult::cost)
        .def_readonly("expanded", &SearchResult::expanded);
    m.def(
        "best_first_search",
        [](int n, int start, int goal,
           const std::function<std::vector<std::pair<int, double>>(int)>& neighbours,
           const std::function<double(int)>& heuristic, SearchAlgorithm algorithm, double weight) {
            return best_first_search(
                n, start, goal,
                [&](int u, std::vector<std::pair<int, double>>& out) {
                    for (const auto& e : neighbours(u)) out.push_back(e);
                },
                heuristic, algorithm, weight);
        },
        "num_nodes"_a, "start"_a, "goal"_a, "neighbours"_a, "heuristic"_a, "algorithm"_a, "weight"_a = 1.0,
        "Generic search; `neighbours(u)` returns a list of (v, cost). Section 4.3.");
    m.def("grid_heuristic", &grid_heuristic, "kind"_a, "a"_a, "b"_a, "resolution"_a = 1.0);

    py::class_<GridSearchOptions>(m, "GridSearchOptions")
        .def(py::init([](SearchAlgorithm a, Connectivity c, Heuristic h, double w, bool cut) {
                 return GridSearchOptions{a, c, h, w, cut};
             }),
             "algorithm"_a = SearchAlgorithm::AStar, "connectivity"_a = Connectivity::Eight,
             "heuristic"_a = Heuristic::Octile, "weight"_a = 1.0, "allow_corner_cutting"_a = false)
        .def_readwrite("algorithm", &GridSearchOptions::algorithm)
        .def_readwrite("connectivity", &GridSearchOptions::connectivity)
        .def_readwrite("heuristic", &GridSearchOptions::heuristic)
        .def_readwrite("weight", &GridSearchOptions::weight)
        .def_readwrite("allow_corner_cutting", &GridSearchOptions::allow_corner_cutting);
    py::class_<GridSearchResult>(m, "GridSearchResult")
        .def_readonly("found", &GridSearchResult::found)
        .def_readonly("cost", &GridSearchResult::cost)
        .def_property_readonly("path", [](const GridSearchResult& r) { return cells_to_array(r.path); })
        .def_property_readonly("expanded",
                               [](const GridSearchResult& r) { return cells_to_array(r.expanded); });
    m.def(
        "grid_search",
        [](const OccupancyGrid& g, const Cell& s, const Cell& t, const GridSearchOptions& o,
           std::optional<FieldArray> cost) { return grid_search(g, s, t, o, cost ? *cost : FieldArray()); },
        "grid"_a, "start"_a, "goal"_a, "options"_a = GridSearchOptions{}, "cell_cost"_a = py::none(),
        "Search an occupancy grid; optional per-cell cost factor. Eq. (4.9).");

    py::class_<Graph>(m, "Graph", "Explicit graph with points in the plane.")
        .def(py::init<>())
        .def("add_node", &Graph::add_node, "p"_a)
        .def("add_edge", &Graph::add_edge, "a"_a, "b"_a, "cost"_a = -1.0, "bidirectional"_a = true)
        .def("__len__", &Graph::size)
        .def_property_readonly("num_edges", &Graph::num_edges)
        .def("point", &Graph::point, "i"_a)
        .def("edges", &Graph::edges, "i"_a)
        .def_property_readonly("points", [](const Graph& g) {
            Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> p(g.size(), 2);
            for (int i = 0; i < g.size(); ++i) p.row(i) = g.point(i).transpose();
            return p;
        });
    m.def("graph_search", &graph_search, "graph"_a, "start"_a, "goal"_a,
          "algorithm"_a = SearchAlgorithm::AStar, "weight"_a = 1.0);
    m.def("point_in_polygon", &point_in_polygon, "p"_a, "polygon"_a);
    m.def("segment_clear_of_polygons", &segment_clear_of_polygons, "a"_a, "b"_a, "polygons"_a);
    m.def("visibility_graph", &visibility_graph, "polygons"_a, "points"_a, "Section 4.6.");
}

using PointArray = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

PointArray points_to_array(const std::vector<Eigen::Vector2d>& pts) {
    PointArray a(static_cast<Eigen::Index>(pts.size()), 2);
    for (std::size_t i = 0; i < pts.size(); ++i) a.row(static_cast<Eigen::Index>(i)) = pts[i].transpose();
    return a;
}

void bind_sampling(py::module_& m) {
    py::class_<PlanningProblem2D>(m, "PlanningProblem2D",
                                  "Point robot in a rectangle; validity checks may be Python callables.")
        .def(py::init<>())
        .def_readwrite("lower", &PlanningProblem2D::lower)
        .def_readwrite("upper", &PlanningProblem2D::upper)
        .def_readwrite("state_valid", &PlanningProblem2D::state_valid)
        .def_readwrite("motion_valid", &PlanningProblem2D::motion_valid)
        .def_readwrite("start", &PlanningProblem2D::start)
        .def_readwrite("goal", &PlanningProblem2D::goal)
        .def_readwrite("goal_radius", &PlanningProblem2D::goal_radius);
    m.def("problem_from_grid", &problem_from_grid, "grid"_a, "start"_a, "goal"_a);
    m.def("problem_from_polygons", &problem_from_polygons, "polygons"_a, "lower"_a, "upper"_a, "start"_a,
          "goal"_a);
    m.def("rewiring_radius", &rewiring_radius, "n"_a, "gamma"_a, "Eq. (5.3).");
    m.def("optimal_gamma", &optimal_gamma, "free_area"_a, "Eq. (5.4).");

    py::class_<PrmOptions>(m, "PrmOptions")
        .def(py::init([](int n, double radius, double gamma, std::uint32_t seed) {
                 return PrmOptions{n, radius, gamma, seed};
             }),
             "num_samples"_a = 500, "connection_radius"_a = -1.0, "gamma"_a = 0.0, "seed"_a = 1)
        .def_readwrite("num_samples", &PrmOptions::num_samples)
        .def_readwrite("connection_radius", &PrmOptions::connection_radius)
        .def_readwrite("gamma", &PrmOptions::gamma)
        .def_readwrite("seed", &PrmOptions::seed);
    py::class_<PrmResult>(m, "PrmResult")
        .def_readonly("roadmap", &PrmResult::roadmap)
        .def_readonly("query", &PrmResult::query)
        .def_readonly("collision_checks", &PrmResult::collision_checks);
    m.def("prm", &prm, "problem"_a, "options"_a = PrmOptions{}, "Section 5.2.");

    py::class_<RrtOptions>(m, "RrtOptions")
        .def(py::init([](int iters, double step, double bias, bool stop, double gamma, std::uint32_t seed) {
                 return RrtOptions{iters, step, bias, stop, gamma, seed};
             }),
             "max_iterations"_a = 5000, "step"_a = 0.5, "goal_bias"_a = 0.05,
             "stop_at_first_solution"_a = true, "gamma"_a = 0.0, "seed"_a = 1)
        .def_readwrite("max_iterations", &RrtOptions::max_iterations)
        .def_readwrite("step", &RrtOptions::step)
        .def_readwrite("goal_bias", &RrtOptions::goal_bias)
        .def_readwrite("stop_at_first_solution", &RrtOptions::stop_at_first_solution)
        .def_readwrite("gamma", &RrtOptions::gamma)
        .def_readwrite("seed", &RrtOptions::seed);
    py::class_<TreeResult>(m, "TreeResult")
        .def_property_readonly("nodes", [](const TreeResult& t) { return points_to_array(t.nodes); })
        .def_readonly("parent", &TreeResult::parent)
        .def_readonly("cost", &TreeResult::cost)
        .def_readonly("found", &TreeResult::found)
        .def_property_readonly("path", [](const TreeResult& t) { return points_to_array(t.path); })
        .def_readonly("path_cost", &TreeResult::path_cost)
        .def_readonly("first_solution_iteration", &TreeResult::first_solution_iteration)
        .def_readonly("best_cost_history", &TreeResult::best_cost_history);
    m.def("rrt", &rrt, "problem"_a, "options"_a = RrtOptions{}, "Section 5.3.");
    m.def("rrt_star", &rrt_star, "problem"_a, "options"_a = RrtOptions{}, "Section 5.4.");
    m.def(
        "path_length",
        [](const PointArray& p) {
            double L = 0.0;
            for (Eigen::Index i = 1; i < p.rows(); ++i) L += (p.row(i) - p.row(i - 1)).norm();
            return L;
        },
        "path"_a);
}

void bind_dubins(py::module_& m) {
    py::enum_<DubinsWord>(m, "DubinsWord")
        .value("LSL", DubinsWord::LSL)
        .value("RSR", DubinsWord::RSR)
        .value("LSR", DubinsWord::LSR)
        .value("RSL", DubinsWord::RSL)
        .value("RLR", DubinsWord::RLR)
        .value("LRL", DubinsWord::LRL);
    py::class_<DubinsPath>(m, "DubinsPath")
        .def_readonly("start", &DubinsPath::start)
        .def_readonly("word", &DubinsPath::word)
        .def_readonly("lengths", &DubinsPath::lengths)
        .def_readonly("radius", &DubinsPath::radius)
        .def("length", &DubinsPath::length)
        .def("at", &DubinsPath::at, "s"_a)
        .def("sample", &DubinsPath::sample, "step"_a)
        .def("truncated", &DubinsPath::truncated, "s"_a)
        .def("__repr__", [](const DubinsPath& p) {
            return "DubinsPath(" + to_string(p.word) + ", length=" + std::to_string(p.length()) + ")";
        });
    m.def("dubins_path", &dubins_path, "q0"_a, "q1"_a, "radius"_a, "word"_a,
          "Eqs. (6.3)-(6.8); None if infeasible.");
    m.def("all_dubins_paths", &all_dubins_paths, "q0"_a, "q1"_a, "radius"_a);
    m.def("shortest_dubins_path", &shortest_dubins_path, "q0"_a, "q1"_a, "radius"_a);
    m.def("dubins_distance", &dubins_distance, "q0"_a, "q1"_a, "radius"_a);
    m.def("dubins_path_valid", &dubins_path_valid, "path"_a, "problem"_a, "step"_a, "Eq. (6.9).");

    py::class_<DubinsPlannerOptions>(m, "DubinsPlannerOptions")
        .def(py::init([](int iters, double radius, double step, double bias, double gamma, double cstep,
                         bool star, std::uint32_t seed) {
                 return DubinsPlannerOptions{iters, radius, step, bias, gamma, cstep, star, seed};
             }),
             "max_iterations"_a = 3000, "radius"_a = 1.0, "step"_a = 3.0, "goal_bias"_a = 0.05,
             "gamma"_a = 0.0, "collision_step"_a = 0.05, "star"_a = true, "seed"_a = 1)
        .def_readwrite("max_iterations", &DubinsPlannerOptions::max_iterations)
        .def_readwrite("radius", &DubinsPlannerOptions::radius)
        .def_readwrite("step", &DubinsPlannerOptions::step)
        .def_readwrite("goal_bias", &DubinsPlannerOptions::goal_bias)
        .def_readwrite("gamma", &DubinsPlannerOptions::gamma)
        .def_readwrite("collision_step", &DubinsPlannerOptions::collision_step)
        .def_readwrite("star", &DubinsPlannerOptions::star)
        .def_readwrite("seed", &DubinsPlannerOptions::seed);
    py::class_<DubinsTreeResult>(m, "DubinsTreeResult")
        .def_readonly("nodes", &DubinsTreeResult::nodes)
        .def_readonly("parent", &DubinsTreeResult::parent)
        .def_readonly("cost", &DubinsTreeResult::cost)
        .def_readonly("found", &DubinsTreeResult::found)
        .def_readonly("segments", &DubinsTreeResult::segments)
        .def_readonly("path_cost", &DubinsTreeResult::path_cost)
        .def_readonly("best_cost_history", &DubinsTreeResult::best_cost_history)
        .def(
            "sample_path",
            [](const DubinsTreeResult& t, double step) {
                std::vector<Pose> poses;
                for (const DubinsPath& s : t.segments) {
                    const Trajectory tr = s.sample(step);
                    for (Eigen::Index k = poses.empty() ? 0 : 1; k < tr.rows(); ++k)
                        poses.push_back(tr.row(k));
                }
                Trajectory out(static_cast<Eigen::Index>(poses.size()), 3);
                for (std::size_t i = 0; i < poses.size(); ++i)
                    out.row(static_cast<Eigen::Index>(i)) = poses[i];
                return out;
            },
            "step"_a = 0.05, "The whole path as poses every `step` metres.");
    m.def("dubins_rrt_star", &dubins_rrt_star, "problem"_a, "start"_a, "goal"_a,
          "options"_a = DubinsPlannerOptions{}, "Section 6.4.");
}

void bind_tracking(py::module_& m) {
    m.def("shortcut_greedy", &shortcut_greedy, "path"_a, "motion_valid"_a, "Section 7.1.");
    m.def("shortcut_random", &shortcut_random, "path"_a, "motion_valid"_a, "iterations"_a = 200, "seed"_a = 1,
          "Section 7.1.");
    m.def("turning_angles", &turning_angles, "path"_a);
    py::class_<PathProjection>(m, "PathProjection")
        .def_readonly("s", &PathProjection::s)
        .def_readonly("point", &PathProjection::point)
        .def_readonly("heading", &PathProjection::heading)
        .def_readonly("lateral", &PathProjection::lateral);
    py::class_<Path2D>(m, "Path2D")
        .def(py::init<const Points2D&>(), "points"_a)
        .def("length", &Path2D::length)
        .def_property_readonly("points", &Path2D::points)
        .def("point_at", &Path2D::point_at, "s"_a)
        .def("heading_at", &Path2D::heading_at, "s"_a)
        .def("project", &Path2D::project, "p"_a, "Eq. (7.1).");
    m.def("pure_pursuit_curvature", &pure_pursuit_curvature, "q"_a, "target"_a, "Eq. (7.3).");
    m.def("stanley_steering", &stanley_steering, "heading_error"_a, "lateral_error"_a, "speed"_a, "gain"_a,
          "softening"_a, "Eq. (7.5).");
    py::enum_<TrackingController>(m, "TrackingController")
        .value("PurePursuit", TrackingController::PurePursuit)
        .value("Stanley", TrackingController::Stanley);
    py::class_<TrackingOptions>(m, "TrackingOptions")
        .def(py::init([](TrackingController c, double speed, double lookahead, double lookahead_gain,
                         double k, double ks, double rate, double dt, int max_steps) {
                 return TrackingOptions{c, speed, lookahead, lookahead_gain, k, ks, rate, dt, max_steps};
             }),
             "controller"_a = TrackingController::PurePursuit, "speed"_a = 1.0, "lookahead"_a = 1.0,
             "lookahead_gain"_a = 0.0, "stanley_gain"_a = 1.0, "softening"_a = 0.1,
             "steering_rate"_a = INFINITY, "dt"_a = 0.02, "max_steps"_a = 20000)
        .def_readwrite("controller", &TrackingOptions::controller)
        .def_readwrite("speed", &TrackingOptions::speed)
        .def_readwrite("lookahead", &TrackingOptions::lookahead)
        .def_readwrite("lookahead_gain", &TrackingOptions::lookahead_gain)
        .def_readwrite("stanley_gain", &TrackingOptions::stanley_gain)
        .def_readwrite("softening", &TrackingOptions::softening)
        .def_readwrite("steering_rate", &TrackingOptions::steering_rate)
        .def_readwrite("dt", &TrackingOptions::dt)
        .def_readwrite("max_steps", &TrackingOptions::max_steps);
    py::class_<TrackingResult>(m, "TrackingResult")
        .def_readonly("states", &TrackingResult::states)
        .def_readonly("cross_track", &TrackingResult::cross_track)
        .def_readonly("steering", &TrackingResult::steering)
        .def_readonly("reached_end", &TrackingResult::reached_end);
    m.def("track_path", &track_path, "vehicle"_a, "path"_a, "start"_a, "options"_a = TrackingOptions{});
}

void bind_dwa(py::module_& m) {
    py::class_<DwaConfig>(m, "DwaConfig",
                          "Robot limits and DWA objective weights; all fields keyword-settable.")
        .def(py::init([](py::kwargs kw) {
            DwaConfig c;
            py::object o = py::cast(&c, py::return_value_policy::reference);
            for (auto item : kw) py::setattr(o, item.first, item.second);
            return c;
        }))
        .def_readwrite("max_speed", &DwaConfig::max_speed)
        .def_readwrite("min_speed", &DwaConfig::min_speed)
        .def_readwrite("max_yaw_rate", &DwaConfig::max_yaw_rate)
        .def_readwrite("max_accel", &DwaConfig::max_accel)
        .def_readwrite("max_yaw_accel", &DwaConfig::max_yaw_accel)
        .def_readwrite("control_period", &DwaConfig::control_period)
        .def_readwrite("horizon", &DwaConfig::horizon)
        .def_readwrite("sim_step", &DwaConfig::sim_step)
        .def_readwrite("v_samples", &DwaConfig::v_samples)
        .def_readwrite("w_samples", &DwaConfig::w_samples)
        .def_readwrite("robot_radius", &DwaConfig::robot_radius)
        .def_readwrite("heading_weight", &DwaConfig::heading_weight)
        .def_readwrite("distance_weight", &DwaConfig::distance_weight)
        .def_readwrite("velocity_weight", &DwaConfig::velocity_weight)
        .def_readwrite("distance_cap", &DwaConfig::distance_cap);
    py::class_<DynamicWindow>(m, "DynamicWindow")
        .def_readonly("v_min", &DynamicWindow::v_min)
        .def_readonly("v_max", &DynamicWindow::v_max)
        .def_readonly("w_min", &DynamicWindow::w_min)
        .def_readonly("w_max", &DynamicWindow::w_max);
    m.def("dynamic_window", &dynamic_window, "current"_a, "config"_a, "Eqs. (8.1)-(8.2).");
    m.def("rollout", &rollout, "q"_a, "v"_a, "w"_a, "config"_a);
    py::class_<DwaCandidate>(m, "DwaCandidate")
        .def_readonly("v", &DwaCandidate::v)
        .def_readonly("w", &DwaCandidate::w)
        .def_readonly("heading", &DwaCandidate::heading)
        .def_readonly("distance", &DwaCandidate::distance)
        .def_readonly("velocity", &DwaCandidate::velocity)
        .def_readonly("total", &DwaCandidate::total)
        .def_readonly("free_distance", &DwaCandidate::free_distance)
        .def_readonly("admissible", &DwaCandidate::admissible);
    py::class_<DwaDecision>(m, "DwaDecision")
        .def_readonly("found", &DwaDecision::found)
        .def_readonly("command", &DwaDecision::command)
        .def_readonly("candidates", &DwaDecision::candidates)
        .def_readonly("best_rollout", &DwaDecision::best_rollout);
    m.def("dwa_step", &dwa_step, "grid"_a, "distance"_a, "q"_a, "current"_a, "goal"_a, "config"_a,
          "Section 8.3.");
    py::class_<DwaRun>(m, "DwaRun")
        .def_readonly("states", &DwaRun::states)
        .def_readonly("commands", &DwaRun::commands)
        .def_readonly("reached_goal", &DwaRun::reached_goal)
        .def_readonly("stuck", &DwaRun::stuck);
    m.def("run_dwa", &run_dwa, "grid"_a, "start"_a, "goal"_a, "config"_a, "global_path"_a = Points2D(),
          "carrot_distance"_a = 1.0, "goal_tolerance"_a = 0.15, "max_time"_a = 120.0, "stuck_time"_a = 10.0,
          "Section 8.4.");
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core of the motion_planning learning module.";
    bind_kinematics(m);
    bind_pose_control(m);
    bind_grid(m);
    bind_potential(m);
    bind_graph_search(m);
    bind_sampling(m);
    bind_dubins(m);
    bind_tracking(m);
    bind_dwa(m);
}
