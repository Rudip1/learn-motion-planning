#include "motion_planning/dubins.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include "motion_planning/angles.hpp"

namespace motion_planning {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Angle in [0, 2 pi). Values within round-off of 2 pi snap to 0: otherwise a tangent case whose segment angle
// comes out as -1e-16 turns into a full extra loop.
double mod2pi(double a) {
    double r = std::fmod(a, 2.0 * kPi);
    if (r < 0.0) r += 2.0 * kPi;
    return r > 2.0 * kPi - 1e-10 ? 0.0 : r;
}

// Turn direction of each segment: +1 left, -1 right, 0 straight.
std::array<int, 3> segment_types(DubinsWord w) {
    switch (w) {
        case DubinsWord::LSL:
            return {1, 0, 1};
        case DubinsWord::RSR:
            return {-1, 0, -1};
        case DubinsWord::LSR:
            return {1, 0, -1};
        case DubinsWord::RSL:
            return {-1, 0, 1};
        case DubinsWord::RLR:
            return {-1, 1, -1};
        case DubinsWord::LRL:
            return {1, -1, 1};
    }
    return {0, 0, 0};
}

// Normalised segment lengths (t, p, q) for unit radius, eqs. (6.3)-(6.8). Empty if the word is infeasible.
std::optional<std::array<double, 3>> word_lengths(DubinsWord w, double d, double a, double b) {
    const double sa = std::sin(a), sb = std::sin(b), ca = std::cos(a), cb = std::cos(b),
                 cab = std::cos(a - b);
    switch (w) {
        case DubinsWord::LSL: {  // eq. (6.3)
            const double p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sa - sb);
            if (p2 < 0.0) return std::nullopt;
            // both turning circles coincide: one arc, and the tangent direction below would be atan2(0, 0)
            if (p2 < 1e-14) return std::array<double, 3>{mod2pi(b - a), 0.0, 0.0};
            const double tmp = std::atan2(cb - ca, d + sa - sb);
            return std::array<double, 3>{mod2pi(-a + tmp), std::sqrt(p2), mod2pi(b - tmp)};
        }
        case DubinsWord::RSR: {  // eq. (6.4)
            const double p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sb - sa);
            if (p2 < 0.0) return std::nullopt;
            if (p2 < 1e-14) return std::array<double, 3>{mod2pi(a - b), 0.0, 0.0};  // as for LSL
            const double tmp = std::atan2(ca - cb, d - sa + sb);
            return std::array<double, 3>{mod2pi(a - tmp), std::sqrt(p2), mod2pi(-b + tmp)};
        }
        case DubinsWord::LSR: {  // eq. (6.5)
            const double p2 = -2.0 + d * d + 2.0 * cab + 2.0 * d * (sa + sb);
            if (p2 < 0.0) return std::nullopt;
            const double p = std::sqrt(p2);
            const double tmp = std::atan2(-ca - cb, d + sa + sb) - std::atan2(-2.0, p);
            return std::array<double, 3>{mod2pi(-a + tmp), p, mod2pi(-b + tmp)};
        }
        case DubinsWord::RSL: {  // eq. (6.6)
            const double p2 = -2.0 + d * d + 2.0 * cab - 2.0 * d * (sa + sb);
            if (p2 < 0.0) return std::nullopt;
            const double p = std::sqrt(p2);
            const double tmp = std::atan2(ca + cb, d - sa - sb) - std::atan2(2.0, p);
            return std::array<double, 3>{mod2pi(a - tmp), p, mod2pi(b - tmp)};
        }
        case DubinsWord::RLR: {  // eq. (6.7)
            const double c = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sa - sb)) / 8.0;
            if (std::abs(c) > 1.0) return std::nullopt;
            const double p = mod2pi(2.0 * kPi - std::acos(c));
            const double t = mod2pi(a - std::atan2(ca - cb, d - sa + sb) + p / 2.0);
            return std::array<double, 3>{t, p, mod2pi(a - b - t + p)};
        }
        case DubinsWord::LRL: {  // eq. (6.8)
            const double c = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sb - sa)) / 8.0;
            if (std::abs(c) > 1.0) return std::nullopt;
            const double p = mod2pi(2.0 * kPi - std::acos(c));
            const double t = mod2pi(-a - std::atan2(ca - cb, d + sa - sb) + p / 2.0);
            return std::array<double, 3>{t, p, mod2pi(b - a - t + p)};
        }
    }
    return std::nullopt;
}

constexpr std::array<DubinsWord, 6> kWords{DubinsWord::LSL, DubinsWord::RSR, DubinsWord::LSR,
                                           DubinsWord::RSL, DubinsWord::RLR, DubinsWord::LRL};

}  // namespace

std::string to_string(DubinsWord w) {
    constexpr const char* names[] = {"LSL", "RSR", "LSR", "RSL", "RLR", "LRL"};
    return names[static_cast<int>(w)];
}

Pose DubinsPath::at(double s) const {
    s = std::clamp(s, 0.0, length());
    const std::array<int, 3> types = segment_types(word);
    Pose q = start;
    for (int i = 0; i < 3 && s > 0.0; ++i) {
        const double ds = std::min(s, lengths[i]);
        q = unicycle_exact_step(q, Input(1.0, types[i] / radius), ds);  // unit speed: time = arc length
        s -= ds;
    }
    return q;
}

Trajectory DubinsPath::sample(double step) const {
    if (step <= 0.0) throw std::invalid_argument("step must be positive");
    const int n = std::max(1, static_cast<int>(std::ceil(length() / step)));
    Trajectory out(n + 1, 3);
    for (int k = 0; k <= n; ++k) out.row(k) = at(length() * k / n).transpose();
    return out;
}

DubinsPath DubinsPath::truncated(double s) const {
    DubinsPath p = *this;
    for (double& l : p.lengths) {
        l = std::min(l, std::max(s, 0.0));
        s -= l;
    }
    return p;
}

std::optional<DubinsPath> dubins_path(const Pose& q0, const Pose& q1, double radius, DubinsWord word) {
    if (radius <= 0.0) throw std::invalid_argument("turning radius must be positive");
    // eq. (6.2): move q0 to the origin, rotate so that q1 lies on the x axis, scale to unit radius
    const double dx = q1[0] - q0[0], dy = q1[1] - q0[1];
    const double d = std::hypot(dx, dy) / radius;
    const double phi = mod2pi(std::atan2(dy, dx));
    const double a = mod2pi(q0[2] - phi), b = mod2pi(q1[2] - phi);
    const auto L = word_lengths(word, d, a, b);
    if (!L) return std::nullopt;
    DubinsPath p;
    p.start = q0;
    p.word = word;
    p.radius = radius;
    for (int i = 0; i < 3; ++i) p.lengths[i] = (*L)[i] * radius;
    return p;
}

std::vector<DubinsPath> all_dubins_paths(const Pose& q0, const Pose& q1, double radius) {
    std::vector<DubinsPath> out;
    for (DubinsWord w : kWords)
        if (auto p = dubins_path(q0, q1, radius, w)) out.push_back(*p);
    return out;
}

DubinsPath shortest_dubins_path(const Pose& q0, const Pose& q1, double radius) {
    const std::vector<DubinsPath> all = all_dubins_paths(q0, q1, radius);
    if (all.empty())
        throw std::logic_error("no Dubins word is feasible");  // cannot happen: CSC always exists
    // exact ties are common (e.g. LSL and RSR when both loop around); break them by word order, not round-off
    const DubinsPath* best = &all.front();
    for (const DubinsPath& p : all)
        if (p.length() < best->length() - 1e-9) best = &p;
    return *best;
}

double dubins_distance(const Pose& q0, const Pose& q1, double radius) {
    return shortest_dubins_path(q0, q1, radius).length();
}

bool dubins_path_valid(const DubinsPath& path, const PlanningProblem2D& problem, double step) {
    const Trajectory pts = path.sample(step);  // eq. (6.9): chords of length <= step
    for (Eigen::Index k = 0; k < pts.rows(); ++k) {
        const Eigen::Vector2d p = pts.row(k).head<2>().transpose();
        if (!problem.state_valid(p)) return false;
        if (k > 0 && !problem.motion_valid(pts.row(k - 1).head<2>().transpose(), p)) return false;
    }
    return true;
}

DubinsTreeResult dubins_rrt_star(const PlanningProblem2D& problem, const Pose& start, const Pose& goal,
                                 const DubinsPlannerOptions& opt) {
    if (!problem.state_valid || !problem.motion_valid)
        throw std::invalid_argument("problem needs validity checks");
    if (!problem.state_valid(start.head<2>()) || !problem.state_valid(goal.head<2>()))
        throw std::invalid_argument("start or goal position is invalid");
    const double r = opt.radius;
    const Eigen::Vector2d size = problem.upper - problem.lower;
    // eq. (6.10): the radius rule of eq. (5.3) with d = 3 and the volume of the (x, y, theta) box
    const double gamma = opt.gamma > 0.0 ? opt.gamma
                                         : 2.0 * std::cbrt(4.0 / 3.0) *
                                               std::cbrt(size.x() * size.y() * 2.0 * kPi / (4.0 * kPi / 3.0));
    std::mt19937 rng(opt.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    DubinsTreeResult t;
    t.nodes = {start};
    t.parent = {-1};
    t.cost = {0.0};
    std::vector<DubinsPath> edge(1);  // edge[i]: path from parent(i) to i
    std::vector<std::vector<int>> children(1);
    std::vector<std::pair<int, DubinsPath>> goal_links;

    auto valid = [&](const DubinsPath& p) { return dubins_path_valid(p, problem, opt.collision_step); };
    auto best_goal = [&]() {
        int best = -1;
        double c = kInf;
        for (int k = 0; k < static_cast<int>(goal_links.size()); ++k) {
            const double ck = t.cost[goal_links[k].first] + goal_links[k].second.length();
            if (ck < c) {
                c = ck;
                best = k;
            }
        }
        return std::make_pair(best, c);
    };

    for (int it = 0; it < opt.max_iterations; ++it) {
        Pose q_rand = goal;
        if (u(rng) >= opt.goal_bias)
            q_rand = Pose(problem.lower.x() + size.x() * u(rng), problem.lower.y() + size.y() * u(rng),
                          -kPi + 2.0 * kPi * u(rng));
        if (problem.state_valid(q_rand.head<2>())) {
            // nearest in the Dubins sense (eq. 6.1 makes it asymmetric: distance *from* the tree)
            int i_near = 0;
            double d_near = kInf;
            for (int i = 0; i < static_cast<int>(t.nodes.size()); ++i) {
                const double d = dubins_distance(t.nodes[i], q_rand, r);
                if (d < d_near) {
                    d_near = d;
                    i_near = i;
                }
            }
            DubinsPath path = shortest_dubins_path(t.nodes[i_near], q_rand, r);
            if (path.length() > opt.step) path = path.truncated(opt.step);
            const Pose q_new = path.at(path.length());
            if (path.length() > 1e-9 && valid(path)) {
                int parent = i_near;
                double c_new = t.cost[i_near] + path.length();
                std::vector<int> near;
                if (opt.star) {
                    const int n = static_cast<int>(t.nodes.size()) + 1;
                    const double rad = std::min(gamma * std::cbrt(std::log(double(n)) / n), opt.step);
                    for (int i = 0; i < static_cast<int>(t.nodes.size()); ++i)  // Dubins length >= Euclidean
                        if ((t.nodes[i].head<2>() - q_new.head<2>()).norm() <= rad) near.push_back(i);
                    for (int i : near) {  // choose parent, eq. (5.5)
                        const DubinsPath p = shortest_dubins_path(t.nodes[i], q_new, r);
                        if (t.cost[i] + p.length() < c_new - 1e-12 && valid(p)) {
                            c_new = t.cost[i] + p.length();
                            parent = i;
                            path = p;
                        }
                    }
                }
                const int k = static_cast<int>(t.nodes.size());
                t.nodes.push_back(q_new);
                t.parent.push_back(parent);
                t.cost.push_back(c_new);
                edge.push_back(path);
                children.emplace_back();
                children[parent].push_back(k);
                if (opt.star) {
                    for (int i : near) {  // rewire, eq. (5.6)
                        if (i == parent || i == 0) continue;
                        const DubinsPath p = shortest_dubins_path(q_new, t.nodes[i], r);
                        if (c_new + p.length() < t.cost[i] - 1e-9 && valid(p)) {
                            auto& sib = children[t.parent[i]];
                            sib.erase(std::find(sib.begin(), sib.end(), i));
                            t.parent[i] = k;
                            children[k].push_back(i);
                            edge[i] = p;
                            std::vector<int> stack{i};
                            const double delta = c_new + p.length() - t.cost[i];
                            while (!stack.empty()) {
                                const int v = stack.back();
                                stack.pop_back();
                                t.cost[v] += delta;
                                for (int c : children[v]) stack.push_back(c);
                            }
                        }
                    }
                }
                if ((q_new.head<2>() - goal.head<2>()).norm() <= opt.step) {
                    const DubinsPath to_goal = shortest_dubins_path(q_new, goal, r);
                    if (to_goal.length() <= opt.step && valid(to_goal)) goal_links.emplace_back(k, to_goal);
                }
            }
        }
        t.best_cost_history.push_back(best_goal().second);
        if (!opt.star && !goal_links.empty()) break;
    }

    const auto [best, cost] = best_goal();
    if (best >= 0) {
        t.found = true;
        t.path_cost = cost;
        for (int v = goal_links[best].first; v > 0; v = t.parent[v]) t.segments.push_back(edge[v]);
        std::reverse(t.segments.begin(), t.segments.end());
        t.segments.push_back(goal_links[best].second);
    }
    return t;
}

}  // namespace motion_planning
