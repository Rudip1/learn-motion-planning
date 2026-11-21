#include "motion_planning/task_planning.hpp"

#include <algorithm>
#include <cctype>
#include <deque>
#include <fstream>
#include <functional>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace motion_planning::task {

// ================================================================================================ parsing

namespace {

/// An S-expression: a symbol or a list.
struct SExpr {
    std::string symbol;
    std::vector<SExpr> list;
    bool is_list = false;
};

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    auto flush = [&] {
        if (!cur.empty()) tokens.push_back(cur), cur.clear();
    };
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == ';') {  // comment to end of line
            flush();
            while (i < text.size() && text[i] != '\n') ++i;
        } else if (c == '(' || c == ')') {
            flush();
            tokens.emplace_back(1, c);
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            flush();
        } else {
            cur += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    flush();
    return tokens;
}

SExpr parse_sexpr(const std::vector<std::string>& tokens, size_t& pos) {
    if (pos >= tokens.size()) throw std::runtime_error("PDDL: unexpected end of input");
    SExpr e;
    if (tokens[pos] == "(") {
        e.is_list = true;
        ++pos;
        while (pos < tokens.size() && tokens[pos] != ")") e.list.push_back(parse_sexpr(tokens, pos));
        if (pos >= tokens.size()) throw std::runtime_error("PDDL: missing ')'");
        ++pos;
    } else if (tokens[pos] == ")") {
        throw std::runtime_error("PDDL: unexpected ')'");
    } else {
        e.symbol = tokens[pos++];
    }
    return e;
}

SExpr parse_text(const std::string& text) {
    const auto tokens = tokenize(text);
    size_t pos = 0;
    SExpr e = parse_sexpr(tokens, pos);
    if (pos != tokens.size()) throw std::runtime_error("PDDL: text after the closing ')'");
    if (!e.is_list || e.list.empty() || e.list[0].symbol != "define")
        throw std::runtime_error("PDDL: expected (define ...)");
    return e;
}

const std::string& sym(const SExpr& e, const char* what) {
    if (e.is_list) throw std::runtime_error(std::string("PDDL: expected a name for ") + what);
    return e.symbol;
}

/// "a b - t c - u d" -> {(a,t),(b,t),(c,u),(d,object)}, in order.
std::vector<std::pair<std::string, std::string>> typed_list(const std::vector<SExpr>& items, size_t begin) {
    std::vector<std::pair<std::string, std::string>> out;
    std::vector<std::string> pending;
    for (size_t i = begin; i < items.size(); ++i) {
        const std::string& s = sym(items[i], "a typed list");
        if (s == "-") {
            if (i + 1 >= items.size()) throw std::runtime_error("PDDL: '-' without a type");
            const std::string& type = sym(items[++i], "a type");
            for (auto& p : pending) out.emplace_back(p, type);
            pending.clear();
        } else {
            pending.push_back(s);
        }
    }
    for (auto& p : pending) out.emplace_back(p, "object");
    return out;
}

Atom parse_atom(const SExpr& e) {
    if (!e.is_list || e.list.empty()) throw std::runtime_error("PDDL: expected an atom (predicate args...)");
    Atom a;
    a.predicate = sym(e.list[0], "a predicate");
    for (size_t i = 1; i < e.list.size(); ++i) a.args.push_back(sym(e.list[i], "an argument"));
    return a;
}

Literal parse_literal(const SExpr& e) {
    if (e.is_list && !e.list.empty() && !e.list[0].is_list && e.list[0].symbol == "not") {
        if (e.list.size() != 2) throw std::runtime_error("PDDL: (not ...) takes one atom");
        return {parse_atom(e.list[1]), false};
    }
    const std::string head = e.is_list && !e.list.empty() && !e.list[0].is_list ? e.list[0].symbol : "";
    if (head == "or" || head == "imply" || head == "forall" || head == "exists" || head == "when" || head == "=")
        throw std::runtime_error("PDDL: '" + head + "' is not supported (STRIPS subset only)");
    return {parse_atom(e), true};
}

/// (and l1 l2 ...), a single literal, or () -> list of literals.
std::vector<Literal> parse_conjunction(const SExpr& e) {
    if (e.is_list && e.list.empty()) return {};
    if (e.is_list && !e.list[0].is_list && e.list[0].symbol == "and") {
        std::vector<Literal> out;
        for (size_t i = 1; i < e.list.size(); ++i) out.push_back(parse_literal(e.list[i]));
        return out;
    }
    return {parse_literal(e)};
}

}  // namespace

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool is_subtype(const Domain& domain, const std::string& type, const std::string& ancestor) {
    std::string t = type;
    for (int guard = 0; guard < 100; ++guard) {
        if (t == ancestor) return true;
        if (t == "object") return false;
        const auto it = domain.types.find(t);
        if (it == domain.types.end()) return false;
        t = it->second;
    }
    throw std::runtime_error("PDDL: cyclic type hierarchy");
}

Domain parse_domain(const std::string& text) {
    const SExpr root = parse_text(text);
    Domain d;
    for (size_t i = 1; i < root.list.size(); ++i) {
        const SExpr& sec = root.list[i];
        if (!sec.is_list || sec.list.empty()) throw std::runtime_error("PDDL: malformed domain section");
        const std::string& key = sym(sec.list[0], "a section");
        if (key == "domain") {
            d.name = sym(sec.list.at(1), "the domain name");
        } else if (key == ":requirements") {
            for (size_t k = 1; k < sec.list.size(); ++k) {
                const std::string& r = sym(sec.list[k], "a requirement");
                if (r != ":strips" && r != ":typing" && r != ":negative-preconditions")
                    throw std::runtime_error("PDDL: requirement " + r + " is not supported");
            }
        } else if (key == ":types") {
            for (auto& [t, parent] : typed_list(sec.list, 1)) d.types[t] = parent;
        } else if (key == ":constants") {
            for (auto& [o, t] : typed_list(sec.list, 1)) d.constants[o] = t;
        } else if (key == ":predicates") {
            for (size_t k = 1; k < sec.list.size(); ++k) {
                const SExpr& p = sec.list[k];
                if (!p.is_list || p.list.empty()) throw std::runtime_error("PDDL: malformed predicate");
                d.predicates[sym(p.list[0], "a predicate")] = static_cast<int>(typed_list(p.list, 1).size());
            }
        } else if (key == ":action") {
            ActionSchema a;
            a.name = sym(sec.list.at(1), "the action name");
            for (size_t k = 2; k + 1 < sec.list.size(); k += 2) {
                const std::string& field = sym(sec.list[k], "an action field");
                const SExpr& value = sec.list[k + 1];
                if (field == ":parameters") {
                    for (auto& [n, t] : typed_list(value.list, 0)) a.parameters.push_back({n, t});
                } else if (field == ":precondition") {
                    a.precondition = parse_conjunction(value);
                } else if (field == ":effect") {
                    for (const Literal& l : parse_conjunction(value)) (l.positive ? a.add : a.del).push_back(l.atom);
                } else {
                    throw std::runtime_error("PDDL: action field " + field + " is not supported");
                }
            }
            d.actions.push_back(std::move(a));
        } else {
            throw std::runtime_error("PDDL: domain section " + key + " is not supported");
        }
    }
    return d;
}

Problem parse_problem(const std::string& text) {
    const SExpr root = parse_text(text);
    Problem p;
    for (size_t i = 1; i < root.list.size(); ++i) {
        const SExpr& sec = root.list[i];
        if (!sec.is_list || sec.list.empty()) throw std::runtime_error("PDDL: malformed problem section");
        const std::string& key = sym(sec.list[0], "a section");
        if (key == "problem") {
            p.name = sym(sec.list.at(1), "the problem name");
        } else if (key == ":domain") {
            p.domain = sym(sec.list.at(1), "the domain name");
        } else if (key == ":objects") {
            for (auto& [o, t] : typed_list(sec.list, 1)) p.objects[o] = t;
        } else if (key == ":init") {
            for (size_t k = 1; k < sec.list.size(); ++k) p.init.push_back(parse_atom(sec.list[k]));
        } else if (key == ":goal") {
            p.goal = parse_conjunction(sec.list.at(1));
        } else {
            throw std::runtime_error("PDDL: problem section " + key + " is not supported");
        }
    }
    return p;
}

// ================================================================================================ grounding

int State::count() const {
    int n = 0;
    for (auto w : bits_) n += __builtin_popcountll(w);
    return n;
}

std::size_t State::hash() const {
    std::size_t h = 1469598103934665603ull;
    for (auto w : bits_) h = (h ^ std::hash<std::uint64_t>{}(w)) * 1099511628211ull;
    return h;
}

int GroundTask::fact_index(const std::string& fact) const {
    const auto it = std::find(facts.begin(), facts.end(), fact);
    return it == facts.end() ? -1 : static_cast<int>(it - facts.begin());
}

std::string GroundTask::describe(const State& s) const {
    std::vector<std::string> on;
    for (int f = 0; f < static_cast<int>(facts.size()); ++f)
        if (s.test(f)) on.push_back(facts[size_t(f)]);
    std::sort(on.begin(), on.end());
    std::string out;
    for (const auto& f : on) out += (out.empty() ? "" : " ") + f;
    return out;
}

namespace {

std::string fact_string(const std::string& predicate, const std::vector<std::string>& args) {
    std::string s = "(" + predicate;
    for (const auto& a : args) s += " " + a;
    return s + ")";
}

}  // namespace

GroundTask ground(const Domain& domain, const Problem& problem) {
    // Objects with their types.
    std::map<std::string, std::string> objects = domain.constants;
    for (const auto& [o, t] : problem.objects) objects[o] = t;
    // Static predicates: never in an effect.
    std::set<std::string> fluent;
    for (const auto& a : domain.actions) {
        for (const auto& at : a.add) fluent.insert(at.predicate);
        for (const auto& at : a.del) fluent.insert(at.predicate);
    }
    std::set<std::string> init_facts;
    for (const auto& at : problem.init) init_facts.insert(fact_string(at.predicate, at.args));

    GroundTask task;
    std::unordered_map<std::string, int> index;
    auto intern = [&](const std::string& f) {
        auto [it, inserted] = index.emplace(f, static_cast<int>(task.facts.size()));
        if (inserted) task.facts.push_back(f);
        return it->second;
    };
    // Fluent facts of the initial state and the goal first, so their indices are stable and readable.
    for (const auto& at : problem.init)
        if (fluent.count(at.predicate)) intern(fact_string(at.predicate, at.args));
    bool static_goal_fails = false;
    for (const auto& l : problem.goal) {
        const std::string f = fact_string(l.atom.predicate, l.atom.args);
        if (!fluent.count(l.atom.predicate)) {
            if (init_facts.count(f) != static_cast<size_t>(l.positive)) static_goal_fails = true;
            continue;
        }
        (l.positive ? task.goal_pos : task.goal_neg).push_back(intern(f));
    }

    for (const ActionSchema& schema : domain.actions) {
        const size_t n = schema.parameters.size();
        std::vector<std::vector<std::string>> candidates(n);
        for (size_t i = 0; i < n; ++i)
            for (const auto& [o, t] : objects)
                if (is_subtype(domain, t, schema.parameters[i].type)) candidates[i].push_back(o);
        std::map<std::string, size_t> var_pos;
        for (size_t i = 0; i < n; ++i) var_pos[schema.parameters[i].name] = i;
        // Static literals, each checked as soon as its last variable is bound (pruning).
        std::vector<std::vector<const Literal*>> static_at(n + 1);
        for (const Literal& l : schema.precondition) {
            if (fluent.count(l.atom.predicate)) continue;
            size_t last = 0;
            for (const auto& arg : l.atom.args)
                if (var_pos.count(arg)) last = std::max(last, var_pos[arg] + 1);
            static_at[last].push_back(&l);
        }
        std::vector<std::string> binding(n);
        auto subst = [&](const Atom& at) {
            std::vector<std::string> args;
            for (const auto& arg : at.args) args.push_back(var_pos.count(arg) ? binding[var_pos[arg]] : arg);
            return fact_string(at.predicate, args);
        };
        auto statics_hold = [&](size_t level) {
            for (const Literal* l : static_at[level])
                if (init_facts.count(subst(l->atom)) != static_cast<size_t>(l->positive)) return false;
            return true;
        };
        std::function<void(size_t)> rec = [&](size_t depth) {
            if (depth == n) {
                GroundAction ga;
                ga.name = "(" + schema.name;
                for (const auto& b : binding) ga.name += " " + b;
                ga.name += ")";
                for (const Literal& l : schema.precondition) {
                    if (!fluent.count(l.atom.predicate)) continue;
                    (l.positive ? ga.pre : ga.pre_neg).push_back(intern(subst(l.atom)));
                }
                for (const Atom& at : schema.add) ga.add.push_back(intern(subst(at)));
                for (const Atom& at : schema.del) ga.del.push_back(intern(subst(at)));
                task.actions.push_back(std::move(ga));
                return;
            }
            for (const auto& o : candidates[depth]) {
                binding[depth] = o;
                if (statics_hold(depth + 1)) rec(depth + 1);
            }
        };
        if (statics_hold(0)) rec(0);
    }
    if (static_goal_fails) task.goal_pos.push_back(intern("(static-goal-unsatisfiable)"));
    task.init = State(static_cast<int>(task.facts.size()));
    for (const auto& at : problem.init) {
        const auto it = index.find(fact_string(at.predicate, at.args));
        if (it != index.end()) task.init.set(it->second);
    }
    return task;
}

bool applicable(const GroundAction& a, const State& s) {
    for (int f : a.pre)
        if (!s.test(f)) return false;
    for (int f : a.pre_neg)
        if (s.test(f)) return false;
    return true;  // eq. (10.3)
}

State apply(const GroundAction& a, const State& s) {
    State out = s;
    for (int f : a.del) out.reset(f);
    for (int f : a.add) out.set(f);  // eq. (10.4): adds win over deletes
    return out;
}

bool is_goal(const GroundTask& task, const State& s) {
    for (int f : task.goal_pos)
        if (!s.test(f)) return false;
    for (int f : task.goal_neg)
        if (s.test(f)) return false;
    return true;
}

std::string validate_plan(const GroundTask& task, const std::vector<int>& plan) {
    State s = task.init;
    for (size_t i = 0; i < plan.size(); ++i) {
        if (plan[i] < 0 || plan[i] >= static_cast<int>(task.actions.size()))
            return "step " + std::to_string(i) + ": no such action";
        const GroundAction& a = task.actions[size_t(plan[i])];
        if (!applicable(a, s)) return "step " + std::to_string(i) + ": " + a.name + " is not applicable";
        s = apply(a, s);
    }
    return is_goal(task, s) ? "" : "the goal is not reached";
}

// ================================================================================================ heuristics

double heuristic(const GroundTask& task, const State& s, Heuristic h) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    if (h == Heuristic::Zero) return 0.0;
    if (h == Heuristic::GoalCount) {  // eq. (10.7)
        int n = 0;
        for (int f : task.goal_pos) n += !s.test(f);
        for (int f : task.goal_neg) n += s.test(f);
        return n;
    }
    // Relaxed costs by fixpoint iteration (eq. 10.8): cost(f) = 0 if f in s, else the cheapest achiever's
    // cost(a) + combine(cost of its preconditions), combine = max (h_max) or sum (h_add).
    const bool use_max = h == Heuristic::HMax;
    std::vector<double> cost(task.facts.size(), inf);
    for (int f = 0; f < static_cast<int>(task.facts.size()); ++f)
        if (s.test(f)) cost[size_t(f)] = 0.0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (const GroundAction& a : task.actions) {
            double c = 0.0;
            for (int f : a.pre) {
                const double cf = cost[size_t(f)];
                if (cf == inf) {
                    c = inf;
                    break;
                }
                c = use_max ? std::max(c, cf) : c + cf;
            }
            if (c == inf) continue;
            c += a.cost;
            for (int f : a.add)
                if (c < cost[size_t(f)]) cost[size_t(f)] = c, changed = true;
        }
    }
    double value = 0.0;  // eqs. (10.9), (10.10)
    for (int f : task.goal_pos) {
        const double cf = cost[size_t(f)];
        if (cf == inf) return inf;
        value = use_max ? std::max(value, cf) : value + cf;
    }
    return value;
}

// ================================================================================================ search

namespace {

struct StateHash {
    std::size_t operator()(const State& s) const { return s.hash(); }
};

struct SearchNode {
    State state;
    int parent = -1;
    int action = -1;
    double g = 0.0;
    double h = 0.0;
};

std::vector<int> extract(const std::vector<SearchNode>& nodes, int n) {
    std::vector<int> plan;
    for (; nodes[size_t(n)].parent >= 0; n = nodes[size_t(n)].parent) plan.push_back(nodes[size_t(n)].action);
    std::reverse(plan.begin(), plan.end());
    return plan;
}

}  // namespace

SearchResult plan(const GroundTask& task, const SearchConfig& config) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    SearchResult result;
    std::vector<SearchNode> nodes;
    std::unordered_map<State, int, StateHash> seen;
    nodes.push_back({task.init, -1, -1, 0.0, heuristic(task, task.init, config.heuristic)});
    seen.emplace(task.init, 0);
    result.h_init = nodes[0].h;
    auto finish = [&](int n) {
        result.solved = true;
        result.plan = extract(nodes, n);
        result.cost = nodes[size_t(n)].g;
        return result;
    };
    if (is_goal(task, task.init)) return finish(0);
    if (result.h_init == inf) return result;  // dead end already in the relaxation

    if (config.algorithm == SearchAlgorithm::BreadthFirst) {
        // Goal test at generation: the first goal state found is at minimal depth.
        std::deque<int> open{0};
        while (!open.empty() && result.expanded < config.max_expansions) {
            const int n = open.front();
            open.pop_front();
            ++result.expanded;
            for (int a = 0; a < static_cast<int>(task.actions.size()); ++a) {
                const GroundAction& act = task.actions[size_t(a)];
                if (!applicable(act, nodes[size_t(n)].state)) continue;
                State next = apply(act, nodes[size_t(n)].state);
                ++result.generated;
                if (seen.count(next)) continue;
                nodes.push_back({next, n, a, nodes[size_t(n)].g + act.cost, 0.0});
                const int id = static_cast<int>(nodes.size()) - 1;
                seen.emplace(std::move(next), id);
                if (is_goal(task, nodes[size_t(id)].state)) return finish(id);
                open.push_back(id);
            }
        }
        return result;
    }

    // A* (f = g + w h, eq. 10.5) or greedy best-first (f = h, eq. 10.6), with re-opening when a cheaper path
    // to a known state is found. Ties are broken by smaller h, then first in.
    using Entry = std::tuple<double, double, long, int>;  // f, h, -order, node
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
    long order = 0;
    auto f_of = [&](const SearchNode& nd) {
        return config.algorithm == SearchAlgorithm::GreedyBestFirst ? nd.h : nd.g + config.weight * nd.h;
    };
    open.emplace(f_of(nodes[0]), nodes[0].h, order++, 0);
    std::vector<double> closed_g;  // g at expansion, per node; inf = not expanded
    while (!open.empty() && result.expanded < config.max_expansions) {
        const auto [f, hval, ord, n] = open.top();
        (void)hval;
        (void)ord;
        open.pop();
        if (f > f_of(nodes[size_t(n)]) + 1e-12) continue;  // stale entry
        if (closed_g.size() < nodes.size()) closed_g.resize(nodes.size(), inf);
        if (closed_g[size_t(n)] <= nodes[size_t(n)].g) continue;  // already expanded with this g
        closed_g[size_t(n)] = nodes[size_t(n)].g;
        if (is_goal(task, nodes[size_t(n)].state)) return finish(n);
        ++result.expanded;
        for (int a = 0; a < static_cast<int>(task.actions.size()); ++a) {
            const GroundAction& act = task.actions[size_t(a)];
            if (!applicable(act, nodes[size_t(n)].state)) continue;
            State next = apply(act, nodes[size_t(n)].state);
            ++result.generated;
            const double g = nodes[size_t(n)].g + act.cost;
            const auto it = seen.find(next);
            if (it != seen.end()) {
                SearchNode& old = nodes[size_t(it->second)];
                if (g >= old.g || config.algorithm == SearchAlgorithm::GreedyBestFirst) continue;
                old.g = g;  // cheaper path: re-open
                old.parent = n;
                old.action = a;
                open.emplace(f_of(old), old.h, order++, it->second);
                continue;
            }
            const double h = heuristic(task, next, config.heuristic);
            if (h == inf) continue;  // relaxed dead end: the goal cannot be reached from here
            nodes.push_back({next, n, a, g, h});
            const int id = static_cast<int>(nodes.size()) - 1;
            seen.emplace(std::move(next), id);
            open.emplace(f_of(nodes[size_t(id)]), h, order++, id);
        }
    }
    return result;
}

std::vector<std::string> plan_names(const GroundTask& task, const std::vector<int>& plan) {
    std::vector<std::string> out;
    for (int a : plan) out.push_back(task.actions.at(size_t(a)).name);
    return out;
}

std::vector<State> plan_states(const GroundTask& task, const std::vector<int>& plan) {
    std::vector<State> out{task.init};
    for (int a : plan) out.push_back(apply(task.actions.at(size_t(a)), out.back()));
    return out;
}

}  // namespace motion_planning::task
