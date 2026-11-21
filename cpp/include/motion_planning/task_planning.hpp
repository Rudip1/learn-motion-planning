#pragma once

/// Chapter 10 — task planning: a STRIPS subset of PDDL, grounding, and forward state-space search with relaxed
/// heuristics. Theory: 1_theory/10_task_planning.md. Equation numbers below refer to that file.
///
/// Supported PDDL: requirements :strips, :typing, :negative-preconditions; typed objects and constants;
/// preconditions and goals that are conjunctions of literals (atoms or (not atom)); effects that are conjunctions
/// of atoms and (not atom). No disjunctions, quantifiers, conditional effects, equality or numbers.

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace motion_planning::task {

// ---------------------------------------------------------------------------------------------------- lifted

/// An atom with variables or objects as arguments, e.g. (on ?x ?y) or (at robot kitchen).
struct Atom {
    std::string predicate;
    std::vector<std::string> args;
};

/// An atom or its negation.
struct Literal {
    Atom atom;
    bool positive = true;
};

/// A parameter of an action schema: name (with the leading '?') and type.
struct Parameter {
    std::string name;
    std::string type = "object";
};

/// An action schema (operator). Section 10.2.
struct ActionSchema {
    std::string name;
    std::vector<Parameter> parameters;
    std::vector<Literal> precondition;
    std::vector<Atom> add;  ///< positive effects
    std::vector<Atom> del;  ///< negative effects
};

struct Domain {
    std::string name;
    std::map<std::string, std::string> types;  ///< type -> parent type ("object" is the root)
    std::map<std::string, std::string> constants;  ///< object -> type
    std::map<std::string, int> predicates;  ///< predicate -> arity
    std::vector<ActionSchema> actions;
};

struct Problem {
    std::string name;
    std::string domain;
    std::map<std::string, std::string> objects;  ///< object -> type
    std::vector<Atom> init;  ///< ground atoms true initially (closed world: all others are false)
    std::vector<Literal> goal;
};

/// Parse a PDDL domain or problem from text. Throws std::runtime_error with a message on unsupported or
/// malformed input.
Domain parse_domain(const std::string& text);
Problem parse_problem(const std::string& text);

/// Read a whole file (helper for the parsers).
std::string read_file(const std::string& path);

/// True if @p type is @p ancestor or one of its subtypes.
bool is_subtype(const Domain& domain, const std::string& type, const std::string& ancestor);

// ---------------------------------------------------------------------------------------------------- ground

/// A ground action: fact indices of its precondition (positive and negative), add and delete lists. Eq. (10.2).
struct GroundAction {
    std::string name;  ///< e.g. "(stack a b)"
    std::vector<int> pre, pre_neg, add, del;
    double cost = 1.0;
};

/// A set of facts as a bitset over the fact indices of a GroundTask.
class State {
  public:
    State() = default;
    explicit State(int num_facts) : bits_((num_facts + 63) / 64, 0) {}
    bool test(int f) const { return (bits_[size_t(f) / 64] >> (f % 64)) & 1u; }
    void set(int f) { bits_[size_t(f) / 64] |= std::uint64_t{1} << (f % 64); }
    void reset(int f) { bits_[size_t(f) / 64] &= ~(std::uint64_t{1} << (f % 64)); }
    int count() const;
    bool operator==(const State& o) const { return bits_ == o.bits_; }
    std::size_t hash() const;

  private:
    std::vector<std::uint64_t> bits_;
};

/// The grounded planning task: facts, actions, initial state and goal. Section 10.3.
struct GroundTask {
    std::vector<std::string> facts;  ///< e.g. "(on a b)"; index = fact id
    std::vector<GroundAction> actions;
    State init;
    std::vector<int> goal_pos, goal_neg;

    int fact_index(const std::string& fact) const;  ///< -1 if the fact does not exist
    std::string describe(const State& s) const;     ///< the true facts, sorted, space separated
};

/// Ground a problem: one action per type-correct binding of the parameters. Facts of static predicates (never
/// added or deleted by any action) are evaluated against the initial state and do not become state variables;
/// actions whose static preconditions fail are dropped. Section 10.3.
GroundTask ground(const Domain& domain, const Problem& problem);

/// s satisfies the precondition of a (positive facts true, negative facts false). Eq. (10.3).
bool applicable(const GroundAction& a, const State& s);

/// Progression: (s \ del(a)) ∪ add(a). Eq. (10.4).
State apply(const GroundAction& a, const State& s);

bool is_goal(const GroundTask& task, const State& s);

/// Check a plan independently of the search: every action applicable in turn and the goal reached at the end.
/// Returns an empty string if valid, else the reason.
std::string validate_plan(const GroundTask& task, const std::vector<int>& plan);

// ---------------------------------------------------------------------------------------------------- search

enum class Heuristic {
    Zero,       ///< h = 0 (A* becomes uniform-cost search)
    GoalCount,  ///< number of unsatisfied goal literals; not admissible in general. Eq. (10.7)
    HMax,       ///< max over goal facts of the relaxed cost; admissible. Eq. (10.9)
    HAdd,       ///< sum over goal facts of the relaxed cost; informative, not admissible. Eq. (10.10)
};

/// Relaxed-planning heuristic value of state s (infinity if the goal is unreachable even when deletes are
/// ignored). Negative preconditions and negative goals are ignored by the relaxation. Eqs. (10.8)-(10.10).
double heuristic(const GroundTask& task, const State& s, Heuristic h);

enum class SearchAlgorithm {
    BreadthFirst,  ///< optimal in the number of actions. Section 10.4
    AStar,         ///< f = g + w h; optimal for w = 1 and an admissible h. Eq. (10.5)
    GreedyBestFirst,  ///< f = h. Eq. (10.6)
};

struct SearchConfig {
    SearchAlgorithm algorithm = SearchAlgorithm::AStar;
    Heuristic heuristic = Heuristic::HMax;
    double weight = 1.0;            ///< w of weighted A*
    long max_expansions = 2000000;  ///< give up after this many expansions
};

struct SearchResult {
    bool solved = false;
    std::vector<int> plan;  ///< indices into GroundTask::actions
    double cost = 0.0;
    long expanded = 0;   ///< states taken from the open list and expanded
    long generated = 0;  ///< successor states created
    double h_init = 0.0; ///< heuristic value of the initial state
};

/// Forward search from the initial state. Duplicate states are detected (graph search).
SearchResult plan(const GroundTask& task, const SearchConfig& config = {});

/// Names of the actions of a plan, e.g. {"(pick ball1 rooma left)", ...}.
std::vector<std::string> plan_names(const GroundTask& task, const std::vector<int>& plan);

/// The states visited by a plan, starting with the initial state.
std::vector<State> plan_states(const GroundTask& task, const std::vector<int>& plan);

}  // namespace motion_planning::task
