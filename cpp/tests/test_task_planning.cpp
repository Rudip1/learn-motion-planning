// Chapter 10 tests: the PDDL parser and grounding on hand-counted examples, progression, the known optimal plan
// lengths of the Sussman anomaly and of Gripper (6 n/2 - 1 actions for n balls), admissibility of h_max against
// the optimal cost, independent plan validation, and unsolvable problems.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <string>

#include "motion_planning/task_planning.hpp"

using namespace motion_planning::task;

namespace {

const std::string kDir = PDDL_DIR;

GroundTask load(const std::string& domain, const std::string& problem) {
    return ground(parse_domain(read_file(kDir + "/" + domain)), parse_problem(read_file(kDir + "/" + problem)));
}

SearchConfig cfg(SearchAlgorithm a, Heuristic h, double w = 1.0) {
    SearchConfig c;
    c.algorithm = a;
    c.heuristic = h;
    c.weight = w;
    return c;
}

}  // namespace

TEST_CASE("parser reads types, constants, predicates and actions", "[task]") {
    const Domain d = parse_domain(read_file(kDir + "/pick_place_domain.pddl"));
    CHECK(d.name == "pick-place");
    CHECK(d.types.size() == 3);
    CHECK(d.predicates.at("stand-at") == 2);
    REQUIRE(d.actions.size() == 3);
    const ActionSchema& place = d.actions[2];
    CHECK(place.name == "place");
    REQUIRE(place.parameters.size() == 3);
    CHECK(place.parameters[1].type == "stand");
    CHECK(place.precondition.size() == 4);
    CHECK_FALSE(place.precondition[3].positive);  // (not (occupied ?s))
    CHECK(place.add.size() == 3);
    CHECK(place.del.size() == 1);
    // Case-insensitive, comments ignored, unsupported features rejected with a message.
    CHECK(parse_domain("(DEFINE (DOMAIN x) ; comment\n (:PREDICATES (P)))").predicates.count("p") == 1);
    CHECK_THROWS(parse_domain("(define (domain x) (:requirements :adl))"));
    CHECK_THROWS(parse_domain("(define (domain x) (:action a :parameters () :precondition (or (p) (q)) :effect (p)))"));
}

TEST_CASE("grounding keeps only the type-correct actions (statics pruned)", "[task]") {
    // Gripper, n balls: move 2x2 (from = to is allowed by the domain), pick and drop n balls x 2 rooms x 2
    // grippers each. Facts: at-robby 2, at 2n, free 2, carry 2n.
    for (int n : {2, 4}) {
        const GroundTask t = load("gripper_domain.pddl", "gripper_n" + std::to_string(n) + ".pddl");
        CHECK(t.actions.size() == size_t(4 + 2 * 4 * n));
        CHECK(t.facts.size() == size_t(4 + 4 * n));
        CHECK(t.init.count() == 1 + n + 2);
        CHECK(t.goal_pos.size() == size_t(n));
    }
    // Blocksworld, 3 blocks: pick-up/put-down 3 each, stack/unstack 3 x 3 each (x = y is not excluded).
    const GroundTask b = load("blocksworld_domain.pddl", "blocksworld_sussman.pddl");
    CHECK(b.actions.size() == size_t(3 + 3 + 9 + 9));
}

TEST_CASE("progression applies deletes, then adds", "[task]") {
    // Eq. (10.4) on the Sussman start: unstack C from A.
    const GroundTask t = load("blocksworld_domain.pddl", "blocksworld_sussman.pddl");
    int unstack_ca = -1;
    for (size_t i = 0; i < t.actions.size(); ++i)
        if (t.actions[i].name == "(unstack c a)") unstack_ca = int(i);
    REQUIRE(unstack_ca >= 0);
    REQUIRE(applicable(t.actions[size_t(unstack_ca)], t.init));
    const State s = apply(t.actions[size_t(unstack_ca)], t.init);
    CHECK(t.describe(s) == "(clear a) (clear b) (holding c) (ontable a) (ontable b)");
}

TEST_CASE("breadth-first search finds the known optimal plans", "[task]") {
    const GroundTask sussman = load("blocksworld_domain.pddl", "blocksworld_sussman.pddl");
    const SearchResult r = plan(sussman, cfg(SearchAlgorithm::BreadthFirst, Heuristic::Zero));
    REQUIRE(r.solved);
    CHECK(r.plan.size() == 6);
    CHECK(validate_plan(sussman, r.plan).empty());
    for (int n : {2, 4, 6}) {
        const GroundTask g = load("gripper_domain.pddl", "gripper_n" + std::to_string(n) + ".pddl");
        const SearchResult rg = plan(g, cfg(SearchAlgorithm::BreadthFirst, Heuristic::Zero));
        REQUIRE(rg.solved);
        CHECK(rg.plan.size() == size_t(6 * n / 2 - 1));
    }
}

TEST_CASE("A* with h_max is optimal and h_max never overestimates", "[task]") {
    for (const auto& [dom, prob] : std::vector<std::pair<std::string, std::string>>{
             {"blocksworld_domain.pddl", "blocksworld_sussman.pddl"},
             {"blocksworld_domain.pddl", "blocksworld_reverse5.pddl"},
             {"gripper_domain.pddl", "gripper_n4.pddl"},
             {"pick_place_domain.pddl", "pick_place_swap.pddl"}}) {
        const GroundTask t = load(dom, prob);
        const SearchResult bfs = plan(t, cfg(SearchAlgorithm::BreadthFirst, Heuristic::Zero));
        const SearchResult astar = plan(t, cfg(SearchAlgorithm::AStar, Heuristic::HMax));
        REQUIRE(bfs.solved);
        REQUIRE(astar.solved);
        CHECK(astar.cost == bfs.cost);
        CHECK(validate_plan(t, astar.plan).empty());
        // Admissibility (eq. 10.11) along the optimal plan: h_max(s_k) <= remaining cost.
        const auto states = plan_states(t, bfs.plan);
        for (size_t k = 0; k < states.size(); ++k)
            CHECK(heuristic(t, states[k], Heuristic::HMax) <= double(bfs.plan.size() - k));
        CHECK(astar.expanded <= bfs.expanded);
    }
}

TEST_CASE("h_add guides greedy search to valid plans with fewer expansions", "[task]") {
    const GroundTask t = load("gripper_domain.pddl", "gripper_n8.pddl");
    const SearchResult gbfs = plan(t, cfg(SearchAlgorithm::GreedyBestFirst, Heuristic::HAdd));
    const SearchResult astar = plan(t, cfg(SearchAlgorithm::AStar, Heuristic::HMax));
    REQUIRE(gbfs.solved);
    REQUIRE(astar.solved);
    CHECK(validate_plan(t, gbfs.plan).empty());
    CHECK(gbfs.expanded < astar.expanded / 10);
    CHECK(gbfs.plan.size() >= astar.plan.size());  // no optimality guarantee
    // h_add overestimates on Gripper: at the start every ball "needs" its own pick, move, drop.
    CHECK(heuristic(t, t.init, Heuristic::HAdd) > double(astar.plan.size()));
}

TEST_CASE("plan validation rejects broken plans", "[task]") {
    const GroundTask t = load("blocksworld_domain.pddl", "blocksworld_sussman.pddl");
    SearchResult r = plan(t, cfg(SearchAlgorithm::BreadthFirst, Heuristic::Zero));
    REQUIRE(r.solved);
    std::vector<int> swapped = r.plan;
    std::swap(swapped[0], swapped[1]);
    CHECK_FALSE(validate_plan(t, swapped).empty());
    std::vector<int> short_plan(r.plan.begin(), r.plan.end() - 1);
    CHECK(validate_plan(t, short_plan) == "the goal is not reached");
}

TEST_CASE("unsolvable problems are recognised", "[task]") {
    const Domain d = parse_domain(read_file(kDir + "/gripper_domain.pddl"));
    // A ball that is not a ball (static type fact missing) can never be picked: h_max is infinite at the start.
    Problem p = parse_problem(read_file(kDir + "/gripper_n2.pddl"));
    p.init.erase(std::remove_if(p.init.begin(), p.init.end(),
                                [](const Atom& a) { return a.predicate == "ball" && a.args[0] == "ball2"; }),
                 p.init.end());
    const GroundTask t = ground(d, p);
    CHECK(std::isinf(heuristic(t, t.init, Heuristic::HMax)));
    CHECK_FALSE(plan(t, cfg(SearchAlgorithm::AStar, Heuristic::HMax)).solved);
    const SearchResult bfs = plan(t, cfg(SearchAlgorithm::BreadthFirst, Heuristic::Zero));
    CHECK_FALSE(bfs.solved);
    CHECK(bfs.expanded > 0);  // blind search has to exhaust the reachable states to find out
}
