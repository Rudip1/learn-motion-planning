// Chapter 9 tests: the tick semantics of every node type against the hand-worked traces of
// 1_theory/09_behavior_trees.md, halting, and a small mission run end to end.
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <deque>

#include "motion_planning/behavior_tree.hpp"

using namespace motion_planning::bt;

namespace {

// A leaf that returns a scripted sequence of statuses (the last one repeats) and counts ticks and halts.
struct Script {
    std::deque<Status> statuses;
    int ticks = 0, halts = 0;
};

NodePtr scripted(const std::string& name, std::shared_ptr<Script> s) {
    return std::make_shared<Action>(
        name,
        [s](Blackboard&) {
            ++s->ticks;
            const Status r = s->statuses.front();
            if (s->statuses.size() > 1) s->statuses.pop_front();
            return r;
        },
        [s]() { ++s->halts; });
}

std::shared_ptr<Script> script(std::initializer_list<Status> l) {
    return std::make_shared<Script>(Script{l});
}

constexpr Status S = Status::Success, F = Status::Failure, R = Status::Running;

}  // namespace

TEST_CASE("sequence and fallback decide on the first child that does not succeed or fail") {
    Blackboard bb;
    auto a = script({S}), b = script({F}), c = script({S});
    Sequence seq("seq", {scripted("a", a), scripted("b", b), scripted("c", c)});
    CHECK(seq.tick(bb) == F);
    CHECK(c->ticks == 0);  // never reached
    auto d = script({F}), e = script({S}), f = script({F});
    Fallback fb("fb", {scripted("d", d), scripted("e", e), scripted("f", f)});
    CHECK(fb.tick(bb) == S);
    CHECK(f->ticks == 0);
    Sequence all("all", {scripted("x", script({S})), scripted("y", script({S}))});
    CHECK(all.tick(bb) == S);
    Fallback none("none", {scripted("x", script({F})), scripted("y", script({F}))});
    CHECK(none.tick(bb) == F);
}

TEST_CASE("worked example: a reactive sequence re-checks its condition and halts the running action") {
    // Sequence(BatteryOk, Patrol): the condition holds for two ticks, then fails; Patrol is always running.
    Blackboard bb;
    auto ok = script({S, S, F}), patrol = script({R});
    Sequence seq("seq", {scripted("BatteryOk", ok), scripted("Patrol", patrol)});
    CHECK(seq.tick(bb) == R);
    CHECK(seq.tick(bb) == R);
    CHECK(seq.tick(bb) == F);
    CHECK(ok->ticks == 3);      // the condition is re-ticked every tick
    CHECK(patrol->ticks == 2);  // the action only while the condition holds
    CHECK(patrol->halts == 1);  // and it is halted when the condition fails
}

TEST_CASE("worked example: a sequence with memory resumes the running child") {
    Blackboard bb;
    auto ok = script({S, F}), patrol = script({R, R, S});
    Sequence seq("seq", {scripted("BatteryOk", ok), scripted("Patrol", patrol)}, /*memory=*/true);
    CHECK(seq.tick(bb) == R);
    CHECK(seq.tick(bb) == R);
    CHECK(seq.tick(bb) == S);
    CHECK(ok->ticks == 1);  // not re-checked while Patrol runs: the failure is never seen
    CHECK(patrol->ticks == 3);
}

TEST_CASE("reactive fallback: a higher-priority child that becomes true preempts the running one") {
    Blackboard bb;
    auto emergency = script({F, F, S}), work = script({R});
    Fallback fb("fb", {scripted("Emergency", emergency), scripted("Work", work)});
    CHECK(fb.tick(bb) == R);
    CHECK(fb.tick(bb) == R);
    CHECK(fb.tick(bb) == S);
    CHECK(work->ticks == 2);
    CHECK(work->halts == 1);
    // with memory the running child keeps the focus
    auto em2 = script({F, S}), work2 = script({R, R, S});
    Fallback fbm("fbm", {scripted("Emergency", em2), scripted("Work", work2)}, true);
    CHECK(fbm.tick(bb) == R);
    CHECK(fbm.tick(bb) == R);
    CHECK(fbm.tick(bb) == S);
    CHECK(em2->ticks == 1);
}

TEST_CASE("parallel thresholds") {
    Blackboard bb;
    auto a = script({S}), b = script({R}), c = script({F});
    Parallel p2("p2", {scripted("a", a), scripted("b", b), scripted("c", c)}, 2);
    CHECK(p2.tick(bb) == R);  // one success, one failure, one running: 2 successes still possible
    Parallel p1("p1", {scripted("a", script({S})), scripted("b", b), scripted("c", script({F}))}, 1);
    CHECK(p1.tick(bb) == S);
    CHECK(b->halts == 1);  // the running child is halted once the parallel has decided
    Parallel p3("p3", {scripted("a", script({S})), scripted("b", script({R})), scripted("c", script({F}))},
                3);
    CHECK(p3.tick(bb) == F);  // one failure makes 3 successes impossible
    CHECK_THROWS(Parallel("bad", {scripted("a", script({S}))}, 2));
}

TEST_CASE("decorators") {
    Blackboard bb;
    CHECK(Inverter("i", scripted("a", script({S}))).tick(bb) == F);
    CHECK(Inverter("i", scripted("a", script({R}))).tick(bb) == R);
    auto flaky = script({F, F, S});
    Retry retry("retry", scripted("flaky", flaky), 3);
    CHECK(retry.tick(bb) == S);  // three attempts within one tick
    CHECK(flaky->ticks == 3);
    auto always_fail = script({F});
    Retry retry2("retry2", scripted("fail", always_fail), 2);
    CHECK(retry2.tick(bb) == F);
    CHECK(always_fail->ticks == 2);
    auto ok = script({S});
    Repeat rep("rep", scripted("ok", ok), 4);
    CHECK(rep.tick(bb) == S);
    CHECK(ok->ticks == 4);
    auto slow = script({R});
    Timeout to("to", scripted("slow", slow), 3);
    CHECK(to.tick(bb) == R);
    CHECK(to.tick(bb) == R);
    CHECK(to.tick(bb) == F);
    CHECK(slow->halts == 1);
    CHECK(Force("fs", scripted("a", script({F})), S).tick(bb) == S);
    CHECK(Force("ff", scripted("a", script({S})), F).tick(bb) == F);
}

TEST_CASE("built-in leaves and the blackboard") {
    Tree t(std::make_shared<Sequence>("root", std::vector<NodePtr>{
                                                  compare("low?", "battery", "<", 20.0),
                                                  set_value("flag", "charging", true),
                                                  wait("charge", 3),
                                              }));
    t.blackboard().set("battery", 10.0);
    CHECK(t.tick() == R);
    CHECK(std::get<bool>(t.blackboard().get("charging")));
    CHECK(t.run(10) == S);
    CHECK(t.ticks() == 3);  // wait(3) needs three ticks
    t.blackboard().set("battery", 50);
    CHECK(t.tick() == F);
    CHECK_THROWS(t.blackboard().get("missing"));
    t.blackboard().set("name", std::string("r1"));
    CHECK_THROWS(t.blackboard().number("name"));
    CHECK(t.to_string().find("Sequence 'root'  [FAILURE]") != std::string::npos);
}

TEST_CASE("worked example: a mission that chatters, and the fix with hysteresis") {
    // Fallback( Sequence(BatteryOk, Patrol), Charge ): Patrol drains 7 % per tick and never ends, Charge adds
    // 30 % per tick and succeeds when full.
    auto make_patrol = []() {
        return std::make_shared<Action>("Patrol", [](Blackboard& bb) {
            bb.set("battery", bb.number("battery") - 7.0);
            return Status::Running;
        });
    };
    auto make_charge = []() {
        return std::make_shared<Action>("Charge", [](Blackboard& bb) {
            const double b = std::min(100.0, bb.number("battery") + 30.0);
            bb.set("battery", b);
            bb.set("charging", b < 100.0);  // only read by the fixed tree
            return b >= 100.0 ? Status::Success : Status::Running;
        });
    };
    auto trace = [](Tree& t) {
        t.blackboard().set("battery", 50.0);
        t.blackboard().set("charging", false);
        std::vector<double> battery;
        for (int k = 0; k < 14; ++k) {
            t.tick();
            battery.push_back(t.blackboard().number("battery"));
        }
        return battery;
    };
    // naive: one tick of charging lifts the battery above 20 %, and the reactive fallback preempts Charge
    Tree naive(std::make_shared<Fallback>(
        "root",
        std::vector<NodePtr>{
            std::make_shared<Sequence>(
                "work", std::vector<NodePtr>{compare("BatteryOk", "battery", ">=", 20.0), make_patrol()}),
            make_charge()}));
    const std::vector<double> b1 = trace(naive);
    CHECK(b1[4] == 15.0);  // 50, 43, 36, 29, 22 -> 15 after five patrol ticks
    CHECK(b1[5] == 45.0);  // one charging tick ...
    CHECK(b1[6] == 38.0);  // ... and the robot is back on patrol
    CHECK(*std::max_element(b1.begin() + 5, b1.end()) < 50.0);  // it never charges fully: it chatters

    // fix: work only while not charging; Charge holds the flag until the battery is full
    Tree fixed(std::make_shared<Fallback>(
        "root",
        std::vector<NodePtr>{
            std::make_shared<Sequence>(
                "work", std::vector<NodePtr>{std::make_shared<Inverter>(
                                                 "NotCharging", compare("Charging", "charging", "==", 1.0)),
                                             compare("BatteryOk", "battery", ">=", 20.0), make_patrol()}),
            make_charge()}));
    const std::vector<double> b2 = trace(fixed);
    CHECK(b2[4] == 15.0);
    CHECK(b2[5] == 45.0);
    CHECK(b2[6] == 75.0);
    CHECK(b2[7] == 100.0);
    CHECK(b2[8] == 93.0);
}
