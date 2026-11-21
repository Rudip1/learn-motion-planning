"""Chapter 10: the Python interface to the planner, against plans worked out by hand."""

import math

from motion_planning import task


def test_examples_are_shipped_and_parse():
    d, p, t = task.load_example("gripper_domain", "gripper_n4")
    assert d.name == "gripper" and p.name == "gripper-n4"
    assert len(t.actions) == 4 + 8 * 4


def test_sussman_plan_by_hand():
    _, _, t = task.load_example("blocksworld_domain", "blocksworld_sussman")
    names = ["(unstack c a)", "(put-down c)", "(pick-up b)", "(stack b c)", "(pick-up a)", "(stack a b)"]
    index = {a.name: i for i, a in enumerate(t.actions)}
    assert task.validate_plan(t, [index[n] for n in names]) == ""
    r = task.plan(t, task.SearchConfig(task.SearchAlgorithm.BreadthFirst, task.Heuristic.Zero))
    assert len(r.plan) == 6


def test_relaxed_heuristics_on_gripper():
    for n in (2, 4, 6):
        _, _, t = task.load_example("gripper_domain", f"gripper_n{n}")
        assert task.heuristic(t, t.init, task.Heuristic.HMax) == 2.0
        assert task.heuristic(t, t.init, task.Heuristic.HAdd) == 3.0 * n
        assert task.heuristic(t, t.init, task.Heuristic.GoalCount) == n
    assert not math.isinf(task.heuristic(t, t.init, task.Heuristic.HMax))
