"""Chapter 5 reference test: our RRT* against OMPL's RRT* on the same problem.

OMPL is used only here, as an independent reference (pip install ompl). Both planners must return paths whose
length is close to the exact optimum around a square, 2 + 2 sqrt(2) (see cpp/tests/test_graph_search.cpp), and
close to each other.
"""

import numpy as np
import pytest

import motion_planning as mp

ob = pytest.importorskip("ompl.base")
og = pytest.importorskip("ompl.geometric")

OPTIMAL = 2.0 + 2.0 * np.sqrt(2.0)
SQUARE = np.array([[1.0, -1.0], [3.0, -1.0], [3.0, 1.0], [1.0, 1.0]])
LOWER, UPPER = np.array([-0.5, -2.5]), np.array([4.5, 2.5])
START, GOAL = np.array([0.0, 0.0]), np.array([4.0, 0.0])
GOAL_TOLERANCE = 0.05


def problem():
    return mp.problem_from_polygons([SQUARE], LOWER, UPPER, START, GOAL)


def ompl_rrt_star_length(seconds=1.0, step=0.5):
    try:
        from ompl import util as ou

        ou.setLogLevel(ou.LOG_ERROR)
    except (ImportError, AttributeError):
        pass
    prob = problem()
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    for i in range(2):
        bounds.setLow(i, LOWER[i])
        bounds.setHigh(i, UPPER[i])
    space.setBounds(bounds)
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(lambda s: prob.state_valid(np.array([s[0], s[1]])))
    si = ss.getSpaceInformation()
    si.setStateValidityCheckingResolution(0.001)  # discretised motion checks every 0.1 % of the space extent
    start, goal = space.allocState(), space.allocState()
    start[0], start[1] = START
    goal[0], goal[1] = GOAL
    ss.setStartAndGoalStates(start, goal, GOAL_TOLERANCE)
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    planner = og.RRTstar(si)
    planner.setRange(step)
    ss.setPlanner(planner)
    assert ss.solve(seconds)
    assert ss.haveExactSolutionPath()
    path = ss.getSolutionPath()
    end = np.array([path.getState(path.getStateCount() - 1)[i] for i in range(2)])
    return path.length() + np.linalg.norm(end - GOAL)  # complete the path to the goal point


def test_rrt_star_matches_ompl():
    ours = np.mean([mp.rrt_star(problem(), mp.RrtOptions(max_iterations=4000, step=0.5, seed=s)).path_cost
                    for s in (1, 2, 3)])
    theirs = ompl_rrt_star_length()
    assert ours >= OPTIMAL - 1e-9
    assert theirs >= OPTIMAL - 2 * GOAL_TOLERANCE
    assert ours <= 1.05 * OPTIMAL
    assert theirs <= 1.05 * OPTIMAL
    assert abs(ours - theirs) <= 0.05 * OPTIMAL
