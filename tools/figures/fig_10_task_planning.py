"""Figures for 1_theory/10_task_planning.md."""

from pathlib import Path

import matplotlib.pyplot as plt

from motion_planning import task
from motion_planning.plotting import SERIES, use_style

OUT = Path(__file__).resolve().parents[2] / "1_theory" / "figures"

PLANNERS = [
    ("breadth-first", task.SearchAlgorithm.BreadthFirst, task.Heuristic.Zero),
    ("A*, $h_{max}$", task.SearchAlgorithm.AStar, task.Heuristic.HMax),
    ("A*, $h_{add}$", task.SearchAlgorithm.AStar, task.Heuristic.HAdd),
    ("greedy, $h_{add}$", task.SearchAlgorithm.GreedyBestFirst, task.Heuristic.HAdd),
    ("greedy, goal count", task.SearchAlgorithm.GreedyBestFirst, task.Heuristic.GoalCount),
]


def gripper_runs():
    ns = [2, 4, 6, 8, 10]
    results = {name: [] for name, _, _ in PLANNERS}
    for n in ns:
        _, _, t = task.load_example("gripper_domain", f"gripper_n{n}")
        for name, alg, h in PLANNERS:
            results[name].append(task.plan(t, task.SearchConfig(alg, h)))
    return ns, results


def main():
    use_style()
    ns, results = gripper_runs()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (name, _, _) in enumerate(PLANNERS):
        r = results[name]
        ax[0].semilogy(ns, [x.expanded for x in r], "o-", color=SERIES[k], label=name)
        ax[1].plot(ns, [x.cost for x in r], "o-", color=SERIES[k], label=name)
    ax[1].plot(ns, [3 * n - 1 for n in ns], "k--", lw=1, label="optimal, $3n - 1$")
    ax[0].set_title("states expanded")
    ax[1].set_title("plan length")
    for a in ax:
        a.set_xlabel("balls $n$")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "10_gripper_search.png", dpi=120)
    print("wrote 1_theory/figures/10_gripper_search.png")


if __name__ == "__main__":
    main()
