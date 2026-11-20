"""Figures for 1_theory/09_behavior_trees.md."""

from pathlib import Path

import matplotlib.pyplot as plt

from motion_planning import bt
from motion_planning.plotting import SERIES, draw_tree, use_style

OUT = Path(__file__).resolve().parents[2] / "1_theory" / "figures"


def patrol(b):
    b["battery"] = b.number("battery") - 7.0
    return bt.Status.Running


def charge(b):
    level = min(100.0, b.number("battery") + 30.0)
    b["battery"] = level
    b["charging"] = level < 100.0
    return bt.Status.Success if level >= 100.0 else bt.Status.Running


def mission(fixed):
    work = [bt.compare("BatteryOk", "battery", ">=", 20.0), bt.Action("Patrol", patrol)]
    if fixed:
        work.insert(0, bt.Inverter("not", bt.compare("Charging", "charging", "==", 1.0)))
    tree = bt.Tree(bt.Fallback("root", [bt.Sequence("work", work), bt.Action("Charge", charge)]))
    tree.blackboard["battery"] = 50.0
    tree.blackboard["charging"] = False
    return tree


def battery_trace(tree, n=16):
    out = []
    for _ in range(n):
        tree.tick()
        out.append(tree.blackboard.number("battery"))
    return out


def mission_figure():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, fixed, ticks, title in [(axes[0], False, 5, "Naive tree, tick 6: Charge runs"),
                                    (axes[1], True, 6, "Fixed tree, tick 7: still charging")]:
        t = mission(fixed)
        battery_trace(t, ticks)
        t.tick()
        draw_tree(ax, t.root, x_gap=1.3)
        ax.set_title(title)
    a = battery_trace(mission(False))
    b = battery_trace(mission(True))
    axes[2].plot(range(1, 17), a, "o-", color=SERIES[0], label="naive: Fallback(Sequence(BatteryOk, Patrol), Charge)")
    axes[2].plot(range(1, 17), b, "s-", color=SERIES[1], label="with a Charging flag (hysteresis)")
    axes[2].axhline(20, color="#8a8985", ls="--", lw=1)
    axes[2].set_xlabel("tick")
    axes[2].set_ylabel("battery [%]")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8)
    axes[2].set_title("Reactive preemption makes the naive tree chatter")
    fig.tight_layout()
    fig.savefig(OUT / "09_mission.png")
    plt.close(fig)


if __name__ == "__main__":
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    mission_figure()
