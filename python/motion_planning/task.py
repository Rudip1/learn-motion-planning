"""Task planning (chapter 10): the C++ PDDL parser, grounder and planners, re-exported so that
``import motion_planning.task`` works, plus access to the example domains shipped with the package."""

from pathlib import Path

from ._core import task as _task

globals().update({name: getattr(_task, name) for name in dir(_task) if not name.startswith("_")})

PDDL_DIR = Path(__file__).resolve().parent / "pddl"


def example(name: str) -> str:
    """Text of a shipped PDDL file, e.g. ``example("gripper_domain")`` (``.pddl`` may be omitted)."""
    path = PDDL_DIR / (name if name.endswith(".pddl") else name + ".pddl")
    return path.read_text()


def load_example(domain: str, problem: str):
    """Parse and ground a shipped domain and problem; returns ``(Domain, Problem, GroundTask)``."""
    d = _task.parse_domain(example(domain))
    p = _task.parse_problem(example(problem))
    return d, p, _task.ground(d, p)


__all__ = [name for name in dir(_task) if not name.startswith("_")] + ["PDDL_DIR", "example", "load_example"]
