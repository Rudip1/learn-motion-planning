"""Reinforcement learning basics (chapter 11): the C++ MDP solvers and TD learners, re-exported so that
``import motion_planning.rl`` works."""

from ._core import rl as _rl

globals().update({name: getattr(_rl, name) for name in dir(_rl) if not name.startswith("_")})
__all__ = [name for name in dir(_rl) if not name.startswith("_")]
