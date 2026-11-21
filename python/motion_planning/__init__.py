"""Motion planning learning module: C++ algorithms (``_core``) plus small plotting helpers.

Everything that is an algorithm lives in C++ and is re-exported here; see ``1_theory/`` for the derivations.
Behaviour trees live in the submodule ``motion_planning.bt``, task planning in ``motion_planning.task``
(with the example PDDL files), reinforcement learning in ``motion_planning.rl``.
"""

from ._core import *  # noqa: F401,F403
from ._core import __doc__  # noqa: F401
from . import bt  # noqa: F401  (the motion_planning.bt submodule)
import importlib as _importlib

# The star import above already bound `task` to the C++ submodule; load the Python wrapper (which adds the example
# PDDL files) explicitly so that it replaces it.
task = _importlib.import_module(f"{__name__}.task")
rl = _importlib.import_module(f"{__name__}.rl")

__version__ = "0.1.0"
