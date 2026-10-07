"""Motion planning learning module: C++ algorithms (``_core``) plus small plotting helpers.

Everything that is an algorithm lives in C++ and is re-exported here; see ``1_theory/`` for the derivations.
Behaviour trees live in the submodule ``motion_planning.bt``.
"""

from ._core import *  # noqa: F401,F403
from ._core import __doc__  # noqa: F401
from . import bt  # noqa: F401  (the motion_planning.bt submodule)

__version__ = "0.1.0"
