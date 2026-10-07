"""Motion planning learning module: C++ algorithms (``_core``) plus small plotting helpers.

Everything that is an algorithm lives in C++ and is re-exported here; see ``1_theory/`` for the derivations.
"""

from ._core import *  # noqa: F401,F403
from ._core import __doc__  # noqa: F401

__version__ = "0.1.0"
