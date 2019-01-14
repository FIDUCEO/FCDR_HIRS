"""Package for analysis and production for the FIDUCEO FCDR HIRS.

This package contains subpackages, modules, classes, and functions for the
production of the FIDUCEO HIRS FCDR.  The main package contains modules
with relatively generic functionality.  The `analysis` subpackage contains
functionality for the analysis, either of L1B data in preparation for FCDR
generation, or analysis of the FCDR itself.  Many of those are plotting
scripts, that the end-user calls as a script (see the section on scripts).
Within the `processing` subpackage are modules containing higher level
functionality to produce the FCDR.  Again, those are available as scripts
such as ``generate_fcdr``.

The documentation is currently short on examples, but all code is being
used, so the source code itself will serve as an example on how to use the
source code.
"""

from . import cached
from . import common
from . import effects
from . import exceptions
from . import _fcdr_defs
from . import fcdr
from . import filters
from . import graphics
from . import _harm_defs
from . import matchups
from . import math
from . import measurement_equation
from . import metrology
from . import models

__all__ = ["analysis", "processing",
    "cached", "common", "effects", "exceptions", "fcdr", "filters",
    "graphics", "matchups", "math", "measurement_equation", "metrology",
    "models", "_fcdr_defs", "_harm_defs"]
