"""
neurocircuitdesk.library
------------------------
Registered templates and algorithms for canonical fly-visual-system circuits.

Importing this package registers everything in the module-level template and
algorithm registries (see ``neurocircuitdesk.registry``). After import,
templates and algorithms can be looked up by string name.

Submodules
----------
optics   PR_col, MVP, ONOFF_col templates + supporting algorithms
motion   T4/T5 motion detector + ``build_motion_pipeline`` motif (Phase 2)
looming  LPLC2 looming detector + ``build_looming_pipeline`` motif (Phase 5)
"""
from . import optics  # noqa: F401  — import for side-effect registration
from . import motion  # noqa: F401

__all__ = ["optics", "motion"]
