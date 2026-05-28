"""
NeuroCircuitDesk — programmable ontology for executable neural circuits.
"""

from neurocircuitdesk.canvas import Canvas, Program
from neurocircuitdesk.microcircuit import MicroCircuit
from neurocircuitdesk.blocks_exe import unified_algorithm
from neurocircuitdesk.io_utils import (
    HexActiveSampleSpring,
    HexActiveSampleKinetic,
    HexViz,
)
from neurocircuitdesk.video_input import VideoInput
from neurocircuitdesk import stimuli
from neurocircuitdesk.libs.algorithms import (
    borst_algorithm,
    hr_algorithm,
    bl_algorithm,
)
from neurocircuitdesk.libs.microcircuit_templates import (
    CMC_photoreceptor_dnp,
    CMC_lamina_l1l2_onoff,
    iCMC_amacrine_mvp,
    iCMC_t4t5_motiondetector,
    iCMC_lplc2_loomingdetector,
)

# mlx is an optional dependency (see pyproject.toml [project.optional-dependencies]).
# If it isn't installed, expose a stub so the rest of the package remains importable
# and direct use surfaces a helpful error instead of ModuleNotFoundError.
try:
    from neurocircuitdesk.mlx_engine import VectorizedProgram
except ImportError as _mlx_import_error:
    class VectorizedProgram:  # type: ignore[no-redef]
        """Placeholder when the optional `mlx` extra is not installed."""

        _import_error = _mlx_import_error

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "VectorizedProgram requires the optional `mlx` dependency. "
                "Install with `pip install -e './neurocircuitdesk[mlx]'`."
            ) from self._import_error

        @classmethod
        def from_program(cls, *args, **kwargs):
            raise ImportError(
                "VectorizedProgram.from_program requires the optional `mlx` "
                "dependency. Install with `pip install -e './neurocircuitdesk[mlx]'`."
            ) from cls._import_error

__all__ = [
    "Canvas",
    "Program",
    "MicroCircuit",
    "VectorizedProgram",
    "unified_algorithm",
    "HexActiveSampleSpring",
    "HexActiveSampleKinetic",
    "HexViz",
    "VideoInput",
    "stimuli",
    "borst_algorithm",
    "hr_algorithm",
    "bl_algorithm",
    "CMC_photoreceptor_dnp",
    "CMC_lamina_l1l2_onoff",
    "iCMC_amacrine_mvp",
    "iCMC_t4t5_motiondetector",
    "iCMC_lplc2_loomingdetector",
]
