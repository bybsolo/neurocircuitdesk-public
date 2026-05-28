"""
flyeyesimulator
---------------
Drosophila-style fly-eye simulator: projects 2D video onto a spherical screen
and samples it through a hex retina of R1-R6 receptive fields using von
Mises-Fisher filters.

Public API:
    Screen                   spherical screen from 2D video
    Retina                   hex grid of R1-R6 receptive fields
    ReceptiveField           single R-cell container
    RetinaRotator            rigid transform of a Retina
    FlyEyeSimulatorStatic    fixed receptive fields
    FlyEyeSimulatorActive    spring-damper dynamic receptive fields
    ScreenViz, RetinaViz, SimulatorViz   Plotly visualizations (optional deps)

Backend:
    BACKEND, xp, xp_ndimage, to_numpy, free_memory  (see .backend)
"""
from .backend import BACKEND, xp, xp_ndimage, to_numpy, free_memory
from .screen import Screen
from .retina import Retina, ReceptiveField
from .retina_rotator import RetinaRotator
from .superposition import superposition_mask, local_order, get_prs
from .flyeyesimulator_static import FlyEyeSimulator as FlyEyeSimulatorStatic
from .flyeyesimulator_active import FlyEyeSimulator as FlyEyeSimulatorActive
from . import phototransduction
from .phototransduction import (
    PhotoreceptorRetina,
    pr,
    aggregate_pr,
    fes_to_pr_voltage,
)

try:
    from .screenviz import ScreenViz
    from .retinaviz import RetinaViz
    from .simulatorviz import SimulatorViz
    from .IO_viz import IOViz
except ImportError:
    ScreenViz = RetinaViz = SimulatorViz = IOViz = None

__all__ = [
    'BACKEND', 'xp', 'xp_ndimage', 'to_numpy', 'free_memory',
    'Screen', 'Retina', 'ReceptiveField', 'RetinaRotator',
    'superposition_mask', 'local_order', 'get_prs',
    'FlyEyeSimulatorStatic', 'FlyEyeSimulatorActive',
    'ScreenViz', 'RetinaViz', 'SimulatorViz', 'IOViz',
    # Phototransduction (RPM + HH(NS)) — backend-selected at import.
    'phototransduction', 'PhotoreceptorRetina',
    'pr', 'aggregate_pr', 'fes_to_pr_voltage',
]
