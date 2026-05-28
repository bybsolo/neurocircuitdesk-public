"""
library/optics.py
-----------------
Templates and algorithms for the early visual stages of the *Drosophila*
optic lobe: photoreceptor divisive normalisation (PR_col + MVP), and the
ON/OFF bandpass split (ONOFF_col).

All entries are registered with the module-level template and algorithm
registries (see ``neurocircuitdesk.registry``). Importing this module is
sufficient to make ``get_template('pr_dnp')`` etc. work.

Migrated from ``docs/demos/demo_motion_circuit.py`` with no behavioural
changes — the functions are byte-equivalent; only the registration
decorators are new.
"""
from __future__ import annotations
import math
from typing import Dict

import numpy as np

from neurocircuitdesk.microcircuit import MicroCircuit
from neurocircuitdesk.blocks_exe import unified_algorithm
from neurocircuitdesk.registry import template


# ── PR_col template (DNP with MVP feedback hooks) ─────────────────────────

@template(
    name='pr_dnp',
    category='columnar',
    description=(
        'Photoreceptor divisive normalisation: T1/T2 polynomial '
        'numerator/denominator feeding a Division block with optional '
        'lateral (MVP) feedback inputs.'
    ),
    default_z=1.3,
)
def pr_dnp_template(mc: MicroCircuit):
    """PR_col template (formerly ``pr_dnp_template`` in the motion demo)."""
    passthrough_pos = (mc.center[0], mc.center[1], mc.center[2] - 0.5)
    mc.add_block('passthrough', *passthrough_pos)

    mc.add_block('T1', *(mc.center[0] - 0.12, mc.center[1], mc.center[2] + 0.75))
    mc.add_block('T2', *(mc.center[0] + 0.12, mc.center[1], mc.center[2] + 0.75))
    mc.add_block('division_block', *mc.center, node_kind='division')

    div_node = mc.get_exec_node('division_block')
    div_node.add_input_port('num_in', port_type='numerator')
    div_node.add_input_port('den_in', port_type='denominator')
    div_node.add_input_port('den_feedback_val',    port_type='denominator',
                            aggregation='weighted_mean')
    div_node.add_input_port('den_feedback_weight', port_type='denominator',
                            aggregation='weighted_mean')

    mc.connect('T1', 'output', 'division_block', 'num_in')
    mc.connect('T2', 'output', 'division_block', 'den_in')

    mc.specify_io(
        inputs=[
            ("input_main",           "T1",             "input"),
            ("input_main",           "T2",             "input"),
            ("input_main",           "passthrough",    "input"),
            ("den_feedback_val",     "division_block", "den_feedback_val"),
            ("den_feedback_weight",  "division_block", "den_feedback_weight"),
        ],
        outputs=[
            ("output_main",        "division_block", "output"),
            ("input_passthrough",  "passthrough",    "output"),
        ],
    )


# ── PR_col polynomial algorithms ───────────────────────────────────────────

@unified_algorithm(
    name='poly2_T1',
    description='Quadratic polynomial: b1 + a1*x + a2*x^2 (PR numerator).',
)
def T1_poly(inputs, p):
    x = inputs['input']
    return {'output': p['b1'] + p['a1'] * x + p['a2'] * (x * x)}


@unified_algorithm(
    name='poly2_T2',
    description='Quadratic polynomial: b2 + c1*x + c2*x^2 (PR denominator).',
)
def T2_poly(inputs, p):
    x = inputs['input']
    return {'output': p['b2'] + p['c1'] * x + p['c2'] * (x * x)}


# Default DNP parameters (from the motion demo).
DNP_PARAMS = {
    'T1': {'b1': 0.0,  'a1': 0.001, 'a2': 1e-7},
    'T2': {'b2': 1.0,  'c1': 0.001, 'c2': 1e-7},
}


# ── MVP template (intercolumnar lateral pooling) ──────────────────────────

@template(
    name='mvp_lateral',
    category='intercolumnar',
    description=(
        'Lateral inhibition pool: aggregates a ring-2 neighbourhood of '
        'upstream columns and emits both a weighted feedback value '
        '(output_val_col_*) and a weight (output_weight_col_*) per neighbour.'
    ),
    default_z=-0.3,
    requires_neighborhood=True,
)
def mvp_microcircuit_template(mc, neighborhood: Dict[int, int]):
    """MVP intercolumnar template (formerly ``mvp_microcircuit_template``)."""
    input_cols = sorted(neighborhood.keys())
    input_port_names = [f'input_col_{i}' for i in input_cols]
    output_port_names = ([f'output_val_col_{i}' for i in input_cols] +
                         [f'output_weight_col_{i}' for i in input_cols])

    mc.add_block(
        'mvp_processor', *mc.center,
        input_names=input_port_names,
        output_names=output_port_names,
    )
    mc.specify_io(
        inputs=[(name, 'mvp_processor', name) for name in input_port_names],
        outputs=[(name, 'mvp_processor', name) for name in output_port_names],
    )


@unified_algorithm(
    name='mvp_lateral_mean',
    description=(
        'Mean over the masked neighbourhood; scales each neighbour by a '
        'small delta(y) factor. Returns weighted_mean values + weights '
        'for downstream DNP feedback.'
    ),
)
def mvp_algorithm(inputs, params):
    F    = inputs['neighbors']
    mask = inputs['neighbor_mask']
    g1   = params['g1']

    n_valid = mask.sum(axis=-1, keepdims=True)
    y       = (F * mask).sum(axis=-1, keepdims=True) / n_valid
    delta   = 1.0165216804198919e-07 * y + 0.001760445128947395 - 0.001

    return {
        'output_val_col_neighbors':    delta * F * 0.33,
        'output_weight_col_neighbors': g1,
    }


# ── ONOFF_col template (bandpass + ON/OFF split) ──────────────────────────

@template(
    name='onoff_bandpass',
    category='columnar',
    description=(
        'Temporal bandpass followed by ON/OFF rectifier split. Outputs '
        '`output_main_on` (positive rectified) and `output_main_off` '
        '(inverted rectified).'
    ),
    default_z=0.3,
)
def onoff_template(mc):
    """ONOFF_col template (formerly ``onoff_template``)."""
    mc.add_block('bp_block',  *(mc.center[0],        mc.center[1],
                                mc.center[2] + 0.75), node_kind='temporal_filter')
    mc.add_block('on_block',  *(mc.center[0] - 0.12, mc.center[1],
                                mc.center[2] + 0.25), node_kind='rectifier_pos')
    mc.add_block('off_block', *(mc.center[0] + 0.12, mc.center[1],
                                mc.center[2] + 0.25), node_kind='rectifier_inv')

    mc.connect('bp_block', 'output', 'on_block',  'input')
    mc.connect('bp_block', 'output', 'off_block', 'input')

    mc.specify_io(
        inputs=[("input_main", "bp_block", "input")],
        outputs=[
            ("output_main_on",  "on_block",  "output"),
            ("output_main_off", "off_block", "output"),
        ],
    )


def bp_filter() -> np.ndarray:
    """Default biphasic-gamma bandpass kernel used by ONOFF_col."""
    def gamma(t, n, tau):
        return ((n * t) ** n * np.exp(-n * t / tau)
                / (math.factorial(n - 1) * tau ** (n + 1)) * (t[1] - t[0]))
    tH = np.arange(0, 20, 1)
    return gamma(tH, 2, 2) - gamma(tH, 6, 4)


def gaussian_kernel_from_distances(neighborhood: Dict[int, int],
                                   sigma: float) -> Dict[int, float]:
    """Per-neighbour Gaussian weight, keyed by col_idx.

    Equivalent to ``Canvas._compute_kernel_value({'type':'gaussian','sigma':s}, nbhd)``
    but kept here for callers that want a free function.
    """
    return {col: float(np.exp(-0.5 * (dist / sigma) ** 2))
            for col, dist in neighborhood.items()}
