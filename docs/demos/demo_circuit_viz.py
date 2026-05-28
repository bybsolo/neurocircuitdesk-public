"""
demo_circuit_viz.py
-------------------
Build the full motion-detection circuit (PR -> MVP -> ONOFF -> T4/T5) and
save an interactive 3D Plotly HTML visualisation.  No input data (.h5)
is required -- this demo only constructs and wires the circuit topology.

Usage:
    python demo_circuit_viz.py

Outputs:
    motion_circuit.html   -- interactive 3D circuit diagram (open in browser)
"""

import os
import sys
import time
import math
from typing import Dict
from collections import deque

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from neurocircuitdesk.microcircuit import MicroCircuit
from neurocircuitdesk.canvas import Canvas
from neurocircuitdesk.blocks_exe import unified_algorithm
from neurocircuitdesk import state_utils as su
from neurocircuitdesk import math as nm

COL_JSON   = os.path.join(_REPO_ROOT, 'neurocircuitdesk', 'libs', 'jsons',
                          'hexcol_l1m3_new_578.json')
GRAPH_JSON = os.path.join(_REPO_ROOT, 'neurocircuitdesk', 'libs', 'jsons',
                          'hex_grid_graph.json')


# ── MicroCircuit templates ─────────────────────────────────────────────────

def pr_dnp_template(mc: MicroCircuit):
    """Photoreceptor with divisive normalization (T1/T2 polynomials + division)."""
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


def onoff_template(mc: MicroCircuit):
    """ON/OFF channel split: bandpass -> positive / inverted rectifiers."""
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


def mvp_microcircuit_template(mc: MicroCircuit, neighborhood: Dict[int, int]):
    """MIMO lateral processor: receives from neighbouring columns, emits feedback."""
    input_cols = sorted(neighborhood.keys())
    input_port_names  = [f'input_col_{i}' for i in input_cols]
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


def make_motion_detector_template(canvas: Canvas):
    """Returns a template that wires a Borst-style motion detector.

    The detector is inter-columnar (centre + 6 ring-1 neighbours), so the
    returned template takes a ``neighborhood`` kwarg and is meant to be
    added via ``Canvas.add_microcircuit_intercolumnar``.
    """
    def motion_detector_template(mc: MicroCircuit, neighborhood):
        del neighborhood  # accepted for iCMC API parity; body uses local_order
        ordered_cols = canvas.graph_utils.local_order(
            mc.col_idx, num_rings=1, require_in_graph=False)
        input_port_names  = [f'input_col_{c}' for c in ordered_cols]
        output_port_names = ['output_a', 'output_b', 'output_c', 'output_d']

        mc.add_block(
            'motion_detector_block', *mc.center,
            input_names=input_port_names,
            output_names=output_port_names,
            stateless=False,
        )
        mc.specify_io(
            inputs=[(name, 'motion_detector_block', name) for name in input_port_names],
            outputs=[(name, 'motion_detector_block', name) for name in output_port_names],
        )
    return motion_detector_template


# ── Algorithms (unified backend-agnostic signature) ────────────────────────

@unified_algorithm
def T1_poly_demo(inputs, p):
    x = inputs['input']
    return {'output': p['b1'] + p['a1'] * x + p['a2'] * (x * x)}


@unified_algorithm
def T2_poly_demo(inputs, p):
    x = inputs['input']
    return {'output': p['b2'] + p['c1'] * x + p['c2'] * (x * x)}


def bp_filter():
    """Gamma-function bandpass kernel."""
    def gamma(t, n, tau):
        return ((n * t) ** n * np.exp(-n * t / tau)
                / (math.factorial(n - 1) * tau ** (n + 1)) * (t[1] - t[0]))
    tH = np.arange(0, 20, 1)
    return gamma(tH, 2, 2) - gamma(tH, 6, 4)


def gaussian_kernel_from_distances(neighborhood: Dict[int, int],
                                   sigma: float) -> Dict[int, float]:
    return {col: np.exp(-0.5 * (dist / sigma) ** 2)
            for col, dist in neighborhood.items()}


@unified_algorithm
def mvp_algorithm_demo(inputs, params):
    """Fan-out MIMO: per-neighbour delta scaled by neighbourhood mean."""
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


@unified_algorithm
def borst_3branch_algorithm(inputs, params, state):
    """Stateful motion detector with four directional branches."""
    F       = inputs['neighbors']
    N_delay = params.get('N', 1)
    alpha   = params.get('alpha', 100)
    beta    = params.get('beta', 100)

    buf = su.ring_buffer_push(state, 'history', F, maxlen=N_delay + 1)
    zeros = F[..., 0] * 0.0

    if su.ring_buffer_len(buf, state, 'history') <= N_delay:
        return ({'output_a': zeros, 'output_b': zeros,
                 'output_c': zeros, 'output_d': zeros}, state)

    y_vals  = su.ring_buffer_get(buf, -1)
    delayed = su.ring_buffer_get(buf,  0)

    def branch(y_mean, x_mean, z_mean):
        return y_mean * (1.0 + x_mean * alpha) / (1.0 + z_mean * beta)

    y_a = (y_vals[..., 0] + y_vals[..., 1] + y_vals[..., 4]) / 3.0
    y_c = (y_vals[..., 0] + y_vals[..., 5] + y_vals[..., 6]
           + y_vals[..., 2] + y_vals[..., 3]) / 5.0
    y_d = (y_vals[..., 0] + y_vals[..., 2] + y_vals[..., 3]
           + y_vals[..., 5] + y_vals[..., 6]) / 5.0

    d_23  = (delayed[..., 2] + delayed[..., 3]) / 2.0
    d_56  = (delayed[..., 5] + delayed[..., 6]) / 2.0
    d_543 = (delayed[..., 5] + delayed[..., 4] + delayed[..., 3]) / 3.0
    d_126 = (delayed[..., 1] + delayed[..., 2] + delayed[..., 6]) / 3.0

    val_a = branch(y_a, d_23,  d_56)
    val_b = branch(y_a, d_56,  d_23)
    val_c = branch(y_c, d_543, d_126)
    val_d = branch(y_d, d_126, d_543)

    return ({'output_a': val_a, 'output_b': val_b,
             'output_c': val_c, 'output_d': val_d}, state)


# ── Parameters ─────────────────────────────────────────────────────────────

DNP_PARAMS   = {'T1': {'b1': 0, 'a1': 0.001, 'a2': 1e-7},
                'T2': {'b2': 1.0, 'c1': 0.001, 'c2': 1e-7}}
BORST_PARAMS = {'N': 2, 'alpha': 100, 'beta': 100}
N_COLS       = 1261


# ── Build circuit ──────────────────────────────────────────────────────────

def build_motion_circuit():
    """Build and wire the full motion-detection circuit (no execution)."""

    cv = Canvas(
        w=900, h=700,
        col_json_path=COL_JSON,
        interconnect_json_path=GRAPH_JSON,
    )
    for t in ('PR_col', 'MVP', 'ONOFF_col', 'MOTION_ON_col', 'MOTION_OFF_col'):
        cv.add_mc_type(t)

    # 1. Photoreceptors (top)
    for col_idx in range(N_COLS):
        cv.add_microcircuit_columnar(col_idx=col_idx, z=1.5,
                                     mc_type='PR_col', template=pr_dnp_template)

    # 2. ON/OFF stage (middle)
    for col_idx in range(N_COLS):
        cv.add_microcircuit_columnar(col_idx=col_idx, z=0.0,
                                     mc_type='ONOFF_col', template=onoff_template)

    # 3. MVPs -- lateral feedback (between PR and ONOFF)
    mvp_centers = cv.graph_utils.calc_mimo_centers(
        limit=N_COLS, step=2, jump=2, num_rings=2, require_in_graph=False)
    for col_idx, neighborhood in mvp_centers.items():
        cv.add_microcircuit_intercolumnar(
            mc_type='MVP', center_col_idx=col_idx,
            neighborhood=neighborhood, z=0.8,
            template=mvp_microcircuit_template)

    # 4. Motion detectors (bottom, ON slightly above OFF). Inter-columnar
    # at the data-flow level — each cell reads centre + 6 ring-1 inputs.
    motion_template = make_motion_detector_template(cv)
    motion_centers = cv.graph_utils.calc_mimo_centers(
        limit=N_COLS, step=1, jump=1, num_rings=1, require_in_graph=False)
    for col_idx, neighborhood in motion_centers.items():
        cv.add_microcircuit_intercolumnar(
            mc_type='MOTION_ON_col', center_col_idx=col_idx,
            neighborhood=neighborhood, z=-0.8, template=motion_template)
        cv.add_microcircuit_intercolumnar(
            mc_type='MOTION_OFF_col', center_col_idx=col_idx,
            neighborhood=neighborhood, z=-1.2, template=motion_template)

    # 5. Configure functions/params
    for mc in cv.mc_types['PR_col']:
        mc.set_block_func('T1', T1_poly_demo, DNP_PARAMS['T1'])
        mc.set_block_func('T2', T2_poly_demo, DNP_PARAMS['T2'])

    for mc in cv.mc_types['MVP']:
        mc.set_block_func('mvp_processor', mvp_algorithm_demo)
        g1 = gaussian_kernel_from_distances(mvp_centers[mc.col_idx], 0.85)
        mc.set_block_params('mvp_processor', {'g1': g1})

    for mc in cv.mc_types['ONOFF_col']:
        mc.set_block_params('bp_block', {'filter': bp_filter()})

    for mc in cv.mc_types['MOTION_ON_col']:
        mc.set_block_func('motion_detector_block', borst_3branch_algorithm)
        mc.set_block_params('motion_detector_block', BORST_PARAMS)
    for mc in cv.mc_types['MOTION_OFF_col']:
        mc.set_block_func('motion_detector_block', borst_3branch_algorithm)
        mc.set_block_params('motion_detector_block', BORST_PARAMS)

    # 6. Wire connections
    print("Wiring PR -> MVP ...")
    for col_idx in mvp_centers.keys():
        cv.connect_utils.mimo(
            'PR_col', 'input_passthrough', 'MVP', 'input_col',
            dst_center_col_idx=col_idx,
            cols=list(mvp_centers[col_idx].keys()))

    print("Wiring MVP -> PR feedback ...")
    for col_idx in mvp_centers.keys():
        cv.connect_utils.mimo(
            'MVP', 'output_val_col', 'PR_col', 'den_feedback_val',
            src_center_col_idx=col_idx,
            cols=list(mvp_centers[col_idx].keys()))
        cv.connect_utils.mimo(
            'MVP', 'output_weight_col', 'PR_col', 'den_feedback_weight',
            src_center_col_idx=col_idx,
            cols=list(mvp_centers[col_idx].keys()))

    print("Wiring PR -> ONOFF ...")
    for col_idx in range(N_COLS):
        cv.connect_utils.siso(
            'PR_col', 'output_main', 'ONOFF_col', 'input_main',
            col_idx=col_idx)

    print("Wiring ONOFF -> MOTION ...")
    for col_idx in range(N_COLS):
        cv.connect_utils.miso(
            'ONOFF_col', 'output_main_on', 'MOTION_ON_col', 'input_col',
            col_idx=col_idx, num_rings=1)
        cv.connect_utils.miso(
            'ONOFF_col', 'output_main_off', 'MOTION_OFF_col', 'input_col',
            col_idx=col_idx, num_rings=1)

    return cv


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    outdir = os.path.join(_THIS_DIR, 'outputs')
    os.makedirs(outdir, exist_ok=True)

    print("Building full motion-detection circuit (N=1261 columns) ...")
    t0 = time.perf_counter()
    cv = build_motion_circuit()
    elapsed = time.perf_counter() - t0
    print(f"  Build + wire: {elapsed:.2f}s")

    out_path = os.path.join(outdir, 'motion_circuit')
    cv.save(out_path)
    html_path = out_path + '.html'
    sz = os.path.getsize(html_path) / 1024 / 1024
    print(f"  Saved -> {html_path} ({sz:.1f} MB)")
    print("  Open the HTML file in a browser to explore the 3D circuit diagram.")
