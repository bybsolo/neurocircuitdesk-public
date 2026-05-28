"""
demo_motion_circuit.py
----------------------
Build, compile, and run the full motion-detection circuit on a natural-scene
video input (highway.h5), then visualise selected outputs as retinotopic
scatter plots.

Signal flow:
    input -> PR_col (divisive normalisation)
                ^v
               MVP  (lateral neighbourhood feedback)
                |
    PR_col.output -> ONOFF_col (bandpass -> ON/OFF rectifiers)
                          |           |
                    MOTION_ON    MOTION_OFF  (Borst T4/T5 detectors)

Usage:
    python demo_motion_circuit.py                  # default: 547 columns
    python demo_motion_circuit.py --n-cols 100     # quick test with fewer columns

Requires:
    data/highway.h5  (shipped with this demo)

Outputs (written to outputs/motion_demo/):
    *.npy            -- probed time-series arrays
    *.png            -- retinotopic scatter plots at selected frames
"""

import argparse
import os
import sys
import json
import math
import time
from typing import Dict
from collections import deque

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import h5py

from neurocircuitdesk.microcircuit import MicroCircuit
from neurocircuitdesk.canvas import Canvas
from neurocircuitdesk.blocks_exe import unified_algorithm
from neurocircuitdesk import state_utils as su

COL_JSON   = os.path.join(_REPO_ROOT, 'neurocircuitdesk', 'libs', 'jsons',
                          'hexcol_l1m3_new_578.json')
GRAPH_JSON = os.path.join(_REPO_ROOT, 'neurocircuitdesk', 'libs', 'jsons',
                          'hex_grid_graph.json')
H5_PATH    = os.path.join(_THIS_DIR, 'data', 'highway.h5')
OUTDIR     = os.path.join(_THIS_DIR, 'outputs', 'motion_demo')


# ── MicroCircuit templates ─────────────────────────────────────────────────

def pr_dnp_template(mc: MicroCircuit):
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
    def motion_detector_template(mc: MicroCircuit, neighborhood):
        # Motion detectors are inter-columnar: one centre + 6 ring-1 inputs.
        # `neighborhood` is accepted for add_microcircuit_intercolumnar API
        # parity; the ring-1 spiral ordering is re-derived inside.
        del neighborhood
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


# ── Algorithms ─────────────────────────────────────────────────────────────

@unified_algorithm
def T1_poly_demo(inputs, p):
    x = inputs['input']
    return {'output': p['b1'] + p['a1'] * x + p['a2'] * (x * x)}


@unified_algorithm
def T2_poly_demo(inputs, p):
    x = inputs['input']
    return {'output': p['b2'] + p['c1'] * x + p['c2'] * (x * x)}


def bp_filter():
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


DNP_PARAMS   = {'T1': {'b1': 0, 'a1': 0.001, 'a2': 1e-7},
                'T2': {'b2': 1.0, 'c1': 0.001, 'c2': 1e-7}}
BORST_PARAMS = {'N': 2, 'alpha': 100, 'beta': 100}


# ── Build and compile ──────────────────────────────────────────────────────

def build_and_compile(n_cols: int):
    cv = Canvas(w=900, h=700,
                col_json_path=COL_JSON,
                interconnect_json_path=GRAPH_JSON)

    for t in ('PR_col', 'MVP', 'ONOFF_col', 'MOTION_ON_col', 'MOTION_OFF_col'):
        cv.add_mc_type(t)

    for col_idx in range(n_cols):
        cv.add_microcircuit_columnar(col_idx=col_idx, z=1.3,
                                     mc_type='PR_col', template=pr_dnp_template)
    for col_idx in range(n_cols):
        cv.add_microcircuit_columnar(col_idx=col_idx, z=0.3,
                                     mc_type='ONOFF_col', template=onoff_template)

    mvp_centers = cv.graph_utils.calc_mimo_centers(
        limit=n_cols, step=2, jump=2, num_rings=2, require_in_graph=False)
    for col_idx, neighborhood in mvp_centers.items():
        cv.add_microcircuit_intercolumnar(
            mc_type='MVP', center_col_idx=col_idx,
            neighborhood=neighborhood, z=-0.3,
            template=mvp_microcircuit_template)

    # Motion detectors are inter-columnar: one centre + 6 ring-1 inputs per
    # cell. The centres + ring-1 neighbourhoods are shared between the ON
    # and OFF stacks.
    motion_template = make_motion_detector_template(cv)
    motion_centers = cv.graph_utils.calc_mimo_centers(
        limit=n_cols, step=1, jump=1, num_rings=1, require_in_graph=False)
    for col_idx, neighborhood in motion_centers.items():
        cv.add_microcircuit_intercolumnar(
            mc_type='MOTION_ON_col', center_col_idx=col_idx,
            neighborhood=neighborhood, z=-1.0, template=motion_template)
        cv.add_microcircuit_intercolumnar(
            mc_type='MOTION_OFF_col', center_col_idx=col_idx,
            neighborhood=neighborhood, z=-1.0, template=motion_template)

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

    # Wire
    print("Wiring circuit ...")
    for col_idx in mvp_centers.keys():
        cv.connect_utils.mimo('PR_col', 'input_passthrough', 'MVP', 'input_col',
                              dst_center_col_idx=col_idx,
                              cols=list(mvp_centers[col_idx].keys()),
                              skip_viz=True)

    for col_idx in mvp_centers.keys():
        cv.connect_utils.mimo('MVP', 'output_val_col', 'PR_col', 'den_feedback_val',
                              src_center_col_idx=col_idx,
                              cols=list(mvp_centers[col_idx].keys()),
                              skip_viz=True)
        cv.connect_utils.mimo('MVP', 'output_weight_col', 'PR_col', 'den_feedback_weight',
                              src_center_col_idx=col_idx,
                              cols=list(mvp_centers[col_idx].keys()),
                              skip_viz=True)

    for col_idx in range(n_cols):
        cv.connect_utils.siso('PR_col', 'output_main', 'ONOFF_col', 'input_main',
                              col_idx=col_idx, skip_viz=True)

    for col_idx in range(n_cols):
        cv.connect_utils.miso('ONOFF_col', 'output_main_on', 'MOTION_ON_col', 'input_col',
                              col_idx=col_idx, num_rings=1, skip_viz=True)
        cv.connect_utils.miso('ONOFF_col', 'output_main_off', 'MOTION_OFF_col', 'input_col',
                              col_idx=col_idx, num_rings=1, skip_viz=True)

    print("Compiling ...")
    t0 = time.perf_counter()
    prog = cv.compile()
    compile_time = time.perf_counter() - t0
    print(f"  Compile time: {compile_time:.2f}s")

    return cv, prog, compile_time


# ── Plotting helpers ───────────────────────────────────────────────────────

def load_hex_positions(n_cols: int) -> np.ndarray:
    with open(COL_JSON) as f:
        col_data = json.load(f)
    hex_coords_id = col_data['hex_coords_id']

    with open(GRAPH_JSON) as f:
        graph_data = json.load(f)
    pos = {node['id']: node['pos'] for node in graph_data['nodes']}

    coords = np.full((n_cols, 2), np.nan)
    for i in range(min(n_cols, len(hex_coords_id))):
        bio_id = hex_coords_id[i]
        if bio_id < 1000 and bio_id in pos:
            coords[i] = pos[bio_id]
    return coords


def scatter_row(coords, panels, suptitle, path,
                vmin=None, vmax=None, cmap='viridis', s=40):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (title, values) in zip(axes, panels):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values,
                        s=s, vmin=vmin, vmax=vmax, cmap=cmap, edgecolors='none')
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(suptitle)
    fig.colorbar(sc, ax=axes, shrink=0.75)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-cols', type=int, default=547,
                        help='Number of retinotopic columns (default: 547)')
    args = parser.parse_args()
    n_cols = args.n_cols

    os.makedirs(OUTDIR, exist_ok=True)

    print(f"\n== Motion circuit demo  N_COLS={n_cols} ==\n")

    # Build & compile
    cv, prog, compile_time = build_and_compile(n_cols)

    # Load input
    with h5py.File(H5_PATH, 'r') as f:
        vid = f['inputs'][:]
    lum = vid[:, :n_cols]
    print(f"Input shape: {lum.shape}")

    # Run
    print("Running program ...")
    t0 = time.perf_counter()
    prog.run_program(inputs=lum, input_microcircuits=cv.mc_types['PR_col'])
    run_time = time.perf_counter() - t0
    print(f"  Run time: {run_time:.2f}s")

    # Probe outputs
    am_T = prog.probe_result(cv.mc_types['PR_col'],         "output_main")
    on   = prog.probe_result(cv.mc_types['ONOFF_col'],      "output_main_on")
    off  = prog.probe_result(cv.mc_types['ONOFF_col'],      "output_main_off")
    T4a  = prog.probe_result(cv.mc_types['MOTION_ON_col'],  "output_a")
    T4b  = prog.probe_result(cv.mc_types['MOTION_ON_col'],  "output_b")
    T4c  = prog.probe_result(cv.mc_types['MOTION_ON_col'],  "output_c")
    T4d  = prog.probe_result(cv.mc_types['MOTION_ON_col'],  "output_d")
    T5a  = prog.probe_result(cv.mc_types['MOTION_OFF_col'], "output_a")
    T5b  = prog.probe_result(cv.mc_types['MOTION_OFF_col'], "output_b")
    T5c  = prog.probe_result(cv.mc_types['MOTION_OFF_col'], "output_c")
    T5d  = prog.probe_result(cv.mc_types['MOTION_OFF_col'], "output_d")

    # Save arrays
    tag = f'n{n_cols}'
    for name, arr in [('am_T', am_T), ('on', on), ('off', off),
                      ('T4a', T4a), ('T4b', T4b), ('T4c', T4c), ('T4d', T4d),
                      ('T5a', T5a), ('T5b', T5b), ('T5c', T5c), ('T5d', T5d)]:
        np.save(os.path.join(OUTDIR, f'{name}_{tag}.npy'), arr)

    # Visualise
    coords = load_hex_positions(n_cols)

    scatter_row(coords,
                panels=[("ON", on[75]), ("OFF", off[75])],
                suptitle=f"ON / OFF @ t=75  (N={n_cols})",
                path=os.path.join(OUTDIR, f'onoff_frame75_{tag}.png'),
                vmin=0, vmax=0.25)

    scatter_row(coords,
                panels=[("T4a", T4a[75]), ("T4b", T4b[75]),
                        ("T4c", T4c[75]), ("T4d", T4d[75])],
                suptitle=f"T4 ON motion @ t=75  (N={n_cols})",
                path=os.path.join(OUTDIR, f't4_frame75_{tag}.png'),
                vmin=0, vmax=0.25)

    scatter_row(coords,
                panels=[("T5a", T5a[75]), ("T5b", T5b[75]),
                        ("T5c", T5c[75]), ("T5d", T5d[75])],
                suptitle=f"T5 OFF motion @ t=75  (N={n_cols})",
                path=os.path.join(OUTDIR, f't5_frame75_{tag}.png'),
                vmin=0, vmax=0.25)

    print(f"\nAll outputs written to {OUTDIR}/")


if __name__ == '__main__':
    main()
