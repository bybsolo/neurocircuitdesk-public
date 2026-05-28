"""
library/motion.py
-----------------
Borst-style T4/T5 motion-detector template + algorithm, plus the canonical
motion-pipeline spec.

The ``motion_pipeline_spec`` function returns a JSON-able spec dict
covering PR_col + MVP + ONOFF_col + MOTION_ON/OFF_col with all wirings.
It is *not* a black-box mutation — the LLM-driven app loads this spec
via ``Canvas.from_spec``, after which the user can read, modify, or
extend it with primitive tools (``wire``, ``bind_algorithm``,
``add_mc_type``).

Migrated from ``docs/demos/demo_motion_circuit.py``.
"""
from __future__ import annotations
from typing import Dict, Any

from neurocircuitdesk.microcircuit import MicroCircuit
from neurocircuitdesk.blocks_exe import unified_algorithm
from neurocircuitdesk.registry import template, motif
from neurocircuitdesk import state_utils as su


# ── Motion-detector template (Borst T4/T5) ────────────────────────────────

@template(
    name='borst_motion_detector',
    category='columnar',
    description=(
        'Borst-style T4/T5 motion detector: reads a ring-1 hexagonal '
        'neighbourhood from upstream ONOFF outputs (via MISO wiring) and '
        'emits four directional motion outputs (a/b/c/d).'
    ),
    default_z=-1.0,
    params_schema={'num_rings': 1},
)
def motion_detector_template(mc: MicroCircuit, num_rings: int = 1):
    """Motion-detector template.

    Originally a factory closure (``make_motion_detector_template``) so it
    could see the canvas. The MC already holds a reference to its canvas,
    so the closure is unnecessary — we read neighbourhood ports straight
    off ``mc.canvas.graph_utils``.
    """
    ordered_cols = mc.canvas.graph_utils.local_order(
        mc.col_idx, num_rings=num_rings, require_in_graph=False)
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


@unified_algorithm(
    name='borst_t4t5',
    signature='stateful',
    description=(
        'T4/T5 motion detector: 1-tap delay + multiplicative combination '
        'over a ring-1 hexagonal neighbourhood. Returns 4 directional '
        'outputs (a/b/c/d) per column.'
    ),
)
def borst_algorithm(inputs, params, state):
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


# Default Borst parameters (from the motion demo).
BORST_PARAMS = {'N': 2, 'alpha': 100, 'beta': 100}


# ── Canonical motion-pipeline spec ────────────────────────────────────────

@motif(
    'motion',
    description=(
        'Standard fly motion-detection pipeline: PR_col (DNP) ↔ MVP '
        '(lateral feedback) → ONOFF_col (bandpass + rectify) → '
        'MOTION_ON_col + MOTION_OFF_col (Borst T4/T5 four directions).'
    ),
    params_schema={'n_cols': {'type': 'int', 'default': 547,
                              'description': 'Number of retinotopic columns'}},
)
def motion_pipeline_spec(n_cols: int = 547) -> Dict[str, Any]:
    """Return a canonical motion-pipeline spec dict.

    Equivalent to the wiring produced by ``demo_motion_circuit.py``:
    PR_col (DNP) ↔ MVP (lateral feedback) → ONOFF_col (bandpass+rectify)
    → MOTION_ON/OFF_col (T4/T5 four directions).

    Apply via::

        spec = motion_pipeline_spec(100)
        cv = Canvas.from_spec(spec, col_json_path=COL_JSON, graph_json_path=GRAPH_JSON)

    The returned dict is a normal spec — the LLM-driven app can ``apply``
    it, then modify with primitives (`wire`, `bind_algorithm`, etc.). It
    is *not* a black-box function on Canvas.
    """
    from neurocircuitdesk.library.optics import DNP_PARAMS

    return {
        'version': 1,
        'canvas': {
            'col_json': None,        # caller may override at from_spec time
            'graph_json': None,
            'n_cols': n_cols,
        },
        'mc_types': [
            {
                'name': 'PR_col',
                'category': 'columnar',
                'z': 1.3,
                'template': 'pr_dnp',
                'template_params': {},
            },
            {
                'name': 'MVP',
                'category': 'intercolumnar',
                'z': -0.3,
                'template': 'mvp_lateral',
                'template_params': {},
                'centers': {
                    'limit': n_cols, 'step': 2, 'jump': 2,
                    'num_rings': 2, 'require_in_graph': False,
                },
                'neighborhood_kernel': {'type': 'gaussian', 'sigma': 0.85},
            },
            {
                'name': 'ONOFF_col',
                'category': 'columnar',
                'z': 0.3,
                'template': 'onoff_bandpass',
                'template_params': {},
            },
            {
                'name': 'MOTION_ON_col',
                'category': 'columnar',
                'z': -1.0,
                'template': 'borst_motion_detector',
                'template_params': {'num_rings': 1},
            },
            {
                'name': 'MOTION_OFF_col',
                'category': 'columnar',
                'z': -1.0,
                'template': 'borst_motion_detector',
                'template_params': {'num_rings': 1},
            },
        ],
        'algorithms': [
            {'mc_type': 'PR_col', 'block': 'T1',
             'algo': 'poly2_T1', 'params': dict(DNP_PARAMS['T1'])},
            {'mc_type': 'PR_col', 'block': 'T2',
             'algo': 'poly2_T2', 'params': dict(DNP_PARAMS['T2'])},
            {'mc_type': 'MVP', 'block': 'mvp_processor',
             'algo': 'mvp_lateral_mean', 'params': {},
             'kernel_param': 'g1'},
            {'mc_type': 'MOTION_ON_col', 'block': 'motion_detector_block',
             'algo': 'borst_t4t5', 'params': dict(BORST_PARAMS)},
            {'mc_type': 'MOTION_OFF_col', 'block': 'motion_detector_block',
             'algo': 'borst_t4t5', 'params': dict(BORST_PARAMS)},
        ],
        'block_params': [
            # bp_filter is stored as a literal list (length 20) — keeps spec
            # self-contained and reproducible without re-running the kernel
            # constructor at load time.
            {'mc_type': 'ONOFF_col', 'block': 'bp_block',
             'params': {'filter': _default_bp_filter_as_list()}},
        ],
        'wirings': [
            # PR_col fans out its passthrough into MVP's neighbourhood inputs
            {'src': 'PR_col', 'src_port': 'input_passthrough',
             'dst': 'MVP', 'dst_port': 'input_col',
             'pattern': 'mimo', 'anchor': 'dst_center'},
            # MVP fans out its per-neighbour val + weight back into PR feedback
            {'src': 'MVP', 'src_port': 'output_val_col',
             'dst': 'PR_col', 'dst_port': 'den_feedback_val',
             'pattern': 'mimo', 'anchor': 'src_center'},
            {'src': 'MVP', 'src_port': 'output_weight_col',
             'dst': 'PR_col', 'dst_port': 'den_feedback_weight',
             'pattern': 'mimo', 'anchor': 'src_center'},
            # PR_col → ONOFF_col same-column
            {'src': 'PR_col', 'src_port': 'output_main',
             'dst': 'ONOFF_col', 'dst_port': 'input_main',
             'pattern': 'siso'},
            # ONOFF_col → MOTION_ON_col with ring-1 fan-in
            {'src': 'ONOFF_col', 'src_port': 'output_main_on',
             'dst': 'MOTION_ON_col', 'dst_port': 'input_col',
             'pattern': 'miso', 'num_rings': 1},
            {'src': 'ONOFF_col', 'src_port': 'output_main_off',
             'dst': 'MOTION_OFF_col', 'dst_port': 'input_col',
             'pattern': 'miso', 'num_rings': 1},
        ],
    }


def _default_bp_filter_as_list():
    """Compute the default biphasic-gamma kernel and return it as a list.

    Encapsulated as a function to keep ``motion_pipeline_spec`` callable
    at import time without forcing the bp_filter import-time computation
    onto every library consumer.
    """
    from neurocircuitdesk.library.optics import bp_filter
    return bp_filter().tolist()
