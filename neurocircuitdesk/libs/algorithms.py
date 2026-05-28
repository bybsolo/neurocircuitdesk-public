"""
neurocircuitdesk.algorithms
---------------------------
Reusable unified-signature algorithms for common circuit motifs.

Currently houses three direction-selective motion detectors that share
the same I/O contract — all of them are drop-in alternatives for one
another inside any MicroCircuit whose ``input_col_<N>`` neighbourhood
follows the standard 7-cell layout (slot 0 = centre, slots 1..6 = ring-1
clockwise from north):

================  ======================================================
``borst``         Borst-style detector with multiplicative gain and
                  divisive normalisation on the delayed arm. The
                  shipped demos default to this one. Tunable via
                  ``alpha`` (gain) and ``beta`` (normalisation).
``hr``            Classic Hassenstein-Reichardt correlator: signed
                  ``x · y_delay − y · x_delay``. Cheap, biologically
                  influential, but can produce negative responses.
``bl``            Barlow-Levick / half-wave-rectified HR. Output is
                  ``max(HR, 0)``, preserving direction selectivity
                  while ensuring non-negative responses.
================  ======================================================

All three are decorated with ``@unified_algorithm`` so they run on both
the scalar (NumPy) engine and the batched (MLX) engine without any
backend branching in the body.

Common contract
~~~~~~~~~~~~~~~

Inputs
    ``inputs['neighbors']`` is the assembled neighbour tensor, shape
    ``(7,)`` on the scalar engine or ``(N_cols, 7)`` on the MLX engine.
    Slot order follows the template's declared port order — see the
    ``motion_detector_template`` in the looming demo.

Returns
    ``{'output_a', 'output_b', 'output_c', 'output_d'}`` — four
    directional response channels. ``a`` and ``b`` are an
    opposing-direction pair on one axis; ``c`` and ``d`` are the same on
    the other.

Use:
    >>> from neurocircuitdesk import borst_algorithm, hr_algorithm, bl_algorithm
    >>> mc.set_block_func('motion_detector_block', hr_algorithm)
    >>> mc.set_block_params('motion_detector_block', {'delay': 2})
"""

from neurocircuitdesk.blocks_exe import unified_algorithm
from neurocircuitdesk import state_utils as su
from neurocircuitdesk import math as nm


# ── Borst ───────────────────────────────────────────────────────────────

@unified_algorithm
def borst_algorithm(inputs, params, state):
    """Borst-style motion detector with tunable gain and normalisation.

    Each output is ``y_mean · (1 + α · x_mean) / (1 + β · z_mean)`` where
    ``y_mean`` is averaged over the current frame, and ``x_mean`` /
    ``z_mean`` are averaged over the same-side and opposite-side delayed
    inputs.

    Parameters
    ----------
    N : int, default 1
        Temporal delay in frames.
    alpha : float, default 100
        Multiplicative gain on the same-side delayed term.
    beta : float, default 100
        Divisive normalisation on the opposite-side delayed term.
    """
    F = inputs['neighbors']
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


# ── Shared HR / BL branch geometry ──────────────────────────────────────

def _hr_branches(y_vals, delayed):
    """Build the four (x, x_delay, y, y_delay) tuples for HR / BL branches.

    Branches a/b: south-axis opponent pair (cells 5,6 vs 2,3) + centre.
    Branches c/d: east-west opponent pair (cell 1 vs 4)       + centre.
    """
    # Vertical axis (a / b)
    y_56 = (y_vals[..., 0] + y_vals[..., 5] + y_vals[..., 6]) / 3.0
    d_23 = (delayed[..., 0] + delayed[..., 2] + delayed[..., 3]) / 3.0
    y_23 = (y_vals[..., 0] + y_vals[..., 2] + y_vals[..., 3]) / 3.0
    d_56 = (delayed[..., 0] + delayed[..., 5] + delayed[..., 6]) / 3.0

    # Horizontal axis (c / d)
    y_n = (y_vals[..., 0] + y_vals[..., 1]) / 2.0
    d_s = (delayed[..., 0] + delayed[..., 4]) / 2.0
    y_s = (y_vals[..., 0] + y_vals[..., 4]) / 2.0
    d_n = (delayed[..., 0] + delayed[..., 1]) / 2.0

    return [
        (y_56, d_23, y_23, d_56),  # a
        (y_23, d_56, y_56, d_23),  # b — opposite direction of a
        (y_n,  d_s,  y_s,  d_n ),  # c
        (y_s,  d_n,  y_n,  d_s ),  # d — opposite direction of c
    ]


def _zero_outputs(F):
    z = F[..., 0] * 0.0
    return {'output_a': z, 'output_b': z, 'output_c': z, 'output_d': z}


# ── Hassenstein-Reichardt ──────────────────────────────────────────────

@unified_algorithm
def hr_algorithm(inputs, params, state):
    """Hassenstein-Reichardt elementary motion detector.

    Each branch computes the signed two-arm correlator
    ``x · y_delay − y · x_delay``. Output is direction-selective and can
    be negative; pair with a downstream rectifier to obtain a
    Barlow-Levick-style non-negative response, or use ``bl_algorithm``
    directly.

    Parameters
    ----------
    delay : int, default 1
        Temporal delay in frames between the direct and delayed arms.
    """
    F = inputs['neighbors']
    N_delay = params.get('delay', 1)

    buf = su.ring_buffer_push(state, 'history', F, maxlen=N_delay + 1)

    if su.ring_buffer_len(buf, state, 'history') <= N_delay:
        return (_zero_outputs(F), state)

    y_vals  = su.ring_buffer_get(buf, -1)
    delayed = su.ring_buffer_get(buf,  0)

    out = {}
    for label, (x, x_d, y, y_d) in zip('abcd', _hr_branches(y_vals, delayed)):
        out[f'output_{label}'] = x * y_d - y * x_d
    return out, state


# ── Barlow-Levick (half-wave-rectified HR) ─────────────────────────────

@unified_algorithm
def bl_algorithm(inputs, params, state):
    """Barlow-Levick detector — half-wave-rectified HR correlator.

    Output is ``max(x · y_delay − y · x_delay, 0)``: preserves the
    direction selectivity of HR but suppresses opposite-direction
    (negative) responses to zero, matching the classical Barlow-Levick
    model.

    Parameters
    ----------
    delay : int, default 1
        Temporal delay in frames.
    """
    F = inputs['neighbors']
    N_delay = params.get('delay', 1)

    buf = su.ring_buffer_push(state, 'history', F, maxlen=N_delay + 1)

    if su.ring_buffer_len(buf, state, 'history') <= N_delay:
        return (_zero_outputs(F), state)

    y_vals  = su.ring_buffer_get(buf, -1)
    delayed = su.ring_buffer_get(buf,  0)

    out = {}
    for label, (x, x_d, y, y_d) in zip('abcd', _hr_branches(y_vals, delayed)):
        out[f'output_{label}'] = nm.maximum(x * y_d - y * x_d, 0.0)
    return out, state


__all__ = ['borst_algorithm', 'hr_algorithm', 'bl_algorithm']
