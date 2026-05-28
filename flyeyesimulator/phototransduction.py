"""
flyeyesimulator.phototransduction
---------------------------------
Single-module photoreceptor model: **RPM phototransduction → HH(NS) membrane**.

This consolidates the three retina*.py prototypes (CuPy / CPU-Numba /
Metal-MLX) into one entry point. The backend is selected at import time
to match :data:`flyeyesimulator.backend.BACKEND`:

    BACKEND value  →  phototransduction backend
    ─────────────────────────────────────────────
    'cupy'         →  CuPy + numba.cuda kernel
    'mlx'          →  mlx.fast.metal_kernel
    'numpy'        →  Numba @njit + prange (CPU)

Pipeline
~~~~~~~~

::

    (T, 6·N_cols)  light intensity   ──pr()──▶  (T, 6·N_cols) voltage
                                                          │
                                          ┌───────────────┘
                                          ▼
                                 aggregate_pr(...)        uses
                                          │             superposition.get_prs
                                          ▼
                                  (T, N_cols)  voltage  ──▶  NCD PR MCs

Time grid defaults assume **100 fps video** (10 ms per outer frame):
``itpl_val=1000`` sub-steps at ``dt=1e-5 s`` per sub-step ⇒
``itpl_val · dt = 1e-2 s = 10 ms = 1 frame``. Override both together if
your video has a different frame rate.

Public API
~~~~~~~~~~

- :class:`PhotoreceptorRetina` — class wrapper, exposes ``.axon_terminal``
  of shape ``(T, N_ch)``. ``Retina`` is provided as an alias for parity
  with the prototype scripts.
- :func:`pr` — ``(T, 6·N_cols)`` light → ``(T, 6·N_cols)`` voltage.
- :func:`aggregate_pr` — ``(T, 6·N_cols)`` voltage → ``(T, N_cols)``
  via :func:`superposition.get_prs` + chosen aggregation (default
  ``mean``).
- :func:`fes_to_pr_voltage` — one-shot pipeline composing the two above.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import numpy as np

from .backend import BACKEND
from .superposition import get_prs

__all__ = [
    'BACKEND',
    'PhotoreceptorRetina',
    'Retina',
    'pr',
    'aggregate_pr',
    'fes_to_pr_voltage',
    'supported_membrane',
]

supported_membrane = ['hh', 'hhns']


# ─────────────────────────────────────────────────────────────────────────
# Backend kernel registration
#
# Each branch defines a single ``_run(input_TN, dt, itpl_val, membrane)``
# callable that returns a host numpy ``(T, N_ch)`` voltage array. The
# wrapper class below routes through whichever ``_run`` got bound.
# ─────────────────────────────────────────────────────────────────────────

_RUN = None
_BACKEND_USED = 'unavailable'


# ── CPU (Numba @njit + prange) ───────────────────────────────────────────
def _build_cpu_kernels():
    """Compile and return the two CPU kernels + a ``_run`` dispatcher."""
    from numba import njit, prange

    @njit(parallel=True, fastmath=True, cache=True)
    def _rpm_hhns(input_raw, output_V, dt, itpl_val):
        T_orig = input_raw.shape[0]
        N_ch   = input_raw.shape[1]
        T_steps = (T_orig - 1) * itpl_val
        ddt = dt * 1000.0
        inv_itpl = 1.0 / itpl_val

        for ch in prange(N_ch):
            E_K   = -85.0
            E_Cl  = -30.0
            G_s   = 1.6
            G_dr  = 3.5
            G_Cl  = 0.006
            G_K   = 0.082
            G_nov = 3.0
            C     = 4.0

            a1, a2, a3 = 0.0001, 150.5, 819.8
            b1, b2, b3 = 279.6, 819.8, 4.51
            K2 = 30.0
            c_hill = 0.51
            gTRP = 1500.0
            ETRP = 0.0
            n_hill = 2.0

            V = 0.0
            sa, si    = 0.2184, 0.9653
            dra, dri  = 0.0117, 0.9998
            nov       = 0.0017
            x1, x2, x3 = 0.0, 0.0, 0.0

            output_V[0, ch] = V

            for i in range(T_steps):
                t = i * inv_itpl
                lo = int(t)
                hi = lo + 1
                if hi >= T_orig:
                    hi = T_orig - 1
                    lo = hi - 1
                frac = t - lo
                inp = input_raw[lo, ch] * (1.0 - frac) + input_raw[hi, ch] * frac

                dx1 = a1 * inp * (1.0 - x1) - b1 * x1
                dx2 = a2 * x1 * (1.0 - x2 - x3) - b2 * x2 - K2 * x2 * x3
                dx3 = -a3 * x3 + b3 * x2
                I_trp = (x2 ** n_hill / (x2 ** n_hill + c_hill ** n_hill)
                         * gTRP * (ETRP - V))

                x1 = max(min(x1 + dt * dx1, 1.0), 0.0)
                x2 = max(min(x2 + dt * dx2, 1.0), 0.0)
                x3 = max(min(x3 + dt * dx3, 1.0), 0.0)

                x_inf = (1.0 / (1.0 + math.exp((-23.7 - V) / 12.8))) ** (1.0 / 3.0)
                tau_x = 0.13 + 3.39 * math.exp(-(-73.0 - V) ** 2 / 400.0)
                dsa = (x_inf - sa) / tau_x

                x_inf = (0.9 / (1.0 + math.exp((-55.0 - V) / -3.9))
                         + 0.1 / (1.0 + math.exp((-74.8 - V) / -10.7)))
                tau_x = 113.0 * math.exp(-(-71.0 - V) ** 2 / 841.0)
                dsi = (x_inf - si) / tau_x

                x_inf = math.sqrt(1.0 / (1.0 + math.exp((-1.0 - V) / 9.1)))
                tau_x = 0.5 + 5.75 * math.exp(-(-25.0 - V) ** 2 / 1024.0)
                ddra = (x_inf - dra) / tau_x

                x_inf = 1.0 / (1.0 + math.exp((-25.7 - V) / -6.4))
                tau_x = 890.0
                ddri = (x_inf - dri) / tau_x

                x_inf = 1.0 / (1.0 + math.exp((-12.0 - V) / 11.0))
                tau_x = 3.0 + 166.0 * math.exp(-(-20.0 - V) ** 2 / 484.0)
                dnov = (x_inf - nov) / tau_x

                dV = (I_trp
                      - G_K * (V - E_K)
                      - G_Cl * (V - E_Cl)
                      - G_s * sa ** 3 * si * (V - E_K)
                      - G_dr * dra ** 2 * dri * (V - E_K)
                      - G_nov * nov * (V - E_K)) / C

                V = V + ddt * dV
                sa  = max(min(sa  + ddt * dsa,  1.0), 0.0)
                si  = max(min(si  + ddt * dsi,  1.0), 0.0)
                dra = max(min(dra + ddt * ddra, 1.0), 0.0)
                dri = max(min(dri + ddt * ddri, 1.0), 0.0)
                nov = max(min(nov + ddt * dnov, 1.0), 0.0)

                if (i + 1) % itpl_val == 0:
                    out_idx = (i + 1) // itpl_val
                    if out_idx < T_orig:
                        output_V[out_idx, ch] = V

    @njit(parallel=True, fastmath=True, cache=True)
    def _rpm_hh(input_raw, output_V, dt, itpl_val):
        T_orig = input_raw.shape[0]
        N_ch   = input_raw.shape[1]
        T_steps = (T_orig - 1) * itpl_val
        ddt = dt * 1000.0
        inv_itpl = 1.0 / itpl_val

        for ch in prange(N_ch):
            E_K, E_Na, E_L = -77.0, 50.0, -54.387
            gmax_K, gmax_Na, g_L = 36.0, 120.0, 0.3

            a1, a2, a3 = 0.0001, 150.5, 819.8
            b1, b2, b3 = 279.6, 819.8, 4.51
            K2 = 30.0
            c_hill = 0.51
            gTRP, ETRP = 1500.0, 0.0
            n_hill = 2.0

            V = 0.0
            m, n_gate, h = 0.0530, 0.3178, 0.5958
            x1, x2, x3 = 0.0, 0.0, 0.0

            output_V[0, ch] = V

            for i in range(T_steps):
                t = i * inv_itpl
                lo = int(t)
                hi = lo + 1
                if hi >= T_orig:
                    hi = T_orig - 1
                    lo = hi - 1
                frac = t - lo
                inp = input_raw[lo, ch] * (1.0 - frac) + input_raw[hi, ch] * frac

                dx1 = a1 * inp * (1.0 - x1) - b1 * x1
                dx2 = a2 * x1 * (1.0 - x2 - x3) - b2 * x2 - K2 * x2 * x3
                dx3 = -a3 * x3 + b3 * x2
                I_trp = (x2 ** n_hill / (x2 ** n_hill + c_hill ** n_hill)
                         * gTRP * (ETRP - V))

                x1 = max(min(x1 + dt * dx1, 1.0), 0.0)
                x2 = max(min(x2 + dt * dx2, 1.0), 0.0)
                x3 = max(min(x3 + dt * dx3, 1.0), 0.0)

                an = 0.01 * (V + 55.0) / (1.0 - math.exp(-0.1 * (V + 55.0)))
                bn = 0.125 * math.exp(-(V + 65.0) / 80.0)
                am = 0.1 * (V + 40.0) / (1.0 - math.exp(-0.1 * (V + 40.0)))
                bm = 4.0 * math.exp(-(V + 65.0) / 18.0)
                ah = 0.07 * math.exp(-0.05 * (V + 65.0))
                bh = 1.0 / (1.0 + math.exp(-0.1 * (V + 35.0)))

                dn = an * (1.0 - n_gate) - bn * n_gate
                dm = am * (1.0 - m) - bm * m
                dh = ah * (1.0 - h) - bh * h

                g_K  = gmax_K  * (n_gate ** 4.0)
                g_Na = gmax_Na * (m ** 3.0) * h
                I_K  = g_K  * (V - E_K)
                I_Na = g_Na * (V - E_Na)
                I_L  = g_L  * (V - E_L)
                dV = I_trp - I_K - I_Na - I_L

                V = V + ddt * dV
                n_gate = max(min(n_gate + ddt * dn, 1.0), 0.0)
                m      = max(min(m      + ddt * dm, 1.0), 0.0)
                h      = max(min(h      + ddt * dh, 1.0), 0.0)

                if (i + 1) % itpl_val == 0:
                    out_idx = (i + 1) // itpl_val
                    if out_idx < T_orig:
                        output_V[out_idx, ch] = V

    def _run(input_TN, dt, itpl_val, membrane):
        kernel = _rpm_hhns if membrane == 'hhns' else _rpm_hh
        arr = np.ascontiguousarray(input_TN, dtype=np.float32)
        out = np.zeros_like(arr)
        kernel(arr, out, np.float32(dt), int(itpl_val))
        return out

    return _run


# ── CuPy + numba.cuda (CUDA) ─────────────────────────────────────────────
def _build_cupy_kernels():
    """Compile and return CUDA kernels + a ``_run`` dispatcher."""
    import cupy as cp
    from numba import cuda

    @cuda.jit
    def _rpm_hhns_cuda(input_l, output_v, time_step, itpl_val_arr):
        ch_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ch_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        ch = ch_y * cuda.gridDim.x * cuda.blockDim.x + ch_x

        T_orig = input_l.shape[0]
        N_ch   = input_l.shape[1]
        if ch >= N_ch:
            return

        dt  = time_step
        ddt = dt * 1000.0
        itpl = itpl_val_arr[0]
        T_steps = (T_orig - 1) * itpl
        inv_itpl = 1.0 / itpl

        E_K, E_Cl = -85.0, -30.0
        G_s, G_dr, G_Cl, G_K, G_nov = 1.6, 3.5, 0.006, 0.082, 3.0
        C = 4.0

        a1, a2, a3 = 0.0001, 150.5, 819.8
        b1, b2, b3 = 279.6, 819.8, 4.51
        K2 = 30.0
        c_hill = 0.51
        gTRP, ETRP = 1500.0, 0.0

        V = 0.0
        sa, si = 0.2184, 0.9653
        dra_v, dri_v = 0.0117, 0.9998
        nov = 0.0017
        x1, x2, x3 = 0.0, 0.0, 0.0

        output_v[0, ch] = V

        for i in range(T_steps):
            t = i * inv_itpl
            lo = int(t)
            hi = lo + 1
            if hi >= T_orig:
                hi = T_orig - 1
                lo = hi - 1
            frac = t - lo
            inp = input_l[lo, ch] * (1.0 - frac) + input_l[hi, ch] * frac

            dx1 = a1 * inp * (1.0 - x1) - b1 * x1
            dx2 = a2 * x1 * (1.0 - x2 - x3) - b2 * x2 - K2 * x2 * x3
            dx3 = -a3 * x3 + b3 * x2
            x2_n = x2 * x2
            cc_n = c_hill * c_hill
            I_trp = x2_n / (x2_n + cc_n) * gTRP * (ETRP - V)

            x1 = max(min(x1 + dt * dx1, 1.0), 0.0)
            x2 = max(min(x2 + dt * dx2, 1.0), 0.0)
            x3 = max(min(x3 + dt * dx3, 1.0), 0.0)

            x_inf = (1.0 / (1.0 + math.exp((-23.7 - V) / 12.8))) ** (1.0 / 3.0)
            tau_x = 0.13 + 3.39 * math.exp(-(-73.0 - V) ** 2 / 400.0)
            dsa = (x_inf - sa) / tau_x

            x_inf = (0.9 / (1.0 + math.exp((-55.0 - V) / -3.9))
                     + 0.1 / (1.0 + math.exp((-74.8 - V) / -10.7)))
            tau_x = 113.0 * math.exp(-(-71.0 - V) ** 2 / 841.0)
            dsi = (x_inf - si) / tau_x

            x_inf = math.sqrt(1.0 / (1.0 + math.exp((-1.0 - V) / 9.1)))
            tau_x = 0.5 + 5.75 * math.exp(-(-25.0 - V) ** 2 / 1024.0)
            ddra = (x_inf - dra_v) / tau_x

            x_inf = 1.0 / (1.0 + math.exp((-25.7 - V) / -6.4))
            tau_x = 890.0
            ddri = (x_inf - dri_v) / tau_x

            x_inf = 1.0 / (1.0 + math.exp((-12.0 - V) / 11.0))
            tau_x = 3.0 + 166.0 * math.exp(-(-20.0 - V) ** 2 / 484.0)
            dnov = (x_inf - nov) / tau_x

            dV = (I_trp
                  - G_K  * (V - E_K)
                  - G_Cl * (V - E_Cl)
                  - G_s  * sa * sa * sa * si * (V - E_K)
                  - G_dr * dra_v * dra_v * dri_v * (V - E_K)
                  - G_nov * nov * (V - E_K)) / C

            V = V + ddt * dV
            sa    = max(min(sa    + ddt * dsa,  1.0), 0.0)
            si    = max(min(si    + ddt * dsi,  1.0), 0.0)
            dra_v = max(min(dra_v + ddt * ddra, 1.0), 0.0)
            dri_v = max(min(dri_v + ddt * ddri, 1.0), 0.0)
            nov   = max(min(nov   + ddt * dnov, 1.0), 0.0)

            if (i + 1) % itpl == 0:
                out_idx = (i + 1) // itpl
                if out_idx < T_orig:
                    output_v[out_idx, ch] = V

    @cuda.jit
    def _rpm_hh_cuda(input_l, output_v, time_step, itpl_val_arr):
        ch_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ch_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        ch = ch_y * cuda.gridDim.x * cuda.blockDim.x + ch_x

        T_orig = input_l.shape[0]
        N_ch   = input_l.shape[1]
        if ch >= N_ch:
            return

        dt  = time_step
        ddt = dt * 1000.0
        itpl = itpl_val_arr[0]
        T_steps = (T_orig - 1) * itpl
        inv_itpl = 1.0 / itpl

        E_K, E_Na, E_L = -77.0, 50.0, -54.387
        gmax_K, gmax_Na, g_L = 36.0, 120.0, 0.3

        a1, a2, a3 = 0.0001, 150.5, 819.8
        b1, b2, b3 = 279.6, 819.8, 4.51
        K2 = 30.0
        c_hill = 0.51
        gTRP, ETRP = 1500.0, 0.0

        V = 0.0
        m, n_gate, h = 0.0530, 0.3178, 0.5958
        x1, x2, x3 = 0.0, 0.0, 0.0

        output_v[0, ch] = V

        for i in range(T_steps):
            t = i * inv_itpl
            lo = int(t)
            hi = lo + 1
            if hi >= T_orig:
                hi = T_orig - 1
                lo = hi - 1
            frac = t - lo
            inp = input_l[lo, ch] * (1.0 - frac) + input_l[hi, ch] * frac

            dx1 = a1 * inp * (1.0 - x1) - b1 * x1
            dx2 = a2 * x1 * (1.0 - x2 - x3) - b2 * x2 - K2 * x2 * x3
            dx3 = -a3 * x3 + b3 * x2
            x2_n = x2 * x2
            cc_n = c_hill * c_hill
            I_trp = x2_n / (x2_n + cc_n) * gTRP * (ETRP - V)

            x1 = max(min(x1 + dt * dx1, 1.0), 0.0)
            x2 = max(min(x2 + dt * dx2, 1.0), 0.0)
            x3 = max(min(x3 + dt * dx3, 1.0), 0.0)

            an = 0.01 * (V + 55.0) / (1.0 - math.exp(-0.1 * (V + 55.0)))
            bn = 0.125 * math.exp(-(V + 65.0) / 80.0)
            am = 0.1 * (V + 40.0) / (1.0 - math.exp(-0.1 * (V + 40.0)))
            bm = 4.0 * math.exp(-(V + 65.0) / 18.0)
            ah = 0.07 * math.exp(-0.05 * (V + 65.0))
            bh = 1.0 / (1.0 + math.exp(-0.1 * (V + 35.0)))

            dn = an * (1.0 - n_gate) - bn * n_gate
            dm = am * (1.0 - m) - bm * m
            dh = ah * (1.0 - h) - bh * h

            g_K  = gmax_K  * (n_gate ** 4.0)
            g_Na = gmax_Na * (m ** 3.0) * h
            I_K  = g_K  * (V - E_K)
            I_Na = g_Na * (V - E_Na)
            I_L  = g_L  * (V - E_L)
            dV = I_trp - I_K - I_Na - I_L

            V = V + ddt * dV
            n_gate = max(min(n_gate + ddt * dn, 1.0), 0.0)
            m      = max(min(m      + ddt * dm, 1.0), 0.0)
            h      = max(min(h      + ddt * dh, 1.0), 0.0)

            if (i + 1) % itpl == 0:
                out_idx = (i + 1) // itpl
                if out_idx < T_orig:
                    output_v[out_idx, ch] = V

    def _run(input_TN, dt, itpl_val, membrane):
        kernel = _rpm_hhns_cuda if membrane == 'hhns' else _rpm_hh_cuda
        arr = cp.asarray(input_TN, dtype=cp.float32)
        out = cp.zeros_like(arr)

        threads_per_block = (16, 16)
        N_ch = arr.shape[1]
        blocks_x = int(np.ceil(N_ch / (threads_per_block[0] * threads_per_block[1])))
        blocks_per_grid = (blocks_x, 1)
        itpl_val_arr = cp.asarray([int(itpl_val)], dtype=cp.int32)

        kernel[blocks_per_grid, threads_per_block](
            arr, out, np.float32(dt), itpl_val_arr)
        return cp.asnumpy(out)

    return _run


# ── Metal (MLX) ──────────────────────────────────────────────────────────
def _build_metal_kernels():
    """Compile and return Metal kernels + a ``_run`` dispatcher."""
    import mlx.core as mx
    import mlx.core.fast as fast

    _HHNS_SOURCE = """
        uint ch = thread_position_in_grid.x;
        uint T_orig = input_raw_shape[0];
        uint N_ch   = input_raw_shape[1];
        if (ch >= N_ch) return;

        float dt_val = dt[0];
        int itpl = (int)itpl_val[0];
        int T_steps = ((int)T_orig - 1) * itpl;
        float ddt = dt_val * 1000.0f;
        float inv_itpl = 1.0f / (float)itpl;

        float E_K = -85.0f, E_Cl = -30.0f;
        float G_s = 1.6f, G_dr = 3.5f, G_Cl = 0.006f, G_K = 0.082f, G_nov = 3.0f;
        float C = 4.0f;

        float a1 = 0.0001f, a2 = 150.5f, a3 = 819.8f;
        float b1 = 279.6f,  b2 = 819.8f, b3 = 4.51f;
        float K2 = 30.0f;
        float cc = 0.51f;
        float gTRP = 1500.0f, ETRP = 0.0f;

        float V = 0.0f;
        float sa = 0.2184f, si = 0.9653f;
        float dra_v = 0.0117f, dri_v = 0.9998f;
        float nov = 0.0017f;
        float x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;

        output_V[ch] = V;

        for (int i = 0; i < T_steps; i++) {
            float t = (float)i * inv_itpl;
            int lo = (int)t;
            int hi = lo + 1;
            if (hi >= (int)T_orig) { hi = (int)T_orig - 1; lo = hi - 1; }
            float frac = t - (float)lo;
            float inp = input_raw[lo * N_ch + ch] * (1.0f - frac)
                       + input_raw[hi * N_ch + ch] * frac;

            float dx1 = a1 * inp * (1.0f - x1) - b1 * x1;
            float dx2 = a2 * x1 * (1.0f - x2 - x3) - b2 * x2 - K2 * x2 * x3;
            float dx3 = -a3 * x3 + b3 * x2;
            float x2_n = x2 * x2;
            float cc_n = cc * cc;
            float I_trp = x2_n / (x2_n + cc_n) * gTRP * (ETRP - V);

            x1 = clamp(x1 + dt_val * dx1, 0.0f, 1.0f);
            x2 = clamp(x2 + dt_val * dx2, 0.0f, 1.0f);
            x3 = clamp(x3 + dt_val * dx3, 0.0f, 1.0f);

            float x_inf = pow(1.0f / (1.0f + exp((-23.7f - V) / 12.8f)), 1.0f/3.0f);
            float tau_x = 0.13f + 3.39f * exp(-(-73.0f - V) * (-73.0f - V) / 400.0f);
            float dsa = (x_inf - sa) / tau_x;

            x_inf = 0.9f / (1.0f + exp((-55.0f - V) / -3.9f))
                  + 0.1f / (1.0f + exp((-74.8f - V) / -10.7f));
            tau_x = 113.0f * exp(-(-71.0f - V) * (-71.0f - V) / 841.0f);
            float dsi = (x_inf - si) / tau_x;

            x_inf = sqrt(1.0f / (1.0f + exp((-1.0f - V) / 9.1f)));
            tau_x = 0.5f + 5.75f * exp(-(-25.0f - V) * (-25.0f - V) / 1024.0f);
            float ddra = (x_inf - dra_v) / tau_x;

            x_inf = 1.0f / (1.0f + exp((-25.7f - V) / -6.4f));
            tau_x = 890.0f;
            float ddri = (x_inf - dri_v) / tau_x;

            x_inf = 1.0f / (1.0f + exp((-12.0f - V) / 11.0f));
            tau_x = 3.0f + 166.0f * exp(-(-20.0f - V) * (-20.0f - V) / 484.0f);
            float dnov = (x_inf - nov) / tau_x;

            float dV = (I_trp
                        - G_K  * (V - E_K)
                        - G_Cl * (V - E_Cl)
                        - G_s  * sa * sa * sa * si * (V - E_K)
                        - G_dr * dra_v * dra_v * dri_v * (V - E_K)
                        - G_nov * nov * (V - E_K)) / C;

            V = V + ddt * dV;
            sa    = clamp(sa    + ddt * dsa,  0.0f, 1.0f);
            si    = clamp(si    + ddt * dsi,  0.0f, 1.0f);
            dra_v = clamp(dra_v + ddt * ddra, 0.0f, 1.0f);
            dri_v = clamp(dri_v + ddt * ddri, 0.0f, 1.0f);
            nov   = clamp(nov   + ddt * dnov, 0.0f, 1.0f);

            if ((i + 1) % itpl == 0) {
                int out_idx = (i + 1) / itpl;
                if (out_idx < (int)T_orig) {
                    output_V[out_idx * N_ch + ch] = V;
                }
            }
        }
    """

    _HH_SOURCE = """
        uint ch = thread_position_in_grid.x;
        uint T_orig = input_raw_shape[0];
        uint N_ch   = input_raw_shape[1];
        if (ch >= N_ch) return;

        float dt_val = dt[0];
        int itpl = (int)itpl_val[0];
        int T_steps = ((int)T_orig - 1) * itpl;
        float ddt = dt_val * 1000.0f;
        float inv_itpl = 1.0f / (float)itpl;

        float E_K = -77.0f, E_Na = 50.0f, E_L = -54.387f;
        float gmax_K = 36.0f, gmax_Na = 120.0f, g_L = 0.3f;

        float a1 = 0.0001f, a2 = 150.5f, a3 = 819.8f;
        float b1 = 279.6f,  b2 = 819.8f, b3 = 4.51f;
        float K2 = 30.0f;
        float cc = 0.51f;
        float gTRP = 1500.0f, ETRP = 0.0f;

        float V = 0.0f;
        float m = 0.0530f, n_gate = 0.3178f, h = 0.5958f;
        float x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;

        output_V[ch] = V;

        for (int i = 0; i < T_steps; i++) {
            float t = (float)i * inv_itpl;
            int lo = (int)t;
            int hi = lo + 1;
            if (hi >= (int)T_orig) { hi = (int)T_orig - 1; lo = hi - 1; }
            float frac = t - (float)lo;
            float inp = input_raw[lo * N_ch + ch] * (1.0f - frac)
                       + input_raw[hi * N_ch + ch] * frac;

            float dx1 = a1 * inp * (1.0f - x1) - b1 * x1;
            float dx2 = a2 * x1 * (1.0f - x2 - x3) - b2 * x2 - K2 * x2 * x3;
            float dx3 = -a3 * x3 + b3 * x2;
            float x2_n = x2 * x2;
            float cc_n = cc * cc;
            float I_trp = x2_n / (x2_n + cc_n) * gTRP * (ETRP - V);

            x1 = clamp(x1 + dt_val * dx1, 0.0f, 1.0f);
            x2 = clamp(x2 + dt_val * dx2, 0.0f, 1.0f);
            x3 = clamp(x3 + dt_val * dx3, 0.0f, 1.0f);

            float an = 0.01f * (V + 55.0f) / (1.0f - exp(-0.1f * (V + 55.0f)));
            float bn = 0.125f * exp(-(V + 65.0f) / 80.0f);
            float am = 0.1f * (V + 40.0f) / (1.0f - exp(-0.1f * (V + 40.0f)));
            float bm = 4.0f * exp(-(V + 65.0f) / 18.0f);
            float ah = 0.07f * exp(-0.05f * (V + 65.0f));
            float bh = 1.0f / (1.0f + exp(-0.1f * (V + 35.0f)));

            float dn = an * (1.0f - n_gate) - bn * n_gate;
            float dm = am * (1.0f - m) - bm * m;
            float dh = ah * (1.0f - h) - bh * h;

            float g_K  = gmax_K  * n_gate * n_gate * n_gate * n_gate;
            float g_Na = gmax_Na * m * m * m * h;
            float I_K  = g_K  * (V - E_K);
            float I_Na = g_Na * (V - E_Na);
            float I_L  = g_L  * (V - E_L);
            float dV = I_trp - I_K - I_Na - I_L;

            V = V + ddt * dV;
            n_gate = clamp(n_gate + ddt * dn, 0.0f, 1.0f);
            m      = clamp(m      + ddt * dm, 0.0f, 1.0f);
            h      = clamp(h      + ddt * dh, 0.0f, 1.0f);

            if ((i + 1) % itpl == 0) {
                int out_idx = (i + 1) / itpl;
                if (out_idx < (int)T_orig) {
                    output_V[out_idx * N_ch + ch] = V;
                }
            }
        }
    """

    _kernel_hhns_metal = fast.metal_kernel(
        name="rpm_hhns",
        input_names=["input_raw", "dt", "itpl_val"],
        output_names=["output_V"],
        source=_HHNS_SOURCE,
    )

    _kernel_hh_metal = fast.metal_kernel(
        name="rpm_hh",
        input_names=["input_raw", "dt", "itpl_val"],
        output_names=["output_V"],
        source=_HH_SOURCE,
    )

    def _run(input_TN, dt, itpl_val, membrane):
        kernel = _kernel_hhns_metal if membrane == 'hhns' else _kernel_hh_metal
        arr = mx.array(np.asarray(input_TN, dtype=np.float32))
        T, N_ch = arr.shape
        dt_arr = mx.array([float(dt)], dtype=mx.float32)
        itpl_arr = mx.array([float(itpl_val)], dtype=mx.float32)

        threadgroup_size = 256
        grid_size = ((N_ch + threadgroup_size - 1) // threadgroup_size) * threadgroup_size

        outputs = kernel(
            inputs=[arr, dt_arr, itpl_arr],
            output_shapes=[(T, N_ch)],
            output_dtypes=[mx.float32],
            grid=(grid_size, 1, 1),
            threadgroup=(threadgroup_size, 1, 1),
        )
        mx.eval(outputs[0])
        return np.array(outputs[0])

    return _run


# ── Activate the matching backend ─────────────────────────────────────────
if BACKEND == 'cupy':
    try:
        _RUN = _build_cupy_kernels()
        _BACKEND_USED = 'cupy'
    except Exception:
        _RUN = None
elif BACKEND == 'mlx':
    try:
        _RUN = _build_metal_kernels()
        _BACKEND_USED = 'metal'
    except Exception:
        _RUN = None
else:
    # 'numpy' or anything else falls back to the CPU Numba path.
    try:
        _RUN = _build_cpu_kernels()
        _BACKEND_USED = 'cpu'
    except Exception:
        _RUN = None


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

class PhotoreceptorRetina:
    """RPM phototransduction + HH(NS) membrane on a (T, N_ch) light array.

    Parameters
    ----------
    intensity_inputs : array
        ``(T, N_ch)`` photon-rate trace per photoreceptor, or a
        ``(T, H, W)`` video. The latter is reshaped internally to
        ``(T, H*W)`` and reshaped back to ``(T, H, W)`` on output (this
        is what the original ``retina.py`` prototype expected).
    membrane : {'hhns', 'hh'}
        ``'hhns'`` — non-spiking 5-gate (default, matches the published
        Drosophila photoreceptor model). ``'hh'`` — classical 3-gate
        spiking Hodgkin-Huxley.
    amacrine, downsample : bool
        Accepted but currently no-ops in the consolidated module. Pass
        ``False`` (the default). The amacrine gain-control loop from
        the original CuPy/Metal prototypes is **not yet ported**; if
        you need it use the prototype files directly for now.
    norm_input : bool
        If ``True``, rescale the input to the ``2e5`` photon-rate range
        used by the model's tuning. Off by default — the FES pipeline
        already produces appropriately-scaled photon rates.

    Attributes
    ----------
    axon_terminal : np.ndarray
        Populated by :meth:`sim`. ``(T, N_ch)`` voltage trace, in mV
        (resting near ``-60`` to ``-80`` mV; depolarises toward
        ``0`` mV under high light).
    """

    supported_membrane = ('hh', 'hhns')

    def __init__(self,
                 intensity_inputs,
                 membrane: str = 'hhns',
                 amacrine: bool = False,
                 downsample: bool = False,
                 norm_input: bool = False):
        if membrane not in self.supported_membrane:
            raise ValueError(
                f"membrane must be one of {self.supported_membrane}; got {membrane!r}.")
        if amacrine:
            raise NotImplementedError(
                "amacrine gain control is not ported into the consolidated "
                "phototransduction module yet; use the prototype "
                "retina.py / retina_metal.py if you need it.")
        if downsample:
            raise NotImplementedError(
                "downsample is not supported in the consolidated module; "
                "pre-downsample your input or use the prototype retina.py.")

        arr = np.asarray(intensity_inputs, dtype=np.float32)
        self._orig_shape = arr.shape
        if arr.ndim == 2:
            self._input_TN = arr
        elif arr.ndim == 3:
            self._input_TN = arr.reshape(arr.shape[0], -1)
        else:
            raise ValueError(
                f"intensity_inputs must be (T, N_ch) or (T, H, W); "
                f"got shape {arr.shape}.")

        if norm_input:
            mn = float(self._input_TN.min())
            mx_ = float(self._input_TN.max())
            if mx_ > mn:
                self._input_TN = (2e5 * (self._input_TN - mn)
                                  / (mx_ - mn)).astype(np.float32)

        self.membrane = membrane
        self.amacrine = amacrine
        self.downsample = downsample
        self.norm_input = norm_input
        self.axon_terminal: Optional[np.ndarray] = None
        self.post_amacrine = None  # placeholder for future port

    def sim(self,
            itpl_val: int = 1000,
            itpl_order: int = 1,  # accepted for parity; only linear used
            dt: float = 1e-5,
            chunks: int = 0,
            amacrine_steps: int = 200) -> None:
        """Run the RPM + HH(NS) model and populate :attr:`axon_terminal`.

        Defaults (``itpl_val=1000``, ``dt=1e-5``) assume a **100 fps**
        input — one outer frame = ``itpl_val · dt`` = 10 ms. Scale
        ``itpl_val`` proportionally for other frame rates and keep
        ``dt`` near ``1e-5`` (the model is stiff in the RPM kinetics).

        ``chunks`` is accepted for parity with the CuPy prototype but
        currently treated as a no-op — the consolidated kernels use
        on-the-fly interpolation and don't need to slice the input to
        avoid OOM. Pass any value; only ``chunks > 1`` will print a
        note that chunking is no-op.
        """
        if _RUN is None:
            raise RuntimeError(
                f"phototransduction has no available backend (BACKEND={BACKEND!r}). "
                f"Install cupy+numba (CUDA), mlx (Metal), or numba (CPU).")
        if chunks and chunks > 1:
            # Kept for API parity; not needed since both fused kernels
            # avoid the upsampled-array memory blowup.
            pass

        out_TN = _RUN(self._input_TN, float(dt), int(itpl_val), self.membrane)

        if len(self._orig_shape) == 3:
            self.axon_terminal = out_TN.reshape(self._orig_shape)
        else:
            self.axon_terminal = out_TN


# Backwards-compatible alias matching the prototype scripts' import name.
Retina = PhotoreceptorRetina


# ─────────────────────────────────────────────────────────────────────────
# Functional helpers
# ─────────────────────────────────────────────────────────────────────────

def pr(res,
       *,
       membrane: str = 'hhns',
       itpl_val: int = 1000,
       dt: float = 1e-5) -> np.ndarray:
    """``(T, 6·N_cols)`` light intensity → ``(T, 6·N_cols)`` voltage.

    Direct wrapper around :class:`PhotoreceptorRetina`. The input is
    the raw FES output (one channel per R-cell, columns interleaved as
    ``[col0_R1, col0_R2, ..., col0_R6, col1_R1, ...]``). Defaults
    assume 100 fps; override ``itpl_val`` and ``dt`` proportionally for
    other frame rates so ``itpl_val · dt`` equals the frame period in
    seconds.
    """
    arr = np.squeeze(np.asarray(res))
    if arr.ndim != 2:
        raise ValueError(
            f"pr() expects (T, N_ch) light intensity; got shape {arr.shape} "
            f"after squeezing.")
    ret = PhotoreceptorRetina(
        arr, membrane=membrane,
        amacrine=False, downsample=False, norm_input=False)
    ret.sim(itpl_val=itpl_val, dt=dt)
    return ret.axon_terminal


def aggregate_pr(V_TN6: np.ndarray,
                 num_cols: int,
                 valid_idx: Optional[Iterable[int]] = None,
                 *,
                 mode: str = 'mean') -> np.ndarray:
    """Aggregate the 6 photoreceptors per superposition group into one trace.

    Calls :func:`flyeyesimulator.superposition.get_prs` for each target
    column, then combines the six returned traces via ``mode``.

    Parameters
    ----------
    V_TN6 : array of shape (T, 6·num_cols)
        Photoreceptor voltage produced by :func:`pr`.
    num_cols : int
        Number of cell columns (``= len(retina.vrfs) // 6``).
    valid_idx : iterable of int, optional
        Set of valid **R-cell channel** indices (into the 6·num_cols
        axis), used by :func:`get_prs` to substitute ``-80.0`` for
        unavailable channels. Defaults to ``range(6 · num_cols)``.
    mode : {'mean'}
        Aggregation mode. Only ``'mean'`` is supported in the first
        cut.

    Returns
    -------
    array of shape ``(T, num_cols)``
        One voltage trace per column.
    """
    if mode != 'mean':
        raise ValueError(
            f"aggregate_pr supports mode='mean' only for now; got {mode!r}.")

    V = np.asarray(V_TN6)
    if V.ndim != 2:
        raise ValueError(
            f"V_TN6 must be 2D (T, 6·num_cols); got shape {V.shape}.")
    if V.shape[1] != 6 * num_cols:
        raise ValueError(
            f"V_TN6 has {V.shape[1]} channels but num_cols={num_cols} "
            f"implies 6·num_cols={6 * num_cols}.")

    if valid_idx is None:
        valid_set: set = set(range(V.shape[1]))
    else:
        valid_set = set(int(i) for i in valid_idx)

    out = np.empty((V.shape[0], num_cols), dtype=V.dtype)
    for c in range(num_cols):
        six = get_prs(c, V, valid_set)            # (T, 6)
        out[:, c] = six.mean(axis=1)
    return out


def fes_to_pr_voltage(res,
                      num_cols: int,
                      valid_idx: Optional[Iterable[int]] = None,
                      *,
                      membrane: str = 'hhns',
                      itpl_val: int = 1000,
                      dt: float = 1e-5,
                      aggregate: Optional[str] = 'mean') -> np.ndarray:
    """One-shot pipeline: FES ``(T, 6·N_cols)`` light → per-column voltage.

    Calls :func:`pr` then :func:`aggregate_pr`. Pass ``aggregate=None``
    to skip aggregation and get the ``(T, 6·N_cols)`` voltage array
    directly.
    """
    V_TN6 = pr(res, membrane=membrane, itpl_val=itpl_val, dt=dt)
    if aggregate is None:
        return V_TN6
    return aggregate_pr(V_TN6, num_cols, valid_idx, mode=aggregate)
