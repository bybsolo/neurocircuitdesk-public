"""
flyeyesimulator.backend
-----------------------
Array backend selector.  Three branches, in priority order:

  1. CuPy  (with CUDA device probe) — preferred on Linux / CUDA boxes.
  2. MLX                             — preferred on Apple Silicon.
  3. NumPy + SciPy                   — fallback everywhere else.

Public symbols:
    xp              array module (cupy, _mlx_shim, or numpy)
    xp_ndimage      namespace with at least ``map_coordinates(...)``
    to_numpy(a)     convert any backend array to a host numpy array
    free_memory()   release cached device memory (no-op on numpy)
    BACKEND         'cupy' | 'mlx' | 'numpy'
"""
import numpy as np


def _try_cupy():
    import cupy as _cp
    from cupyx.scipy import ndimage as _cp_ndimage
    # Probe for a real CUDA device — degrades gracefully when cupy is
    # installed but no GPU is present (CI, Docker without --gpus, etc.).
    _cp.cuda.runtime.getDevice()
    return _cp, _cp_ndimage


def _try_mlx():
    from . import _mlx_shim
    return _mlx_shim


# ── Resolve backend ─────────────────────────────────────────────────────
xp = xp_ndimage = None
BACKEND = ''

try:
    _cp, _cp_ndimage = _try_cupy()
    xp = _cp
    xp_ndimage = _cp_ndimage
    BACKEND = 'cupy'
except Exception:
    try:
        _shim = _try_mlx()
        xp = _shim

        class _MlxNdimage:
            map_coordinates = staticmethod(_shim.mlx_map_coordinates)

        xp_ndimage = _MlxNdimage()
        BACKEND = 'mlx'
    except ImportError:
        from scipy import ndimage as _scipy_ndimage
        xp = np
        xp_ndimage = _scipy_ndimage
        BACKEND = 'numpy'


# ── Helpers ──────────────────────────────────────────────────────────────

def to_numpy(a):
    """Return ``a`` as a host numpy array, regardless of backend."""
    if BACKEND == 'cupy' and isinstance(a, xp.ndarray):
        return xp.asnumpy(a)
    if BACKEND == 'mlx':
        # np.array() on an mlx.core.array forces evaluation and copies out.
        return np.array(a, copy=True)
    return np.asarray(a)


def free_memory():
    """Release cached device memory pools. No-op on numpy backend."""
    if BACKEND == 'cupy':
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()
    elif BACKEND == 'mlx':
        try:
            import mlx.core as _mx
            if hasattr(_mx, 'clear_cache'):
                _mx.clear_cache()
            elif hasattr(_mx, 'metal') and hasattr(_mx.metal, 'clear_cache'):
                _mx.metal.clear_cache()
        except (ImportError, AttributeError):
            pass
