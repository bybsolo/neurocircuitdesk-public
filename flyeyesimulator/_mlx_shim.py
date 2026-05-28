"""
flyeyesimulator._mlx_shim
-------------------------
Thin compatibility layer on top of ``mlx.core``.  Re-exports every symbol the
flyeyesimulator package uses under names that match numpy / cupy idioms, and
provides one non-trivial function — ``mlx_map_coordinates`` — that mirrors
``scipy.ndimage.map_coordinates(video, coords, order=1, mode='nearest')``
for the specific use-case in ``screen.py``.
"""
import math

import mlx.core as _mx
import numpy as np

# ── Re-export every mlx.core symbol used by the simulators ──────────────
from mlx.core import (
    array, asarray, zeros, ones, full,
    arange, linspace, repeat, tile,
    exp, log, sqrt, power,
    sin, cos, arcsin, arctan2,
    maximum, minimum, clip, where,
    stack, concatenate,
    sum, mean,
    meshgrid,
    float32, float64,
    int32,
    eval,
)

# mlx uses ``radians`` / ``degrees`` instead of ``deg2rad`` / ``rad2deg``
deg2rad = _mx.radians
rad2deg = _mx.degrees

# ── numpy / cupy idiom aliases ──────────────────────────────────────────
newaxis = None
pi = math.pi
bool_ = _mx.bool_
ndarray = _mx.array  # for isinstance checks in to_numpy


def empty(shape, dtype=None):
    """MLX has no ``empty``; use ``zeros`` with identical contract."""
    return _mx.zeros(shape, dtype=dtype) if dtype is not None else _mx.zeros(shape)


def exp2(x):
    return _mx.power(2.0, x)


class linalg:
    @staticmethod
    def norm(x, axis=None, keepdims=False):
        return _mx.sqrt(_mx.sum(x * x, axis=axis, keepdims=keepdims))


def dot(a, b):
    """``np.dot``-style for 1-D / 2-D arguments — wraps ``mx.matmul``."""
    return _mx.matmul(a, b)


# ── map_coordinates bilinear shim ───────────────────────────────────────

def mlx_map_coordinates(video, coordinates, order=1, mode='nearest'):
    """Bilinear interpolation matching ``scipy.ndimage.map_coordinates``
    for the ``screen.py`` use-case.

    ``coordinates`` has shape ``(3, K)`` where row 0 is the (integer-valued)
    frame index and rows 1–2 are fractional ``(y, x)`` positions.
    ``mode='nearest'`` is implemented by clipping indices to valid bounds.

    Parameters
    ----------
    video : mx.array, shape (T, H, W)
    coordinates : mx.array, shape (3, K)
    order : must be 1 (bilinear)
    mode : must be 'nearest'

    Returns
    -------
    mx.array of shape (K,)
    """
    if order != 1:
        raise NotImplementedError("Only order=1 (bilinear) is supported.")
    if mode != 'nearest':
        raise NotImplementedError("Only mode='nearest' is supported.")

    T, H, W = video.shape
    t_coord = coordinates[0]
    y_coord = coordinates[1]
    x_coord = coordinates[2]

    t_idx = t_coord.astype(_mx.int32)

    y_floor = _mx.floor(y_coord).astype(_mx.int32)
    x_floor = _mx.floor(x_coord).astype(_mx.int32)
    y_ceil = y_floor + 1
    x_ceil = x_floor + 1

    # Bilinear weights
    wy = y_coord - y_floor.astype(y_coord.dtype)
    wx = x_coord - x_floor.astype(x_coord.dtype)

    # 'nearest' ≡ clip to image bounds
    y0 = _mx.clip(y_floor, 0, H - 1)
    y1 = _mx.clip(y_ceil, 0, H - 1)
    x0 = _mx.clip(x_floor, 0, W - 1)
    x1 = _mx.clip(x_ceil, 0, W - 1)

    # Gather four corner samples
    v00 = video[t_idx, y0, x0]
    v01 = video[t_idx, y0, x1]
    v10 = video[t_idx, y1, x0]
    v11 = video[t_idx, y1, x1]

    # Bilinear combination
    v0 = v00 * (1.0 - wx) + v01 * wx
    v1 = v10 * (1.0 - wx) + v11 * wx
    return v0 * (1.0 - wy) + v1 * wy
