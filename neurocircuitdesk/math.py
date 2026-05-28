"""
Backend-agnostic math façade for unified-signature algorithms.

Numpy array methods like ``.sum(axis=-1)``, ``.mean(axis=-1)``, and plain
arithmetic operators work uniformly on both numpy arrays (scalar engine)
and ``mlx.core.array`` (batched engine). Free functions do not — numpy
has ``np.where`` / ``np.clip`` and mlx has ``mx.where`` / ``mx.clip``,
and they are not interchangeable.

This module provides a thin dispatching layer that picks the right
backend based on the array type of its first array-like argument, so
unified algorithms can write ``from neurocircuitdesk import math as nm``
and then ``nm.where(mask, a, b)`` without branching on backend.

Only the helpers actually needed by the shipped algorithms are
implemented. Extend as new algorithms require more primitives.
"""

from typing import Any
import numpy as np


def _is_mx(x: Any) -> bool:
    """Return True if ``x`` is an mlx.core.array (lazy-import)."""
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return isinstance(x, mx.array)


def _pick(*args: Any):
    """Pick the mlx backend if any arg is an mx.array, else numpy."""
    for a in args:
        if _is_mx(a):
            import mlx.core as mx
            return mx
    return np


def where(cond, a, b):
    """Backend-agnostic ``where(cond, a, b)``."""
    return _pick(cond, a, b).where(cond, a, b)


def clip(x, lo, hi):
    """Backend-agnostic ``clip(x, lo, hi)``. Either bound may be ``None``."""
    return _pick(x).clip(x, lo, hi)


def exp(x):
    return _pick(x).exp(x)


def log(x):
    return _pick(x).log(x)


def sqrt(x):
    return _pick(x).sqrt(x)


def maximum(a, b):
    return _pick(a, b).maximum(a, b)


def minimum(a, b):
    return _pick(a, b).minimum(a, b)


def stack(arrays, axis=0):
    return _pick(*arrays).stack(list(arrays), axis=axis)


def concatenate(arrays, axis=0):
    return _pick(*arrays).concatenate(list(arrays), axis=axis)


def power(base, exponent):
    return _pick(base, exponent).power(base, exponent)


def sign(x):
    return _pick(x).sign(x)


def abs(x):  # noqa: A001 — intentional shadow for API symmetry
    return _pick(x).abs(x)


def zeros_like(x):
    return _pick(x).zeros_like(x)


def ones_like(x):
    return _pick(x).ones_like(x)
