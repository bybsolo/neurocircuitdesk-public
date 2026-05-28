"""
Backend-agnostic state helpers for unified-signature algorithms.

These helpers dispatch polymorphically on the buffer type, so the same
algorithm body can run in scalar mode (with Python ``collections.deque``
buffers) and in batched MLX mode (with rolling ``mx.array`` buffers).

Usage pattern in a unified algorithm::

    from neurocircuitdesk import state_utils as su

    def my_algo(inputs, params, state):
        x = inputs['input']
        buf = su.ring_buffer_push(state, 'history', x, maxlen=params['N'] + 1)
        if su.ring_buffer_len(buf, state, 'history') < params['N'] + 1:
            return {'output': x * 0}, state
        delayed = su.ring_buffer_get(buf, 0)   # oldest entry
        return {'output': delayed}, state

Both backends pre-fill the buffer with zeros and use a sidecar state key
``f'{key}__fill'`` to track how many entries have been pushed so far. This
guarantees identical warmup semantics: ``buf[0]`` returns zero in both
backends until the buffer has been filled, and ``ring_buffer_len`` returns
the same count regardless of backend.
"""

from collections import deque
from typing import Any, Dict, Optional
import numpy as np


def _is_mx_array(value: Any) -> bool:
    """True if ``value`` is an ``mlx.core.array``. Lazy-imports mlx."""
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return isinstance(value, mx.array)


def ring_buffer_push(state: Dict[str, Any], key: str, value: Any, maxlen: int) -> Any:
    """Append ``value`` to the ring buffer at ``state[key]``.

    On the first call the buffer is allocated based on the type of
    ``value``:

    - **scalar/numpy**: a ``deque`` of length ``maxlen``, pre-filled with
      zeros matching the shape of ``value``.
    - **mlx**: a zero-initialised ``mx.array`` of shape
      ``(maxlen,) + value.shape``.

    Both backends use a sidecar counter ``state[f'{key}__fill']`` to track
    how many entries have been pushed so far (capped at ``maxlen``). This
    ensures ``ring_buffer_len`` and ``ring_buffer_get`` have identical
    semantics regardless of backend.

    Returns the (possibly newly allocated) buffer. The caller does not
    need to write it back to ``state`` — this function does that.
    """
    buf = state.get(key)
    fill_key = f'{key}__fill'

    if buf is None:
        if _is_mx_array(value):
            import mlx.core as mx
            shape = (maxlen,) + tuple(value.shape)
            buf = mx.zeros(shape, dtype=value.dtype)
        else:
            # Pre-fill with zeros matching the value's shape/type so that
            # buf[0] during warmup returns zero — identical to the batched
            # mx.array branch which is also zero-initialised.
            if isinstance(value, np.ndarray):
                zero = np.zeros_like(value)
            else:
                zero = type(value)(0) if isinstance(value, (int, float)) else 0.0
            buf = deque([zero] * maxlen, maxlen=maxlen)
        state[key] = buf
        state[fill_key] = 0

    if isinstance(buf, deque):
        buf.append(value)
        state[fill_key] = min(maxlen, state.get(fill_key, 0) + 1)
        return buf

    # mx.array branch: shift-left and assign the new tail.
    import mlx.core as mx
    buf = mx.concatenate([buf[1:], value[None]], axis=0)
    state[key] = buf
    state[fill_key] = min(maxlen, state.get(fill_key, 0) + 1)
    return buf


def ring_buffer_get(buf: Any, idx: int) -> Any:
    """Index into a ring buffer returned by :func:`ring_buffer_push`.

    ``idx=0`` is the oldest entry, ``idx=-1`` is the newest. Works for
    both ``deque`` and ``mx.array`` buffers because both support the
    same integer-indexing semantics.
    """
    return buf[idx]


def ring_buffer_len(buf: Any, state: Optional[Dict[str, Any]] = None,
                    key: Optional[str] = None) -> int:
    """Return the number of entries pushed so far (capped at ``maxlen``).

    Uses the sidecar counter ``state[f'{key}__fill']`` written by
    :func:`ring_buffer_push`. Both deque and mx.array backends track fill
    identically via this counter.

    Falls back to ``len(buf)`` (deque) or ``buf.shape[0]`` (mx.array) if
    the sidecar is unavailable, but callers should always pass ``state``
    and ``key`` for correct warmup-aware length.
    """
    if state is not None and key is not None:
        fill = state.get(f'{key}__fill')
        if fill is not None:
            return int(fill)
    # Fallback for callers that don't pass state/key.
    if isinstance(buf, deque):
        return len(buf)
    return int(buf.shape[0])
