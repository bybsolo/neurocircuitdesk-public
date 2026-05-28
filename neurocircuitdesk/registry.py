"""
registry.py
-----------
Discoverable, string-addressable registries for MicroCircuit templates and
unified algorithms. Enables circuit specifications and the chat-driven app
to refer to building blocks by name instead of passing Python callables.

Two registries live here:

- ``_TEMPLATES``     name -> (callable, metadata)
- ``_ALGORITHMS``    name -> (callable, metadata)

Reverse lookup (callable -> name) is also supported so legacy callers that
still pass bare functions to ``Canvas.add_microcircuit_columnar`` can be
auto-resolved when the same function was previously registered.

Collision policy (qualname-aware)
---------------------------------
All three ``register_*`` paths share the same rule when a name is already
taken:

1. ``stored_fn is fn``                                 → idempotent no-op
   (Python module re-import — the function object hasn't changed).
2. Same ``__qualname__`` and same ``__module__``       → ``DeprecationWarning``,
   then overwrite (notebook re-execution / ``importlib.reload``;
   the source location is the same, only the function object differs).
3. Different ``__qualname__`` or ``__module__``        → ``ValueError``
   (genuine cross-module shadowing — refuse to silently shadow another
   module's registration).

This keeps notebook re-execution friendly without losing protection
against accidental shadowing.

Public API
----------
template(name, *, category, description, ...)   decorator for templates
register_algorithm(fn, *, name=None, ...)       used by @unified_algorithm
motif(name, *, description, ...)                decorator for spec functions
get_template(name) / get_algorithm(name) / get_motif(name)
list_templates() / list_algorithms() / list_motifs()
template_name_of(fn) / algorithm_name_of(fn)    reverse lookup
"""
from __future__ import annotations
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Any


_TEMPLATES: Dict[str, Tuple[Callable, Dict[str, Any]]] = {}
_ALGORITHMS: Dict[str, Tuple[Callable, Dict[str, Any]]] = {}
_MOTIFS: Dict[str, Tuple[Callable, Dict[str, Any]]] = {}

_TEMPLATE_REVERSE: Dict[int, str] = {}   # id(fn) -> name
_ALGORITHM_REVERSE: Dict[int, str] = {}  # id(fn) -> name


def _resolve_collision(
    registry: Dict[str, Tuple[Callable, Dict[str, Any]]],
    key: str,
    fn: Callable,
    kind: str,
) -> str:
    """Apply the qualname-aware collision policy.

    Returns one of: ``'fresh'`` (no prior entry — caller proceeds with
    registration), ``'noop'`` (caller should return early without
    re-registering), or ``'overwrite'`` (caller proceeds and clobbers the
    prior entry, after we emit a DeprecationWarning here). Raises
    ``ValueError`` for genuine cross-module shadowing.

    ``kind`` is one of ``'algorithm'`` / ``'template'`` / ``'motif'`` and
    is used only for error / warning messages.
    """
    existing = registry.get(key)
    if existing is None:
        return 'fresh'

    stored_fn = existing[0]
    if stored_fn is fn:
        return 'noop'                                 # identical re-import

    same_location = (
        getattr(stored_fn, '__qualname__', None) == getattr(fn, '__qualname__', None)
        and getattr(stored_fn, '__module__', None) == getattr(fn, '__module__', None)
    )
    if same_location:
        warnings.warn(
            f"re-registering {kind} {key!r} from "
            f"{getattr(fn, '__module__', '<unknown>')} "
            "(presumed notebook re-execution / hot reload)",
            DeprecationWarning,
            stacklevel=4,
        )
        return 'overwrite'

    raise ValueError(
        f"{kind} name {key!r} is already registered to a function in "
        f"{getattr(stored_fn, '__module__', '<unknown>')}; "
        "rename one of them.",
    )


# ── Templates ──────────────────────────────────────────────────────────────

def template(
    name: str,
    *,
    category: str,
    description: str = "",
    default_z: float = 0.0,
    requires_neighborhood: bool = False,
    default_num_rings: Optional[int] = None,
    params_schema: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable], Callable]:
    """Register an MC template under a stable string name.

    Parameters
    ----------
    name : str
        Unique registry key. Convention: ``{CMC,iCMC}_<biology>_<variant>``.
    category : {'columnar', 'intercolumnar'}
        Determines which Canvas adder is used (``add_microcircuit_columnar``
        vs ``add_microcircuit_intercolumnar``). User-facing docs render
        these as **CMC** / **iCMC**.
    description : str
        Human-readable summary shown to discovery callers (incl. the LLM).
    default_z : float
        Suggested z-plane for this MC type in retinotopic stacking.
    requires_neighborhood : bool
        True if the template's signature is ``(mc, neighborhood, **kwargs)``;
        the Canvas will supply the neighborhood dict.
    default_num_rings : int, optional
        For iCMC templates: the natural neighbourhood size used by
        ``mc_lib.preview(...)`` when the caller doesn't supply an explicit
        ``neighborhood``. Matches the ring count the demos pass to
        ``calc_mimo_centers(num_rings=...)`` for this template type.
        ``None`` (default) for CMC.
    params_schema : dict, optional
        Free-form schema describing template kwargs. Used only for LLM
        discovery — not validated here.
    """
    if category not in ('columnar', 'intercolumnar'):
        raise ValueError(
            f"template '{name}' has invalid category {category!r}; "
            "must be 'columnar' or 'intercolumnar'."
        )

    def deco(fn: Callable) -> Callable:
        action = _resolve_collision(_TEMPLATES, name, fn, 'template')
        if action == 'noop':
            return fn
        meta = dict(
            category=category,
            description=description,
            default_z=default_z,
            requires_neighborhood=requires_neighborhood,
            default_num_rings=default_num_rings,
            params_schema=params_schema or {},
        )
        _TEMPLATES[name] = (fn, meta)
        _TEMPLATE_REVERSE[id(fn)] = name
        return fn

    return deco


def get_template(name: str) -> Callable:
    """Return the template callable for ``name``."""
    if name not in _TEMPLATES:
        raise KeyError(
            f"unknown template {name!r}. Known: {sorted(_TEMPLATES.keys())}"
        )
    return _TEMPLATES[name][0]


def list_templates() -> List[Dict[str, Any]]:
    """Return template catalog as ``[{name, category, description, ...}, ...]``.

    No callables included — safe to JSON-serialise.
    """
    return [{"name": n, **meta} for n, (_, meta) in sorted(_TEMPLATES.items())]


def template_name_of(fn: Callable) -> Optional[str]:
    """Reverse lookup: return the registered name for ``fn`` or None."""
    return _TEMPLATE_REVERSE.get(id(fn))


# ── Algorithms ─────────────────────────────────────────────────────────────

def register_algorithm(
    fn: Callable,
    *,
    name: Optional[str] = None,
    signature: str = "stateless",
    ports: Optional[Dict[str, Any]] = None,
    description: str = "",
) -> Callable:
    """Register an algorithm. Used by @unified_algorithm; rarely called directly.

    If ``name`` is None, falls back to ``fn.__name__``.
    """
    key = name or fn.__name__
    action = _resolve_collision(_ALGORITHMS, key, fn, 'algorithm')
    if action == 'noop':
        return fn
    meta = dict(signature=signature, ports=ports or {}, description=description)
    _ALGORITHMS[key] = (fn, meta)
    _ALGORITHM_REVERSE[id(fn)] = key
    return fn


def get_algorithm(name: str) -> Callable:
    if name not in _ALGORITHMS:
        raise KeyError(
            f"unknown algorithm {name!r}. Known: {sorted(_ALGORITHMS.keys())}"
        )
    return _ALGORITHMS[name][0]


def list_algorithms() -> List[Dict[str, Any]]:
    return [{"name": n, **meta} for n, (_, meta) in sorted(_ALGORITHMS.items())]


def algorithm_name_of(fn: Callable) -> Optional[str]:
    return _ALGORITHM_REVERSE.get(id(fn))


# ── Motifs ─────────────────────────────────────────────────────────────────
# A motif is a function returning a spec dict (NOT a Canvas mutation).
# Motifs are scaffolds the LLM can load + modify with primitives; they are
# also reference documentation for "how to wire this kind of circuit."

def motif(
    name: str,
    *,
    description: str,
    params_schema: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable], Callable]:
    """Register a motif (spec-producing function) under ``name``.

    The decorated function MUST return a JSON-able spec dict suitable for
    ``Canvas.from_spec``. It must NOT mutate any Canvas — produce-only.

    Example::

        @motif('motion', description='Standard motion-detection pipeline.',
               params_schema={'n_cols': {'type': 'int', 'default': 547}})
        def motion_pipeline_spec(n_cols=547): ...
    """
    def deco(fn: Callable) -> Callable:
        action = _resolve_collision(_MOTIFS, name, fn, 'motif')
        if action == 'noop':
            return fn
        meta = dict(description=description, params_schema=params_schema or {})
        _MOTIFS[name] = (fn, meta)
        return fn
    return deco


def get_motif(name: str) -> Callable:
    if name not in _MOTIFS:
        raise KeyError(
            f"unknown motif {name!r}. Known: {sorted(_MOTIFS.keys())}"
        )
    return _MOTIFS[name][0]


def list_motifs() -> List[Dict[str, Any]]:
    return [{"name": n, **meta} for n, (_, meta) in sorted(_MOTIFS.items())]
