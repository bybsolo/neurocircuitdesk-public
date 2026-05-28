"""
spec.py
-------
Canvas specification: a canonical JSON-able description of a circuit.

The actual serialise/deserialise logic lives on ``Canvas`` itself
(``to_spec`` / ``from_spec``); this module owns the version constant,
the schema validator, and the migration hook for future schema bumps.

Usage::

    from neurocircuitdesk.canvas import Canvas
    from neurocircuitdesk.spec import SPEC_VERSION, validate_spec, save_spec, load_spec

    spec = cv.to_spec()
    validate_spec(spec)
    save_spec(spec, 'my_circuit.json')

    spec2 = load_spec('my_circuit.json')
    cv2   = Canvas.from_spec(spec2)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Union


SPEC_VERSION = 1


_REQUIRED_TOP = ('version', 'canvas', 'mc_types', 'algorithms', 'wirings')
_REQUIRED_CANVAS = ('n_cols',)
_VALID_CATEGORIES = ('columnar', 'intercolumnar')


def validate_spec(spec: Dict[str, Any]) -> None:
    """Raise ``ValueError`` with a useful message if ``spec`` is malformed.

    Validates structure, not semantics — does not check that template /
    algorithm names exist in the registry, since that's a runtime concern
    handled by ``Canvas.from_spec``.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"spec must be a dict, got {type(spec).__name__}")

    for key in _REQUIRED_TOP:
        if key not in spec:
            raise ValueError(f"spec missing required key {key!r}")

    if spec['version'] != SPEC_VERSION:
        raise ValueError(
            f"spec version {spec['version']} != current {SPEC_VERSION}; "
            f"call migrate_spec() first."
        )

    canvas = spec['canvas']
    if not isinstance(canvas, dict):
        raise ValueError(f"spec['canvas'] must be a dict, got {type(canvas).__name__}")
    for key in _REQUIRED_CANVAS:
        if key not in canvas:
            raise ValueError(f"spec['canvas'] missing required key {key!r}")

    if not isinstance(spec['mc_types'], list):
        raise ValueError("spec['mc_types'] must be a list")
    for i, td in enumerate(spec['mc_types']):
        for key in ('name', 'category', 'z', 'template'):
            if key not in td:
                raise ValueError(f"spec['mc_types'][{i}] missing key {key!r}")
        if td['category'] not in _VALID_CATEGORIES:
            raise ValueError(
                f"spec['mc_types'][{i}].category = {td['category']!r}; "
                f"must be one of {_VALID_CATEGORIES}"
            )

    if not isinstance(spec['algorithms'], list):
        raise ValueError("spec['algorithms'] must be a list")
    for i, ad in enumerate(spec['algorithms']):
        for key in ('mc_type', 'block', 'algo'):
            if key not in ad:
                raise ValueError(f"spec['algorithms'][{i}] missing key {key!r}")

    if not isinstance(spec['wirings'], list):
        raise ValueError("spec['wirings'] must be a list")


def migrate_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade an older-version spec to the current ``SPEC_VERSION``.

    Identity at v1. Add migration steps here as the schema evolves.
    """
    v = spec.get('version', 0)
    if v == SPEC_VERSION:
        return spec
    raise ValueError(
        f"don't know how to migrate spec from version {v} to {SPEC_VERSION}"
    )


def save_spec(spec: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write ``spec`` to ``path`` as pretty-printed JSON."""
    validate_spec(spec)
    Path(path).write_text(json.dumps(spec, indent=2, default=str))


def load_spec(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate a spec from JSON."""
    spec = json.loads(Path(path).read_text())
    spec = migrate_spec(spec)
    validate_spec(spec)
    return spec
