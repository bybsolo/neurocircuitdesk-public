"""
Phase 1 acceptance gate.

Builds a 50-column PR-only canvas, round-trips through ``to_spec`` /
``from_spec``, and asserts that the second spec matches the first
byte-for-byte. Also exercises ``summary()`` and ``save_spec`` /
``load_spec``.

Run from the core repo root::

    pytest tests/test_spec_roundtrip.py -v

or directly::

    python tests/test_spec_roundtrip.py
"""
from __future__ import annotations
import json
import os
import sys
import tempfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from neurocircuitdesk import library  # noqa: F401 — registers templates/algos
from neurocircuitdesk.canvas import Canvas
from neurocircuitdesk.library.optics import DNP_PARAMS
from neurocircuitdesk.registry import list_templates, list_algorithms
from neurocircuitdesk.spec import (
    SPEC_VERSION, validate_spec, save_spec, load_spec,
)

COL_JSON = os.path.join(
    _REPO_ROOT, 'neurocircuitdesk', 'libs', 'jsons', 'hexcol_l1m3_new_578.json'
)
GRAPH_JSON = os.path.join(
    _REPO_ROOT, 'neurocircuitdesk', 'libs', 'jsons', 'hex_grid_graph.json'
)
N_COLS = 50


def build_pr_only_canvas() -> Canvas:
    cv = Canvas(
        w=400, h=400,
        col_json_path=COL_JSON,
        interconnect_json_path=GRAPH_JSON,
    )
    cv.add_mc_type('PR_col')
    for col_idx in range(N_COLS):
        cv.add_microcircuit_columnar(
            col_idx=col_idx, z=1.3, mc_type='PR_col', template='pr_dnp',
        )
    cv.bind_algorithm('PR_col', 'T1', 'poly2_T1', DNP_PARAMS['T1'])
    cv.bind_algorithm('PR_col', 'T2', 'poly2_T2', DNP_PARAMS['T2'])
    return cv


def test_registry_populated():
    tpl_names = {t['name'] for t in list_templates()}
    algo_names = {a['name'] for a in list_algorithms()}
    assert 'pr_dnp' in tpl_names, tpl_names
    assert 'poly2_T1' in algo_names and 'poly2_T2' in algo_names, algo_names


def test_spec_validates():
    cv = build_pr_only_canvas()
    spec = cv.to_spec()

    assert spec['version'] == SPEC_VERSION
    assert spec['canvas']['n_cols'] == N_COLS
    assert len(spec['mc_types']) == 1
    assert spec['mc_types'][0]['name'] == 'PR_col'
    assert spec['mc_types'][0]['template'] == 'pr_dnp'
    assert len(spec['algorithms']) == 2
    validate_spec(spec)


def test_spec_roundtrip_identity():
    cv1 = build_pr_only_canvas()
    spec1 = cv1.to_spec()

    cv2 = Canvas.from_spec(spec1)
    spec2 = cv2.to_spec()

    assert spec1 == spec2, (
        "round-trip changed the spec:\n"
        f"--- before ---\n{json.dumps(spec1, indent=2, default=str)}\n"
        f"--- after ---\n{json.dumps(spec2, indent=2, default=str)}"
    )

    assert len(cv2.microcircuits) == N_COLS
    assert len(cv2.mc_types['PR_col']) == N_COLS


def test_disk_roundtrip():
    cv1 = build_pr_only_canvas()
    spec1 = cv1.to_spec()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    try:
        save_spec(spec1, path)
        spec_loaded = load_spec(path)
        assert spec1 == spec_loaded
    finally:
        os.unlink(path)


def test_summary_is_compact():
    cv = build_pr_only_canvas()
    s = cv.summary()
    assert isinstance(s, str) and len(s) > 0
    assert len(s) < 2048, f"summary too long ({len(s)} chars):\n{s}"
    assert 'PR_col' in s and 'pr_dnp' in s and 'poly2_T1' in s
    print('\nsummary:\n' + s)


def test_canvas_compiles():
    """Sanity check: a spec-built canvas can still compile to a Program.

    No execution — just verify the graph is well-formed and scheduling
    completes. Unwired ports default to 0.0 at runtime.
    """
    cv = build_pr_only_canvas()
    prog = cv.compile()
    assert prog is not None
    assert len(prog.nodes) > 0
    print(f"\ncompiled program: {len(prog.nodes)} nodes, "
          f"dag={len(prog.dag_schedule)} scc={len(prog.scc_schedule)}")


if __name__ == '__main__':
    # Plain-script driver — handy when pytest isn't available.
    tests = [
        test_registry_populated,
        test_spec_validates,
        test_spec_roundtrip_identity,
        test_disk_roundtrip,
        test_summary_is_compact,
        test_canvas_compiles,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok    {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
