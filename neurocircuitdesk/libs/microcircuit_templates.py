"""
neurocircuitdesk.libs.microcircuit_templates
--------------------------------------------
The curated library of named ``MicroCircuit`` templates ("mc_lib").

Templates encode *topology only* — they add blocks, wire intra-MC
connections, and declare public I/O. Block algorithms and parameters are
assigned by the caller via :meth:`MicroCircuit.set_block_func` /
:meth:`MicroCircuit.set_block_params` after instantiation, so the same
template can host different algorithm choices (e.g. ``borst`` vs ``hr``
vs ``bl`` inside ``iCMC_t4t5_motiondetector``).

Naming convention
~~~~~~~~~~~~~~~~~
``{CMC,iCMC}_<biology>_<variant>`` — see the project-wide CMC / iCMC
glossary (``docs/microcircuit_construction.md``):

- **CMC** — Columnar MicroCircuit; inputs come from a single column.
  Registered via :meth:`Canvas.add_microcircuit_columnar`.
- **iCMC** — Inter-Columnar MicroCircuit; inputs span a neighbourhood.
  Registered via :meth:`Canvas.add_microcircuit_intercolumnar` and the
  template accepts a ``neighborhood`` kwarg.

Inspection API
~~~~~~~~~~~~~~

>>> from neurocircuitdesk.libs import microcircuit_templates as mc_lib
>>> mc_lib.list()                                    # all template names
>>> mc_lib.list(category='iCMC')                     # filter
>>> mc_lib.describe('CMC_photoreceptor_dnp')         # TemplateInfo
>>> mc_lib.show('CMC_photoreceptor_dnp')             # source viewer
>>> mc_lib.get('iCMC_t4t5_motiondetector')           # callable
>>> mc_lib.preview('iCMC_lplc2_loomingdetector')     # 3D Plotly figure

Typical usage
~~~~~~~~~~~~~

>>> from neurocircuitdesk import Canvas
>>> from neurocircuitdesk.libs import microcircuit_templates as mc_lib
>>> from neurocircuitdesk import borst_algorithm
>>>
>>> cv = Canvas(...)
>>> cv.add_mc_type('PR_col')
>>> for col_idx in range(N):
...     cv.add_microcircuit_columnar(
...         col_idx=col_idx, z=1.3, mc_type='PR_col',
...         template=mc_lib.get('CMC_photoreceptor_dnp'),
...     )
>>>
>>> cv.add_mc_type('MOTION_ON_col')
>>> motion_centers = cv.graph_utils.calc_mimo_centers(
...     limit=N, step=1, jump=1, num_rings=1, require_in_graph=False)
>>> for col_idx, nb in motion_centers.items():
...     cv.add_microcircuit_intercolumnar(
...         center_col_idx=col_idx, neighborhood=nb,
...         z=-1.0, mc_type='MOTION_ON_col',
...         template=mc_lib.get('iCMC_t4t5_motiondetector'),
...     )
>>> for mc in cv.mc_types['MOTION_ON_col']:
...     mc.set_block_func('motion_detector_block', borst_algorithm)
...     mc.set_block_params('motion_detector_block', {'N': 2, 'alpha': 100, 'beta': 100})
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
import builtins as _builtins  # we shadow `list` at module level for the public API
from typing import Any, Callable, Dict, List, Optional

from neurocircuitdesk import registry
from neurocircuitdesk.microcircuit import MicroCircuit


# ════════════════════════════════════════════════════════════════════════
#  Templates
# ════════════════════════════════════════════════════════════════════════

# ── CMC_photoreceptor_dnp ──────────────────────────────────────────────

@registry.template(
    name='CMC_photoreceptor_dnp',
    category='columnar',
    description=(
        'Photoreceptor with divisive normalisation. T1/T2 polynomial pre-shapers '
        'feed a Division block whose denominator also accepts a weighted-mean '
        'feedback pair (closes the MVP lateral loop).'),
    default_z=1.3,
)
def CMC_photoreceptor_dnp(mc: MicroCircuit) -> None:
    """Photoreceptor + amacrine-style divisive normalisation.

    Blocks: ``passthrough``, ``T1``, ``T2`` (polynomial pre-shapers),
    ``division_block`` (the divisive normaliser). The denominator
    accepts a weighted-mean feedback pair ``den_feedback_val`` and
    ``den_feedback_weight`` for closing the MVP lateral loop.

    Public I/O:
        inputs:   ``input_main`` (broadcast to T1, T2, passthrough),
                  ``den_feedback_val``, ``den_feedback_weight``
        outputs:  ``output_main``, ``input_passthrough``
    """
    passthrough_pos = (mc.center[0], mc.center[1], mc.center[2] - 0.5)
    mc.add_block('passthrough', *passthrough_pos)

    mc.add_block('T1', *(mc.center[0] - 0.12, mc.center[1], mc.center[2] + 0.75))
    mc.add_block('T2', *(mc.center[0] + 0.12, mc.center[1], mc.center[2] + 0.75))
    mc.add_block('division_block', *mc.center, node_kind='division')

    div_node = mc.get_exec_node('division_block')
    div_node.add_input_port('num_in', port_type='numerator')
    div_node.add_input_port('den_in', port_type='denominator')
    div_node.add_input_port('den_feedback_val',    port_type='denominator',
                            aggregation='weighted_mean')
    div_node.add_input_port('den_feedback_weight', port_type='denominator',
                            aggregation='weighted_mean')

    mc.connect('T1', 'output', 'division_block', 'num_in')
    mc.connect('T2', 'output', 'division_block', 'den_in')

    mc.specify_io(
        inputs=[
            ('input_main',           'T1',             'input'),
            ('input_main',           'T2',             'input'),
            ('input_main',           'passthrough',    'input'),
            ('den_feedback_val',     'division_block', 'den_feedback_val'),
            ('den_feedback_weight',  'division_block', 'den_feedback_weight'),
        ],
        outputs=[
            ('output_main',        'division_block', 'output'),
            ('input_passthrough',  'passthrough',    'output'),
        ],
    )


# ── CMC_lamina_l1l2_onoff ──────────────────────────────────────────────

@registry.template(
    name='CMC_lamina_l1l2_onoff',
    category='columnar',
    description=(
        'L1/L2-style lamina split: temporal bandpass followed by positive and '
        'inverted rectifiers, producing transient ON and OFF channels.'),
    default_z=0.3,
)
def CMC_lamina_l1l2_onoff(mc: MicroCircuit) -> None:
    """Temporal bandpass followed by positive / inverted rectifiers.

    The bandpass kernel is parameterless at the template level — assign
    it after instantiation via ``mc.set_block_params('bp_block',
    {'filter': bp_filter()})``.

    Public I/O:
        inputs:   ``input_main``
        outputs:  ``output_main_on``, ``output_main_off``
    """
    mc.add_block('bp_block',  *(mc.center[0],        mc.center[1],
                                mc.center[2] + 0.75), node_kind='temporal_filter')
    mc.add_block('on_block',  *(mc.center[0] - 0.12, mc.center[1],
                                mc.center[2] + 0.25), node_kind='rectifier_pos')
    mc.add_block('off_block', *(mc.center[0] + 0.12, mc.center[1],
                                mc.center[2] + 0.25), node_kind='rectifier_inv')

    mc.connect('bp_block', 'output', 'on_block',  'input')
    mc.connect('bp_block', 'output', 'off_block', 'input')

    mc.specify_io(
        inputs=[('input_main', 'bp_block', 'input')],
        outputs=[
            ('output_main_on',  'on_block',  'output'),
            ('output_main_off', 'off_block', 'output'),
        ],
    )


# ── iCMC_amacrine_mvp ──────────────────────────────────────────────────

@registry.template(
    name='iCMC_amacrine_mvp',
    category='intercolumnar',
    description=(
        'Amacrine-style lateral feedback unit (MVP). A single inter-columnar MIMO '
        'block over a ring-2 neighbourhood: one input_col_<N> per neighbour, two '
        'output channels (val + weight) per neighbour for closing the divisive '
        'normalisation loop on each upstream PR cell.'),
    default_z=-0.3,
    requires_neighborhood=True,
    default_num_rings=2,
)
def iCMC_amacrine_mvp(mc: MicroCircuit, neighborhood: Dict[int, int]) -> None:
    """Single inter-columnar MIMO block over a ring-2 neighbourhood.

    For each column in ``neighborhood``, declare one input
    (``input_col_<N>``) and two outputs (``output_val_col_<N>``,
    ``output_weight_col_<N>``). The defaults are tuned for the MVP
    amacrine-style lateral feedback loop in the looming demo.

    Parameters
    ----------
    neighborhood : Dict[int, int]
        Mapping ``col_idx → ring_distance`` for the cells covered by
        this MVP centre. ``Canvas.graph_utils.calc_mimo_centers``
        returns exactly this shape.
    """
    input_cols = sorted(neighborhood.keys())
    input_port_names  = [f'input_col_{i}' for i in input_cols]
    output_port_names = ([f'output_val_col_{i}'    for i in input_cols] +
                         [f'output_weight_col_{i}' for i in input_cols])

    mc.add_block(
        'mvp_processor', *mc.center,
        input_names=input_port_names,
        output_names=output_port_names,
    )
    mc.specify_io(
        inputs=[(name, 'mvp_processor', name) for name in input_port_names],
        outputs=[(name, 'mvp_processor', name) for name in output_port_names],
    )


# ── iCMC_t4t5_motiondetector ───────────────────────────────────────────

@registry.template(
    name='iCMC_t4t5_motiondetector',
    category='intercolumnar',
    description=(
        'T4/T5-style motion detector host (one centre + 6 ring-1 inputs → four '
        'directional outputs). Hosts the Borst / HR / BL algorithm (assigned by '
        'caller). NCD instantiates one per column but the cell is iCMC at the '
        'data-flow level.'),
    default_z=-1.0,
    requires_neighborhood=True,
    default_num_rings=1,
)
def iCMC_t4t5_motiondetector(mc: MicroCircuit, neighborhood: Dict[int, int]) -> None:
    """Seven-input MISO host for any of the shipped motion detectors.

    Despite NCD instantiating one motion detector per retinotopic column,
    the cell is **inter-columnar** at the data-flow level — it consumes
    inputs from its centre column plus the six ring-1 neighbours. The
    template therefore takes ``neighborhood`` as an argument and is meant
    to be added via :meth:`Canvas.add_microcircuit_intercolumnar`. The
    ``neighborhood`` argument is accepted but not used inside the body —
    the ring-1 column ordering is re-derived from
    ``mc.canvas.graph_utils.local_order`` so the spiral convention
    (slot 0 = centre, 1..6 = CW from north) is enforced regardless of
    how the caller iterates the neighborhood dict.

    Wires its inputs into a single stateful block whose function is
    supplied separately via
    ``mc.set_block_func('motion_detector_block', borst_algorithm)`` (or
    ``hr_algorithm`` / ``bl_algorithm``). Outputs are the four canonical
    directional channels.

    Public I/O:
        inputs:   ``input_col_<centre>`` plus 6 ring-1 neighbours, in
                  spiral order (slot 0 = centre, 1..6 = CW from north)
        outputs:  ``output_a``, ``output_b``, ``output_c``, ``output_d``
    """
    del neighborhood  # accepted for iCMC API parity; body uses local_order
    ordered_cols = mc.canvas.graph_utils.local_order(
        mc.col_idx, num_rings=1, require_in_graph=False)
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


# ── iCMC_lplc2_loomingdetector ─────────────────────────────────────────

@registry.template(
    name='iCMC_lplc2_loomingdetector',
    category='intercolumnar',
    description=(
        'LPLC2-style looming detector: four cardinal dendritic branches, each '
        'sampling a ring-3 neighbourhood centred on a displaced cardinal point '
        '(offsets 32, 23, 19, 28 in the centre\'s ring-3 spiral ordering). An '
        'axon block multiplies the four dendrite outputs to produce one '
        'looming-selective output.'),
    default_z=-3.0,
    requires_neighborhood=True,
    default_num_rings=6,  # demo's calc_mimo_centers(num_rings=6) for the centre's neighbourhood
)
def iCMC_lplc2_loomingdetector(mc: MicroCircuit,
                                neighborhood: Dict[int, int]) -> None:
    """LPLC2-style looming detector with four cardinal dendritic branches.

    Each branch (a/b/c/d) samples a ring-3 neighbourhood centred on a
    displaced cardinal point (offsets 32, 23, 19, 28 in the centre's
    ring-3 spiral ordering). The axon block multiplies the four
    dendrites to produce one looming-selective output.

    The ``neighborhood`` argument is kept for API parity with
    ``Canvas.add_microcircuit_intercolumnar`` but is not used inside the
    template — branch geometry is derived purely from
    ``mc.canvas.graph_utils.local_order(mc.col_idx, num_rings=3)``.

    Public I/O:
        inputs:   ``input_<label>_col_<N>`` for label ∈ {a, b, c, d} and
                  N spanning each branch's ring-3 neighbourhood
        outputs:  ``output_looming``
    """
    del neighborhood  # accepted for iCMC API parity; body uses local_order
    ordered_cols = mc.canvas.graph_utils.local_order(
        mc.col_idx, num_rings=3, require_in_graph=False)

    branch_offsets = {'a': 32, 'b': 23, 'c': 19, 'd': 28}
    branch_cols: Dict[str, list] = {}
    for label, offset in branch_offsets.items():
        if offset < len(ordered_cols):
            branch_cols[label] = mc.canvas.graph_utils.local_order(
                ordered_cols[offset], num_rings=3, require_in_graph=False)
        else:
            branch_cols[label] = ordered_cols

    branch_inputs = {label: [f'input_{label}_col_{c}' for c in cols]
                     for label, cols in branch_cols.items()}

    for label in 'abcd':
        mc.add_block(f'layer_{label}', *mc.center,
                     input_names=branch_inputs[label],
                     output_names=['output'], stateless=True)

    mc.add_block('looming_detector_block', *mc.center,
                 input_names=[f'input_{c}' for c in 'abcd'],
                 output_names=['output'], stateless=True)

    for label in 'abcd':
        mc.connect(f'layer_{label}', 'output',
                   'looming_detector_block', f'input_{label}')

    mc.specify_io(
        inputs=([(n, 'layer_a', n) for n in branch_inputs['a']] +
                [(n, 'layer_b', n) for n in branch_inputs['b']] +
                [(n, 'layer_c', n) for n in branch_inputs['c']] +
                [(n, 'layer_d', n) for n in branch_inputs['d']]),
        outputs=[('output_looming', 'looming_detector_block', 'output')],
    )


# ════════════════════════════════════════════════════════════════════════
#  Inspection API — list / get / describe / show / preview
# ════════════════════════════════════════════════════════════════════════

_CATEGORY_INTERNAL_TO_PUBLIC = {'columnar': 'CMC', 'intercolumnar': 'iCMC'}
_CATEGORY_PUBLIC_TO_INTERNAL = {v: k for k, v in _CATEGORY_INTERNAL_TO_PUBLIC.items()}


def list(category: Optional[str] = None) -> List[str]:  # noqa: A001 — intentional shadow
    """Return the names of all registered MC templates.

    Parameters
    ----------
    category : {'CMC', 'iCMC'}, optional
        If given, restrict the result to that category.
    """
    names = sorted(registry._TEMPLATES.keys())
    if category is None:
        return names
    if category not in _CATEGORY_PUBLIC_TO_INTERNAL:
        raise ValueError(
            f"category must be 'CMC' or 'iCMC', got {category!r}")
    target = _CATEGORY_PUBLIC_TO_INTERNAL[category]
    return [n for n in names if registry._TEMPLATES[n][1]['category'] == target]


def get(name: str) -> Callable:
    """Return the template callable for ``name``."""
    return registry.get_template(name)


# ── describe — TemplateInfo dataclass ──────────────────────────────────

@dataclass
class TemplateInfo:
    """Structured description of a registered template.

    Dict-like access (``info['category']`` or ``info.category``) plus
    rich notebook rendering via ``_repr_html_``.
    """
    name: str
    category: str                    # 'CMC' or 'iCMC'
    description: str
    block_kinds: List[str]
    io_ports: Dict[str, List[str]]   # {'inputs': [...], 'outputs': [...]}
    default_num_rings: Optional[int] = None
    default_z: float = 0.0
    source_path: str = ''
    requires_neighborhood: bool = False

    def __repr__(self) -> str:
        return (f'TemplateInfo({self.name!r}, {self.category}, '
                f'{len(self.block_kinds)} block kinds)')

    def _repr_html_(self) -> str:
        rows = [
            ('Name',                  f'<code>{self.name}</code>'),
            ('Category',              self.category),
            ('Description',           self.description),
            ('Block kinds',           ', '.join(self.block_kinds) or '—'),
            ('Public inputs',         ', '.join(f'<code>{p}</code>' for p in self.io_ports['inputs']) or '—'),
            ('Public outputs',        ', '.join(f'<code>{p}</code>' for p in self.io_ports['outputs']) or '—'),
            ('Default z',             f'{self.default_z:g}'),
            ('Requires neighborhood', 'yes' if self.requires_neighborhood else 'no'),
            ('Default num_rings',     ('—' if self.default_num_rings is None
                                       else str(self.default_num_rings))),
            ('Source',                f'<code>{self.source_path}</code>'),
        ]
        body = '\n'.join(
            f'<tr><th style="text-align:left;padding-right:1em;vertical-align:top">{k}</th>'
            f'<td>{v}</td></tr>'
            for k, v in rows
        )
        return f'<table style="border-collapse:collapse">{body}</table>'

    # dict-like access
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self):
        return _builtins.list(self.__dataclass_fields__)


def describe(name: str) -> TemplateInfo:
    """Return a :class:`TemplateInfo` for the named template.

    Block kinds and public I/O ports are discovered by instantiating the
    template once in a stub canvas — accurate but does a tiny amount of
    work per call.
    """
    if name not in registry._TEMPLATES:
        raise KeyError(
            f"unknown template {name!r}. "
            f"Known: {sorted(registry._TEMPLATES.keys())}")
    fn, meta = registry._TEMPLATES[name]

    _, mc = _instantiate_for_inspection(name)
    block_kinds = sorted({type(node).__name__ for node in mc._exec_nodes.values()})
    io_ports = {
        'inputs':  _builtins.list(mc.input_ports.keys()),
        'outputs': _builtins.list(mc.output_ports.keys()),
    }

    try:
        source_path = f'{inspect.getsourcefile(fn)}:{inspect.getsourcelines(fn)[1]}'
    except (OSError, TypeError):
        source_path = '<unknown>'

    return TemplateInfo(
        name=name,
        category=_CATEGORY_INTERNAL_TO_PUBLIC[meta['category']],
        description=meta.get('description', '').strip(),
        block_kinds=block_kinds,
        io_ports=io_ports,
        default_num_rings=meta.get('default_num_rings'),
        default_z=float(meta.get('default_z', 0.0)),
        source_path=source_path,
        requires_neighborhood=bool(meta.get('requires_neighborhood', False)),
    )


# ── show — source viewer ───────────────────────────────────────────────

class _TemplateSource:
    """Source-code viewer: plain string in terminal, syntax-highlighted in notebook."""

    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source

    def __repr__(self) -> str:
        return self.source

    def _repr_html_(self) -> str:
        try:
            from IPython.display import Code  # type: ignore
            return Code(self.source, language='python')._repr_html_()
        except ImportError:
            from html import escape
            return f'<pre><code class="python">{escape(self.source)}</code></pre>'


def show(name: str) -> _TemplateSource:
    """Return a notebook-rendering source viewer for the named template."""
    if name not in registry._TEMPLATES:
        raise KeyError(
            f"unknown template {name!r}. "
            f"Known: {sorted(registry._TEMPLATES.keys())}")
    fn = registry._TEMPLATES[name][0]
    src = textwrap.dedent(inspect.getsource(fn))
    return _TemplateSource(name, src)


# ── preview — 3D visualisation ─────────────────────────────────────────

def preview(name: str, *, col_idx: int = 0,
            neighborhood: Optional[Dict[int, int]] = None):
    """Instantiate the named template into a stub canvas and render it.

    Returns a Plotly :class:`Figure`. For iCMC templates without an
    explicit ``neighborhood``, the registered ``default_num_rings`` is
    used to build a default ring-K neighbourhood centred on ``col_idx``.
    """
    _, mc = _instantiate_for_inspection(name,
                                         col_idx=col_idx,
                                         neighborhood=neighborhood)
    # Lazy-import to avoid pulling Plotly when only list/describe are used.
    from neurocircuitdesk.microcircuit_viz import MicroCircuitViz
    return MicroCircuitViz(mc).plot()


# ── Internal helpers ───────────────────────────────────────────────────

def _stub_canvas():
    """Minimal canvas used for inspection / preview. JSON paths default to shipped."""
    from neurocircuitdesk.canvas import Canvas
    libs = Path(__file__).parent
    return Canvas(
        w=400, h=300,
        col_json_path=str(libs / 'jsons' / 'hexcol_l1m3_new_578.json'),
        interconnect_json_path=str(libs / 'jsons' / 'hex_grid_graph.json'),
    )


def _instantiate_for_inspection(name: str, *, col_idx: int = 0,
                                  neighborhood: Optional[Dict[int, int]] = None):
    """Build one MC of the named template in a fresh stub canvas.

    Returns the ``(canvas, mc)`` pair. Used by ``describe`` and
    ``preview`` so the work of instantiation is shared.
    """
    if name not in registry._TEMPLATES:
        raise KeyError(
            f"unknown template {name!r}. "
            f"Known: {sorted(registry._TEMPLATES.keys())}")
    fn, meta = registry._TEMPLATES[name]

    cv = _stub_canvas()
    cv.add_mc_type('_inspect')

    z = float(meta.get('default_z', 0.0))
    if meta['category'] == 'columnar':
        cv.add_microcircuit_columnar(
            col_idx=col_idx, z=z, mc_type='_inspect', template=fn)
    else:
        if neighborhood is None:
            num_rings = meta.get('default_num_rings') or 1
            neighborhood = cv.graph_utils.get_neighbors_in_rings(
                col_idx, num_rings=num_rings, require_in_graph=False)
        cv.add_microcircuit_intercolumnar(
            center_col_idx=col_idx, z=z, mc_type='_inspect',
            neighborhood=neighborhood, template=fn)

    mc = next(iter(cv.microcircuits.values()))
    return cv, mc


__all__ = [
    # templates (registered + importable)
    'CMC_photoreceptor_dnp',
    'CMC_lamina_l1l2_onoff',
    'iCMC_amacrine_mvp',
    'iCMC_t4t5_motiondetector',
    'iCMC_lplc2_loomingdetector',
    # inspection API
    'list', 'get', 'describe', 'show', 'preview',
    'TemplateInfo',
]
