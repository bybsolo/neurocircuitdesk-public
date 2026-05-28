# Building a MicroCircuit

This guide is for **writing your own `MicroCircuit` templates** in NCD —
the kind of thing that gets called from
`Canvas.add_microcircuit_columnar(template=...)` or
`add_microcircuit_intercolumnar(...)`. If you just want to drop in one
of the shipped templates by name, use the library directly:

```python
from neurocircuitdesk.libs import microcircuit_templates as mc_lib
mc_lib.list()                                       # see what's available
mc_lib.describe('CMC_photoreceptor_dnp')            # inspect one
mc_lib.preview('iCMC_t4t5_motiondetector')          # 3D visualisation
```

The rest of this guide is about defining new templates.

## §1. CMC vs iCMC

Every MicroCircuit belongs to one of two categories:

| Term | Meaning | Registered via |
| --- | --- | --- |
| **CMC** — Columnar MicroCircuit | inputs come from a **single column**. Outputs may have multiple ports but they all live in the same column. | `Canvas.add_microcircuit_columnar(col_idx, …)` |
| **iCMC** — Inter-Columnar MicroCircuit | inputs span **multiple columns** (a neighbourhood). The template must accept a `neighborhood` kwarg even if the body re-derives the spiral ordering. | `Canvas.add_microcircuit_intercolumnar(center_col_idx, neighborhood, …)` |

The distinction is decided by **how many columns the inputs span**, not
by per-port arity within a column. A motion detector with 4 output
channels but inputs from 7 columns is iCMC. The lamina L1/L2 split with
1 input and 2 outputs is CMC.

The template function signature follows the category:

```python
def my_cmc_template(mc):                       # CMC — Canvas calls with no kwargs
    ...

def my_icmc_template(mc, neighborhood):        # iCMC — Canvas passes the neighborhood
    ...
```

Even iCMC templates that don't actually use `neighborhood` inside the
body (e.g. they re-derive the spiral ordering from
`mc.canvas.graph_utils.local_order(...)`) still **must accept the
kwarg** — Canvas's `add_microcircuit_intercolumnar` always passes it.
Use `del neighborhood` at the top of the function to make the intent
explicit.

## §2. Anatomy of a template

A template is a function with three responsibilities:

1. **Add blocks** to the microcircuit (`mc.add_block(...)`).
2. **Wire intra-MC connections** between blocks (`mc.connect(...)`).
3. **Declare the public I/O** that other microcircuits will see
   (`mc.specify_io(...)`).

A template **does not** assign computational algorithms or parameters
to the blocks — that's left to the caller after instantiation:

```python
for mc in cv.mc_types['MY_TYPE']:
    mc.set_block_func('some_block', my_algorithm)
    mc.set_block_params('some_block', {'gain': 0.5})
```

This separation is what lets the same template host different
algorithms (e.g. `iCMC_t4t5_motiondetector` can host Borst, HR, or BL).

The microcircuit's `mc` object also exposes:

- `mc.center` — `(x, y, z)` tuple of the MC's retinotopic position.
  Used as a reference when placing blocks (`mc.add_block('foo',
  *mc.center)` puts the block at the MC centre; offsets like
  `mc.center[0] + 0.12` are typical).
- `mc.canvas` — the parent canvas, useful for `mc.canvas.graph_utils.local_order(...)`
  when an iCMC template needs the spiral neighbour ordering of its
  inputs.
- `mc.col_idx` — the column index this MC was instantiated at.

## §3. Available block kinds

Every block you add via `mc.add_block(...)` is one of these kinds.
The `node_kind` kwarg selects which:

### `node_kind='default'` (the default) — `FuncBlock`

Wraps a `@unified_algorithm` function. Default ports: `input` /
`output`. Override with `input_names=[...]` / `output_names=[...]` for
multi-port blocks (this is how MIMO blocks like motion detectors are
built).

```python
mc.add_block('my_block', x, y, z)                          # SISO
mc.add_block('my_mimo_block', x, y, z,                     # multi-port
             input_names=['a', 'b'], output_names=['c'])
mc.add_block('stateful_block', x, y, z,                    # stateful FuncBlock
             input_names=['input_col_0', 'input_col_1'],
             output_names=['output'], stateless=False)
```

The algorithm is assigned later via `mc.set_block_func(block_id, fn)`,
where `fn` must be `@unified_algorithm`-decorated.

### `node_kind='temporal_filter'` — `TemporalFilter`

Convolution with a user-supplied kernel. Ports: `input` / `output`.
The filter kernel **must** be set after instantiation:

```python
mc.add_block('bp', *pos, node_kind='temporal_filter')
# elsewhere:
mc.set_block_params('bp', {'filter': bp_filter()})
```

### `node_kind='rectifier_pos'` and `'rectifier_inv'` — `Rectifier`

Half-wave rectification. `rectifier_pos` zeros negatives;
`rectifier_inv` zeros positives and negates the rest. Ports: `input` /
`output`.

```python
mc.add_block('on',  *pos, node_kind='rectifier_pos')
mc.add_block('off', *pos, node_kind='rectifier_inv')
```

### `node_kind='division'` — `Division`

Numerator / denominator with **dynamic ports** added via
`mc.get_exec_node('div').add_input_port(...)`:

```python
mc.add_block('div', *pos, node_kind='division')
div = mc.get_exec_node('div')
div.add_input_port('num_in',  port_type='numerator')
div.add_input_port('den_in',  port_type='denominator')
div.add_input_port('fb_val',  port_type='denominator',  # weighted-mean feedback pair
                   aggregation='weighted_mean')
div.add_input_port('fb_wt',   port_type='denominator',
                   aggregation='weighted_mean')
```

Output port is always `output`. Supported `aggregation` modes per port
group: `'sum'`, `'mean'`, `'product'`, `'subtract'`, `'weighted_mean'`
(needs paired `_val` / `_wt` ports).

### `node_kind='aggregator'` — `Aggregator`

Multi-input combine. Modes: `'sum'`, `'mean'`, `'product'`,
`'subtract'`. Input ports added dynamically; one output port named
`output`.

```python
mc.add_block('sum', *pos, node_kind='aggregator', mode='sum')
mc.get_exec_node('sum').add_input_port('a')
mc.get_exec_node('sum').add_input_port('b')
```

### `node_kind='derivative'` — `TemporalDerivative`

Three-point central difference. Stateful. Ports: `input` / `output`.

```python
mc.add_block('ddt', *pos, node_kind='derivative')
```

## §4. The MIMO port-name convention

For inter-columnar MicroCircuits, ports that fan in or fan out across
the column neighbourhood follow a strict naming pattern:

- **Inputs**: `input_col_<N>` or `input_<channel>_col_<N>` — one port
  per source column.
- **Outputs**: `output_<channel>_col_<N>` — one port per target
  column.

`<N>` is the source/target column index; `<channel>` is an arbitrary
short name when there's more than one logical channel (e.g.
`input_a_col_…` / `input_b_col_…` for the LPLC2 dendrites).

The MLX engine and the 2D-diagram generator both detect MIMO blocks
by matching these patterns, so **following the convention is what
unlocks MIMO acceleration and visualisation**:

| Pattern | Treated as | Effect |
| --- | --- | --- |
| `input_col_<N>` everywhere | MIMO input | engine builds `inputs['neighbors']` + `neighbor_mask`; visualisation uses hex layout |
| `<channel>_col_<N>` outputs everywhere | MIMO fan-out | engine assigns per-slot outputs from a single tensor; algorithm returns `<channel>_neighbors` key |
| Anything else | scalar port | normal feed dict, normal port markers |

Non-MIMO ports (e.g. `input_main`, `output_main`, `den_feedback_val`)
go through the engine without any neighbour bookkeeping.

## §5. Worked example — a CMC from scratch

Goal: a CMC that takes one input, applies a temporal filter, then runs
it through a positive rectifier. (A simplified version of
`CMC_lamina_l1l2_onoff`, with only the ON branch.)

```python
from neurocircuitdesk.microcircuit import MicroCircuit


def CMC_filter_rectify(mc: MicroCircuit) -> None:
    """Bandpass filter → positive rectifier.

    Public I/O:
        inputs:   `input_main`
        outputs:  `output_main`
    """
    # Two blocks: filter on top, rectifier below.
    mc.add_block('bp',  *(mc.center[0], mc.center[1], mc.center[2] + 0.5),
                 node_kind='temporal_filter')
    mc.add_block('rec', *(mc.center[0], mc.center[1], mc.center[2] - 0.5),
                 node_kind='rectifier_pos')

    # Wire bp.output → rec.input
    mc.connect('bp', 'output', 'rec', 'input')

    # Public I/O: one input feeds bp, one output comes from rec.
    mc.specify_io(
        inputs=[('input_main',  'bp',  'input')],
        outputs=[('output_main', 'rec', 'output')],
    )
```

To use this template:

```python
cv = Canvas(...)
cv.add_mc_type('LAMINA')
for col_idx in range(N_COLS):
    cv.add_microcircuit_columnar(
        col_idx=col_idx, z=0.3, mc_type='LAMINA',
        template=CMC_filter_rectify,
    )

# After all MCs are added, set the filter kernel on each:
for mc in cv.mc_types['LAMINA']:
    mc.set_block_params('bp', {'filter': bp_filter()})
```

The rectifier needs no extra config — the `'rectifier_pos'` kind is
already complete.

To preview it before adding to a real canvas:

```python
mc.show()       # 3D Plotly figure of one instance's wiring
```

## §6. Worked example — an iCMC from scratch

Goal: an iCMC that takes inputs from its 7 ring-1 neighbours
(centre + 6), computes their mean, then runs it through a temporal
derivative. (Roughly an "averaging motion detector".)

```python
from typing import Dict
from neurocircuitdesk.microcircuit import MicroCircuit


def iCMC_neighbours_meanddt(mc: MicroCircuit, neighborhood: Dict[int, int]) -> None:
    """Mean of ring-1 inputs → temporal derivative.

    Public I/O:
        inputs:   `input_col_<N>` for the centre + ring-1 cols (7 total)
        outputs:  `output_main`
    """
    del neighborhood  # accepted for iCMC API parity; we re-derive below

    ordered_cols = mc.canvas.graph_utils.local_order(
        mc.col_idx, num_rings=1, require_in_graph=False)
    input_port_names = [f'input_col_{c}' for c in ordered_cols]

    # An aggregator in mean mode, plus a temporal derivative
    mc.add_block('avg', *(mc.center[0], mc.center[1], mc.center[2] + 0.5),
                 node_kind='aggregator', mode='mean')
    avg = mc.get_exec_node('avg')
    for name in input_port_names:
        avg.add_input_port(name)

    mc.add_block('ddt', *(mc.center[0], mc.center[1], mc.center[2] - 0.5),
                 node_kind='derivative')

    mc.connect('avg', 'output', 'ddt', 'input')

    mc.specify_io(
        inputs=[(name, 'avg', name) for name in input_port_names],
        outputs=[('output_main', 'ddt', 'output')],
    )
```

To use this template:

```python
cv.add_mc_type('AVG_DDT')
centres = cv.graph_utils.calc_mimo_centers(
    limit=N_COLS, step=1, jump=1, num_rings=1, require_in_graph=False)
for col_idx, nb in centres.items():
    cv.add_microcircuit_intercolumnar(
        center_col_idx=col_idx, neighborhood=nb,
        z=-1.0, mc_type='AVG_DDT',
        template=iCMC_neighbours_meanddt,
    )
```

Note how the inputs follow the `input_col_<N>` MIMO pattern from §4 —
this lets the engine batch the neighbour gather efficiently and lets
`MicroCircuitViz` lay the ports out as a local hex pattern.

## §7. Patterns and pitfalls

### Feedback loops live across MCs, not within

Intra-MC `mc.connect(...)` should be a DAG — feedback loops are formed
when multiple MCs reference each other via `cv.connect_microcircuits`,
not via self-referencing connections inside one template. Canvas's
compiler detects strongly-connected components and applies a one-step
delay automatically.

The Division block's `weighted_mean` feedback pair (`fb_val` / `fb_wt`)
is the canonical example: PR_dnp declares them as denominator inputs,
and MVP wires its outputs back into them at the canvas level.

### Don't share block IDs across MCs

Inside a template the IDs (`'T1'`, `'bp'`, `'division_block'`) are
local to the MC; Canvas namespaces them by prepending the MC name on
compile (`PR_col_42/T1`). Two MCs of the same type can have a block
called `T1` without colliding.

### Multiple internal targets for one public input ≠ broadcast

`mc.specify_io(inputs=[(pub, 'block_a', 'input'), (pub, 'block_b',
'input')])` means "the public port `pub` broadcasts to both
`block_a.input` and `block_b.input`". This is how `CMC_photoreceptor_dnp`
gets `input_main` to T1, T2, and passthrough simultaneously.

### iCMC templates should always accept `neighborhood`

Even if you don't use it, accept it. `add_microcircuit_intercolumnar`
always passes it. If you omit the parameter, the template breaks at
the Canvas call site, not at the template definition site — confusing.

### Don't reach for `mc.canvas` in CMC templates

CMC templates don't get neighborhood info anyway; reaching for
`mc.canvas.graph_utils.local_order(...)` inside a CMC template is a
sign that the template should actually be an iCMC.

### Port-name patterns control MIMO behaviour

If you accidentally name an input `input_col_5_secondary` or similar,
the MIMO regex (`(.+)_col_(\d+)$`) won't match, and the engine will
treat the port as scalar. Stick to `<prefix>_col_<N>` with N at the
end of the string.

## §8. Registering with `mc_lib`

Once a template is working, you can register it under a stable string
name so it shows up in `mc_lib.list()`, can be discovered via
`mc_lib.describe(name)`, and can be referenced by name from agent
workflows or `Canvas.from_spec(...)` round-trips:

```python
from neurocircuitdesk import registry

@registry.template(
    name='CMC_my_filter_rectify',
    category='columnar',                  # or 'intercolumnar'
    description='Bandpass filter into positive rectifier.',
    default_z=0.3,                        # suggested z-plane
    requires_neighborhood=False,          # True for iCMC
    default_num_rings=None,               # set to an int for iCMC (see preview docs)
)
def CMC_my_filter_rectify(mc):
    ...
```

Registered templates immediately appear in:

```python
mc_lib.list()                 # name in the list
mc_lib.list(category='CMC')   # name in the filtered list
mc_lib.describe('CMC_my_filter_rectify')
mc_lib.show('CMC_my_filter_rectify')
mc_lib.preview('CMC_my_filter_rectify')
```

For iCMC templates, set `default_num_rings` to the neighbourhood size
the template expects — this is what `mc_lib.preview(...)` uses to
auto-build a neighbourhood when the caller doesn't supply one
explicitly:

```python
@registry.template(
    name='iCMC_my_neighbours',
    category='intercolumnar',
    description='...',
    requires_neighborhood=True,
    default_num_rings=1,                  # ring-1 ⇒ 7 cols (centre + 6)
)
def iCMC_my_neighbours(mc, neighborhood):
    ...
```

If you decide later that the template should live in the shipped
library, move it from your local file into
`neurocircuitdesk/libs/microcircuit_templates.py` — no other change
needed.

## See also

- `docs/unified_algorithm_syntax.md` — how to write the
  `@unified_algorithm`-decorated functions you'll assign to
  `mc.set_block_func(...)` on these blocks.
- `neurocircuitdesk/libs/microcircuit_templates.py` — the source of
  the five shipped templates; the cleanest reading reference.
