"""
diagram2d.py
------------
Publication-quality 2D schematic diagrams for any Canvas.

Renders an illustrative 3-column slice of the circuit. Per-MC internal
block structure is preserved verbatim from ``mc._exec_nodes`` /
``mc._exec_edges``. Inter-stage connectivity is reduced to one
channel-per-arrow-per-slot so the picture stays readable. Sparse
microcircuit types (MVP, LPLC2) are rendered identically to dense ones —
the diagram is a high-level architecture sketch, not a faithful
projection of the 3D layout.

Public surface
--------------
DiagramOptions   dataclass of layout/style knobs
gen_flat_diagram build a matplotlib Figure from a Canvas

Usage
-----
    fig = canvas.gen_flat_diagram(cols=3)
    fig.savefig('outputs/circuit.png', dpi=200)
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ── MIMO port aggregation ──────────────────────────────────────────────────

_COL_PORT_RE = re.compile(r'^(.+?)_col_\d+$')


def _aggregate_port_name(port_name: str) -> str:
    """Group MIMO ports by prefix.

    ``input_col_42`` → ``input``;  ``output_val_col_3`` → ``output_val``;
    ``num_in`` → ``num_in`` (unchanged because there is no ``_col_<n>`` suffix).
    """
    m = _COL_PORT_RE.match(port_name)
    return m.group(1) if m else port_name


# ── Drawing primitives (lifted from circuit_diagram.py) ────────────────────

def _draw_block(ax, cx, cy, w, h, label, *, color='#d0e8ff',
                edge_color='#333333', fontsize=8, alpha=0.85,
                ports_top=None, ports_bottom=None, ports_left=None,
                ports_right=None, label_color='black'):
    """Draw a labeled rounded box, return ``{port_name: (x, y)}``."""
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=edge_color,
        linewidth=1.0, alpha=alpha, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color=label_color, zorder=3)

    ports: Dict[str, Tuple[float, float]] = {}

    def place(names, axis, side):
        if not names:
            return
        n = len(names)
        for i, pn in enumerate(names):
            if axis == 'x':
                frac = (i + 1) / (n + 1)
                px = cx - w / 2 + w * frac
                py = cy + h / 2 if side == 'top' else cy - h / 2
            else:
                frac = (i + 1) / (n + 1)
                py = cy - h / 2 + h * frac
                px = cx - w / 2 if side == 'left' else cx + w / 2
            ports[pn] = (px, py)

    place(ports_top,    'x', 'top')
    place(ports_bottom, 'x', 'bottom')
    place(ports_left,   'y', 'left')
    place(ports_right,  'y', 'right')
    return ports


def _draw_arrow(ax, src, dst, *, color='#333333', style='->', lw=1.0,
                linestyle='-', connectionstyle='arc3,rad=0',
                label=None, label_fontsize=6, label_color='#555555',
                zorder=1, shrinkA=3, shrinkB=3, alpha=1.0):
    arrow = FancyArrowPatch(
        src, dst, arrowstyle=style, color=color,
        linewidth=lw, linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=shrinkA, shrinkB=shrinkB,
        zorder=zorder, mutation_scale=10,
        alpha=alpha,
    )
    ax.add_patch(arrow)
    if label:
        mx = (src[0] + dst[0]) / 2
        my = (src[1] + dst[1]) / 2
        ax.text(mx + 0.05, my, label, fontsize=label_fontsize,
                color=label_color, ha='left', va='center', zorder=4)


def _draw_stage_label(ax, y, label, x=-0.3, fontsize=10):
    ax.text(x, y, label, fontsize=fontsize, fontweight='bold',
            ha='right', va='center', color='#444444')


def _draw_stage_sep(ax, y, xmin, xmax):
    ax.plot([xmin, xmax], [y, y], '--', color='#cccccc', lw=0.8, zorder=0)


def _draw_column_header(ax, x, y, label, fontsize=9):
    ax.text(x, y, label, ha='center', va='bottom', fontsize=fontsize,
            fontweight='bold', color='#666666')


# ── Options ────────────────────────────────────────────────────────────────

@dataclass
class DiagramOptions:
    cols:           Union[int, Sequence[int], None] = 3
    centre_col:     Optional[int] = None
    stage_order:    Optional[List[str]] = None
    show_io_labels: bool = True
    figsize:        Optional[Tuple[float, float]] = None
    title:          Optional[str] = None
    save_dir:       Optional[str] = None
    save_name:      str = 'flat_diagram'


# ── Selection helpers ──────────────────────────────────────────────────────

def _select_cols(canvas, cols, centre):
    """Return the list of ``col_idx`` values that drive the slot layout.

    Always returns exactly the requested count; pads from arbitrary cols
    in the canvas if the centre's spiral neighbourhood is exhausted.
    """
    if isinstance(cols, (list, tuple)):
        return list(cols)
    n = int(cols) if cols is not None else 3

    all_canvas_cols = sorted({mc.col_idx for mc in canvas.microcircuits.values()})
    if not all_canvas_cols:
        return []

    # Pick the centre column.
    if centre is None:
        if canvas.hex_lookup:
            best_bio = min(canvas.hex_lookup,
                           key=lambda i: canvas.hex_lookup[i][0] ** 2
                                          + canvas.hex_lookup[i][1] ** 2)
            try:
                centre = canvas._hex_coords_id.index(best_bio)
            except ValueError:
                centre = all_canvas_cols[0]
        else:
            centre = all_canvas_cols[0]

    # Walk a spiral from the centre and keep cols that the canvas has.
    spiral = canvas.graph_utils.local_order(
        centre, num_rings=max(1, (n + 4) // 6),
        require_in_graph=False,
    )
    in_canvas = set(all_canvas_cols)
    selected: List[int] = []
    for ci in spiral:
        if ci in in_canvas and ci not in selected:
            selected.append(ci)
            if len(selected) >= n:
                return selected
    for ci in all_canvas_cols:
        if ci not in selected:
            selected.append(ci)
            if len(selected) >= n:
                break
    return selected[:n]


def _stage_order(canvas, override=None):
    """Order mc_type names by forward dataflow depth.

    Builds the stage-level DAG from forward (non-feedback) inter-MC edges,
    computes BFS depth from the sources, and breaks ties at the same
    depth by promoting stages that feed back to an upstream stage. This
    places lateral side-branches (MVP) directly under their parent (PR)
    even when the parent also feeds forward to a sibling at the same
    depth (ONOFF). Final tiebreaker is descending mean ``mc.center[2]``.
    """
    if override is not None:
        return [t for t in override if t in canvas.mc_types and canvas.mc_types[t]]

    types = [t for t, mcs in canvas.mc_types.items() if mcs]
    if not types:
        return []

    mc_to_type = _mc_type_map(canvas)
    fwd: set = set()                  # (src_type, dst_type) — forward edges
    feedback_stages: set = set()      # stages that emit a feedback edge

    for s_mc, _s_port, d_mc, d_port in canvas.inter_microcircuit_edges:
        s_type = mc_to_type.get(s_mc)
        d_type = mc_to_type.get(d_mc)
        if not s_type or not d_type or s_type == d_type:
            continue
        if 'feedback' in d_port:
            feedback_stages.add(s_type)
        else:
            fwd.add((s_type, d_type))

    in_degree: Dict[str, int] = {t: 0 for t in types}
    succs: Dict[str, set] = defaultdict(set)
    for s, d in fwd:
        succs[s].add(d)
        in_degree[d] += 1

    depths: Dict[str, int] = {}
    frontier = [t for t in types if in_degree[t] == 0]
    for t in frontier:
        depths[t] = 0
    visited = set(frontier)
    while frontier:
        nxt = []
        for t in frontier:
            for s in succs[t]:
                if s not in visited:
                    visited.add(s)
                    depths[s] = depths[t] + 1
                    nxt.append(s)
        frontier = nxt
    # Anything unreached (true cycle, no source) ends up last.
    fallback_depth = (max(depths.values()) + 1) if depths else 0
    for t in types:
        depths.setdefault(t, fallback_depth)

    def sort_key(t):
        mean_z = sum(mc.center[2] for mc in canvas.mc_types[t]) / len(canvas.mc_types[t])
        return (depths[t], 0 if t in feedback_stages else 1, -mean_z)

    return sorted(types, key=sort_key)


def _pick_slot_mcs(mc_list, selected_cols):
    """For one mc_type, pick one MC per slot.

    Dense: use the MC at the literal slot col_idx when present.
    Sparse: fall back to the MC of the closest col_idx (may repeat).
    """
    by_col = {mc.col_idx: mc for mc in mc_list}
    slot_mcs = []
    for ci in selected_cols:
        if ci in by_col:
            slot_mcs.append(by_col[ci])
        else:
            slot_mcs.append(min(mc_list, key=lambda m: abs(m.col_idx - ci)))
    return slot_mcs


def _is_dense(mc_list, selected_cols) -> bool:
    """An mc_type is dense iff every selected col_idx has its own MC."""
    cols_with_mc = {mc.col_idx for mc in mc_list}
    return all(ci in cols_with_mc for ci in selected_cols)


def _block_depths(mc):
    """Topological depth per block in an MC, robust to cycles."""
    nodes = list(mc._exec_nodes.keys())
    preds: Dict[str, List[str]] = {n: [] for n in nodes}
    for src, _, dst, _ in mc._exec_edges:
        if src in preds and dst in preds:
            preds[dst].append(src)
    depths: Dict[str, int] = {}
    visiting: set = set()

    def compute(n):
        if n in depths:
            return depths[n]
        if n in visiting:
            return 0
        visiting.add(n)
        if not preds[n]:
            d = 0
        else:
            d = max(compute(p) for p in preds[n]) + 1
        depths[n] = d
        visiting.discard(n)
        return d

    for n in nodes:
        compute(n)
    return depths


# ── Block colour heuristic ─────────────────────────────────────────────────

def _block_color(block) -> str:
    cls = type(block).__name__
    if cls == 'Division':
        return '#ffcdd2'
    if cls == 'Rectifier':
        return '#ffe0b2' if getattr(block, 'mode', 'on') == 'on' else '#e1bee7'
    if cls == 'TemporalFilter':
        return '#c8e6c9'
    if cls == 'Aggregator':
        return '#fff9c4'
    if cls == 'TemporalDerivative':
        return '#d1c4e9'
    if cls == 'FuncBlock':
        return '#bbdefb'
    return '#d0e8ff'


# ── Per-MC layout ──────────────────────────────────────────────────────────

def _aggregated_ports(block):
    """Return (top_inputs, bottom_outputs, left_inputs) for a block.

    Feedback inputs (name contains ``feedback``) move to the left side so
    incoming recurrent arrows can curve around without crossing the box.
    """
    seen_in, seen_out = set(), set()
    inputs_agg, outputs_agg = [], []
    for pn in block.inputs:
        a = _aggregate_port_name(pn)
        if a not in seen_in:
            seen_in.add(a)
            inputs_agg.append(a)
    for pn in block.outputs:
        a = _aggregate_port_name(pn)
        if a not in seen_out:
            seen_out.add(a)
            outputs_agg.append(a)

    top_in   = [p for p in inputs_agg if 'feedback' not in p]
    left_in  = [p for p in inputs_agg if 'feedback' in p]
    return top_in, outputs_agg, left_in


def _short_label(block_id: str, max_chars: int = 14) -> str:
    """Compact label for a block: strip ``_block`` suffix, then truncate."""
    s = block_id
    if s.endswith('_block'):
        s = s[:-len('_block')]
    if len(s) > max_chars:
        s = s[:max_chars - 1] + '…'
    return s


def _layout_mc(ax, mc, cell_cx, cell_y_top, cell_y_bot, cell_w,
               block_h=0.35):
    """Render the blocks of one MC inside a vertical cell.

    Block width scales down when many blocks share a depth row so e.g. the
    four LPLC2 dendrites fit without overlap. Labels are auto-truncated.
    Returns ``{block_id: {port_name: (x, y)}}`` for downstream arrow routing.
    """
    depths = _block_depths(mc)
    if not depths:
        return {}
    max_depth = max(depths.values())
    by_depth: Dict[int, List[str]] = defaultdict(list)
    for bid, d in depths.items():
        by_depth[d].append(bid)

    cell_h = cell_y_top - cell_y_bot
    n_rows = max_depth + 1
    y_step = cell_h / (n_rows + 1)
    y_at = [cell_y_top - (d + 1) * y_step for d in range(n_rows)]

    # Adaptive block width: fit the widest depth row within ~95% of the cell.
    widest = max((len(v) for v in by_depth.values()), default=1)
    horizontal_room = cell_w * 0.92
    block_w = min(0.95, horizontal_room / (widest + 0.4))
    # Scale font down for narrow blocks so labels still read.
    block_font = 7 if block_w >= 0.7 else (6 if block_w >= 0.55 else 5)

    ports_map: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for d in range(n_rows):
        block_ids = by_depth.get(d, [])
        n = len(block_ids)
        for j, bid in enumerate(block_ids):
            if n == 1:
                bx = cell_cx
            else:
                # Spread evenly across the available width.
                frac = (j + 0.5) / n
                bx = cell_cx - horizontal_room / 2 + horizontal_room * frac
            by = y_at[d]

            block = mc._exec_nodes[bid]
            top_in, bot_out, left_in = _aggregated_ports(block)
            ports = _draw_block(
                ax, bx, by, block_w, block_h, _short_label(bid),
                color=_block_color(block), fontsize=block_font,
                ports_top=top_in if top_in else None,
                ports_bottom=bot_out if bot_out else None,
                ports_left=left_in if left_in else None,
            )
            ports_map[bid] = ports
    return ports_map


# ── Channel extraction ─────────────────────────────────────────────────────

def _mc_type_map(canvas) -> Dict[str, str]:
    out = {}
    for t, mcs in canvas.mc_types.items():
        for mc in mcs:
            out[mc.name] = t
    return out


def _extract_channels(canvas):
    """Collapse all inter-MC edges into unique stage-to-stage channels.

    Each channel records whether the underlying ports were MIMO (had a
    ``_col_<N>`` suffix on src, dst, or both). Channels flagged as MIMO
    are rendered all-to-all across slots to imply neighbourhood fan-in /
    fan-out; non-MIMO channels render one arrow per slot.
    """
    mc_to_type = _mc_type_map(canvas)
    channels: Dict[Tuple, dict] = {}
    for s_mc_n, s_port_pub, d_mc_n, d_port_pub in canvas.inter_microcircuit_edges:
        s_mc = canvas.microcircuits.get(s_mc_n)
        d_mc = canvas.microcircuits.get(d_mc_n)
        if s_mc is None or d_mc is None:
            continue
        s_type = mc_to_type.get(s_mc_n)
        d_type = mc_to_type.get(d_mc_n)
        if s_type is None or d_type is None:
            continue
        if s_port_pub not in s_mc.output_ports:
            continue
        s_block, s_internal = s_mc.output_ports[s_port_pub]
        s_prefix = _aggregate_port_name(s_internal)
        s_is_mimo = bool(_COL_PORT_RE.match(s_internal))
        for d_block, d_internal in d_mc.input_ports.get(d_port_pub, []):
            d_prefix = _aggregate_port_name(d_internal)
            d_is_mimo = bool(_COL_PORT_RE.match(d_internal))
            key = (s_type, s_block, s_prefix, d_type, d_block, d_prefix)
            if key not in channels:
                channels[key] = {
                    'src_type':   s_type,
                    'src_block':  s_block,
                    'src_prefix': s_prefix,
                    'dst_type':   d_type,
                    'dst_block':  d_block,
                    'dst_prefix': d_prefix,
                    'feedback':   'feedback' in d_prefix,
                    'is_mimo':    s_is_mimo or d_is_mimo,
                }
            else:
                # Promote to MIMO if any contributing edge is MIMO.
                channels[key]['is_mimo'] |= (s_is_mimo or d_is_mimo)
    return list(channels.values())


# ── External I/O detection ─────────────────────────────────────────────────

def _external_endpoints(canvas):
    """Return ``(set(external input MC public ports), set(external output ...))``.

    An MC public input is "external" iff no inter-MC edge ends at it.
    Symmetric for outputs. Iterates the full canvas, not just the slice.
    """
    used_in: set = set()
    used_out: set = set()
    for s_mc, s_port, d_mc, d_port in canvas.inter_microcircuit_edges:
        used_out.add((s_mc, s_port))
        used_in.add((d_mc, d_port))
    return used_in, used_out


# ── Main entry ─────────────────────────────────────────────────────────────

def gen_flat_diagram(canvas, **kwargs) -> plt.Figure:
    """Build a flat 2D schematic of ``canvas`` as a matplotlib Figure.

    See ``DiagramOptions`` for the supported keyword arguments.
    """
    opts = DiagramOptions(**kwargs)

    selected_cols = _select_cols(canvas, opts.cols, opts.centre_col)
    if not selected_cols:
        raise ValueError("Canvas has no microcircuits — nothing to draw.")
    n_slots = len(selected_cols)

    stage_types = _stage_order(canvas, opts.stage_order)
    if not stage_types:
        raise ValueError("Canvas has no populated mc_types.")

    # One MC per slot per stage, and a dense/sparse classification.
    stage_mcs: Dict[str, List] = {
        t: _pick_slot_mcs(canvas.mc_types[t], selected_cols)
        for t in stage_types
    }
    is_dense_map: Dict[str, bool] = {
        t: _is_dense(canvas.mc_types[t], selected_cols)
        for t in stage_types
    }
    centre_slot = n_slots // 2

    # Allocate y-band per stage from the prototype MC's block depth.
    stage_rows: Dict[str, int] = {}
    for t in stage_types:
        d = _block_depths(stage_mcs[t][0])
        stage_rows[t] = (max(d.values()) if d else 0) + 1

    # X layout. Widen col_spacing so multi-block depth rows + slot headers
    # both fit without overlap.
    col_spacing = 3.5
    col_xs = [(i + 1) * col_spacing for i in range(n_slots)]
    cell_w = col_spacing * 0.92
    xmin_axis = col_xs[0] - cell_w * 0.6
    xmax_axis = col_xs[-1] + cell_w * 0.6

    # Y layout — top down.
    row_h = 0.9
    stage_gap = 0.6
    y_cursor = 0.0
    stage_bounds: Dict[str, Tuple[float, float]] = {}
    for t in stage_types:
        h = max(row_h, row_h * stage_rows[t])
        stage_bounds[t] = (y_cursor, y_cursor - h)
        y_cursor -= h + stage_gap

    # Figsize: auto unless overridden.
    if opts.figsize is None:
        total_h = abs(y_cursor) + 3.0
        total_w = (xmax_axis - xmin_axis) + 3.0
        figsize = (max(8.0, total_w * 0.9), max(6.0, total_h * 0.7))
    else:
        figsize = opts.figsize

    # Vertical layout above the first stage: slot headers, input label,
    # external-input arrows. Stack so they never collide.
    Y_SLOT_HEADER = 2.0
    Y_INPUT_LABEL = 1.4
    Y_INPUT_SRC   = 0.9          # tail of every external-input arrow

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xmin_axis - 1.5, xmax_axis + 1.0)
    ax.set_ylim(y_cursor - 0.5, Y_SLOT_HEADER + 0.8)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.suptitle(opts.title or 'Circuit Schematic',
                 fontsize=13, fontweight='bold', y=0.97)

    # Column headers
    for ci, cx in enumerate(col_xs):
        _draw_column_header(ax, cx, Y_SLOT_HEADER,
                            f'slot {ci}  (col {selected_cols[ci]})')

    # Render each stage. Dense types: one MC per slot. Sparse types: one
    # MC at the row centre spanning the full slot row (rendering three
    # identical wide blocks adds no information). For sparse types every
    # (type, slot) key in ports_global points to the same rendered pmap
    # so downstream channel routing is uniform.
    ports_global: Dict[Tuple[str, int], Dict[str, Dict[str, Tuple[float, float]]]] = {}
    full_row_w = (col_xs[-1] - col_xs[0]) + cell_w
    centre_x = (col_xs[0] + col_xs[-1]) / 2.0
    for t in stage_types:
        y_top, y_bot = stage_bounds[t]
        mid_y = (y_top + y_bot) / 2
        _draw_stage_label(ax, mid_y, t, x=xmin_axis - 0.5, fontsize=9)
        _draw_stage_sep(ax, y_top + 0.1, xmin_axis, xmax_axis)
        if is_dense_map[t]:
            for ci, mc in enumerate(stage_mcs[t]):
                pmap = _layout_mc(ax, mc, col_xs[ci], y_top, y_bot, cell_w)
                ports_global[(t, ci)] = pmap
        else:
            # One wide block at the row centre.
            mc = stage_mcs[t][centre_slot]
            pmap = _layout_mc(ax, mc, centre_x, y_top, y_bot, full_row_w)
            for ci in range(n_slots):
                ports_global[(t, ci)] = pmap

    # Effective slots per type: dense → every slot has its own rendering;
    # sparse → all slots share the centre rendering, so draw intra-MC
    # arrows / external markers only once.
    def _effective_slots(t):
        return list(range(n_slots)) if is_dense_map[t] else [centre_slot]

    # Intra-MC arrows — drawn once per effective slot
    for t in stage_types:
        for ci in _effective_slots(t):
            mc = stage_mcs[t][ci]
            pmap = ports_global.get((t, ci), {})
            for src, sport, dst, dport in mc._exec_edges:
                if src not in pmap or dst not in pmap:
                    continue
                sp = _aggregate_port_name(sport)
                dp = _aggregate_port_name(dport)
                if sp not in pmap[src] or dp not in pmap[dst]:
                    continue
                is_fb = 'feedback' in dp
                _draw_arrow(
                    ax, pmap[src][sp], pmap[dst][dp],
                    color='#c62828' if is_fb else '#333333',
                    lw=1.0,
                    linestyle='--' if is_fb else '-',
                    connectionstyle='arc3,rad=0.0',
                )

    # Inter-MC channels. Pairs depend on (1) MIMO vs non-MIMO and (2) the
    # effective slot count of each side. Dense=3 slots, sparse=1 slot
    # (centre). MIMO is rendered as effective_src × effective_dst; non-MIMO
    # zips dense↔dense and broadcasts dense↔sparse.
    for ch in _extract_channels(canvas):
        s_type, d_type = ch['src_type'], ch['dst_type']
        if s_type not in stage_mcs or d_type not in stage_mcs:
            continue
        is_fb = ch['feedback']
        is_mimo = ch['is_mimo']
        s_slots_eff = _effective_slots(s_type)
        d_slots_eff = _effective_slots(d_type)
        if is_mimo:
            pairs = [(s, d) for s in s_slots_eff for d in d_slots_eff]
        else:
            if len(s_slots_eff) == len(d_slots_eff):
                pairs = list(zip(s_slots_eff, d_slots_eff))
            elif len(s_slots_eff) == 1:
                pairs = [(s_slots_eff[0], d) for d in d_slots_eff]
            elif len(d_slots_eff) == 1:
                pairs = [(s, d_slots_eff[0]) for s in s_slots_eff]
            else:
                pairs = list(zip(s_slots_eff, d_slots_eff))

        for s_slot, d_slot in pairs:
            s_ports = ports_global.get((s_type, s_slot), {})
            d_ports = ports_global.get((d_type, d_slot), {})
            if ch['src_block'] not in s_ports or ch['dst_block'] not in d_ports:
                continue
            if ch['src_prefix'] not in s_ports[ch['src_block']]:
                continue
            if ch['dst_prefix'] not in d_ports[ch['dst_block']]:
                continue
            sxy = s_ports[ch['src_block']][ch['src_prefix']]
            dxy = d_ports[ch['dst_block']][ch['dst_prefix']]

            # Curvature: keep crossings clean. Diagonals curve outward;
            # same-slot arrows stay straight (or small feedback bend).
            offset = d_slot - s_slot
            if offset == 0:
                rad = 0.35 if (is_fb and s_slot != n_slots // 2) else 0.0
                if is_fb and s_slot > n_slots // 2:
                    rad = -rad
            else:
                # Diagonals: curve right if going right, left if left.
                rad = 0.15 * offset
            lw = 0.7 if (is_mimo and offset != 0) else 1.1
            alpha = 0.55 if (is_mimo and offset != 0) else 1.0
            _draw_arrow(
                ax, sxy, dxy,
                color='#c62828' if is_fb else '#333333',
                lw=lw, linestyle='--' if is_fb else '-',
                connectionstyle=f'arc3,rad={rad}',
                alpha=alpha,
            )

    # External I/O annotations
    if opts.show_io_labels:
        used_in, used_out = _external_endpoints(canvas)

        # External inputs — at the top stage. Sparse top stages only need
        # the centre rendering's markers.
        top_t = stage_types[0]
        for ci in _effective_slots(top_t):
            mc = stage_mcs[top_t][ci]
            for pub_name, connections in mc.input_ports.items():
                if (mc.name, pub_name) in used_in:
                    continue
                if not connections:
                    continue
                block_id, port_name = connections[0]
                pmap = ports_global.get((top_t, ci), {})
                if block_id not in pmap:
                    continue
                agg = _aggregate_port_name(port_name)
                if agg not in pmap[block_id]:
                    continue
                tgt = pmap[block_id][agg]
                src = (tgt[0], Y_INPUT_SRC)
                _draw_arrow(ax, src, tgt, color='black', lw=1.2)
        # Single label
        ax.text(col_xs[n_slots // 2], Y_INPUT_LABEL,
                'input', ha='center', va='center',
                fontsize=10, color='#444444', fontweight='bold')

        # External outputs — at the bottom stage. When a block has many
        # external outputs (e.g. T4/T5 with output_a..d), text labels at
        # consecutive ports overlap. We strip the common "output_" prefix
        # for compactness; if even the suffix overlaps we fall back to a
        # triangle marker without text.
        bot_t = stage_types[-1]
        _, y_bot = stage_bounds[bot_t]
        for ci in _effective_slots(bot_t):
            mc = stage_mcs[bot_t][ci]
            # Pre-count externals per block so we can decide labelling style.
            by_block: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
            for pub_name, (block_id, port_name) in mc.output_ports.items():
                if (mc.name, pub_name) in used_out:
                    continue
                by_block[block_id].append((pub_name, port_name))

            pmap = ports_global.get((bot_t, ci), {})
            for block_id, entries in by_block.items():
                if block_id not in pmap:
                    continue
                compact = len(entries) >= 3
                for pub_name, port_name in entries:
                    agg = _aggregate_port_name(port_name)
                    if agg not in pmap[block_id]:
                        continue
                    src = pmap[block_id][agg]
                    tgt = (src[0], src[1] - 0.25)
                    if compact:
                        ax.plot(tgt[0], tgt[1], 'v', color='#555555',
                                markersize=4, zorder=3)
                        short = pub_name.replace('output_', '').replace('input_', '')
                        ax.text(tgt[0], tgt[1] - 0.18, short,
                                ha='center', va='top', fontsize=6,
                                color='#666666')
                    else:
                        _draw_arrow(ax, src, tgt, color='black', lw=1.0)
                        ax.text(tgt[0], tgt[1] - 0.1, pub_name,
                                ha='center', va='top', fontsize=7,
                                color='#555555')

    # Optional save
    if opts.save_dir:
        os.makedirs(opts.save_dir, exist_ok=True)
        base = os.path.join(opts.save_dir, opts.save_name)
        for ext in ('png', 'pdf'):
            fig.savefig(f'{base}.{ext}', dpi=200,
                        bbox_inches='tight', facecolor='white')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
