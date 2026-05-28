"""
mlx_engine.py
-------------
Generic vectorized execution engine for NCD circuits.

Key idea: instead of executing ~2500 nodes one scalar at a time (Program),
this engine groups nodes by type (e.g. all 1261 'PR_col/pow_block' instances)
and executes each group as a single batched MLX array call per timestep.

The outer Python loop shrinks from ~2500 node iterations to the number of
distinct block kinds in the circuit (~4-8 for the demo).

Works for any circuit compiled by Canvas — no hardcoded topology.
"""

import re
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

import mlx.core as mx
import numpy as np

from neurocircuitdesk.blocks_exe import (
    InputNode, OutputNode, FuncBlock, Node,
    Rectifier, TemporalFilter, Division,
    rectifier_batched, temporal_filter_batched, division_batched,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COL_PORT_RE     = re.compile(r'^input_col_(\d+)$')
# Fan-out MIMO output ports: 'output_val_col_42' → channel='output_val_col', col=42.
# The '_col_' separator is the marker — more permissive patterns would
# accidentally match names like 'output_a'.
_OUT_COL_PORT_RE = re.compile(r'^(.+_col)_(\d+)$')


def _strip_col_idx(mc_name: str) -> str:
    """'PR_col_42' → 'PR_col'"""
    return re.sub(r'_\d+$', '', mc_name)


def _col_idx_from_name(mc_name: str) -> int:
    """'PR_col_42' → 42"""
    m = re.search(r'_(\d+)$', mc_name)
    return int(m.group(1)) if m else 0


def _parse_fq(fq: str) -> Tuple[str, str, str, int]:
    """'PR_col_42/pow_block' → ('PR_col_42', 'pow_block', 'PR_col', 42)"""
    i = fq.index('/')
    mc = fq[:i]
    blk = fq[i + 1:]
    return mc, blk, _strip_col_idx(mc), _col_idx_from_name(mc)


def _is_col_indexed_mimo(node: Node) -> bool:
    """True if every input port matches 'input_col_N' (MIMO neighbourhood node)."""
    return bool(node.inputs) and all(_COL_PORT_RE.match(p) for p in node.inputs)


def _is_col_indexed_mimo_out(node: Node) -> bool:
    """
    True if every output port matches '<channel>_col_<N>'  — this is the
    fan-out MIMO mode: one or more semantic channels, each with a neighbor
    axis (N outputs per node indexed by col_idx).
    """
    return bool(node.outputs) and all(_OUT_COL_PORT_RE.match(p) for p in node.outputs)


def _decode_out_channels(output_port_names: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    For a fan-out MIMO node, group its output ports by channel.
    Returns (channels_in_insertion_order, {channel: [col_idx, col_idx, ...]}).
    Slot order within each channel follows node.outputs insertion order
    (== template declaration order).
    """
    channels: List[str] = []
    cols_per_channel: Dict[str, List[int]] = {}
    for p in output_port_names:
        m = _OUT_COL_PORT_RE.match(p)
        ch  = m.group(1)
        col = int(m.group(2))
        if ch not in cols_per_channel:
            channels.append(ch)
            cols_per_channel[ch] = []
        cols_per_channel[ch].append(col)
    return channels, cols_per_channel


# ---------------------------------------------------------------------------
# Routing descriptors
# ---------------------------------------------------------------------------

@dataclass
class _PortRouting:
    """
    Routing for one source contribution to a port of a NodeGroup.
    A port with fan-in > 1 has multiple _PortRouting entries whose values
    are summed by the engine.

    src_group_key    : '__external__' | '__unconnected__' | group key string
    src_port         : port/channel name in the source (channel name for mimo_out sources)
    src_indices      : (N_group,) int — index into source's node axis per member
    src_slot_indices : (N_group,) int or None — slot index within the source's
                       fan-out output axis. None means the source is 1-D (scalar
                       per node); set means the source is a (N_src, max_out) tensor.
    is_feedback      : read from previous timestep (_prev) instead of _values
    """
    src_group_key: str
    src_port: str
    src_indices: Any              # list[int] during build, mx.array after freeze()
    src_slot_indices: Any = None  # None, list[int], or mx.array
    is_feedback: bool = False

    def freeze(self):
        if isinstance(self.src_indices, list):
            self.src_indices = mx.array(self.src_indices, dtype=mx.int32)
        if isinstance(self.src_slot_indices, list):
            self.src_slot_indices = mx.array(self.src_slot_indices, dtype=mx.int32)


# ---------------------------------------------------------------------------
# NodeGroup
# ---------------------------------------------------------------------------

class NodeGroup:
    """
    A batch of same-kind nodes that share function, port structure, and params.

    Input routing modes:
      is_mimo=False  →  per-port routing: routings[port_name] = List[_PortRouting]
      is_mimo=True   →  neighbourhood tensor: nbr_routing (N, max_nbrs) int32
                        Engine assembles {'__nbr_F__': ..., '__nbr_mask__': ...}
                        and calls fn with that dict.

    Output routing modes:
      is_mimo_out=False → output_port_names hold real port names; values are (N,)
      is_mimo_out=True  → output_port_names hold channel names (e.g.
                          'output_val_col'); values are (N, max_out_nbrs) tensors
                          and downstream consumers gather with (node_idx, slot_idx).
    """

    def __init__(self, key: str, is_mimo: bool = False, is_mimo_out: bool = False):
        self.key = key
        self.is_mimo = is_mimo
        self.is_mimo_out = is_mimo_out

        self.node_names:  List[str] = []
        self.col_indices: List[int] = []

        self.fn: Optional[Callable] = None
        self.params: Dict = {}
        self.state: Dict = {}
        self.is_stateless: bool = True

        # Standard routing (non-MIMO)
        self.input_port_names:  List[str] = []
        self.output_port_names: List[str] = []   # channel names when is_mimo_out
        self.routings: Dict[str, List[_PortRouting]] = {}

        # MIMO (input) neighbourhood routing
        self.nbr_src_group: str = ''
        self.nbr_src_port:  str = ''
        self._nbr_rows: List[List[int]] = []     # built per member, frozen later
        self.nbr_routing: Optional[mx.array] = None   # (N, max_nbrs) int32
        self.nbr_mask:    Optional[mx.array] = None   # (N, max_nbrs) float32
        # Slot axis used when the MIMO source group is itself is_mimo_out:
        # gather is then 2-D (src[node_idx, slot_idx]) instead of 1-D.
        self._nbr_slot_rows: List[List[int]] = []
        self.nbr_slot_routing: Optional[mx.array] = None  # (N, max_nbrs) int32

        # Fan-out MIMO (output) axis
        # _out_rows[i] = list of col_idx values occupying each slot in node i's
        #                output axis. Built at compile time by decoding
        #                output port names through _OUT_COL_PORT_RE.
        self._out_rows: List[List[int]] = []
        self.max_out_nbrs: int = 0
        # Fast reverse lookup: _out_slot_of[node_pos][col_idx] = slot
        self._out_slot_of: List[Dict[int, int]] = []

    @property
    def N(self) -> int:
        return len(self.node_names)

    def _freeze(self):
        """Convert all raw index lists to MLX arrays (called once after build)."""
        for routing_list in self.routings.values():
            for rt in routing_list:
                rt.freeze()

        if self.is_mimo and self._nbr_rows:
            max_w = max(len(row) for row in self._nbr_rows)
            padded = [row + [-1] * (max_w - len(row)) for row in self._nbr_rows]
            arr = mx.array(padded, dtype=mx.int32)
            self.nbr_routing = arr
            # Mask marks slots within each node's DECLARED port count, not just
            # connected-to-FuncBlock slots. InputNode-only ports (declared but
            # with no internal FuncBlock predecessor) still count as valid
            # neighbour slots contributing 0 — matching scalar engine semantics
            # where len(sorted_keys) = number of declared ports. F is separately
            # zeroed at unconnected (nr < 0) slots inside _assemble_feed.
            mask_rows = [[1.0] * len(row) + [0.0] * (max_w - len(row))
                         for row in self._nbr_rows]
            self.nbr_mask = mx.array(mask_rows, dtype=mx.float32)

            # Slot axis (populated only when MIMO source is is_mimo_out)
            if self._nbr_slot_rows:
                slot_padded = [row + [-1] * (max_w - len(row))
                               for row in self._nbr_slot_rows]
                self.nbr_slot_routing = mx.array(slot_padded, dtype=mx.int32)

        if self.is_mimo_out and self._out_rows:
            self.max_out_nbrs = max(len(row) for row in self._out_rows)


# ---------------------------------------------------------------------------
# VectorizedProgram
# ---------------------------------------------------------------------------

class VectorizedProgram:
    """
    Generic vectorized execution engine for any NCD circuit compiled by Canvas.

    Build with:
        vprog = VectorizedProgram.from_program(program, canvas)

    Run with:
        outputs = vprog.run_mlx(inputs, output_group='PR_col/pow_block')
    """

    def __init__(self,
                 groups: Dict[str, NodeGroup],
                 group_schedule: List[str],
                 scc_group_keys: set,
                 external_keys: Optional[set] = None):
        self.groups = groups
        self.group_schedule = group_schedule
        self.scc_group_keys = scc_group_keys
        self.external_keys: set = external_keys or set()
        self._values: Dict[str, Dict[str, mx.array]] = {}
        self._prev:   Dict[str, Dict[str, mx.array]] = {}
        self._input_mc_types: Optional[List[str]] = None

    # ---------------------------------------------------------------------- #
    # Construction                                                             #
    # ---------------------------------------------------------------------- #

    @classmethod
    def from_program(cls, program, canvas) -> "VectorizedProgram":
        """
        Inspect a compiled Program to build a VectorizedProgram automatically.

        Steps
        -----
        A  Invert canvas_inputs: input_node_fq → (col_idx, channel_name)
        B  Group nodes by (base_mc_type, block_id), skipping Input/Output nodes
        C  Sort each group's members by col_idx
        D  Build fq_name → (group_key, position) lookup
        E  Build routing arrays for every group's input ports
        F  Derive group execution schedule from existing dag/scc order
        """
        nodes       = program.nodes
        fan_in      = program._fan_in
        scc_nodes   = program.scc_nodes
        canvas_inputs = program.canvas_inputs   # (mc, pub) → (inp_fq, 'output')

        # ── A. Invert canvas_inputs ──────────────────────────────────────────
        # Maps input_node_fq → (col_idx, channel_name, mc_type). External
        # channels are keyed by f'{mctype}/{chan}' so distinct mc_types (e.g.
        # PR_col and ONOFF_col both declaring public 'input_main') do not
        # collide — the caller decides which mc_type receives the external
        # feed via input_mc_types.
        inp_node_col:    Dict[str, int] = {}
        inp_node_chan:   Dict[str, str] = {}
        inp_node_mctype: Dict[str, str] = {}
        external_keys:   set = set()
        for (mc_name, pub_name), (inp_fq, _) in canvas_inputs.items():
            mctype = _strip_col_idx(mc_name)
            inp_node_col[inp_fq]    = _col_idx_from_name(mc_name)
            inp_node_chan[inp_fq]   = pub_name
            inp_node_mctype[inp_fq] = mctype
            external_keys.add(f'{mctype}/{pub_name}')

        # ── B. Build groups ──────────────────────────────────────────────────
        groups: Dict[str, NodeGroup] = {}

        for fq, node in nodes.items():
            if isinstance(node, (InputNode, OutputNode)):
                continue
            _, blk_id, base_type, col_idx = _parse_fq(fq)
            gk          = f"{base_type}/{blk_id}"
            is_mimo     = _is_col_indexed_mimo(node)
            is_mimo_out = _is_col_indexed_mimo_out(node)

            if gk not in groups:
                g = NodeGroup(gk, is_mimo=is_mimo, is_mimo_out=is_mimo_out)
                if is_mimo_out:
                    # Fan-out: expose channel names (e.g. 'output_val_col')
                    # as the group's ports; per-node slot axes are built in
                    # section C2 below.
                    channels, _cols = _decode_out_channels(list(node.outputs.keys()))
                    g.output_port_names = channels
                else:
                    g.output_port_names = list(node.outputs.keys())

                if isinstance(node, FuncBlock):
                    # Unified algorithms are called directly in batched
                    # mode (no separate register_batched variant needed).
                    # Legacy algorithms fall back to the registered batched
                    # variant for compatibility.
                    if getattr(node.f, '_unified', False):
                        g.fn = node.f
                    else:
                        g.fn = FuncBlock.get_batched(node.f)
                    g.params      = dict(node.params)
                    g.is_stateless = node.stateless
                elif isinstance(node, Rectifier):
                    g.fn           = rectifier_batched
                    g.params       = {'mode': node.mode}
                    g.is_stateless = True
                elif isinstance(node, Division):
                    g.fn     = division_batched
                    g.params = {
                        'port_groups':         node.port_groups,
                        'weighted_mean_pairs': node.weighted_mean_pairs,
                        'eps':                 node.params.get('eps', 1e-9),
                    }
                    g.is_stateless = True
                elif isinstance(node, TemporalFilter):
                    filter_np = node.params.get('filter')
                    if filter_np is None:
                        raise ValueError(
                            f"TemporalFilter '{fq}' has no 'filter' param set; "
                            "call mc.set_block_params(block, {'filter': ...}) before compile_mlx().")
                    filter_np = np.asarray(filter_np, dtype=np.float32)
                    # scipy.ndimage.convolve1d(buf, filter, mode='nearest',
                    # origin=-(F//2)) sampled at the last position reduces to
                    # dot(reverse(filter), buffer) when len(buffer)==len(filter).
                    rev = mx.array(np.flip(filter_np).copy())
                    g.fn           = temporal_filter_batched
                    g.params       = {'filter_rev': rev, 'F_len': int(rev.shape[0])}
                    g.is_stateless = False

                if not is_mimo:
                    g.input_port_names = list(node.inputs.keys())
                    # Pre-populate routing lists so we can append per-member later
                    for p in g.input_port_names:
                        g.routings[p] = []     # will hold List[_PortRouting]

                groups[gk] = g

            groups[gk].node_names.append(fq)
            groups[gk].col_indices.append(col_idx)

        # ── C. Sort each group by col_idx ────────────────────────────────────
        for g in groups.values():
            order          = sorted(range(g.N), key=lambda i: g.col_indices[i])
            g.node_names   = [g.node_names[i]  for i in order]
            g.col_indices  = [g.col_indices[i] for i in order]

        # ── C2. Build fan-out output axis + auto-pack per-neighbor dict params
        for g in groups.values():
            if not g.is_mimo_out:
                continue

            # For each member node, decode its output port names into the
            # slot→col_idx ordering for the FIRST channel, then validate that
            # every other channel uses the same ordering.
            first_channel = g.output_port_names[0]
            for fq in g.node_names:
                node_out_names = list(nodes[fq].outputs.keys())
                channels, cols_per_channel = _decode_out_channels(node_out_names)
                if channels[0] != first_channel or set(channels) != set(g.output_port_names):
                    raise ValueError(
                        f"Fan-out MIMO group '{g.key}' has inconsistent channels "
                        f"across nodes: {channels} vs {g.output_port_names}")
                canonical = cols_per_channel[first_channel]
                for ch in channels[1:]:
                    if cols_per_channel[ch] != canonical:
                        raise ValueError(
                            f"Fan-out MIMO node '{fq}' channel '{ch}' slot order "
                            f"{cols_per_channel[ch]} differs from '{first_channel}' "
                            f"slot order {canonical}. Per-channel axis mismatch is "
                            "not supported.")
                g._out_rows.append(canonical)
                g._out_slot_of.append({c: k for k, c in enumerate(canonical)})

            # Reciprocal neighborhood check: for groups that are BOTH input
            # and output MIMO, the input neighbor axis and output slot axis
            # must match per node. This lets batched algorithms do elementwise
            # math on the (N, max) input tensor and return (N, max) outputs
            # aligned with the output axis.
            # (_nbr_rows is built later in section E, so we can only verify
            # the shape dimension here; axis equality is checked after E.)

            # Auto-pack dict-valued params into (N, max_out) tensors.
            # Any param whose value is a dict is interpreted as a
            # per-neighbor-col lookup and packed against each node's slot
            # axis. The packed array overwrites the original dict under
            # the SAME param name, so user algorithms read params[pname]
            # identically in scalar and batched mode.
            representative = nodes[g.node_names[0]]
            if isinstance(representative, FuncBlock):
                # Use get_original_param to see the pre-packed dict even if
                # a scalar run already converted it to an ndarray.
                dict_param_names = [pname for pname in representative.params
                                    if isinstance(representative.get_original_param(pname), dict)]
                for pname in dict_param_names:
                    rows = []
                    for node_i, fq in enumerate(g.node_names):
                        pval = nodes[fq].get_original_param(pname)
                        if not isinstance(pval, dict):
                            pval = {}
                        slot_cols = g._out_rows[node_i]
                        rows.append([float(pval.get(c, 0.0)) for c in slot_cols])
                    max_out = max(len(r) for r in rows)
                    padded  = [r + [0.0] * (max_out - len(r)) for r in rows]
                    g.params[pname] = mx.array(padded, dtype=mx.float32)

        # ── D. Position lookup ───────────────────────────────────────────────
        node_to_pos: Dict[str, Tuple[str, int]] = {}
        for g in groups.values():
            for pos, fq in enumerate(g.node_names):
                node_to_pos[fq] = (g.key, pos)

        # ── E. Build routing ─────────────────────────────────────────────────
        for g in groups.values():

            if g.is_mimo:
                # ── MIMO: build a 2-D neighbourhood routing tensor ──────────
                # For each group member (in sorted col_idx order):
                #   collect the position of each source in its source group.
                # If the source group is itself is_mimo_out, we also build a
                # parallel slot-axis tensor so gather becomes 2-D:
                #   F = src[node_idx, slot_idx]   instead of   F = src[node_idx]
                src_gk      = None
                src_port    = None    # channel name if src is mimo_out, else real port
                src_is_fout = False

                for fq in g.node_names:
                    node     = nodes[fq]
                    row:      List[int] = []
                    slot_row: List[int] = []
                    consumer_col = _col_idx_from_name(fq)
                    # Use insertion order (== template declared order), not
                    # sorted-by-col-idx. borst_algorithm relies on spiral order
                    # (center, ring-1 CW from north); sorting by col_idx would
                    # scramble that to an arbitrary hex-grid ID order.
                    for port_name in node.inputs.keys():
                        raw_preds = fan_in.get((fq, port_name), [])
                        # Canvas.compile() creates direct inter-MC block edges
                        # that bypass InputNodes. When both kinds of predecessor
                        # exist, the InputNode is stale — drop it.
                        block_preds = [(sf, sp) for sf, sp in raw_preds
                                       if sf not in inp_node_col]
                        preds = block_preds if block_preds else raw_preds
                        # Iterate all preds: skip InputNodes, take first FuncBlock source
                        placed = False
                        for src_fq, src_p in preds:
                            if src_fq in node_to_pos:
                                sk, spos = node_to_pos[src_fq]
                                src_grp  = groups[sk]
                                if src_gk is None:
                                    src_gk = sk
                                    if src_grp.is_mimo_out:
                                        m = _OUT_COL_PORT_RE.match(src_p)
                                        src_port    = m.group(1)  # channel name
                                        src_is_fout = True
                                    else:
                                        src_port    = src_p
                                        src_is_fout = False
                                row.append(spos)
                                if src_grp.is_mimo_out:
                                    # Target col encoded in source port name
                                    # (mimo_out src always sends a named port
                                    # for each downstream col). Consumer's own
                                    # col is the target.
                                    slot = src_grp._out_slot_of[spos].get(consumer_col, -1)
                                    slot_row.append(slot)
                                else:
                                    slot_row.append(-1)
                                placed = True
                                break
                        if not placed:
                            row.append(-1)       # unconnected or InputNode-only
                            slot_row.append(-1)
                    g._nbr_rows.append(row)
                    g._nbr_slot_rows.append(slot_row)

                g.nbr_src_group = src_gk   or ''
                g.nbr_src_port  = src_port or 'output'
                if not src_is_fout:
                    # Reset slot rows to empty so freeze() skips building a
                    # slot-axis tensor.
                    g._nbr_slot_rows = []

            else:
                # ── Standard: per-port routing with fan-in support ──────────
                # For each port, we may have multiple sources (fan-in > 1).
                # Build one _PortRouting entry per "source layer" per port.
                #
                # We iterate members in sorted order, then iterate their
                # fan-in sources in order. Source layer i collects the i-th
                # source for each member (or -1 if that member has fewer than
                # i+1 sources).

                for port_name in g.input_port_names:
                    # Collect each member's source list for this port.
                    # Each entry is a 5-tuple:
                    #   (src_group_key, channel_or_port, node_pos, slot_or_-1, is_fb)
                    # slot is -1 when the source group is not is_mimo_out;
                    # otherwise it's the slot index in the source's (N, max_out)
                    # output tensor, looked up from the src port name suffix.
                    member_sources: List[List[Tuple]] = []
                    for fq in g.node_names:
                        raw_preds = fan_in.get((fq, port_name), [])
                        # Drop stale InputNode preds bypassed by direct inter-MC
                        # block edges (see canvas.compile()). Only when no block
                        # source exists does the InputNode matter — then it
                        # carries external feed.
                        block_preds = [(sf, sp) for sf, sp in raw_preds
                                       if sf not in inp_node_col]
                        preds = block_preds if block_preds else raw_preds
                        srcs: List[Tuple] = []
                        for src_fq, src_p in preds:
                            if src_fq in inp_node_col:
                                ext_key = f'{inp_node_mctype[src_fq]}/{inp_node_chan[src_fq]}'
                                srcs.append(('__external__',
                                             ext_key,
                                             inp_node_col[src_fq],
                                             -1,
                                             False))
                            elif src_fq in node_to_pos:
                                sk, spos = node_to_pos[src_fq]
                                src_grp  = groups[sk]
                                if src_grp.is_mimo_out:
                                    m = _OUT_COL_PORT_RE.match(src_p)
                                    channel     = m.group(1)
                                    target_col  = int(m.group(2))
                                    slot        = src_grp._out_slot_of[spos].get(target_col, -1)
                                    srcs.append((sk, channel, spos, slot,
                                                 src_fq in scc_nodes))
                                else:
                                    srcs.append((sk, src_p, spos, -1,
                                                 src_fq in scc_nodes))
                        member_sources.append(srcs)

                    max_fan_in = max((len(s) for s in member_sources), default=0)

                    layer_routings: List[_PortRouting] = []
                    for layer in range(max_fan_in):
                        indices:   List[int] = []
                        slots:     List[int] = []
                        src_gk    = '__unconnected__'
                        src_port  = port_name
                        is_fb     = False
                        has_slots = False
                        for srcs in member_sources:
                            if layer < len(srcs):
                                sk, sp, spos, slot, fb = srcs[layer]
                                if src_gk == '__unconnected__':
                                    src_gk    = sk
                                    src_port  = sp
                                    is_fb     = fb
                                    has_slots = (slot >= 0)
                                indices.append(spos)
                                slots.append(slot)
                            else:
                                indices.append(-1)   # member has no source at this layer
                                slots.append(-1)

                        layer_routings.append(_PortRouting(
                            src_group_key=src_gk,
                            src_port=src_port,
                            src_indices=indices,
                            src_slot_indices=(slots if has_slots else None),
                            is_feedback=is_fb,
                        ))

                    # Ensure at least one (zero-valued) routing entry
                    if not layer_routings:
                        layer_routings = [_PortRouting('__unconnected__', port_name,
                                                       [-1] * g.N, None, False)]

                    g.routings[port_name] = layer_routings

        # Freeze all routing index lists to mx.arrays
        for g in groups.values():
            g._freeze()

        # ── F. Derive group schedule from existing program schedule ──────────
        seen: set = set()
        group_schedule: List[str] = []
        for node_name in (program.dag_schedule + program.scc_schedule):
            if node_name in node_to_pos:
                gk, _ = node_to_pos[node_name]
                if gk not in seen:
                    group_schedule.append(gk)
                    seen.add(gk)

        scc_group_keys: set = set()
        for node_name in program.scc_schedule:
            if node_name in node_to_pos:
                scc_group_keys.add(node_to_pos[node_name][0])

        return cls(groups=groups,
                   group_schedule=group_schedule,
                   scc_group_keys=scc_group_keys,
                   external_keys=external_keys)

    # ---------------------------------------------------------------------- #
    # Execution                                                                #
    # ---------------------------------------------------------------------- #

    def _get_val(self, gk: str, port: str, use_prev: bool) -> mx.array:
        """
        Retrieve a group's output array from the value store. Shape is
        (N,) for normal groups and (N, max_out_nbrs) for is_mimo_out groups.
        """
        g     = self.groups[gk]
        store = self._prev if use_prev else self._values
        port_store = store.get(gk, {})
        if port in port_store:
            return port_store[port]
        if g.is_mimo_out:
            return mx.zeros((g.N, g.max_out_nbrs))
        return mx.zeros((g.N,))

    def _assemble_feed(self, g: NodeGroup) -> Dict[str, mx.array]:
        """Gather all input arrays for a group from the current value store."""
        feed: Dict[str, mx.array] = {}

        if g.is_mimo:
            # Gather neighbour values into a 2-D tensor
            src_arr = self._get_val(g.nbr_src_group, g.nbr_src_port, use_prev=False)
            nr   = g.nbr_routing                         # (N, max_nbrs)
            mask = g.nbr_mask                            # (N, max_nbrs) float32
            safe_n = mx.where(nr >= 0, nr, mx.zeros_like(nr))

            if g.nbr_slot_routing is not None:
                # Source is fan-out mimo_out: 2-D gather
                sr     = g.nbr_slot_routing
                safe_s = mx.where(sr >= 0, sr, mx.zeros_like(sr))
                F      = src_arr[safe_n, safe_s]          # (N, max_nbrs)
            else:
                F = src_arr[safe_n]                       # (N, max_nbrs)
            F = mx.where(nr >= 0, F, mx.zeros_like(F))
            feed['neighbors']     = F
            feed['neighbor_mask'] = mask

        else:
            for port_name, routing_list in g.routings.items():
                total = mx.zeros((g.N,))
                for rt in routing_list:
                    if rt.src_group_key == '__unconnected__':
                        continue   # contributes zero

                    if rt.src_group_key == '__external__':
                        ext  = self._values.get('__external__', {})
                        src  = ext.get(rt.src_port, mx.zeros((g.N,)))
                    else:
                        src = self._get_val(rt.src_group_key, rt.src_port,
                                            use_prev=rt.is_feedback)

                    idx   = rt.src_indices                  # (N,) int32
                    valid = (idx >= 0)
                    safe  = mx.where(valid, idx, mx.zeros_like(idx))

                    if rt.src_slot_indices is not None:
                        # Fan-out mimo_out source: 2-D gather (node, slot)
                        slot      = rt.src_slot_indices
                        safe_slot = mx.where(slot >= 0, slot, mx.zeros_like(slot))
                        gathered  = src[safe, safe_slot]
                    else:
                        gathered  = src[safe]
                    total = total + mx.where(valid, gathered, mx.zeros((g.N,)))

                feed[port_name] = total

        return feed

    def run_step(self, t: float, dt: float, x: mx.array) -> None:
        """
        Execute one timestep in-place.

        x : external input array, shape (N_cols,) indexed by col_idx.
        Results are stored in self._values; probe with get_group_output().
        """
        self._prev = {gk: dict(vd) for gk, vd in self._values.items()}
        self._values.clear()
        # Build external feed using mc_type-qualified keys (e.g. 'PR_col/input_main').
        # If no input_mc_types filter was set, broadcast x to every discovered
        # external channel — the simple single-mc_type case. When multiple
        # mc_types declare the same public input, the caller MUST pass
        # input_mc_types to disambiguate which mc_type receives the feed.
        if self._input_mc_types is None:
            ext = {key: x for key in self.external_keys}
        else:
            ext = {}
            for mct in self._input_mc_types:
                prefix = f'{mct}/'
                for key in self.external_keys:
                    if key.startswith(prefix):
                        ext[key] = x
        self._values['__external__'] = ext

        for gk in self.group_schedule:
            g = self.groups[gk]

            if g.fn is None:
                # No registered batched function — pass zeros through
                if g.is_mimo_out:
                    shape = (g.N, g.max_out_nbrs)
                else:
                    shape = (g.N,)
                self._values[gk] = {p: mx.zeros(shape)
                                    for p in g.output_port_names}
                continue

            feed = self._assemble_feed(g)

            if g.is_stateless:
                out         = g.fn(feed, g.params)
            else:
                out, g.state = g.fn(feed, g.params, g.state)

            # Fan-out MIMO: unified algorithms return '<channel>_neighbors'
            # keys. Engine stores under the bare channel name. Only strip
            # the suffix when the stripped result matches a declared output
            # channel — other keys ending in '_neighbors' pass through
            # unchanged to avoid collisions with user-defined port names.
            if g.is_mimo_out:
                out_channels = set(g.output_port_names)
                normalized: Dict[str, mx.array] = {}
                for key, val in out.items():
                    if key.endswith('_neighbors'):
                        bare = key[: -len('_neighbors')]
                        if bare in out_channels:
                            normalized[bare] = val
                            continue
                    normalized[key] = val
                out = normalized

            self._values[gk] = out

    def get_group_output(self, group_key: str, port: str = 'output') -> mx.array:
        """Return the current output array for a group after run_step()."""
        return self._values.get(group_key, {}).get(port, mx.zeros((self.groups[group_key].N,)))

    def run_mlx(self,
                inputs: np.ndarray,
                output_group: str,
                output_port: str = 'output',
                input_mc_types: Optional[List[str]] = None) -> np.ndarray:
        """
        Run all T timesteps and return the output array.

        Parameters
        ----------
        inputs : (T, N_cols) float32/float64 numpy array
            External input.  Row t is passed as 'input_main' and indexed
            by col_idx — identical semantics to Program.run_program().
        output_group : str
            NodeGroup key to probe, e.g. 'PR_col/pow_block'.
        output_port : str
            Output port name within that group, default 'output'.

        Returns
        -------
        (T, N_cols) float32 numpy array — outputs in col_idx order.
        """
        T, N_cols = inputs.shape
        g         = self.groups[output_group]

        result = np.zeros((T, N_cols), dtype=np.float32)

        # Reset all group states
        for grp in self.groups.values():
            grp.state = {}
        self._values.clear()
        self._prev.clear()
        self._input_mc_types = input_mc_types

        for t_idx in range(T):
            x = mx.array(inputs[t_idx], dtype=mx.float32)
            self.run_step(float(t_idx) / 60.0, 1.0 / 60.0, x)

            raw = self.get_group_output(output_group, output_port)
            mx.eval(raw)

            # g.col_indices[k] is the col_idx of position k in raw
            result[t_idx, g.col_indices] = np.array(raw, copy=False)

        return result

    def run_mlx_multi(self,
                      inputs: np.ndarray,
                      probes: List[Tuple[str, str]],
                      n_cols_out: Optional[int] = None,
                      input_mc_types: Optional[List[str]] = None,
                      ) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Run all T timesteps and return multiple outputs from a single pass.

        Parameters
        ----------
        inputs : (T, N_cols) numpy array
            External 'input_main' channel, indexed by col_idx.
        probes : list of (group_key, output_port)
            Each probe is collected into its own (T, N_cols_out) result array
            using that group's col_indices for placement.
        n_cols_out : int, optional
            Width of the returned arrays. Defaults to inputs.shape[1].

        Returns
        -------
        {(group_key, output_port): (T, n_cols_out) float32 np.array}
        """
        T, N_cols_in = inputs.shape
        if n_cols_out is None:
            n_cols_out = N_cols_in

        results: Dict[Tuple[str, str], np.ndarray] = {
            p: np.zeros((T, n_cols_out), dtype=np.float32) for p in probes
        }
        probe_groups = {p: self.groups[p[0]] for p in probes}

        # Reset all group states
        for grp in self.groups.values():
            grp.state = {}
        self._values.clear()
        self._prev.clear()
        self._input_mc_types = input_mc_types

        for t_idx in range(T):
            x = mx.array(inputs[t_idx], dtype=mx.float32)
            self.run_step(float(t_idx) / 60.0, 1.0 / 60.0, x)

            for probe in probes:
                gk, port = probe
                raw = self.get_group_output(gk, port)
                mx.eval(raw)
                g = probe_groups[probe]
                results[probe][t_idx, g.col_indices] = np.array(raw, copy=False)

        return results
