"""
neurocircuitdesk.microcircuit_viz
---------------------------------
Standalone 3D Plotly visualisation for a *single* :class:`MicroCircuit`.

Lives outside Canvas — no canvas-level batching, no retinotopic
placement — so it can render a template's internal structure as soon as
the MC has been built (i.e. after the template's ``add_block`` /
``connect`` / ``specify_io`` calls have populated ``mc._viz_nodes`` and
``mc.input_ports`` / ``mc.output_ports``).

Auto-layout is driven by the MC's category, inferred from port names:

- **CMC** — every public input goes through one column. Input port
  markers stack vertically above the block cluster; output port markers
  below.
- **iCMC** — at least one public input port matches
  ``<prefix>_col_<N>``. Port names are grouped by ``<prefix>`` and each
  group is laid out in a local **axial-hex** pattern derived from each
  column's spiral position relative to ``mc.col_idx`` (so the centre
  appears at the centre of the hex, ring-1 cells around it, etc.).
  Multiple channels (e.g. LPLC2's ``input_a/b/c/d``) sit side-by-side
  along the x-axis.

Reused: ``mc._viz_nodes`` (block positions established by ``add_block``)
and ``mc._viz_edges`` (intra-MC arrows from ``mc.connect``). No
positions are recomputed for blocks; only port-marker positions are
synthesised by the layout.

Usage
~~~~~

>>> from neurocircuitdesk.microcircuit_viz import MicroCircuitViz
>>> fig = MicroCircuitViz(mc).plot()      # returns a plotly.graph_objs.Figure
>>>
>>> # Or via the MicroCircuit.show() shortcut:
>>> fig = mc.show()
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


_COL_PORT_RE = re.compile(r'(.+)_col_(\d+)$')


# ── Hex / spiral helpers ───────────────────────────────────────────────

def _spiral_to_axial(spiral_idx: int) -> Tuple[int, int]:
    """Spiral index (0=centre, 1..6=ring-1 CW from north) → axial (q, r).

    Mirrors the ordering produced by
    :meth:`Canvas.graph_utils.local_order` / ``_local_order_pure``.
    """
    if spiral_idx == 0:
        return (0, 0)
    k = 1
    while 1 + 3 * k * (k + 1) < spiral_idx + 1:
        k += 1
    start = 1 + 3 * k * (k - 1)
    offset = spiral_idx - start
    side = offset // k
    step = offset % k
    corners = [(0, -k), (k, -k), (k, 0), (0, k), (-k, k), (-k, 0)]
    a = corners[side]
    b = corners[(side + 1) % 6]
    dq = 0 if (b[0] - a[0]) == 0 else (b[0] - a[0]) // abs(b[0] - a[0])
    dr = 0 if (b[1] - a[1]) == 0 else (b[1] - a[1]) // abs(b[1] - a[1])
    return (a[0] + dq * step, a[1] + dr * step)


def _axial_to_xy(q: int, r: int, spacing: float = 0.2) -> Tuple[float, float]:
    """Flat-topped axial → pixel (x, y) for local hex layout."""
    size = spacing / np.sqrt(3)
    x = 1.5 * size * q
    y = np.sqrt(3) * size * (r + 0.5 * q)
    return float(x), float(y)


# ── MicroCircuitViz ────────────────────────────────────────────────────

class MicroCircuitViz:
    """Standalone 3D viz for one :class:`MicroCircuit` (canvas-free).

    Parameters
    ----------
    mc : MicroCircuit
        Must have its ``_viz_nodes``, ``_viz_edges``, ``input_ports``,
        ``output_ports`` populated (i.e. its template body has finished
        executing). Does not need to be wired into a Canvas.
    width, height : int
        Figure dimensions in pixels.
    title : str, optional
        Figure title. Defaults to ``mc.name`` plus a ``(CMC)`` / ``(iCMC)``
        annotation.

    Notes
    -----
    The visualisation deliberately does NOT show per-block port markers
    or intra-MC port positions — for sanity-checking topology, the
    public I/O and block-to-block arrows are enough, and per-port noise
    dominates the picture quickly. If you want the full Canvas-level
    rendering with port markers, use ``Canvas.show()`` after the MC has
    been added to a canvas.
    """

    INPUT_Z_OFFSET = 0.6     # added on top of the top-most block's z
    OUTPUT_Z_OFFSET = -0.6   # subtracted from the bottom-most block's z

    def __init__(self, mc, *, width: int = 600, height: int = 450,
                 title: Optional[str] = None):
        self.mc = mc
        self.width = width
        self.height = height
        self._title_override = title

    # ── Public entry point ──

    def plot(self) -> go.Figure:
        """Return a :class:`plotly.graph_objs.Figure` of this MC."""
        fig = go.Figure()
        self._add_blocks(fig)
        self._add_intra_mc_arrows(fig)
        category = self._classify()
        input_positions = self._lay_out_input_ports(category)
        output_positions = self._lay_out_output_ports(category)
        self._add_port_markers(fig, input_positions, symbol='circle',
                                name='public inputs', size=6, color='#3366cc')
        self._add_port_markers(fig, output_positions, symbol='diamond',
                                name='public outputs', size=6, color='#cc3333')
        self._add_io_routing_arrows(fig, input_positions, output_positions)

        title = self._title_override or f'{self.mc.name or "MicroCircuit"}  ({category})'
        fig.update_layout(
            title=title,
            scene=dict(
                aspectmode='data',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            width=self.width, height=self.height,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
        )
        return fig

    # ── Classification ──

    def _classify(self) -> str:
        """Return ``'CMC'`` or ``'iCMC'`` based on port-name pattern."""
        for name in self.mc.input_ports:
            if _COL_PORT_RE.match(name):
                return 'iCMC'
        return 'CMC'

    # ── Blocks ──

    def _add_blocks(self, fig: go.Figure) -> None:
        by_color = defaultdict(lambda: ([], [], [], []))
        for block_id, block in self.mc._viz_nodes.items():
            meta = block.get('meta', {})
            x, y, z = meta.get('x', 0.0), meta.get('y', 0.0), meta.get('z', 0.0)
            color = meta.get('color', 'cyan')
            xs, ys, zs, ht = by_color[color]
            xs.append(float(x))
            ys.append(float(y))
            zs.append(float(z))
            ht.append(block_id)
        for color, (xs, ys, zs, ht) in by_color.items():
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='markers+text',
                marker=dict(size=18, color=color, opacity=0.9,
                            line=dict(color='#333333', width=1)),
                text=ht, textposition='middle center',
                textfont=dict(size=10, color='black'),
                hoverinfo='text', hovertext=ht,
                name=f'blocks ({color})',
            ))

    def _add_intra_mc_arrows(self, fig: go.Figure) -> None:
        by_color = defaultdict(lambda: ([], [], []))
        for arrow_list in self.mc._viz_edges.values():
            for arr in arrow_list:
                if not isinstance(arr, dict) or arr.get('kind') != 'arrow':
                    continue
                color = arr.get('color', '#333333')
                src = arr['src']
                dst = arr['dst']
                xs, ys, zs = by_color[color]
                xs += [src[0], dst[0], None]
                ys += [src[1], dst[1], None]
                zs += [src[2], dst[2], None]
        for color, (xs, ys, zs) in by_color.items():
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines',
                line=dict(color=color, width=2),
                hoverinfo='none',
                name=f'intra-MC ({color})',
            ))

    # ── Public-port layout ──

    def _block_top_z(self) -> float:
        if not self.mc._viz_nodes:
            return 0.0
        return max(b.get('meta', {}).get('z', 0.0)
                   for b in self.mc._viz_nodes.values())

    def _block_bot_z(self) -> float:
        if not self.mc._viz_nodes:
            return 0.0
        return min(b.get('meta', {}).get('z', 0.0)
                   for b in self.mc._viz_nodes.values())

    def _col_to_spiral_lookup(self) -> Dict[int, int]:
        """Map col_idx → its position in mc.canvas.graph_utils.local_order().

        Returns an empty dict if no canvas is attached or graph utilities
        aren't available — the iCMC layout then degrades gracefully to
        spiral_idx=0 for all cols.
        """
        try:
            spiral = self.mc.canvas.graph_utils.local_order(
                self.mc.col_idx, num_rings=10, require_in_graph=False)
            return {int(c): i for i, c in enumerate(spiral)}
        except Exception:
            return {}

    def _group_ports_by_channel(self, port_names: List[str]
                                  ) -> Dict[str, List[Tuple[str, Optional[int]]]]:
        """Split ``input_col_5`` / ``input_a_col_7`` etc. into ``{prefix: [(name, col), ...]}``.

        Port names that don't match ``_col_<N>`` go under ``'<scalar>'``.
        """
        out: Dict[str, List[Tuple[str, Optional[int]]]] = defaultdict(list)
        for name in port_names:
            m = _COL_PORT_RE.match(name)
            if m:
                out[m.group(1)].append((name, int(m.group(2))))
            else:
                out['<scalar>'].append((name, None))
        return out

    def _lay_out_input_ports(self, category: str
                              ) -> Dict[str, Tuple[float, float, float]]:
        z_in = self._block_top_z() + self.INPUT_Z_OFFSET
        if category == 'CMC':
            return self._lay_out_scalar_ports(
                list(self.mc.input_ports.keys()), z=z_in, dz_per_port=0.18)
        return self._lay_out_hex_ports(
            list(self.mc.input_ports.keys()), z=z_in)

    def _lay_out_output_ports(self, category: str
                               ) -> Dict[str, Tuple[float, float, float]]:
        z_out = self._block_bot_z() + self.OUTPUT_Z_OFFSET
        if category == 'CMC':
            return self._lay_out_scalar_ports(
                list(self.mc.output_ports.keys()), z=z_out, dz_per_port=-0.18)
        return self._lay_out_hex_ports(
            list(self.mc.output_ports.keys()), z=z_out)

    def _lay_out_scalar_ports(self, names: List[str], *, z: float,
                                dz_per_port: float
                                ) -> Dict[str, Tuple[float, float, float]]:
        """Stack non-MIMO ports vertically along z."""
        return {name: (0.0, 0.0, z + i * dz_per_port)
                for i, name in enumerate(names)}

    def _lay_out_hex_ports(self, names: List[str], *, z: float,
                            channel_x_offset: float = 1.1,
                            hex_spacing: float = 0.18
                            ) -> Dict[str, Tuple[float, float, float]]:
        """Lay out MIMO ports in per-channel hex patterns at z.

        Each channel prefix gets its own hex cluster, offset along x so
        they don't overlap. Within a cluster, each column's position is
        derived from its spiral index relative to ``mc.col_idx``.
        Scalar ports (no ``_col_<N>``) get a small fan above the hex
        cluster.
        """
        positions: Dict[str, Tuple[float, float, float]] = {}
        groups = self._group_ports_by_channel(names)
        col_to_spiral = self._col_to_spiral_lookup()

        channel_keys = list(groups.keys())
        n_channels = len(channel_keys)
        for ci, prefix in enumerate(channel_keys):
            channel_dx = (ci - (n_channels - 1) / 2.0) * channel_x_offset
            for name, col in groups[prefix]:
                if col is None:
                    # scalar port within an iCMC: float just above the
                    # channel column
                    positions[name] = (channel_dx, 0.0, z + 0.15)
                else:
                    spiral_idx = col_to_spiral.get(col, 0)
                    q, r = _spiral_to_axial(spiral_idx)
                    dx, dy = _axial_to_xy(q, r, spacing=hex_spacing)
                    positions[name] = (channel_dx + dx, dy, z)
        return positions

    # ── Port markers + routing arrows ──

    def _add_port_markers(self, fig: go.Figure,
                           positions: Dict[str, Tuple[float, float, float]],
                           *, symbol: str, name: str,
                           size: int, color: str) -> None:
        if not positions:
            return
        xs, ys, zs, ht = [], [], [], []
        for label, (x, y, z) in positions.items():
            xs.append(x); ys.append(y); zs.append(z); ht.append(label)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers',
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(color='#222222', width=1), opacity=0.9),
            text=ht, hoverinfo='text', name=name,
        ))

    def _add_io_routing_arrows(self, fig: go.Figure,
                                 input_positions: Dict[str, Tuple[float, float, float]],
                                 output_positions: Dict[str, Tuple[float, float, float]]
                                 ) -> None:
        """Dashed light-grey arrows from input markers to internal blocks
        and from internal blocks to output markers."""
        xs, ys, zs = [], [], []
        for pub_name, src in input_positions.items():
            for block_id, _port in self.mc.input_ports.get(pub_name, []):
                if block_id not in self.mc._viz_nodes:
                    continue
                meta = self.mc._viz_nodes[block_id].get('meta', {})
                dst = (float(meta.get('x', 0)),
                       float(meta.get('y', 0)),
                       float(meta.get('z', 0)))
                xs += [src[0], dst[0], None]
                ys += [src[1], dst[1], None]
                zs += [src[2], dst[2], None]
        for pub_name, dst in output_positions.items():
            mapping = self.mc.output_ports.get(pub_name)
            if mapping is None:
                continue
            block_id, _port = mapping
            if block_id not in self.mc._viz_nodes:
                continue
            meta = self.mc._viz_nodes[block_id].get('meta', {})
            src = (float(meta.get('x', 0)),
                   float(meta.get('y', 0)),
                   float(meta.get('z', 0)))
            xs += [src[0], dst[0], None]
            ys += [src[1], dst[1], None]
            zs += [src[2], dst[2], None]
        if xs:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines',
                line=dict(color='#999999', width=1, dash='dot'),
                hoverinfo='none', name='I/O routes',
            ))


__all__ = ['MicroCircuitViz']
