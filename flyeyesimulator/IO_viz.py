"""
flyeyesimulator.IO_viz
----------------------
``IOViz`` — combined screen + retina-panel grid as a single animated
Plotly figure with one synchronised time slider.

Composes :class:`ScreenViz` / :class:`RetinaViz` outputs into one figure
without re-implementing their mesh assembly. Two supported layouts:

  - **1-row**: Screen on the far left, ``N`` retina panels in cols 2..N+1.
  - **2-row**: Screen on the far left spanning both rows (big), ``N`` retina
    panels per row on the right (rows may have different counts).

Each retina panel can have its own colour range; the screen has its own.
Per-panel colorbars are placed adjacent to their scenes — left of the
screen, right of each retina.

Typical use::

    from flyeyesimulator import IOViz

    iv = IOViz(screen=screen, retina_viz=ret_viz)
    fig = iv.save_video(
        retina_values=[lum_TN, V_R1_TN, V_avg_TN],
        titles=['R1 photon', 'R1 voltage', 'R1–R6 voltage'],
        retina_crange=[(0, 3e5), (-80, 0), (-80, 0)],
        screen_crange=(0, 3e5),
        fps=15, dt_ms=10,
        html_path='out.html',
    )

Pass nested lists for the 2-row layout::

    fig = iv.save_video(
        retina_values=[[a1, a2, a3], [b1, b2, b3]],
        titles=[['Static R1', 'Static avg', 'LMC'],
                ['Active R1', 'Active avg', 'LMC ama']],
        retina_crange=[[(-80, -20), (-80, -20), (-65, -45)],
                       [(-80, -20), (-80, -20), (-65, -45)]],
    )
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backend import to_numpy
from .retina import Retina
from .retinaviz import RetinaViz
from .screen import Screen

__all__ = ['IOViz']

CRange = Tuple[Optional[float], Optional[float]]
Values1Row = Sequence[np.ndarray]
Values2Row = Sequence[Sequence[np.ndarray]]


# Camera defaults match the ones in `_run_final_vids.ipynb` — head-on
# views that frame the retina + hemisphere together.
_DEFAULT_SCREEN_CAMERA = dict(
    eye=dict(x=0, y=-3.5, z=0),
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
)
_DEFAULT_RETINA_CAMERA = dict(
    eye=dict(x=0, y=-2.35, z=0),
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
)
_FLAT_LIGHTING = dict(ambient=1.0, diffuse=0.0, specular=0.0)
_HIDDEN_AXIS = dict(showbackground=False, visible=False)


class IOViz:
    """Screen + retina-panel grid as one synchronised Plotly video.

    Parameters
    ----------
    screen : :class:`flyeyesimulator.Screen`
        Source of the spherical projection. ``screen.intensities`` must
        still be alive (i.e. don't call any simulator's ``.run()`` with
        ``release=True`` before constructing the ``IOViz``).
    retina_viz : :class:`flyeyesimulator.RetinaViz`
        Mesh builder + the retina geometry that every retina panel
        inherits. All retina panels share one mesh; the only thing that
        varies frame-to-frame is each panel's intensity array.
    screen_camera, retina_camera : dict, optional
        Plotly scene cameras. Defaults match the ``_run_final_vids``
        prototype's framing.
    colorscale, screen_colorscale : str
        Default colour scales for retina panels and the screen,
        respectively. ``save_video(colorscale=...)`` overrides the
        retina default for one call.
    lighting : dict, optional
        Custom per-trace ``lighting`` override. Defaults to flat
        shading (``ambient=1, diffuse=0, specular=0``).
    screen_colorbar_title, retina_colorbar_title : str
        Strings used on the colorbars. The retina colorbar title is
        shared by every retina panel.
    screen_downsample : int
        Spatial downsample factor for the screen surface (every Nth
        row and column of ``parallels × meridians``). Default ``4``
        is what the FES demo uses to keep inline Plotly rendering
        responsive — at full resolution a 300-frame 256×256 screen
        produces ~150 MB of surfacecolor data, which crashes most
        browsers when displayed inline. Pass ``1`` for full
        resolution (HTML export only; do not display inline).

    Notes
    -----
    Construction precomputes screen geometry via
    :meth:`Screen.get_sphere_geometry` and retina mesh via
    :meth:`RetinaViz.get_r1_mesh` — both publicly cached helpers. No
    private attribute access on either class.
    """

    def __init__(
        self,
        screen: Screen,
        retina_viz: RetinaViz,
        *,
        screen_camera: Optional[dict] = None,
        retina_camera: Optional[dict] = None,
        colorscale: str = 'Viridis',
        screen_colorscale: str = 'gray',
        lighting: Optional[dict] = None,
        screen_colorbar_title: str = 'photon/s',
        retina_colorbar_title: str = 'mV',
        screen_downsample: int = 4,
    ):
        self.screen = screen
        self.retina_viz = retina_viz

        # Screen geometry + intensities (numpy, T x P x M).
        ds = max(1, int(screen_downsample))
        self._screen_downsample = ds
        X, Y, Z = screen.get_sphere_geometry()
        self._sc_X = X[::ds, ::ds]
        self._sc_Y = Y[::ds, ::ds]
        self._sc_Z = Z[::ds, ::ds]
        self._sc_intens = to_numpy(screen.intensities)[:, ::ds, ::ds]
        self._T_screen = int(self._sc_intens.shape[0])

        # Retina mesh (cached on RetinaViz; we just unpack it).
        mesh = retina_viz.get_r1_mesh()
        self._mx = mesh['x']
        self._my = mesh['y']
        self._mz = mesh['z']
        self._mi = mesh['i']
        self._mj = mesh['j']
        self._mk = mesh['k']
        self._kept_col_idx = mesh['kept_col_idx']
        self._verts_per_lens = int(mesh['verts_per_lens'])

        self._screen_camera = screen_camera or _DEFAULT_SCREEN_CAMERA
        self._retina_camera = retina_camera or _DEFAULT_RETINA_CAMERA
        self._colorscale = colorscale
        self._screen_colorscale = screen_colorscale
        self._lighting = lighting or _FLAT_LIGHTING
        self._screen_cbar_title = screen_colorbar_title
        self._retina_cbar_title = retina_colorbar_title

    # ─── public API ──────────────────────────────────────────────────────

    def save_video(
        self,
        retina_values: Union[Values1Row, Values2Row],
        titles: Union[Sequence[str], Sequence[Sequence[str]]],
        *,
        screen_crange: CRange = (None, None),
        retina_crange: Optional[Union[Sequence[CRange],
                                      Sequence[Sequence[CRange]]]] = None,
        fps: int = 30,
        dt_ms: float = 10,
        html_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        colorscale: Optional[str] = None,
        frame_stride: int = 1,
    ) -> go.Figure:
        """Build the animated figure and (optionally) write HTML.

        Parameters
        ----------
        retina_values : list of (T, N_cols) arrays, or nested list
            Flat list → 1-row layout. Nested list of two lists → 2-row
            layout (top row, bottom row).
        titles : list of str, or nested list of str
            Subplot titles. Must mirror the shape of ``retina_values``.
        screen_crange : (float, float)
            ``(cmin, cmax)`` for the screen colour scale. Either entry
            may be ``None`` to use the data extremum.
        retina_crange : optional, list of (cmin, cmax) tuples
            Per-panel colour ranges, matching the shape of
            ``retina_values``. Use ``None`` for a panel to auto-fit.
            Pass ``None`` for the whole arg to auto-fit every panel.
        fps : int
            Animation rate. Affects the ``Play`` button's frame
            duration, not the slider step density.
        dt_ms : float
            Simulator step duration in milliseconds. Used to label the
            time slider in ms (``label = int(t * dt_ms)``).
        html_path : str or Path, optional
            If given, ``fig.write_html`` is called with the path.
        width, height : int, optional
            Figure dimensions in px. Layout-aware defaults if omitted.
        colorscale : str, optional
            Override for retina panel colour scale on this call only.
            Defaults to the constructor's ``colorscale``.
        frame_stride : int
            Keep every Nth frame in the animation. Default ``1`` (no
            stride). For very long sequences (≥ 300 frames at full
            resolution) bump to ``2`` or ``5`` to keep the figure
            small enough for inline Plotly rendering. Independent of
            ``screen_downsample`` (which is spatial).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        layout_kind, flat_panels, flat_titles, flat_crange = self._flatten_panels(
            retina_values, titles, retina_crange)

        # Validate shapes against the FULL-rate screen length.
        T_full = self._T_screen
        for label, arr in zip(flat_titles, flat_panels):
            if arr.ndim != 2:
                raise ValueError(
                    f"retina panel {label!r} must be 2D (T, N_cols); "
                    f"got shape {arr.shape}.")
            if arr.shape[0] != T_full:
                raise ValueError(
                    f"retina panel {label!r} has T={arr.shape[0]}, but the "
                    f"screen has T={T_full}. The simulator must have been run "
                    f"with release=False (so screen.intensities is alive) "
                    f"and the panel array must come from the same run.")

        # Apply frame stride.
        stride = max(1, int(frame_stride))
        T = (T_full + stride - 1) // stride
        sc_intens = self._sc_intens[::stride]
        flat_panels = [arr[::stride] for arr in flat_panels]
        # dt_ms label still scales with the original simulator step, so
        # multiply by stride so the displayed time matches wall-clock.
        effective_dt_ms = dt_ms * stride

        colorscale = colorscale or self._colorscale
        s_cmin, s_cmax = self._resolve_crange(sc_intens, screen_crange)
        panel_cranges = [
            self._resolve_crange(arr, cr if cr is not None else (None, None))
            for arr, cr in zip(flat_panels, flat_crange)
        ]

        # Build the subplot grid + per-panel cell coordinates.
        fig, screen_cell, panel_cells = self._make_grid(
            layout_kind, flat_titles, num_panels_top=self._n_top,
            num_panels_bot=self._n_bot)

        # Screen trace (col 1).
        sx_dom = fig.layout['scene'].domain.x
        fig.add_trace(
            go.Surface(
                x=self._sc_X, y=self._sc_Y, z=self._sc_Z,
                surfacecolor=sc_intens[0],
                colorscale=self._screen_colorscale,
                cmin=s_cmin, cmax=s_cmax,
                colorbar=dict(
                    title=self._screen_cbar_title,
                    x=sx_dom[0] - 0.005,
                    xanchor='right',
                    len=0.5 if layout_kind == '2rows' else 0.7,
                    thickness=15,
                ),
                showscale=True,
                hoverinfo='skip',
                lighting=self._lighting,
            ),
            row=screen_cell[0], col=screen_cell[1],
        )

        # Retina traces.
        for panel_idx, (data, title, (cmin, cmax), cell) in enumerate(
                zip(flat_panels, flat_titles, panel_cranges, panel_cells)):
            scene_key = self._scene_key(panel_idx + 2)  # +1 for screen, +1 for 1-indexed
            scene_dom_x = fig.layout[scene_key].domain.x
            scene_dom_y = fig.layout[scene_key].domain.y
            cbar_y = 0.5 * (scene_dom_y[0] + scene_dom_y[1])
            cbar_len = max(0.25, min(0.7, scene_dom_y[1] - scene_dom_y[0] - 0.05))

            fig.add_trace(
                go.Mesh3d(
                    x=self._mx, y=self._my, z=self._mz,
                    i=self._mi, j=self._mj, k=self._mk,
                    intensity=self._panel_intensity(data, 0),
                    colorscale=colorscale,
                    cmin=cmin, cmax=cmax,
                    colorbar=dict(
                        title=self._retina_cbar_title,
                        x=scene_dom_x[1] + 0.005,
                        xanchor='left',
                        y=cbar_y,
                        len=cbar_len,
                        thickness=10,
                    ),
                    showscale=True,
                    flatshading=True,
                    hoverinfo='skip',
                    lighting=self._lighting,
                ),
                row=cell[0], col=cell[1],
            )

        # Frames (synchronised across all traces).
        n_traces = 1 + len(flat_panels)
        frames: List[go.Frame] = []
        for t in range(T):
            frame_data: List[go.BaseTraceType] = [
                go.Surface(surfacecolor=sc_intens[t],
                           cmin=s_cmin, cmax=s_cmax)
            ]
            for data, (cmin, cmax) in zip(flat_panels, panel_cranges):
                frame_data.append(go.Mesh3d(
                    intensity=self._panel_intensity(data, t),
                    cmin=cmin, cmax=cmax,
                    lighting=self._lighting,
                ))
            frames.append(go.Frame(
                name=str(t),
                data=frame_data,
                traces=list(range(n_traces)),
            ))
        fig.frames = frames

        # Layout (sizing, cameras, controls, slider).
        self._apply_layout(
            fig, layout_kind, n_total_cols=self._n_grid_cols,
            T=T, fps=fps, dt_ms=effective_dt_ms,
            width=width, height=height,
        )

        if html_path is not None:
            fig.write_html(str(html_path))

        return fig

    # ─── internals ───────────────────────────────────────────────────────

    def _flatten_panels(self, retina_values, titles, retina_crange):
        """Detect 1-row vs 2-row layout and produce flat panel lists.

        Returns ``(layout_kind, panels, titles, cranges)`` where panels
        / titles / cranges are flat lists. Also stashes ``self._n_top``,
        ``self._n_bot``, ``self._n_grid_cols`` for grid construction.
        """
        if len(retina_values) == 0:
            raise ValueError("retina_values is empty — pass at least one panel.")

        first = retina_values[0]
        is_2rows = isinstance(first, (list, tuple)) and not isinstance(first, np.ndarray)

        if is_2rows:
            if len(retina_values) != 2:
                raise ValueError(
                    f"2-row layout requires exactly 2 rows of panels; "
                    f"got {len(retina_values)}.")
            top_vals, bot_vals = retina_values
            top_titles, bot_titles = titles
            if len(top_titles) != len(top_vals) or len(bot_titles) != len(bot_vals):
                raise ValueError(
                    f"titles shape doesn't match retina_values: "
                    f"got {len(top_titles)}/{len(bot_titles)} titles for "
                    f"{len(top_vals)}/{len(bot_vals)} panels.")

            self._n_top = len(top_vals)
            self._n_bot = len(bot_vals)
            self._n_grid_cols = 1 + max(self._n_top, self._n_bot)

            if retina_crange is None:
                top_cr = [(None, None)] * self._n_top
                bot_cr = [(None, None)] * self._n_bot
            else:
                if len(retina_crange) != 2:
                    raise ValueError(
                        "retina_crange must be a 2-element list for 2-row layout.")
                top_cr, bot_cr = retina_crange
                top_cr = list(top_cr) if top_cr is not None else [(None, None)] * self._n_top
                bot_cr = list(bot_cr) if bot_cr is not None else [(None, None)] * self._n_bot
                if len(top_cr) != self._n_top or len(bot_cr) != self._n_bot:
                    raise ValueError(
                        f"retina_crange shape doesn't match: "
                        f"got {len(top_cr)}/{len(bot_cr)} ranges for "
                        f"{self._n_top}/{self._n_bot} panels.")

            panels = [np.asarray(v) for v in top_vals] + [np.asarray(v) for v in bot_vals]
            flat_titles = list(top_titles) + list(bot_titles)
            flat_crange = list(top_cr) + list(bot_cr)
            return '2rows', panels, flat_titles, flat_crange

        # 1-row.
        if len(titles) != len(retina_values):
            raise ValueError(
                f"titles length ({len(titles)}) doesn't match retina_values "
                f"length ({len(retina_values)}).")
        self._n_top = len(retina_values)
        self._n_bot = 0
        self._n_grid_cols = 1 + self._n_top

        if retina_crange is None:
            flat_crange = [(None, None)] * self._n_top
        else:
            if len(retina_crange) != self._n_top:
                raise ValueError(
                    f"retina_crange length ({len(retina_crange)}) doesn't match "
                    f"retina_values length ({self._n_top}).")
            flat_crange = list(retina_crange)

        panels = [np.asarray(v) for v in retina_values]
        return '1row', panels, list(titles), flat_crange

    def _make_grid(self, layout_kind, flat_titles, num_panels_top, num_panels_bot):
        """Make the subplot grid + return (screen_cell, panel_cells) lookups.

        ``screen_cell`` is the (row, col) for the screen scene.
        ``panel_cells`` is a list of (row, col) tuples, one per panel,
        flattened in the same order as the flat panel list (top row
        first, then bottom row in the 2-row case).
        """
        N = self._n_grid_cols
        if layout_kind == '1row':
            specs = [[{'type': 'scene'} for _ in range(N)]]
            fig = make_subplots(
                rows=1, cols=N,
                specs=specs,
                subplot_titles=['Screen'] + list(flat_titles),
                horizontal_spacing=0.032,
            )
            screen_cell = (1, 1)
            panel_cells = [(1, c) for c in range(2, N + 1)]
            return fig, screen_cell, panel_cells

        # 2-row.
        specs = [[None] * N, [None] * N]
        specs[0][0] = {'type': 'scene', 'rowspan': 2}
        for c in range(1, num_panels_top + 1):
            specs[0][c] = {'type': 'scene'}
        for c in range(1, num_panels_bot + 1):
            specs[1][c] = {'type': 'scene'}

        # Subplot titles: make_subplots reads them in scan order (row-major).
        # We need to thread Nones into the empty cells so labels land on the
        # right scenes. For the 2-row layout, only scenes with specs receive
        # titles; the first scene is "Screen" and then top-row + bot-row.
        titles_in_order: List[str] = ['Screen']
        titles_in_order += list(flat_titles[:num_panels_top])
        titles_in_order += list(flat_titles[num_panels_top:])

        fig = make_subplots(
            rows=2, cols=N,
            specs=specs,
            subplot_titles=titles_in_order,
            horizontal_spacing=0.04,
            vertical_spacing=0.08,
        )

        screen_cell = (1, 1)
        panel_cells = (
            [(1, c) for c in range(2, num_panels_top + 2)]
            + [(2, c) for c in range(2, num_panels_bot + 2)]
        )
        return fig, screen_cell, panel_cells

    def _apply_layout(self, fig, layout_kind, *, n_total_cols, T, fps, dt_ms,
                      width, height):
        # Per-scene camera + hidden axes. Scene 1 = screen, scene 2..n = retinas.
        n_scenes = 1 + self._n_top + self._n_bot
        for i in range(1, n_scenes + 1):
            key = self._scene_key(i)
            if key in fig.layout:
                fig.layout[key].update(
                    xaxis=_HIDDEN_AXIS, yaxis=_HIDDEN_AXIS, zaxis=_HIDDEN_AXIS,
                    aspectmode='data',
                    camera=self._screen_camera if i == 1 else self._retina_camera,
                )

        steps = [
            dict(
                method='animate',
                label=str(int(round(t * dt_ms))),
                args=[[str(t)],
                      {'mode': 'immediate',
                       'frame': {'duration': 0, 'redraw': True},
                       'transition': {'duration': 0}}],
            )
            for t in range(T)
        ]

        if layout_kind == '1row':
            default_w = 350 * n_total_cols
            default_h = 500
            margin = dict(l=50, r=100, b=20, t=80)
        else:
            default_w = 400 * n_total_cols
            default_h = 850
            margin = dict(l=80, r=80, b=50, t=100)

        fig.update_layout(
            width=width or default_w,
            height=height or default_h,
            margin=margin,
            hovermode=False,
            updatemenus=[dict(
                type='buttons',
                x=0.05, y=0,
                xanchor='right', yanchor='top',
                showactive=False,
                buttons=[
                    dict(label='Play', method='animate',
                         args=[None, {'fromcurrent': True,
                                      'frame': {'duration': int(1000 / fps),
                                                'redraw': True},
                                      'transition': {'duration': 0}}]),
                    dict(label='Pause', method='animate',
                         args=[[None], {'mode': 'immediate',
                                        'frame': {'duration': 0, 'redraw': True},
                                        'transition': {'duration': 0}}]),
                ],
            )],
            sliders=[dict(
                active=0,
                currentvalue={'prefix': 'Time (ms): '},
                steps=steps,
                pad={'t': 30},
            )],
        )

    def _panel_intensity(self, values_TN, t: int) -> np.ndarray:
        per_lens = values_TN[t, self._kept_col_idx].astype(np.float32)
        return np.repeat(per_lens, self._verts_per_lens)

    @staticmethod
    def _resolve_crange(arr: np.ndarray, crange: CRange) -> Tuple[float, float]:
        cmin = crange[0] if crange[0] is not None else float(np.nanmin(arr))
        cmax = crange[1] if crange[1] is not None else float(np.nanmax(arr))
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
            cmin, cmax = 0.0, 1.0
        return cmin, cmax

    @staticmethod
    def _scene_key(scene_idx: int) -> str:
        """Plotly's scene keys: ``scene`` for #1, ``scene2`` for #2, ..."""
        return 'scene' if scene_idx == 1 else f'scene{scene_idx}'
