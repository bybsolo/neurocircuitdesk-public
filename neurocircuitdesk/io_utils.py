"""
io_utils.py
-----------
Standalone hex-grid video sampler and retinotopic visualizer for
NeuroCircuitDesk.  Provides a lightweight alternative to the full
``flyeyesimulator`` pipeline when only direct-from-video sampling and
2-D retinotopic visualization are needed.

Three classes are provided:

- **HexActiveSampleSpring**: Spring-mass-damper RF dynamics driven by
  luminance via a Hill-equation activation and gain compensation.
- **HexActiveSampleKinetic**: Two-phase kinetic RF dynamics (fast
  contraction / slow recovery) driven by a luminance sigmoid.
- **HexViz**: Visualization helper — luminance overlays, RF circle
  videos, and retinotopic scatter plots (same layout as FES RetinaViz).

Both samplers produce ``(T, N)`` luminance arrays compatible with
``Program.run_program()`` and store the full ``(T, N, 4)`` state
(x, y, R, lum) as an attribute.

Public API
----------
HexActiveSampleSpring     spring-damper active sampler
HexActiveSampleKinetic    two-phase kinetic active sampler
HexViz                    hex-grid visualization
"""

import json
import os
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
from tqdm import tqdm

# ── Default JSON paths (same ones Canvas uses) ──────────────────────────
_LIBS_DIR = os.path.join(os.path.dirname(__file__), "libs", "jsons")
_DEFAULT_COL_JSON = os.path.join(_LIBS_DIR, "hexcol_l1m3_new_578.json")
_DEFAULT_GRAPH_JSON = os.path.join(_LIBS_DIR, "hex_grid_graph.json")


# ── Hex grid helpers ─────────────────────────────────────────────────────

def _hex_ring_indices(radius: int) -> List[Tuple[int, int]]:
    """Generate axial (q, r) coordinates ring-by-ring, CW from north."""
    coords = [(0, 0)]
    ring_dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for k in range(1, radius + 1):
        q, r = 0, -k
        for d_q, d_r in ring_dirs:
            for _ in range(k):
                coords.append((q, r))
                q += d_q
                r += d_r
    return coords


def _axial_to_pixel_flat(q: int, r: int, spacing: float) -> Tuple[float, float]:
    """Axial hex -> pixel (x, y) using flat-topped orientation."""
    size = spacing / np.sqrt(3)
    x = 1.5 * size * q
    y = np.sqrt(3) * size * (r + 0.5 * q)
    return x, y


# ── Bio-column mask ──────────────────────────────────────────────────────

def _build_bio_mask(N: int, col_json_path: str) -> np.ndarray:
    """Return active_mask[N] where True iff hex_coords_id < 1000."""
    with open(col_json_path, "r") as f:
        data = json.load(f)
    hex_ids = np.array(data["hex_coords_id"])
    n_common = min(len(hex_ids), N)
    mask = np.full(N, False)
    mask[:n_common] = hex_ids[:n_common] < 1000
    return mask


def _load_hex_positions(
    n_cols: int,
    col_json_path: str = _DEFAULT_COL_JSON,
    graph_json_path: str = _DEFAULT_GRAPH_JSON,
) -> np.ndarray:
    """Return (n_cols, 2) retinotopic (x, y) positions.

    Applies the same ``(x, y) -> (-y, x)`` rotation that ``Canvas`` uses when
    placing microcircuits, so the retinotopic scatter aligns with both the
    Canvas's 3D layout and the flat-topped axial layout used by
    ``home_centers`` (where col_idx=1 sits above the centre).

    Non-bio columns get NaN so they are silently skipped by matplotlib.
    """
    with open(col_json_path) as f:
        col_data = json.load(f)
    hex_ids = col_data["hex_coords_id"]
    with open(graph_json_path) as f:
        graph_data = json.load(f)
    pos = {node["id"]: node["pos"] for node in graph_data["nodes"]}

    coords = np.full((n_cols, 2), np.nan)
    for i in range(min(n_cols, len(hex_ids))):
        bio_id = hex_ids[i]
        if bio_id < 1000 and bio_id in pos:
            x, y = pos[bio_id]
            coords[i] = (-y, x)
    return coords


# ── Shared base for both samplers ────────────────────────────────────────

class _HexSamplerBase:
    """Common grid + mask setup shared by Spring and Kinetic samplers."""

    def __init__(
        self,
        center_rc: Tuple[float, float],
        *,
        video_T: Optional[np.ndarray] = None,
        bio_cols_only: bool = True,
        col_json_path: Optional[str] = _DEFAULT_COL_JSON,
        radius: int = 20,
        window: int = 8,
        spacing: float = 8.0,
        dt: float = 10.0,
    ):
        self.video_T = video_T
        self.center_rc = (float(center_rc[0]), float(center_rc[1]))
        self.radius = int(radius)
        self.window = int(window)
        self.spacing = float(spacing)
        self.dt = float(dt)
        self.bio_cols_only = bio_cols_only
        self.col_json_path = col_json_path
        self.shrink_ratio = 0.1
        self.shift_ratio = 1

        # Hex grid
        self.axial = _hex_ring_indices(self.radius)
        self.N = 1 + 3 * self.radius * (self.radius + 1)

        # Bio mask
        self.active_mask = np.full(self.N, True)
        if self.bio_cols_only:
            if not self.col_json_path:
                raise ValueError(
                    "bio_cols_only=True requires a valid col_json_path.")
            self.active_mask = _build_bio_mask(self.N, self.col_json_path)

        # Home positions
        cy, cx = self.center_rc
        self.home_centers = np.empty((self.N, 2), dtype=float)
        for i, (q, r) in enumerate(self.axial):
            dx, dy = _axial_to_pixel_flat(q, r, self.spacing)
            self.home_centers[i] = (cx + dx, cy + dy)

        # Filled after sample()
        self.state: Optional[np.ndarray] = None   # (T, N, 4)
        self.values: Optional[np.ndarray] = None  # (T, N)

    def _get_luminance(self, x: float, y: float, R: float,
                       frame: np.ndarray) -> float:
        """Gaussian-weighted sum of luminance under the RF."""
        H, W = frame.shape
        sigma = R / 2.35
        patch_radius = 3.3 * sigma
        r_int = int(round(patch_radius))
        two_sigma_sq = 2.0 * sigma * sigma

        cx, cy = int(round(x)), int(round(y))
        y_min = max(0, cy - r_int)
        y_max = min(H, cy + r_int + 1)
        x_min = max(0, cx - r_int)
        x_max = min(W, cx + r_int + 1)

        Y, X = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (X - x) ** 2 + (Y - y) ** 2
        weights = np.exp(-dist_sq / two_sigma_sq)
        mask = dist_sq <= (patch_radius ** 2)
        if not np.any(mask):
            return 0.0

        return float(np.sum(frame[y_min:y_max, x_min:x_max][mask]
                            * weights[mask]))

    def _resolve_video(self, video_T: Optional[np.ndarray]) -> np.ndarray:
        """Return a (T, H, W) grayscale array from the provided or stored video."""
        if video_T is not None:
            self.video_T = video_T
        elif self.video_T is None:
            raise ValueError(
                "video_T must be provided at init or to sample().")
        v = self.video_T
        if v.ndim == 4:
            v = np.mean(v, axis=-1)
        return v

    # ── I/O ──────────────────────────────────────────────────────────────

    @staticmethod
    def save_h5(path: str, **arrays: np.ndarray):
        """Save named arrays to an HDF5 file.

        Example::

            sampler.save_h5("out.h5", inputs=lum_T, outputs=am_T)
        """
        with h5py.File(path, "w") as f:
            for name, arr in arrays.items():
                f.create_dataset(name, data=arr)


# ═════════════════════════════════════════════════════════════════════════
# HexActiveSampleSpring — spring-mass-damper dynamics
# ═════════════════════════════════════════════════════════════════════════

class HexActiveSampleSpring(_HexSamplerBase):
    """Active hex-grid sampler with spring-mass-damper RF dynamics.

    Each receptive field is driven by luminance through a Hill-equation
    activation with gain compensation, asymmetric exponential damping,
    and an activity-dependent stiffness.
    """

    def __init__(self, center_rc, **kwargs):
        super().__init__(center_rc, **kwargs)
        # Spring parameters
        self.k0 = 0.00067
        self.k_coef = 0.0032
        self.F_max = 0.031 * self.shift_ratio * self.window / 8.0
        self.D_coef = 0.09
        self.D_exp = 2.0
        self.half_pk = 120000.0
        self.gain_slope = 0.40

    def _update_receptive_field(self, phys_x, phys_v, x0, y0, lum):
        """Spring-mass-damper step.  Returns (pixel_x, pixel_y, pixel_R,
        new_phys_x, new_phys_v)."""
        compensated_act = lum * (1.0 + self.gain_slope * phys_x)
        u_term = compensated_act / (self.half_pk + compensated_act + 1e-9)

        k = self.k0 + (self.k_coef * u_term)
        F_drive = self.F_max * u_term
        F_damp = self.D_coef * (np.power(2, -self.D_exp * phys_v) - 1)

        accel = F_drive + F_damp - (k * phys_x)
        new_phys_v = phys_v + (accel * self.dt)
        new_phys_x = max(0.0, phys_x + (new_phys_v * self.dt))

        pixel_x = x0 + new_phys_x
        pixel_y = y0
        pixel_R = max(0.1, self.window - (self.shrink_ratio * new_phys_x))

        return pixel_x, pixel_y, pixel_R, new_phys_x, new_phys_v

    def sample(self, video_T: Optional[np.ndarray] = None,
               shift: bool = True, shrink: bool = True) -> np.ndarray:
        """Run the spring-damper sampling loop.

        Returns ``(T, N)`` luminance array.  Full ``(T, N, 4)`` state
        ``[x, y, R, lum]`` is stored in ``self.state``.
        """
        video = self._resolve_video(video_T)
        T = video.shape[0]

        state_T = np.zeros((T, self.N, 4), dtype=np.float32)
        phys_x = np.zeros(self.N)
        phys_v = np.zeros(self.N)
        delay = max(1, int(round(10.0 / self.dt)))

        for t in tqdm(range(T), desc="Spring sampling"):
            frame = video[t]
            for i in range(self.N):
                if not self.active_mask[i]:
                    state_T[t, i] = [self.home_centers[i, 0],
                                     self.home_centers[i, 1], 0.0, 0.0]
                    continue

                x0, y0 = self.home_centers[i]
                delayed_lum = (state_T[t - delay, i, 3]
                               if t >= delay else 0.0)

                pix_x, pix_y, pix_R, new_px, new_pv = \
                    self._update_receptive_field(
                        phys_x[i], phys_v[i], x0, y0, delayed_lum)

                final_x = pix_x if shift else x0
                final_y = pix_y if shift else y0
                final_R = pix_R if shrink else float(self.window)

                phys_x[i] = new_px
                phys_v[i] = new_pv

                lum = self._get_luminance(final_x, final_y, final_R, frame)
                state_T[t, i] = [final_x, final_y, final_R, lum]

        self.state = state_T
        self.values = state_T[:, :, 3]
        return self.values


# ═════════════════════════════════════════════════════════════════════════
# HexActiveSampleKinetic — two-phase kinetic dynamics
# ═════════════════════════════════════════════════════════════════════════

class HexActiveSampleKinetic(_HexSamplerBase):
    """Active hex-grid sampler with two-phase kinetic RF dynamics.

    Fast contraction phase (50 ms) driven by a luminance sigmoid,
    slow recovery phase (375 ms) returning to rest.
    """

    def _update_receptive_field(self, x, y, x0, y0, current_R, lum,
                                instant: bool):
        """Two-phase kinetic step.  Returns (new_x, new_y, new_R)."""
        fast_phase = 50.0   # ms
        slow_phase = 375.0  # ms
        r = float(self.window)

        safe_lum = np.clip(lum, 1e-3, np.inf)
        log_center = 3.0
        r_lum = r / (1.0 + np.exp(-2.0 * (np.log10(safe_lum) - log_center)))

        x_target = x0 + self.shift_ratio * r_lum
        R_target = r - (self.shrink_ratio * r_lum)

        fast_rate = self.shift_ratio * r_lum / (fast_phase / self.dt)
        R_dec = (self.shrink_ratio * r_lum) / (fast_phase / self.dt)
        slow_rate = self.shift_ratio * r / (slow_phase / self.dt)
        R_inc = (self.shrink_ratio * r) / (slow_phase / self.dt)

        if current_R > R_target:
            new_x = min(x + fast_rate, x_target)
            new_R = max(current_R - R_dec, R_target)
        else:
            new_x = max(x - slow_rate, x0)
            new_R = min(current_R + R_inc, r)

        new_y = y0

        if instant:
            return x_target, new_y, R_target
        return new_x, new_y, new_R

    def sample(self, video_T: Optional[np.ndarray] = None,
               shift: bool = True, shrink: bool = True,
               instant: bool = False) -> np.ndarray:
        """Run the kinetic sampling loop.

        Returns ``(T, N)`` luminance array.  Full ``(T, N, 4)`` state
        ``[x, y, R, lum]`` is stored in ``self.state``.
        """
        video = self._resolve_video(video_T)
        T = video.shape[0]

        state_T = np.zeros((T, self.N, 4), dtype=np.float32)
        r = float(self.window)
        current_centers = self.home_centers.astype(float).copy()
        current_Rs = np.full(self.N, r)
        delay = max(1, int(round(10.0 / self.dt)))

        for t in tqdm(range(T), desc="Kinetic sampling"):
            frame = video[t]
            for i in range(self.N):
                if not self.active_mask[i]:
                    current_centers[i] = self.home_centers[i]
                    current_Rs[i] = 0.0
                    state_T[t, i] = [self.home_centers[i, 0],
                                     self.home_centers[i, 1], 0.0, 0.0]
                    continue

                x0, y0 = self.home_centers[i]
                x, y = current_centers[i]

                delayed_lum = (state_T[t - delay, i, 3]
                               if t >= delay else 0.0)

                _x, _y, _R = self._update_receptive_field(
                    x, y, x0, y0, current_Rs[i], delayed_lum, instant)

                new_x = _x if shift else x0
                new_y = _y if shift else y0
                new_R = _R if shrink else r

                lum = self._get_luminance(new_x, new_y, new_R, frame)
                current_centers[i] = [new_x, new_y]
                current_Rs[i] = new_R
                state_T[t, i] = [new_x, new_y, new_R, lum]

        self.state = state_T
        self.values = state_T[:, :, 3]
        return self.values


# ═════════════════════════════════════════════════════════════════════════
# HexViz — visualization for sampler outputs
# ═════════════════════════════════════════════════════════════════════════

class HexViz:
    """Visualization helper for hex-grid sampler output (state_T).

    Handles luminance overlay plots, RF circle plots, retinotopic scatter
    plots, and video export.  Respects the biological-column mask.
    """

    def __init__(
        self,
        center_rc: Tuple[float, float],
        *,
        video_T: Optional[np.ndarray] = None,
        bio_cols_only: bool = True,
        col_json_path: Optional[str] = _DEFAULT_COL_JSON,
        radius: int = 20,
        window: int = 8,
        spacing: float = 8.0,
    ):
        self.video_T = video_T
        self.center_rc = (float(center_rc[0]), float(center_rc[1]))
        self.radius = int(radius)
        self.window = int(window)
        self.spacing = float(spacing)
        self.bio_cols_only = bio_cols_only
        self.col_json_path = col_json_path

        # Hex grid
        self.axial = _hex_ring_indices(self.radius)
        self.N = 1 + 3 * self.radius * (self.radius + 1)

        # Bio mask
        self.active_mask = np.full(self.N, True)
        if self.bio_cols_only:
            if not self.col_json_path:
                raise ValueError(
                    "bio_cols_only=True requires a valid col_json_path.")
            self.active_mask = _build_bio_mask(self.N, self.col_json_path)

        # Home positions
        cy, cx = self.center_rc
        self.home_centers = np.empty((self.N, 2), dtype=float)
        for i, (q, r) in enumerate(self.axial):
            dx, dy = _axial_to_pixel_flat(q, r, self.spacing)
            self.home_centers[i] = (cx + dx, cy + dy)

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_video(self, video_T: Optional[np.ndarray]) -> np.ndarray:
        if video_T is not None:
            return video_T
        if self.video_T is not None:
            return self.video_T
        raise ValueError(
            "video_T must be provided at init or to the plotting method.")

    # ── Per-cell overlay ─────────────────────────────────────────────────

    def plot_frame(
        self,
        data: np.ndarray,
        frame_idx: int,
        *,
        video_T: Optional[np.ndarray] = None,
        title: str = "Hex overlay",
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        s: float = 12.0,
        ax=None,
        show: bool = True,
    ):
        """Scatter per-cell ``data`` at hex home positions over the video frame.

        ``data`` is ``(T, N)`` — pass ``sampler.values`` (= ``state[..., 3]``)
        for the input luminance, or any per-column time-series from
        ``Program.probe_result(...)`` for circuit outputs.
        """
        video_T = self._get_video(video_T)
        t = int(frame_idx)
        bg = video_T[t]
        bg_vmin = float(np.nanmin(video_T))
        bg_vmax = float(np.nanmax(video_T))

        centers_x = self.home_centers[self.active_mask, 0]
        centers_y = self.home_centers[self.active_mask, 1]
        vals = data[t, self.active_mask]

        if vmin is None:
            vmin = float(np.nanmin(data[:, self.active_mask]))
        if vmax is None:
            vmax = float(np.nanmax(data[:, self.active_mask]))

        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            ax.clear()

        ax.imshow(bg, cmap="gray", interpolation="nearest",
                  vmin=bg_vmin, vmax=bg_vmax)
        if len(centers_x) > 0:
            ax.scatter(centers_x, centers_y, c=vals,
                       cmap=cmap, s=s, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

        if show and created:
            plt.show()
        return ax

    def save_frame_video(
        self,
        data: np.ndarray,
        fname: str,
        *,
        video_T: Optional[np.ndarray] = None,
        fps: int = 12,
        title: str = "Hex overlay",
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        s: float = 12.0,
        dpi: int = 150,
        bitrate: int = 1800,
    ):
        """Save a video of the per-cell overlay (one ``plot_frame`` per step)."""
        video_T = self._get_video(video_T)
        T = video_T.shape[0]

        if vmin is None:
            vmin = float(np.nanmin(data[:, self.active_mask]))
        if vmax is None:
            vmax = float(np.nanmax(data[:, self.active_mask]))

        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        writer = FFMpegWriter(fps=fps, bitrate=bitrate)
        with writer.saving(fig, fname, dpi=dpi):
            for t in tqdm(range(T), desc="Saving frame video"):
                self.plot_frame(
                    data, t, video_T=video_T, title=title, cmap=cmap,
                    vmin=vmin, vmax=vmax, s=s, ax=ax, show=False)
                writer.grab_frame()
        plt.close(fig)

    # ── Inline-embedded video (notebook helpers) ─────────────────────────

    def show_video_inline(
        self,
        data: np.ndarray,
        fname: str,
        *,
        title: str = "Hex overlay",
        fps: int = 20,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        s: float = 12.0,
        width: int = 420,
        video_T: Optional[np.ndarray] = None,
    ):
        """Save ``data`` (T, N) as an MP4 to ``fname`` and return an inline
        :class:`IPython.display.Video` for cell-level embedding.

        Wraps :meth:`save_frame_video` (which writes the MP4) with
        ``IPython.display.Video(..., embed=True)`` so the MP4 bytes are
        base64-encoded into the notebook output — the cell stays
        self-contained when the ``.ipynb`` is saved or shared.

        Designed to be the cell's return value (the notebook displays the
        returned :class:`Video` automatically) or passed to
        ``display(...)``.

        Parameters
        ----------
        data : array of shape ``(T, N)``
            Per-cell time series.
        fname : str or Path
            Where to write the MP4. Required — embedding needs a file on
            disk.
        title, fps, vmin, vmax, cmap, s : as in :meth:`save_frame_video`.
        width : int
            Inline display width in pixels.
        video_T : array, optional
            Background video. Forwarded to :meth:`save_frame_video`.
        """
        from IPython.display import Video  # lazy: only needed in notebooks
        self.save_frame_video(data, str(fname),
                              video_T=video_T,
                              fps=fps, title=title, cmap=cmap,
                              vmin=vmin, vmax=vmax, s=s)
        return Video(str(fname), embed=True, width=width)

    def show_videos_row_inline(
        self,
        datas: List[np.ndarray],
        fnames: List[str],
        titles: List[str],
        *,
        fps: int = 20,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        s: float = 12.0,
        width: int = 300,
        video_T: Optional[np.ndarray] = None,
    ) -> None:
        """Save each ``(data, fname, title)`` triple as an MP4 and embed
        them side-by-side via ``ipywidgets.HBox``. Displays directly; no
        return value.

        Uses one shared ``vmin``/``vmax`` across all panels so the colour
        scales are comparable.
        """
        from IPython.display import Video, display      # lazy
        from ipywidgets import HBox, Output             # lazy
        if not (len(datas) == len(fnames) == len(titles)):
            raise ValueError(
                f"datas/fnames/titles length mismatch: "
                f"{len(datas)}/{len(fnames)}/{len(titles)}")

        items = []
        for data, fname, title in zip(datas, fnames, titles):
            self.save_frame_video(data, str(fname),
                                  video_T=video_T,
                                  fps=fps, title=title, cmap=cmap,
                                  vmin=vmin, vmax=vmax, s=s)
            out = Output()
            with out:
                display(Video(str(fname), embed=True, width=width))
            items.append(out)
        display(HBox(items))

    # ── RF circles ───────────────────────────────────────────────────────

    def plot_rf_frame(
        self,
        data: np.ndarray,
        frame_idx: int,
        *,
        video_T: Optional[np.ndarray] = None,
        title: str = "Active Receptive Fields",
        edge_color: str = "red",
        edge_width: float = 0.5,
        fill: bool = False,
        alpha: float = 1.0,
        ax=None,
        show: bool = True,
    ):
        """Draw RF circles at current (x, y) with current R."""
        video_T = self._get_video(video_T)
        t = int(frame_idx)
        bg = video_T[t]
        bg_vmin = float(np.nanmin(video_T))
        bg_vmax = float(np.nanmax(video_T))

        active_state = data[t, self.active_mask, :]

        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(6, 6))

        if not hasattr(ax, "_rf_bg_im"):
            ax._rf_bg_im = ax.imshow(bg, cmap="gray",
                                     interpolation="nearest",
                                     vmin=bg_vmin, vmax=bg_vmax)
            ax.set_title(title)
            ax.axis("off")
            circles = []
            for x, y, R, _ in active_state:
                c = Circle((x, y), radius=R, fill=fill,
                           edgecolor=edge_color, linewidth=edge_width,
                           alpha=alpha)
                ax.add_patch(c)
                circles.append(c)
            ax._rf_circles = circles
        else:
            ax._rf_bg_im.set_data(bg)
            ax._rf_bg_im.set_clim(bg_vmin, bg_vmax)
            ax.set_title(title)
            circles = ax._rf_circles
            n_existing = len(circles)
            n_needed = len(active_state)
            if n_needed > n_existing:
                for _ in range(n_needed - n_existing):
                    c = Circle((0, 0), radius=0.0, fill=fill,
                               edgecolor=edge_color, linewidth=edge_width,
                               alpha=alpha)
                    ax.add_patch(c)
                    circles.append(c)
                ax._rf_circles = circles
            elif n_needed < n_existing:
                for extra in circles[n_needed:]:
                    extra.set_visible(False)
            for circ, (x, y, R, _) in zip(circles, active_state):
                circ.set_visible(True)
                circ.center = (x, y)
                circ.radius = R
                circ.set_edgecolor(edge_color)
                circ.set_linewidth(edge_width)
                circ.set_alpha(alpha)

        if show and created:
            plt.show()
        return ax

    def save_rf_video(
        self,
        data: np.ndarray,
        fname: str,
        *,
        video_T: Optional[np.ndarray] = None,
        fps: int = 12,
        title: str = "Active Receptive Fields",
        edge_color: str = "red",
        edge_width: float = 0.5,
        fill: bool = False,
        alpha: float = 1.0,
        dpi: int = 150,
        bitrate: int = 1800,
    ):
        """Save an MP4 showing the moving/resizing RF circles."""
        video_T = self._get_video(video_T)
        T = video_T.shape[0]

        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        writer = FFMpegWriter(fps=fps, bitrate=bitrate)
        self.plot_rf_frame(data, 0, video_T=video_T, title=title,
                           edge_color=edge_color, edge_width=edge_width,
                           fill=fill, alpha=alpha, ax=ax, show=False)
        with writer.saving(fig, fname, dpi=dpi):
            for t in tqdm(range(T), desc="Saving RF video"):
                self.plot_rf_frame(data, t, video_T=video_T, title=title,
                                   edge_color=edge_color,
                                   edge_width=edge_width, fill=fill,
                                   alpha=alpha, ax=ax, show=False)
                writer.grab_frame()
        plt.close(fig)

    # ── I/O ──────────────────────────────────────────────────────────────

    @staticmethod
    def save_h5(path: str, **arrays: np.ndarray):
        """Save named arrays to an HDF5 file.

        Example::

            HexViz.save_h5("out.h5", inputs=lum_T, outputs=am_T)
        """
        with h5py.File(path, "w") as f:
            for name, arr in arrays.items():
                f.create_dataset(name, data=arr)
