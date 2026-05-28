import numpy as np
import plotly.graph_objects as go
from matplotlib import cm

from .backend import to_numpy


class ScreenViz:
    """
    Visualizes the spherical screen intensities in 3D hemisphere space.
    """
    def __init__(self, screen_obj):
        """
        Args:
            screen_obj: An instance of the Screen class.
        """
        self.screen = screen_obj
        self.default_camera = dict(
            eye=dict(x=1.5, y=1.5, z=0.8),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )

    def _downsample(self, arr, factor):
        """Downsample 2D or 3D array by taking every factor-th element along spatial dims."""
        if factor <= 1:
            return arr
        if arr.ndim == 2:
            return arr[::factor, ::factor]
        return arr[:, ::factor, ::factor]

    def _get_xyz_and_intensity(self, frame_idx, downsample=1):
        """Returns (X, Y, Z, intensities) as numpy arrays for a given frame."""
        el_grid, az_grid = self.screen.get_coordinates()
        intensities = self.screen.get_intensity_frame(frame_idx)

        el_grid = to_numpy(el_grid)
        az_grid = to_numpy(az_grid)
        intensities = to_numpy(intensities)

        if downsample > 1:
            el_grid = self._downsample(el_grid, downsample)
            az_grid = self._downsample(az_grid, downsample)
            intensities = self._downsample(intensities, downsample)

        el_rad = np.deg2rad(el_grid)
        az_rad = np.deg2rad(az_grid)
        r = float(self.screen.radius)

        X = r * np.cos(el_rad) * np.cos(az_rad)
        Y = r * np.cos(el_rad) * np.sin(az_rad)
        Z = r * np.sin(el_rad)
        return X, Y, Z, intensities

    def plot(self, frame_idx=0, cmap='Viridis', camera=None, downsample=1):
        """
        Plots the 3D hemisphere for a single frame.

        Args:
            frame_idx: Index of the frame to display.
            cmap: Plotly colorscale name (e.g. 'Viridis', 'Gray', 'Turbo').
            camera: Optional dict with eye, up, center; uses default if None.
            downsample: Spatial downsampling factor (e.g. 4 = 4x fewer points).
        """
        X, Y, Z, intensities = self._get_xyz_and_intensity(frame_idx, downsample)

        surf = go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=intensities,
            colorscale=cmap,
            cmin=float(np.nanmin(intensities)),
            cmax=float(np.nanmax(intensities)),
            showscale=True,
        )

        fig = go.Figure(data=[surf])
        no_axis = dict(
            showbackground=True, showgrid=False, showline=False,
            showticklabels=False, title='', visible=False,
        )

        cam = camera if camera is not None else self.default_camera
        fig.update_layout(
            scene=dict(
                xaxis=no_axis, yaxis=no_axis, zaxis=no_axis,
                aspectmode='data',
                camera=cam,
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            title=dict(text=f"Screen (Frame {frame_idx})"),
            width=800, height=600,
        )
        fig.show()

    def save_video(
        self,
        fps=30,
        html_path=None,
        cmin=None,
        cmax=None,
        colorscale="Viridis",
        camera=None,
        downsample=4,
        frame_stride=1,
        include_plotlyjs="cdn",
    ):
        """
        Saves an interactive HTML video of the screen intensities over time.

        Args:
            fps: Frames per second for playback.
            html_path: If provided, writes HTML to this path.
            cmin, cmax: Color range; if None, computed from all frames.
            colorscale: Plotly colorscale name.
            camera: Optional dict with eye, up, center; uses default if None.
            downsample: Spatial factor (e.g. 4 → 75x75 from 300x300); reduces size & speeds render.
            frame_stride: Use every Nth frame (e.g. 5 → 5x fewer frames).
            include_plotlyjs: "cdn" (smaller file, needs internet) or True (portable, ~3MB larger).
        """
        intensities = to_numpy(self.screen.intensities)
        if downsample > 1:
            intensities = self._downsample(intensities, downsample)
        if frame_stride > 1:
            intensities = intensities[::frame_stride]

        T, P, M = intensities.shape

        el_grid, az_grid = self.screen.get_coordinates()
        el_grid = to_numpy(el_grid)
        az_grid = to_numpy(az_grid)
        if downsample > 1:
            el_grid = self._downsample(el_grid, downsample)
            az_grid = self._downsample(az_grid, downsample)

        el_rad = np.deg2rad(el_grid)
        az_rad = np.deg2rad(az_grid)
        r = float(self.screen.radius)

        X = r * np.cos(el_rad) * np.cos(az_rad)
        Y = r * np.cos(el_rad) * np.sin(az_rad)
        Z = r * np.sin(el_rad)

        if cmin is None:
            cmin = float(np.nanmin(intensities))
        if cmax is None:
            cmax = float(np.nanmax(intensities))
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
            cmin = 0.0
            cmax = 1.0

        surf = go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=intensities[0],
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=True,
        )

        frames = [
            go.Frame(
                name=str(t),
                data=[go.Surface(surfacecolor=intensities[t])],
                traces=[0],
            )
            for t in range(T)
        ]

        steps = [
            dict(
                method="animate",
                args=[[str(t)], {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                }],
                label=str(t),
            )
            for t in range(T)
        ]

        fig = go.Figure(data=[surf], frames=frames)
        no_axis = dict(
            showbackground=True, showgrid=False, showline=False,
            showticklabels=False, title='', visible=False,
        )
        cam = camera if camera is not None else self.default_camera

        fig.update_layout(
            scene=dict(
                xaxis=no_axis, yaxis=no_axis, zaxis=no_axis,
                aspectmode='data',
                camera=cam,
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            width=800,
            height=600,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                    "frame": {"duration": int(1000 / fps), "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                ),
            ],
            sliders=[dict(active=0, currentvalue={"prefix": "t = "}, steps=steps)],
        )

        if html_path is not None:
            fig.write_html(
                html_path,
                include_plotlyjs=include_plotlyjs,
                full_html=True,
                auto_play=False,
            )

        return fig

    def save_mp4(
        self,
        path,
        fps=30,
        cmin=None,
        cmax=None,
        cmap="viridis",
        frame_stride=1,
        width=512,
        height=512,
    ):
        """
        Exports a 2D (el, az) projection as MP4 video. Much smaller and faster than HTML.
        Requires ffmpeg (matplotlib animation backend).

        Args:
            path: Output file path (e.g. 'screen.mp4').
            fps: Frames per second.
            cmin, cmax: Color range; if None, computed from all frames.
            cmap: Matplotlib colormap name (e.g. 'viridis', 'gray').
            frame_stride: Use every Nth frame.
            width, height: Output resolution.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation

        intensities = to_numpy(self.screen.intensities)
        if frame_stride > 1:
            intensities = intensities[::frame_stride]
        T = intensities.shape[0]

        if cmin is None:
            cmin = float(np.nanmin(intensities))
        if cmax is None:
            cmax = float(np.nanmax(intensities))
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
            cmin = 0.0
            cmax = 1.0

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_axis_off()
        im = ax.imshow(intensities[0], cmap=cmap, aspect="auto", vmin=cmin, vmax=cmax)

        def update(t):
            im.set_array(intensities[t])
            return [im]

        anim = FuncAnimation(fig, update, frames=T, blit=True, interval=0)
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(path, writer=writer)
        plt.close(fig)
