"""
SimulatorViz: Plotly-based visualization of retinas + screen geometry.
Retinas shown as uniform-color hex prisms; screen as transparent hemisphere.
"""
import numpy as np
import plotly.graph_objects as go
from .retina import Retina
from .retina_rotator import RetinaRotator


def _rotation_matrix_from_vectors(vec1, vec2):
    """Build rotation matrix to align vec1 with vec2."""
    a = np.asarray(vec1, dtype=np.float64) / (np.linalg.norm(vec1) + 1e-12)
    b = np.asarray(vec2, dtype=np.float64) / (np.linalg.norm(vec2) + 1e-12)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))


def _create_hex_lens_mesh(center, normal, lens_radius, lens_length):
    """Create hex prism mesh at center with normal direction."""
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    hex_x = lens_radius * np.cos(angles)
    hex_y = lens_radius * np.sin(angles)

    target_normal = np.array(normal, dtype=np.float64)
    R = _rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), target_normal)

    eps = 1e-12
    if abs(lens_length) <= eps:
        z0 = np.zeros_like(hex_x)
        verts_local = np.column_stack([hex_x, hex_y, z0])
        verts_world = verts_local @ R.T + np.array(center, dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]], dtype=int)
        return verts_world, faces

    z_base = np.zeros_like(hex_x)
    z_tip = np.full_like(hex_x, lens_length)
    verts_local = np.vstack([
        np.column_stack([hex_x, hex_y, z_base]),
        np.column_stack([hex_x, hex_y, z_tip]),
    ])
    verts_world = (verts_local @ R.T) + np.array(center, dtype=np.float64)

    faces = []
    for i in range(6):
        next_i = (i + 1) % 6
        faces.append([i, next_i, next_i + 6])
        faces.append([i, next_i + 6, i + 6])
    for i in range(1, 5):
        faces.append([0, i + 1, i])
        faces.append([6, 6 + i, 6 + i + 1])
    return verts_world, np.array(faces, dtype=int)


def _axis_polar_to_normal(az, el):
    """Convert axis_polar (az, el) in radians to unit direction vector (hex facing direction)."""
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ], dtype=np.float64)


def _create_hemisphere_mesh(radius, parallels=24, meridians=48):
    """
    Create hemisphere mesh (el [-90, 90], az [0, 180]).
    Returns (X, Y, Z) for Surface, and vertices for optional wireframe.
    """
    el_rad = np.linspace(-np.pi / 2, np.pi / 2, parallels)
    az_rad = np.linspace(0, np.pi, meridians)
    el_grid, az_grid = np.meshgrid(el_rad, az_rad, indexing="ij")

    X = radius * np.cos(el_grid) * np.cos(az_grid)
    Y = radius * np.cos(el_grid) * np.sin(az_grid)
    Z = radius * np.sin(el_grid)
    return X, Y, Z


class SimulatorViz:
    """
    Visualizes retinas (as uniform-color hex prisms) and screen (transparent hemisphere).
    Optional rays show the viewing direction for selected RF types.
    """

    def __init__(
        self,
        lens_radius=0.04,
        lens_length=0.01,
        rays=False,
        ray_mode="R1",
        ray_length=0.3,
        ray_width=3,
    ):
        """
        Args:
            lens_radius: Radius of each hex lens (same as RetinaViz).
            lens_length: Extrusion length of hex prism (0 for flat hex).
            rays: If True, draw direction rays from each selected RF.
            ray_mode: Which RF set to use for rays. Options: 'R1'...'R6', 'R7'.
                      R1-R6 use axis_polar of vrfs[offset::6]; R7 uses coord_polar of vrfs[::6].
            ray_length: Length of each ray segment (same units as retina coords).
            ray_width: Line width for rays (Plotly units, default 3).
        """
        self.lens_radius = lens_radius
        self.lens_length = lens_length
        self.retinas = []  # list of (retina, color, name, ray_color)
        self.screen_radius = None  # set when add_screen called
        self.show_rays = rays
        self.ray_mode = ray_mode.upper()
        if self.ray_mode not in {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}:
            raise ValueError("ray_mode must be one of R1..R7")
        self.ray_length = float(ray_length)
        self.ray_width = float(ray_width)
        self.custom_ray_specs = []  # list of (retina_name, col_index, rf_type, color)

        self.default_camera = dict(
            eye=dict(x=2.0, y=2.0, z=1.2),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )

    def add_retina(self, retina: Retina, color="steelblue", name=None, ray_color=None):
        """
        Add a retina to the visualization. Renders as uniform-color hex prisms (no intensity).

        Args:
            retina: Retina object.
            color: Plotly color (e.g. 'steelblue', 'crimson', 'rgb(100,150,200)').
            name: Optional legend name.
            ray_color: Color for direction rays (defaults to black if not provided).
        """
        self.retinas.append(
            (retina, color, name or f"retina_{len(self.retinas)}", ray_color)
        )

    def add_rays(self, specs):
        """
        Add specific rays when show_rays=False. Each spec: (retina_name, col_index, rf_type, color).
        R1-R6: use axis_polar of that RF; R7: use coord_polar of column's first RF.

        Example:
            viz.add_rays([
                ('left', 0, 'R1', 'red'),
                ('right', 1, 'R1', 'blue'),
            ])
        """
        for spec in specs:
            if len(spec) != 4:
                raise ValueError("Each spec must be (retina_name, col_index, rf_type, color)")
            retina_name, col_index, rf_type, color = spec
            rf_type = rf_type.upper()
            if rf_type not in {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}:
                raise ValueError("rf_type must be one of R1..R7")
            self.custom_ray_specs.append((retina_name, int(col_index), rf_type, color))

    def add_screen(self, screen=None, radius=None):
        """
        Add screen as a transparent hemisphere.

        Args:
            screen: Screen object (uses screen.radius). If None, radius must be provided.
            radius: Hemisphere radius when screen is None.
        """
        if screen is not None:
            self.screen_radius = float(screen.radius)
        elif radius is not None:
            self.screen_radius = float(radius)
        else:
            raise ValueError("Either screen or radius must be provided.")

    def _build_retina_trace(self, retina, color, name):
        """Build Mesh3d trace for one retina: 1/6 of receptors (one per column, like RetinaViz)."""
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        vertex_offset = 0

        # Plot one per column (vrfs[::6] = R1 of each column)
        for vrf in retina.vrfs[::6]:
            if vrf.enabled:
                center = vrf.coord_xyz
                # Hex faces in the direction the RF looks (axis_polar)
                normal = _axis_polar_to_normal(vrf.axis_polar["az"], vrf.axis_polar["el"])
                verts, faces = _create_hex_lens_mesh(
                    center, normal, self.lens_radius, self.lens_length
                )
                nv = verts.shape[0]
                all_x.extend(verts[:, 0])
                all_y.extend(verts[:, 1])
                all_z.extend(verts[:, 2])
                for f in faces:
                    all_i.append(f[0] + vertex_offset)
                    all_j.append(f[1] + vertex_offset)
                    all_k.append(f[2] + vertex_offset)
                vertex_offset += nv

        if len(all_x) == 0:
            return None

        return go.Mesh3d(
            x=all_x, y=all_y, z=all_z,
            i=all_i, j=all_j, k=all_k,
            color=color,
            opacity=1.0,
            flatshading=True,
            name=name,
            hoverinfo="skip",
            hovertemplate=None,
        )

    def _build_custom_ray_traces(self):
        """Build ray traces from add_rays specs (used when show_rays=False)."""
        name_to_retina = {name: (retina, ray_color) for retina, _, name, ray_color in self.retinas}
        traces = []
        for retina_name, col_index, rf_type, color in self.custom_ray_specs:
            if retina_name not in name_to_retina:
                continue
            retina, _ = name_to_retina[retina_name]
            if rf_type == "R7":
                vrf_idx = col_index * 6
                direction_source = "coord"
            else:
                offset = int(rf_type[1]) - 1
                vrf_idx = col_index * 6 + offset
                direction_source = "axis"

            if vrf_idx >= len(retina.vrfs) or not retina.vrfs[vrf_idx].enabled:
                continue

            vrf = retina.vrfs[vrf_idx]
            origin = vrf.coord_xyz
            if direction_source == "axis":
                az, el = vrf.axis_polar["az"], vrf.axis_polar["el"]
            else:
                az, el = vrf.coord_polar["az"], vrf.coord_polar["el"]
            direction = _axis_polar_to_normal(az, el)
            end = origin + direction * self.ray_length

            traces.append(go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                line=dict(color=color, width=self.ray_width),
                showlegend=False,
                hoverinfo="skip",
            ))
        return traces

    def _build_ray_trace(self, retina, color, name, ray_color=None):
        """Optional rays showing viewing direction (all rays, used when show_rays=True)."""
        if not self.show_rays:
            return None

        mode = self.ray_mode
        if mode == "R7":
            vrf_iter = retina.vrfs[::6]
            direction_source = "coord"
        else:
            offset = int(mode[1]) - 1  # R1 -> 0, ..., R6 -> 5
            vrf_iter = retina.vrfs[offset::6]
            direction_source = "axis"

        xs, ys, zs = [], [], []
        for vrf in vrf_iter:
            if not vrf.enabled:
                continue

            origin = vrf.coord_xyz
            if direction_source == "axis":
                az = vrf.axis_polar["az"]
                el = vrf.axis_polar["el"]
            else:
                az = vrf.coord_polar["az"]
                el = vrf.coord_polar["el"]

            direction = _axis_polar_to_normal(az, el)
            end = origin + direction * self.ray_length

            xs.extend([origin[0], end[0], None])
            ys.extend([origin[1], end[1], None])
            zs.extend([origin[2], end[2], None])

        if not xs:
            return None

        ray_color = ray_color or "black"
        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=ray_color, width=self.ray_width),
            name=f"{name}_rays",
            showlegend=False,
            hoverinfo="skip",
        )

    def _build_screen_trace(self):
        """Build transparent hemisphere Surface trace."""
        X, Y, Z = _create_hemisphere_mesh(self.screen_radius)
        return go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=np.ones_like(Z),
            colorscale=[[0, "rgba(220,220,220,0.25)"], [1, "rgba(220,220,220,0.25)"]],
            showscale=False,
            name="screen",
            hoverinfo="skip",
        )

    def plot(self, camera=None):
        """Render and show the combined scene."""
        data = []

        for retina, color, name, ray_color in self.retinas:
            trace = self._build_retina_trace(retina, color, name)
            if trace is not None:
                data.append(trace)
            if self.show_rays:
                ray_trace = self._build_ray_trace(retina, color, name, ray_color)
                if ray_trace is not None:
                    data.append(ray_trace)

        if self.custom_ray_specs and not self.show_rays:
            data.extend(self._build_custom_ray_traces())

        if self.screen_radius is not None:
            data.append(self._build_screen_trace())

        fig = go.Figure(data=data)
        no_axis = dict(
            showbackground=True, showgrid=False, showline=False,
            showticklabels=False, title="", visible=False,
        )
        cam = camera if camera is not None else self.default_camera
        fig.update_layout(
            scene=dict(
                xaxis=no_axis, yaxis=no_axis, zaxis=no_axis,
                aspectmode="data",
                camera=cam,
            ),
            margin=dict(l=0, r=0, b=0, t=20),
            paper_bgcolor="white",
            width=900,
            height=700,
        )
        fig.show()

    @staticmethod
    def create_dual_eye_setup(
        base_retina: Retina,
        screen=None,
        radius=10,
        left_offset=(-0.5, 0, 0),
        left_euler_deg=(0, 0, 0),
        right_offset=(0.5, 0, 0),
        right_euler_deg=(0, 0, 0),
        left_color="steelblue",
        right_color="crimson",
        left_ray_color=None,
        right_ray_color=None,
        lens_radius=0.04,
        lens_length=0.01,
    ):
        """
        Create a SimulatorViz with two retinas (left and right) from a base retina,
        plus a transparent screen hemisphere.

        Args:
            base_retina: Source retina (centered at origin).
            screen: Screen object for radius; if None, uses radius.
            radius: Screen hemisphere radius when screen is None.
            left_offset, right_offset: (x, y, z) for each eye.
            left_euler_deg, right_euler_deg: (rx, ry, rz) in degrees for each eye.
            left_color, right_color: Plotly colors for each retina.
            lens_radius, lens_length: Hex prism geometry.

        Returns:
            SimulatorViz with both retinas and screen added; call .plot() to show.
        """
        left = RetinaRotator(base_retina, offset=left_offset, euler_deg=left_euler_deg).apply()
        right = RetinaRotator(base_retina, offset=right_offset, euler_deg=right_euler_deg).apply()

        viz = SimulatorViz(lens_radius=lens_radius, lens_length=lens_length)
        viz.add_retina(left, color=left_color, name="left", ray_color=left_ray_color)
        viz.add_retina(right, color=right_color, name="right", ray_color=right_ray_color)

        if screen is not None:
            viz.add_screen(screen=screen)
        else:
            viz.add_screen(radius=radius)

        return viz


if __name__ == "__main__":
    # Demo: dual-eye setup with base retina
    base = Retina(num_rings=8, radius=1)
    viz = SimulatorViz.create_dual_eye_setup(
        base_retina=base,
        radius=10,
        left_offset=(-0.6, 0, 0),
        left_euler_deg=(0, 0, 0),
        right_offset=(0.6, 0, 0),
        right_euler_deg=(0, 0, 0),
        left_color="steelblue",
        right_color="crimson",
    )
    viz.plot()
