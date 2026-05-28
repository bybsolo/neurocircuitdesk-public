import numpy as np
import plotly.graph_objects as go
import json
import pandas as pd
from .retina import Retina, ReceptiveField
from plotly.subplots import make_subplots

class RetinaViz:
    """
    Visualization for retina VRFs.
    """
    def __init__(self, retina, lens_radius=0.04, lens_length=0.01):
        self.retina = retina
        self.lens_radius = lens_radius
        self.lens_length = lens_length
        
        self.default_camera = dict(
            eye=dict(x=0, y=2.0, z=0),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )

    def _rotation_matrix_from_vectors(self, vec1, vec2):
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s == 0: return np.eye(3)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    # def _create_hex_lens_mesh(self, center, normal):
    #     size = self.lens_radius
    #     length = self.lens_length
    #     angles = np.linspace(0, 2*np.pi, 7)[:-1]
    #     hex_x = size * np.cos(angles)
    #     hex_y = size * np.sin(angles)
    #     z_base = np.zeros_like(hex_x)
    #     z_tip = np.full_like(hex_x, length)
    #     verts_local = np.vstack([
    #         np.column_stack([hex_x, hex_y, z_base]), 
    #         np.column_stack([hex_x, hex_y, z_tip])
    #     ])
    #     target_normal = np.array(normal)
    #     R = self._rotation_matrix_from_vectors(np.array([0, 0, 1]), target_normal)
    #     verts_rotated = verts_local @ R.T
    #     verts_world = verts_rotated + np.array(center)
    #     faces = []
    #     for i in range(6):
    #         next_i = (i + 1) % 6
    #         faces.append([i, next_i, next_i + 6])
    #         faces.append([i, next_i + 6, i + 6])
    #     for i in range(1, 5):
    #         faces.append([0, i+1, i]) 
    #         faces.append([6, 6+i, 6+i+1])
    #     return verts_world, np.array(faces)
    
    def create_hex_lens_mesh(self, center, normal):
        """Public mesh builder for a single hexagonal lens at ``center``
        oriented along ``normal``.

        Returns ``(verts, faces)`` where ``verts`` is ``(6, 3)`` for a flat
        hex (``lens_length == 0``) or ``(12, 3)`` for an extruded prism, and
        ``faces`` is a triangle index array referencing ``verts``.

        Used by external composers (e.g., ncd-fes-pipeline.PipelineComposer)
        that need to assemble retina meshes without reaching into private
        helpers.
        """
        return self._create_hex_lens_mesh(center, normal)

    def get_r1_mesh(self):
        """Pre-assembled mesh for all enabled R1 cells across the retina.

        Returns a dict::

            {
                'x': (V,) np.ndarray of x coords,
                'y': (V,) np.ndarray of y coords,
                'z': (V,) np.ndarray of z coords,
                'i': (F,) np.ndarray of triangle vertex-0 indices,
                'j': (F,) np.ndarray of triangle vertex-1 indices,
                'k': (F,) np.ndarray of triangle vertex-2 indices,
                'kept_col_idx': (N_R1,) np.ndarray of vrf.index for each
                    R1 cell included (use this to index per-column value
                    arrays when colouring the mesh),
                'verts_per_lens': int (6 or 12),
            }

        Cached on first call. Designed for external composers that need to
        animate the retina mesh across time — they call this once for the
        geometry, then re-feed ``intensity = np.repeat(values[kept_col_idx],
        verts_per_lens)`` per frame.
        """
        if not hasattr(self, '_r1_mesh_cache'):
            verts_per_lens = 6 if abs(float(self.lens_length)) <= 1e-12 else 12
            all_x, all_y, all_z = [], [], []
            all_i, all_j, all_k = [], [], []
            kept_col_idx = []
            v_offset = 0

            for vrf in self.retina.vrfs:
                if not (vrf.rf_type == 'R1' and vrf.enabled):
                    continue
                verts, faces = self._create_hex_lens_mesh(vrf.coord_xyz, vrf.coord_xyz)
                all_x.extend(verts[:, 0]); all_y.extend(verts[:, 1]); all_z.extend(verts[:, 2])
                all_i.extend(faces[:, 0] + v_offset)
                all_j.extend(faces[:, 1] + v_offset)
                all_k.extend(faces[:, 2] + v_offset)
                v_offset += verts.shape[0]
                kept_col_idx.append(vrf.index)

            self._r1_mesh_cache = {
                'x': np.asarray(all_x),
                'y': np.asarray(all_y),
                'z': np.asarray(all_z),
                'i': np.asarray(all_i, dtype=int),
                'j': np.asarray(all_j, dtype=int),
                'k': np.asarray(all_k, dtype=int),
                'kept_col_idx': np.asarray(kept_col_idx, dtype=int),
                'verts_per_lens': verts_per_lens,
            }
        return self._r1_mesh_cache

    def _create_hex_lens_mesh(self, center, normal):
        size = float(self.lens_radius)
        length = float(self.lens_length)

        # Hex in local XY, centered at origin
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        hex_x = size * np.cos(angles)
        hex_y = size * np.sin(angles)

        target_normal = np.array(normal, dtype=float)
        R = self._rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), target_normal)

        eps = 1e-12
        if abs(length) <= eps:
            # -------------------------
            # 2D flat hexagon (6 verts)
            # -------------------------
            z0 = np.zeros_like(hex_x)
            verts_local = np.column_stack([hex_x, hex_y, z0])          # (6,3)
            verts_world = verts_local @ R.T + np.array(center, dtype=float)

            # Triangulate the hex into 4 triangles using a fan from vertex 0
            # (0,1,2), (0,2,3), (0,3,4), (0,4,5)
            faces = np.array([
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 5],
            ], dtype=int)

            return verts_world, faces
        # -------------------------
        # 3D extruded hex prism
        # -------------------------
        z_base = np.zeros_like(hex_x)
        z_tip = np.full_like(hex_x, length)

        verts_local = np.vstack([
            np.column_stack([hex_x, hex_y, z_base]),  # base  (6,3)
            np.column_stack([hex_x, hex_y, z_tip])    # top   (6,3)
        ])  # (12,3)

        verts_world = (verts_local @ R.T) + np.array(center, dtype=float)

        faces = []
        # Side faces (2 triangles per side)
        for i in range(6):
            next_i = (i + 1) % 6
            faces.append([i, next_i, next_i + 6])
            faces.append([i, next_i + 6, i + 6])

        # Caps (fan triangulation on base and top)
        for i in range(1, 5):
            faces.append([0, i + 1, i])             # base
            faces.append([6, 6 + i, 6 + i + 1])     # top

        return verts_world, np.array(faces, dtype=int)

    def plot(self, input_values, title="Ommatidia", camera = None):
        """
        Plots the retina for all enabled VRFs.
        
        Args:
            input_values: Array of intensity values. 
                          Length must match the number of COLUMNS.
            title:        Title for the plot.
        """
        # Calculate total columns to validate input
        num_columns = len(self.retina.vrfs) // 6
        
        if len(input_values) != num_columns:
            print(f"Warning: Input data length ({len(input_values)}) does not match "
                  f"number of columns ({num_columns}).")
            return

        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        all_intensity = []
        vertex_offset = 0

        # Loop through ALL VRFs in the retina
        for vrf in self.retina.vrfs:
            
            # Filter for R1 type only
            if vrf.rf_type == 'R1' and vrf.enabled:
                
                # Get intensity using the column index
                intensity = input_values[vrf.index]

                # Create Geometry
                center = vrf.coord_xyz
                normal = center 
                verts, faces = self._create_hex_lens_mesh(center, normal)
                
                all_x.extend(verts[:, 0])
                all_y.extend(verts[:, 1])
                all_z.extend(verts[:, 2])
                
                faces_offset = faces + vertex_offset
                all_i.extend(faces_offset[:, 0])
                all_j.extend(faces_offset[:, 1])
                all_k.extend(faces_offset[:, 2])
                
                all_intensity.extend([intensity] * 12)
                vertex_offset += 12

        # Viz Setup
        fig = go.Figure(data=[
            go.Mesh3d(
                x=all_x, y=all_y, z=all_z,
                i=all_i, j=all_j, k=all_k,
                intensity=all_intensity,
                colorscale='Viridis',
                opacity=1.0,
                flatshading=True,
                name=title,
                hoverinfo="skip",
                hovertemplate=None,
            )
        ])

        no_axis = dict(showbackground=True, showgrid=False, showline=False,
                       showticklabels=False, title='', visible=False)
        
        if camera is None:
            camera = self.default_camera
        
        fig.update_layout(
            scene=dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, aspectmode='data', camera=camera,),
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor='rgba(0,0,0,0)',
            width=800, height=600
        )
        fig.show()
        
        
########################################################################################


    def _build_static_mesh(self, rf_type='R1'):
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        vrfs_kept = []
        vertex_offset = 0

        for vrf in self.retina.vrfs:
            if vrf.rf_type == rf_type and vrf.enabled:
                center = vrf.coord_xyz
                normal = center
                verts, faces = self._create_hex_lens_mesh(center, normal)

                all_x.extend(verts[:, 0])
                all_y.extend(verts[:, 1])
                all_z.extend(verts[:, 2])

                faces_offset = faces + vertex_offset
                all_i.extend(faces_offset[:, 0])
                all_j.extend(faces_offset[:, 1])
                all_k.extend(faces_offset[:, 2])

                vertex_offset += verts.shape[0]  # 12
                vrfs_kept.append(vrf)

        return (
            np.asarray(all_x), np.asarray(all_y), np.asarray(all_z),
            np.asarray(all_i), np.asarray(all_j), np.asarray(all_k),
            vrfs_kept
        )

    def save_video(
        self,
        values_TN,
        title="Ommatidia",
        fps=30,
        html_path=None,
        cmin=None,
        cmax=None,
        colorscale="Viridis",
        camera = None
    ):
        """
        values_TN: (T, Ncols) where Ncols = len(retina.vrfs)//6
                  Only the COLUMN values are provided; we expand to per-vertex intensities internally.
        cmin/cmax: If None, computed globally from values_TN.
        """
        values_TN = np.asarray(values_TN)
        if values_TN.ndim != 2:
            raise ValueError("values_TN must be 2D: (T, Ncols).")

        T, N = values_TN.shape
        num_columns = len(self.retina.vrfs) // 6
        if N != num_columns:
            raise ValueError(f"values_TN has N={N}, expected {num_columns} columns.")

        # --- global color range (consistent across all frames) ---
        if cmin is None:
            cmin = float(np.nanmin(values_TN))
        if cmax is None:
            cmax = float(np.nanmax(values_TN))
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
            raise ValueError(f"Bad color range: cmin={cmin}, cmax={cmax}")

        # --- build geometry once (R1 type only) ---
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        vrfs_kept = []
        vertex_offset = 0

        for vrf in self.retina.vrfs:
            if vrf.rf_type == 'R1' and vrf.enabled:
                center = vrf.coord_xyz
                normal = center
                verts, faces = self._create_hex_lens_mesh(center, normal)

                all_x.extend(verts[:, 0])
                all_y.extend(verts[:, 1])
                all_z.extend(verts[:, 2])

                faces_offset = faces + vertex_offset
                all_i.extend(faces_offset[:, 0])
                all_j.extend(faces_offset[:, 1])
                all_k.extend(faces_offset[:, 2])

                vertex_offset += verts.shape[0]
                vrfs_kept.append(vrf)

        x = np.asarray(all_x)
        y = np.asarray(all_y)
        z = np.asarray(all_z)
        i = np.asarray(all_i)
        j = np.asarray(all_j)
        k = np.asarray(all_k)

        # Map each kept VRF to its column index; repeat each lens value for its 12 vertices
        kept_col_idx = np.array([vrf.index for vrf in vrfs_kept], dtype=int)

        verts_per_lens = 6 if abs(self.lens_length) <= 1e-12 else 12

        def intensity_for_t(t: int) -> np.ndarray:
            per_lens = values_TN[t, kept_col_idx].astype(np.float32)
            return np.repeat(per_lens, verts_per_lens)

        intensity0 = intensity_for_t(0)

        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            intensity=intensity0,
            colorscale=colorscale,
            cmin=cmin,          # <-- fixed global min
            cmax=cmax,          # <-- fixed global max
            showscale=True,     # show one consistent colorbar
            opacity=1.0,
            flatshading=True,
            hoverinfo="skip",
            hovertemplate=None,
            name=title,
        )

        # Frames only update intensity (and nothing else)
        frames = [
            go.Frame(
                name=str(t),
                data=[go.Mesh3d(intensity=intensity_for_t(t))],
                traces=[0],
            )
            for t in range(T)
        ]

        # Slider steps
        steps = [
            dict(
                method="animate",
                args=[[str(t)], {"mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0}}],
                label=str(t),
            )
            for t in range(T)
        ]

        fig = go.Figure(data=[mesh], frames=frames)

        no_axis = dict(showbackground=True, showgrid=False, showline=False,
                       showticklabels=False, title="", visible=False)

        if camera is None:
            camera = self.default_camera
        
        fig.update_layout(
            scene=dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, aspectmode="data", camera=camera,),
            margin=dict(l=0, r=0, b=0, t=0),
            width=800, height=600,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {
                            "fromcurrent": True,
                            "mode": "immediate",
                            "frame": {"duration": int(1000 / fps), "redraw": True},
                            "transition": {"duration": 0},
                        }],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        }],
                    ),
                ],
            )],
            sliders=[dict(active=0, currentvalue={"prefix": "t = "}, steps=steps)],
        )

        if html_path is not None:
            # Standalone (bigger but portable). Use "cdn" if you're ok requ, iring internet.
            fig.data[0].intensity = intensity_for_t(0)
            fig.layout.sliders[0].active = 0
            fig.write_html(html_path, include_plotlyjs=True, full_html=True, auto_play=False, )

        return fig

    def save_video_row(
            self,
            list_values_TN,
            title_list,
            fps=30,
            html_path=None,
            cmin=None,
            cmax=None,
            colorscale="Viridis",
            camera = None,
            dt_ms = 10
        ):
        verts_per_lens = 6 if abs(self.lens_length) <= 1e-12 else 12
        ncols = len(list_values_TN)

        if len(title_list) != ncols:
            raise ValueError(
                f"Expected {ncols} titles in title_list (one per column). "
                f"Got {len(title_list)} titles."
            )

        T = list_values_TN[0].shape[0]

        # 1. Global Color Range
        if cmin is None or cmax is None:
            all_data = np.concatenate([np.asarray(v) for v in list_values_TN])
            if cmin is None: cmin = float(np.nanmin(all_data))
            if cmax is None: cmax = float(np.nanmax(all_data))

        # 2. Setup Subplots
        fig = make_subplots(
            rows=1, cols=ncols,
            specs=[[{'type': 'scene'}] * ncols],
            subplot_titles=title_list,
            horizontal_spacing=0.02
        )

        # 3. Build geometry once for R1 type only
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        vrfs_kept = []
        vertex_offset = 0

        for vrf in self.retina.vrfs:
            if vrf.rf_type == 'R1' and vrf.enabled:
                center = vrf.coord_xyz
                normal = center
                verts, faces = self._create_hex_lens_mesh(center, normal)

                all_x.extend(verts[:, 0])
                all_y.extend(verts[:, 1])
                all_z.extend(verts[:, 2])

                faces_offset = faces + vertex_offset
                all_i.extend(faces_offset[:, 0])
                all_j.extend(faces_offset[:, 1])
                all_k.extend(faces_offset[:, 2])

                vertex_offset += verts.shape[0]
                vrfs_kept.append(vrf)

        x = np.asarray(all_x)
        y = np.asarray(all_y)
        z = np.asarray(all_z)
        i = np.asarray(all_i)
        j = np.asarray(all_j)
        k = np.asarray(all_k)
        kept_col_idx = np.array([vrf.index for vrf in vrfs_kept], dtype=int)

        def get_intensities_at_t(t):
            intensities = []
            for idx in range(ncols):
                per_lens = list_values_TN[idx][t, kept_col_idx].astype(np.float32)
                intensities.append(np.repeat(per_lens, verts_per_lens))
            return intensities

        # 4. Add Initial Traces (t=0)
        initial_intensities = get_intensities_at_t(0)
        for idx in range(ncols):
            fig.add_trace(
                go.Mesh3d(
                    x=x, y=y, z=z, i=i, j=j, k=k,
                    intensity=initial_intensities[idx],
                    colorscale=colorscale, cmin=cmin, cmax=cmax,
                    showscale=(idx == ncols - 1), flatshading=True,
                    hoverinfo="skip",
                    hovertemplate=None
                ),
                row=1, col=idx + 1
            )

        # 5. Build Frames (Synchronized)
        frames = []
        for t in range(T):
            t_intensities = get_intensities_at_t(t)
            frames.append(go.Frame(
                name=str(t),
                data=[go.Mesh3d(intensity=t_intensities[idx], hoverinfo="skip", hovertemplate=None) for idx in range(ncols)],
                traces=list(range(ncols)) 
            ))

        # 6. Define Camera Angle (Centered at Y-axis)
        # Eye: (0, 2, 0) looks from the positive Y direction towards the origin
        # Up:  (0, 0, 1) keeps the Z-axis pointing 'up'

        # 7. Layout, Slider, and Menus
        no_axis = dict(showbackground=False, showgrid=False, showline=False, 
                       showticklabels=False, title="", visible=False)
        if camera is None:
            camera = self.default_camera
        # Apply camera and axis settings to all scenes
        scene_updates = {}
        for i in range(1, ncols + 1):
            scene_key = f'scene{i if i > 1 else ""}'
            scene_updates[scene_key] = dict(
                xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, 
                aspectmode='data', camera=camera
            )

        # Slider logic
        steps = [
            dict(
                method="animate",
                label=str(int(round(t * dt_ms))),
                args=[[str(t)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]
            ) for t in range(T)
        ]

        fig.update_layout(
            **scene_updates,
            margin=dict(l=10, r=10, b=10, t=50),
            width=int(1600 * ncols / 6), height=500,
            updatemenus=[dict(
                type="buttons",
                x=0.1, y=0, xanchor="right", yanchor="top",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": int(1000/fps), "redraw": True}}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": True}}])
                ]
            )],
            sliders=[dict(active=0, currentvalue={"prefix": "Time (ms): "}, steps=steps, pad={"t": 50})]
        )

        fig.frames = frames

        if html_path:
            fig.write_html(html_path)

        return fig

    def save_video_rows(
        self,
        row1_list_values_TN,
        row2_list_values_TN,
        row1_title_list,
        row2_title_list,
        fps=30,
        html_path=None,
        cmin1=None,
        cmax1=None,
        cmin2=None,
        cmax2=None,
        colorscale="Viridis",
        camera=None,
        dt_ms=None,  # optional: if you want slider labels in ms
    ):
        ncols = len(row1_list_values_TN)

        if len(row2_list_values_TN) != ncols:
            raise ValueError(
                f"Expected {ncols} arrays in row2_list_values_TN (must match row1). "
                f"Got {len(row2_list_values_TN)} arrays."
            )
        if len(row1_title_list) != ncols or len(row2_title_list) != ncols:
            raise ValueError(
                f"Expected {ncols} titles per row. "
                f"Got row1_titles={len(row1_title_list)}, row2_titles={len(row2_title_list)}."
            )

        row1_list_values_TN = [np.asarray(v) for v in row1_list_values_TN]
        row2_list_values_TN = [np.asarray(v) for v in row2_list_values_TN]

        for r, lst in enumerate([row1_list_values_TN, row2_list_values_TN], start=1):
            for c, v in enumerate(lst):
                if v.ndim != 2:
                    raise ValueError(f"Row {r}, col {c}: values must be 2D (T, Ncols). Got shape={v.shape}.")

        T1 = row1_list_values_TN[0].shape[0]
        T2 = row2_list_values_TN[0].shape[0]
        if T1 != T2:
            raise ValueError(f"Row 1 T={T1} does not match Row 2 T={T2}.")
        T = T1

        num_columns = len(self.retina.vrfs) // 6
        for r, lst in enumerate([row1_list_values_TN, row2_list_values_TN], start=1):
            for c, v in enumerate(lst):
                if v.shape[1] != num_columns:
                    raise ValueError(
                        f"Row {r}, col {c}: Ncols={v.shape[1]} but expected {num_columns} (=len(retina.vrfs)//6)."
                    )

        verts_per_lens = 6 if abs(float(self.lens_length)) <= 1e-12 else 12

        # Per-row color ranges
        if cmin1 is None or cmax1 is None:
            all1 = np.concatenate([v.reshape(-1) for v in row1_list_values_TN])
            if cmin1 is None: cmin1 = float(np.nanmin(all1))
            if cmax1 is None: cmax1 = float(np.nanmax(all1))

        if cmin2 is None or cmax2 is None:
            all2 = np.concatenate([v.reshape(-1) for v in row2_list_values_TN])
            if cmin2 is None: cmin2 = float(np.nanmin(all2))
            if cmax2 is None: cmax2 = float(np.nanmax(all2))

        def _check_range(name, a, b):
            if not np.isfinite(a) or not np.isfinite(b) or a == b:
                raise ValueError(f"Bad {name} color range: cmin={a}, cmax={b}")

        _check_range("row1", cmin1, cmax1)
        _check_range("row2", cmin2, cmax2)

        # Build geometry once for R1 type only; reuse for all columns
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        vrfs_kept = []
        vertex_offset = 0

        for vrf in self.retina.vrfs:
            if vrf.rf_type == 'R1' and vrf.enabled:
                center = vrf.coord_xyz
                normal = center
                verts, faces = self._create_hex_lens_mesh(center, normal)

                all_x.extend(verts[:, 0])
                all_y.extend(verts[:, 1])
                all_z.extend(verts[:, 2])

                faces_offset = faces + vertex_offset
                all_i.extend(faces_offset[:, 0])
                all_j.extend(faces_offset[:, 1])
                all_k.extend(faces_offset[:, 2])

                vertex_offset += verts.shape[0]
                vrfs_kept.append(vrf)

        x = np.asarray(all_x)
        y = np.asarray(all_y)
        z = np.asarray(all_z)
        i = np.asarray(all_i)
        j = np.asarray(all_j)
        k = np.asarray(all_k)
        kept_col_idx = np.array([vrf.index for vrf in vrfs_kept], dtype=int)

        def intensities_row_at_t(row_list_values_TN, t):
            out = []
            for idx in range(ncols):
                per_lens = row_list_values_TN[idx][t, kept_col_idx].astype(np.float32)
                out.append(np.repeat(per_lens, verts_per_lens))
            return out

        subplot_titles = row1_title_list + row2_title_list

        fig = make_subplots(
            rows=2,
            cols=ncols,
            specs=[[{"type": "scene"}] * ncols, [{"type": "scene"}] * ncols],
            subplot_titles=subplot_titles,
            horizontal_spacing=0.02,
            vertical_spacing=0.06,
        )

        if camera is None:
            camera = self.default_camera

        no_axis = dict(
            showbackground=False, showgrid=False, showline=False,
            showticklabels=False, title="", visible=False
        )

        # Put two independent colorbars on the far right
        def colorbar_for_row(row_idx):
            y = 0.78 if row_idx == 1 else 0.22
            return dict(x=1.02, y=y, len=0.40, thickness=14)

        # Add initial traces (t=0), with hover disabled
        # Add all row1 traces first, then all row2 traces to match frame data order
        row1_int0 = intensities_row_at_t(row1_list_values_TN, 0)
        row2_int0 = intensities_row_at_t(row2_list_values_TN, 0)

        # Add row1 traces
        for col in range(ncols):
            fig.add_trace(
                go.Mesh3d(
                    x=x, y=y, z=z, i=i, j=j, k=k,
                    intensity=row1_int0[col],
                    colorscale=colorscale,
                    cmin=cmin1, cmax=cmax1,
                    showscale=(col == ncols - 1),
                    colorbar=(colorbar_for_row(1) if col == ncols - 1 else None),
                    flatshading=True,
                    opacity=1.0,
                    hoverinfo="skip",
                    hovertemplate=None,
                ),
                row=1, col=col + 1
            )

        # Add row2 traces
        for col in range(ncols):
            fig.add_trace(
                go.Mesh3d(
                    x=x, y=y, z=z, i=i, j=j, k=k,
                    intensity=row2_int0[col],
                    colorscale=colorscale,
                    cmin=cmin2, cmax=cmax2,
                    showscale=(col == ncols - 1),
                    colorbar=(colorbar_for_row(2) if col == ncols - 1 else None),
                    flatshading=True,
                    opacity=1.0,
                    hoverinfo="skip",
                    hovertemplate=None,
                ),
                row=2, col=col + 1
            )

        # Frames: update intensities using dict(type="mesh3d", ...) so hover stays off
        frames = []
        for t in range(T):
            r1 = intensities_row_at_t(row1_list_values_TN, t)
            r2 = intensities_row_at_t(row2_list_values_TN, t)

            frame_data = []
            for col in range(ncols):
                frame_data.append(dict(type="mesh3d", intensity=r1[col]))
            for col in range(ncols):
                frame_data.append(dict(type="mesh3d", intensity=r2[col]))

            frames.append(go.Frame(
                name=str(t),
                data=frame_data,
                traces=list(range(2 * ncols)),
            ))

        # Apply scene settings to all scenes
        scene_updates = {}
        for r in range(1, 3):
            for c in range(1, ncols + 1):
                scene_number = (r - 1) * ncols + c
                key = "scene" if scene_number == 1 else f"scene{scene_number}"
                scene_updates[key] = dict(
                    xaxis=no_axis, yaxis=no_axis, zaxis=no_axis,
                    aspectmode="data",
                    camera=camera,
                )

        # Slider labels (frames by default; ms if dt_ms provided)
        def step_label(t):
            if dt_ms is None:
                return str(t)
            return str(int(round(t * float(dt_ms))))

        steps = [
            dict(
                method="animate",
                label=step_label(t),
                args=[[str(t)], {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                }],
            )
            for t in range(T)
        ]

        current_prefix = "Time: " if dt_ms is None else "Time (ms): "

        fig.update_layout(
            **scene_updates,
            margin=dict(l=10, r=80, b=10, t=60),
            width=int(1600 * ncols / 6),
            height=900,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.12, y=0.02,
                xanchor="left", yanchor="bottom",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {
                            "fromcurrent": True,
                            "mode": "immediate",
                            "frame": {"duration": int(1000 / fps), "redraw": True},
                            "transition": {"duration": 0},
                        }],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        }],
                    ),
                ],
            )],
            sliders=[dict(
                active=0,
                currentvalue={"prefix": current_prefix},
                steps=steps,
                pad={"t": 40},
            )],
        )

        fig.frames = frames

        if html_path:
            fig.write_html(html_path, include_plotlyjs=True, full_html=True, auto_play=False)

        return fig
