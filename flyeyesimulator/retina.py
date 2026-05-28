import numpy as np
import plotly.graph_objects as go
import json
import pandas as pd 
from .superposition import superposition_mask

class ReceptiveField:
    """
    Represents a single photoreceptor cell (R1-R6) within an ommatidium column.
    """
    def __init__(self, col_index, rf_type, x, y, z, az, el, enabled=True):
        self.index = col_index      # Shared Column ID (0 to N-1)
        self.rf_type = rf_type      # 'R1', 'R2', 'R3', 'R4', 'R5', or 'R6'
        
        # Coordinates (Shared by all RFs in this column)
        self.coord_xyz = np.array([x, y, z])
        self.coord_polar = {'az': az, 'el': el}
        
        # Axis normal vector (Initially same as coord; updated by superposition)
        self.axis_polar = {'az': az, 'el': el} 
        
        self.enabled = enabled

    def __repr__(self):
        return f"VRF(col={self.index}, type={self.rf_type}, enabled={self.enabled})"


class Retina:
    """
    Defines the geometry of the eye.
    Generates a grid of Columns, where each Column contains 6 ReceptiveFields.
    """
    def __init__(self, num_rings=14, radius = 1,
                 inter_ommatidia_angle_deg=4.4, 
                 bio_cols_only=False, col_json_path=None):
        self.num_rings = num_rings
        self.radius = radius
        self.inter_ommatidia_angle_deg = inter_ommatidia_angle_deg
        
        # This list will be populated with 6 RFs per column
        self.vrfs = [] 
        
        # 1. Generate the geometry
        self._generate_grid()
        
        # 2. Apply Neural Superposition (overwrite axes)
        self._apply_superposition()
        
        # 3. Apply mask if requested
        if bio_cols_only:
            self._apply_mask(col_json_path)

        # 4. Build flattened arrays for vectorized projection (coords_xyz, axis_az, axis_el)
        self._build_vrf_arrays()

    @classmethod
    def from_vrfs(cls, vrfs, radius=1, num_rings=None, inter_ommatidia_angle_deg=4.4):
        """
        Construct a Retina from an existing list of ReceptiveField objects.
        Used by RetinaRotator to create a transformed copy without re-running grid generation.
        """
        self = cls.__new__(cls)
        self.vrfs = vrfs
        self.radius = radius
        self.num_rings = num_rings if num_rings is not None else len(vrfs) // 6
        self.inter_ommatidia_angle_deg = inter_ommatidia_angle_deg
        self._build_vrf_arrays()
        return self

    def _generate_grid(self):
        """Generates spherical coordinates and populates R1-R6 for each position."""
        phi_step = np.radians(self.inter_ommatidia_angle_deg)
        
        # --- 1. Generate 2D Hex Grid Positions (Spiral) ---
        basis_angles_deg = [90, 30, -30, -90, -150, 150]
        basis_vectors = [
            (np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))) 
            for a in basis_angles_deg
        ]
        
        points_2d = [(0.0, 0.0)] 
        
        for r in range(1, self.num_rings + 1):
            x = r * basis_vectors[0][0]
            y = r * basis_vectors[0][1]
            walk_dirs = [2, 3, 4, 5, 0, 1] 
            
            for d_idx in walk_dirs:
                dx, dy = basis_vectors[d_idx]
                for _ in range(r):
                    points_2d.append((x, y))
                    x += dx
                    y += dy
                    
        points_np = np.array(points_2d)
        
        # --- 2. Map to 3D Spherical ---
        rho_2d = np.sqrt(points_np[:,0]**2 + points_np[:,1]**2)
        psi_2d = np.arctan2(points_np[:,1], points_np[:,0])
        
        arc_angle = rho_2d * phi_step
        
        # Cartesian Projection
        y_cart = np.cos(arc_angle)
        r_plane = np.sin(arc_angle)
        x_cart = r_plane * np.cos(psi_2d) 
        z_cart = r_plane * np.sin(psi_2d) 
        
        # User Azimuth/Elevation
        user_az = np.arctan2(y_cart, x_cart)
        user_el = np.arcsin(z_cart)
        
        # --- 3. Create Objects (6 per column) ---
        rf_types = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
        
        for col_idx in range(len(user_az)):
            # Create 6 RFs for this single column location
            for r_type in rf_types:
                vrf = ReceptiveField(
                    col_index=col_idx,   
                    rf_type=r_type,       
                    x=x_cart[col_idx],
                    y=y_cart[col_idx],
                    z=z_cart[col_idx],
                    az=user_az[col_idx],
                    el=user_el[col_idx],
                    enabled=True         
                )
                self.vrfs.append(vrf)

    def _apply_superposition(self):
        """
        Second pass: Overwrites axis_polar for VRFs based on neural superposition.
        For every column 'i', we project its coordinates onto the R-cells of 
        neighboring columns defined by superposition_mask(i).
        """
        # Calculate total columns (assuming 6 RFs per column)
        num_columns = len(self.vrfs) // 6
        
        for i in range(num_columns):
            # 1. Get the source coordinates from column i 
            # (We grab the first RF in the column, as all share the same coords)
            source_vrf_idx = i * 6
            source_coord_polar = self.vrfs[source_vrf_idx].coord_polar.copy()
            
            # 2. Get the list of 6 target columns [col_for_R1, col_for_R2... col_for_R6]
            try:
                target_col_indices = superposition_mask(i)
            except Exception as e:
                print(f"Warning: superposition_mask failed for column {i}: {e}")
                continue
            
            # 3. Update the specific R-type in the target columns
            # k=0 updates R1, k=1 updates R2, etc.
            for k, target_col_idx in enumerate(target_col_indices):
                
                # CHECK: Does this target column actually exist in our grid?
                if 0 <= target_col_idx < num_columns:
                    # Calculate the index in the flat list self.vrfs
                    # Target Index = (Column ID * 6) + (R-type offset k)
                    target_vrf_idx = (target_col_idx * 6) + k
                    
                    # Overwrite the axis with the source (column i) coordinates
                    self.vrfs[target_vrf_idx].axis_polar = source_coord_polar
                
                # If target_col_idx is out of bounds, we do nothing (skip),
                # leaving that RF's axis as its original value.

    def _apply_mask(self, col_json_path):
        """Disables ALL VRFs (R1-R6) associated with masked columns."""
        if not col_json_path:
            raise ValueError("bio_cols_only=True requires a valid col_json_path.")
            
        try:
            with open(col_json_path, 'r') as f:
                data = json.load(f)
            
            loaded_hex_ids = np.array(data['hex_coords_id'])
            
            # Determine max column index needed
            num_columns = len(self.vrfs) // 6
            
            # Map mask to columns
            M = len(loaded_hex_ids)
            n_common = min(M, num_columns)
            
            # Identify active COLUMNS (< 1000)
            active_cols_mask = loaded_hex_ids[:n_common] < 1000
            
            # Apply to VRFs
            for vrf in self.vrfs:
                c_idx = vrf.index
                if c_idx < n_common:
                    vrf.enabled = bool(active_cols_mask[c_idx])
                else:
                    vrf.enabled = False
                    
        except FileNotFoundError:
            print(f"Warning: JSON path {col_json_path} not found. All VRFs enabled.")

    def _build_vrf_arrays(self):
        """
        Build flattened numpy arrays from VRFs for vectorized projection.
        Call after superposition (so axis_polar is final).
        """
        self.coords_xyz = np.array([vrf.coord_xyz for vrf in self.vrfs], dtype=np.float32)
        self.axis_az = np.array([vrf.axis_polar['az'] for vrf in self.vrfs], dtype=np.float32)
        self.axis_el = np.array([vrf.axis_polar['el'] for vrf in self.vrfs], dtype=np.float32)

 