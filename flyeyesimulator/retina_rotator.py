"""
RetinaRotator: Apply mirror, rotation, and translation to a Retina.
Pipeline: (1) rotate at origin, (2) apply superposition on rotated coord_polar,
          (3) translate positions. axis_polar is direction-only; unchanged by translation.
"""
import numpy as np
from .retina import Retina, ReceptiveField
from .superposition import superposition_mask


def _euler_xyz_to_rotation_matrix(rx, ry, rz):
    """
    Build rotation matrix from Euler angles (radians) in XYZ intrinsic order.
    Applies: R = Rz(rz) @ Ry(ry) @ Rx(rx)
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def _polar_to_unit_vector(az, el):
    """Convert (az, el) in radians to unit direction vector [x, y, z]."""
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ], dtype=np.float64)


def _unit_vector_to_polar(v):
    """Convert unit vector [x, y, z] to (az, el) in radians."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 0.0, 0.0
    v = v / n
    el = np.arcsin(np.clip(v[2], -1.0, 1.0))
    az = np.arctan2(v[1], v[0])
    return az, el


def _xyz_to_polar(x, y, z):
    """Convert Cartesian (x,y,z) to polar (az, el) in radians (direction from origin)."""
    r = np.sqrt(x * x + y * y + z * z)
    if r < 1e-12:
        return 0.0, 0.0
    el = np.arcsin(np.clip(z / r, -1.0, 1.0))
    az = np.arctan2(y, x)
    return az, el


def _mirror_matrix(axes):
    """
    Build reflection matrix. axes: 'x', 'y', 'z' or list thereof.
    - 'x': reflect in yz-plane (x -> -x)
    - 'y': reflect in xz-plane (y -> -y)
    - 'z': reflect in xy-plane (z -> -z)
    """
    M = np.eye(3, dtype=np.float64)
    axes = (axes,) if isinstance(axes, str) else axes
    for a in axes:
        if a == "x":
            M[0, 0] *= -1
        elif a == "y":
            M[1, 1] *= -1
        elif a == "z":
            M[2, 2] *= -1
    return M


class RetinaRotator:
    """
    Applies offset and Euler rotation to a Retina, returning a new Retina
    with transformed coord_xyz, coord_polar, and axis_polar for each VRF.
    """

    def __init__(
        self,
        retina: Retina,
        offset=(0, 0, 0),
        euler_rad=None,
        euler_deg=None,
        mirror=None,
    ):
        """
        Args:
            retina: Source Retina object (centered at origin, default orientation).
            offset: Translation (x, y, z) applied after rotation.
            euler_rad: Euler angles (rx, ry, rz) in radians, XYZ intrinsic order.
            euler_deg: Euler angles in degrees (overrides euler_rad if provided).
            mirror: Reflection axis/axes: 'x', 'y', 'z' or list (e.g. ['x']).
                    Applied before rotation. 'x' = reflect in yz-plane (x -> -x).
        """
        self.retina = retina
        self.offset = np.asarray(offset, dtype=np.float64)
        if euler_deg is not None:
            self.euler_rad = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
        elif euler_rad is not None:
            self.euler_rad = np.asarray(euler_rad, dtype=np.float64)
        else:
            self.euler_rad = np.zeros(3)
        self.R = _euler_xyz_to_rotation_matrix(*self.euler_rad)
        self.M = _mirror_matrix(mirror) if mirror is not None else np.eye(3)

    def apply(self) -> Retina:
        """
        Pipeline:
        1. Rotate (with optional mirror) at origin: coord_xyz, coord_polar recomputed.
        2. Apply superposition: use rotated coord_polar to overwrite axis_polar (same logic as Retina).
        3. Translate: add offset to coord_xyz; coord_polar, axis_polar unchanged (direction only).
        """
        T = self.R @ self.M  # mirror then rotate (no translation)

        # --- Step 1: Rotate at origin ---
        rotated_vrfs = []
        for vrf in self.retina.vrfs:
            p_rot = T @ np.asarray(vrf.coord_xyz, dtype=np.float64)
            az, el = _xyz_to_polar(p_rot[0], p_rot[1], p_rot[2])

            new_vrf = ReceptiveField(
                col_index=vrf.index,
                rf_type=vrf.rf_type,
                x=float(p_rot[0]),
                y=float(p_rot[1]),
                z=float(p_rot[2]),
                az=az,
                el=el,
                enabled=vrf.enabled,
            )
            new_vrf.axis_polar = {"az": az, "el": el}  # initially same as coord_polar
            rotated_vrfs.append(new_vrf)

        # --- Step 2: Superposition (same logic as Retina._apply_superposition) ---
        num_columns = len(rotated_vrfs) // 6
        for i in range(num_columns):
            source_vrf_idx = i * 6
            source_coord_polar = rotated_vrfs[source_vrf_idx].coord_polar.copy()

            try:
                target_col_indices = superposition_mask(i)
            except Exception:
                continue

            for k, target_col_idx in enumerate(target_col_indices):
                if 0 <= target_col_idx < num_columns:
                    target_vrf_idx = (target_col_idx * 6) + k
                    rotated_vrfs[target_vrf_idx].axis_polar = source_coord_polar.copy()

        # --- Step 3: Translation (coord_xyz only; coord_polar, axis_polar unchanged) ---
        offset = self.offset
        for vrf in rotated_vrfs:
            vrf.coord_xyz = vrf.coord_xyz + offset
            # az, el = _xyz_to_polar(vrf.coord_xyz[0], vrf.coord_xyz[1], vrf.coord_xyz[2])
            # vrf.coord_polar = {"az": az, "el": el}
            # coord_polar, axis_polar left unchanged - it is direction-only for ray projection

        return Retina.from_vrfs(
            rotated_vrfs,
            radius=self.retina.radius,
            num_rings=self.retina.num_rings,
            inter_ommatidia_angle_deg=self.retina.inter_ommatidia_angle_deg,
        )
