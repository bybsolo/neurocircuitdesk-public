import numpy as np
from tqdm import tqdm

from .backend import xp, to_numpy, free_memory, BACKEND
from .screen import Screen
from .retina import Retina


class FlyEyeSimulator:
    def __init__(self, screen: Screen, retina: Retina, acceptance_angle_deg=5.0, M=1.0):
        """
        Initializes the simulator, pre-computes filters, and pre-flattens video data.
        """
        self.screen = screen
        self.retina = retina

        self.R_ret = self.retina.radius
        self.R_scr = self.screen.radius

        # 1. Geometry & Unit Setup
        parallels = self.screen.parallels
        meridians = self.screen.meridians
        self.num_pixels = parallels * meridians
        self.num_neurons = len(self.retina.vrfs)

        el_min_rad = -xp.pi / 2.0
        el_max_rad = xp.pi / 2.0
        az_min_rad = 0.0
        az_max_rad = xp.pi

        # --- Calculate Grid Spacing (dxy) ---
        el_span = el_max_rad - el_min_rad
        az_span = az_max_rad - az_min_rad

        d_el = el_span / (parallels - 1) if parallels > 1 else el_span
        d_az = az_span / (meridians - 1) if meridians > 1 else az_span
        self.dxy = d_el * d_az

        # --- Pre-flatten Screen Intensity ---
        print("Pre-flattening screen video tensor...")
        self.flat_intensities = self.screen.intensities.reshape(self.screen.num_frames, self.num_pixels)

        # --- Pre-compute Screen Coordinates ---
        el_grid_deg, az_grid_deg = self.screen.get_coordinates()

        # Convert to Radians and Flatten
        self.el_rad_flat = xp.deg2rad(el_grid_deg).reshape(-1)
        self.az_rad_flat = xp.deg2rad(az_grid_deg).reshape(-1)

        # Pre-compute Trigonometry
        self.sin_el_scr = xp.sin(self.el_rad_flat)
        self.cos_el_scr = xp.cos(self.el_rad_flat)  # Correct Jacobian term
        # self.cos_az_scr removed as it caused the sign error

        acc_angle_rad = np.deg2rad(acceptance_angle_deg)

        # 1. Pre-compute Screen Unit Vectors (Target points for the filters)
        # These are the positions of the pixels on the screen sphere
        # Standard spherical-to-Cartesian: x=cos(el)cos(az), y=cos(el)sin(az), z=sin(el)
        self.P_pixel = xp.stack([
            self.cos_el_scr * xp.cos(self.az_rad_flat),  # X
            self.cos_el_scr * xp.sin(self.az_rad_flat),  # Y
            self.sin_el_scr                              # Z
        ], axis=1) * self.R_scr

        # Compute Kappa (plain float to avoid numpy/mlx type clashes)
        self.kappa = float(np.log(2) / (1 - np.cos(acc_angle_rad / 2.0 / M)))

        # Normalization Constant
        one_over_2pi = 1.0 / (2.0 * np.pi)
        self.norm_const = float((self.kappa * one_over_2pi) / (1.0 - np.exp(-2.0 * self.kappa)))

        self.filters = xp.zeros((self.num_pixels, self.num_neurons), dtype=xp.float32)
        self.enabled_mask = xp.array([vrf.enabled for vrf in self.retina.vrfs], dtype=xp.bool_).reshape(1, -1)
        print(f"Generating filters for {self.num_neurons} VRFs")

        # Vectorized path (used at init)
        self._generate_filters_vectorized()

    def _compute_projection_centers_vectorized(self):
        """
        Vectorized ray-sphere intersection: for each VRF, trace from coord_xyz
        along axis_polar to intersect the screen sphere; return polar coords (el, az)
        of the intersection on the screen.
        """
        p_ret = xp.asarray(self.retina.coords_xyz)  # (N, 3)
        axis_az = xp.asarray(self.retina.axis_az)
        axis_el = xp.asarray(self.retina.axis_el)
        d_hat = xp.stack([
            xp.cos(axis_el) * xp.cos(axis_az),
            xp.cos(axis_el) * xp.sin(axis_az),
            xp.sin(axis_el),
        ], axis=1)  # (N, 3)

        b = 2.0 * xp.sum(p_ret * d_hat, axis=1)
        c = xp.sum(p_ret * p_ret, axis=1) - self.R_scr ** 2
        discriminant = b * b - 4.0 * c

        t = (-b + xp.sqrt(xp.maximum(discriminant, 0.0))) / 2.0
        p_int = p_ret + t[:, None] * d_hat

        norm_p = xp.linalg.norm(p_int, axis=1, keepdims=True)
        norm_p = xp.maximum(norm_p, 1e-9)
        center_uv = xp.where(discriminant[:, None] >= 0, p_int / norm_p, d_hat)

        el = xp.arcsin(xp.clip(center_uv[:, 2], -1.0, 1.0))
        az = xp.arctan2(center_uv[:, 1], center_uv[:, 0])
        return el.astype(xp.float32), az.astype(xp.float32)

    def _generate_filters_vectorized(self):
        """
        Vectorized filter computation: precompute all projection centers once,
        then compute vMF filters in one batch.
        """
        proj_el, proj_az = self._compute_projection_centers_vectorized()
        proj_el = proj_el.reshape(1, -1)
        proj_az = proj_az.reshape(1, -1)

        cos_el_scr = self.cos_el_scr.reshape(-1, 1)
        sin_el_scr = self.sin_el_scr.reshape(-1, 1)
        az_scr_vec = self.az_rad_flat.reshape(-1, 1)

        term1 = cos_el_scr * xp.cos(proj_el) * xp.cos(proj_az - az_scr_vec)
        term2 = sin_el_scr * xp.sin(proj_el)
        innerM1 = term1 + term2 - 1.0

        decay = xp.exp(self.kappa * innerM1)
        self.filters = self.norm_const * decay * self.dxy * cos_el_scr
        self.filters = self.filters * self.enabled_mask

    def _get_projection_center(self, vrf):
        """
        Calculates the 3D intersection point on the screen for a VRF.
        """
        p_ret = xp.array(vrf.coord_xyz)
        az2, el2 = vrf.axis_polar['az'], vrf.axis_polar['el']
        d_hat = xp.array([
                xp.cos(el2) * xp.cos(az2),  # X
                xp.cos(el2) * xp.sin(az2),  # Y
                xp.sin(el2)                 # Z
            ])

        # C. Ray-Sphere Intersection: |p_ret + t*d_hat|^2 = R_scr^2
        # Solve quadratic: t^2 + 2t(p_ret . d_hat) + (|p_ret|^2 - R_scr^2) = 0
        b = 2.0 * xp.dot(p_ret, d_hat)
        c = xp.dot(p_ret, p_ret) - self.R_scr**2

        discriminant = b**2 - 4*c
        if discriminant < 0:
            # Handle edge case where axis doesn't hit screen (pointing away)
            # Default to the axis direction as a fallback
            return d_hat
        # We take the positive root (forward projection)
        t = (-b + xp.sqrt(discriminant)) / 2.0
        p_int = p_ret + t * d_hat

        # Normalize to get the unit vector center for vMF
        return p_int / xp.linalg.norm(p_int)

    def _generate_single_filter(self, vrf):
        """
        Computes the Von Mises-Fisher distribution for a single Receptive Field.
        """
        center_uv = self._get_projection_center(vrf)

        # Screen pixel unit vectors
        pixel_uvs = self.P_pixel / self.R_scr

        # Spherical Dot Product (Cosine similarity)
        cos_theta = xp.dot(pixel_uvs, center_uv)

        innerM1 = cos_theta - 1.0
        decay_component = xp.exp(self.kappa * innerM1)

        return (self.norm_const * decay_component * self.dxy * self.cos_el_scr)

    def run_step(self, t):
        """
        Computes response at time index t using the pre-flattened intensity array.
        """
        frame_flat = self.flat_intensities[t:t+1]
        response = xp.dot(frame_flat, self.filters)
        return response.reshape(-1)

    def run(self, release=True):
        """
        Computes the neural response for the entire video sequence.

        Args:
            release: If True (default), returns a numpy array and releases
                     device-side buffers (filters, flat_intensities, screen
                     intensities) at the end of the run. Set False to keep the
                     simulator alive for more calls.

        Returns:
            (T, N) float32 array — numpy when ``release=True``, otherwise
            backend-native.
        """
        num_frames = self.screen.num_frames

        print(f"Computing response for {num_frames} frames...")

        res_frames = []
        for t in tqdm(range(num_frames), desc="Processing Frames"):
            res_frames.append(self.run_step(t))
            if BACKEND == 'mlx' and (t % 4 == 0):
                xp.eval(res_frames[-1])
        res = xp.stack(res_frames, axis=0)

        if release:
            res_np = to_numpy(res)
            del res, res_frames
            self.release_device_memory()
            return res_np
        return res

    def release_device_memory(self):
        """
        Drop large backend-side buffers held by this simulator and the
        attached screen, then flush the memory pool.
        """
        for attr in ('filters', 'flat_intensities', 'P_pixel',
                     'el_rad_flat', 'az_rad_flat', 'sin_el_scr', 'cos_el_scr',
                     'enabled_mask'):
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self.screen, 'release_device_memory'):
            self.screen.release_device_memory()
        free_memory()
