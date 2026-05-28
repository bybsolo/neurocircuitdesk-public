import numpy as np
from tqdm import tqdm

from .backend import xp, to_numpy, free_memory, BACKEND
from .screen import Screen
from .retina import Retina


class FlyEyeSimulator:
    def __init__(self, screen: Screen, retina: Retina,
                 initial_acceptance_angle_deg=5.0, M=1.0,
                 shrink_ratio=0.1,
                 shift_ratio=1):
        """
        Initializes the simulator with vectorized physics-based receptive field dynamics.
        """
        self.screen = screen
        self.retina = retina
        self.R_ret = self.retina.radius
        self.R_scr = self.screen.radius
        self.M = M
        self.initial_window_deg = initial_acceptance_angle_deg

        # =========================================================================
        # 1. Physics Parameters (Tuned for Angular Units)
        # =========================================================================
        # Treating phys_x as DEGREES.
        # F_max scales with window size (larger RFs have more "muscle")
        self.shrink_ratio = shrink_ratio
        self.shift_ratio = shift_ratio

        self.k0 = 0.00067           # Base Stiffness
        self.k_coef = 0.0032        # Active Stiffness Gain

        # self.F_max = 0.031 * self.shift_ratio * (self.initial_window_deg / 8.0)  # Max Force
        # if shift_ratio >0:
        #     self.F_max = 0.031 * self.shift_ratio * (self.initial_window_deg / 8.0)  # Max Force
        # else:
        #     # this would be for special cases where we have try to see the effect of shrink only, then we still need to make sure the shrink is doable, we use a default
        #     self.F_max = 0.031 * (self.initial_window_deg / 8.0)
        self.F_max = 0.031 * (self.initial_window_deg / 8.0)

        self.D_coef = 0.09          # Damping Coefficient
        self.D_exp = 2.0            # Damping Exponent

        # Saturation point for luminance.
        # Note: Screen intensity might need normalization depending on input range (0-1 vs 0-255).
        # Assuming sum of luminance over a large RF, 120,000 is reasonable for 255-scale images.
        self.half_pk = 120000.0

        self.gain_slope = 0.40      # Signal boost per degree of stretch

        self.dt = self.screen.dt              # Integration time step (ms)

        # =========================================================================
        # 2. Geometry & Unit Setup
        # =========================================================================
        parallels = self.screen.parallels
        meridians = self.screen.meridians
        self.num_pixels = parallels * meridians
        self.num_neurons = len(self.retina.vrfs)

        # --- Grid Spacing (dxy) ---
        el_min_rad = -xp.pi / 2.0
        el_max_rad = xp.pi / 2.0
        az_min_rad = 0.0
        az_max_rad = xp.pi

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

        self.el_rad_flat = xp.deg2rad(el_grid_deg).reshape(-1)
        self.az_rad_flat = xp.deg2rad(az_grid_deg).reshape(-1)

        # --- Pre-compute Screen Trigonometry (Reshaped for Broadcasting) ---
        self.sin_el_scr = xp.sin(self.el_rad_flat).reshape(-1, 1)
        self.cos_el_scr = xp.cos(self.el_rad_flat).reshape(-1, 1)  # Jacobian
        self.az_scr_vec = self.az_rad_flat.reshape(-1, 1)

        # =========================================================================
        # 3. State Initialization (Backend Arrays)
        # =========================================================================
        print("Initializing Receptive Field State...")

        # Compute projection centers (ray from coord_xyz along axis_polar -> screen intersection)
        # Stored as polar coords on the screen; replaces naive axis_polar assumption
        proj_el, proj_az = self._compute_projection_centers_vectorized()

        enabled_list = [vrf.enabled for vrf in self.retina.vrfs]

        # -- Constant State --
        self.enabled_mask = xp.array(enabled_list, dtype=xp.bool_).reshape(1, -1)
        self.original_el_rad = proj_el.reshape(1, -1)
        self.original_az_rad = proj_az.reshape(1, -1)

        # -- Physics State (x=displacement, v=velocity) --
        # Shape: (1, N) to allow easy broadcasting
        self.phys_x = xp.zeros((1, self.num_neurons), dtype=xp.float32)
        self.phys_v = xp.zeros((1, self.num_neurons), dtype=xp.float32)

        # -- Derived State (Current Positions & Angles) --
        self.current_el_rad = xp.array(self.original_el_rad)
        self.current_az_rad = xp.array(self.original_az_rad)
        self.current_acc_angle_deg = xp.full((1, self.num_neurons), self.initial_window_deg, dtype=xp.float32)

        # Pre-allocate Filter Matrix
        self.filters = xp.zeros((self.num_pixels, self.num_neurons), dtype=xp.float32)

    def _compute_projection_centers_vectorized(self):
        """
        Vectorized ray-sphere intersection: for each VRF, trace from coord_xyz
        along axis_polar to intersect the screen sphere; return polar coords (el, az)
        of the intersection on the screen.
        """
        # Use precomputed retina arrays; explicit xp.asarray avoids np/xp mixing in xp.stack
        p_ret = xp.asarray(self.retina.coords_xyz)  # (N, 3)
        axis_az = xp.asarray(self.retina.axis_az)
        axis_el = xp.asarray(self.retina.axis_el)
        d_hat = xp.stack([
            xp.cos(axis_el) * xp.cos(axis_az),
            xp.cos(axis_el) * xp.sin(axis_az),
            xp.sin(axis_el),
        ], axis=1)  # (N, 3)

        # Ray-sphere: |p_ret + t*d_hat|^2 = R_scr^2
        b = 2.0 * xp.sum(p_ret * d_hat, axis=1)
        c = xp.sum(p_ret * p_ret, axis=1) - self.R_scr ** 2
        discriminant = b * b - 4.0 * c

        t = (-b + xp.sqrt(xp.maximum(discriminant, 0.0))) / 2.0
        p_int = p_ret + t[:, None] * d_hat

        # Normalize; fallback to d_hat when ray misses (discriminant < 0)
        norm_p = xp.linalg.norm(p_int, axis=1, keepdims=True)
        norm_p = xp.maximum(norm_p, 1e-9)  # avoid div/0
        center_uv = xp.where(discriminant[:, None] >= 0, p_int / norm_p, d_hat)

        # Convert to polar on screen
        el = xp.arcsin(xp.clip(center_uv[:, 2], -1.0, 1.0))
        az = xp.arctan2(center_uv[:, 1], center_uv[:, 0])

        return el.astype(xp.float32), az.astype(xp.float32)

    def _generate_filters_vectorized(self):
        """
        Recomputes the entire filter matrix using the CURRENT state arrays.
        """
        # 1. Update Kappa based on current acceptance angles
        acc_angle_rad = xp.deg2rad(self.current_acc_angle_deg)

        # Kappa Formula
        # Prevent div/0 if acc_angle becomes 0 (though clamped to 0.1 later)
        kappa_vec = xp.log(2) / (1.0 - xp.cos(acc_angle_rad / 2.0 / self.M))

        # Normalization Constant
        one_over_2pi = 1.0 / (2.0 * xp.pi)
        norm_const_vec = (kappa_vec * one_over_2pi) / (1.0 - xp.exp(-2.0 * kappa_vec))

        # 2. Compute Inner Product
        term1 = self.cos_el_scr * xp.cos(self.current_el_rad) * xp.cos(self.current_az_rad - self.az_scr_vec)
        term2 = self.sin_el_scr * xp.sin(self.current_el_rad)
        innerM1 = term1 + term2 - 1.0

        # 3. Apply VMF Distribution
        self.filters = xp.exp(kappa_vec * innerM1)
        self.filters = self.filters * norm_const_vec
        self.filters = self.filters * self.dxy
        self.filters = self.filters * self.cos_el_scr
        self.filters = self.filters * self.enabled_mask


    def _update_receptive_fields(self, response_vector):
        """
        Vectorized implementation of the Spring-Damper physics model.
        Args:
            response_vector: 1D array of shape (Neurons,) representing 'lum'.
        """
        # Reshape lum to (1, N)
        lum = response_vector.reshape(1, -1)

        # 1. BIOLOGICAL FEEDBACK
        # Boosts signal sensitivity as the sensor stretches
        compensated_act = lum * (1.0 + self.gain_slope * self.phys_x)

        # 2. ACTIVITY NORMALIZATION (Hill Equation)
        # Avoid div/0 with epsilon
        u_term = compensated_act / (self.half_pk + compensated_act + 1e-9)

        # 3. FORCE CALCULATIONS
        # Spring stiffness increases with activity
        k = self.k0 + (self.k_coef * u_term)

        # Driving Force pushed by light intensity
        F_drive = self.F_max * u_term

        # Damping (F_damp) - resistance against velocity
        # Formula: D_coef * (2^(-D_exp * v) - 1)
        # Using exp2 for base-2 power
        F_damp = self.D_coef * (xp.exp2(-self.D_exp * self.phys_v) - 1.0)

        # 4. EULER INTEGRATION
        # F = ma -> accel = Drive + Drag - SpringForce
        accel = F_drive + F_damp - (k * self.phys_x)

        self.phys_v = self.phys_v + accel * self.dt
        self.phys_x = self.phys_x + self.phys_v * self.dt

        # 5. CONSTRAINTS & MAPPING
        # Prevent negative extension
        self.phys_x = xp.maximum(0.0, self.phys_x)

        # Map Physics to Geometry
        # A. Elevation Shift (pixel_x = x0 + phys_x)
        # Convert phys_x (Degrees) to Radians and add to original elevation
        shift_rad = xp.deg2rad(self.phys_x)
        # self.current_el_rad = self.original_el_rad + shift_rad
        # self.current_el_rad = self.original_el_rad - shift_rad
        self.current_el_rad = self.original_el_rad - self.shift_ratio*shift_rad


        # B. Radius Shrinkage (pixel_R = window - shrink * phys_x)
        new_radius = self.initial_window_deg - (self.shrink_ratio * self.phys_x)
        # Clamp to minimum 0.1 degrees to prevent singularities
        self.current_acc_angle_deg = xp.maximum(0.1, new_radius)

    def run(self, release=True):
        """
        Runs the dynamic simulation loop.

        Args:
            release: If True (default), returns a numpy array and releases
                     device-side buffers at the end of the run.

        Returns:
            (T, N) float32 array — numpy when ``release=True``, otherwise
            backend-native.
        """
        num_frames = self.screen.num_frames

        print(f"Starting physics-driven simulation for {num_frames} frames...")

        # Initial Filter Generation
        self._generate_filters_vectorized()

        res_frames = []
        for t in tqdm(range(num_frames), desc="Simulating"):

            # 1. Get Frame
            frame_flat = self.flat_intensities[t:t+1]

            # 2. Compute Response
            step_response = xp.dot(frame_flat, self.filters)

            # 3. Store Result
            res_1d = step_response.reshape(-1)
            res_frames.append(res_1d)

            # 4. Update Physics State for NEXT frame
            self._update_receptive_fields(res_1d)

            # 5. Recompute Filters (using NEW state)
            if t < num_frames - 1:
                self._generate_filters_vectorized()

            # Lazy-eval checkpoint (prevents graph memory blowup on MLX)
            if BACKEND == 'mlx' and (t % 4 == 0):
                xp.eval(self.filters)

        full_response = xp.stack(res_frames, axis=0)

        if release:
            res_np = to_numpy(full_response)
            del full_response, res_frames
            self.release_device_memory()
            return res_np
        return full_response

    def run_more(self, release=True):
        """
        Runs the dynamic simulation loop and records state history.

        Args:
            release: If True (default), returns numpy arrays and releases
                     device-side buffers at the end of the run.

        Returns:
            Tuple (responses, angles, elevations):
            - responses: (T, N) float32 array of neural outputs.
            - angles:    (T, N) float32 array of acceptance angles (Degrees).
            - elevations:(T, N) float32 array of RF elevations (Degrees).
        """
        num_frames = self.screen.num_frames

        print(f"Starting physics-driven simulation for {num_frames} frames...")

        # Initial Filter Generation (using initial state t=0)
        self._generate_filters_vectorized()

        res_frames = []
        ang_frames = []
        elev_frames = []

        for t in tqdm(range(num_frames), desc="Simulating"):

            # --- A. RECORD STATE (Before Update) ---
            ang_frames.append(self.current_acc_angle_deg.reshape(-1))
            elev_frames.append(xp.rad2deg(self.current_el_rad).reshape(-1))

            # --- B. PROCESS FRAME ---
            frame_flat = self.flat_intensities[t:t+1]
            step_response = xp.dot(frame_flat, self.filters)
            res_1d = step_response.reshape(-1)
            res_frames.append(res_1d)

            # --- C. UPDATE STATE ---
            self._update_receptive_fields(res_1d)

            if t < num_frames - 1:
                self._generate_filters_vectorized()

            # Lazy-eval checkpoint
            if BACKEND == 'mlx' and (t % 4 == 0):
                xp.eval(self.filters)

        full_response = xp.stack(res_frames, axis=0)
        full_angles = xp.stack(ang_frames, axis=0)
        full_elevations = xp.stack(elev_frames, axis=0)

        if release:
            out = (to_numpy(full_response), to_numpy(full_angles), to_numpy(full_elevations))
            del full_response, full_angles, full_elevations
            del res_frames, ang_frames, elev_frames
            self.release_device_memory()
            return out
        return full_response, full_angles, full_elevations

    def release_device_memory(self):
        """
        Drop large backend-side buffers held by this simulator and the
        attached screen, then flush the memory pool.
        """
        for attr in ('filters', 'flat_intensities',
                     'el_rad_flat', 'az_rad_flat',
                     'sin_el_scr', 'cos_el_scr', 'az_scr_vec',
                     'enabled_mask',
                     'original_el_rad', 'original_az_rad',
                     'current_el_rad', 'current_az_rad', 'current_acc_angle_deg',
                     'phys_x', 'phys_v'):
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self.screen, 'release_device_memory'):
            self.screen.release_device_memory()
        free_memory()
