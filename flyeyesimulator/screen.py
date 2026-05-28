import numpy as np
from .backend import xp, xp_ndimage, to_numpy, free_memory


class Screen:
    """
    Represents a 3D spherical screen that has been illuminated by a 2D video.
    All projections are calculated immediately upon instantiation.
    """

    def __init__(self, video_tensor, radius=10, parallels=300, meridians=300, dt=10):
        """
        Initializes the screen geometry and projects the input video onto it.

        Args:
            video_tensor: Array of shape (T, H, W). Accepted as either numpy or
                          backend-native; will be moved to the active backend.
            radius: Radius of the projection sphere.
            parallels: Number of elevation grid points.
            meridians: Number of azimuth grid points.
        """
        # 1. Store base geometry parameters
        self.radius = radius
        self.parallels = parallels
        self.meridians = meridians
        self.dt = dt

        # Normalize input to active backend
        video_tensor = xp.asarray(video_tensor)

        # 2. Extract video dimensions
        self.num_frames, self.img_h, self.img_w = video_tensor.shape

        # 3. Define the spherical coordinate grid
        # Elevation: [-90, 90], Azimuth: [0, 180]
        el_range = xp.linspace(-90, 90, parallels, dtype=xp.float32)
        az_range = xp.linspace(0, 180, meridians, dtype=xp.float32)
        self.el_grid, self.az_grid = xp.meshgrid(el_range, az_range, indexing='ij')

        # 4. Compute 2D mapping and Project Video
        self._precompute_albers_map()
        self._intensities = self._project_video(video_tensor)

    def _precompute_albers_map(self):
        el_rad = xp.deg2rad(self.el_grid)
        az_rad = xp.deg2rad(self.az_grid)

        # 1. Convert to 3D Cartesian coordinates
        x0 = xp.cos(el_rad) * xp.cos(az_rad)
        y0 = xp.cos(el_rad) * xp.sin(az_rad)
        z0 = xp.sin(el_rad)

        # 2. Rotate 90 degrees around X-axis.
        # This shifts the point (el=0, az=90) to the "North Pole" (0, 0, 1)
        x1 = x0
        y1 = -z0
        z1 = y0

        # 3. Convert back to shifted spherical coordinates
        el_prime = xp.arcsin(z1)
        az_prime = xp.arctan2(y1, x1)

        # 4. Standard Albers Projection on the shifted coordinates
        rxy = self.radius * xp.sqrt(1 - xp.sin(el_prime))
        x_proj = -rxy * xp.cos(az_prime)
        y_proj = -rxy * xp.sin(az_prime)

        # 5. Fit to the image center
        # Since we are mapping a hemisphere, the max radius is simply 'self.radius'
        max_r = self.radius

        center_x = (self.img_w - 1) / 2.0
        center_y = (self.img_h - 1) / 2.0

        # Scale so the hemisphere perfectly touches the 128x128 image boundaries
        scale = center_x / max_r

        self.map_x = center_x + (x_proj * scale)
        self.map_y = center_y + (y_proj * scale)

        # Flatten for the 3D map_coordinates function
        self.map_x = self.map_x.reshape(-1)
        self.map_y = self.map_y.reshape(-1)

    def _project_video(self, video_tensor):
        """
        Internal: Runs the GPU interpolation for the entire video volume.
        """
        num_pixels = self.parallels * self.meridians

        # Create Time Coordinates
        t_coords = xp.arange(self.num_frames, dtype=xp.float32).reshape(self.num_frames, 1)
        t_coords = xp.repeat(t_coords, num_pixels, axis=1).reshape(-1)

        # Create Spatial Coordinates
        y_coords = xp.tile(self.map_y, self.num_frames)
        x_coords = xp.tile(self.map_x, self.num_frames)

        coordinates = xp.stack([t_coords, y_coords, x_coords])

        # Fast interpolation (GPU on cupy, CPU on numpy+scipy)
        output_flat = xp_ndimage.map_coordinates(
            video_tensor, coordinates, order=1, mode='nearest'
        )

        return output_flat.reshape(self.num_frames, self.parallels, self.meridians)

    # ==========================================
    # Public Getters & Properties
    # ==========================================

    @property
    def intensities(self):
        """Returns the full 3D array of screen intensities (T, Parallels, Meridians)."""
        return self._intensities

    def get_intensity_frame(self, frame_idx):
        """
        Returns the screen intensities for a specific time step.
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of bounds (0 to {self.num_frames-1}).")
        return self._intensities[frame_idx]

    def get_coordinates(self):
        """
        Returns the underlying sphere coordinates as a tuple: (elevation_grid, azimuth_grid).
        Useful for mapping receptive field centers.
        """
        return self.el_grid, self.az_grid

    def get_sphere_geometry(self):
        """Return ``(X, Y, Z)`` arrays for plotting the sphere as a surface.

        Returned arrays are NumPy regardless of backend (CuPy / MLX / NumPy)
        and shaped ``(parallels, meridians)`` matching ``intensities[t]``.

        Cached on first call (sphere geometry is static across time).
        """
        if not hasattr(self, '_sphere_xyz_cache'):
            el_grid, az_grid = self.get_coordinates()
            el_np = to_numpy(el_grid)
            az_np = to_numpy(az_grid)
            el_rad = np.deg2rad(el_np)
            az_rad = np.deg2rad(az_np)
            r = float(self.radius)
            X = r * np.cos(el_rad) * np.cos(az_rad)
            Y = r * np.cos(el_rad) * np.sin(az_rad)
            Z = r * np.sin(el_rad)
            self._sphere_xyz_cache = (X, Y, Z)
        return self._sphere_xyz_cache

    def release_device_memory(self):
        """
        Drop the large intensity tensor and interpolation map from the active
        backend. Call after downstream consumers have copied what they need;
        subsequent calls to ``intensities`` / ``get_intensity_frame`` will fail.
        """
        for attr in ('_intensities', 'map_x', 'map_y', 'el_grid', 'az_grid'):
            if hasattr(self, attr):
                delattr(self, attr)
        free_memory()
