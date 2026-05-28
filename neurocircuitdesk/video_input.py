"""
neurocircuitdesk.video_input
----------------------------
``VideoInput`` — universal video loader for NCD / FES input pipelines.

Accepts:
  - HDF5 files (``.h5`` / ``.hdf5``) with a 3-D dataset
  - Standard video files (``.mp4`` / ``.avi`` / ``.mov`` / …) via imageio
  - A directory of image frames (PNG / JPG / TIF)
  - A raw numpy array

Always produces a ``(T, H, W)`` ``float32`` monochrome array internally.
Provides inspection, single-frame visualisation with optional ROI
overlays, and chainable transforms (``crop``, ``resize``, ``grayscale``,
``normalize``). Standardised ``save_h5`` / ``load_h5`` for handoff to
downstream sampling skills (``HexActiveSample*`` or
``flyeyesimulator.Screen``).

Typical usage::

    from neurocircuitdesk import VideoInput

    vi = VideoInput('raw.mp4').grayscale().normalize()
    vi.info()                                       # → "VideoInput(T=300, H=480, W=720, ...)"
    vi.show_frame(t=0)                              # matplotlib preview
    vi.show_roi((140, 220, 256, 256), t=0)          # preview with crop box

    vi_crop = vi.crop((140, 220, 256, 256))         # apply crop
    vi_crop.save_h5('outputs/stim.h5')              # standardised (T, H, W)

    # Later — feed into a sampler
    video_T = VideoInput.load_h5('outputs/stim.h5').video
    screen  = Screen(video_T, radius=10, parallels=256, meridians=256, dt=10)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ['VideoInput']

ROI = Tuple[int, int, int, int]   # (y0, x0, h, w)
PathLike = Union[str, os.PathLike]

# File-extension hints.
_H5_EXT     = {'.h5', '.hdf5'}
_VIDEO_EXT  = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif'}
_IMAGE_EXT  = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


class VideoInput:
    """Universal video loader for NCD / FES pipelines.

    Parameters
    ----------
    source : str | Path | np.ndarray
        - Path to ``.h5`` / video file / image directory.
        - Or a numpy array ``(T, H, W)`` or ``(T, H, W, C)`` already in
          memory.
    source_type : {'h5', 'video', 'frames_dir', 'array'}, optional
        Force-select the loader. Auto-detected from path extension if
        ``None``.
    dataset : str, default 'video'
        HDF5 dataset name (only used for the ``'h5'`` source type).
    grayscale_mode : {'mean', 'luma'}, default 'mean'
        How to collapse RGB → mono on load. ``'mean'`` averages
        channels equally; ``'luma'`` applies ITU-R BT.601 weights
        (``0.299·R + 0.587·G + 0.114·B``).
    fps : float, optional
        Source frame rate. Stored on the instance and propagated to
        ``save_h5`` (as a dataset attribute). Auto-detected from video
        files when possible; otherwise ``None``.

    Attributes
    ----------
    video : np.ndarray of shape (T, H, W) float32
        The loaded video, always monochrome.
    source : object
        The original source argument (path or array) for provenance.
    source_type : str
        Resolved source type.
    fps : float | None
        Frame rate metadata.
    """

    def __init__(
        self,
        source,
        *,
        source_type: Optional[str] = None,
        dataset: str = 'video',
        grayscale_mode: str = 'mean',
        fps: Optional[float] = None,
    ):
        if grayscale_mode not in ('mean', 'luma'):
            raise ValueError(f"grayscale_mode must be 'mean' or 'luma'; got {grayscale_mode!r}")

        self.source = source
        self.fps = fps
        self._dataset = dataset
        self._grayscale_mode = grayscale_mode

        if source_type is None:
            source_type = self._detect_source_type(source)
        self.source_type = source_type

        arr = self._load(source, source_type, dataset)
        self._video = self._coerce_mono(arr, grayscale_mode).astype(np.float32, copy=False)

    # ─── detection / loading ─────────────────────────────────────────────

    @staticmethod
    def _detect_source_type(source) -> str:
        if isinstance(source, np.ndarray):
            return 'array'
        p = Path(source)
        if p.is_dir():
            return 'frames_dir'
        ext = p.suffix.lower()
        if ext in _H5_EXT:
            return 'h5'
        if ext in _VIDEO_EXT:
            return 'video'
        raise ValueError(
            f"Cannot detect source type from {source!r}. "
            f"Pass source_type= explicitly. "
            f"Recognised extensions: {sorted(_H5_EXT | _VIDEO_EXT)}, or pass a directory.")

    def _load(self, source, source_type: str, dataset: str) -> np.ndarray:
        if source_type == 'array':
            return np.asarray(source)
        if source_type == 'h5':
            import h5py
            with h5py.File(source, 'r') as f:
                if dataset not in f:
                    avail = list(f.keys())
                    raise KeyError(
                        f"HDF5 file {source!r} has no dataset {dataset!r}. "
                        f"Available: {avail}")
                arr = f[dataset][:]
                # Pick up fps attribute if present and not user-overridden.
                if self.fps is None and 'fps' in f[dataset].attrs:
                    self.fps = float(f[dataset].attrs['fps'])
            return arr
        if source_type == 'video':
            try:
                import imageio.v3 as iio
            except ImportError:
                import imageio as iio                                  # legacy
            try:
                # imageio.v3 path
                arr = iio.imread(source, plugin='pyav')
            except Exception:
                # legacy path: mimread returns list of frames
                arr = np.stack(iio.mimread(source, memtest=False), axis=0)
            return np.asarray(arr)
        if source_type == 'frames_dir':
            return self._load_frames_dir(Path(source))
        raise ValueError(f"unknown source_type {source_type!r}")

    @staticmethod
    def _load_frames_dir(d: Path) -> np.ndarray:
        from PIL import Image
        frames = sorted(p for p in d.iterdir()
                        if p.suffix.lower() in _IMAGE_EXT)
        if not frames:
            raise ValueError(
                f"No image frames found in {d!r}. "
                f"Looked for extensions: {sorted(_IMAGE_EXT)}")
        stack = [np.asarray(Image.open(p)) for p in frames]
        return np.stack(stack, axis=0)

    @staticmethod
    def _coerce_mono(arr: np.ndarray, mode: str) -> np.ndarray:
        if arr.ndim == 3:
            return arr                                  # (T, H, W) already mono
        if arr.ndim == 4 and arr.shape[-1] in (3, 4):
            rgb = arr[..., :3]                          # drop alpha if present
            if mode == 'luma':
                w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
                return (rgb.astype(np.float32) * w).sum(axis=-1)
            return rgb.astype(np.float32).mean(axis=-1)
        raise ValueError(
            f"Unsupported video shape {arr.shape}. "
            f"Need (T, H, W) or (T, H, W, 3/4).")

    # ─── properties ──────────────────────────────────────────────────────

    @property
    def video(self) -> np.ndarray:
        return self._video

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self._video.shape)  # type: ignore[return-value]

    @property
    def T(self) -> int: return int(self._video.shape[0])

    @property
    def H(self) -> int: return int(self._video.shape[1])

    @property
    def W(self) -> int: return int(self._video.shape[2])

    @property
    def dtype(self):
        return self._video.dtype

    def info(self) -> str:
        """One-line text summary suitable for printing."""
        vmin, vmax = float(self._video.min()), float(self._video.max())
        src = (self.source if not isinstance(self.source, np.ndarray)
               else f'<ndarray {self.source.shape}>')
        fps_str = f", fps={self.fps:g}" if self.fps else ""
        return (f"VideoInput(T={self.T}, H={self.H}, W={self.W}, "
                f"dtype={self._video.dtype}, range=[{vmin:.3g}, {vmax:.3g}]"
                f"{fps_str}, source_type={self.source_type!r}, source={src!r})")

    def __repr__(self) -> str:
        return self.info()

    # ─── inspection / visualisation ──────────────────────────────────────

    def show_frame(
        self,
        t: int = 0,
        *,
        ax=None,
        title: Optional[str] = None,
        cmap: str = 'gray',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Render frame ``t`` as a matplotlib image. Returns the Axes."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self._video[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title or f"t = {t}")
        ax.set_xticks([]); ax.set_yticks([])
        return ax

    def show_roi(
        self,
        roi: ROI,
        t: int = 0,
        *,
        ax=None,
        title: Optional[str] = None,
        edge_color: str = 'red',
        edge_width: float = 2.0,
        fill: bool = False,
        cmap: str = 'gray',
    ):
        """Render frame ``t`` with the ROI rectangle overlaid.

        ``roi`` is ``(y0, x0, h, w)`` — origin is top-left pixel.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        y0, x0, h, w = self._validate_roi(roi)
        ax = self.show_frame(t=t, ax=ax, title=title, cmap=cmap)
        rect = Rectangle((x0 - 0.5, y0 - 0.5), w, h,
                         linewidth=edge_width, edgecolor=edge_color,
                         facecolor=edge_color if fill else 'none',
                         alpha=0.3 if fill else 1.0)
        ax.add_patch(rect)
        return ax

    def show_grid(
        self,
        ts: Sequence[int],
        *,
        cols: int = 4,
        cmap: str = 'gray',
        figsize_per: Tuple[float, float] = (3, 3),
        title: Optional[str] = None,
    ):
        """Render multiple frames in a grid."""
        import matplotlib.pyplot as plt
        ts = list(ts)
        rows = (len(ts) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * figsize_per[0], rows * figsize_per[1]))
        axes = np.asarray(axes).reshape(-1)
        for ax, t in zip(axes, ts):
            ax.imshow(self._video[t], cmap=cmap)
            ax.set_title(f"t = {t}")
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes[len(ts):]:
            ax.axis('off')
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        return fig

    # ─── transforms (chainable; return new VideoInput) ───────────────────

    def crop(self, roi: ROI) -> 'VideoInput':
        """Return a new VideoInput cropped to ``roi=(y0, x0, h, w)``."""
        y0, x0, h, w = self._validate_roi(roi)
        return self._spawn(self._video[:, y0:y0 + h, x0:x0 + w])

    def resize(self, h: int, w: int) -> 'VideoInput':
        """Return a new VideoInput resized to ``(h, w)`` via bilinear."""
        try:
            from skimage.transform import resize as _sk_resize
        except ImportError as e:
            raise ImportError(
                "VideoInput.resize requires scikit-image. "
                "Install with `pip install scikit-image`.") from e
        out = np.empty((self.T, h, w), dtype=np.float32)
        for t in range(self.T):
            out[t] = _sk_resize(self._video[t], (h, w),
                                anti_aliasing=True, preserve_range=True)
        return self._spawn(out)

    def grayscale(self) -> 'VideoInput':
        """Idempotent — VideoInput is always mono. Returns ``self``."""
        return self

    def normalize(self, target: Tuple[float, float] = (0.0, 1.0)) -> 'VideoInput':
        """Return a new VideoInput linearly scaled to ``target=(lo, hi)``."""
        lo, hi = target
        vmin = float(self._video.min())
        vmax = float(self._video.max())
        if vmax <= vmin:
            return self._spawn(np.full_like(self._video, lo, dtype=np.float32))
        out = (self._video - vmin) / (vmax - vmin) * (hi - lo) + lo
        return self._spawn(out.astype(np.float32, copy=False))

    def _spawn(self, video: np.ndarray) -> 'VideoInput':
        """Create a new VideoInput sharing metadata with self."""
        new = object.__new__(VideoInput)
        new.source = self.source
        new.source_type = self.source_type
        new.fps = self.fps
        new._dataset = self._dataset
        new._grayscale_mode = self._grayscale_mode
        new._video = video
        return new

    # ─── persistence ─────────────────────────────────────────────────────

    def save_h5(
        self,
        path: PathLike,
        *,
        dataset: str = 'video',
        dtype=np.float32,
        compression: Optional[str] = 'gzip',
    ) -> str:
        """Save as standardised ``(T, H, W)`` HDF5 dataset.

        Returns the saved path as a string.
        """
        import h5py
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, 'w') as f:
            ds = f.create_dataset(dataset, data=self._video.astype(dtype),
                                  compression=compression)
            ds.attrs['T'] = self.T
            ds.attrs['H'] = self.H
            ds.attrs['W'] = self.W
            if self.fps is not None:
                ds.attrs['fps'] = float(self.fps)
            ds.attrs['source_type'] = str(self.source_type)
        return path

    @classmethod
    def load_h5(cls, path: PathLike, *, dataset: str = 'video') -> 'VideoInput':
        """Equivalent to ``VideoInput(path, source_type='h5', dataset=dataset)``."""
        return cls(str(path), source_type='h5', dataset=dataset)

    # ─── internals ───────────────────────────────────────────────────────

    def _validate_roi(self, roi: ROI) -> Tuple[int, int, int, int]:
        if len(roi) != 4:
            raise ValueError(f"roi must be (y0, x0, h, w); got {roi!r}")
        y0, x0, h, w = (int(v) for v in roi)
        if not (0 <= y0 <= self.H and 0 <= x0 <= self.W):
            raise ValueError(
                f"roi origin ({y0}, {x0}) outside frame "
                f"({self.H}, {self.W}).")
        if h <= 0 or w <= 0:
            raise ValueError(f"roi h={h}, w={w} must be positive.")
        if y0 + h > self.H or x0 + w > self.W:
            raise ValueError(
                f"roi ({y0},{x0},{h},{w}) exceeds frame "
                f"({self.H},{self.W}).")
        return y0, x0, h, w
