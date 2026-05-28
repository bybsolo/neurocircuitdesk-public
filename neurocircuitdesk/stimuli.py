"""
neurocircuitdesk.stimuli
------------------------
Stimulus generators for NCD / FES input pipelines.

Every generator returns a ``(T, H, W)`` ``float32`` array in the
``[0, 1]`` range. Downstream input processing (FES ``Screen`` or
``HexActiveSample*``) handles photon-rate scaling — these generators
stay luminance-normalised so they're composable.

Shipped generators (all functional, no class state):

  - :func:`looming_dot`           — dark dot expanding over a background
  - :func:`saccade_scan`          — slide a window over a wider background
  - :func:`drifting_grating`      — sinusoidal grating drifting at an angle
  - :func:`flicker`               — whole-field temporal modulation
  - :func:`fixational_drift`      — static scene with small camera jitter
  - :func:`natural_scene_background` — load + crop an image file as a
    background

Pattern of use::

    from neurocircuitdesk.stimuli import looming_dot
    from neurocircuitdesk import VideoInput

    video = looming_dot(T=150, H=256, W=256)        # (T, H, W) float32 in [0, 1]
    VideoInput(video).save_h5('outputs/looming.h5') # standardised input H5

The saved H5 is then consumed by the ``input_generation`` skill.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    'looming_dot',
    'saccade_scan',
    'drifting_grating',
    'flicker',
    'fixational_drift',
    'natural_scene_background',
]


# ─────────────────────────────────────────────────────────────────────────
# Looming dot
# ─────────────────────────────────────────────────────────────────────────

def looming_dot(
    T: int = 150,
    H: int = 256,
    W: int = 256,
    *,
    bg: Optional[np.ndarray] = None,
    bg_value: float = 0.5,
    r_start: float = 2.0,
    r_end: float = 60.0,
    center: Optional[Tuple[int, int]] = None,
    dot_value: float = 0.05,
    interp: str = 'linear',
) -> np.ndarray:
    """A dark dot expanding over a background.

    Parameters
    ----------
    T, H, W : int
        Frame count + spatial size.
    bg : (H, W) array, optional
        Background image in ``[0, 1]``. If ``None``, a uniform field at
        ``bg_value`` is used.
    bg_value : float
        Value of the uniform background when ``bg`` is ``None``.
    r_start, r_end : float
        Dot radius in pixels at ``t=0`` and ``t=T-1``.
    center : (row, col) tuple, optional
        Dot centre in pixel coordinates. Defaults to ``(H//2, W//2)``.
    dot_value : float
        Multiplicative factor applied to the background inside the dot.
        ``0.0`` = pure black, ``1.0`` = invisible. Default ``0.05``.
    interp : {'linear', 'exp'}
        Radius growth profile. ``'linear'`` interpolates uniformly
        between ``r_start`` and ``r_end``. ``'exp'`` grows
        exponentially (closer to a constant-velocity approaching
        object).

    Returns
    -------
    (T, H, W) float32 in ``[0, 1]``.
    """
    if bg is None:
        bg = np.full((H, W), bg_value, dtype=np.float32)
    else:
        bg = np.asarray(bg, dtype=np.float32)
        if bg.shape != (H, W):
            raise ValueError(f"bg shape {bg.shape} != (H, W) = ({H}, {W}).")

    cy, cx = center if center is not None else (H // 2, W // 2)

    if interp == 'linear':
        r_t = np.linspace(r_start, r_end, T, dtype=np.float32)
    elif interp == 'exp':
        # exponential growth from r_start to r_end
        if r_end <= r_start:
            raise ValueError("interp='exp' requires r_end > r_start.")
        log_ratio = np.log(r_end / r_start)
        r_t = r_start * np.exp(np.linspace(0, log_ratio, T, dtype=np.float32))
    else:
        raise ValueError(f"interp must be 'linear' or 'exp'; got {interp!r}.")

    yy, xx = np.ogrid[:H, :W]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2

    video = np.empty((T, H, W), dtype=np.float32)
    for t in range(T):
        mask = d2 <= r_t[t] ** 2
        frame = bg.copy()
        frame[mask] *= dot_value
        video[t] = frame
    return np.clip(video, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────
# Saccade scan
# ─────────────────────────────────────────────────────────────────────────

def saccade_scan(
    bg_img: np.ndarray,
    *,
    T: int = 300,
    H: int = 256,
    W: int = 256,
    speed: float = 4.0,
    direction: str = 'right',
    y_start: Optional[int] = None,
    x_start: Optional[int] = None,
) -> np.ndarray:
    """Slide an ``(H, W)`` window across a wider background image.

    Mirrors ``demo_fes.ipynb::vid_gen_saccade``. Pure spatial
    translation — no foreground dots.

    Parameters
    ----------
    bg_img : (H_bg, W_bg) array
        Background image in ``[0, 1]``. Should be larger than ``(H, W)``
        in the scan direction.
    T, H, W : int
        Frame count + viewport size.
    speed : float
        Pixels per frame.
    direction : {'right', 'left', 'up', 'down'}
        Direction of the saccade.
    y_start, x_start : int, optional
        Initial top-left corner of the viewport. Defaults pick a
        sensible starting position based on the direction.

    Returns
    -------
    (T, H, W) float32 in ``[0, 1]``.
    """
    bg_img = np.asarray(bg_img, dtype=np.float32)
    bg_h, bg_w = bg_img.shape

    if direction not in ('right', 'left', 'up', 'down'):
        raise ValueError(
            f"direction must be 'right'/'left'/'up'/'down'; got {direction!r}.")

    vertical = direction in ('up', 'down')
    going_negative = direction in ('left', 'up')

    if y_start is None:
        y_start = (bg_h - H) // 2 if not vertical else (bg_h - H if going_negative else 0)
    if x_start is None:
        x_start = (bg_w - W) // 2 if vertical else (bg_w - W if going_negative else 0)

    sign = -1 if going_negative else 1
    video = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        off = int(round(t * speed * sign))
        y0 = y_start + (off if vertical else 0)
        x0 = x_start + (off if not vertical else 0)

        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(bg_h, y0 + H), min(bg_w, x0 + W)
        if sy1 > sy0 and sx1 > sx0:
            dy0, dx0 = sy0 - y0, sx0 - x0
            video[t, dy0:dy0 + (sy1 - sy0),
                    dx0:dx0 + (sx1 - sx0)] = bg_img[sy0:sy1, sx0:sx1]
    return video


# ─────────────────────────────────────────────────────────────────────────
# Drifting grating
# ─────────────────────────────────────────────────────────────────────────

def drifting_grating(
    T: int = 100,
    H: int = 256,
    W: int = 256,
    *,
    spatial_period: float = 32.0,
    temporal_freq: float = 2.0,
    fps: float = 60.0,
    direction_deg: float = 0.0,
    contrast: float = 1.0,
    mean: float = 0.5,
) -> np.ndarray:
    """Sinusoidal grating drifting at ``direction_deg``.

    Parameters
    ----------
    T, H, W : int
        Frame count + spatial size.
    spatial_period : float
        Pixels per spatial cycle.
    temporal_freq : float
        Cycles per second (Hz).
    fps : float
        Frame rate (used together with ``temporal_freq`` to set the
        per-frame phase advance).
    direction_deg : float
        Drift direction in degrees. ``0`` = rightward,
        ``90`` = upward, ``180`` = leftward, ``-90`` / ``270`` = downward.
    contrast : float
        Peak-to-mean amplitude. ``1.0`` gives ``[mean - 1, mean + 1]``
        modulation before clipping; the result is clipped to
        ``[0, 1]``.
    mean : float
        DC offset (background luminance).

    Returns
    -------
    (T, H, W) float32 in ``[0, 1]``.
    """
    if spatial_period <= 0:
        raise ValueError("spatial_period must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")

    theta = np.deg2rad(direction_deg)
    kx = np.cos(theta) / spatial_period   # cycles per pixel along x
    ky = -np.sin(theta) / spatial_period  # negative because rows grow downward

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    spatial_phase = 2.0 * np.pi * (kx * xx + ky * yy)   # (H, W)

    dt = 1.0 / fps
    temporal_phase = 2.0 * np.pi * temporal_freq * dt * np.arange(T, dtype=np.float32)

    video = np.empty((T, H, W), dtype=np.float32)
    for t in range(T):
        video[t] = mean + contrast * 0.5 * np.sin(spatial_phase - temporal_phase[t])
    return np.clip(video, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────
# Whole-field flicker
# ─────────────────────────────────────────────────────────────────────────

def flicker(
    T: int = 100,
    H: int = 256,
    W: int = 256,
    *,
    frequency: float = 4.0,
    fps: float = 60.0,
    contrast: float = 1.0,
    mean: float = 0.5,
    waveform: str = 'sine',
) -> np.ndarray:
    """Whole-field temporal modulation — uniform brightness over space.

    Parameters
    ----------
    T, H, W : int
        Frame count + spatial size.
    frequency : float
        Hz.
    fps : float
        Frame rate.
    contrast : float
        Peak-to-mean amplitude. Clipped to ``[0, 1]`` after offset.
    mean : float
        DC offset.
    waveform : {'sine', 'square'}
        Temporal waveform.

    Returns
    -------
    (T, H, W) float32 in ``[0, 1]`` — every frame is spatially uniform.
    """
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if waveform not in ('sine', 'square'):
        raise ValueError(f"waveform must be 'sine' or 'square'; got {waveform!r}.")

    t = np.arange(T, dtype=np.float32) / fps
    phase = 2.0 * np.pi * frequency * t
    if waveform == 'sine':
        signal = mean + contrast * 0.5 * np.sin(phase)
    else:
        signal = mean + contrast * 0.5 * np.sign(np.sin(phase))
    signal = np.clip(signal, 0.0, 1.0).astype(np.float32)

    return np.broadcast_to(signal[:, None, None], (T, H, W)).copy()


# ─────────────────────────────────────────────────────────────────────────
# Fixational drift
# ─────────────────────────────────────────────────────────────────────────

def fixational_drift(
    bg_img: Optional[np.ndarray] = None,
    *,
    T: int = 300,
    H: int = 256,
    W: int = 256,
    sigma: float = 1.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Static scene with small Brownian camera jitter.

    Parameters
    ----------
    bg_img : (H_bg, W_bg) array, optional
        Background image in ``[0, 1]``. If ``None``, uses a random-noise
        background (Gaussian, mean 0.5, std 0.15, clipped).
    T, H, W : int
        Frame count + viewport size.
    sigma : float
        Standard deviation of the per-step pixel jitter (Brownian
        increment). Total drift after ``T`` steps is roughly
        ``sigma * sqrt(T)`` pixels.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    (T, H, W) float32 in ``[0, 1]``.
    """
    rng = np.random.default_rng(seed)

    if bg_img is None:
        bg = np.clip(rng.normal(0.5, 0.15, (max(H * 2, 256), max(W * 2, 256))),
                     0.0, 1.0).astype(np.float32)
    else:
        bg = np.asarray(bg_img, dtype=np.float32)
    bg_h, bg_w = bg.shape

    # Brownian walk centred at the middle of the background
    cy0 = (bg_h - H) // 2
    cx0 = (bg_w - W) // 2
    dy = np.cumsum(rng.normal(0.0, sigma, T)).astype(np.float32)
    dx = np.cumsum(rng.normal(0.0, sigma, T)).astype(np.float32)

    video = np.empty((T, H, W), dtype=np.float32)
    for t in range(T):
        y0 = int(round(cy0 + dy[t]))
        x0 = int(round(cx0 + dx[t]))
        y0 = max(0, min(bg_h - H, y0))
        x0 = max(0, min(bg_w - W, x0))
        video[t] = bg[y0:y0 + H, x0:x0 + W]
    return video


# ─────────────────────────────────────────────────────────────────────────
# Background loader
# ─────────────────────────────────────────────────────────────────────────

def natural_scene_background(
    path: str,
    *,
    target_shape: Optional[Tuple[int, int]] = None,
    crop_origin: str = 'centre',
    mat_key: str = 'im',
    normalize: bool = True,
) -> np.ndarray:
    """Load an image (PNG / JPG / TIFF / MAT) as a 2-D ``(H, W)`` array
    in ``[0, 1]``.

    Useful as the ``bg`` / ``bg_img`` argument to other generators.

    Parameters
    ----------
    path : str or Path
        Image path. ``.mat`` files load via ``scipy.io.loadmat`` and
        pull the dataset named ``mat_key`` (default ``'im'`` — matches
        the shipped ``image1.mat``).
    target_shape : (h, w), optional
        If given, the loaded image is cropped to this shape via
        ``crop_origin``. If the source is smaller than ``target_shape``,
        ``ValueError`` is raised.
    crop_origin : {'centre', 'top-left'}
        How to anchor the crop when ``target_shape`` is set.
    mat_key : str
        Dataset key inside a ``.mat`` file.
    normalize : bool
        Rescale to ``[0, 1]`` before returning.
    """
    path = str(path)
    if path.lower().endswith('.mat'):
        import scipy.io
        mat = scipy.io.loadmat(path)
        arr = np.asarray(mat[mat_key], dtype=np.float32)
    else:
        from PIL import Image
        arr = np.asarray(Image.open(path).convert('L'), dtype=np.float32)

    if arr.ndim != 2:
        # Drop trailing channel dim if present
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = arr[..., :3].mean(axis=-1)
        else:
            raise ValueError(f"Loaded image has unsupported shape {arr.shape}.")

    if target_shape is not None:
        h, w = target_shape
        bg_h, bg_w = arr.shape
        if h > bg_h or w > bg_w:
            raise ValueError(
                f"target_shape {target_shape} exceeds source {arr.shape}.")
        if crop_origin == 'centre':
            y0 = (bg_h - h) // 2
            x0 = (bg_w - w) // 2
        elif crop_origin == 'top-left':
            y0 = 0
            x0 = 0
        else:
            raise ValueError(f"crop_origin must be 'centre' or 'top-left'.")
        arr = arr[y0:y0 + h, x0:x0 + w]

    if normalize and arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr.astype(np.float32)
