"""
demo_active_sampling.py
-----------------------
Demonstrates the flyeyesimulator optical pipeline: construct a Retina and
Screen, then run both the static and active (physics-driven) simulators to
produce per-photoreceptor photon-rate time series (lambda).

The demo:
  1. Loads a natural image (image1.mat) and synthesises a moving-dot video
  2. Projects the video onto a spherical Screen
  3. Builds a biological Retina (890 columns, 7566 R1-R6 receptive fields)
  4. Runs FlyEyeSimulatorStatic  -> fixed receptive fields
  5. Runs FlyEyeSimulatorActive  -> spring-damper dynamic receptive fields
  6. Compares the two outputs

Usage:
    python demo_active_sampling.py

Requires:
    data/image1.mat  (natural image, shipped with this demo)

Outputs (written to outputs/active_sampling/):
    static_lambda.npy    -- (T, 6*N_cols) static photon rates
    active_lambda.npy    -- (T, 6*N_cols) active photon rates
    comparison.png       -- side-by-side R1 response for selected columns
"""

import os
import sys
import time

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from flyeyesimulator import (
    Retina, Screen,
    FlyEyeSimulatorStatic, FlyEyeSimulatorActive,
    xp, to_numpy,
)

COL_JSON  = os.path.join(_REPO_ROOT, 'neurocircuitdesk', 'libs', 'jsons',
                         'hexcol_l1m3_new_578.json')
MAT_PATH  = os.path.join(_THIS_DIR, 'data', 'image1.mat')
OUTDIR    = os.path.join(_THIS_DIR, 'outputs', 'active_sampling')


# ── Synthetic video generation ─────────────────────────────────────────────

def make_looming_dot_video(bg_img: np.ndarray, T: int = 150,
                           H: int = 256, W: int = 256) -> np.ndarray:
    """Generate a (T, H, W) video of a dark dot expanding over a natural image.

    The dot starts small at the center and grows linearly, simulating a
    looming stimulus approaching the observer.
    """
    # Crop / resize background to (H, W)
    from PIL import Image
    bg = Image.fromarray((bg_img * 255).astype(np.uint8))
    bg = bg.resize((W, H), Image.BILINEAR)
    bg = np.array(bg, dtype=np.float32) / 255.0

    video = np.empty((T, H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2

    for t in range(T):
        frame = bg.copy()
        # Dot radius grows from 2 to 60 pixels
        r = 2.0 + (58.0 * t / T)
        yy, xx = np.ogrid[:H, :W]
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
        frame[mask] *= 0.05  # dark dot
        video[t] = frame

    return video


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("== Active sampling demo ==\n")

    # 1. Load natural image
    print("Loading background image ...")
    mat = scipy.io.loadmat(MAT_PATH)
    img = np.array(mat['im'], dtype=np.float64)[325:725]   # crop to ~400x1536
    img = (img - img.min()) / (img.max() - img.min())       # normalise to [0, 1]

    # 2. Synthesise looming-dot video
    T, H, W = 150, 256, 256
    print(f"Generating looming-dot video ({T} frames, {H}x{W}) ...")
    video = make_looming_dot_video(img, T=T, H=H, W=W)

    # Scale to photon rates (order 1e5)
    photon_video = 3e5 * video / video.max()

    # 3. Build Retina (biological columns enabled)
    print("Building retina ...")
    retina = Retina(
        num_rings=20,
        inter_ommatidia_angle_deg=4.4,
        bio_cols_only=True,
        col_json_path=COL_JSON,
    )
    n_vrfs = len(retina.vrfs)
    print(f"  Retina: {n_vrfs} receptive fields "
          f"({n_vrfs // 6} columns x 6 R-cells)")

    # 4. Build Screens (one per simulator -- run() releases device memory)
    photon_arr = xp.asarray(photon_video)
    print("Building spherical screens ...")
    screen_static = Screen(photon_arr, radius=10, parallels=H, meridians=W, dt=10)
    screen_active = Screen(photon_arr, radius=10, parallels=H, meridians=W, dt=10)
    print(f"  Screen: {screen_static.num_frames} frames, "
          f"{screen_static.parallels}x{screen_static.meridians} grid")

    # 5. Run static simulator
    print("\nRunning static simulator ...")
    fes_static = FlyEyeSimulatorStatic(
        screen=screen_static, retina=retina, acceptance_angle_deg=4.4)
    t0 = time.perf_counter()
    res_static = fes_static.run()
    static_time = time.perf_counter() - t0
    res_static_np = to_numpy(res_static)
    print(f"  Static: {static_time:.2f}s, output shape {res_static_np.shape}")

    # 6. Run active simulator (spring-damper physics)
    print("\nRunning active simulator ...")
    fes_active = FlyEyeSimulatorActive(
        screen=screen_active, retina=retina,
        initial_acceptance_angle_deg=4.4,
        shift_ratio=1, shrink_ratio=0.5)
    t0 = time.perf_counter()
    res_active = fes_active.run()
    active_time = time.perf_counter() - t0
    res_active_np = to_numpy(res_active)
    print(f"  Active: {active_time:.2f}s, output shape {res_active_np.shape}")

    # 7. Save outputs
    np.save(os.path.join(OUTDIR, 'static_lambda.npy'), res_static_np)
    np.save(os.path.join(OUTDIR, 'active_lambda.npy'), res_active_np)

    # 8. Plot comparison: R1 channel (every 6th column) for selected columns
    print("\nPlotting comparison ...")
    # R1 is every 6th entry starting at 0: indices 0, 6, 12, ...
    n_cols = n_vrfs // 6
    sample_cols = [0, n_cols // 4, n_cols // 2, 3 * n_cols // 4]

    fig, axes = plt.subplots(len(sample_cols), 1, figsize=(10, 3 * len(sample_cols)),
                             sharex=True)
    for ax, col in zip(axes, sample_cols):
        r1_idx = col * 6  # R1 channel for this column
        ax.plot(res_static_np[:, r1_idx], label='Static', alpha=0.8)
        ax.plot(res_active_np[:, r1_idx], label='Active', alpha=0.8)
        ax.set_ylabel(f'Col {col} R1')
        ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Frame')
    fig.suptitle('Static vs Active Sampling: R1 photon rates (lambda)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'comparison.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {os.path.join(OUTDIR, 'comparison.png')}")

    # Summary
    print(f"\nStatic vs Active max absolute difference: "
          f"{np.abs(res_static_np - res_active_np).max():.2f}")
    print(f"All outputs written to {OUTDIR}/")


if __name__ == '__main__':
    main()
