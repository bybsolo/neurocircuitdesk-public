# FlyEyeSimulator (FES)

FES projects a 2D video onto a spherical screen and samples it through
a Drosophila-style compound eye — a hex grid of R1–R6 receptive fields
with neural superposition. The output is a `(T, 6 · N_cols)` array of
per-photoreceptor **photon-rate** time series. For NCD circuits that
expect voltage, pipe it through the `phototransduction` module
(§9) which adds the RPM phototransduction + HH(NS) membrane model
and aggregates the 6 superposed R-cells into a single per-column
voltage trace.

Two simulator flavours:

- **Static** — fixed RFs through the run. Cheapest baseline.
- **Active** — every step the per-VRF acceptance angle and elevation
  are updated by a spring-damper driven by the cell's own luminance.
  Models pupil-style angular dynamics.

The demos under `docs/demos/` are the canonical worked examples —
this doc orients you to the API surface and points you at the right
demo for each task.

## 1. The pipeline

```
        video (T, H, W)
              │
              ▼
   ┌──────────────────┐
   │ Screen           │  hemisphere projection
   │   (parallels,    │  ─ Albers map, GPU-friendly
   │    meridians)    │
   └────────┬─────────┘
            │
            │   intensities (T, P, M)
            │
   ┌────────┴─────────┐    ┌──────────────────┐
   │ FlyEyeSimulator  │◄───┤ Retina           │  hex grid, R1–R6
   │   Static         │    │   (num_rings)    │  neural superposition
   │   or Active      │    └──────────────────┘  optional bio mask
   └────────┬─────────┘
            │
            ▼
       photon rates
       (T, 6·N_cols)
```

Each stage is one class:

| Class | Constructor |
|---|---|
| `Retina` | `Retina(num_rings, inter_ommatidia_angle_deg=4.4, bio_cols_only=False, col_json_path=None)` |
| `Screen` | `Screen(video_T, radius=10, parallels, meridians, dt=10)` |
| `FlyEyeSimulatorStatic` | `FlyEyeSimulatorStatic(screen, retina, acceptance_angle_deg=5.0, M=1.0)` |
| `FlyEyeSimulatorActive` | `FlyEyeSimulatorActive(screen, retina, initial_acceptance_angle_deg=5.0, M=1.0, shrink_ratio=0.1, shift_ratio=1)` |

Both simulators expose `.run(release=True)` → `(T, 6 · N_cols)` array.
With `release=True` (default) you get a host NumPy array and
device-side buffers are freed; with `release=False` the result stays
on the backend so you can chain another run.

`FlyEyeSimulatorActive` also has `.run_more(release=True)` →
`(responses, acceptance_angles_deg, elevations_deg)` if you want to
inspect the RF state trajectory in addition to the photon rates.

## 2. Minimum end-to-end

```python
from flyeyesimulator import Retina, Screen, FlyEyeSimulatorStatic, xp

retina = Retina(num_rings=20, inter_ommatidia_angle_deg=4.4,
                bio_cols_only=True, col_json_path=COL_JSON)

photon_video = 3e5 * video_T / video_T.max()           # ~photon-rate scale
photon_arr   = xp.asarray(photon_video)                # to active backend
screen       = Screen(photon_arr, radius=10,
                      parallels=H, meridians=W, dt=10)

fes = FlyEyeSimulatorStatic(screen=screen, retina=retina,
                            acceptance_angle_deg=4.4)
res = fes.run()                                        # (T, 6·N_cols), numpy
```

Three things that matter:

1. **Scale the video to photon rates** before building the `Screen`.
   The active simulator's physics is tuned for sums in the ~10⁵
   range. The reference scale is `3e5 * video / video.max()`.
2. **Push to the active backend** with `xp.asarray(...)`. `xp` is
   CuPy on CUDA boxes, MLX on Apple Silicon, NumPy elsewhere — see
   §6.
3. **`bio_cols_only=True`** with the shipped column JSON
   (`neurocircuitdesk/libs/jsons/hexcol_l1m3_new_578.json`) masks the
   hex spiral to the biological column set (~890 active out of 1261
   positions). Required if you plan to feed the output into an NCD
   canvas that was also built with the same column set.

For the full walkthrough — video synthesis, geometry visualisation,
output animation — see [`docs/demos/demo_fes.ipynb`](demos/demo_fes.ipynb).

## 3. Output shape — what `(T, 6·N_cols)` means

The output is laid out as: column 0 R1, column 0 R2, …, column 0 R6,
column 1 R1, …. So for any column `c`:

```python
r1_c = res[:, c * 6 + 0]      # R1 trace for column c
r2_c = res[:, c * 6 + 1]      # R2 trace
...
r6_c = res[:, c * 6 + 5]      # R6 trace
```

Vectorised slicing:

```python
all_r1   = res[:, ::6]                                  # (T, N_cols) — R1 only
per_col  = res.reshape(res.shape[0], -1, 6).mean(-1)    # (T, N_cols) — avg R1..R6
```

Pick which slicing pattern to use based on what the downstream
consumer expects. NCD photoreceptor microcircuits typically take a
single per-column value, so most pipelines use either `[:, ::6]` (R1
only) or the R1..R6 mean.

## 4. Static vs Active — picking one

| Need | Use |
|---|---|
| Quick baseline / debugging the rest of the pipeline | Static |
| You only care about the optical projection geometry | Static |
| Want a circuit-comparable baseline with no RF dynamics | Static |
| Simulating saccade / fixation / pupil-style angular dynamics | Active |
| Receptive fields should shrink under high luminance | Active |
| You want to record `(responses, angles, elevations)` jointly | Active + `run_more()` |
| You need the cheapest run that gets a reasonable response | Static |

The two share a screen and retina; you can run both on the same
inputs and compare directly — that's exactly what
[`docs/demos/demo_active_sampling.py`](demos/demo_active_sampling.py)
does (it saves a `comparison.png` per-column R1 trace overlay) and
what `demo_fes.ipynb` §VI–§VII shows interactively.

## 5. Dual-eye and rotated retinas

`RetinaRotator` produces a transformed copy of a retina without
re-running the grid generator. Used to compose left/right eye pairs:

```python
from flyeyesimulator import Retina, RetinaRotator

retina = Retina(num_rings=20, bio_cols_only=True, col_json_path=COL_JSON)

retinaL = RetinaRotator(retina, offset=(-0.5, 5, 0),
                        euler_deg=(-60,  60, 0), mirror='x').apply()
retinaR = RetinaRotator(retina, offset=( 0.5, 5, 0),
                        euler_deg=(-60, -60, 0)).apply()
```

Each rotated retina is a self-contained `Retina` you can pass to a
separate simulator, or render in `SimulatorViz` alongside the
original. See `demo_flyeye.py::demo_dual_picked_rays` and the
`dual_pick` demo argument for a complete dual-eye geometry walk.

## 6. Backend selection

FES is backend-agnostic via `flyeyesimulator.backend`:

| `BACKEND` value | When | Notes |
|---|---|---|
| `'cupy'` | CuPy installed **and** a CUDA device is detected | preferred on Linux/CUDA |
| `'mlx'` | CuPy unavailable, MLX installed | preferred on Apple Silicon |
| `'numpy'` | neither of the above | works everywhere, slow |

Use `xp`, `xp_ndimage`, and `to_numpy(a)` to write code that runs on
all three:

```python
from flyeyesimulator import xp, to_numpy, BACKEND

photon_arr = xp.asarray(photon_video)        # numpy → backend
res_native = fes.run(release=False)          # stays on backend
res_np     = to_numpy(res_native)            # → host numpy

print(f'running on {BACKEND}')
```

Most user code only needs `xp.asarray(...)` to push the input video
onto the backend; the rest happens internally.

## 7. Geometry visualisation — `SimulatorViz`

Before running a simulation, you usually want to verify the retina
and screen geometry are sensible. `SimulatorViz` is the all-in-one
Plotly 3D scene for that:

```python
from flyeyesimulator import SimulatorViz

viz = SimulatorViz(lens_radius=0.04, lens_length=0.01,
                   rays=False, ray_mode='R7', ray_length=8, ray_width=2)
viz.add_retina(retina, color='steelblue', name='eye')
viz.add_screen(radius=10)
viz.add_rays([
    ('eye', 0, r_type, 'red') for r_type in
    ('R7', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6')
])
viz.plot()
```

The R-axis fan (R1 of col 5, R2 of col 6, R3 of col 18, R4 of col 1,
R5 of col 2, R6 of col 3 all aimed at column 0) is the canonical
visualisation of **neural superposition** — six neighbouring
ommatidia's R-cells co-pointing at one world location. See
`demo_flyeye.py::demo_single_fan` and `demo_fes.ipynb` §II.

## 8. Output visualisation — `ScreenViz`, `RetinaViz`, `IOViz`

After running, you'll want to look at what the screen saw or what the
retina sampled. Three renderers, each consuming the simulator's
standard output shapes:

| Object | Method | Input |
|---|---|---|
| `ScreenViz(screen)` | `.plot(frame_idx)` | one screen frame |
| `ScreenViz(screen)` | `.save_video(fps, html_path, downsample)` | animated Plotly surface |
| `RetinaViz(retina)` | `.plot(values_at_t)` | per-column array, length `N_cols` |
| `RetinaViz(retina)` | `.save_video(values_TN)` | per-column `(T, N_cols)` |
| `RetinaViz(retina)` | `.save_video_row([data_a, …], title_list)` | side-by-side retina panels |
| `RetinaViz(retina)` | `.save_video_rows(row1, row2, t1, t2)` | 2×K retina grid |
| **`IOViz(screen, retina_viz)`** | `.save_video(retina_values, titles, ...)` | **screen + retina panels in one synchronised figure** |

`IOViz` is the right choice when you want the **screen and the retina
panels in one animation** (so the time slider scrubs the input video
and the responses together). 1-row layout puts the screen on the
left + N retina panels to the right; 2-row layout makes the screen
big on the left (`rowspan=2`) with two rows of retina panels next to
it. Each panel has its own colour range and colorbar. See §8.1.

`HexViz` (in `neurocircuitdesk.io_utils`) is the matplotlib-over-video
alternative for cases where you want the input video as the
background and a flat 2D hex scatter for the response. The shipped
demos use `IOViz` for the canonical screen+retina+phototransduction
view (`demo_fes.ipynb` §VII).

### 8.1. `IOViz` — screen + retina-panel grid

```python
from flyeyesimulator import IOViz

iv = IOViz(
    screen=screen,
    retina_viz=ret_viz,                  # any RetinaViz instance you already have
    colorscale='Viridis',                # default for retina panels
    screen_colorscale='gray',
)

# 1-row layout: flat list of (T, N_cols) arrays + matching titles
fig = iv.save_video(
    retina_values=[lum_R1_TN, V_R1_TN, V_avg_TN],
    titles=['R1 photon', 'R1 voltage', 'R1–R6 voltage'],
    screen_crange=(0, 3e5),
    retina_crange=[None, (-80, 0), (-80, 0)],   # `None` → auto-fit
    fps=15, dt_ms=10,
    html_path='out_1row.html',
)

# 2-row layout: nested lists. Screen spans both rows (big on the left).
fig = iv.save_video(
    retina_values=[
        [static_R1_TN, static_avg_TN, static_LMC_TN],
        [active_R1_TN, active_avg_TN, active_LMC_TN],
    ],
    titles=[
        ['Static R1', 'Static avg', 'Static LMC'],
        ['Active R1', 'Active avg', 'Active LMC'],
    ],
    screen_crange=(0, 3e5),
    fps=30, dt_ms=10,
    html_path='out_2rows.html',
)
```

Layout is auto-detected from `retina_values`: flat list ⇒ 1-row, nested
list ⇒ 2-row. `titles` and `retina_crange` mirror that shape. Each entry
in `retina_crange` is `(cmin, cmax)` or `None` (auto-fit).

Constructor caches once: `screen.get_sphere_geometry()` +
`retina_viz.get_r1_mesh()`. You can build many figures from one
`IOViz` instance — geometry is computed once.

**Inline rendering vs HTML export.** At full resolution
(`screen_downsample=1`, `frame_stride=1`), a 300-frame 256×256 screen
+ four retina panels produces a ~300 MB JSON figure — Jupyter will
choke trying to render that inline. Two knobs trim the payload:

| Parameter | Where | Effect |
|---|---|---|
| `screen_downsample` | constructor (default 4) | Spatial subsample of the screen surface (every Nth row/col). 4 cuts the screen 16× |
| `frame_stride` | `save_video` (default 1) | Keep every Nth frame of the animation. `frame_stride=5` cuts the payload 5× and labels the slider in scaled ms |

Defaults (`screen_downsample=4`, `frame_stride=1`) are tuned for the
shipped demo; pass `frame_stride=5` for ≥ 300-frame sequences if
inline rendering still chokes. For full-resolution HTML deliverables,
pass `screen_downsample=1, frame_stride=1` and consume the figure via
`html_path=` only — don't `display(fig)`.

## 9. From photon rates to photoreceptor voltage — `phototransduction`

`FlyEyeSimulator{Static,Active}.run()` outputs **photon rates** in the
~10⁵ regime — that's the *light* reaching each rhabdomere, not the
*voltage* of the photoreceptor cell. For circuit work that needs
voltage (most of NCD's photoreceptor microcircuits do), pipe the photon
rates through the `phototransduction` module:

```
(T, 6·N_cols)  photon rates  ──pr()──▶  (T, 6·N_cols)  voltage  (mV)
                                                  │
                                  ┌───────────────┘
                                  ▼
                      aggregate_pr(V_TN6, num_cols)        uses
                                  │             superposition.get_prs
                                  ▼
                          (T, N_cols)  voltage  ──▶  NCD PR MCs
```

The model is **RPM phototransduction → HH(NS) membrane**:

- **RPM** — a 3-state ODE (`x1, x2, x3`) converting light into a
  transient TRP current.
- **HH(NS)** — by default the non-spiking 5-gate Hodgkin–Huxley
  membrane published for Drosophila photoreceptors. Pass
  `membrane='hh'` for the classical spiking 3-gate variant.

Backend selection matches FES's own `BACKEND` — CuPy on CUDA, Metal
on Apple Silicon, Numba CPU otherwise. You don't pick the backend; the
module picks it for you at import time. (Inspect via
`flyeyesimulator.phototransduction._BACKEND_USED` if you need to confirm
which path was activated.)

### 9.1. The one-liner

```python
from flyeyesimulator import pr, aggregate_pr

V_TN6 = pr(res_static)                            # (T, 6·N_cols) voltage
V_TN  = aggregate_pr(V_TN6, num_cols=N_cols)      # (T, N_cols) per-column
```

`pr(res, *, membrane='hhns', itpl_val=1000, dt=1e-5)` accepts the raw
FES output. Defaults assume a **100 fps** video (one frame = `itpl_val
· dt` = 10 ms); scale `itpl_val` for other frame rates and keep `dt`
near `1e-5` (the RPM kinetics are stiff).

`aggregate_pr(V_TN6, num_cols, valid_idx=None, mode='mean')` uses
`flyeyesimulator.superposition.get_prs` to collect the 6 photoreceptors
in neural superposition for each target column, then combines them
(currently `mode='mean'` only). For each target column, `get_prs`
returns `(T, 6)` — one trace per R-cell from the matching ring-1
neighbours — and the mean across that axis gives the per-column
voltage trace. Missing channels (e.g. cells outside the biological
mask) fill with `-80.0 mV` placeholder before averaging.

### 9.2. The one-shot

```python
from flyeyesimulator import fes_to_pr_voltage

V_TN = fes_to_pr_voltage(res_static, num_cols=N_cols)             # mean aggregation
V_TN6 = fes_to_pr_voltage(res_static, num_cols=N_cols, aggregate=None)  # skip aggregation
```

### 9.3. The class form

The functional `pr` / `aggregate_pr` helpers wrap a class — exposed
under two names:

```python
from flyeyesimulator.phototransduction import PhotoreceptorRetina, Retina

# Both names refer to the same class — `Retina` is a short alias.
ret = PhotoreceptorRetina(res, amacrine=False, downsample=False, norm_input=False)
ret.sim(itpl_val=1000, dt=1e-5)
V_TN6 = ret.axon_terminal                # (T, 6·N_cols)
```

`amacrine=True` and `downsample=True` raise `NotImplementedError` —
the amacrine gain-control feedback isn't ported into the consolidated
module yet. Pass `False` (the default).

### 9.4. Passing it into NCD

`(T, N_cols)` voltage is exactly what an NCD photoreceptor microcircuit
expects at its `input_main` port. The bridge:

```python
from flyeyesimulator import pr, aggregate_pr

# 1. Photon rates → per-photoreceptor voltage → per-column voltage
V_TN  = aggregate_pr(pr(res_static), num_cols=len(retina.vrfs) // 6)

# 2. Feed into the canvas
cv = Canvas(w=…, h=…, col_json_path=COL_JSON)     # same col_json as retina
# … build the circuit (PR_col MC type) …

inputs_by_step = [
    {(f'PR_col_{c}', 'input_main'): float(V_TN[t, c])
     for c in range(V_TN.shape[1])}
    for t in range(V_TN.shape[0])
]
program = cv.compile()
out     = program.run_series(T=len(inputs_by_step), dt=1/60.0,
                             inputs_by_step=inputs_by_step)
```

The canvas's `col_json_path` and the retina's `col_json_path` **must
match** — otherwise the column indices the retina samples and the
indices the circuit's MCs sit on won't be the same set. See
`demo_looming.ipynb` for the full bridge.

### 9.5. Skipping phototransduction

If your downstream NCD circuit already expects photon-rate / lux input
(not voltage), you can skip the phototransduction layer entirely and
use the older "R1 slice" or "R1..R6 mean" bridge:

```python
lum_per_col = res[:, ::6]                                  # R1 only
# or:
lum_per_col = res.reshape(res.shape[0], -1, 6).mean(-1)    # R1..R6 average
```

The choice depends on what the photoreceptor MC's algorithm expects.
Most shipped circuits assume voltage — go through `phototransduction`.

## 10. Demos as use-case documentation

The shipped demos are the use-case documentation:

| File | What it shows |
|---|---|
| `docs/demos/demo_flyeye.py` | Geometry only: single retina, R7 column fan, R1–R6 superposition fan, dual-eye rotated + mirrored. |
| `docs/demos/demo_active_sampling.py` | Full Static + Active pipeline on a synthesised looming-dot video; saves `.npy` arrays and a per-column R1 comparison plot. |
| `docs/demos/demo_fes.ipynb` | Interactive notebook: retina, screen, saccade video, Static + Active, side-by-side animation with `RetinaViz.save_video_row`. The longest worked example; read this for the canonical end-to-end. |
| `docs/demos/demo_looming.ipynb` | Bridges FES output into an NCD looming circuit — see §9 above. |

Use these as your reference. The classes have docstrings for
parameter-level detail, but the demos are the right place to read for
"how do I actually use it."

## 11. Pitfalls

- **Forgot to scale the video** — if you pass a `[0, 1]`-range video
  straight into `Screen`, the active simulator's spring-damper sees
  ~1e3 instead of ~1e5 input scale and barely moves. Always scale:
  `3e5 * video / video.max()` (or whatever your data's appropriate
  peak photon rate is).
- **Backend mismatch on input** — `Screen(...)` does call
  `xp.asarray(...)` internally, so passing a NumPy array works. But
  if you're benchmarking, prefer pre-converting once via
  `photon_arr = xp.asarray(photon_video)` so the conversion isn't
  inside your timed region.
- **`release=True` invalidates the simulator** — calling `.run()` a
  second time on the same instance after `release=True` raises (the
  filters / screen buffers are gone). Build a fresh simulator for a
  second run, or pass `release=False` and call `.release_device_memory()`
  manually when done.
- **`bio_cols_only` mismatch with the column JSON** — if you set
  `bio_cols_only=True` but pass `col_json_path=None`, construction
  raises. The two have to agree.
- **Output column count** is `N_cols = num_rings * (num_rings + 1) * 3 + 1`
  (the hex-spiral count) regardless of `bio_cols_only` — disabled
  cells still occupy a slot in the output array, they just have a
  zero filter. Slice by the canvas's MC col indices, not by a dense
  range.

## See also

- Source: `flyeyesimulator/{retina,screen,flyeyesimulator_static,flyeyesimulator_active,retina_rotator,backend,retinaviz,screenviz,simulatorviz,phototransduction,IO_viz}.py`.
- Shipped demos: `docs/demos/demo_fes.ipynb`, `docs/demos/demo_active_sampling.py`, `docs/demos/demo_flyeye.py`.
