# Writing algorithms with the unified signature

> **Status.** This document is the user-facing reference for the
> unified `@unified_algorithm` signature, which is what the shipped
> algorithms in `neurocircuitdesk/libs/algorithms.py` use today. The
> scalar and MLX engines both dispatch unified-signature functions
> directly; the legacy 4-convention dispatch remains available for
> non-decorated functions during migration but is not exercised by
> any shipped block.

This guide covers:

1. [The two function shapes](#the-two-function-shapes)
2. [The `inputs` dict — reference](#the-inputs-dict--reference)
3. [The `params` dict — reference](#the-params-dict--reference)
4. [Return-value conventions](#return-value-conventions)
5. [State management for stateful algorithms](#state-management-for-stateful-algorithms)
6. [Backend-agnostic math](#backend-agnostic-math)
7. [Decorating and registering](#decorating-and-registering)
8. [Worked examples](#worked-examples)
9. [Common pitfalls and FAQ](#common-pitfalls-and-faq)
10. [Migration cheat sheet](#migration-cheat-sheet-from-the-old-conventions)

---

## Core promise

**You write one function. It runs on either backend.**

- Construct your microcircuits and wire your canvas the same way you always did.
- Choose the backend at compile time: `cv.compile()` for scalar, `cv.compile_mlx()` for MLX.
- The engine presents inputs, params, and state to your function in a shape that works in both modes. You write the algorithm body without branching on backend.

You do **not** need to:

- Write two versions of your algorithm (no `*_batched` twin).
- Call `FuncBlock.register_batched(...)` for the common case.
- Care whether `inputs['neighbors']` is a `np.ndarray` or an `mx.array` — the operators you need work on both.
- Know about padding, slot indexing, or declared port order — the engine handles all of that before your function is called.

---

## The two function shapes

Every unified algorithm matches one of exactly **two** signatures:

```python
from neurocircuitdesk import unified_algorithm

# Stateless — most algorithms
@unified_algorithm
def my_algorithm(inputs, params):
    ...
    return {'output_port_a': value_a, 'output_port_b': value_b}


# Stateful — anything that needs history, buffers, counters
@unified_algorithm
def my_stateful_algorithm(inputs, params, state):
    ...
    return {'output': value}, state
```

That's the whole API. No 4-arg variant. No SISO raw-value shortcut. No branch on `is_mimo`.

The `@unified_algorithm` decorator is how the engine knows your function is written in this form (as opposed to an old-convention scalar function). Forgetting the decorator falls back to legacy dispatch and will surface a clear error at compile time.

---

## The `inputs` dict — reference

`inputs` is a `Dict[str, T]` where `T` is whatever shape the current backend uses:

| Backend | Shape of a scalar-per-node input | Shape of a per-neighbor input |
|---|---|---|
| Scalar engine | Python `float` | `np.ndarray` of shape `(n_nbrs,)` |
| MLX engine | `mx.array` of shape `(N,)` where N = number of node instances | `mx.array` of shape `(N, max_nbrs)` |

You don't branch on this. The operators you use (`+`, `-`, `*`, `/`, `**`, `.sum(axis=-1)`, `[..., k]`, etc.) behave identically across all four shapes.

### Standard named ports

Any port you declared in the template shows up under its real name:

```python
def T1_poly(inputs, params):
    x = inputs['input']       # present because the template declared port 'input'
    return {'output': params['b1'] + params['a1'] * x + params['a2'] * x * x}
```

If your block has multiple named inputs, they each get their own key:

```python
def division_like(inputs, params):
    num = inputs['numerator']
    den = inputs['denominator']
    return {'output': num / (den + params['eps'])}
```

### MIMO blocks — `inputs['neighbors']` and `inputs['neighbor_mask']`

A **MIMO block** is one whose input ports are all named `input_col_<N>` (e.g., a motion detector reading ring-1 neighbours, or an MVP reading its ring-2 neighbourhood). For these blocks, the engine assembles the neighbourhood into a single array and exposes it under a fixed key:

```python
def mvp_algorithm(inputs, params):
    F    = inputs['neighbors']        # scalar: (n_nbrs,) | batched: (N, max_nbrs)
    mask = inputs['neighbor_mask']    # same shape as F — values are 0.0 or 1.0

    # Compute mean over the neighbour axis, ignoring pad slots:
    y = (F * mask).sum(axis=-1) / mask.sum(axis=-1)
    ...
```

**Key contract:** axis `-1` is **always** the neighbour axis. In scalar mode `F` is 1-D so `axis=-1` reduces to a scalar; in batched mode `F` is 2-D so `axis=-1` reduces along the last axis giving shape `(N,)`. Same code path.

**The order along the neighbour axis is the template's declared port order** — i.e. the order you passed to `add_block(..., input_names=[...])` or declared via `specify_io`. This is NOT sorted by `col_idx`. If your template author wrote the ports in spiral order (centre first, then ring-1 CW from north), your algorithm sees them in that order and `F[..., 0]` is always the centre.

**`neighbor_mask` semantics:** `mask[k] == 1` if slot `k` is within your node's declared port count; `mask[k] == 0` only in the pad region beyond that count (batched mode only — scalar mode has no padding). **Important:** mask is 1 even for slots whose input port is unconnected or only fed by an off-canvas InputNode. Those slots carry `0.0` in `F`, but they **count** in the mask sum. This matches the scalar reference's `np.mean(F) = sum(F) / len(declared_ports)` semantics exactly.

The rule of thumb: if you want a plain mean over the neighbourhood, write:

```python
y = (F * mask).sum(axis=-1) / mask.sum(axis=-1)
```

and it will match the scalar engine's `np.mean(F)` divisor (declared port count), not the smaller "connected ports only" count.

### Ordered positional access for non-MIMO blocks (future)

If a future block declares multiple named inputs but wants positional access (e.g., to index by template-declared slot), the engine will optionally expose `inputs['__ordered__']` with the same contract as `inputs['neighbors']`. This is **not shipped in the first cut** — every current block that needs positional access is already a MIMO block, so `neighbors` covers it. The key is reserved.

---

## The `params` dict — reference

`params` is your block's parameter dict, exactly as you set it with `mc.set_block_params(block_id, {...})`. Three kinds of values:

### Scalar params

Plain Python numbers, strings, tuples — passed through unchanged. Usable identically in both engines.

```python
mc.set_block_params('motion_detector', {'N': 2, 'alpha': 100, 'beta': 100})

def borst_algorithm(inputs, params, state):
    alpha = params['alpha']    # int, same in both backends
    ...
```

### Array params

`np.ndarray` values — the MLX engine converts them to `mx.array` at compile time; the scalar engine leaves them as numpy. You access them the same way in both:

```python
mc.set_block_params('bp_block', {'filter': bp_filter()})   # numpy array

def temporal_filter(inputs, params, state):
    f = params['filter']       # scalar: np.ndarray | batched: mx.array
    ...
```

### Per-neighbour dict params (auto-packed)

This is the important one for MIMO blocks. If you write a dict whose keys are column indices:

```python
g1 = {col_idx: np.exp(-0.5 * (d / 0.85) ** 2) for col_idx, d in neighborhood.items()}
mc.set_block_params('mvp_processor', {'g1': g1})
```

the engine **auto-packs** it at compile time into an array aligned with your declared port axis:

| Backend | Shape of `params['g1']` |
|---|---|
| Scalar | `np.ndarray` of shape `(n_nbrs,)` |
| MLX | `mx.array` of shape `(N, max_nbrs)` |

In both cases, `params['g1'][..., k]` is the per-neighbour gain for slot `k`, which aligns slot-for-slot with `inputs['neighbors'][..., k]`. You do elementwise math:

```python
def mvp_algorithm(inputs, params):
    F  = inputs['neighbors']
    g1 = params['g1']                        # already packed, SAME key name
    return {'val_col_neighbors': F * g1 * 0.33}
```

**You do not see the dict form inside your algorithm.** That's intentional — packing is an engine detail. The dict only exists at `set_block_params` call time as a convenient way for the template author to specify per-column values by column ID.

> **Contract:** a dict param whose keys are `int` col indices is auto-packed against the node's output slot axis. Any other dict param (e.g., `{'a1': 0.001, 'a2': 1e-7}`) is passed through unchanged — it's just a regular dict of named constants.

---

## Return-value conventions

Stateless algorithms return a dict. Stateful ones return `(dict, state)`. The dict keys are output port or channel names; the values are the same shape as your inputs (scalar `float` / `(N,)` in batched / `(n_nbrs,)` or `(N, max_nbrs)` for per-neighbour).

### Simple named outputs

```python
return {'output': value}
return {'output_a': val_a, 'output_b': val_b, 'output_c': val_c, 'output_d': val_d}
```

Each key must match an output port name your template declared.

### Per-neighbour outputs (fan-out MIMO)

Some blocks emit a value **per neighbour** — the canonical case is MVP's DNP feedback, where each MVP node sends one `val` and one `weight` back to each upstream PR column. Your template declared output ports like `output_val_col_935`, `output_val_col_936`, …, `output_weight_col_935`, `output_weight_col_936`, …

Instead of returning a dict with one key per port, you return **one key per channel** with a `_neighbors` suffix and a per-neighbour array:

```python
def mvp_algorithm(inputs, params):
    F  = inputs['neighbors']
    g1 = params['g1']
    y  = (F * inputs['neighbor_mask']).sum(axis=-1) / inputs['neighbor_mask'].sum(axis=-1)
    delta = 1.0165e-07 * y + 7.60e-4

    return {
        'val_col_neighbors':    delta * F * 0.33,    # shape: (N, max) batched, (n_nbrs,) scalar
        'weight_col_neighbors': g1,                   # shape: (N, max) batched, (n_nbrs,) scalar
    }
```

The engine unpacks `val_col_neighbors[..., k]` into the output port whose slot is `k` — which, by [the reciprocity contract](#the-reciprocity-contract), is the same port that fed `inputs['neighbors'][..., k]`.

### The reciprocity contract

For any block that is **both** a MIMO input consumer (reads `inputs['neighbors']`) **and** a fan-out MIMO output producer (returns `*_neighbors` keys), the engine guarantees:

> **The input neighbour axis and the output neighbour axis are aligned slot-for-slot per node.**

This is what lets you write `delta * F * 0.33` elementwise and just return the result. Slot `k` of the output is guaranteed to be the correct per-neighbour output for the source that fed slot `k` of the input.

If your input and output neighbourhoods cover different sets of neighbours (rare — no example in the motion demo), the contract doesn't apply and you need an explicit per-slot scatter. That's out of scope for the first cut of the unified signature.

### Channel naming rule

- **Real output port name** (e.g., `'output'`, `'output_a'`) → value is a per-node scalar/array.
- **Channel name ending in `_neighbors`** (e.g., `'val_col_neighbors'`) → value is a per-neighbour tensor; the engine unpacks to all output ports of the form `<channel>_<col_idx>`.

Mixing both in one return dict is allowed — a block can emit some per-node outputs and some per-neighbour outputs.

---

## State management for stateful algorithms

Stateful algorithms receive a `state` dict and return `(outputs, state)`. The catch: in scalar mode, state values can be Python objects like `collections.deque`; in batched mode, they need to be `mx.array`s that can't be dynamically resized. Writing backend-polymorphic state code by hand is tedious and error-prone.

**Solution:** use the helpers in `neurocircuitdesk.state_utils`, which hide the backend divergence entirely.

### Ring buffers — the 95% case

Most stateful algorithms just need a rolling window of the last `N+1` values (for N-step delays, temporal filters, derivatives, etc.):

```python
from neurocircuitdesk.state_utils import ring_buffer_push, ring_buffer_get, ring_buffer_len

@unified_algorithm
def col_power_algorithm(inputs, params, state):
    N = params['N']
    p_current = inputs['pow_input']

    buf = ring_buffer_push(state, 'history', p_current, maxlen=N + 1)

    # Pre-warm: first N steps return a default before the buffer is full
    if ring_buffer_len(buf, state, 'history') <= N:
        return {'output': 1e-3}, state

    p_delayed = ring_buffer_get(buf, 0)      # oldest
    p_now     = ring_buffer_get(buf, -1)     # newest (just pushed)

    return {'output': some_fn(p_now, p_delayed, params)}, state
```

What the helpers do:

- **`ring_buffer_push(state, key, value, maxlen)`** — first call allocates a `deque` (scalar) or a rolling `mx.array` (batched) based on the type of `value`. Subsequent calls append/shift. Updates `state[key]` in place **and returns the buffer**, so you can pass it straight to the `get` / `len` helpers below.
- **`ring_buffer_get(buf, idx)`** — returns the entry at position `idx`. `0` = oldest, `-1` = newest, negative indexing supported. Works on both backends. Takes the **buffer** (returned by `push`), not `(state, key, idx)`.
- **`ring_buffer_len(buf, state, key)`** — returns the number of entries pushed so far (capped at `maxlen`). Pass `state` and `key` so the helper can read its sidecar fill counter and give the correct warmup-aware length. The 1-arg fallback `ring_buffer_len(buf)` returns the storage size and will misreport during pre-warm.

You never see `deque.append` or `mx.concatenate`. Your algorithm body reads the same in both backends.

### More complex state

For state that isn't a ring buffer — a running counter, a learned parameter, a lookup table — just store it directly in the `state` dict. Use plain Python types (`int`, `float`, `np.ndarray`) in scalar mode and `mx.array` in batched mode. If you need both backends to share exact code, store the value and dispatch only on the backend's native constructor at first-touch:

```python
if 'counter' not in state:
    state['counter'] = 0
state['counter'] += 1
```

Plain counters work identically in both backends.

For array-valued state that isn't a rolling window (e.g., a persistent weight matrix that's updated elementwise), follow the same "allocate on first use" pattern, but be careful about dtype — in batched mode always use `mx.zeros(...)` not `np.zeros(...)`, otherwise you can't assign `mx.array` values into an ndarray slot. The cleanest pattern is to defer to a `state_utils` helper (`state_array_init(state, key, shape_like)`) once we add more of them. For now, ring buffers cover every shipped algorithm.

---

## Backend-agnostic math

The unified signature only works because a large subset of numerical operators behave identically on `float`, `np.ndarray`, and `mx.array`. Here's what's safe and what isn't.

### Always safe — use these freely

**Arithmetic operators:** `+`, `-`, `*`, `/`, `//`, `%`, `**`, unary `-`. All work on scalars and arrays of either backend, including broadcasting.

**Comparison operators:** `<`, `<=`, `==`, `>=`, `>`, `!=`. Return booleans or boolean arrays.

**Array methods (call as `arr.method(...)`):**

- `.sum(axis=...)`, `.mean(axis=...)`, `.max(axis=...)`, `.min(axis=...)`, `.prod(axis=...)`
- `.reshape(...)`, `.transpose(...)`, `.flatten()`
- `.astype(dtype)`
- Indexing: `arr[i]`, `arr[i, j]`, `arr[..., k]`, `arr[mask]`, `arr[i:j]`
- Negative indexing, ellipsis, slicing — all standard.

**Constants:** numeric literals, `math.pi`, `math.e`.

### Usually safe — one caveat

**`.mean()` without axis** works on both backends but reduces to a 0-D array / scalar. Be explicit with `axis=-1` (or whichever) to keep dimensionality predictable.

**Fancy indexing with integer arrays:** `F[..., [0, 1, 4]]` works on both numpy and mlx, but the resulting shape can differ subtly for higher-rank arrays. Stick to 1-D index lists along the last axis and you'll be fine.

### NOT safe — use the `ncd.math` façade instead

These are namespaced differently in numpy vs mlx, so you can't write backend-agnostic code against them directly:

- `np.where(...)` vs `mx.where(...)`
- `np.clip(...)` vs `mx.clip(...)`
- `np.exp(...)`, `np.log(...)`, `np.sqrt(...)` vs `mx.*` equivalents
- `np.stack(...)`, `np.concatenate(...)` vs `mx.*`
- `np.maximum(...)`, `np.minimum(...)` (elementwise, not reductions) vs `mx.*`

For these, import from the façade:

```python
from neurocircuitdesk.math import where, clip, exp, log, sqrt, stack, concatenate, maximum, minimum

def my_algorithm(inputs, params):
    x = inputs['input']
    return {'output': where(x > 0, exp(x), 0.0)}
```

The façade is thin — each wrapper dispatches based on the argument's type at call time. Cost is negligible compared to the underlying op.

### Rule of thumb

If the operation is a method on the array (`arr.sum(axis=-1)`, `arr[i, j]`, `arr * 0.5`), write it directly. If it's a free function that takes an array (`np.where(mask, a, b)`), import the same name from `ncd.math`.

---

## Decorating and registering

### The decorator

```python
from neurocircuitdesk import unified_algorithm

@unified_algorithm
def my_algorithm(inputs, params):
    ...
```

What it does:

- Marks the function so the engine dispatches it as unified (no SISO raw-value branch, no 4-arg stateful MIMO branch).
- Records the call signature (stateless vs stateful) so the engine knows whether to pass `state`.

Without the decorator, the engine falls back to legacy dispatch and your function is expected to match one of the old four conventions. This lets old code keep running during migration, but new code should always decorate.

### Using the function in a template

Same as today — pass it to `mc.set_block_func`:

```python
@unified_algorithm
def mvp_algorithm(inputs, params):
    ...

def mvp_template(mc):
    mc.add_block('mvp_processor', *mc.center,
                 input_names=input_names,
                 output_names=output_names)
    mc.set_block_func('mvp_processor', mvp_algorithm)
    mc.set_block_params('mvp_processor', {'g1': g1_dict})
```

The template is fully backend-neutral — it never references numpy vs mlx.

### When you need a hand-tuned batched variant

The common case is that you write one unified function and let both engines call it directly. But for a performance-critical block where the natural unified form is suboptimal on MLX (e.g., you want to fuse operations by hand), you can still register a batched override:

```python
@unified_algorithm
def mvp_algorithm(inputs, params):
    """Default: used by the scalar engine and by MLX if no override is registered."""
    ...

def mvp_algorithm_mlx(inputs, params):
    """Hand-tuned MLX variant. Same signature; the engine prefers this in batched mode."""
    ...

FuncBlock.register_batched(mvp_algorithm, mvp_algorithm_mlx)
```

This is the **escape hatch**, not the common path. The payoff is only worth it for algorithms where the unified form has observable overhead in profiling. For everything in the motion demo, the unified form is fast enough — the auto-packed params, pre-assembled `neighbors` tensor, and elementwise ops all fuse into the same MLX kernels you'd write by hand.

---

## Worked examples

Five complete algorithms, one per pattern we ship.

### Example 1 — Stateless SISO: `T1_poly`

```python
from neurocircuitdesk import unified_algorithm

@unified_algorithm
def T1_poly(inputs, params):
    x = inputs['input']
    return {'output': params['b1'] + params['a1'] * x + params['a2'] * x * x}
```

That's the whole algorithm. `x` is a float in scalar mode and an `(N,)` array in batched mode. The operators work on both. No neighbours, no state, no special parameters.

Template usage:

```python
mc.add_block('T1', x, y, z)
mc.set_block_func('T1', T1_poly, {'b1': 0.0, 'a1': 0.001, 'a2': 1e-7})
```

### Example 2 — Stateless SISO passthrough

```python
@unified_algorithm
def placeholder_passthrough(inputs, params):
    return {'output': inputs['input']}
```

Trivially small but canonical — even the pass-through uses the full dict-in / dict-out form.

### Example 3 — Stateless MIMO with fan-out outputs and per-neighbour params: `mvp_algorithm`

This is the most feature-dense example. It reads a neighbour tensor, computes a mean, and returns a per-neighbour output tensor that feeds back to each upstream column.

```python
from neurocircuitdesk import unified_algorithm

@unified_algorithm
def mvp_algorithm(inputs, params):
    F    = inputs['neighbors']         # (n_nbrs,) or (N, max_nbrs)
    mask = inputs['neighbor_mask']
    g1   = params['g1']                # auto-packed; same shape as F

    # Mean over the neighbour axis using the declared-port-count mask
    y     = (F * mask).sum(axis=-1) / mask.sum(axis=-1)
    delta = 1.0165216804198919e-07 * y + 0.001760445128947395 - 0.001

    # delta has shape (,) scalar or (N,). We need to broadcast it against F
    # which has one extra axis (the neighbour axis). Add a trailing axis:
    if hasattr(delta, 'shape') and len(delta.shape) > 0:
        delta = delta[..., None]

    return {
        'val_col_neighbors':    delta * F * 0.33,   # per-neighbour output
        'weight_col_neighbors': g1,                 # per-neighbour output
    }
```

Template usage:

```python
def mvp_microcircuit_template(mc, neighborhood):
    input_cols = sorted(neighborhood.keys())
    input_port_names  = [f'input_col_{i}' for i in input_cols]
    output_port_names = [f'output_val_col_{i}' for i in input_cols] + \
                        [f'output_weight_col_{i}' for i in input_cols]

    mc.add_block('mvp_processor', *mc.center,
                 input_names=input_port_names,
                 output_names=output_port_names)
    mc.set_block_func('mvp_processor', mvp_algorithm)
    mc.set_block_params('mvp_processor', {'g1': gaussian_kernel(neighborhood, sigma=0.85)})
    mc.specify_io(
        inputs=[(n, 'mvp_processor', n) for n in input_port_names],
        outputs=[(n, 'mvp_processor', n) for n in output_port_names],
    )
```

What happens under the hood:

- At compile time, the engine sees that every output port matches `<channel>_col_<N>` and registers two channels: `val_col` and `weight_col`.
- It packs the `g1` dict into an array aligned with the declared port order.
- At each step, the engine builds `inputs['neighbors']` from the real values feeding each `input_col_<N>` port, in declared order.
- The algorithm does elementwise math on `F` and `g1`, which have aligned shapes by the reciprocity contract.
- The returned `val_col_neighbors` array, shape `(n_nbrs,)` scalar or `(N, max_nbrs)` batched, is unpacked slot-by-slot into `output_val_col_<c>` for each `c` in the declared port order.

Compare to the old two-function form in `test_motion_mlx_full.py` — the unified version is ~15 lines shorter, doesn't mention numpy or mlx, and doesn't reference internal keys like `__nbr_F__` or `g1_packed`.

### Example 4 — Stateful MIMO with positional indexing: `borst_algorithm`

Reads the last N+1 neighbourhood snapshots, indexes specific neighbours for each branch's directional preference:

```python
from neurocircuitdesk import unified_algorithm
from neurocircuitdesk.state_utils import ring_buffer_push, ring_buffer_get, ring_buffer_len

@unified_algorithm
def borst_algorithm(inputs, params, state):
    N     = params['N']
    alpha = params['alpha']
    beta  = params['beta']

    F = inputs['neighbors']   # (7,) scalar or (N, 7) batched — centre + 6 ring-1

    buf = ring_buffer_push(state, 'history', F, maxlen=N + 1)

    # Pre-warm: first N timesteps emit zeros until the buffer fills
    if ring_buffer_len(buf, state, 'history') <= N:
        zero = F[..., 0] * 0.0   # shape matches per-node output
        return ({'output_a': zero, 'output_b': zero,
                 'output_c': zero, 'output_d': zero}, state)

    y_vals       = ring_buffer_get(buf, -1)   # newest
    delayed_vals = ring_buffer_get(buf,  0)   # oldest

    def branch(y, x, z):
        return y * (1 + x * alpha) / (1 + z * beta)

    val_a = branch(
        y_vals[..., [0, 1, 4]].mean(axis=-1),
        delayed_vals[..., [2, 3]].mean(axis=-1),
        delayed_vals[..., [5, 6]].mean(axis=-1))
    val_b = branch(
        y_vals[..., [0, 1, 4]].mean(axis=-1),
        delayed_vals[..., [5, 6]].mean(axis=-1),
        delayed_vals[..., [2, 3]].mean(axis=-1))
    val_c = branch(
        y_vals[..., [0, 5, 6, 2, 3]].mean(axis=-1),
        delayed_vals[..., [5, 4, 3]].mean(axis=-1),
        delayed_vals[..., [1, 2, 6]].mean(axis=-1))
    val_d = branch(
        y_vals[..., [0, 2, 3, 5, 6]].mean(axis=-1),
        delayed_vals[..., [1, 2, 6]].mean(axis=-1),
        delayed_vals[..., [5, 4, 3]].mean(axis=-1))

    return ({'output_a': val_a, 'output_b': val_b,
             'output_c': val_c, 'output_d': val_d}, state)
```

Key things to notice:

- **Positional indexing with ellipsis:** `y_vals[..., 0]` reads slot 0 of the neighbour axis regardless of whether `y_vals` is shape `(7,)` scalar or `(N, 7)` batched.
- **Fancy indexing:** `y_vals[..., [0, 1, 4]]` picks three specific neighbours. Works on both backends along the last axis.
- **`.mean(axis=-1)`** reduces over the picked slots, giving shape `()` scalar or `(N,)` batched.
- **State** is managed entirely through the helpers — no `deque`, no `mx.concatenate`.
- **Pre-warm zero** is written as `F[..., 0] * 0.0` so it inherits the correct shape for broadcasting. (The engine can also handle bare `0.0` in the dict, but `F[..., 0] * 0.0` is more explicit about shape intent.)

### Example 5 — Stateful SISO with ring buffer: `col_power_algorithm`

```python
from neurocircuitdesk import unified_algorithm
from neurocircuitdesk.state_utils import ring_buffer_push, ring_buffer_get, ring_buffer_len
from neurocircuitdesk.math import exp, log

@unified_algorithm
def col_power_algorithm(inputs, params, state):
    N = params['N']
    p_current = inputs['pow_input']

    buf = ring_buffer_push(state, 'history', p_current, maxlen=N + 1)

    if ring_buffer_len(buf, state, 'history') <= N:
        return ({'output': p_current * 0.0 + 1e-3}, state)

    p_delayed = ring_buffer_get(buf, 0)
    ratio     = exp(log(p_current + 1e-9) - log(p_delayed + 1e-9))   # safe log-space ratio
    return ({'output': ratio * params['gain']}, state)
```

Illustrates:

- Reading a regular named input (`'pow_input'`) — not every stateful algorithm is MIMO.
- Using the `ncd.math` façade for `exp` and `log` because they're free functions (numpy) vs methods (mlx) differently.
- Emitting a broadcast-compatible default during the pre-warm window by multiplying the current value by zero.

---

## Common pitfalls and FAQ

### "My algorithm works in scalar mode but outputs NaN in batched mode at border cells."

You're probably dividing by `mask.sum()` and assuming it's never zero. In batched mode, a pad-only row (no declared ports at all — shouldn't happen in practice but can during malformed templates) has `mask.sum() == 0`. Use the standard safe pattern:

```python
ms = mask.sum(axis=-1)
safe_ms = where(ms > 0, ms, 1.0)     # from ncd.math
y = (F * mask).sum(axis=-1) / safe_ms
```

### "Why does `inputs['neighbors']` have padding in batched mode but not in scalar mode?"

Because batched mode packs N nodes into a single `(N, max_nbrs)` tensor and `max_nbrs` is the maximum across all nodes. Nodes with fewer declared ports get pad slots at the end, with `F = 0` and `mask = 0`. Scalar mode processes one node at a time so there's no need for padding.

The `(F * mask).sum(axis=-1) / mask.sum(axis=-1)` idiom handles both cases identically — pad slots contribute 0 to the numerator and 0 to the denominator.

### "I need to index by col_idx, not by slot position."

You shouldn't. The template declares ports in a meaningful order (e.g., spiral, alphabetical, whatever the author chose). Your algorithm should use that order. If you need "the centre" and "the north neighbour", the template should put them at slots 0 and 1 — that's a template concern, not an algorithm concern.

If you genuinely need col_idx access for debugging, you can reach into the block's metadata at compile time, but don't do it in the step function.

### "What happens if I return a dict key that isn't declared as an output port?"

Error at runtime (scalar) or silent drop (batched, unless the channel name has `_neighbors` suffix). Declare every output port you intend to emit. The template is the contract.

### "Can I access the previous step's output in a non-stateful algorithm?"

No. Stateless algorithms are just that — no memory. If you need history, mark the algorithm stateful and use `ring_buffer_push`. The engine's feedback-loop support (SCC detection + one-step delay) handles algebraic loops between different blocks, but a single algorithm's temporal state goes through the `state` parameter.

### "How do I know if my block is MIMO?"

By construction. If every input port name matches `input_col_<N>`, the engine treats it as MIMO and builds `inputs['neighbors']`. If your ports are named anything else (`'input'`, `'numerator'`, `'den_feedback_val'`, etc.), the engine passes them under their real names and there is no `inputs['neighbors']` key. You don't declare MIMO-ness explicitly — it falls out of your port naming.

### "Can a block mix MIMO ports and non-MIMO ports?"

Not supported in the first cut. A block is either fully MIMO (all input ports match `input_col_<N>`) or fully standard. Mixing would require two separate feed-assembly paths per call and isn't needed for any shipped algorithm.

### "Where do I set `stateless=True` or `stateless=False`?"

At `add_block(..., stateless=True)` call time, same as today. The decorator knows whether your function expects a `state` argument by introspecting its signature, but the block itself still needs the flag so the engine can decide whether to preserve state between steps. Set it to match your function's signature.

---

## Migration cheat sheet from the old conventions

| Old convention | New unified form |
|---|---|
| `def f(x, params):` (SISO raw value) | `def f(inputs, params): x = inputs['input']` |
| `def f(x, params, state):` (stateful SISO raw value) | `def f(inputs, params, state): x = inputs['input']` |
| `def f(inputs, params, state, ordered_input_names):` (stateful MIMO) | `def f(inputs, params, state):` — drop the 4th arg, read `inputs['neighbors']` |
| `F = np.array([inputs[k] for k in sorted_keys])` | `F = inputs['neighbors']` |
| `F = inputs['__nbr_F__']` (MLX batched) | `F = inputs['neighbors']` — same key in both backends |
| `mask = inputs['__nbr_mask__']` (MLX batched) | `mask = inputs['neighbor_mask']` |
| `g1_pack = params['g1_packed']` (MLX batched) | `g1 = params['g1']` — no `_packed` suffix |
| Returning `{f'output_val_col_{c}': ... for c in ...}` per-port | Returning `{'val_col_neighbors': array}` once |
| `state['history'] = deque([], maxlen=N+1)` (scalar) | `buf = ring_buffer_push(state, 'history', value, maxlen=N+1)` |
| `np.concatenate([hist[:,1:], v[:,None]], axis=1)` (MLX) | `buf = ring_buffer_push(state, 'history', value, maxlen=N+1)` |
| `len(state['history']) <= N` | `ring_buffer_len(buf, state, 'history') <= N` |
| `state['history'][0]` | `ring_buffer_get(buf, 0)` |
| `FuncBlock.register_batched(f, f_batched)` + two function bodies | `@unified_algorithm` + one function body |
| `np.where(cond, a, b)` | `from ncd.math import where; where(cond, a, b)` |
| `np.mean(F)` | `F.mean(axis=-1)` — explicit axis |

Every migration is local to the function body. No template code changes. No wiring code changes. The same `build_canvas_skeleton` and `wire_canvas` functions that work today continue to work unchanged — only the algorithm implementations and the `set_block_func` function references change.

---

## See also

- **`UNIFIED_SIGNATURE_ANALYSIS.md`** — the design rationale and the list of engine invariants the implementation must preserve.
- **`UNIFIED_SIGNATURE_IMPLEMENTATION.md`** — the phased implementation plan for landing this in the engine.
- **`test_motion_benchmark.py`** — the scalar reference, pre-migration.
- **`test_motion_mlx_full.py`** — the two-function form with `register_batched_variants()`, to be replaced by the unified single-function form after Phase 5 of the implementation plan.
