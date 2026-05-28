from typing import List, Tuple, Dict, Optional, Callable, Any
from collections import deque
import re
import numpy as np
from scipy.ndimage import convolve1d


# ---------------------------------------------------------------------------
# Unified-signature decorator
# ---------------------------------------------------------------------------
# Mark an algorithm as "unified": it accepts the canonical signature
#     stateless: f(inputs: dict, params: dict) -> dict
#     stateful:  f(inputs: dict, params: dict, state: dict) -> (dict, dict)
# and is backend-agnostic (runs identically on the scalar and MLX engines).
#
# The FuncBlock dispatcher uses the canonical signature unconditionally —
# every algorithm MUST be decorated. The MLX engine checks the `_unified`
# attribute to skip the legacy `register_batched` lookup and call the
# function directly in batched mode.
# ---------------------------------------------------------------------------

def unified_algorithm(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    signature: str = "stateless",
    ports: Optional[dict] = None,
    description: str = "",
) -> Callable:
    """Mark a function as a unified-signature algorithm.

    Two forms supported::

        @unified_algorithm                              # legacy bare form
        def my_algo(inputs, params): ...

        @unified_algorithm(name='borst_t4t5',           # registry form
                           signature='stateful',
                           description='T4/T5 motion detector')
        def my_algo(inputs, params, state): ...

    Either form sets ``fn._unified = True`` (required by FuncBlock). The
    registry form additionally registers the function under ``name`` in
    ``neurocircuitdesk.registry._ALGORITHMS`` so it can be looked up by
    string from a spec or chat tool call.
    """
    # Import here to avoid a circular import at module load (registry imports
    # from typing only, but blocks_exe is imported very early).
    from neurocircuitdesk.registry import register_algorithm

    def _wrap(f: Callable) -> Callable:
        f._unified = True
        register_algorithm(
            f, name=name, signature=signature,
            ports=ports, description=description,
        )
        return f

    # Called as bare decorator: @unified_algorithm
    if fn is not None and callable(fn):
        return _wrap(fn)

    # Called with kwargs: @unified_algorithm(name=...)
    return _wrap


# Regex shared with mlx_engine: matches fan-out MIMO output ports of the
# form '<channel>_<col_idx>' (channel must end in '_col').
_OUT_COL_PORT_RE = re.compile(r'^(.+_col)_(\d+)$')
_IN_COL_PORT_RE  = re.compile(r'^input_col_(\d+)$')


@unified_algorithm
def placeholder_passthrough(inputs, params):
    """Identity SISO passthrough. Useful for forwarding values between MCs."""
    return {'output': inputs['input']}

class Node:
    """
    Base class for executable nodes in the graph.
    Each node has input and output ports (Port objects),
    a params dict (configuration), 
    and a state dict (for stateful blocks).
    """

    def __init__(self, name: str):
        self.name = name
        self.inputs: Dict[str, Port] = {}
        self.outputs: Dict[str, Port] = {}
        self.params: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}

    def step(self, t: float, dt: float, feed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute outputs from inputs.
        Subclasses (e.g., FuncBlock, Division, Sum) must override this.
        """
        raise NotImplementedError(f"step() not implemented for {self.__class__.__name__}")

    def __repr__(self):
        return f"Node(name={self.name}, type={self.__class__.__name__})"


class Port:
    def __init__(
        self,
        name: str,
        direction: str, # "in" or "out",
        port_type: str = 'default', # e.g. 'numerator', 'denominator'
        dtype=float,
        shape=(),
        coord: tuple[float, float, float] = None  
    ):
        self.name = name
        self.direction = direction
        self.dtype = dtype
        self.shape = shape
        self.port_type = port_type
        self.coord = coord  # (x, y, z) position for visualization

    def __repr__(self):
        return f"Port(name={self.name}, dir={self.direction}, coord={self.coord})"

        
        

class Sum(Node):
    def __init__(self, name: str):
        super().__init__(name)
        self.outputs['output'] = Port('output', 'out')

    def add_input_port(self, pname: str, port_type: str = 'default'):
        self.inputs[pname] = Port(pname, 'in', port_type=port_type)

    def step(self, t, dt, feed):
        total = 0.0
        for v in feed.values():
            if v is not None:
                total += v
        return {'output': total}


class Aggregator(Node):
    """
    A node with one output and dynamically added input ports.
    Aggregation mode (sum, mean, product, subtract) is fixed at creation;
    input ports are added via add_input_port() when building the microcircuit.
    """
    VALID_MODES = ('sum', 'mean', 'product', 'subtract')

    def __init__(self, name: str, mode: str = 'sum'):
        super().__init__(name)
        if mode not in self.VALID_MODES:
            raise ValueError(f"Aggregator mode must be one of {self.VALID_MODES}, got '{mode}'.")
        self.mode = mode
        self.outputs['output'] = Port('output', 'out')

    def add_input_port(self, pname: str):
        self.inputs[pname] = Port(pname, 'in', port_type='default')

    def step(self, t: float, dt: float, feed: Dict[str, Any]) -> Dict[str, Any]:
        values = [v for v in feed.values() if v is not None]
        if not values:
            if self.mode == 'sum':
                result = 0.0
            elif self.mode == 'mean':
                result = 0.0
            elif self.mode == 'product':
                result = 1.0
            else:  # subtract
                result = 0.0
            return {'output': result}
        if self.mode == 'sum':
            result = sum(values)
        elif self.mode == 'mean':
            result = sum(values) / len(values)
        elif self.mode == 'product':
            result = 1.0
            for v in values:
                result *= v
        else:  # subtract: first value minus the rest
            result = values[0] - sum(values[1:])
        return {'output': float(result)}


class InputNode(Node):
    """
    A special node representing a single input port of a microcircuit.
    It has one output port named 'output'. Its step function is a passthrough.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.outputs['output'] = Port('output', 'out')

    def step(self, t, dt, feed):
        return {} 
    
class OutputNode(Node):
    """
    A special node representing a single output port of a microcircuit.
    It has one input port named 'input'. It acts as a sink.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.inputs['input'] = Port('input', 'in')

    def step(self, t, dt, feed):
        return {}
    
# stateless functional blocks (old)
# class FuncBlock(Node):
#     """
#     Stateless scalar function y = f(x; params)
#     Ports: input -> 'input', output -> 'output'
#     """
#     def __init__(self, name:str, f: Callable[[float, Dict[str,Any]], float], params=None):
#         super().__init__(name)
#         self.inputs['input']  = Port('input',  'in')
#         self.outputs['output']= Port('output', 'out')
#         self.f = f
#         self.params = params or {}

#     def step(self, t, dt, feed):
#         x = feed['input']  # scalar
#         return {'output': float(self.f(x, self.params))}


class FuncBlock(Node):
    """
    Wraps a user algorithm as an executable graph node.

    All algorithms use the unified backend-agnostic signature:

        stateless: f(inputs: dict, params: dict) -> dict
        stateful:  f(inputs: dict, params: dict, state: dict) -> (dict, dict)

    where ``inputs`` is keyed by the block's declared input port names
    (plus ``'neighbors'`` / ``'neighbor_mask'`` for MIMO blocks — see below)
    and the returned dict is keyed by the block's declared output ports
    (or by ``'<channel>_neighbors'`` for fan-out MIMO blocks — see below).

    Algorithms must be decorated with ``@unified_algorithm`` so the MLX
    engine can call them directly without a registered batched variant.

    MIMO inputs (blocks whose input ports are all ``input_col_<N>``):
        The scalar engine pre-assembles::

            inputs['neighbors']     : np.ndarray shape (n_nbrs,)
            inputs['neighbor_mask'] : np.ndarray shape (n_nbrs,) — all 1.0
                                      in scalar mode (no padding).

        Slot order follows the block's declared port order (template
        insertion order), which preserves geometry-sensitive orderings
        like the borst spiral.

    Fan-out MIMO outputs (blocks whose output ports are all
    ``<channel>_col_<N>``):
        A unified algorithm may return a single key
        ``'<channel>_neighbors'`` → ``np.ndarray`` shape ``(n_nbrs,)``,
        which FuncBlock unpacks into the individual per-slot output
        ports. Returning the per-slot ports directly also works.

    Auto-packed per-neighbor dict params:
        Any param whose value is a ``Dict[col_idx, float]`` is converted
        once at construction time into a ``(n_nbrs,)`` ``np.ndarray``
        aligned with the fan-out output slot order. The dict is
        overwritten by the packed array under the same key.
    """

    # Legacy registry kept as an escape hatch for hand-tuned batched
    # variants that outperform the unified implementation. If a batched
    # variant is registered AND the algorithm is not `@unified_algorithm`,
    # the MLX engine uses the registered variant. Otherwise the unified
    # function is called directly.
    _batched_registry: Dict = {}

    @classmethod
    def register_batched(cls, scalar_fn: Callable, batched_fn: Callable) -> None:
        """Register a batched MLX variant for a scalar algorithm function."""
        cls._batched_registry[id(scalar_fn)] = batched_fn

    @classmethod
    def get_batched(cls, scalar_fn: Callable) -> Optional[Callable]:
        """Return the registered batched variant, or None if not registered."""
        return cls._batched_registry.get(id(scalar_fn))

    def __init__(self, name: str, f: Callable, params=None,
                 input_names: Optional[List[str]] = None,
                 output_names: Optional[List[str]] = None,
                 stateless: bool = True):
        super().__init__(name)
        self.f = f
        self.params = params or {}
        self.state = {}
        self.stateless = stateless

        # Port names in declared order (== template insertion order).
        # Order matters for geometry-sensitive algorithms like borst.
        self.input_names = input_names or ['input']
        self.output_names = output_names or ['output']

        for pname in self.input_names:
            self.inputs[pname] = Port(pname, 'in')
        for pname in self.output_names:
            self.outputs[pname] = Port(pname, 'out')

        if not getattr(f, '_unified', False):
            # Fail loud — the unified signature is mandatory.
            raise TypeError(
                f"FuncBlock '{name}' wraps '{getattr(f, '__name__', f)}' which "
                "is not decorated with @unified_algorithm. All algorithms must "
                "use the unified signature "
                "f(inputs, params[, state]) → dict[, state].")

        # --- MIMO structure detection ---------------------------------------
        # is_mimo_in: every input port is 'input_col_<N>'
        self._is_mimo_in = (
            len(self.input_names) > 0
            and all(_IN_COL_PORT_RE.match(p) for p in self.input_names)
        )

        # is_mimo_out: every output port is '<channel>_col_<N>'
        mout = [_OUT_COL_PORT_RE.match(p) for p in self.output_names]
        self._is_mimo_out = len(mout) > 0 and all(m is not None for m in mout)

        if self._is_mimo_out:
            # Group output ports by channel, preserving per-channel slot
            # order (== declared order within the channel).
            channels: List[str] = []
            cols_per_channel: Dict[str, List[int]] = {}
            for p, m in zip(self.output_names, mout):
                ch  = m.group(1)
                col = int(m.group(2))
                if ch not in cols_per_channel:
                    channels.append(ch)
                    cols_per_channel[ch] = []
                cols_per_channel[ch].append(col)
            # Every channel must share the same slot order (reciprocity
            # with the input neighbor axis for blocks that are both
            # is_mimo_in and is_mimo_out).
            canonical = cols_per_channel[channels[0]]
            for ch in channels[1:]:
                if cols_per_channel[ch] != canonical:
                    raise ValueError(
                        f"FuncBlock '{name}': fan-out MIMO channel '{ch}' slot "
                        f"order {cols_per_channel[ch]} differs from '{channels[0]}' "
                        f"slot order {canonical}. Per-channel axis mismatch is "
                        "not supported.")
            self._out_channels = channels          # e.g. ['output_val_col', 'output_weight_col']
            self._out_slot_cols = canonical        # e.g. [41, 42, 43, ...]
        else:
            self._out_channels = []
            self._out_slot_cols = []

        # --- Neighbor-mask (scalar = all 1.0, no padding) ------------------
        if self._is_mimo_in:
            self._neighbor_mask = np.ones(len(self.input_names), dtype=np.float32)
        else:
            self._neighbor_mask = None

        # --- Lazy auto-pack of per-neighbor dict params --------------------
        # Any param whose value is a Dict[col_idx, float] will be converted
        # to a (n_nbrs,) np.ndarray aligned with the fan-out output slot
        # order on the FIRST scalar step() call. We cannot pack here in
        # __init__ because mlx_engine.compile_mlx() needs to see the
        # original dicts to build its own (N, max_out) packed tensors.
        self._dict_params_packed = False
        self._original_dict_params: Dict[str, dict] = {}

    def _pack_dict_params(self) -> None:
        """Convert any Dict[col_idx, float] param into an aligned np.ndarray.

        Called lazily on the first scalar step so mlx_engine.compile_mlx()
        sees the original dicts at compile time. Originals are preserved in
        ``_original_dict_params`` so compile_mlx() can still read them even
        after a scalar run.
        """
        if self._dict_params_packed:
            return
        if self._is_mimo_out:
            for pname, pval in list(self.params.items()):
                if isinstance(pval, dict):
                    self._original_dict_params[pname] = pval
                    self.params[pname] = np.array(
                        [float(pval.get(c, 0.0)) for c in self._out_slot_cols],
                        dtype=np.float32,
                    )
        self._dict_params_packed = True

    def get_original_param(self, pname: str) -> Any:
        """Return the original (pre-packed) value of a param.

        If the param was a Dict[col_idx, float] that got packed into an
        ndarray by the scalar engine, this returns the original dict so
        compile_mlx() can build its own packed tensors correctly.
        """
        return self._original_dict_params.get(pname, self.params.get(pname))

    # ----- Feed enrichment (scalar engine only) ---------------------------
    def _enrich_feed(self, feed: Dict[str, Any]) -> Dict[str, Any]:
        """Backfill missing ports with 0.0 and add neighbors/neighbor_mask."""
        # Ensure every declared port has a value (default 0.0 for unconnected).
        for pname in self.input_names:
            if pname not in feed:
                feed[pname] = 0.0

        if self._is_mimo_in:
            # Assemble the neighbor tensor in declared port order.
            feed['neighbors'] = np.array(
                [feed[p] for p in self.input_names],
                dtype=np.float32,
            )
            feed['neighbor_mask'] = self._neighbor_mask
        return feed

    # ----- Output unpacking (scalar engine only) --------------------------
    def _unpack_outputs(self, out: Dict[str, Any]) -> Dict[str, Any]:
        """Expand any '<channel>_neighbors' key into per-slot output ports.

        Only keys whose channel (after stripping ``_neighbors``) matches a
        declared fan-out output channel are unpacked. Other keys — even if
        they happen to end in ``_neighbors`` — pass through unchanged, so
        user-defined port names like ``'count_neighbors'`` are safe.
        """
        if not self._is_mimo_out:
            return out
        expanded: Dict[str, Any] = {}
        for key, value in out.items():
            if key.endswith('_neighbors'):
                channel = key[: -len('_neighbors')]
                if channel in self._out_channels:
                    # Genuine fan-out: unpack per-slot.
                    arr = np.asarray(value)
                    if arr.shape[-1] != len(self._out_slot_cols):
                        raise ValueError(
                            f"FuncBlock '{self.name}': '{key}' has last-axis "
                            f"length {arr.shape[-1]} but "
                            f"{len(self._out_slot_cols)} slots were declared.")
                    for slot, col in enumerate(self._out_slot_cols):
                        expanded[f'{channel}_{col}'] = float(arr[slot])
                    continue
            # Regular key (or _neighbors suffix that doesn't match a channel).
            expanded[key] = value
        return expanded

    def step(self, t, dt, feed):
        """Unified dispatch: f(inputs, params[, state]) → dict[, state]."""
        self._pack_dict_params()
        feed = self._enrich_feed(feed)
        if self.stateless:
            out = self.f(feed, self.params)
        else:
            out, self.state = self.f(feed, self.params, self.state)
        return self._unpack_outputs(out)



class Rectifier(Node):
    """
    A stateless node that performs positive or inverted rectification on its input.
    - 'on': output = max(0, x)
    - 'off': output = max(0, -x)
    """
    def __init__(self, name: str, mode: str = 'on'):
        super().__init__(name)
        # Define standard input/output ports
        self.inputs['input'] = Port('input', 'in')
        self.outputs['output'] = Port('output', 'out')
        
        if mode not in ['on', 'off']:
            raise ValueError("Rectifier mode must be 'on' or 'off'")
        self.mode = mode

    def step(self, t: float, dt: float, feed: Dict[str, Any]) -> Dict[str, Any]:
        """Applies the configured rectification logic."""
        x = feed.get('input', 0.0)
        
        if self.mode == 'on':
            # Standard positive rectification
            output = x if x > 0 else 0.0
        else: # 'off' mode
            # Inverted rectification: zero out positives, invert negatives
            output = -x if x < 0 else 0.0
            
        return {'output': output}
    
    
class TemporalDerivative(Node):
    """
    A stateful node that approximates a first-order derivative using a
    three-point central difference formula.
    """
    def __init__(self, name: str):
        super().__init__(name)
        # Define standard input/output ports
        self.inputs['input'] = Port('input', 'in')
        self.outputs['output'] = Port('output', 'out')
        
        # Initialize the state with a buffer (deque) to hold the last 3 values
        self.state['history'] = deque([0.0, 0.0, 0.0], maxlen=3)

    def step(self, t: float, dt: float, feed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates the history buffer and calculates the new derivative.
        """
        # Get the current input value
        x_t = feed.get('input', 0.0)
        
        # Add the new value to our history buffer.
        self.state['history'].append(x_t)
        
        # Unpack the history for the calculation
        # These represent x at (t-2), x at (t-1), and x at (t)
        x_t_minus_2, x_t_minus_1, x_t_current = self.state['history']
        
        # --- This is the three-point slope calculation ---
        # It's the central difference formula, which approximates
        # the derivative at the middle point (t-1).
        if dt > 0:
            slope = (x_t_current - x_t_minus_2) / (2 * dt)
        else:
            slope = 0.0
            
        return {'output': slope}
    
    
class Division(Node):
    """
    output = (sum of numerators) / (sum of denominators + eps)
    """
    def __init__(self, name:str, eps=1e-9):
        super().__init__(name)
        # This node no longer has fixed ports; they are added dynamically.
        self.outputs['output'] = Port('output', 'out')
        self.params['eps'] = eps
        self.port_groups = {
            'numerator': {'sum': [], 'mean': [], 'product': [], 'subtract': []},
            'denominator': {'sum': [], 'mean': [], 'product': [], 'subtract': []}
        }
        self.weighted_mean_pairs = {'numerator': {}, 'denominator': {}}
        self._unmatched_val_ports = {'numerator': set(), 'denominator': set()}

    def add_input_port(self, pname: str, 
                       port_type: str, 
                       aggregation: str = 'sum',
                       weight_port_name: Optional[str] = None):
        """
        Dynamically add an input port with a specified aggregation mode.

        Args:
            pname (str): The name of the port.
            port_type (str): 'numerator' or 'denominator'.
            aggregation (str): The method to combine inputs for this group.
                               Options: 'sum', 'mean', 'product', 'subtract', 'weighted_mean'. Defaults to 'sum'.
        """
        if port_type not in self.port_groups:
            raise ValueError("port_type must be 'numerator' or 'denominator'")
        valid_aggregations = list(self.port_groups[port_type].keys()) + ['weighted_mean']
        if aggregation not in valid_aggregations:
            raise ValueError(f"Unsupported aggregation mode: '{aggregation}' in {self.name}")
  
            
        self.inputs[pname] = Port(pname, 'in', port_type=port_type)

        # Handle the 'weighted_mean' case
        if aggregation == 'weighted_mean':
            if pname.endswith('_val'):
                self._unmatched_val_ports[port_type].add(pname)
            elif pname.endswith('_weight'):
                base_name = pname[:-7] # Remove '_weight'
                expected_val_port = f"{base_name}_val"
                if expected_val_port in self._unmatched_val_ports[port_type]:
                    self.weighted_mean_pairs[port_type][expected_val_port] = pname
                    self._unmatched_val_ports[port_type].remove(expected_val_port)
                else:
                    raise ValueError(f"Weight port '{pname}' was added before its value partner '{expected_val_port}' in {self.name}.")
            else:
                raise ValueError(f"Port '{pname}' with 'weighted_mean' aggregation must end in '_val' or '_weight' in {self.name}.")
        else:
            # Standard aggregations are added to their respective groups
            self.port_groups[port_type][aggregation].append(pname)
            
    def step(self, t, dt, feed):
        def calculate_total(port_type: str) -> float:
            """Helper function to calculate the total for numerator or denominator."""
            total = 0.0
            
            # --- Summation Group ---
            sum_ports = self.port_groups[port_type]['sum']
            if sum_ports:
                total += sum(feed.get(p, 0.0) for p in sum_ports)
            
            # --- Subtraction Group ---
            subtract_ports = self.port_groups[port_type]['subtract']
            if subtract_ports:
                total -= sum(feed.get(p, 0.0) for p in subtract_ports)

            # --- Mean (Average) Group ---
            mean_ports = self.port_groups[port_type]['mean']
            if mean_ports:
                mean_val = sum(feed.get(p, 0.0) for p in mean_ports) / len(mean_ports)
                total += mean_val

            # --- Product (Multiplication) Group ---
            product_ports = self.port_groups[port_type]['product']
            if product_ports:
                # Note: Product is multiplicative, so it's added to the total.
                product_val = 1.0
                for p in product_ports:
                    product_val *= feed.get(p, 1.0) # Default to 1.0 for multiplication
                total += product_val
            
            weighted_pairs = self.weighted_mean_pairs[port_type]
            if weighted_pairs:
                sum_of_products = 0.0
                sum_of_weights = 0.0
                for val_port, weight_port in weighted_pairs.items():
                    value = feed.get(val_port, 0.0)
                    weight = feed.get(weight_port, 0.0)
                    sum_of_products += value * weight
                    sum_of_weights += weight
                
                if sum_of_weights != 0:
                    total += sum_of_products / sum_of_weights
            return total

        num_total = calculate_total('numerator')
        den_total = calculate_total('denominator')
        
        eps = self.params['eps']
        return {'output': num_total / (den_total + np.sign(den_total) * eps if den_total != 0 else eps)}

    
class TemporalFilter(Node):
    """
    A stateful node that convolves its input history with a user-defined,
    static temporal filter. The filter must be set in the node's `params`
    before the simulation runs.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.inputs['input'] = Port('input', 'in')
        self.outputs['output'] = Port('output', 'out')
        
        # The buffer is minimal, as it will be resized when a filter is set.
        self.state['buffer'] = deque([0.0], maxlen=1)
        # The filter is now initialized to None. It MUST be set by the user.
        self.params['filter'] = None

    def step(self, t: float, dt: float, feed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates the buffer, checks for filter changes, and performs convolution.
        """
        current_filter = self.params.get('filter')
        if current_filter is None:
            raise ValueError(f"TemporalFilter node '{self.name}' has not been configured. "
                             "Please set a filter using `microcircuit.set_block_params()`.")
        
        current_filter = np.asarray(current_filter)
        filter_len = len(current_filter)
        
        # --- Dynamic Buffer Resizing ---
        # If the user changes the filter, resize the buffer to match.
        if len(self.state['buffer']) != filter_len:
            old_buffer = list(self.state['buffer'])
            fill_value = old_buffer[0] if old_buffer else 0.0
            new_buffer = deque(np.full(filter_len, fill_value), maxlen=filter_len)
            
            start_index = max(0, len(old_buffer) - filter_len)
            new_buffer.extend(old_buffer[start_index:])
            self.state['buffer'] = new_buffer

        x_t = feed.get('input', 0.0)
        self.state['buffer'].append(x_t)
        
        buffer_array = np.array(self.state['buffer'])

        convolved_array = convolve1d(
            buffer_array, 
            current_filter, 
            axis=0, 
            mode='nearest',
            origin=-(filter_len // 2)
        )
        
        return {'output': convolved_array[-1]}


# ---------------------------------------------------------------------------
# Batched MLX variants for built-in Node types (Division, Rectifier,
# TemporalFilter). These are engine-internal — user algorithms use the
# unified signature instead and do not need a separate batched variant.
# ---------------------------------------------------------------------------

def division_batched(inputs: Dict, params: Dict) -> Dict:
    """
    Batched MLX version of the Division node.

    Mirrors Division.step() on (N,)-shaped feed values. Fan-in is already
    summed upstream by the engine (same as the scalar Program), so per-port
    inputs arrive as single (N,) arrays.

    params : {
        'port_groups'        : {'numerator': {sum: [...], mean: [...],
                                              product: [...], subtract: [...]},
                                'denominator': {same}},
        'weighted_mean_pairs': {'numerator': {val_port: weight_port, ...},
                                'denominator': {same}},
        'eps'                : float,
    }
    """
    import mlx.core as mx

    eps          = params.get('eps', 1e-9)
    port_groups  = params['port_groups']
    wmean_pairs  = params['weighted_mean_pairs']

    any_v  = next(iter(inputs.values()))
    n_cols = any_v.shape[0]
    zeros  = mx.zeros((n_cols,))
    ones   = mx.ones((n_cols,))

    def calc_total(side: str):
        total = zeros
        groups = port_groups[side]

        for p in groups.get('sum', []):
            total = total + inputs.get(p, zeros)

        for p in groups.get('subtract', []):
            total = total - inputs.get(p, zeros)

        mean_ports = groups.get('mean', [])
        if mean_ports:
            acc = zeros
            for p in mean_ports:
                acc = acc + inputs.get(p, zeros)
            total = total + acc / float(len(mean_ports))

        prod_ports = groups.get('product', [])
        if prod_ports:
            acc = ones
            for p in prod_ports:
                acc = acc * inputs.get(p, ones)
            total = total + acc

        wpairs = wmean_pairs.get(side, {})
        if wpairs:
            sum_products = zeros
            sum_weights  = zeros
            for val_port, weight_port in wpairs.items():
                v = inputs.get(val_port, zeros)
                w = inputs.get(weight_port, zeros)
                sum_products = sum_products + v * w
                sum_weights  = sum_weights + w
            # Per-column safe divide; where sum_weights == 0, contribute 0.
            nonzero = sum_weights != 0
            safe_w  = mx.where(nonzero, sum_weights, ones)
            wm      = mx.where(nonzero, sum_products / safe_w, zeros)
            total   = total + wm

        return total

    num = calc_total('numerator')
    den = calc_total('denominator')

    # Scalar Division does:
    #   num / (den + sign(den)*eps  if den != 0  else  eps)
    nonzero  = den != 0
    den_safe = mx.where(nonzero, den + mx.sign(den) * eps,
                        mx.full((n_cols,), eps))
    return {'output': num / den_safe}


def rectifier_batched(inputs: Dict, params: Dict) -> Dict:
    """
    Batched MLX version of the Rectifier node.

    params : {'mode': 'on' | 'off'}
    inputs : {'input': (N,) mx.array}
    """
    import mlx.core as mx
    x = inputs['input']
    if params.get('mode', 'on') == 'on':
        return {'output': mx.maximum(x, 0.0)}
    else:
        return {'output': mx.maximum(-x, 0.0)}


def temporal_filter_batched(inputs: Dict, params: Dict, state: Dict):
    """
    Batched MLX version of the TemporalFilter node.

    Uses a ring buffer shifted one sample per call. With a filter of length F,
    the output equals dot(reverse(filter), buffer), which matches
    scipy.ndimage.convolve1d(buf, filter, mode='nearest', origin=-(F//2))[-1]
    when len(buf) == F and buf is zero-initialised (the warmup regime that
    the scalar TemporalFilter also uses on first resize).

    params : {'filter_rev': (F,) mx.array — already reversed, 'F_len': int}
    state  : {'buffer': (N, F) mx.array}  — allocated on first call
    """
    import mlx.core as mx

    filter_rev = params['filter_rev']     # (F,)
    F_len      = params['F_len']
    x          = inputs['input']          # (N,)
    n_cols     = x.shape[0]

    if 'buffer' not in state:
        state = {'buffer': mx.zeros((n_cols, F_len))}

    buf     = state['buffer']                                   # (N, F)
    new_buf = mx.concatenate([buf[:, 1:], x[:, None]], axis=1)  # shift-left
    out     = mx.sum(new_buf * filter_rev[None, :], axis=1)     # (N,)

    return {'output': out}, {'buffer': new_buf}
