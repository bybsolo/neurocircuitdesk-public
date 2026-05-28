from typing import List, Tuple, Dict, Optional, Callable, Any
from collections import defaultdict, OrderedDict
from functools import lru_cache
import plotly.graph_objects as go
import numpy as np
from neurocircuitdesk.blocks_exe import *
from neurocircuitdesk.microcircuit import MicroCircuit
from neurocircuitdesk.blocks_viz import Blocks
from neurocircuitdesk.registry import (
    get_template, template_name_of,
    get_algorithm, algorithm_name_of,
)
import json
import networkx as nx

class Program:
    """
    An executable, flattened graph representing the final COMPILED CIRCUIT ready for execution.
    Handles algebraic loops by applying a single time-step delay to feedback signals.
    """
    def __init__(self,
                 nodes: Dict[str, "Node"],
                 edges: List[Tuple[str, str, str, str]],
                 dag_schedule: List[str],
                 scc_schedule: List[str],
                 canvas_inputs: Dict[Tuple[str, str], Tuple[str, str]],
                 canvas_outputs: Dict[Tuple[str, str], Tuple[str, str]]):
        self.nodes = nodes
        self.edges = edges
        self.dag_schedule = dag_schedule
        self.scc_schedule = scc_schedule
        self.scc_nodes = set(scc_schedule)
        self.canvas_inputs = canvas_inputs
        self.canvas_outputs = canvas_outputs
        
        self.inputs = []
        self.outputs = []

        self._fan_in: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
        for s_n, s_p, d_n, d_p in edges:
            self._fan_in[(d_n, d_p)].append((s_n, s_p))
        
        self._node_inputs: Dict[str, List[str]] = {
            n: list(self.nodes[n].inputs.keys()) for n in (dag_schedule + scc_schedule)
        }
        
        self._values: Dict[Tuple[str, str], Any] = {}
        self._previous_values: Dict[Tuple[str, str], Any] = {} # For feedback loops

    def _execute_node(self, n: str, t: float, dt: float):
        node = self.nodes[n]
        feed = {}
        
        for p_name in node.inputs:
            k = (n, p_name)
            preds = self._fan_in.get(k, [])
            if not preds: continue

            total_input = 0.0
            for s_n, s_p in preds:
                # Use previous value if the predecessor is part of the feedback loop (SCC)
                source_dict = self._previous_values if s_n in self.scc_nodes else self._values
                v = source_dict.get((s_n, s_p), 0.0)
                total_input += v if v is not None else 0.0
            feed[p_name] = total_input

        out = node.step(t, dt, feed)
        for p_name, v in out.items():
            self._values[(n, p_name)] = v

    def run_step(self, t: float, dt: float, x: Dict[Tuple[str, str], Any]) -> Dict[Tuple[str, str], Any]:
        """
        Execution runtime at a given time step
        """
        self._previous_values = self._values.copy()
        self._values.clear()

        # feed external inputs using the public-to-internal mapping
        for pub_key, val in x.items():
            if pub_key in self.canvas_inputs:
                internal_key = self.canvas_inputs[pub_key]
                self._values[internal_key] = val

        # execute nodes in scheduled order
        for n in self.dag_schedule:
            self._execute_node(n, t, dt)
        for n in self.scc_schedule:
            self._execute_node(n, t, dt)

        # collect and map outputs from internal names back to public names
        y = {}
        for pub_key, internal_key in self.canvas_outputs.items():
            if internal_key in self._values:
                y[pub_key] = self._values[internal_key]
        return y
    
    def run_series(self, T: int, dt: float,
                   inputs_by_step: List[Dict[Tuple[str, str], Any]],
                   t0: float = 0.0) -> List[Dict[Tuple[str, str], Any]]:
        """
        Wraps step runtime run_step over T;
        Returns a list of output dicts.
        """
        out_series: List[Dict[Tuple[str, str], Any]] = []
        t = t0
        for k in range(T):
            y = self.run_step(t, dt, inputs_by_step[k])
            out_series.append(y)
            t += dt
            
        self.outputs = out_series
        return out_series
    
    def run_program(self, inputs, input_microcircuits, input_port_name = 'input_main'):
        """
        Wraps run_series explicitly with direct input.
        Specify which microcircuits take in global input
        """
        self.inputs = inputs
        inputs_by_step = []
        for k in range(inputs.shape[0]):
            feeds_for_step_k = {}
            for microcircuit in input_microcircuits:
                u = float(inputs[k, microcircuit.col_idx])
                feeds_for_step_k[(microcircuit.name, input_port_name)] = u
            inputs_by_step.append(feeds_for_step_k)
        return self.run_series(T=inputs.shape[0], dt=1/60.0, inputs_by_step=inputs_by_step, t0=0)
    
    def probe_result(self, microcircuits, port_name: str):
        """
        Probe results at nodes specified by microcircuit and port name.

        Always returns a ``(T, N_cols)`` array matching the program's input
        column axis, regardless of how densely ``microcircuits`` covers it:

        - **Dense types** (one MC per column, e.g. ``PR_col``, ``ONOFF_col``,
          ``MOTION_ON_col``): every column index in ``[0, N_cols)`` gets the
          corresponding MC's value.
        - **Sparse types** (e.g. ``LOOMING``, where MCs are placed only at
          tiling centres): cells without an MC remain ``0.0`` and cells at
          a centre col_idx hold that MC's output. The returned array shares
          its column axis with the dense outputs, so it can be fed straight
          into ``HexViz.plot_retinotopic`` without reshaping.

        Missing keys in the output stream are tolerated via ``dict.get`` and
        default to ``0.0`` — useful during a stateful block's warmup window.
        """
        if self.outputs == []:
            print("Program not executed!")
            return np.zeros_like(self.inputs)

        outs = np.zeros_like(self.inputs)
        n_cols = outs.shape[1] if outs.ndim > 1 else 0
        for k, yk in enumerate(self.outputs):
            for mc in microcircuits:
                if 0 <= mc.col_idx < n_cols:
                    outs[k, mc.col_idx] = float(yk.get((mc.name, port_name), 0.0))
        return outs
        

class Canvas:
    """
    Manages microcircuit creation, visualization, composition, and compilation.
    This is the main user-facing class for building circuits.
    """
    def __init__(self, w : int = 1000, h : int =1000,
                 col_json_path: Optional[str] = None,
                 interconnect_json_path: Optional[str] = None):
        self.microcircuits: Dict[str, "MicroCircuit"] = {}
        self.mc_types: Dict[str, List["MicroCircuit"]] = {}
        self.inter_microcircuit_edges: List[Tuple[str, str, str, str]] = []

        # Spec-round-trip metadata (populated as types/algos are added).
        # Recorded on first instance of each mc_type; assumes all instances
        # of the same type share template+z (true for current Canvas usage).
        self.mc_type_meta: Dict[str, Dict[str, Any]] = {}
        self.algo_bindings: List[Dict[str, Any]] = []
        # High-level wirings recorded by _Connector.{siso,miso,simo,mimo}.
        # Each entry is the kwargs of a single call, plus a 'pattern' key.
        # These are what `to_spec` emits and `from_spec` replays — the
        # low-level `inter_microcircuit_edges` are derived from these.
        self.wirings_meta: List[Dict[str, Any]] = []
        # Non-algorithm block-level param overrides (e.g., TemporalFilter
        # 'filter' kernel). Recorded by set_block_params_all.
        self.block_params_meta: List[Dict[str, Any]] = []

        # Record source JSON paths so to_spec/from_spec can round-trip the
        # retinotopy + interconnect graph references.
        self._col_json_path = col_json_path
        self._interconnect_json_path = interconnect_json_path
        self.fig = go.Figure()
        self.hexgrid_visible = False
        # Pending inter-microcircuit arrows accumulated for batch rendering.
        # Each entry: (src_coord, dst_coord, color).
        self._pending_arrows: List[Tuple] = []
        self._viz_dirty = True   # set True when microcircuits are added/wired
            
        self.fig.update_layout(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True),
                aspectmode='data',
            ),
            width= w,
            height= h,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        self.hex_lookup = {}
        self._hex_coords_id: List[int] = []  # col_idx → bio identity
        if col_json_path:
            try:
                with open(col_json_path, 'r') as f: data = json.load(f)
                hex_ids = np.array(data['hex_coords_id'])
                self._hex_coords_id = [int(h) for h in hex_ids]
                raw_coords = np.array(data['hex_coords'])
                hex_coords = np.array([-raw_coords[:, 1], raw_coords[:, 0]]).T
                self.hex_lookup = {int(hex_ids[i]): tuple(hex_coords[i])
                                   for i in range(len(hex_ids)) if int(hex_ids[i]) < 1000}
            except (FileNotFoundError, KeyError):
                print(f"Warning: Could not load or parse retinotopy file at {col_json_path}.")
                self.hex_lookup = {i: (np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)) for i in range(100)}

        if interconnect_json_path:
            try:
                with open(interconnect_json_path, 'r') as f: graph_data = json.load(f)
                self.G = nx.readwrite.json_graph.node_link_graph(graph_data)
            except (FileNotFoundError, KeyError):
                self.G = nx.Graph()
        else:
            self.G = nx.Graph()

        self._draw_hexgrid()

    def add_mc_type(self, mc_type: str):
        """
        Register a microcircuit type for quick access.
        Creates an entry in self.mc_types if it does not already exist.
        """
        if mc_type in self.mc_types:
            raise ValueError(f"Microcircuit type '{mc_type}' already exists.")
        else:
            self.mc_types[mc_type] = []

    def add_microcircuit(self, name: str, col_idx: int, x: float, y: float, z: float, template: Callable, **template_kwargs):
        """
        Create and position a microcircuit instance on the canvas.
        This is the core function for microcircuit instantiation.
        """
        if name in self.microcircuits:
            raise ValueError(f"MicroCircuit with name '{name}' already exists.")

        microcircuit = MicroCircuit(
            canvas=self, 
            name=name, 
            col_idx=col_idx, 
            x=x, y=y, z=z, 
            template=lambda m: template(m, **template_kwargs)
        )

        # Store the created microcircuit. Visualization is deferred to
        # _render_batched() — no per-instance Plotly traces are created here.
        self.microcircuits[name] = microcircuit
        self._viz_dirty = True
        return microcircuit

    def _col_idx_to_pos(self, col_idx: int, fallback=(0.0, 0.0)):
        """Map a simulation col_idx to (x, y) via hex_coords_id indirection."""
        if col_idx < len(self._hex_coords_id):
            bio_id = self._hex_coords_id[col_idx]
            if bio_id in self.hex_lookup:
                return self.hex_lookup[bio_id]
        return fallback

    def add_microcircuit_columnar(self, col_idx: int, z: float, mc_type: str,
                                   template, **template_kwargs):
        """
        Retonotopically add microcircuit instances, calls add_microcircuit.
        This function will create a microcircuit whose name is f"{mc_type}_{col_idx}".

        ``template`` may be a callable (back-compat) or a string name registered
        with the template registry. Templates passed by name (or by callable
        that was registered with ``@template``) populate ``mc_type_meta`` so
        the canvas can be round-tripped through ``to_spec`` / ``from_spec``.
        Unregistered callables work as before but produce a partial spec.
        """
        if isinstance(template, str):
            template_name = template
            template_fn = get_template(template)
        else:
            template_fn = template
            template_name = template_name_of(template)

        x, y = self._col_idx_to_pos(col_idx, fallback=(float('nan'), float('nan')))
        microcircuit = self.add_microcircuit(
            name = f"{mc_type}_{col_idx}",
            col_idx=col_idx,
            x=x, y=y, z=z,
            template=template_fn,
            **template_kwargs
        )
        if mc_type not in self.mc_types:
            raise ValueError(f"Microcircuit type '{mc_type}' is not registered.")
        self.mc_types[mc_type].append(microcircuit)

        # Record type-level metadata on first instance.
        if mc_type not in self.mc_type_meta:
            self.mc_type_meta[mc_type] = {
                'category': 'columnar',
                'z': z,
                'template': template_name,
                'template_params': dict(template_kwargs),
            }
        return microcircuit
    
    def add_microcircuit_intercolumnar(self, center_col_idx: int, z: float, mc_type: str,
                                       neighborhood: Dict, template, **template_kwargs):
        """
        A high-level helper to create a single, complex microcircuit with a receptive field.
        It calculates the geometric neighborhood and passes it to the microcircuit's template.

        ``template`` may be a callable or a registry name (see
        ``add_microcircuit_columnar`` for details). Full round-trip of
        intercolumnar layers also requires the caller to record the
        ``centers`` config used to generate ``neighborhood`` — Phase 2 adds
        a higher-level ``add_intercolumnar_layer`` helper that does both.
        """
        if isinstance(template, str):
            template_name = template
            template_fn = get_template(template)
        else:
            template_fn = template
            template_name = template_name_of(template)

        x, y = self._col_idx_to_pos(center_col_idx, fallback=(float('nan'), float('nan')))
        microcircuit = self.add_microcircuit(
            name = f"{mc_type}_{center_col_idx}",
            col_idx=center_col_idx,
            x=x, y=y, z=z,
            neighborhood=neighborhood,
            template=template_fn,
            **template_kwargs
        )
        if mc_type not in self.mc_types:
            raise ValueError(f"Microcircuit type '{mc_type}' is not registered.")
        self.mc_types[mc_type].append(microcircuit)

        if mc_type not in self.mc_type_meta:
            self.mc_type_meta[mc_type] = {
                'category': 'intercolumnar',
                'z': z,
                'template': template_name,
                'template_params': dict(template_kwargs),
                # centers + neighborhood_kernel filled by add_intercolumnar_layer
            }

        # Cache the neighborhood for later kernel-aware algorithm binding.
        if not hasattr(self, '_intercolumnar_neighborhoods'):
            self._intercolumnar_neighborhoods = {}
        self._intercolumnar_neighborhoods.setdefault(mc_type, {})[center_col_idx] = neighborhood

        return microcircuit

    def add_intercolumnar_layer(self, *,
                                mc_type: str,
                                template,
                                z: float,
                                centers_config: dict,
                                neighborhood_kernel: Optional[dict] = None,
                                template_kwargs: Optional[dict] = None) -> List["MicroCircuit"]:
        """
        High-level helper to add a full intercolumnar layer in one call.

        Runs ``graph_utils.calc_mimo_centers(**centers_config)`` to derive
        centers + neighborhoods, then creates an MC at each center. Records
        ``centers`` config and ``neighborhood_kernel`` in ``mc_type_meta``
        so the layer round-trips via ``to_spec`` / ``from_spec``.

        Parameters
        ----------
        mc_type : str
            Must already be registered via ``add_mc_type``.
        template : str or callable
            Template name (registry) or callable.
        z : float
            Z-plane for this layer.
        centers_config : dict
            Passed to ``graph_utils.calc_mimo_centers``. Recognised keys:
            ``limit, step, jump, num_rings, require_in_graph``.
        neighborhood_kernel : dict, optional
            Spec for per-instance kernel weighting (e.g.,
            ``{'type': 'gaussian', 'sigma': 0.85}``). Used later by
            ``bind_algorithm(..., kernel_param=...)``.
        template_kwargs : dict, optional
            Extra kwargs forwarded to the template (apart from
            ``neighborhood`` which is supplied automatically).

        Returns
        -------
        list of MicroCircuit instances created.
        """
        if mc_type not in self.mc_types:
            raise KeyError(f"mc_type {mc_type!r} not registered. "
                           f"Call add_mc_type first.")

        template_kwargs = dict(template_kwargs or {})

        # Stamp the type-level meta upfront so it captures centers config
        # even if no instances are created (sparse graphs).
        if isinstance(template, str):
            template_name = template
        else:
            template_name = template_name_of(template)
        self.mc_type_meta[mc_type] = {
            'category': 'intercolumnar',
            'z': z,
            'template': template_name,
            'template_params': dict(template_kwargs),
            'centers': dict(centers_config),
            'neighborhood_kernel': dict(neighborhood_kernel) if neighborhood_kernel else None,
        }

        centers = self.graph_utils.calc_mimo_centers(**centers_config)

        created = []
        for col_idx, neighborhood in centers.items():
            mc = self.add_microcircuit_intercolumnar(
                center_col_idx=col_idx,
                z=z,
                mc_type=mc_type,
                neighborhood=neighborhood,
                template=template,
                **template_kwargs,
            )
            created.append(mc)
        return created

    def connect_microcircuits(self, src_mod_name: str, src_port_name: str,
                              dst_mod_name: str, dst_port_name: str,
                              skip_viz: bool = False, **kwargs):
        """
        Create a logical connection between the public ports of two microcircuits.

        Parameters
        ----------
        skip_viz : bool, default False
            If True, only the logical edge is recorded (no visualization work).
            Use this for large connection loops (e.g. ring-3 neighbourhoods)
            where rendering individual arrows would be slow and unreadable.
            If False, the connection is queued for batch rendering; call
            flush_arrows() (or compile(), which does so automatically) to
            materialise all queued arrows as a single Scatter3d trace.
        """
        # Logical wiring — always recorded
        self.inter_microcircuit_edges.append(
            (src_mod_name, src_port_name, dst_mod_name, dst_port_name)
        )

        if not skip_viz:
            src_mod = self.microcircuits[src_mod_name]
            dst_mod = self.microcircuits[dst_mod_name]
            src_coord = src_mod.get_port_coord(src_port_name, 'output')
            dst_coord = dst_mod.get_port_coord(dst_port_name, 'input')
            color = kwargs.get('color', 'black')
            self._pending_arrows.append((src_coord, dst_coord, color))

    def flush_arrows(self) -> None:
        """
        Render all queued connection arrows as a single batched Scatter3d trace
        per colour group, then clear the queue.

        Called automatically by compile() and compile_mlx().  Call manually
        if you want to visualise the circuit before compiling.

        Batching replaces the previous approach of one curved_arrow trace per
        connection, which caused super-linear slowdown for large neighbourhoods.
        Straight-line segments are used (Bezier curves are indistinguishable at
        circuit scales where batching matters).
        """
        if not self._pending_arrows:
            return

        # Group by colour so different connection types stay visually distinct
        from collections import defaultdict as _dd
        by_color: Dict[str, Tuple[list, list, list]] = _dd(lambda: ([], [], []))

        for src, dst, color in self._pending_arrows:
            xs, ys, zs = by_color[color]
            xs += [src[0], dst[0], None]
            ys += [src[1], dst[1], None]
            zs += [src[2], dst[2], None]

        for color, (xs, ys, zs) in by_color.items():
            self.fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(color=color, width=1),
                hoverinfo='none',
                name=f'connections ({color})',
            ))

        self._pending_arrows.clear()

    def _render_batched(self) -> None:
        """Render all microcircuit blocks, ports, and intra-MC arrows as a
        small number of batched Plotly traces.

        Replaces the old approach of one ``go.Surface`` + ``go.Scatter3d``
        per block instance (~33k traces at N=1261) with one trace per visual
        category (~10–20 traces total).

        Called automatically by :meth:`show` and :meth:`to_html`.
        """
        if not self._viz_dirty:
            return

        from collections import defaultdict as _dd

        # Accumulate metadata across all microcircuits
        block_by_color: Dict[str, Tuple[list, list, list, list]] = \
            _dd(lambda: ([], [], [], []))     # x, y, z, hover
        port_x, port_y, port_z, port_text = [], [], [], []
        arrow_by_color: Dict[str, Tuple[list, list, list]] = \
            _dd(lambda: ([], [], []))         # x, y, z (with None seps)

        for mc in self.microcircuits.values():
            viz = mc.get_viz_metadata()

            for bm in viz['blocks']:
                bx, by, bz = bm['x'], bm['y'], bm['z']
                color = bm.get('color', 'cyan')
                xs, ys, zs, ht = block_by_color[color]
                xs.append(bx)
                ys.append(by)
                zs.append(bz)
                ht.append(bm.get('name', ''))

            for px, py, pz, label in viz['ports']:
                port_x.append(px)
                port_y.append(py)
                port_z.append(pz)
                port_text.append(label)

            for src, dst, color in viz['arrows']:
                xs, ys, zs = arrow_by_color[color]
                xs += [src[0], dst[0], None]
                ys += [src[1], dst[1], None]
                zs += [src[2], dst[2], None]

        # --- Emit batched traces ---

        # Blocks: one Scatter3d marker trace per colour
        for color, (xs, ys, zs, ht) in block_by_color.items():
            self.fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='markers',
                marker=dict(size=5, color=color, opacity=0.8),
                text=ht,
                hoverinfo='text',
                name=f'blocks ({color})',
            ))

        # Ports: single Scatter3d marker trace
        if port_x:
            self.fig.add_trace(go.Scatter3d(
                x=port_x, y=port_y, z=port_z,
                mode='markers',
                marker=dict(size=2, color='black', opacity=0.5),
                text=port_text,
                hoverinfo='text',
                name='ports',
            ))

        # Intra-MC arrows: one line trace per colour
        for color, (xs, ys, zs) in arrow_by_color.items():
            self.fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(color=color, width=1),
                hoverinfo='none',
                name=f'intra-MC arrows ({color})',
            ))

        self._viz_dirty = False

    def show(self, **kwargs):
        """Render batched visualization and display the figure."""
        self._render_batched()
        self.flush_arrows()
        return self.fig.show(**kwargs)

    def to_html(self, **kwargs):
        """Render batched visualization and return HTML string."""
        self._render_batched()
        self.flush_arrows()
        return self.fig.to_html(**kwargs)

    @property
    def graph_utils(self):
        """Provides access to graph utility functions."""
        if not hasattr(self, '_graph_utils_instance'):
            self._graph_utils_instance = _GraphUtils(self)
        return self._graph_utils_instance
    
    @property
    def connect_utils(self):
        """Provides access to the high-level connection API."""
        return _Connector(self)


    def compile(self) -> Program:
        """
        Compiles all microcircuits on the canvas into a single, flat, executable graph.
        This version creates direct connections between internal blocks of different microcircuits.
        """
        self.flush_arrows()   # materialise any queued viz arrows before compiling
        global_nodes: Dict[str, Node] = {}
        global_edges: List[Tuple[str, str, str, str]] = []
        canvas_inputs: Dict[Tuple[str, str], Tuple[str, str]] = {}
        canvas_outputs: Dict[Tuple[str, str], Tuple[str, str]] = {}

        # 1. Ingest all nodes and internal edges from each microcircuit
        for mod in self.microcircuits.values():
            mg = mod.emit_exec_graph()
            global_nodes.update(mg["nodes"])
            global_edges.extend(mg["edges"])

        # 2. Wire up the top-level (external) inputs and outputs of the canvas
        for mod in self.microcircuits.values():
            for pub_name in mod.input_ports:
                # The public input (mod.name, pub_name) is mapped to the output of its internal InputNode
                canvas_inputs[(mod.name, pub_name)] = (f"{mod.name}/{pub_name}_INPUT", 'output')
            for pub_name, (block_id, port_name) in mod.output_ports.items():
                # The public output is mapped directly to the internal source block and port
                canvas_outputs[(mod.name, pub_name)] = (f"{mod.name}/{block_id}", port_name)

        # 3. Create direct inter-microcircuit connections, bypassing the Input/Output nodes
        for s_mod_n, s_port_n, d_mod_n, d_port_n in self.inter_microcircuit_edges:
            src_mod = self.microcircuits[s_mod_n]
            dst_mod = self.microcircuits[d_mod_n]

            # Find the true internal source of the output port
            s_block_id, s_internal_port = src_mod.output_ports[s_port_n]
            fq_source_node = f"{s_mod_n}/{s_block_id}"

            # Find all internal destinations for the input port (handles fan-out)
            destinations = dst_mod.input_ports[d_port_n]
            for d_block_id, d_internal_port in destinations:
                fq_dest_node = f"{d_mod_n}/{d_block_id}"
                # Add a direct edge from the internal source to the internal destination
                global_edges.append((fq_source_node, s_internal_port, fq_dest_node, d_internal_port))

        # 4. Schedule the final, flattened graph for execution
        dag_schedule, scc_schedule = self._schedule_graph(global_nodes, global_edges)
        
        if scc_schedule:
            print(f"Algebraic loop detected with {len(scc_schedule)} nodes. Applying single time-step delay.")

        return Program(
            nodes=global_nodes, edges=global_edges,
            dag_schedule=dag_schedule, scc_schedule=scc_schedule,
            canvas_inputs=canvas_inputs, canvas_outputs=canvas_outputs
        )

    @staticmethod
    def _schedule_graph(nodes: Dict[str, "Node"], edges: List[Tuple[str, str, str, str]]) -> Tuple[List[str], List[str]]:
        """
        Plan the correct execution order and identify feedback loops using NX
        Return a nx.DiGraph object
        Uses a robust method to correctly partition the graph into a DAG and
        one or more Strongly Connected Components (SCCs) for feedback loops.
        """
        # build a graph representation for networkx
        G = nx.DiGraph()
        G.add_nodes_from(nodes.keys())
        unique_edges = set((s_n, d_n) for s_n, _, d_n, _ in edges if s_n in nodes and d_n in nodes)
        G.add_edges_from(unique_edges)

        # find all nodes that are part of a true cycle (not just a single node)
        scc_nodes = {
            node
            for component in nx.strongly_connected_components(G)
            if len(component) > 1 or G.has_edge(list(component)[0], list(component)[0])
            for node in component
        }
        
        # all other nodes are part of the DAG; topologically sort them.
        # or we can create a subgraph of only DAG nodes to sort it correctly.
        dag_graph = G.subgraph([n for n in nodes if n not in scc_nodes])
        try:
            dag_schedule = list(nx.topological_sort(dag_graph))
        except nx.NetworkXUnfeasible:
             raise RuntimeError("Execution graph has a cycle that was not correctly identified as an SCC.")

        # SCC schedule can be sorted for deterministic execution (optional)
        scc_schedule = sorted(list(scc_nodes))
        
        return dag_schedule, scc_schedule

    def _draw_hexgrid(self):
        if not self.hex_lookup: return
        coords = np.array(list(self.hex_lookup.values()))
        ids = list(self.hex_lookup.keys())
        self.fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=np.zeros(len(coords)),
            mode='markers', name='hexgrid', visible=self.hexgrid_visible,
            marker=dict(size=2, color='gray'), hoverinfo='text',
            hovertext=[f'col_{i}' for i in ids]
        ))
    
    def toggle_hexgrid(self, visible: bool):
        self.hexgrid_visible = visible
        self.fig.for_each_trace(
            lambda trace: trace.update(visible=self.hexgrid_visible) if trace.name == 'hexgrid' else ()
        )

    def compile_mlx(self) -> "VectorizedProgram":
        """
        Compile the circuit into a VectorizedProgram for MLX execution.

        Calls compile() internally to obtain the scheduled graph, then wraps
        it in a VectorizedProgram that executes all column instances of each
        node kind as a single batched MLX array operation.

        Returns
        -------
        VectorizedProgram
            Call vprog.run_mlx(inputs, output_group='..') to execute.

        Notes
        -----
        Batched algorithm variants must be registered with
        FuncBlock.register_batched(scalar_fn, batched_fn) *before* this call.
        Unregistered FuncBlocks emit zeros and print a warning.
        """
        try:
            from neurocircuitdesk.mlx_engine import VectorizedProgram
        except ImportError as e:
            raise ImportError(
                "compile_mlx() requires the optional `mlx` dependency. "
                "Install with `pip install -e './neurocircuitdesk[mlx]'`."
            ) from e
        program = self.compile()
        vprog   = VectorizedProgram.from_program(program, self)

        # Warn about any FuncBlock groups with no registered batched function
        missing = [gk for gk, g in vprog.groups.items()
                   if g.fn is None and gk not in ('',)]
        if missing:
            print(f"Warning: no batched function registered for group(s): {missing}")
            print("  These groups will output zeros. Call FuncBlock.register_batched() before compile_mlx().")

        return vprog

    # ── Algorithm binding (spec-aware) ──────────────────────────────────

    def bind_algorithm(self, mc_type: str, block: str, algo,
                       params: Optional[dict] = None,
                       *, kernel_param: Optional[str] = None):
        """
        Bind an algorithm to a block on every instance of ``mc_type``, and
        record the binding for ``to_spec``.

        ``algo`` may be a registry name (str) or a ``@unified_algorithm``
        callable. Binding a string is preferred — callables are accepted
        for backwards compatibility but only round-trip cleanly if they
        were registered with ``@unified_algorithm(name=...)``.

        Re-binding the same ``(mc_type, block)`` overwrites the previous
        binding.

        Parameters
        ----------
        kernel_param : str, optional
            Name of a param that should be computed *per-instance* from the
            MC's neighborhood + the mc_type's ``neighborhood_kernel`` spec.
            Currently supports kernel ``type='gaussian'`` which produces a
            ``Dict[col_idx, float]`` value. Only valid for intercolumnar
            types that were created via ``add_intercolumnar_layer`` with
            ``neighborhood_kernel`` set.
        """
        if isinstance(algo, str):
            algo_name = algo
            algo_fn = get_algorithm(algo)
        else:
            algo_fn = algo
            algo_name = algorithm_name_of(algo)

        if mc_type not in self.mc_types:
            raise KeyError(
                f"mc_type {mc_type!r} not registered. "
                f"Known: {sorted(self.mc_types.keys())}"
            )

        for mc in self.mc_types[mc_type]:
            mc.set_block_func(block, algo_fn, params)

        if kernel_param is not None:
            kernel_spec = (self.mc_type_meta.get(mc_type, {})
                                            .get('neighborhood_kernel'))
            if kernel_spec is None:
                raise ValueError(
                    f"bind_algorithm(kernel_param={kernel_param!r}) requires "
                    f"mc_type {mc_type!r} to have a neighborhood_kernel set "
                    f"(via add_intercolumnar_layer)."
                )
            nbhds = getattr(self, '_intercolumnar_neighborhoods', {}).get(mc_type, {})
            for mc in self.mc_types[mc_type]:
                nbhd = nbhds.get(mc.col_idx, {})
                kval = _compute_kernel_value(kernel_spec, nbhd)
                mc.set_block_params(block, {kernel_param: kval})

        # Replace existing binding for the same (mc_type, block) if any.
        self.algo_bindings = [
            b for b in self.algo_bindings
            if not (b['mc_type'] == mc_type and b['block'] == block)
        ]
        binding = {
            'mc_type': mc_type,
            'block': block,
            'algo': algo_name,
            'params': dict(params or {}),
        }
        if kernel_param is not None:
            binding['kernel_param'] = kernel_param
        self.algo_bindings.append(binding)

    def set_block_params_all(self, mc_type: str, block: str, params: dict) -> None:
        """Set ``params`` on ``block`` for every instance of ``mc_type``.

        Used for non-algorithm blocks (TemporalFilter, Rectifier, etc.) that
        need static numerical params. Records the call in ``block_params_meta``
        so it round-trips through ``to_spec`` / ``from_spec``. NumPy arrays
        in ``params`` are stored as Python lists in the spec record but
        passed through unchanged to the executing block.
        """
        if mc_type not in self.mc_types:
            raise KeyError(f"mc_type {mc_type!r} not registered.")

        for mc in self.mc_types[mc_type]:
            mc.set_block_params(block, params)

        spec_params = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in (params or {}).items()
        }
        self.block_params_meta = [
            b for b in self.block_params_meta
            if not (b['mc_type'] == mc_type and b['block'] == block)
        ]
        self.block_params_meta.append({
            'mc_type': mc_type, 'block': block, 'params': spec_params,
        })

    # ── Spec round-trip ─────────────────────────────────────────────────

    def to_spec(self) -> dict:
        """Serialise this Canvas to a canonical JSON-able dict.

        Round-trip stable for circuits built via ``add_microcircuit_columnar``
        + ``bind_algorithm`` with registered templates/algorithms. Wirings
        and intercolumnar centers metadata are Phase 2 — they appear as
        empty lists / missing keys in v1 specs and are tolerated by
        ``from_spec``.
        """
        n_cols = (max((mc.col_idx for mc in self.microcircuits.values()), default=-1) + 1)
        return {
            'version': 1,
            'canvas': {
                'col_json': self._col_json_path,
                'graph_json': self._interconnect_json_path,
                'n_cols': n_cols,
            },
            'mc_types': [
                {'name': name, **meta}
                for name, meta in self.mc_type_meta.items()
            ],
            'algorithms': list(self.algo_bindings),
            'block_params': list(self.block_params_meta),
            'wirings': list(self.wirings_meta),
        }

    @classmethod
    def from_spec(cls, spec: dict, *,
                  col_json_path: Optional[str] = None,
                  graph_json_path: Optional[str] = None) -> 'Canvas':
        """Reconstruct a Canvas from a spec dict.

        Path overrides take precedence over the spec's recorded paths, so a
        spec produced on one machine can be loaded on another by passing
        explicit paths. If neither is given, the recorded path is used.
        """
        cj = col_json_path if col_json_path is not None else spec.get('canvas', {}).get('col_json')
        gj = graph_json_path if graph_json_path is not None else spec.get('canvas', {}).get('graph_json')
        n_cols = spec['canvas']['n_cols']

        cv = cls(col_json_path=cj, interconnect_json_path=gj)

        # 1. Materialise mc_types (columnar + intercolumnar).
        for tdef in spec.get('mc_types', []):
            tname = tdef['name']
            if tname not in cv.mc_types:
                cv.add_mc_type(tname)

            if tdef['category'] == 'columnar':
                for col_idx in range(n_cols):
                    cv.add_microcircuit_columnar(
                        col_idx=col_idx,
                        z=tdef['z'],
                        mc_type=tname,
                        template=tdef['template'],
                        **(tdef.get('template_params') or {}),
                    )
            elif tdef['category'] == 'intercolumnar':
                centers_config = tdef.get('centers')
                if centers_config is None:
                    # Pre-Phase-2 specs may lack centers; skip the layer.
                    print(f"Warning: intercolumnar mc_type {tname!r} has no "
                          f"centers config; skipping instance creation.")
                    continue
                cv.add_intercolumnar_layer(
                    mc_type=tname,
                    template=tdef['template'],
                    z=tdef['z'],
                    centers_config=dict(centers_config),
                    neighborhood_kernel=tdef.get('neighborhood_kernel'),
                    template_kwargs=tdef.get('template_params'),
                )

        # 2. Bind algorithms (including kernel-aware bindings).
        for adef in spec.get('algorithms', []):
            cv.bind_algorithm(
                mc_type=adef['mc_type'],
                block=adef['block'],
                algo=adef['algo'],
                params=adef.get('params'),
                kernel_param=adef.get('kernel_param'),
            )

        # 3. Apply non-algorithm block params (e.g., temporal filters).
        for bp in spec.get('block_params', []):
            cv.set_block_params_all(
                mc_type=bp['mc_type'],
                block=bp['block'],
                params=bp.get('params') or {},
            )

        # 4. Replay wirings via connect_utils.
        cv._apply_wirings(spec.get('wirings', []), n_cols=n_cols)

        return cv

    def _apply_wirings(self, wirings: List[dict], *, n_cols: int) -> None:
        """Replay deduped wiring records by iterating over per-instance loops.

        For each wiring entry, the appropriate connect_utils method is
        called with the per-instance index/centers re-derived from the
        canvas state.
        """
        for w in wirings:
            pattern = w['pattern']
            src, src_port = w['src'], w['src_port']
            dst, dst_port = w['dst'], w['dst_port']

            if pattern == 'siso':
                for col_idx in range(n_cols):
                    self.connect_utils.siso(src, src_port, dst, dst_port,
                                            col_idx=col_idx, skip_viz=True)

            elif pattern == 'miso':
                num_rings = w['num_rings']
                for col_idx in range(n_cols):
                    self.connect_utils.miso(src, src_port, dst, dst_port,
                                            col_idx=col_idx, num_rings=num_rings,
                                            skip_viz=True)

            elif pattern == 'simo':
                num_rings = w['num_rings']
                for col_idx in range(n_cols):
                    self.connect_utils.simo(src, src_port, dst, dst_port,
                                            col_idx=col_idx, num_rings=num_rings,
                                            skip_viz=True)

            elif pattern == 'mimo':
                anchor = w['anchor']
                # MIMO is centred on the *intercolumnar* mc_type's centres.
                if anchor == 'dst_center':
                    centers = list(self._intercolumnar_neighborhoods.get(dst, {}).keys())
                    for c in centers:
                        cols = list(self._intercolumnar_neighborhoods[dst][c].keys())
                        self.connect_utils.mimo(src, src_port, dst, dst_port,
                                                dst_center_col_idx=c, cols=cols,
                                                skip_viz=True)
                elif anchor == 'src_center':
                    centers = list(self._intercolumnar_neighborhoods.get(src, {}).keys())
                    for c in centers:
                        cols = list(self._intercolumnar_neighborhoods[src][c].keys())
                        self.connect_utils.mimo(src, src_port, dst, dst_port,
                                                src_center_col_idx=c, cols=cols,
                                                skip_viz=True)
                else:
                    raise ValueError(f"unknown mimo anchor {anchor!r}")

            else:
                raise ValueError(f"unknown wiring pattern {pattern!r}")

    def summary(self) -> str:
        """Compact text view of canvas state — for LLM tool results.

        Format::

            Canvas(n_cols=547, mc_instances=547)
              Types:    PR_col(547,z=1.3,tpl=pr_dnp)
              Algos:    PR_col.T1<-poly2_T1, PR_col.T2<-poly2_T2
              Wirings:  0 declared
        """
        lines = []
        n_cols = max((mc.col_idx for mc in self.microcircuits.values()), default=-1) + 1
        lines.append(
            f"Canvas(n_cols={n_cols}, mc_instances={len(self.microcircuits)})"
        )

        if self.mc_types:
            type_strs = []
            for tname, mcs in self.mc_types.items():
                meta = self.mc_type_meta.get(tname, {})
                z = meta.get('z')
                tpl = meta.get('template') or '?'
                z_str = f"{z}" if z is not None else "?"
                type_strs.append(f"{tname}({len(mcs)},z={z_str},tpl={tpl})")
            lines.append(f"  Types:    {', '.join(type_strs)}")

        if self.algo_bindings:
            binds = [f"{b['mc_type']}.{b['block']}<-{b['algo']}"
                     for b in self.algo_bindings]
            lines.append(f"  Algos:    {', '.join(binds)}")

        if self.wirings_meta:
            wires = [f"{w['src']}.{w['src_port']}->{w['dst']}.{w['dst_port']}({w['pattern']})"
                     for w in self.wirings_meta]
            lines.append(f"  Wirings:  {len(self.wirings_meta)} types ({len(self.inter_microcircuit_edges)} edges)")
            # Show first few wirings inline if compact
            if len(wires) <= 6:
                for w in wires:
                    lines.append(f"            {w}")
        else:
            lines.append(f"  Wirings:  0 declared")

        return '\n'.join(lines)

    def diff(self, other: 'Canvas | dict') -> str:
        """Human-readable spec diff between this canvas and another.

        ``other`` may be a Canvas or a spec dict. Returns a multi-line
        string listing added/removed mc_types, changed algorithm
        params, and added/removed wirings. Designed for chat output —
        clear at a glance, not a structured diff format.
        """
        a = self.to_spec()
        b = other.to_spec() if isinstance(other, Canvas) else other

        lines: List[str] = []

        # mc_types
        a_types = {t['name']: t for t in a.get('mc_types', [])}
        b_types = {t['name']: t for t in b.get('mc_types', [])}
        for name in sorted(set(a_types) | set(b_types)):
            if name not in b_types:
                lines.append(f"+ mc_type {name} (in self only)")
            elif name not in a_types:
                lines.append(f"- mc_type {name} (in other only)")
            elif a_types[name] != b_types[name]:
                lines.append(f"~ mc_type {name} changed")

        # algorithms
        def _algo_key(d): return (d['mc_type'], d['block'])
        a_algos = {_algo_key(d): d for d in a.get('algorithms', [])}
        b_algos = {_algo_key(d): d for d in b.get('algorithms', [])}
        for key in sorted(set(a_algos) | set(b_algos)):
            if key not in b_algos:
                lines.append(f"+ algo {key[0]}.{key[1]} = {a_algos[key]['algo']}")
            elif key not in a_algos:
                lines.append(f"- algo {key[0]}.{key[1]} (was {b_algos[key]['algo']})")
            elif a_algos[key] != b_algos[key]:
                old_p = b_algos[key].get('params', {})
                new_p = a_algos[key].get('params', {})
                lines.append(
                    f"~ algo {key[0]}.{key[1]}: "
                    f"params {old_p} -> {new_p}"
                )

        # wirings
        def _wire_key(d): return (d['pattern'], d['src'], d['src_port'],
                                  d['dst'], d['dst_port'])
        a_w = {_wire_key(d): d for d in a.get('wirings', [])}
        b_w = {_wire_key(d): d for d in b.get('wirings', [])}
        for key in sorted(set(a_w) | set(b_w)):
            if key not in b_w:
                lines.append(f"+ wiring {key[1]}.{key[2]} -> {key[3]}.{key[4]} ({key[0]})")
            elif key not in a_w:
                lines.append(f"- wiring {key[1]}.{key[2]} -> {key[3]}.{key[4]} ({key[0]})")

        if not lines:
            return "(no differences)"
        return '\n'.join(lines)

    def save(self, title: str = 'canvas_view'):
        self._render_batched()
        self.flush_arrows()
        self.fig.write_html(title + '.html', full_html=True, include_plotlyjs='cdn')

    def gen_flat_diagram(self, **kwargs):
        """Render a flat 2D circuit schematic.

        Thin wrapper around :func:`neurocircuitdesk.diagram2d.gen_flat_diagram`
        that renders an illustrative slice of the canvas (default 3 columns)
        with each MC's internal block structure preserved and inter-stage
        connectivity collapsed to one channel-per-slot. See
        :class:`DiagramOptions` for keyword arguments.
        """
        from neurocircuitdesk.diagram2d import gen_flat_diagram
        return gen_flat_diagram(self, **kwargs)

def _compute_kernel_value(kernel_spec: dict, neighborhood: dict) -> dict:
    """Compute a per-neighbor kernel weighting from a spec + neighborhood.

    ``neighborhood`` maps ``col_idx -> distance_in_rings`` (as produced by
    ``GraphUtils.get_neighbors_in_rings``).

    Returns a ``Dict[col_idx, float]`` of kernel weights.

    Supported kernel types:
      - ``{'type': 'gaussian', 'sigma': float}``
    """
    ktype = kernel_spec['type']
    if ktype == 'gaussian':
        sigma = float(kernel_spec['sigma'])
        return {col: float(np.exp(-0.5 * (dist / sigma) ** 2))
                for col, dist in neighborhood.items()}
    raise ValueError(f"unknown neighborhood_kernel type {ktype!r}")


class _Connector:
    """
    A helper class providing a high-level WRAPPER API for connecting groups of microcircuits.
    (Prob should build on this idea more)

    Each wiring method records its (src_type, src_port, dst_type, dst_port,
    pattern, ...) tuple in ``canvas.wirings_meta`` on first invocation;
    subsequent calls with the same tuple are dedup'd. This lets per-column
    loops (the common usage pattern) collapse to a single spec entry that
    ``Canvas.from_spec`` can replay.
    """
    def __init__(self, canvas: Canvas):
        self._canvas = canvas

    def _record_wiring(self, *, pattern: str,
                       src: str, src_port: str,
                       dst: str, dst_port: str, **extras):
        """Append a deduped wiring entry to canvas.wirings_meta.

        Dedup key is (pattern, src, src_port, dst, dst_port, sorted extras).
        """
        # Filter out keys that are per-instance (col_idx, src_center_col_idx,
        # dst_center_col_idx, cols, skip_viz) — these get re-derived by
        # from_spec, not stored.
        _PER_INSTANCE = {'col_idx', 'src_center_col_idx', 'dst_center_col_idx',
                         'cols', 'skip_viz'}
        record_extras = {k: v for k, v in extras.items() if k not in _PER_INSTANCE}

        entry = {
            'pattern': pattern,
            'src': src, 'src_port': src_port,
            'dst': dst, 'dst_port': dst_port,
            **record_extras,
        }
        # Dedup
        for existing in self._canvas.wirings_meta:
            if existing == entry:
                return
        self._canvas.wirings_meta.append(entry)

    def siso(self, src_mc_type: str, src_port: str,
                   dst_mc_type: str, dst_port: str,
                   col_idx: int, skip_viz: bool = False, **kwargs):
        """
        Connects a Single-Input to a Single-Output at the same column index.
        This connects f"{src_mc_type}_{col_idx}" -> f"{dst_mc_type}_{col_idx}".
        """
        self._record_wiring(pattern='siso',
                            src=src_mc_type, src_port=src_port,
                            dst=dst_mc_type, dst_port=dst_port,
                            **kwargs)
        src_name = f"{src_mc_type}_{col_idx}"
        dst_name = f"{dst_mc_type}_{col_idx}"
        if src_name in self._canvas.microcircuits and dst_name in self._canvas.microcircuits:
            self._canvas.connect_microcircuits(src_name, src_port, dst_name, dst_port,
                                               skip_viz=skip_viz, **kwargs)
        else:
            print(f"Warning: Skipping SISO connection for col {col_idx}. Missing {src_name} or {dst_name}.")

    def miso(self, src_mc_type: str, src_port: str,
                   dst_mc_type: str, dst_port: str,
                   col_idx: int, num_rings: int, skip_viz: bool = False, **kwargs):
        """
        Connects Multiple-Inputs to a Single-Output.
        The inputs are all microcircuits of src_mc_type within num_rings of col_idx.
        The output is the single microcircuit at col_idx.
        """
        self._record_wiring(pattern='miso',
                            src=src_mc_type, src_port=src_port,
                            dst=dst_mc_type, dst_port=dst_port,
                            num_rings=num_rings, **kwargs)
        dst_name = f"{dst_mc_type}_{col_idx}"
        if dst_name not in self._canvas.microcircuits:
            return
        neighbor_cols = self._canvas.graph_utils.local_order(
            col_idx, num_rings, require_in_graph=False)
        for neighbor_col in neighbor_cols:
            src_name    = f"{src_mc_type}_{neighbor_col}"
            dst_port_col = f"{dst_port}_{neighbor_col}"
            if src_name in self._canvas.microcircuits:
                self._canvas.connect_microcircuits(src_name, src_port, dst_name, dst_port_col,
                                                   skip_viz=skip_viz, **kwargs)

    def simo(self, src_mc_type: str, src_port: str,
                   dst_mc_type: str, dst_port: str,
                   col_idx: int, num_rings: int, skip_viz: bool = False, **kwargs):
        """
        Connects a Single-Input to Multiple-Outputs.
        The input is the single microcircuit at col_idx.
        The outputs are all microcircuits of dst_mc_type within num_rings of col_idx.
        """
        self._record_wiring(pattern='simo',
                            src=src_mc_type, src_port=src_port,
                            dst=dst_mc_type, dst_port=dst_port,
                            num_rings=num_rings, **kwargs)
        src_name = f"{src_mc_type}_{col_idx}"
        if src_name not in self._canvas.microcircuits:
            return
        neighbor_cols = self._canvas.graph_utils.local_order(
            col_idx, num_rings, require_in_graph=False)
        for neighbor_col in neighbor_cols:
            dst_name     = f"{dst_mc_type}_{neighbor_col}"
            src_port_col = f"{src_port}_{neighbor_col}"
            if dst_name in self._canvas.microcircuits:
                self._canvas.connect_microcircuits(src_name, src_port_col, dst_name, dst_port,
                                                   skip_viz=skip_viz, **kwargs)

    def mimo(
        self,
        src_mc_type: str, src_port: str,
        dst_mc_type: str, dst_port: str,
        src_center_col_idx: int = None, dst_center_col_idx: int = None,
        cols: list = None,
        skip_viz: bool = False,
        **kwargs):
        if (src_center_col_idx is None) == (dst_center_col_idx is None):
            raise ValueError("Exactly one of src_center_col_idx or dst_center_col_idx must be not None.")

        anchor = 'src_center' if src_center_col_idx is not None else 'dst_center'
        self._record_wiring(pattern='mimo',
                            src=src_mc_type, src_port=src_port,
                            dst=dst_mc_type, dst_port=dst_port,
                            anchor=anchor, **kwargs)

        if src_center_col_idx is not None:
            src_name = f"{src_mc_type}_{src_center_col_idx}"
            if src_name not in self._canvas.microcircuits:
                return
            for idx in cols:
                dst_name = f"{dst_mc_type}_{idx}"
                if dst_name in self._canvas.microcircuits:
                    self._canvas.connect_microcircuits(
                        src_name, f"{src_port}_{idx}", dst_name, dst_port,
                        skip_viz=skip_viz, **kwargs)

        elif dst_center_col_idx is not None:
            dst_name = f"{dst_mc_type}_{dst_center_col_idx}"
            if dst_name not in self._canvas.microcircuits:
                return
            for idx in cols:
                src_name = f"{src_mc_type}_{idx}"
                if src_name in self._canvas.microcircuits:
                    self._canvas.connect_microcircuits(
                        src_name, src_port, dst_name, f"{dst_port}_{idx}",
                        skip_viz=skip_viz, **kwargs)

@lru_cache(maxsize=None)
def _local_order_pure(n: int, num_rings: int) -> tuple:
    """
    Cached, pure-geometry local ordering around spiral index n.

    Returns a tuple of spiral indices:
      (center, ring1_cw_from_north..., ring2_cw_from_north..., ...)

    No graph membership check — the result depends only on hex arithmetic,
    so it is safe to cache at module level and share across all Canvas instances.
    """
    def ring_start(k: int) -> int:
        if k == 0:
            return 0
        return 1 + 3 * k * (k - 1)

    def index_to_axial(idx: int) -> tuple:
        if idx == 0:
            return (0, 0)
        k = 1
        while ring_start(k + 1) <= idx:
            k += 1
        start  = ring_start(k)
        offset = idx - start
        side   = offset // k
        step   = offset % k
        corners = [(0, -k), (k, -k), (k, 0), (0, k), (-k, k), (-k, 0)]
        a = corners[side]
        b = corners[(side + 1) % 6]
        dq = 0 if (b[0] - a[0]) == 0 else (b[0] - a[0]) // abs(b[0] - a[0])
        dr = 0 if (b[1] - a[1]) == 0 else (b[1] - a[1]) // abs(b[1] - a[1])
        return (a[0] + dq * step, a[1] + dr * step)

    def axial_to_index(q: int, r: int) -> int:
        if (q, r) == (0, 0):
            return 0
        k     = max(abs(q), abs(r), abs(q + r))
        start = ring_start(k)
        corners = [(0, -k), (k, -k), (k, 0), (0, k), (-k, k), (-k, 0)]
        for i in range(6):
            a  = corners[i]
            b  = corners[(i + 1) % 6]
            dq = 0 if (b[0] - a[0]) == 0 else (b[0] - a[0]) // abs(b[0] - a[0])
            dr = 0 if (b[1] - a[1]) == 0 else (b[1] - a[1]) // abs(b[1] - a[1])
            for t in range(k):
                if (a[0] + dq * t, a[1] + dr * t) == (q, r):
                    return start + i * k + t
        raise ValueError("Axial coordinate not found on computed ring.")

    cq, cr = index_to_axial(n)

    def ring_coords(k: int):
        if k == 0:
            yield (cq, cr)
            return
        corners = [
            (cq + 0, cr - k), (cq + k, cr - k), (cq + k, cr + 0),
            (cq + 0, cr + k), (cq - k, cr + k), (cq - k, cr + 0),
        ]
        for i in range(6):
            a  = corners[i]
            b  = corners[(i + 1) % 6]
            dq = 0 if (b[0] - a[0]) == 0 else (b[0] - a[0]) // abs(b[0] - a[0])
            dr = 0 if (b[1] - a[1]) == 0 else (b[1] - a[1]) // abs(b[1] - a[1])
            for t in range(k):
                yield (a[0] + dq * t, a[1] + dr * t)

    ordered = []
    for k in range(num_rings + 1):
        for q, r in ring_coords(k):
            ordered.append(axial_to_index(q, r))
    return tuple(ordered)


class _GraphUtils:
    """
    A helper class providing graph traversal and query functions for the Canvas.
    Automates retinotopically constrained port identifty assignment
    It operates on the connectivity graph G.
    """
    def __init__(self, canvas: Canvas):
        self._canvas = canvas
        self._G = canvas.G

    def get_neighbors_in_rings(self, center_col_idx: int, num_rings: int, require_in_graph: bool) -> Dict[int, int]:
        """
        Finds all column indices within a specified number of rings (hops)
        from a central column, and returns their distance from the center.

        Args:
            center_col_idx: The integer index of the starting column.
            num_rings: The number of rings to search outwards.
                       num_rings=0 returns {center_col_idx: 0}.
                       num_rings=1 returns the center and its direct neighbors.

        Returns:
            A dictionary mapping {column_index: distance_in_rings}.
            The distance is 0 for the center, 1 for the first ring, etc.
        """
        if require_in_graph:
            if center_col_idx not in self._G:
                print(f"Warning: Center column index {center_col_idx} not found in the graph.")
                return {}

            try:
                # This networkx function is highly efficient and returns exactly the
                # dictionary format we need: {node: distance}.
                path_lengths = nx.single_source_shortest_path_length(self._G, center_col_idx, cutoff=num_rings)
                return path_lengths
            except nx.NetworkXNoPath:
                 # This case is unlikely with an undirected graph but is good practice.
                return {center_col_idx: 0}
            
        else:
            ordered_neighbours = self.local_order(center_col_idx, num_rings = num_rings, require_in_graph = False)
            path_lengths = {}
            idx = 0
            path_lengths[ordered_neighbours[idx]] = 0
            idx += 1
            for r in range(1, num_rings + 1):
                for _ in range(6 * r):
                    path_lengths[ordered_neighbours[idx]] = r
                    idx += 1
            return path_lengths
            
        

    def _order_neighbors_by_local_order(self, center: int, num_rings: int, neighbors: dict, require_in_graph:bool) -> OrderedDict:
        """
        Reorder `neighbors` (a dict keyed by hex indices) to match
        self.local_order(center, num_rings, require_in_graph=True).
        """
        order = self.local_order(center, num_rings, require_in_graph=require_in_graph)
        pos = {idx: i for i, idx in enumerate(order)}
        # keep only keys that appear in local order and sort by that order
        items = ((k, neighbors[k]) for k in neighbors.keys() if k in pos)
        return OrderedDict(sorted(items, key=lambda kv: pos[kv[0]]))

    def calc_mimo_centers(self, limit: int = 329, jump: int = 2, step: int = 2, num_rings: int = 2, require_in_graph: bool = True):
        """
        Return center indices for a tiling pattern on a hex grid, plus neighborhoods.

        Parameters
        ----------
        limit : int
            Exclusive upper bound on returned indices — callers typically pass
            their canvas's column count ``N_COLS``, and the result is
            constrained to valid indices ``[0, N_COLS)``.
        jump : int
            How many rings to skip between each batch (e.g., 2 = even rings, 4 = every 4th ring).
        step : int
            How far to step along each ring (default 2 = every other cell).
        num_rings : int
            Neighborhood radius (in rings) around each center to fetch.

        Returns
        -------
        centers : list[int]
            List of selected hex indices (ordered, unique).
        neighborhoods : dict[int, Any]
            Mapping center -> self.graph_utils.get_neighbors_in_rings(center, num_rings).
        """
        centers = [0]
        r = jump
        while True:
            start = 1 + 3 * r * (r - 1)  # first index in ring r
            if start >= limit:
                break
            end = start + 6 * r - 1      # last index in ring r
            # limit is exclusive, matching N_COLS semantics (valid range [0, limit)).
            centers.extend(range(start, min(end + 1, limit), step))
            r += jump

        neighborhoods: dict[int, list[int]] = {}
        for c in centers:
            raw = self.get_neighbors_in_rings(c, num_rings, require_in_graph)
            if require_in_graph:
                # networkx returns keys in arbitrary BFS order — sort into local_order.
                # _local_order_pure is cached, so this call is O(1) after the first hit.
                neighborhoods[c] = self._order_neighbors_by_local_order(
                    c, num_rings, neighbors=raw, require_in_graph=True
                )
            else:
                # get_neighbors_in_rings with require_in_graph=False iterates
                # local_order internally, so keys are already in spiral order.
                # Wrapping in OrderedDict preserves that order without a second
                # local_order call.
                neighborhoods[c] = OrderedDict(raw)

        return neighborhoods


    def local_order(self, n: int, num_rings: int, require_in_graph: bool = False) -> list:
        """
        Clockwise local ordering around center n for any number of rings.

        Returns a flat list:
          [center,
           ring1_clockwise_starting_north...,   # length 6
           ring2_clockwise_starting_north...,   # length 12
           ...
           ringK_clockwise_starting_north...]   # length 6*K

        If require_in_graph=True, only indices present in self._G are kept
        (the list may then be shorter than 1 + 3K(K+1)).
        If require_in_graph=False (default), all mathematically valid spiral
        indices are returned — a fixed-length, position-stable ordering that
        templates and ring-based wiring utilities can rely on even near the
        graph boundary. Unconnected source columns are handled downstream by
        the engines (unfed ports default to 0.0).

        The pure geometry is computed by the module-level _local_order_pure,
        which is @lru_cache'd so repeated calls with the same (n, num_rings)
        are O(1) lookups.
        """
        ordered = list(_local_order_pure(n, num_rings))
        if require_in_graph:
            ordered = [idx for idx in ordered if idx in self._G]
        return ordered

