from typing import List, Tuple, Dict, Optional, Callable, Any
from neurocircuitdesk.blocks_viz import Blocks
from neurocircuitdesk.blocks_exe import *

class MicroCircuit:
    """
    A unified, self-contained computational unit.
    It manages both its visual representation and its underlying executable graph (exec_nodes). 
    It can be configured via templates.
    """
    def __init__(self, canvas: "Canvas", name: str, col_idx: int,
                 x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 template: Optional[Callable[["MicroCircuit"], None]] = None):
        self.canvas = canvas
        self.name = name
        self.col_idx = col_idx
        self.center = (x, y, z)

        # Visualization objects
        self._viz_nodes: Dict[str, Any] = {}
        self._viz_edges: Dict[str, List[Any]] = {}

        # Public I/O declaration
        self.input_ports: Dict[str, List[Tuple[str, str]]] = {}
        self.output_ports: Dict[str, Tuple[str, str]] = {}

        # Executable Graph 
        self._exec_nodes: Dict[str, "Node"] = {} # holds the executable note object such as a funcblock or operator, name such as 'T1'
        self._exec_edges: list[tuple[str, str, str, str]] = [] # list of tuples that defines the computational connections between nodes
        
        # This dictionary is for introspection, to easily see an block's ports.
        self.block_ports: Dict[str, Dict[str, List[str]]] = {}
        
        # Build microcircuit from template if provided
        if template:
            template(self)
            

    def add_block(self, block_id: str, *block_pos,
                  input_names: Optional[List[str]] = None,
                  output_names: Optional[List[str]] = None,
                  node_kind: str = 'default',
                  **kwargs):
        """
        Adds a computational and visual block to the microcircuit.
        Includes a validation check for the node_kind.
        """
        supported_kinds = ['default', 'division', 'derivative', 'rectifier_pos', 'rectifier_inv', 'temporal_filter', 'aggregator']
        if node_kind not in supported_kinds:
            raise ValueError(f"Unsupported node_kind: '{node_kind}'. "
                             f"Supported kinds are: {supported_kinds}")

        if block_id in self._exec_nodes:
            raise ValueError(f"Block ID '{block_id}' already exists.")

        fq_name = f"{self.name}/{block_id}"

        if node_kind == 'division':
            block_obj = Blocks.division(*block_pos, name=block_id)
            exec_node = Division(fq_name)
            # division ports are dynamic, so block_ports is not populated here.
            
        elif node_kind == 'derivative':
            block_obj = Blocks.block_siso(*block_pos, name=block_id)
            exec_node = TemporalDerivative(fq_name)
            self.block_ports[block_id] = {'inputs': ['input'], 'outputs': ['output']}

        elif node_kind == 'rectifier_pos':
            block_obj = Blocks.block_siso(*block_pos, name=block_id, r=0.08, h=0.2)
            exec_node = Rectifier(fq_name, mode='on')
            self.block_ports[block_id] = {'inputs': ['input'], 'outputs': ['output']}

        elif node_kind == 'rectifier_inv':
            block_obj = Blocks.block_siso(*block_pos, name=block_id, r=0.08, h=0.2)
            exec_node = Rectifier(fq_name, mode='off')
            self.block_ports[block_id] = {'inputs': ['input'], 'outputs': ['output']}
        
        elif node_kind == 'temporal_filter':
            block_obj = Blocks.block_siso(*block_pos, name=block_id, h=0.25)
            exec_node = TemporalFilter(fq_name)
            self.block_ports[block_id] = {'inputs': ['input'], 'outputs': ['output']}

        elif node_kind == 'aggregator':
            mode = kwargs.get('mode', 'sum')
            exec_node = Aggregator(fq_name, mode=mode)
            block_obj = Blocks.block_multiport(*block_pos, name=block_id,
                                               input_names=[], output_names=['output'])
            block_obj['block_pos'] = tuple(block_pos)
            self.block_ports[block_id] = {'inputs': [], 'outputs': ['output']}

        # multiport FuncBlocks (MIMO, MISO, etc.)
        # distinc case since the i/o port names need to contain col identify info, unlike SISO
        elif input_names is not None and output_names is not None:
            block_obj = Blocks.block_multiport(*block_pos, name=block_id,
                                               input_names=input_names, output_names=output_names)
            exec_node = FuncBlock(fq_name, f=placeholder_passthrough, params={},
                                  input_names=input_names, output_names=output_names, **kwargs)
            self.block_ports[block_id] = {'inputs': input_names, 'outputs': output_names}
            
        # default case for a standard SISO FuncBlock
        else: 
            # node_kind == 'default'
            block_obj = Blocks.block_siso(*block_pos, name=block_id)
            exec_node = FuncBlock(fq_name, f=placeholder_passthrough, params={}, **kwargs)
            self.block_ports[block_id] = {'inputs': ['input'], 'outputs': ['output']}

        self._viz_nodes[block_id] = block_obj
        self._exec_nodes[block_id] = exec_node
        return block_id

    # def add_aggregator_input(self, block_id: str, port_name: str):
    #     """
    #     Adds an input port to an aggregator block. Call this when building the
    #     microcircuit (e.g. in a template) for each input the aggregator will have.
    #     """
    #     exec_node = self.get_exec_node(block_id)
    #     if not isinstance(exec_node, Aggregator):
    #         raise TypeError(f"Block '{block_id}' is not an Aggregator.")
    #     exec_node.add_input_port(port_name)
    #     self.block_ports[block_id]['inputs'].append(port_name)
    #     # Refresh visual so the new port has a position
    #     block_pos = self._viz_nodes[block_id].get('block_pos', (0, 0, 0))
    #     block_obj = Blocks.block_multiport(*block_pos, name=block_id,
    #                                        input_names=self.block_ports[block_id]['inputs'],
    #                                        output_names=['output'])
    #     block_obj['block_pos'] = block_pos
    #     self._viz_nodes[block_id] = block_obj

    def get_port_coord(self, public_port_name: str, port_direction: str) -> Tuple[float, float, float]:
        """
        Retrieves the 3D world coordinates of a public-facing microcircuit port,
        For viz arrow connection 
        """
        try:
            if port_direction == 'output':
                block_id, internal_port_name = self.output_ports[public_port_name]
            elif port_direction == 'input':
                connections = self.input_ports[public_port_name]
                if not connections:
                    raise KeyError("Input port is defined but has no internal connections.")
                block_id, internal_port_name = connections[0]
            else:
                raise ValueError(f"Invalid port_direction '{port_direction}'. Must be 'input' or 'output'.")

            # dynamic division ports 
            exec_node = self.get_exec_node(block_id)
            viz_port_name = internal_port_name
            if isinstance(exec_node, Division) and internal_port_name in exec_node.inputs:
                # explicitly map the execution port name (e.g., 'den_feedback') to the visual port
                # name (e.g., 'denominator') for coordinate lookup.
                viz_port_name = exec_node.inputs[internal_port_name].port_type
            
            viz_block = self._viz_nodes[block_id]
            coordinates = viz_block['ports'][viz_port_name]
            return coordinates

        except KeyError:
            raise KeyError(f"Public port '{public_port_name}' of type '{port_direction}' not found or not connected in microcircuit '{self.name}'.")

            
    def get_exec_node(self, block_id: str) -> "Node":
        """
        Retrieves a reference to an executable node from the microcircuit's internal graph.
        """
        if block_id not in self._exec_nodes:
            raise KeyError(f"Executable node '{block_id}' not found in microcircuit '{self.name}'.")
        return self._exec_nodes[block_id]


    def connect(self, src_id: str, src_port: str, dst_id: str, dst_port: str, **kwargs):
        """
        Connects two blocks both visually and computationally.
        """
        # visualization connection 
        src_block = self._viz_nodes[src_id]
        dst_block = self._viz_nodes[dst_id]
        src_pos = src_block['ports'][src_port]

        dst_exec_node = self.get_exec_node(dst_id)
        viz_dst_port = dst_port
        port_type = None

        if isinstance(dst_exec_node, Division):
            if dst_port in dst_exec_node.inputs:
                port_type = dst_exec_node.inputs[dst_port].port_type
                viz_dst_port = port_type
            else:
                raise KeyError(f"Port '{dst_port}' not found on Division node '{dst_id}'. Did you call add_input_port()?")

        dst_pos = dst_block['ports'][viz_dst_port]

        color = 'red' if port_type == 'denominator' else 'black'
        curved = kwargs.get('curved', False)
        arrow_traces = Blocks.curved_arrow(src_pos, dst_pos, color=color) if curved else Blocks.arrow(src_pos, dst_pos, color=color)
        
        connection_name = f'{src_id}:{src_port} -> {dst_id}:{dst_port}'
        self._viz_edges[connection_name] = arrow_traces

        # executable graph connection 
        if src_id not in self._exec_nodes or dst_id not in self._exec_nodes:
            raise KeyError("Source or destination node not found in executable graph.")
        self._exec_edges.append((src_id, src_port, dst_id, dst_port))

    def set_block_func(self, block_id: str, f: Callable, params: Optional[dict] = None):
        """
        Set a exec node's function between input and output, using a CALLBLE
        """
        node = self.get_exec_node(block_id)
        if not isinstance(node, FuncBlock):
            raise TypeError(f"Node '{block_id}' is not a FuncBlock.")
        if not getattr(f, '_unified', False):
            raise TypeError(
                f"Function '{f.__name__}' passed to set_block_func must be "
                f"decorated with @unified_algorithm."
            )
        node.f = f
        node.params = params or {}

    def set_block_params(self, block_id: str, params: dict):
        """
        Set the parameters of nodes's function
        """
        node = self.get_exec_node(block_id)
        node.params.update(params or {})

    def specify_io(self, inputs: List[Tuple[str, str, str]], outputs: List[Tuple[str, str, str]]):
        """
        Specifies which internal ports are exposed to the outside world.
        Populate the self.input_ports and self.output_ports dictionaries and establish mapping between public name and internal component's port
        allowing Canvas to route inter-microcircuits connections
        This is a user-side declaration/specificaiton, does not create comp obj
        """
        for pub_name, block_id, port_name in inputs:
            if pub_name not in self.input_ports:
                self.input_ports[pub_name] = []
            self.input_ports[pub_name].append((block_id, port_name))

        for pub_name, block_id, port_name in outputs:
            if pub_name in self.output_ports:
                raise ValueError(f"Output port name '{pub_name}' is already defined.")
            self.output_ports[pub_name] = (block_id, port_name)

    def emit_exec_graph(self) -> dict:
        """
        Build comp backend:
        Generates the final, self-contained executable graph for this microcircuit,
        including special Input/Output nodes that represent its public interface.
        """
        # converts all internal nodes and edges to use their fully qualified names ensuring no name conflicts when canvas compiles
        nodes_fq = {f"{self.name}/{ln}": n for ln, n in self._exec_nodes.items()}
        edges_fq = [(f"{self.name}/{s}", sp, f"{self.name}/{d}", dp) for s, sp, d, dp in self._exec_edges]

        # add special InputNodes for each public input port, using specify_io
        for pub_name, connections in self.input_ports.items():
            input_node_name = f"{self.name}/{pub_name}_INPUT"
            nodes_fq[input_node_name] = InputNode(input_node_name)
            for block_id, port_name in connections:
                dst_node_fq = f"{self.name}/{block_id}"
                edges_fq.append((input_node_name, 'output', dst_node_fq, port_name))

        # add special OutputNodes for each public output port, using specify_io
        for pub_name, (block_id, port_name) in self.output_ports.items():
            output_node_name = f"{self.name}/{pub_name}_OUTPUT"
            nodes_fq[output_node_name] = OutputNode(output_node_name)
            src_node_fq = f"{self.name}/{block_id}"
            edges_fq.append((src_node_fq, port_name, output_node_name, 'input'))

        # the canvas uses these to know what to connect to
        canvas_inputs = list(self.input_ports.keys())
        canvas_outputs = list(self.output_ports.keys())

        return {
            "nodes": nodes_fq,
            "edges": edges_fq,
            "inputs": canvas_inputs,
            "outputs": canvas_outputs,
        }

    def show(self, *, width: int = 600, height: int = 450,
             title: Optional[str] = None):
        """Return a standalone 3D Plotly figure of this MC's wiring.

        Independent of Canvas — works as soon as the template body has
        populated ``_viz_nodes`` / ``_viz_edges`` / ``input_ports`` /
        ``output_ports``. Auto-layout:

        - **CMC** — input port markers stack above the block cluster,
          output port markers below.
        - **iCMC** — port names matching ``<prefix>_col_<N>`` are grouped
          by channel prefix and laid out as a local hex pattern derived
          from each column's spiral position relative to ``self.col_idx``.

        Returns
        -------
        plotly.graph_objs.Figure
            Inline-renderable Plotly figure.
        """
        from neurocircuitdesk.microcircuit_viz import MicroCircuitViz
        return MicroCircuitViz(self, width=width, height=height,
                                title=title).plot()

    def get_viz_metadata(self):
        """Return lightweight metadata for batched rendering.

        Returns a dict with:
          - ``blocks``: list of block meta dicts (kind, x, y, z, color, ...)
          - ``ports``:  list of (x, y, z, label) for all port markers
          - ``arrows``: list of (src, dst, color) for intra-MC arrows
        """
        blocks = []
        ports = []
        arrows = []

        for block in self._viz_nodes.values():
            if 'meta' in block:
                blocks.append(block['meta'])
            for pname, coord in block['ports'].items():
                ports.append((*coord, pname))

        for arrow_list in self._viz_edges.values():
            for a in arrow_list:
                if isinstance(a, dict) and a.get('kind') == 'arrow':
                    arrows.append((a['src'], a['dst'], a['color']))

        return {'blocks': blocks, 'ports': ports, 'arrows': arrows}




