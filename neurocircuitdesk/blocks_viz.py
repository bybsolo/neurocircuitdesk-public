from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Any
import numpy as np
import plotly.graph_objects as go
import json
import networkx as nx
import re


class Blocks:
    """Lightweight block shape primitives.

    Each static method returns a dict with:
      - ``ports``: Dict[str, (x, y, z)] — port coordinates for arrow routing
      - ``meta``:  dict with keys used by the batched renderer:
            kind, x, y, z, color, name, r, h, ...

    No ``go.*`` objects are created here — all Plotly trace generation is
    deferred to :meth:`Canvas._render_batched`.
    """

    @staticmethod
    def siso_encase(x=0, y=0, z_center=0, rad=0.5, height=1.0,
                    name="siso_encase", color="green", alpha=0.05):
        ports = {
            "input":  (x, y, z_center + height / 2),
            "output": (x, y, z_center - height / 2),
        }
        meta = dict(kind='encase', x=x, y=y, z=z_center,
                    r=rad, h=height, color=color, alpha=alpha, name=name)
        return {"ports": ports, "meta": meta}

    @staticmethod
    def mimo_encase(x=0, y=0, z=0, radius=1.0, height=0.15, offset=0,
                    name='mimo_encase', col_coords=None, alpha=0.1, flip=False):
        if col_coords is not None and len(col_coords) > 0:
            coords = np.array(list(col_coords.values()))
            x, y = coords.mean(axis=0)
            dists = np.sqrt(np.sum((coords - [x, y])**2, axis=1))
            radius = np.max(dists) + 0.2

        if col_coords is None or len(col_coords) == 0:
            raise ValueError("Columnar targets for MIMO component not specified!")

        ports = {}
        for col_name, (px, py) in col_coords.items():
            ppx, ppy = px + offset, py + offset
            if flip:
                in_z, out_z = z - height / 2, z + height / 2
            else:
                in_z, out_z = z + height / 2, z - height / 2
            ports[f"input_col_{col_name}"] = (ppx, ppy, in_z)
            ports[f"output_col_{col_name}"] = (ppx, ppy, out_z)

        meta = dict(kind='encase', x=x + offset, y=y + offset, z=z,
                    r=radius, h=height, color='yellow', alpha=alpha, name=name)
        return {"ports": ports, "meta": meta}

    @staticmethod
    def block_siso(x=0, y=0, z=0, name='default', r=0.1, h=0.3):
        ports = {
            'input':  (x, y, z + h / 2),
            'output': (x, y, z - h / 2),
        }
        meta = dict(kind='block', x=x, y=y, z=z, r=r, h=h,
                    color='cyan', alpha=0.9, name=name)
        return {"ports": ports, "meta": meta}

    @staticmethod
    def block_mimo(x=0, y=0, z=0, col_idxs=None, name='block_mimo'):
        N = len(col_idxs)
        radius = 1
        height = 0.15

        theta_ring = np.linspace(0, 2 * np.pi, N, endpoint=False)
        input_coords = [(x + radius * np.cos(t), y + radius * np.sin(t), z + height / 2)
                        for t in theta_ring]
        output_coords = [(x + radius * np.cos(t), y + radius * np.sin(t), z - height / 2)
                         for t in theta_ring]

        ports = {}
        for i, idx in enumerate(col_idxs):
            ports[f'{name}_input_col_{idx}'] = input_coords[i]
            ports[f'{name}_output_col_{idx}'] = output_coords[i]

        meta = dict(kind='block', x=x, y=y, z=z, r=radius, h=height,
                    color='yellow', alpha=0.9, name=name)
        return {"ports": ports, "meta": meta}

    @staticmethod
    def block_multiport(x=0, y=0, z=0, name='multiport_block',
                        input_names: List[str] = [],
                        output_names: List[str] = [],
                        radius=0.2, height=0.2):
        ports = {}

        num_inputs = len(input_names)
        if num_inputs > 0:
            theta_in = np.linspace(0, 2 * np.pi, num_inputs, endpoint=False)
            for i, pn in enumerate(input_names):
                ports[pn] = (x + radius * np.cos(theta_in[i]),
                             y + radius * np.sin(theta_in[i]),
                             z + height / 2)

        num_outputs = len(output_names)
        if num_outputs > 0:
            theta_out = np.linspace(0, 2 * np.pi, num_outputs, endpoint=False) + \
                        (np.pi / num_outputs if num_outputs > 1 else 0)
            for i, pn in enumerate(output_names):
                ports[pn] = (x + radius * np.cos(theta_out[i]),
                             y + radius * np.sin(theta_out[i]),
                             z - height / 2)

        meta = dict(kind='block', x=x, y=y, z=z, r=radius, h=height,
                    color='purple', alpha=0.8, name=name)
        return {"ports": ports, "meta": meta}

    @staticmethod
    def division(x=0, y=0, z=0, name='default'):
        radius = 0.1
        height = 0.2

        numerator_pos = (x - 0.05, y, z + height / 2)
        denominator_pos = (x + 0.05, y, z + height / 2)
        output_pos = (x, y, z - height / 2)

        ports = {
            'numerator': numerator_pos,
            'denominator': denominator_pos,
            'output': output_pos,
        }
        meta = dict(kind='block', x=x, y=y, z=z, r=radius, h=height,
                    color='red', alpha=0.5, name=name)
        return {"ports": ports, "meta": meta}

    @staticmethod
    def block_neurite(x=0, y=0, z=0, r=0.1, h=0.1,
                      name='neurite_block', color='yellow', flip=False):
        in_z = z - h / 2 if flip else z + h / 2
        out_z = z + h / 2 if flip else z - h / 2

        ports = {
            'input': (x, y, in_z),
            'output': (x, y, out_z),
            'interconnect': (x, y, z),
        }
        meta = dict(kind='block', x=x, y=y, z=z, r=r, h=h,
                    color=color, alpha=0.9, name=name)
        return {"ports": ports, "meta": meta}

    # ------------------------------------------------------------------
    # Arrow helpers (used only for intra-MC connections in MicroCircuit.connect)
    # ------------------------------------------------------------------
    @staticmethod
    def arrow(src, dest, color='black', width=2):
        """Return metadata for a straight arrow (no Plotly objects)."""
        return [{'kind': 'arrow', 'src': src, 'dst': dest, 'color': color}]

    @staticmethod
    def curved_arrow(src, dst, color='black', width=4, bend=0.3):
        """Return metadata for a curved arrow (no Plotly objects)."""
        return [{'kind': 'arrow', 'src': src, 'dst': dst, 'color': color}]
