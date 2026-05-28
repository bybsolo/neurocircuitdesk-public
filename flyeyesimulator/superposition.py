from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Callable, Any
import json


def local_order(n: int, num_rings: int) -> list[int]:
    """
    Clockwise local ordering around center n for any number of rings.

    Returns a flat list:
      [center,
       ring1_clockwise_starting_north...,   # length 6
       ring2_clockwise_starting_north...,   # length 12
       ...
       ringK_clockwise_starting_north...]   # length 6*K

    If require_in_graph=True, only indices present in self._G are kept.
    Otherwise, all mathematically valid spiral indices are returned.
    """
    def ring_start(k: int) -> int:
        if k == 0:
            return 0
        return 1 + 3 * k * (k - 1)

    def index_to_axial(n: int) -> tuple[int,int]:
        if n == 0:
            return (0,0)
        k = 1
        while ring_start(k+1) <= n:
            k += 1
        start = ring_start(k)
        offset = n - start  # 0..6k-1
        side = offset // k  # 0..5
        step = offset % k   # 0..k-1
        corners = [(0,-k),(k,-k),(k,0),(0,k),(-k,k),(-k,0)]
        a = corners[side]
        b = corners[(side+1) % 6]
        dq = 0 if (b[0]-a[0]) == 0 else (b[0]-a[0]) // abs(b[0]-a[0])
        dr = 0 if (b[1]-a[1]) == 0 else (b[1]-a[1]) // abs(b[1]-a[1])
        return (a[0] + dq*step, a[1] + dr*step)

    def axial_to_index(q: int, r: int) -> int:
        if (q, r) == (0, 0):
            return 0
        k = max(abs(q), abs(r), abs(q + r))
        start = ring_start(k)
        corners = [(0,-k),(k,-k),(k,0),(0,k),(-k,k),(-k,0)]
        for i in range(6):
            a = corners[i]
            b = corners[(i+1) % 6]
            dq = 0 if (b[0]-a[0]) == 0 else (b[0]-a[0]) // abs(b[0]-a[0])
            dr = 0 if (b[1]-a[1]) == 0 else (b[1]-a[1]) // abs(b[1]-a[1])
            for t in range(k):
                if (a[0] + dq*t, a[1] + dr*t) == (q, r):
                    return start + i*k + t
        raise ValueError("Axial coordinate not found on computed ring (shouldn't happen).")
    # ------------------------------------

    # center in axial
    cq, cr = index_to_axial(n)

    # helper: enumerate coords on ring k around (cq,cr), CW from "north"
    def ring_coords(k: int):
        if k == 0:
            yield (cq, cr)
            return
        corners = [(cq + 0,   cr - k),
                   (cq + k,   cr - k),
                   (cq + k,   cr + 0),
                   (cq + 0,   cr + k),
                   (cq - k,   cr + k),
                   (cq - k,   cr + 0)]
        for i in range(6):
            a = corners[i]
            b = corners[(i+1) % 6]
            dq = 0 if (b[0]-a[0]) == 0 else (b[0]-a[0]) // abs(b[0]-a[0])
            dr = 0 if (b[1]-a[1]) == 0 else (b[1]-a[1]) // abs(b[1]-a[1])
            for t in range(k):
                yield (a[0] + dq*t, a[1] + dr*t)

    ordered = []
    for k in range(0, num_rings + 1):
        for q, r in ring_coords(k):
            idx = axial_to_index(q, r)
            # if not require_in_graph or (idx in self._G):
            ordered.append(idx)

    return ordered

def superposition_mask(n):
    local = local_order(n,2)
    local = np.array(local)
    return local[[5,6,18,1,2,3]]

def get_prs(n, pr_array, all_idx):
    T = pr_array.shape[0]
    group_indices = superposition_mask(n)
    offsets = np.arange(6)
    target_cols = np.array(group_indices) * 6 + offsets
    
    extracted_data = []
    for col_idx in target_cols:
        if col_idx in all_idx:
            extracted_data.append(pr_array[:, int(col_idx)])
        else:
            extracted_data.append(np.full(T, -80.0))
    return np.stack(extracted_data, axis=1)

