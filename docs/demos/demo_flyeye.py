"""
demo_flyeye.py
--------------
Interactive 3D visualisation of the fly compound eye geometry.
Each demo opens a Plotly figure in the browser.

Usage:
    python demo_flyeye.py                # all demos
    python demo_flyeye.py single         # one retina, R1-R6 rays fanned across neighbours
    python demo_flyeye.py column         # one retina, all 7 axes through column 0
    python demo_flyeye.py dual_all       # two retinas with every R7 ray drawn
    python demo_flyeye.py dual_pick      # two retinas with a hand-picked ray set
"""

import argparse
import os
import sys

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flyeyesimulator import Retina, RetinaRotator, SimulatorViz

COL_JSON = os.path.join(_REPO_ROOT, 'neurocircuitdesk', 'libs', 'jsons',
                        'hexcol_l1m3_new_578.json')


def _make_retina():
    return Retina(
        num_rings=20,
        inter_ommatidia_angle_deg=4,
        bio_cols_only=True,
        col_json_path=COL_JSON,
    )


def demo_single_fan():
    """Single retina with R1-R6 rays fanned across six different columns."""
    retina = _make_retina()

    viz = SimulatorViz(lens_radius=0.04, lens_length=0.01,
                       rays=False, ray_mode='R7', ray_length=8, ray_width=2)
    viz.add_retina(retina, color="steelblue", name="left", ray_color='blue')
    viz.add_screen(radius=10)
    viz.add_rays([
        ('left', 0,  'R7', 'blue'),
        ('left', 5,  'R1', 'blue'),
        ('left', 6,  'R2', 'blue'),
        ('left', 18, 'R3', 'blue'),
        ('left', 1,  'R4', 'blue'),
        ('left', 2,  'R5', 'blue'),
        ('left', 3,  'R6', 'blue'),
    ])
    viz.plot()


def demo_single_column():
    """Single retina, all seven R-axes (R1-R7) drawn through column 0."""
    retina = _make_retina()

    viz = SimulatorViz(lens_radius=0.04, lens_length=0.01,
                       rays=False, ray_mode='R7', ray_length=8, ray_width=2)
    viz.add_retina(retina, color="steelblue", name="left", ray_color='blue')
    viz.add_screen(radius=10)
    viz.add_rays([
        ('left', 0, 'R7', 'red'),
        ('left', 0, 'R1', 'red'),
        ('left', 0, 'R2', 'red'),
        ('left', 0, 'R3', 'red'),
        ('left', 0, 'R4', 'red'),
        ('left', 0, 'R5', 'red'),
        ('left', 0, 'R6', 'red'),
    ])
    viz.plot()


def _build_dual_eyes():
    retina = _make_retina()
    rotatorL = RetinaRotator(
        retina,
        offset=(-0.5, 5, 0.0),
        euler_deg=(-60, 60, 0),
        mirror='x',
    )
    rotatorR = RetinaRotator(
        retina,
        offset=(0.5, 5, 0.0),
        euler_deg=(-60, -60, 0),
    )
    return rotatorL.apply(), rotatorR.apply()


def demo_dual_all_rays():
    """Dual-eye (rotated + mirrored) setup with every R7 ray drawn."""
    retinaL, retinaR = _build_dual_eyes()

    viz = SimulatorViz(lens_radius=0.04, lens_length=0.01,
                       rays=True, ray_mode='R7', ray_length=8, ray_width=0.1)
    viz.add_retina(retinaL, color="steelblue", name="left",  ray_color='blue')
    viz.add_retina(retinaR, color="crimson",   name="right", ray_color='red')
    viz.add_screen(radius=15)
    viz.plot()


def demo_dual_picked_rays():
    """Dual-eye setup with hand-picked rays per eye."""
    retinaL, retinaR = _build_dual_eyes()

    viz = SimulatorViz(lens_radius=0.04, lens_length=0.01,
                       rays=False, ray_mode='R7', ray_length=10, ray_width=2)
    viz.add_retina(retinaL, color="steelblue", name="left",  ray_color='blue')
    viz.add_retina(retinaR, color="crimson",   name="right", ray_color='red')
    viz.add_screen(radius=10)
    viz.add_rays([
        ('left', 0, 'R7', 'red'),
        ('left', 0, 'R1', 'blue'),
        ('left', 0, 'R2', 'blue'),
        ('left', 0, 'R3', 'blue'),
        ('left', 0, 'R4', 'blue'),
        ('left', 0, 'R5', 'blue'),
        ('left', 0, 'R6', 'blue'),
    ])
    viz.add_rays([
        ('right', 0,  'R7', 'red'),
        ('right', 5,  'R1', 'blue'),
        ('right', 6,  'R2', 'blue'),
        ('right', 18, 'R3', 'blue'),
        ('right', 1,  'R4', 'blue'),
        ('right', 2,  'R5', 'blue'),
        ('right', 3,  'R6', 'blue'),
    ])
    viz.plot()


DEMOS = {
    'single':    demo_single_fan,
    'column':    demo_single_column,
    'dual_all':  demo_dual_all_rays,
    'dual_pick': demo_dual_picked_rays,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('demo', nargs='?', choices=list(DEMOS.keys()) + ['all'],
                        default='all',
                        help='Which demo to run (default: all).')
    args = parser.parse_args()

    names = list(DEMOS.keys()) if args.demo == 'all' else [args.demo]
    for name in names:
        print(f"-- Running demo: {name} --")
        DEMOS[name]()


if __name__ == '__main__':
    main()
