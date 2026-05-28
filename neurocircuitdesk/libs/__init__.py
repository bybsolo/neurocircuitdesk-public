"""
neurocircuitdesk.libs
---------------------
Shipped libraries of reusable assets:

- ``algorithms`` — unified-signature algorithms (motion detectors, …) ready
  to be assigned via ``mc.set_block_func(...)``.
- ``microcircuit_templates`` — curated ``MicroCircuit`` templates following
  the ``{CMC,iCMC}_<biology>_<variant>`` naming convention. Plus the
  ``mc_lib`` inspection API (``list / get / describe / show / preview``).
- ``jsons/``     — retinotopic placement and connectivity-graph JSON files.
"""

from neurocircuitdesk.libs import algorithms, microcircuit_templates

__all__ = ['algorithms', 'microcircuit_templates']
