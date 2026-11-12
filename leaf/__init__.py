'''leaf: Landscape Ecohydrology with Agent-based Flora (LEAF).

Phase 0 (independent, NumPy + stdlib only; no numba/cupy/MAPLE/SYRUP imports):
  leaf.traits          versioned species catalog JSON, power-law allometry, canonical catalog hash
  leaf.state           Domain, canonical PlantState (SoA, stable int64 IDs), plant-state CSV, SURFACE anchors
  leaf.canopy          derived geometry, shared canopy_view (one aboveground representation for MAPLE AND
                       SYRUP), thin MAPLE adapter maple_vegetation_view, bounded union-cover measurement
  leaf.initialization  cover-driven multispecies generator (overlap allowed / global canopy exclusion)
  leaf.recording       cadence snapshot writer (atomic, no overwrite, no fabricated states)
  leaf.events          MAPLE/SYRUP surface-event diagnostics + ecological response socket
  leaf.runtime         ID-keyed bridge to the existing growth drivers (snapshot capture, legacy inputs)
  python -m leaf       CLI (validate-species, init, generate, inspect, export-canopy)

Phase 1 part A (NumPy-only imports; numba / cupy load lazily per backend):
  leaf.uptake          sparse CSR root map, stateless water proposal / accepted feedback (numpy, numba, cuda)
  leaf.roots           root-map builder from PlantState + catalog (shared allometry, labelled uniform overlap law)
  leaf.forcing         strict fixed-offset interevent timestep records (stdlib only)

Phase 3 (NumPy-only, host reference; lazily imported):
  leaf.recruitment     opt-in species-local simple seed establishment per ecological patch and nearby vegetative
                       daughter plants (law simple_seed_clonal_v1; keyed counter-based randomness; UNCALIBRATED)
  IntereventSession.apply_reproduction  the explicit annual reproduction boundary (complete checkpoints are layout
                       version 4: a plain plants.csv plus a companion plants.metadata.json; versions 1-3 are refused)

Legacy coupled growth (optional dependencies numba / cupy / maple_syrup) stays reachable through LAZY
exports: `from leaf import CoupledSimulation, CoupledPhaseError, stepwat` and `leaf.coupling` import the
native-coupling modules only when first touched. `import leaf` alone imports nothing beyond NumPy.
'''
from __future__ import annotations

import importlib

__version__ = '0.1.0'

_LAZY_ATTRS = {'CoupledSimulation': ('leaf.coupling', 'CoupledSimulation'),
               'CoupledPhaseError': ('leaf.coupling', 'CoupledPhaseError'),
               'stepwat': ('leaf.stepwat', None),
               'coupling': ('leaf.coupling', None)}
for _name in ('traits', 'state', 'canopy', 'initialization', 'recording', 'events', 'runtime',
              'uptake', 'roots', 'forcing', 'recruitment'):
    _LAZY_ATTRS[_name] = ('leaf.' + _name, None)
del _name

__all__ = ['CoupledSimulation', 'CoupledPhaseError', 'stepwat', 'coupling', 'traits', 'state', 'canopy',
           'initialization', 'recording', 'events', 'runtime', 'uptake', 'roots', 'forcing', 'recruitment',
           '__version__']


def __getattr__(name):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module = importlib.import_module(target[0])
    value = module if target[1] is None else getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
