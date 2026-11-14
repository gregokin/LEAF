'''Deterministic genuine-projection fixtures shared by the compact-helper tests (unique module name). `build(name, wv_module)` runs the given water_vegetation module so the
frozen pre-refactor source can produce the golden hashes.'''
from __future__ import annotations

import json

import numpy as np

from leaf import state, traits
from tests import native_leaf_fixture as nfx
from tests.test_water_vegetation import catalog, flat, plants_state

KEY = traits.RECEIVER_WEIGHTS_KEY
EDGES = 1 << 16
BIG = 1 << 28
NAMES = ('masked3', 'periodic_origin', 'session', 'empty', 'dead_only', 'zero_weights')


def _kw(**more):
    return dict(max_edges=EDGES, max_bytes=BIG, **more)


def build(name, wv):
    if name == 'masked3':
        cat = catalog(layers=3, grass=[0, 1, 1], shrub=[0, 0, 0], fractions={'grass': [0.5, 0.3, 0.2], 'shrub': [0.2, 0.3, 0.5]})
        st = plants_state(cat, [(1, 'grass', 2.5, 2.5, 0.4, 'live'), (2, 'shrub', 6.5, 6.5, 0.4, 'live'), (3, 'grass', 8.5, 1.5, 0.3, 'dead_standing'), (4, 'shrub', 0.5, 0.5, 0.2, 'live')])
        mask = np.ones((10, 10), dtype=bool)
        mask[0, 0] = False
        return wv.project_from_initial_state(st, cat, flat(st), active_mask=mask, **_kw()), mask
    if name == 'periodic_origin':
        cat = catalog(layers=4, grass=[0, 1, 2, 1], shrub=[1, 0, 0, 3])
        rng = np.random.default_rng(11)
        pl = [(i, 'grass' if i % 3 else 'shrub', float(rng.uniform(40, 59)), float(rng.uniform(70, 89)), float(rng.uniform(0.3, 1.3)), 'live' if i % 7 else 'dead_standing')
              for i in range(1, 400)]
        st = plants_state(cat, pl, nx=20, ny=20, dx=1.0, origin=(40.0, 70.0), periodic=(True, True))
        mask = np.ones((20, 20), dtype=bool)
        mask[:, 17:] = False
        return wv.project_from_initial_state(st, cat, flat(st), active_mask=mask, **_kw()), mask
    if name == 'session':
        doc = json.loads(nfx.FIXTURES['phase1'].read_text())
        nfx.simple_edit(doc)
        doc['schema_version'] = 2
        for sid in doc['species']:
            doc['species'][sid]['hydrology'] = {KEY: [0.0, 1.0]}
        cat = traits.SpeciesCatalog(doc)
        dom = state.Domain(12, 10, 1.0)
        recs = [{'plant_id': i + 1, 'species': 'grass', 'x_m': 1.5 + (i % 10), 'y_m': 3.5, 'establishment_ground_m': 0.0, 'agb_dry_kg': 0.1, 'seasonal_agb_dry_kg': 0.0, 'age_days': 100.0,
                 'status': s} for i, s in enumerate(('live', 'dead_standing', 'removed', 'live'))]
        st0 = state.PlantState.from_records(cat, dom, recs)
        sess = nfx.start_session(st0, nfx.make_config(cat), nfx.anchor(), bound=True)
        return wv.project_from_plant_session(sess, np.zeros(dom.shape), **_kw()), None
    if name == 'empty':
        cat = catalog(layers=2, grass=[0, 1], shrub=[1, 1])
        st = plants_state(cat, [])
        return wv.project_from_initial_state(st, cat, flat(st), **_kw()), None
    if name == 'dead_only':
        cat = catalog(layers=2, grass=[0, 1], shrub=[1, 1])
        st = plants_state(cat, [(1, 'grass', 3.5, 3.5, 0.5, 'dead_standing')])
        return wv.project_from_initial_state(st, cat, flat(st), **_kw()), None
    if name == 'zero_weights':
        cat = catalog(layers=3, grass=[0, 0, 0], shrub=[0, 2, 1], fractions={'grass': [1.0, 0.0, 0.0], 'shrub': [0.1, 0.1, 0.8]})
        st = plants_state(cat, [(1, 'grass', 3.5, 3.5, 0.5, 'live'), (2, 'shrub', 3.9, 3.6, 0.5, 'live')])
        return wv.project_from_initial_state(st, cat, flat(st), **_kw()), None
    raise KeyError(name)
