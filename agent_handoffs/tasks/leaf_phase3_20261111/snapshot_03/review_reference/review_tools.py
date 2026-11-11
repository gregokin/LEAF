"""Independent test helpers; not part of LEAF's runtime or Claude's write scope."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
from leaf import state, traits, forcing
from leaf.interevent import IntereventSession
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st

UTC = dt.timezone.utc
DAY = dt.timedelta(days=1)
MAX_BYTES = 150_000_000

def host(array):
    return array.get() if hasattr(array, 'get') else np.asarray(array)

def make(raw, records, *, backend='numba', start=None, domain=None, seed=123, overrides=None,
         max_bytes=MAX_BYTES, max_edges=100_000, patch_size_m=2.):
    catalog = traits.SpeciesCatalog(raw)
    domain = domain or state.Domain(4, 4, 1.)
    plants = state.PlantState.from_records(catalog, domain, records)
    xp = np
    if backend == 'cuda':
        import cupy as xp
    layers = ss.soil_layers([0., .25, 1.], theta_residual=.05, theta_field_capacity=.3,
                            theta_saturation=.5, diffusivity_m2_s=0., theta_evaporation_stress=.1)
    column = ss.prepare_column(layers, xp.ones(domain.ncells, dtype=bool), domain.dx_m * domain.dy_m,
                               backend=backend)
    operator = st.prepare_transpiration(column)
    water = ss.column_state(column, xp.asarray(np.broadcast_to([[.05], [.15]], (2, domain.ncells)).copy()))
    session = IntereventSession(plants, catalog, operator=operator, native_state=water,
                                start=start or dt.datetime(2024, 12, 31, tzinfo=UTC), backend=backend,
                                growth_cadence=7*DAY, max_edges=max_edges, max_bytes=max_bytes,
                                root_builder='compiled', rng_seed=seed, layer_fraction_overrides=overrides,
                                reproduction_patch_size_m=patch_size_m)
    return session, catalog, operator, xp

def record(pid=1, species='grass', *, x=1.5, y=1.5, biomass=.1, seasonal=.02, age=1000., status='live'):
    return dict(plant_id=pid, species=species, x_m=x, y_m=y, establishment_ground_m=-.25,
                agb_dry_kg=biomass, seasonal_agb_dry_kg=seasonal, age_days=age, status=status)

def step_day(session, xp, temperature=25., pet=5.):
    rec=forcing.TimestepRecord(session.cursor, session.cursor+DAY, temperature, pet_mm_per_day=pet)
    receipts=[]
    for part in session.split_record(rec):
        receipts.append(session.step_interevent(part, added_water_m=xp.zeros((session._L,session._N)),
                                               potential_evaporation_m_s=xp.zeros(session._N)))
    return receipts

def fingerprint(session):
    """Entire accepted public state plus relevant owned arrays, by value (no ID cache construction)."""
    snap=session.snapshot()
    arrays={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in snap._arrays().items()}
    for name in session._b.__slots__:
        value=getattr(session._b,name,None)
        if hasattr(value,'dtype') and hasattr(value,'shape'):
            array=np.asarray(host(value));arrays['boundary.'+name]=(str(array.dtype),array.shape,
                                                                      hashlib.sha256(array.tobytes()).hexdigest())
    scalars={name:repr(getattr(session._b,name,None)) for name in session._b.__slots__
             if name not in ('native_state','root_map') and not hasattr(getattr(session._b,name,None),'dtype')}
    arrays['water']=hashlib.sha256(host(session.native_state.water_m).tobytes()).hexdigest()
    roots=session.root_map
    if roots is not None:
        for name in ('ids','offsets','cells','weights','resources'):
            value=getattr(roots,name,None)
            if value is not None:
                arrays['root.'+name]=hashlib.sha256(host(value).tobytes()).hexdigest()
    return {'arrays':arrays,'describe':session.describe(),'native_t':float(session.native_state.t_s),
            'snapshot_t':float(snap.time_days),'overrides':repr(session._overrides),'boundary_scalars':scalars,
            'anchor':session._anchor,'time0':session._time0,'last_receipt':session.last_receipt}

def by_id(snapshot):
    order=np.argsort(snapshot.ids,kind='stable')
    return {key:value[order] for key,value in snapshot._arrays().items()}

def same_population(a,b):
    for name,left in by_id(a).items():np.testing.assert_array_equal(left,by_id(b)[name])
