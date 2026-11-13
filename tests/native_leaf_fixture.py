'''Shared test fixtures of the daily plant interface: (1) SYNTHETIC LEAF-side helpers (catalog, plants, session config, availability and
accepted-withdrawal VALUE OBJECTS built from LEAF's own proposal: no SYRUP, no water physics) and (2) the REAL native SYRUP Phase 1.5 harness
(`native_daily_harness`), which runs only when the environment configures the actual native case. A CONFIGURED but unusable native source is a
setup failure (`require_native` raises), never a skip.

The synthetic accepted withdrawal is a TEST double of SYRUP's answer for the LEAF-side logic (proportional limits, identities); it is never
presented as hydrology. The native harness never fakes a lifecycle state: every state comes from the genuine factories.

Native requirements (public snapshot docs/manuals/examples/native_leaf_calendar_cli + native_daily_recharge, same a5b8b18 commit):
  LEAF_NATIVE_BUNDLE      materialized MAPLE 9f bundle directory (CPU or CUDA profile bundle)
  LEAF_NATIVE_CASE_DIR    directory holding `case/` (native 9f case) and `layered_water.yaml` (the example's PUBLIC dir)
  LEAF_NATIVE_BACKEND     numpy | numba | cuda   (default numpy)
  LEAF_NATIVE_DEVICE      CUDA device id (default 0)
  PYTHONPATH              the SYRUP snapshot `src` (package 82e9c011...) and the bundle's MAPLE sources; run CPU tests in an environment that does
                          NOT import CuPy (the genuine CPU source guard refuses otherwise; it must not be bypassed)
'''
from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import json
import os
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leaf import growth, lifecycle_bridge as lb, plant_session as ps, state, traits  # noqa: E402

BIG = 1 << 30
UTC = _dt.timezone.utc
HEX_A, HEX_B, HEX_D = 'a' * 64, 'b' * 64, 'd' * 64
FIXTURES = {'phase1': ROOT / 'examples' / 'phase1' / 'species.json', 'phase2': ROOT / 'examples' / 'phase2' / 'species.json',
            'phase3': ROOT / 'examples' / 'phase3' / 'species.json'}


def simple_edit(doc):
    '''Round numbers (same as tests/test_growth): grass mature 1 kg, rate .5, support 10, live .5, extra .2, activity 1.'''
    g = doc['species']['grass']
    g['growth']['traits'].update({'mature_biomass_kg': 1.0, 'intrinsic_rate_per_year': 0.5, 'biomass_support_kg_year_per_m3': 10.0,
                                  'live_fraction': 0.5, 'extra_growth_rate_per_year': 0.2, 'monthly_activity': [1.0] * 12,
                                  'temperature_min_c': 0.0, 'temperature_opt_c': 20.0, 'temperature_max_c': 40.0})
    g['water_use']['traits']['transpiration_coefficient'] = 0.5


def make_catalog(n_layers=None, fixture='phase1'):
    with open(FIXTURES[fixture], 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    if fixture == 'phase1':
        simple_edit(doc)
    if n_layers is not None:                               # root layer fractions matching the native layer count (a fixture choice)
        for sp in doc['species'].values():
            sp.setdefault('roots', {})['layer_fractions'] = [1.0 / n_layers] * n_layers
    return traits.SpeciesCatalog(doc)


def make_state(cat, *, nx=12, ny=10, dx=1.0, plants=None, species='grass'):
    dom = state.Domain(nx, ny, dx, origin_x_m=0.0, origin_y_m=0.0)
    plants = plants if plants is not None else [(11, 3.5, 3.5, 0.1), (12, 4.2, 3.5, 0.1), (13, 8.5, 6.5, 0.1)]
    recs = [{'plant_id': pid, 'species': species, 'x_m': x, 'y_m': y, 'establishment_ground_m': 0.0, 'agb_dry_kg': b,
             'seasonal_agb_dry_kg': 0.0, 'age_days': 100.0, 'status': 'live'} for pid, x, y, b in plants]
    return state.PlantState.from_records(cat, dom, recs)


def make_config(cat, *, backend='numpy', cadence=7, canopy=True, max_bytes=1 << 28, device=None, reserve=0, rng_seed=None,
                patch_size_m=None, overrides=None):
    table = growth.compile_species(cat, backend, BIG, device=device)
    can = {'max_edges': 1 << 20, 'max_bytes': 1 << 27} if canopy else None
    return ps.SessionConfig(cat, table, backend, 1 << 20, max_bytes, cadence, reserve_bytes=reserve, canopy=can, rng_seed=rng_seed,
                            patch_size_m=patch_size_m, layer_fraction_overrides=overrides)


def anchor(year=2027, month=6, day=1):
    return _dt.datetime(year, month, day, tzinfo=UTC)


def day_ctx(a, i, *, T=20.0, t=True, g=True, hours=0.0, duration_h=None):
    '''Open fragment of nominal day i; `hours` offsets the start inside the day, `duration_h` is the enabled duration (default: to midnight).'''
    start = a + _dt.timedelta(days=i, hours=hours)
    return ps.DayContext(start, T, t, g, duration=None if duration_h is None else _dt.timedelta(hours=duration_h))


def pet_field(n_cells, value=5.0, backend='numpy'):
    f = np.full(n_cells, float(value), dtype=np.float64)
    if backend == 'cuda':
        import cupy as cp
        return cp.asarray(f)
    return f


def identity(lop='c' * 64, hydro=1, seq=0, n_req=0, rev=1, digest=HEX_A, t_s=0.0, ordinal=None, run=HEX_D):
    return lb.ResourceIdentity(lop, digest, rev, seq, hydro, n_req, t_s, 'interevent', hydro if ordinal is None else ordinal, run, False)


def initial_identity(run=HEX_D, lop='c' * 64):
    '''The explicit INITIAL resource cursor: the started daily lineage (no occurrence consumed yet), T history known consumed (0 / 0).'''
    return lb.ResourceIdentity(lop, HEX_A, 0, 0, 0, 0, 0.0, 'interevent', None, run, False)


def start_session(st, cfg, a=None, *, bound=True, session_id=None):
    '''Water-connected session (explicit DayBinding at the nominal origin + initial resource) or a transport-only one (`bound=False`).'''
    a = a or anchor()
    kw = dict(day_binding=ps.DayBinding(HEX_D, a.date().isoformat()), initial_resource=initial_identity()) if bound else {}
    return ps.PlantSession.start(st, cfg, anchor=a, session_id=session_id, **kw)


def make_read(sess, *, hydro, seq=0, n_req=0, rev=1, supply=None, area=1.0, digest=HEX_A, lop='c' * 64, ordinal=None, run=HEX_D):
    if ordinal is None and sess.day_binding is not None:
        ordinal = sess.day_binding.ordinal_of(sess.cursor.date())      # the occurrence of the nominal day being served
    rm = sess.pop.geometry.root_map
    L, N = int(rm.n_layers), int(rm.n_cells)
    sup = np.full((L, N), 1.0e3, dtype=np.float64) if supply is None else np.ascontiguousarray(supply, dtype=np.float64)
    be = sess.config.backend
    if be == 'cuda':                                       # explicit test-side upload of the synthetic availability
        import cupy as cp
        sup = cp.asarray(sup)
    return lb.AvailabilityRead(identity(lop, hydro, seq, n_req, rev, digest, ordinal=ordinal, run=run), sup, area, L, N, be)


def make_accepted(proposal, factor=1.0, *, new_digest=HEX_B):
    '''TEST double of SYRUP's answer for the LEAF-side logic: actual = factor * proposal (a proportional shortfall when factor < 1).'''
    rd = proposal.read
    p = proposal.proposed_m3.reshape(rd.n_layers, rd.n_cells)          # backend array: no implicit transfer
    actual = p * float(factor)
    b = rd.identity
    after = dataclasses.replace(b, parent_digest=new_digest, revision=b.revision + 1, request_sequence=b.request_sequence + 1,
                                n_requests=b.n_requests + 1)
    req, act = float(p.sum()), float(actual.sum())
    return lb.AcceptedWithdrawal(proposal.identity, b, after, actual, p - actual, req, act, float(req - act), rd.cell_area_m2)


def next_read_args(sess):
    '''Identity scalars of the NEXT fresh synthetic read after the last accepted debit (a new daily occurrence).'''
    cur = sess.resource_cursor
    if cur is None:
        return dict(hydro=1, seq=0, n_req=0, rev=1)
    return dict(hydro=cur.hydro_count + 1, seq=cur.request_sequence, n_req=cur.n_requests, rev=cur.revision, lop=cur.lop_sha256)


def hydro_only_read(sess, **kw):
    '''A later physical state with the SAME T counts (hydrology / close / storm transitions are legitimate).'''
    return make_read(sess, **{**next_read_args(sess), **kw})


def run_day(sess, a, i, *, factor=1.0, T=20.0, t=True, g=True, pet=5.0, supply=None, publish=True, hours=0.0, duration_h=None, same_day=False):
    '''One synthetic open fragment; returns (new_or_staged, plan, proposal, staged). `same_day=True` models a LATER fragment of the same SYRUP
    daily occurrence: same ordinal and hydrology count, read from the post-debit state (request sequence = the cursor's).'''
    day = day_ctx(a, i, T=T, t=t, g=g, hours=hours, duration_h=duration_h)
    plan = sess.prepare_daily(day, terrain_pet=pet_field(sess.pop.geometry.root_map.n_cells, pet, sess.config.backend))
    if not t:
        st = sess.stage_dry_day(plan)
        return (sess.publish(st) if publish else st), plan, None, st
    kw = next_read_args(sess)
    cur = sess.resource_cursor
    if same_day:
        kw = dict(hydro=cur.hydro_count, seq=cur.request_sequence, n_req=cur.n_requests, rev=cur.revision, lop=cur.lop_sha256,
                  digest=cur.parent_digest, ordinal=cur.daily_ordinal)
    prop = sess.propose_withdrawal(plan, make_read(sess, supply=supply, **kw))
    st = sess.stage_day(prop, make_accepted(prop, factor, new_digest=(HEX_D if kw.get('digest') == HEX_B else HEX_B)))
    return (sess.publish(st) if publish else st), plan, prop, st


def to_host(a):
    '''TEST-ONLY explicit host transfer (production code never converts implicitly).'''
    return np.array(a.get() if hasattr(a, 'get') else a, copy=True)


# ------------------------------------------------------------------ REAL native SYRUP Phase 1.5 harness
def native_configured():
    return bool(os.environ.get('LEAF_NATIVE_BUNDLE')) and bool(os.environ.get('LEAF_NATIVE_CASE_DIR'))


def require_native():
    '''Setup check of a CONFIGURED native environment: any failure raises (it is not a skip).'''
    import maple_syrup.daily_water  # noqa: F401
    import maple_syrup.water_lifecycle  # noqa: F401
    import yaml  # noqa: F401
    return True


def native_backend():
    return os.environ.get('LEAF_NATIVE_BACKEND', 'numpy')


def native_daily_harness(workdir, *, theta=0.20, bottom='open'):
    '''Genuine factory chain of the SYRUP N15 example (no test helpers, no copied hydrology): returns (lop, accepted boundary state, profile).'''
    import yaml
    from maple_syrup import layered_case as lc
    from maple_syrup import native_case_profile as ncp

    backend = native_backend()
    bundle, public = os.environ['LEAF_NATIVE_BUNDLE'], pathlib.Path(os.environ['LEAF_NATIVE_CASE_DIR'])
    workdir = pathlib.Path(workdir)
    case_dir = workdir / 'case'
    shutil.copytree(public / 'case', case_dir)
    raw = yaml.safe_load((public / 'layered_water.yaml').read_text())
    raw['compute']['backend'] = backend
    raw['soil']['bottom'] = bottom
    raw['initial']['soil_moisture']['by_layer'] = [theta] * len(raw['soil']['layers']['thickness_m'])
    layered = workdir / 'layered_water.yaml'
    layered.write_text(yaml.safe_dump(raw))
    mib = 1 << 20
    digest = lb.QUALIFIED_SYRUP_PACKAGE_SHA256
    if backend == 'cuda':
        from maple_syrup import layered_case_cuda as lcc
        from maple_syrup import native_dense_cuda_profile as ndc
        dev = int(os.environ.get('LEAF_NATIVE_DEVICE', '0'))
        profile = ncp.verify_source_profile(bundle, expected_syrup_digest=digest, profile=ncp.CUDA_SOURCE_PROFILE_NAME)
        prep = ndc.prepare_native_dense_cuda_case(ncp.admit_package_case(profile, case_dir), device_id=dev, caller_host_reserve_bytes=mib,
                                                  caller_device_reserve_bytes=mib)
        setup = lcc.prepare_layered_cuda_case(prep, lc.load_layered_cuda_config(layered), caller_host_reserve_bytes=mib,
                                              caller_device_reserve_bytes=mib, lifecycle_extra_reserve_bytes=mib, native_reserve_bytes=4 * mib)
    else:
        from maple_syrup import native_dense_profile as nd
        profile = ncp.verify_source_profile(bundle, expected_syrup_digest=digest)
        prep = nd.prepare_native_dense_case(ncp.admit_package_case(profile, case_dir), caller_reserve_bytes=mib)
        setup = lc.prepare_layered_case(prep, lc.load_layered_config(layered), caller_reserve_bytes=mib, lifecycle_extra_reserve_bytes=mib)
    return setup.lifecycle_operator, setup.state, profile


def native_plants_for(lop, cat, n=3, B=0.1):
    '''A small LEAF population on the native domain (cell centres).'''
    ny, nx = lop.shape
    dx = float(lop.topology.dx_m)
    pts = [(1 + (3 * k) % max(nx - 1, 1), 1 + (2 * k) % max(ny - 1, 1)) for k in range(n)]
    plants = [(100 + k, (i + 0.5) * dx, (j + 0.5) * dx, B) for k, (i, j) in enumerate(pts)]
    return make_state(cat, nx=nx, ny=ny, dx=dx, plants=plants)
