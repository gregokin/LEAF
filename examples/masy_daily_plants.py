#!/usr/bin/env python
"""MASY-style daily plant loop on a REAL native SYRUP Phase 1.5 lifecycle (a developer HANDOFF example; not a MASY scheduler, not a case run).

Every physical call is VISIBLE here, at the "MASY" call site: LEAF never calls SYRUP hydrology or a withdrawal. Per nominal day:

    1  MASY completes the day's wind / precipitation treatment (here: a plain rain-free or small-rain day from --p-mm)
    2  SYRUP ONCE:   dw.daily_water_action(...)                       (small rain, redistribution, drainage, E)
    3  fresh read:   lb.read_availability(...)  == lifecycle_availability.layer_availability_m (pure, zero time)
    4  LEAF:         session.prepare_daily(...) -> session.propose_withdrawal(plan, read)        (water-free)
    5  SYRUP ONCE:   wl.request_withdrawal(lop, day, **lb.request_args(read, lop, day), depth_m=..., identity=..., retained_feedback_bytes=...)
                     (the expected-parent capability minted at READ time; ZERO time: t_s and hydro_count unchanged, T request count + 1)
    6  LEAF:         acc = lb.accepted_withdrawal(...); staged = session.stage_day(proposal, acc)  (pure candidate)
    7  MASY accepts the pair:  closed = wl.close_interevent(...); session = session.publish(staged)

Usage (CPU; run with the SYRUP N15 snapshot `src` and the bundle's MAPLE sources on PYTHONPATH):
    python examples/masy_daily_plants.py --bundle <materialized-9f-bundle> --case-dir <native_leaf_calendar_cli dir> \
        --workdir /tmp/masy_daily_NEW --days 3 [--backend numpy|numba|cuda]
P / I / T, the PET value and the plant biomass below are ILLUSTRATIVE test values, not calibrated weather or ecology.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT =Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))          # reuses the genuine factory chain of tests/native_leaf_fixture.py (no copied hydrology)

RUN_DIGEST = 'ab' * 32


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--case-dir', required=True, help='directory with case/ and layered_water.yaml (the SYRUP example public dir)')
    ap.add_argument('--workdir', required=True, help='a NEW directory (nothing is overwritten)')
    ap.add_argument('--backend', choices=('numpy', 'numba', 'cuda'), default='numpy')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--days', type=int, default=3)
    ap.add_argument('--p-mm', type=float, default=0.0, help='illustrative daily precipitation, mm (0 = no-op day; 0 < P <= T recharges)')
    ap.add_argument('--i-mm-h', type=float, default=10.0)
    ap.add_argument('--t-mm', type=float, default=5.0, help='amount threshold T, mm (no scientific default exists)')
    ap.add_argument('--pet-m-s', type=float, default=1e-8, help='illustrative terrain PET, m/s (applied to every cell)')
    ap.add_argument('--checkpoint-dir', help='optional NEW directory for the biological-only checkpoint of the last accepted session')
    a = ap.parse_args(argv)
    if os.path.lexists(a.workdir):
        raise SystemExit(f'refusing: {a.workdir} exists')
    os.environ.update(LEAF_NATIVE_BUNDLE=a.bundle, LEAF_NATIVE_CASE_DIR=a.case_dir, LEAF_NATIVE_BACKEND=a.backend,
                      LEAF_NATIVE_DEVICE=str(a.device))
    Path(a.workdir).mkdir(parents=False, exist_ok=False)

    import native_leaf_fixture as nf
    from leaf import lifecycle_bridge as lb
    from leaf import plant_checkpoint as pc
    from leaf import plant_session as ps
    from maple_syrup import daily_water as dw
    from maple_syrup import water_lifecycle as wl

    lop, boundary, profile = nf.native_daily_harness(a.workdir)
    src = lb.bind_source(profile.syrup_package_digest)                         # explicit Phase 1.5 source selection
    xp = lop.xp
    cat = nf.make_catalog(n_layers=int(lop.n_layers))
    cfg = nf.make_config(cat, backend=a.backend, device=a.device if a.backend == 'cuda' else None, cadence=7)
    anchor = nf.anchor()
    cap = cfg.max_bytes
    # explicit MASY nominal-day <-> SYRUP daily-run mapping: occurrence ordinal k belongs to origin_date + k days (never native seconds)
    state = dw.start_daily_lineage(lop, boundary, expected_parent=wl.expect_parent(boundary), run_digest=RUN_DIGEST)
    # water connection is EXPLICIT: the binding AND the initial resource (the started lineage: its T history is known consumed)
    session = ps.PlantSession.start(nf.native_plants_for(lop, cat), cfg, anchor=anchor,
                                    day_binding=ps.DayBinding(RUN_DIGEST, anchor.date().isoformat()),
                                    initial_resource=lb.resource_identity(lop, state))
    pet_m_s = xp.full(lop.n_cells, a.pet_m_s, dtype=np.float64)
    pet_mm = lb.pet_m_s_to_mm_per_day(pet_m_s)                                  # caller-owned unit conversion
    out = []
    for i in range(a.days):
        # 1+2: MASY's day, then ONE SYRUP daily action
        plan = dw.next_plan(state, a.p_mm, a.i_mm_h, a.t_mm)
        res = dw.daily_water_action(lop, state, expected_parent=wl.expect_parent(state), plan=plan, pet_terrain_m_s=pet_m_s)
        if res.status != 'complete':
            raise SystemExit(f'day {i}: SYRUP daily action did not complete ({res.status}); a storm branch needs the begin / finish '
                             'storm calls at this call site (not shown)')
        day = res.state
        before = (day.t_s, day.hydro_count, day.n_requests)
        # 3: fresh availability (pure read), 4: LEAF demand + proposal
        # ONE shared byte cap for native + LEAF: `reserve` is what the caller keeps live besides the call (the accepted LEAF parent + its reserve)
        reserve = session.config.reserve_bytes + session.resident_bytes()
        read = lb.read_availability(src, lop, day, max_bytes=cap, reserve_bytes=reserve)
        lplan = session.prepare_daily(nf.day_ctx(anchor, i, T=20.0, t=True, g=True), terrain_pet=pet_mm)
        prop = session.propose_withdrawal(lplan, read)
        # 5: ONE zero-time SYRUP T request (visible here); the capability tied to the READ is passed in, so SYRUP itself refuses another branch
        new_day, cand = wl.request_withdrawal(lop, day, **lb.request_args(read, lop, day),
                                              depth_m=lb.request_depth_m(read, prop.proposed_m3, max_bytes=cap, reserve_bytes=reserve),
                                              identity=prop.identity, retained_feedback_bytes=prop.retained_feedback_bytes)
        assert (new_day.t_s, new_day.hydro_count, new_day.n_requests) == (before[0], before[1], before[2] + 1)
        # 6: LEAF stage from the ACTUAL result;  7: MASY accepts both halves
        acc = lb.accepted_withdrawal(read, prop.identity, new_day, cand, lop=lop, max_bytes=cap, reserve_bytes=reserve)
        staged = session.stage_day(prop, acc)
        state = wl.close_interevent(lop, new_day, expected_parent=wl.expect_parent(new_day))
        session = session.publish(staged)
        fb = staged.feedback
        out.append({'date': (anchor + timedelta(days=i)).date().isoformat(), 'day': i, 'requested_m3': fb.totals['requested_m3'],
                    'delivered_m3': fb.totals['delivered_m3'], 'unmet_m3': fb.totals['unmet_m3'],
                    'n_plants_unsatisfied': fb.totals['n_plants_unsatisfied'], 'syrup_t_requests': new_day.n_requests,
                    'syrup_clock_s': new_day.t_s, 'syrup_hydro_count': new_day.hydro_count, 'growth_updates': session.counters()['growth_updates']})
    if a.checkpoint_dir:
        # biological only: SYRUP checkpoints its own water. `external` is the EXACT accepted SYRUP reference paired with this biology (here the
        # boundary closed after the last T), kept distinct from the last consumed T cursor stored inside the session.
        pc.save_biology(a.checkpoint_dir, session, external=lb.resource_identity(lop, state), limits=pc.CheckpointLimits(1 << 30),
                        syrup_source=src.package_sha256)
    audit = wl.audit_lifecycle(lop, state)
    print(json.dumps({'days': out, 'counters': session.counters(), 'due_boundaries': session.due_boundaries(),
                      'syrup_closure': {'residual_m3': audit['residual_m3'], 'bound_m3': audit['bound_m3'],
                                        'transpiration_m3': audit['transpiration_m3']}}, indent=2, default=float))


if __name__ == '__main__':
    sys.exit(main())
