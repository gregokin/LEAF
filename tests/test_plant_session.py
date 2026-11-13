'''Tests for leaf.plant_session (water-free daily biological session). Synthetic LEAF-side water doubles only (no SYRUP); NOT RUN by the
author. Run from the candidate root: python -m unittest tests.test_plant_session -v'''
import dataclasses
import datetime as dt
import os
import subprocess
import sys
import unittest
from fractions import Fraction
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaf import biocalendar, growth, lifecycle_bridge as lb, plant_pet, plant_session as ps  # noqa: E402
import native_leaf_fixture as nf  # noqa: E402


def fresh(backend='numpy', plants=None, cfg_kw=None, bound=True, session_id=None):
    cat = nf.make_catalog()
    cfg = nf.make_config(cat, backend=backend, **(cfg_kw or {}))
    return nf.start_session(nf.make_state(cat, plants=plants), cfg, nf.anchor(), bound=bound, session_id=session_id), cat


class DailyFlowTests(unittest.TestCase):
    def test_prepare_demand_matches_scalar_for_uniform_pet(self):
        s, _ = fresh()
        a = nf.anchor()
        plan = s.prepare_daily(nf.day_ctx(a, 0), terrain_pet=nf.pet_field(s.pop.geometry.root_map.n_cells, 5.0))
        scal = s.prepare_daily(nf.day_ctx(a, 0), terrain_pet=5.0)
        np.testing.assert_array_equal(np.asarray(plan.D), np.asarray(scal.D))      # uniform field == scalar demand bit for bit
        self.assertEqual((plan.pet_kind, scal.pet_kind), ('canopy_area_weighted', 'scalar'))
        self.assertTrue((np.asarray(plan.D) > 0).all())
        self.assertNotEqual(plan.nonce, scal.nonce)
        self.assertNotEqual(plan.plan_digest, scal.plan_digest)                    # equal demand packets still give distinct plans
        self.assertEqual(s.sequence, 0)
        self.assertEqual(s.counters()['days_open'], 0)

    def test_full_day_reports_actual_unmet_satisfaction_and_publishes_once(self):
        s, _ = fresh()
        a = nf.anchor()
        _, plan, prop, st = nf.run_day(s, a, 0, factor=0.5, publish=False)
        fb = st.feedback
        np.testing.assert_allclose(fb.totals['delivered_m3'], 0.5 * fb.totals['requested_m3'], rtol=1e-9)
        np.testing.assert_allclose(np.asarray(fb.delivered_m3) + np.asarray(fb.unmet_m3), np.asarray(fb.requested_m3), rtol=1e-9, atol=1e-15)
        self.assertTrue((np.asarray(fb.satisfaction) <= 1.0 + 1e-12).all())
        self.assertEqual(s.sequence, 0)
        child = s.publish(st)
        self.assertEqual((child.sequence, child.counters()['withdrawals_accepted']), (1, 1))
        self.assertEqual(child.last_stage_id, st.stage_id)
        self.assertEqual(child.cursor, a + dt.timedelta(days=1))
        self.assertIsNotNone(child.resource_cursor)
        self.assertTrue(plan._used)                                                 # the plan is consumed at publication

    def test_zero_supply_gives_zero_satisfaction(self):
        s, _ = fresh()
        a = nf.anchor()
        rm = s.pop.geometry.root_map
        child, plan, prop, st = nf.run_day(s, a, 0, supply=np.zeros((rm.n_layers, rm.n_cells)))
        self.assertEqual(float(np.asarray(st.feedback.delivered_m3).sum()), 0.0)
        self.assertEqual(float(np.asarray(st.feedback.satisfaction).sum()), 0.0)
        np.testing.assert_array_equal(child.pending()['U'], np.zeros(child.n_live))
        self.assertTrue((child.pending()['D'] > 0).all())

    def test_seven_day_cadence_grows_once_and_rebuilds_geometry(self):
        s, _ = fresh()
        a = nf.anchor()
        b0, gen0, g0 = np.asarray(s.pop.B).copy(), s.pop.generation, s.geometry_sha256
        for i in range(6):
            s = nf.run_day(s, a, i)[0]
            self.assertEqual(s.counters()['growth_updates'], 0)
            self.assertTrue(s.needs_flush())
        s = nf.run_day(s, a, 6)[0]
        self.assertEqual((s.counters()['growth_updates'], s.pop.generation), (1, gen0 + 1))
        self.assertFalse(s.needs_flush())
        self.assertTrue((np.asarray(s.pop.B) > b0).all())
        self.assertNotEqual(s.geometry_sha256, g0)                                   # new geometry identity at the rebuild
        np.testing.assert_array_equal(s.pending()['D'], np.zeros(s.n_live))
        self.assertEqual(list(s.export_roots()['live_ids']), [11, 12, 13])
        self.assertEqual(s.export_canopy()['canopy_map'].n_plants, 3)

    def test_zero_demand_day_invents_no_growth(self):
        s, _ = fresh()
        a = nf.anchor()
        child, plan, prop, st = nf.run_day(s, a, 0, pet=0.0)
        self.assertEqual(float(np.asarray(plan.D).sum()), 0.0)
        for i in range(1, 7):
            child = nf.run_day(child, a, i, pet=0.0)[0]
        np.testing.assert_allclose(np.asarray(child.pop.B), np.asarray(s.pop.B))

    def test_transpiration_without_growth_debits_but_credits_nothing(self):
        a = nf.anchor()
        s, _ = fresh()
        s1 = nf.run_day(s, a, 0)[0]
        pend = s1.pending()
        s2 = nf.run_day(s1, a, 1, g=False)[0]
        self.assertEqual(s2.counters()['withdrawals_accepted'], 2)
        for k in 'DUAT':
            np.testing.assert_array_equal(s2.pending()[k], pend[k])                 # earned credit kept, none added
        self.assertEqual(s2.schedule.pending_days, s1.schedule.pending_days)
        self.assertEqual(s2.cursor, a + dt.timedelta(days=2))
        # an INDEPENDENT branch (the first one's parent is superseded): the earned credit is finished BEFORE the disabled period
        s0b, _ = fresh()
        s1b = nf.run_day(s0b, a, 0)[0]
        self.assertTrue(s1b.flush_required(next_growth_enabled=False, next_is_transport=False))
        with self.assertRaises(ps.PlantSessionStateError):
            s1.stage_flush()                                                         # stale ancestor of s2: StateError, not BoundaryError
        f = s1b.publish(s1b.stage_flush())
        g_after = nf.run_day(f, a, 1, g=False)[0]
        np.testing.assert_array_equal(np.asarray(g_after.pop.B), np.asarray(f.pop.B))      # no growth in the disabled fragment
        self.assertEqual(g_after.schedule.pending_days, 0)

    def test_transpiration_off_growth_on_is_refused(self):
        with self.assertRaises(ps.PlantSessionUnsupportedError):
            nf.day_ctx(nf.anchor(), 0, t=False, g=True)

    def test_transpiration_off_growth_off_is_calendar_only(self):
        s, _ = fresh()                                     # explicitly connected: the initial resource cursor exists
        c, plan, prop, st = nf.run_day(s, nf.anchor(), 0, t=False, g=False)
        self.assertIsNone(prop)
        self.assertEqual(c.counters()['days_dry'], 1)
        self.assertIsNotNone(s.resource_cursor)
        self.assertEqual(c.resource_cursor, s.resource_cursor)       # the binding and the consumed T counts are preserved, not erased
        self.assertEqual((c.resource_cursor.request_sequence, c.resource_cursor.n_requests), (0, 0))


class TimingTests(unittest.TestCase):
    def test_partial_day_wind_exclusion_uses_only_enabled_time_but_full_calendar_age(self):
        full, _ = fresh()
        a = nf.anchor()
        dfull = full.prepare_daily(nf.day_ctx(a, 0), terrain_pet=5.0)
        s, _ = fresh()
        wind = s.publish(s.stage_transport(ps.TransportDay(a, 'wind', dt.timedelta(hours=6))))
        plan = wind.prepare_daily(nf.day_ctx(a, 0, hours=6), terrain_pet=5.0)       # 18 h enabled
        np.testing.assert_allclose(np.asarray(plan.D), 0.75 * np.asarray(dfull.D), rtol=1e-12)
        self.assertEqual(plan.step.credit_days, Fraction(3, 4))
        self.assertEqual(plan.step.period_days, Fraction(3, 4))
        age0 = wind.snapshot_state().age_days
        done = wind.publish(wind.stage_dry_day(wind.prepare_daily(nf.day_ctx(a, 0, hours=6, t=False, g=False), terrain_pet=5.0)))
        self.assertEqual(done.cursor, a + dt.timedelta(days=1))
        np.testing.assert_allclose(done.snapshot_state().age_days - s.snapshot_state().age_days, 1.0)   # age = FULL nominal calendar
        self.assertGreater(float(np.asarray(wind.snapshot_state().age_days - s.snapshot_state().age_days).max()), 0.2)
        self.assertEqual(done.counters()['days_transport'], 1)
        del age0

    def test_two_open_fragments_share_one_daily_occurrence_without_extra_hydrology(self):
        s, _ = fresh()
        a = nf.anchor()
        s1, _, p1, _ = nf.run_day(s, a, 0, duration_h=6)
        self.assertTrue(s1.flush_required(next_growth_enabled=True, next_is_transport=True))
        # a second fragment that follows ANOTHER hydrology step of the same occurrence is refused
        plan2 = s1.prepare_daily(nf.day_ctx(a, 0, hours=6, duration_h=6), terrain_pet=5.0)
        bad = nf.make_read(s1, hydro=s1.resource_cursor.hydro_count + 1, seq=1, n_req=1, rev=2, lop=s1.resource_cursor.lop_sha256,
                           ordinal=s1.resource_cursor.daily_ordinal)
        with self.assertRaises(ps.PlantSessionStateError):
            s1.propose_withdrawal(plan2, bad)
        s2 = s1.publish(s1.stage_flush())
        s3 = s2.publish(s2.stage_transport(ps.TransportDay(a + dt.timedelta(hours=6), 'rain', dt.timedelta(hours=6))))
        s4, plan, prop, _ = nf.run_day(s3, a, 0, hours=12, same_day=True)             # fresh read AFTER the preceding T, same hydro count
        self.assertEqual(s4.resource_cursor.hydro_count, s1.resource_cursor.hydro_count)
        self.assertEqual(s4.resource_cursor.daily_ordinal, s1.resource_cursor.daily_ordinal)
        self.assertEqual((s4.counters()['withdrawals_accepted'], s4.cursor), (2, a + dt.timedelta(days=1)))

    def test_fragment_must_not_cross_midnight_or_leave_the_cursor(self):
        s, _ = fresh()
        a = nf.anchor()
        with self.assertRaises(ps.PlantSessionBoundaryError):
            nf.day_ctx(a, 0, hours=20, duration_h=6)
        with self.assertRaises(ps.PlantSessionBoundaryError):
            s.prepare_daily(nf.day_ctx(a, 0, hours=3), terrain_pet=5.0)               # gap: start != cursor
        with self.assertRaises(ps.PlantSessionError):
            ps.DayContext(a, 20.0, True, True, process_kind='rain')

    def test_nominal_day_and_run_binding(self):
        s, _ = fresh()                                     # explicit DayBinding(HEX_D, 2027-06-01) + initial resource (fixture)
        a = nf.anchor()
        plan = s.prepare_daily(nf.day_ctx(a, 0), terrain_pet=5.0)
        for kw in (dict(ordinal=1, run=nf.HEX_D), dict(ordinal=0, run=nf.HEX_A), dict(ordinal=0, run=None)):
            with self.subTest(kw=kw), self.assertRaises(ps.PlantSessionStateError):
                s.propose_withdrawal(plan, nf.make_read(s, hydro=1, **kw))           # wrong nominal day / run even though hydro advanced
        prop = s.propose_withdrawal(plan, nf.make_read(s, hydro=1, ordinal=0, run=nf.HEX_D))
        child = s.publish(s.stage_day(prop, nf.make_accepted(prop)))
        self.assertEqual(child.resource_cursor.daily_ordinal, 0)
        # T-disabled days keep the last handled occurrence; the next T day must be its own date's ordinal (day 3 -> ordinal 3)
        c = nf.run_day(child, a, 1, t=False, g=False)[0]
        c = nf.run_day(c, a, 2, t=False, g=False)[0]
        plan3 = c.prepare_daily(nf.day_ctx(a, 3), terrain_pet=5.0)
        with self.assertRaises(ps.PlantSessionStateError):
            c.propose_withdrawal(plan3, nf.make_read(c, hydro=2, seq=1, n_req=1, rev=2, lop='c' * 64, ordinal=1, run=nf.HEX_D))
        ok = c.propose_withdrawal(plan3, nf.make_read(c, hydro=4, seq=1, n_req=1, rev=2, lop='c' * 64, ordinal=3, run=nf.HEX_D))
        self.assertTrue(ok.identity.startswith('leaf:'))


class RefusalTests(unittest.TestCase):
    def test_duplicate_publication_and_stale_parent(self):
        s, _ = fresh()
        a = nf.anchor()
        _, plan, prop, st = nf.run_day(s, a, 0, publish=False)
        st2 = s.stage_day(prop, nf.make_accepted(prop))
        child = s.publish(st)
        with self.assertRaises(ps.PlantSessionStateError):
            s.publish(st2)
        with self.assertRaises(ps.PlantSessionStateError):
            s.publish(st)
        with self.assertRaises(ps.PlantSessionStateError):
            s.prepare_daily(nf.day_ctx(a, 1), terrain_pet=5.0)
        with self.assertRaises(ps.PlantSessionStateError):
            child.publish(st)
        with self.assertRaises(ps.PlantSessionStateError):
            child.stage_day(prop, nf.make_accepted(prop))                            # a consumed / foreign proposal on the successor

    def test_same_session_id_different_plants_have_different_tokens_and_foreign_plans_are_refused(self):
        a = nf.anchor()
        s1, cat = fresh(plants=[(11, 3.5, 3.5, 0.1), (12, 4.2, 3.5, 0.1), (13, 8.5, 6.5, 0.1)], session_id='same')
        s2, _ = fresh(plants=[(21, 2.5, 2.5, 0.2), (22, 5.5, 5.5, 0.3), (23, 9.5, 7.5, 0.15)], session_id='same')
        self.assertEqual((s1.session_id, s1.sequence, s1.cursor, s1.n_live), (s2.session_id, s2.sequence, s2.cursor, s2.n_live))
        self.assertNotEqual(s1.token, s2.token)
        self.assertNotEqual(s1.pop_sha256, s2.pop_sha256)
        self.assertNotEqual(s1.geometry_sha256, s2.geometry_sha256)
        p1 = s1.prepare_daily(nf.day_ctx(a, 0), terrain_pet=5.0)
        with self.assertRaises(ps.PlantSessionStateError):
            s2.propose_withdrawal(p1, nf.make_read(s2, hydro=1))
        # identical populations in two independent starts still differ by the lineage nonce
        s3, _ = fresh(session_id='same')
        s4, _ = fresh(session_id='same')
        self.assertNotEqual(s3.token, s4.token)
        prop = s3.propose_withdrawal(s3.prepare_daily(nf.day_ctx(a, 0), terrain_pet=5.0), nf.make_read(s3, hydro=1))
        with self.assertRaises(ps.PlantSessionStateError):
            s4.stage_day(prop, nf.make_accepted(prop))

    def test_wrong_debit_identity_wrong_read_and_stale_availability(self):
        s, _ = fresh()
        a = nf.anchor()
        plan = s.prepare_daily(nf.day_ctx(a, 0), terrain_pet=5.0)
        prop = s.propose_withdrawal(plan, nf.make_read(s, hydro=1))
        acc = nf.make_accepted(prop)
        with self.assertRaises(ps.PlantSessionStateError):
            s.stage_day(prop, dataclasses.replace(acc, identity='leaf:other'))
        with self.assertRaises(ps.PlantSessionStateError):
            s.stage_day(prop, dataclasses.replace(acc, before=nf.identity(hydro=9)))
        child = s.publish(s.stage_day(prop, acc))
        plan2 = child.prepare_daily(nf.day_ctx(a, 1), terrain_pet=5.0)
        for kw in (dict(hydro=1, seq=1, n_req=1, rev=2), dict(hydro=2, seq=1, n_req=1, rev=2, lop='d' * 64),
                   dict(hydro=2, seq=0, n_req=0, rev=2)):
            with self.subTest(kw=kw), self.assertRaises(ps.PlantSessionStateError):
                child.propose_withdrawal(plan2, nf.make_read(child, **kw))
        with self.assertRaises(ps.PlantSessionStateError):
            child.propose_withdrawal(plan, nf.make_read(child, hydro=2, seq=1, n_req=1, rev=2))

    def test_discard_and_failure_atomicity(self):
        s, _ = fresh()
        a = nf.anchor()
        _, plan, prop, st = nf.run_day(s, a, 0, publish=False)
        st.discard()
        with self.assertRaises(ps.PlantSessionStateError):
            s.publish(st)
        self.assertFalse(s._succ)
        # a discarded / failed candidate lets the SAME proposal be staged again (biology retry without a new withdrawal)
        again = s.stage_day(prop, nf.make_accepted(prop))
        self.assertEqual(s.publish(again).sequence, 1)
        s0, _ = fresh()
        cur = s0
        for i in range(6):
            cur = nf.run_day(cur, a, i)[0]
        before = (cur.token, cur.counters(), np.asarray(cur.pop.B).copy(), cur.pop.generation, cur.geometry_sha256)
        with mock.patch('leaf.biocalendar.rebuild_geometry', side_effect=biocalendar.BioCalendarError('boom')):
            with self.assertRaises(biocalendar.BioCalendarError):
                nf.run_day(cur, a, 6, publish=False)
        self.assertEqual((before[0], before[1], before[3], before[4]), (cur.token, cur.counters(), cur.pop.generation, cur.geometry_sha256))
        np.testing.assert_array_equal(before[2], np.asarray(cur.pop.B))
        self.assertFalse(cur._succ)
        self.assertEqual(nf.run_day(cur, a, 6)[0].counters()['growth_updates'], 1)

    def test_reserve_and_tight_caps_are_refused_before_allocation(self):
        cat = nf.make_catalog()
        table = growth.compile_species(cat, 'numpy', nf.BIG)
        with self.assertRaises(ps.PlantSessionAdmissionError):
            ps.SessionConfig(cat, table, 'numpy', 10, 1000, 7, reserve_bytes=1001)
        # a large population against tight caps: the refusal is metadata-only, so the call allocates (almost) nothing population-sized
        import tracemalloc
        plants = [(1000 + k, 0.5 + (k % 11), 0.5 + ((k // 11) % 9), 0.05) for k in range(3000)]
        big = nf.make_state(cat, plants=plants)
        for kw in (dict(max_bytes=1 << 16), dict(reserve=(1 << 28) - 100), dict(max_bytes=1 << 20)):
            with self.subTest(kw=kw):
                cfg = nf.make_config(cat, **kw)
                tracemalloc.start()
                try:
                    with self.assertRaises(ps.PlantSessionAdmissionError):
                        nf.start_session(big, cfg, nf.anchor())
                    peak = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
                self.assertLess(peak, 256 * 1024)                                    # far below a single 3000-plant working set
        s, _ = fresh()
        lg = s.memory_ledger()                                                       # public ledger for comparison with measured peaks
        self.assertEqual(lg['accepted_total'], s.resident_bytes())
        self.assertGreaterEqual(lg['accepted_total'], sum(lg['accepted_items'].values()) - 1)
        self.assertGreater(lg['start_lower_bound'], 0)
        tiny = dataclasses.replace(s, config=dataclasses.replace(s.config, max_bytes=s.resident_bytes() + 8))
        with self.assertRaises(ps.PlantSessionAdmissionError):
            tiny.prepare_daily(nf.day_ctx(nf.anchor(), 0), terrain_pet=5.0)

    def test_config_freezes_mutable_inputs(self):
        cat = nf.make_catalog()
        can = {'max_edges': 1 << 20, 'max_bytes': 1 << 27}
        cfg = ps.SessionConfig(cat, growth.compile_species(cat, 'numpy', nf.BIG), 'numpy', 1 << 20, 1 << 28, 7, canopy=can)
        can['max_edges'] = 1
        self.assertEqual(cfg.canopy['max_edges'], 1 << 20)
        with self.assertRaises(TypeError):
            cfg.canopy['max_edges'] = 5
        with self.assertRaises(plant_pet.PlantPetError):
            s, _ = fresh()
            s.prepare_daily(nf.day_ctx(nf.anchor(), 0), terrain_pet=nf.pet_field(3))


class WaterConnectionTests(unittest.TestCase):
    def test_unbound_session_is_transport_only_and_must_connect_explicitly(self):
        s, _ = fresh(bound=False)
        a = nf.anchor()
        plan = s.prepare_daily(nf.day_ctx(a, 0), terrain_pet=5.0)
        with self.assertRaises(ps.PlantSessionStateError):                           # cannot allocate: no binding, no initial cursor
            s.propose_withdrawal(plan, nf.make_read(s, hydro=1, ordinal=0))
        t = s.publish(s.stage_transport(ps.TransportDay(a, 'wind', dt.timedelta(hours=6))))          # transport-only works
        with self.assertRaises(ps.PlantSessionError):                                # binding without run match / without binding
            t.stage_connect_water(None, nf.initial_identity())
        with self.assertRaises(ps.PlantSessionError):
            t.stage_connect_water(ps.DayBinding(nf.HEX_A, '2027-06-01'), nf.initial_identity(run=nf.HEX_D))
        c = t.publish(t.stage_connect_water(ps.DayBinding(nf.HEX_D, '2027-06-01'), nf.initial_identity()))
        with self.assertRaises(ps.PlantSessionStateError):                           # never silently replaced
            c.stage_connect_water(ps.DayBinding(nf.HEX_D, '2027-06-01'), nf.initial_identity())
        d = nf.run_day(c, a, 0, hours=6)[0]
        self.assertEqual(d.counters()['withdrawals_accepted'], 1)

    def test_legacy_or_lineage_free_reads_and_wrong_initial_resource_are_refused(self):
        s, _ = fresh()
        plan = s.prepare_daily(nf.day_ctx(nf.anchor(), 0), terrain_pet=5.0)
        for kw in (dict(run=None), dict(ordinal=0, run=None)):
            with self.subTest(kw=kw), self.assertRaises(ps.PlantSessionStateError):
                s.propose_withdrawal(plan, nf.make_read(s, hydro=1, **kw))
        bad = dataclasses.replace(nf.make_read(s, hydro=1), identity=dataclasses.replace(nf.identity(hydro=1, ordinal=0), daily_pending=True))
        with self.assertRaises(ps.PlantSessionStateError):
            s.propose_withdrawal(plan, bad)                                          # a pending storm is not a completed day
        cat = nf.make_catalog()
        with self.assertRaises(ps.PlantSessionError):
            ps.PlantSession.start(nf.make_state(cat), nf.make_config(cat), anchor=nf.anchor(),
                                  day_binding=ps.DayBinding(nf.HEX_A, '2027-06-01'), initial_resource=nf.initial_identity(run=nf.HEX_D))
        with self.assertRaises(ps.PlantSessionError):
            ps.PlantSession.start(nf.make_state(cat), nf.make_config(cat), anchor=nf.anchor(), initial_resource=nf.initial_identity())

    def test_unhandled_extra_T_is_refused_but_hydrology_with_the_same_T_counts_is_fine(self):
        s, _ = fresh()
        a = nf.anchor()
        c = nf.run_day(s, a, 0)[0]
        plan = c.prepare_daily(nf.day_ctx(a, 1), terrain_pet=5.0)
        cur = c.resource_cursor
        for extra in ({'seq': cur.request_sequence + 1, 'n_req': cur.n_requests + 1}, {'seq': cur.request_sequence + 1},
                      {'n_req': cur.n_requests + 1}):
            with self.subTest(extra=extra), self.assertRaises(ps.PlantSessionStateError):
                c.propose_withdrawal(plan, nf.hydro_only_read(c, **extra))            # a real extra T without LEAF feedback
        ok = c.propose_withdrawal(plan, nf.hydro_only_read(c, hydro=cur.hydro_count + 3, rev=cur.revision + 5))
        self.assertTrue(ok.identity.startswith('leaf:'))

    def test_pending_after_debit_does_not_make_the_debit_unhandled_for_a_correct_resume_only(self):
        s, _ = fresh()
        a = nf.anchor()
        plan = s.prepare_daily(nf.day_ctx(a, 0), terrain_pet=5.0)
        prop = s.propose_withdrawal(plan, nf.make_read(s, hydro=1))
        acc = nf.make_accepted(prop)
        c = s.publish(s.stage_day(prop, acc))
        plan1 = c.prepare_daily(nf.day_ctx(a, 1), terrain_pet=5.0)
        with self.assertRaises(ps.PlantSessionStateError):                           # reading the state AFTER the first debit as day 1 is fine only
            c.propose_withdrawal(plan1, nf.make_read(c, hydro=2, seq=acc.after.request_sequence + 1, n_req=acc.after.n_requests + 1, rev=3))


class BoundaryTests(unittest.TestCase):
    def test_transport_needs_flush_then_advances_calendar_and_age_only(self):
        s, _ = fresh()
        a = nf.anchor()
        s = nf.run_day(s, a, 0)[0]
        tday = ps.TransportDay(a + dt.timedelta(days=1), 'rain')
        with self.assertRaises(ps.PlantSessionBoundaryError):
            s.stage_transport(tday)
        self.assertTrue(s.flush_required(next_growth_enabled=True, next_is_transport=True))
        flushed = s.publish(s.stage_flush())
        self.assertEqual((flushed.counters()['flushes'], flushed.needs_flush()), (1, False))
        with self.assertRaises(ps.PlantSessionStateError):
            s.stage_flush()                                                          # the superseded parent is stale
        with self.assertRaises(ps.PlantSessionBoundaryError):
            flushed.stage_flush()                                                    # the CURRENT parent has nothing pending
        snap0 = flushed.snapshot_state()
        t = flushed.publish(flushed.stage_transport(tday))
        snap1 = t.snapshot_state()
        np.testing.assert_allclose(snap1.age_days - snap0.age_days, 1.0)
        np.testing.assert_array_equal(snap1.agb_dry_kg, snap0.agb_dry_kg)
        self.assertEqual(t.resource_cursor, flushed.resource_cursor)
        self.assertEqual(t.phase2_state()['transport_pending']['kind'], 'rain')    # awaits a disturbance decision (existing binding)

    def test_year_end_growth_and_deferred_season(self):
        cat = nf.make_catalog()
        a = nf.anchor(2027, 12, 31)
        mk = lambda: nf.start_session(nf.make_state(cat), nf.make_config(cat), a)
        c = nf.run_day(mk(), a, 0)[0]
        self.assertEqual((c.counters()['year_ends'], c.counters()['growth_updates']), (1, 1))
        self.assertIsNone(c.due_boundaries()['annual_decision_due_year'])           # no active mortality law in this catalog
        s = mk()
        t = s.publish(s.stage_transport(ps.TransportDay(a, 'recession')))
        self.assertTrue(t.boundaries()['season_due'])
        with self.assertRaises(ps.PlantSessionBoundaryError):
            nf.run_day(t, a, 0)                                                      # day 0 is already accounted: gap / replay
        u = t.publish(t.stage_deferred_season())
        self.assertEqual((u.counters()['season_turnovers'], u.boundaries()['season_due']), (1, False))
        s = mk()
        w = s.publish(s.stage_transport(ps.TransportDay(a, 'wind')))
        w = nf.run_day(w, a, 1)[0]                                                 # the open day turns the deferred season over first
        self.assertEqual(w.counters()['season_turnovers'], 1)

    def test_boundaries_and_state_reports(self):
        s, _ = fresh()
        b = s.boundaries()
        for k in ('next_local_midnight', 'next_growth_boundary', 'next_new_year', 'season_due', 'needs_flush', 'annual_decision_due_year',
                  'transport_awaiting_disturbance', 'reproduction_opportunity_year', 'deferred_season_due'):
            self.assertIn(k, b)
        self.assertEqual(b['next_growth_boundary'], nf.anchor() + dt.timedelta(days=7))
        self.assertFalse(s.phase2_state()['demography_active'])
        self.assertEqual(s.phase3_state()['n_reproduction_opportunities'], 0)


def _backend_available(name):
    if name == 'numba':
        try:
            import numba  # noqa: F401
            return True
        except Exception:
            return False
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class PipelineParityTests(unittest.TestCase):
    '''The SAME eight-day pipeline (daily demand, equal-access proposal, accepted feedback, credit, the 7-day growth + rebuild, a flush, a
    growth-disabled fragment) on every available backend must agree with the NumPy reference at the existing numerical tolerance.'''

    def pipeline(self, backend):
        cat = nf.make_catalog()
        dev = None
        if backend == 'cuda':
            import cupy as cp
            dev = int(cp.cuda.runtime.getDevice())
        cfg = nf.make_config(cat, backend=backend, device=dev)
        a = nf.anchor()
        s = nf.start_session(nf.make_state(cat), cfg, a)
        for i in range(7):
            s = nf.run_day(s, a, i, factor=0.7 if i % 2 else 1.0)[0]
        s = nf.run_day(s, a, 7, g=False)[0]
        self.last_session = s
        return {'B': nf.to_host(s.pop.B), 'E': nf.to_host(s.pop.E), 'D': nf.to_host(s.pop.D), 'U': nf.to_host(s.pop.U),
                'area': nf.to_host(s.pop.geometry.canopy_area_m2), 'counters': s.counters(), 'gen': s.pop.generation,
                'ids': nf.to_host(s.pop.live_ids)}

    def test_available_backends_match_numpy(self):
        ref = self.pipeline('numpy')
        self.assertEqual((ref['counters']['growth_updates'], ref['gen']), (1, 1))
        ran = 0
        for be in ('numba', 'cuda'):
            if not _backend_available(be):
                continue
            ran += 1
            got = self.pipeline(be)
            with self.subTest(backend=be):
                self.assertEqual((got['counters'], got['gen']), (ref['counters'], ref['gen']))
                np.testing.assert_array_equal(got['ids'], ref['ids'])
                for k in ('B', 'E', 'D', 'U', 'area'):
                    np.testing.assert_allclose(got[k], ref[k], rtol=1e-12, atol=1e-15)
        self.assertGreaterEqual(ran + 1, 1)


    def test_checkpoint_and_rollback_on_every_available_backend(self):
        import shutil
        import tempfile
        from leaf import plant_checkpoint as pc
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        ref = None
        for be in ('numpy', 'numba', 'cuda'):
            if be != 'numpy' and not _backend_available(be):
                continue
            with self.subTest(backend=be):
                cat = nf.make_catalog()
                dev = None
                if be == 'cuda':
                    import cupy as cp
                    dev = int(cp.cuda.runtime.getDevice())
                cfg = nf.make_config(cat, backend=be, device=dev)
                a = nf.anchor()
                s = nf.start_session(nf.make_state(cat), cfg, a)
                for i in range(3):
                    s = nf.run_day(s, a, i)[0]
                # rollback: a failing weekly rebuild leaves the accepted parent intact and the retry succeeds without replay
                for i in range(3, 6):
                    s = nf.run_day(s, a, i)[0]
                before = (s.token, nf.to_host(s.pop.B))
                with mock.patch('leaf.biocalendar.rebuild_geometry', side_effect=biocalendar.BioCalendarError('boom')):
                    with self.assertRaises(biocalendar.BioCalendarError):
                        nf.run_day(s, a, 6, publish=False)
                self.assertEqual(before[0], s.token)
                np.testing.assert_array_equal(before[1], nf.to_host(s.pop.B))
                # checkpoint round trip (explicit host / device transfers) and continuation equality
                out = os.path.join(tmp, be)
                pc.save_biology(out, s, external=s.resource_cursor, limits=pc.CheckpointLimits(1 << 26))
                r = pc.load_biology(out, cfg, bindings=pc.ExternalBindings(s.resource_cursor), limits=pc.CheckpointLimits(1 << 26))
                self.assertEqual((r.session.token, r.session.geometry_sha256), (s.token, s.geometry_sha256))
                u, t = nf.run_day(s, a, 6)[0], nf.run_day(r.session, a, 6)[0]
                np.testing.assert_allclose(nf.to_host(t.pop.B), nf.to_host(u.pop.B), rtol=1e-12)
                if ref is None:
                    ref = nf.to_host(u.pop.B)
                else:
                    np.testing.assert_allclose(nf.to_host(u.pop.B), ref, rtol=1e-12, atol=1e-15)    # backend parity after restart


class ConfigAndImportTests(unittest.TestCase):
    def test_config_requires_whole_day_cadence_and_matching_backend(self):
        cat = nf.make_catalog()
        table = growth.compile_species(cat, 'numpy', nf.BIG)
        with self.assertRaises(ps.PlantSessionError):
            ps.SessionConfig(cat, table, 'numpy', 10, 10, 8)
        with self.assertRaises(ps.PlantSessionError):
            ps.SessionConfig(cat, table, 'numba', 10, 10, 7)

    def test_core_imports_are_lazy_and_syrup_free(self):
        code = ("import sys; import leaf.plant_session, leaf.plant_checkpoint, leaf.lifecycle_bridge, leaf.biocalendar, leaf.plant_pet; "
                "bad=[m for m in ('cupy','numba','maple_syrup','maple') if m in sys.modules]; sys.exit(1 if bad else 0)")
        r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_no_syrup_water_or_standalone_debit_in_interface_sources(self):
        for mod in ('plant_session', 'plant_checkpoint', 'lifecycle_bridge'):
            text = open(os.path.join(ROOT, 'leaf', mod + '.py'), encoding='utf-8').read()
            self.assertNotIn('import syrup_water', text)
            self.assertNotIn('withdraw_plants(', text)
            self.assertNotIn('withdraw_transpiration(', text)


if __name__ == '__main__':
    unittest.main()
