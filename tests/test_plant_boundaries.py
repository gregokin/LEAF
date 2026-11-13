'''Tests of the EXISTING Phase 2 / 3 boundary mechanics exposed by leaf.plant_session (annual decision, surface event gate, reproduction,
deferred season) with their counters, keyed RNG and checkpoint state. They use the shipped example catalogs (examples/phase2, phase3); a test
that needs an active law is SKIPPED with a named reason when the catalog has none. No law, parameter or response is added. NOT RUN by the
author. Run from the candidate root: python -m unittest tests.test_plant_boundaries -v'''
import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaf import events, plant_checkpoint as pc, plant_session as ps  # noqa: E402
import native_leaf_fixture as nf  # noqa: E402

LIM = pc.CheckpointLimits(1 << 26)
YEAR_END = dict(year=2027, month=12, day=31)


def build(fixture, *, seed=12345, patch=4.0, a=None):
    cat = nf.make_catalog(fixture=fixture)
    sp = cat.species_ids[0]
    a = a or nf.anchor(**YEAR_END)
    for p in (patch, None):
        try:
            cfg = nf.make_config(cat, rng_seed=seed, patch_size_m=p)
            break
        except ps.PlantSessionError:
            continue
    st = nf.make_state(cat, species=sp, plants=[(11, 3.5, 3.5, 0.05), (12, 6.5, 3.5, 0.05), (13, 8.5, 6.5, 0.05)])
    return nf.start_session(st, cfg, a), cat, a


def pet_ok(s):
    return nf.pet_field(s.pop.geometry.root_map.n_cells, 4.0)


def year_end_day(s, a):
    kw = nf.next_read_args(s)
    plan = s.prepare_daily(nf.day_ctx(a, 0), terrain_pet=pet_ok(s))
    prop = s.propose_withdrawal(plan, nf.make_read(s, **kw))
    return s.publish(s.stage_day(prop, nf.make_accepted(prop)))


class AnnualDecisionTests(unittest.TestCase):
    def test_year_end_sets_the_due_gate_and_blocks_open_fragments_until_the_existing_decision_runs(self):
        s, cat, a = build('phase2')
        if not s.laws.demo_on:
            self.skipTest('phase2 example catalog has no active mortality law')
        self.assertTrue(s.phase2_state()['demography_active'])
        c = year_end_day(s, a)
        self.assertEqual(c.due_boundaries()['annual_decision_due_year'], 2027)
        with self.assertRaises(ps.PlantSessionBoundaryError):
            c.prepare_daily(nf.day_ctx(a, 1), terrain_pet=pet_ok(c))                  # growth of the new year must not mix into the old year
        with self.assertRaises(ps.PlantSessionBoundaryError):
            c.stage_transport(ps.TransportDay(a + dt.timedelta(days=1), 'wind'))       # an outstanding decision is never relabelled
        for bad in (2026, 2028):
            with self.assertRaises(ps.PlantSessionStateError):
                c.stage_annual_decision(bad)
        self.assertEqual(c.bio['n_annual'], 0)                                         # refusals consume no random state
        st = c.stage_annual_decision(2027)
        d = c.publish(st)
        self.assertEqual((d.bio['n_annual'], d.bio['annual_due']), (1, None))
        self.assertEqual(d.bio['last_decision']['kind'], 'annual_decision')
        with self.assertRaises(ps.PlantSessionStateError):
            d.stage_annual_decision(2027)                                              # replay refused
        if d.n_live:
            np.testing.assert_array_equal(d.annual_accumulators()['required_support_kg_year'], np.zeros(d.n_live))   # reset
        nf_day = d.prepare_daily(nf.day_ctx(a, 1), terrain_pet=pet_ok(d)) if d.n_live else None
        self.assertTrue(nf_day is None or nf_day.sequence == d.sequence + 1)

    def test_decision_is_keyed_and_deterministic_for_the_same_seed(self):
        out = []
        for _ in range(2):
            s, _, a = build('phase2', seed=777)
            if not s.laws.demo_on:
                self.skipTest('phase2 example catalog has no active mortality law')
            c = year_end_day(s, a)
            d = c.publish(c.stage_annual_decision(2027))
            out.append((d.bio['n_annual'], d.n_live, int(d.base.n), d.stress_counters().tolist() if d.stress_counters() is not None else None))
        self.assertEqual(out[0], out[1])

    def test_transport_year_end_and_deferred_season_order(self):
        s, cat, a = build('phase2')
        if not s.laws.demo_on:
            self.skipTest('phase2 example catalog has no active mortality law')
        t = s.publish(s.stage_transport(ps.TransportDay(a, 'wind')))
        self.assertEqual(t.due_boundaries()['annual_decision_due_year'], 2027)
        self.assertTrue(t.due_boundaries()['deferred_season_due'])
        self.assertEqual(t.phase2_state()['transport_pending']['source'], 'maple_wind')
        u = t.publish(t.stage_deferred_season())                                       # old season is finalised BEFORE the decision
        self.assertFalse(u.due_boundaries()['deferred_season_due'])
        v = u.publish(u.stage_annual_decision(2027))
        self.assertIsNone(v.due_boundaries()['annual_decision_due_year'])


class SurfaceEventTests(unittest.TestCase):
    def event(self, source, start_days, end_days):
        e = mock.create_autospec(events.SurfaceEventDiagnostics, instance=True)
        e.source, e.time_start_days, e.time_end_days, e.n, e.event_id = source, start_days, end_days, 0, 'ev1'
        return e

    def test_event_applies_only_after_the_matching_transport_fragment(self):
        s, cat, a = build('phase2')
        surf = np.zeros((10, 12))
        with self.assertRaises(ps.PlantSessionStateError):                             # no transport awaits a decision
            s.stage_surface_event(self.event('maple_wind', 0.0, 1.0), surf)
        with self.assertRaises(ps.PlantSessionError):
            s.stage_surface_event(object(), surf)
        t = s.publish(s.stage_transport(ps.TransportDay(a, 'wind')))
        tp = t.phase2_state()['transport_pending']
        with self.assertRaises(ps.PlantSessionError):                                  # wrong source
            t.stage_surface_event(self.event('syrup_water', tp['start_days'], tp['end_days']), surf)
        with self.assertRaises(ps.PlantSessionStateError):                             # stale / wrong interval
            t.stage_surface_event(self.event('maple_wind', tp['start_days'] + 3.0, tp['end_days'] + 3.0), surf)
        self.assertIsNotNone(t.phase2_state()['transport_pending'])                    # refusals changed nothing


class ReproductionTests(unittest.TestCase):
    ONE_PATCH = 12.0        # one ecological patch covers the whole 12 x 10 m domain: the existing max_new_plants=5 request fits

    def to_next_year(self, seed=4242, patch=ONE_PATCH):
        s, cat, a = build('phase3', seed=seed, patch=patch)
        if not s.laws.repro_on:
            self.skipTest('phase3 example catalog has no active reproduction law')
        c = year_end_day(s, a)
        if c.due_boundaries()['annual_decision_due_year'] is not None:
            c = c.publish(c.stage_annual_decision(2027))
        return c, a

    def test_opportunity_is_gated_sequential_keyed_and_counted(self):
        s0, cat, a0 = build('phase3', patch=self.ONE_PATCH)
        if not s0.laws.repro_on:
            self.skipTest('phase3 example catalog has no active reproduction law')
        self.assertEqual(s0.due_boundaries()['reproduction_opportunity_year'], None)
        with self.assertRaises(ps.PlantSessionStateError):                             # not yet due (cursor has not crossed 1 Jan 2028)
            s0.stage_reproduction(2027, np.zeros((10, 12)), max_new_plants=5, max_total_records=100, max_attempts=50)
        c, a = self.to_next_year()
        self.assertEqual(c.due_boundaries()['reproduction_opportunity_year'], 2027)
        with self.assertRaises(ps.PlantSessionStateError):                             # skipped / replayed years are refused
            c.stage_reproduction(2028, np.zeros((10, 12)), max_new_plants=5, max_total_records=100, max_attempts=50)
        ids0 = list(c.base.ids)
        st = c.stage_reproduction(2027, np.zeros((10, 12)), max_new_plants=5, max_total_records=100, max_attempts=50)
        d = c.publish(st)
        self.assertEqual((d.bio['n_reproduction'], d.bio['repro_last_year']), (1, 2027))
        self.assertEqual(list(d.base.ids[:len(ids0)]), ids0)                           # existing identities preserved, births appended
        n_new = int(d.base.n) - len(ids0)
        self.assertEqual(d.bio['last_reproduction']['n_births'], n_new)
        if n_new:
            self.assertTrue((d.base.age_days[len(ids0):] == 0.0).all())
            self.assertEqual(d.n_live, c.n_live + n_new)
            self.assertEqual(d.pop.generation, c.pop.generation + 1)
        with self.assertRaises(ps.PlantSessionStateError):
            d.stage_reproduction(2027, np.zeros((10, 12)), max_new_plants=5, max_total_records=100, max_attempts=50)   # replay

    def test_restart_before_the_annual_opportunity_keeps_the_same_keyed_draws_and_refuses_other_seed_or_patch(self):
        import dataclasses
        c, a = self.to_next_year(seed=31)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        out = os.path.join(tmp, 'ck')
        pc.save_biology(out, c, external=c.resource_cursor, limits=LIM)
        bind = pc.ExternalBindings(c.resource_cursor)
        r = pc.load_biology(out, c.config, bindings=bind, limits=LIM)
        args = (2027, np.zeros((10, 12)))
        kw = dict(max_new_plants=5, max_total_records=100, max_attempts=50)
        u = c.publish(c.stage_reproduction(*args, **kw))
        t = r.session.publish(r.session.stage_reproduction(*args, **kw))
        self.assertEqual((int(t.base.n), t.bio['last_reproduction'], t.bio['n_reproduction']),
                         (int(u.base.n), u.bio['last_reproduction'], u.bio['n_reproduction']))
        np.testing.assert_array_equal(t.base.ids, u.base.ids)
        np.testing.assert_array_equal(nf.to_host(t.pop.B), nf.to_host(u.pop.B))
        for name, cfg in (('seed', dataclasses.replace(c.config, rng_seed=32)),
                          ('patch', dataclasses.replace(c.config, patch_size_m=2.0))):
            with self.subTest(name), self.assertRaises(pc.PlantCheckpointError):
                pc.load_biology(out, cfg, bindings=bind, limits=LIM)

    def test_capacity_refusal_is_explicit_and_changes_nothing(self):
        c, a = self.to_next_year(patch=2.0)                  # many small patches: the REQUEST may exceed max_new_plants=5
        before = (c.token, c.bio, int(c.base.n))
        try:
            st = c.stage_reproduction(2027, np.zeros((10, 12)), max_new_plants=5, max_total_records=100, max_attempts=50)
        except ps.PlantSessionError:
            pass                                             # the existing bounded-work refusal: no truncation, no births dropped
        else:
            self.assertLessEqual(int(st.next.base.n) - int(c.base.n), 5)
        self.assertEqual(before, (c.token, c.bio, int(c.base.n)))                 # the accepted parent is untouched either way
        self.assertFalse(c._succ)

    def test_same_seed_same_births_and_decline_consumes_no_random_state(self):
        res = []
        for _ in range(2):
            c, a = self.to_next_year(seed=99)
            d = c.publish(c.stage_reproduction(2027, np.zeros((10, 12)), max_new_plants=5, max_total_records=100, max_attempts=50))
            res.append((int(d.base.n), d.bio['last_reproduction']['n_births']))
        self.assertEqual(res[0], res[1])
        c, a = self.to_next_year()
        d = c.publish(c.stage_decline_reproduction(2027))
        self.assertEqual((d.bio['n_reproduction'], d.bio['repro_last_year'], int(d.base.n)), (0, 2027, int(c.base.n)))
        with self.assertRaises(ps.PlantSessionStateError):
            d.stage_decline_reproduction(2027)

    def test_phase2_3_state_round_trips_through_the_biological_checkpoint(self):
        c, a = self.to_next_year()
        d = c.publish(c.stage_reproduction(2027, np.zeros((10, 12)), max_new_plants=5, max_total_records=100, max_attempts=50))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        ext = d.resource_cursor                      # the exact external reference: same T counts as LEAF consumed
        out = os.path.join(tmp, 'ck')
        man = pc.save_biology(out, d, external=ext, limits=LIM)
        self.assertEqual(man['bio'], d.bio)
        r = pc.load_biology(out, d.config, bindings=pc.ExternalBindings(ext), limits=LIM)
        t = r.session
        self.assertEqual((t.token, t.bio, int(t.base.n), t.n_live, t.geometry_sha256), (d.token, d.bio, int(d.base.n), d.n_live,
                                                                                         d.geometry_sha256))
        if d.stress_counters() is not None:
            np.testing.assert_array_equal(t.stress_counters(), d.stress_counters())
            for k, v in d.annual_accumulators().items():
                np.testing.assert_array_equal(t.annual_accumulators()[k], v)
        self.assertEqual(t.phase3_state()['n_reproduction_opportunities'], d.phase3_state()['n_reproduction_opportunities'])
        # a catalog with active laws needs the explicit seed: a config without it is refused (laws are never silently suppressed)
        cat = nf.make_catalog(fixture='phase3')
        with self.assertRaises(ps.PlantSessionError):
            nf.start_session(nf.make_state(cat, species=cat.species_ids[0]), nf.make_config(cat, rng_seed=None), a)


class CompatibilityTests(unittest.TestCase):
    def test_standalone_modules_are_unchanged_and_importable(self):
        from leaf import checkpoint, interevent, syrup_water  # noqa: F401
        self.assertTrue(hasattr(interevent, 'IntereventSession'))
        self.assertTrue(hasattr(interevent.IntereventSession, 'apply_annual_decision'))
        self.assertTrue(callable(getattr(syrup_water, 'withdraw_plants', None)))      # the old pin keeps its standalone driver API

    def test_session_never_imports_the_standalone_water_bridge(self):
        import subprocess
        code = ("import sys; import leaf.plant_session; sys.exit(1 if 'leaf.syrup_water' in sys.modules or 'leaf.interevent' in sys.modules else 0)")
        r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == '__main__':
    unittest.main()
