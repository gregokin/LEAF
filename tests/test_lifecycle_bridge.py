'''Tests for leaf.lifecycle_bridge: pure value-object / validation tests with live-capability stand-ins (no SYRUP import) and the REAL native SYRUP
Phase 1.5 harness (CPU / CUDA through LEAF_NATIVE_BACKEND). The native tests are skipped ONLY when the native environment is not configured; a
configured but unusable source is a setup failure. NOT RUN by the author. Run from the candidate root:
  python -m unittest tests.test_lifecycle_bridge -v   (CPU native tests: use an environment that does not import CuPy)'''
import dataclasses
import os
import shutil
import sys
import tempfile
import unittest
import weakref
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leaf import lifecycle_bridge as lb, plant_session as ps  # noqa: E402
import native_leaf_fixture as nf  # noqa: E402

CAP = 1 << 30


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def live_read(area=4.0, L=2, N=3):
    '''AvailabilityRead with LIVE capability stand-ins (a parent object, its expected-parent capability and an operator object).'''
    ident = nf.identity(hydro=1)
    leaves = dict(surface_m=Obj(), qx_m2_s=Obj(), qy_m2_s=Obj(), history=Obj(), bed=Obj())
    parent = Obj(ident=ident, payload=Obj(digest=nf.HEX_A, revision=1, **leaves), leaves=leaves)
    lop = Obj()
    exp = Obj(_ref=lambda: parent)
    return lb.AvailabilityRead(ident, np.ones((L, N)) * area, area, L, N, 'numpy', weakref.ref(parent), exp, lop), parent, lop


def fake_result(read, parent, **kw):
    L, N, area = read.n_layers, read.n_cells, read.cell_area_m2
    actual = kw.get('actual', np.full((L, N), 0.25))
    unmet = kw.get('unmet', np.full((L, N), 0.125))
    req = kw.get('requested', float((actual + unmet).sum() * area))
    cs = Obj(revision=2)
    pl = Obj(digest=nf.HEX_B, revision=2, **({k: (v if k not in kw.get('unshare', ()) else Obj()) for k, v in parent.leaves.items()}))
    new_id = dataclasses.replace(read.identity, parent_digest=kw.get('digest', nf.HEX_B), revision=kw.get('rev', 2),
                                 request_sequence=kw.get('seq', 1), n_requests=kw.get('nreq', 1), t_s=kw.get('t_s', 0.0),
                                 hydro_count=kw.get('hydro', 1))
    new_state = Obj(ident=new_id, payload=cs if not kw.get('mix') else pl)
    for k, v in pl.__dict__.items():
        if k not in ('digest', 'revision'):
            setattr(cs, k, v)
    rec = Obj(identity=kw.get('ident', 'leaf:t'), sequence=read.identity.request_sequence, requested_m3=req,
              actual_m3=float(actual.sum() * area), unmet_m3=float(unmet.sum() * area), revision_before=1, revision_after=2,
              parent_token=kw.get('token', nf.HEX_A), t_s=0.0)
    cand = Obj(state=cs, actual_m=actual, unmet_m=unmet, receipt=rec, budget=Obj(bound_m3=1e-12))
    return new_state, cand


def ident_of(lop, st):
    return st.ident


class ValueObjectTests(unittest.TestCase):
    def test_identity_round_trip_and_closed_validation(self):
        i = nf.identity(hydro=3, seq=2, n_req=2, rev=5, run=nf.HEX_D, ordinal=7)
        self.assertEqual(lb.ResourceIdentity.from_dict(i.to_dict()), i)
        with self.assertRaises(lb.BridgeError):
            lb.ResourceIdentity.from_dict({**i.to_dict(), 'extra': 1})
        for kw in (dict(lop_sha256='XYZ'), dict(hydro_count=True), dict(t_s=float('nan')), dict(daily_run='zz'), dict(daily_pending=1)):
            with self.assertRaises(lb.BridgeError):
                dataclasses.replace(i, **kw)

    def test_debit_identity_bounds(self):
        self.assertEqual(lb.debit_identity('a' * 128), 'a' * 128)
        for bad in ('', 'a' * 129, 'é', 'a\nb', 5):
            with self.assertRaises(lb.BridgeError):
                lb.debit_identity(bad)

    def test_single_depth_volume_conversion_and_admission_before_allocation(self):
        r, _, _ = live_read(area=4.0)
        proposed = np.arange(6, dtype=np.float64)
        depth = lb.request_depth_m(r, proposed, max_bytes=CAP)
        self.assertEqual(depth.shape, (2, 3))
        np.testing.assert_allclose(depth * 4.0, proposed.reshape(2, 3))
        with self.assertRaises(lb.BridgeError):
            lb.request_depth_m(r, np.zeros(5), max_bytes=CAP)
        with self.assertRaises(lb.BridgeAdmissionError):
            lb.request_depth_m(r, proposed, max_bytes=10)
        with self.assertRaises(lb.BridgeAdmissionError):
            lb.request_depth_m(r, proposed, max_bytes=CAP, reserve_bytes=CAP)
        e = lb.estimate_conversion_bytes(2, 3, operation='accept')
        self.assertEqual(e['fresh_bytes'], 2 * 8 * 6)

    def test_source_selection_refuses_before_any_import_and_needs_a_verified_record(self):
        for bad in ('0' * 64, 'not-hex'):
            with self.assertRaises(lb.BridgeError):
                lb.bind_source(bad)
        self.assertEqual(lb.QUALIFIED_SYRUP_PACKAGE_SHA256, '82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107')
        with self.assertRaises(lb.BridgeSourceError):
            lb.read_availability(object(), object(), object(), max_bytes=CAP)


class CapabilityAndResultTests(unittest.TestCase):
    def accept(self, **kw):
        r, parent, lop = live_read()
        new, cand = fake_result(r, parent, **kw)
        with mock.patch.object(lb, 'resource_identity', side_effect=ident_of):
            return lb.accepted_withdrawal(r, 'leaf:t', new, cand, lop=lop, max_bytes=CAP)

    def test_request_args_binds_the_read_instance(self):
        r, parent, lop = live_read()
        self.assertIs(lb.request_args(r, lop, parent)['expected_parent']._ref(), parent)
        other = Obj(ident=r.identity, payload=parent.payload)                        # equal scalars, another instance
        with self.assertRaises(lb.BridgeResultError):
            lb.request_args(r, lop, other)
        with self.assertRaises(lb.BridgeResultError):
            lb.request_args(r, Obj(), parent)
        dead = dataclasses.replace(r, _parent_ref=None)
        with self.assertRaises(lb.BridgeError):
            lb.request_args(dead, lop, parent)
        self.assertFalse(lb.AvailabilityRead(r.identity, r.supply_m3, 4.0, 2, 3, 'numpy').is_live)

    def test_resource_identity_validates_the_actual_operator_association(self):
        class LS(Obj):
            pass

        class LO(Obj):
            pass
        fake_wl = Obj(LifecycleState=LS, LifecycleOperator=LO, expect_parent=lambda p: p)
        parent = LS(lop_sha256=nf.HEX_A, digest=nf.HEX_B, revision=1, request_sequence=0, hydro_count=0, n_requests=0, t_s=0.0,
                    role='interevent', daily=None)
        with mock.patch.object(lb, '_modules', return_value=(fake_wl, Obj())):
            with self.assertRaises(lb.BridgeError):                                  # a caller-supplied lop with another hash is refused
                lb.resource_identity(LO(sha256=nf.HEX_B), parent)
            with self.assertRaises(lb.BridgeError):                                  # not a genuine operator object
                lb.resource_identity(Obj(sha256=nf.HEX_A), parent)
            with self.assertRaises(lb.BridgeError):
                lb.resource_identity(LO(sha256=nf.HEX_A), Obj(lop_sha256=nf.HEX_A))
            ident = lb.resource_identity(LO(sha256=nf.HEX_A), parent)
            self.assertEqual((ident.lop_sha256, ident.daily_run, ident.daily_ordinal), (nf.HEX_A, None, None))
            d = Obj(ordinal=3, run_digest=nf.HEX_D, pending=None)
            parent.daily = d
            ident = lb.resource_identity(LO(sha256=nf.HEX_A), parent)
            self.assertEqual((ident.daily_run, ident.daily_ordinal, ident.daily_pending), (nf.HEX_D, 2, False))   # LAST consumed ordinal
            # the water-connected read additionally needs a completed lineage (no cursor / pending storm / zero occurrences)
            fake_lal = Obj(required_availability_bytes=lambda lop, extra_reserve_bytes=0: Obj(required_bytes=1))
            src = lb.BridgeSource(lb.QUALIFIED_SYRUP_PACKAGE_SHA256, 'c', 'b', '.', ())
            with mock.patch.object(lb, '_check_source'), mock.patch.object(lb, '_modules', return_value=(fake_wl, fake_lal)):
                for daily in (None, Obj(ordinal=0, run_digest=nf.HEX_D, pending=None), Obj(ordinal=2, run_digest=nf.HEX_D, pending=Obj())):
                    parent.daily = daily
                    lop = LO(sha256=nf.HEX_A, n_layers=2, n_cells=3)
                    with self.assertRaises(lb.BridgeError):
                        lb.read_availability(src, lop, parent, max_bytes=CAP)

    def test_valid_successor_is_accepted_and_converted_once(self):
        acc = self.accept()
        np.testing.assert_allclose(acc.actual_m3, 0.25 * 4.0)
        np.testing.assert_allclose(acc.unmet_m3, 0.125 * 4.0)
        self.assertEqual((acc.after.request_sequence, acc.after.n_requests), (1, 1))

    def test_refusals(self):
        cases = [dict(t_s=1.0), dict(hydro=2), dict(seq=2), dict(nreq=2), dict(seq=0, nreq=0), dict(digest=nf.HEX_A),
                 dict(ident='leaf:other'), dict(token=nf.HEX_D), dict(unshare=('surface_m',)), dict(unshare=('bed',)), dict(unshare=('history',)), dict(mix=True),
                 dict(actual=np.full((2, 3), -0.25), unmet=np.full((2, 3), 0.625)), dict(requested=99.0),
                 dict(actual=np.full((2, 3), np.nan))]
        for kw in cases:
            with self.subTest(kw=list(kw)), self.assertRaises(lb.BridgeResultError):
                self.accept(**kw)

    def test_candidate_of_another_withdrawal_cannot_be_paired_with_this_successor(self):
        r, parent, lop = live_read()
        new, _ = fake_result(r, parent)
        _, other_cand = fake_result(r, parent, actual=np.full((2, 3), 0.5), unmet=np.full((2, 3), 0.0))
        with mock.patch.object(lb, 'resource_identity', side_effect=ident_of):
            with self.assertRaises(lb.BridgeResultError):                            # new_state.payload is not other_cand.state
                lb.accepted_withdrawal(r, 'leaf:t', new, other_cand, lop=lop, max_bytes=CAP)
            with self.assertRaises(lb.BridgeAdmissionError):
                new, cand = fake_result(r, parent)
                lb.accepted_withdrawal(r, 'leaf:t', new, cand, lop=lop, max_bytes=10)


@unittest.skipUnless(nf.native_configured(), 'native SYRUP environment not configured (LEAF_NATIVE_BUNDLE / LEAF_NATIVE_CASE_DIR)')
class NativeDailyTests(unittest.TestCase):
    '''REAL native SYRUP Phase 1.5 (genuine factories, no fake lifecycle state). CPU or CUDA by LEAF_NATIVE_BACKEND.'''

    @classmethod
    def setUpClass(cls):
        nf.require_native()                      # configured but unusable = setup FAILURE (not a skip)
        from maple_syrup import daily_water as dw
        from maple_syrup import water_lifecycle as wl
        cls.dw, cls.wl = dw, wl
        cls.tmp = tempfile.mkdtemp()
        cls.lop, cls.boundary, cls.profile = nf.native_daily_harness(cls.tmp)
        cls.src = lb.bind_source(cls.profile.syrup_package_digest)               # cold verification of the imported package
        cls.xp = cls.lop.xp
        cls.backend = nf.native_backend()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def day_from(self, parent_boundary, ordinal_plan=(2.0, 10.0, 5.0)):
        dw, wl = self.dw, self.wl
        plan = dw.next_plan(parent_boundary, *ordinal_plan)
        res = dw.daily_water_action(self.lop, parent_boundary, expected_parent=wl.expect_parent(parent_boundary), plan=plan,
                                    pet_terrain_m_s=self.xp.full(self.lop.n_cells, 1e-8, dtype=np.float64))
        self.assertEqual(res.status, 'complete')
        return res.state

    def start(self):
        dw, wl = self.dw, self.wl
        return dw.start_daily_lineage(self.lop, self.boundary, expected_parent=wl.expect_parent(self.boundary), run_digest=nf.HEX_D)

    def leaf_session(self, s0=None):
        '''Water-connected LEAF session: explicit DayBinding + the initial resource = the started daily lineage (T history known: none).'''
        s0 = s0 or self.start()
        cat = nf.make_catalog(n_layers=int(self.lop.n_layers))
        dev = int(os.environ.get('LEAF_NATIVE_DEVICE', '0')) if self.backend == 'cuda' else None
        cfg = nf.make_config(cat, backend=self.backend, device=dev)
        return ps.PlantSession.start(nf.native_plants_for(self.lop, cat), cfg, anchor=nf.anchor(),
                                     day_binding=ps.DayBinding(nf.HEX_D, nf.anchor().date().isoformat()),
                                     initial_resource=lb.resource_identity(self.lop, s0))

    def test_visible_daily_sequence_counts_water_closure_clock_and_shared_pet(self):
        wl = self.wl
        s0 = self.start()
        day = self.day_from(s0)
        self.assertEqual(day.daily.ordinal, s0.daily.ordinal + 1)                    # the daily action ran exactly once
        hydro, t_s, n_req = day.hydro_count, day.t_s, day.n_requests
        sess = self.leaf_session(s0)
        pet_m_s = self.xp.full(self.lop.n_cells, 1e-8, dtype=np.float64)             # the SAME shared terrain PET field SYRUP used
        lplan = sess.prepare_daily(nf.day_ctx(nf.anchor(), 0), terrain_pet=lb.pet_m_s_to_mm_per_day(pet_m_s))
        rd = lb.read_availability(self.src, self.lop, day, max_bytes=1 << 30)
        self.assertEqual((day.hydro_count, day.t_s, day.n_requests), (hydro, t_s, n_req))
        self.assertEqual((rd.identity.daily_ordinal, rd.identity.daily_run), (0, nf.HEX_D))
        prop = sess.propose_withdrawal(lplan, rd)
        depth = lb.request_depth_m(rd, prop.proposed_m3, max_bytes=1 << 30)
        new_day, cand = wl.request_withdrawal(self.lop, day, **lb.request_args(rd, self.lop, day), depth_m=depth, identity=prop.identity,
                                              retained_feedback_bytes=prop.retained_feedback_bytes)
        self.assertEqual((new_day.t_s, new_day.hydro_count, new_day.n_requests), (t_s, hydro, n_req + 1))   # zero time, T once
        acc = lb.accepted_withdrawal(rd, prop.identity, new_day, cand, lop=self.lop, max_bytes=1 << 30)
        staged = sess.stage_day(prop, acc)
        child = sess.publish(staged)
        fb = staged.feedback
        # native / allocator bounds, not an arbitrary absolute delta
        self.assertLessEqual(abs(fb.totals['delivered_m3'] - acc.actual_total_m3), float(fb.closure['tolerance_m3']))
        audit = wl.audit_lifecycle(self.lop, new_day)
        self.assertLessEqual(abs(audit['residual_m3']), audit['bound_m3'])
        self.assertLessEqual(abs(audit['transpiration_m3'] - acc.actual_total_m3), audit['bound_m3'])
        self.assertEqual(child.counters()['withdrawals_accepted'], 1)
        # No stored water inventory / SYRUP state in the accepted LEAF sessions: inspect the DATA (method names such as stage_connect_water are
        # only binding metadata): every stored array is 1-D (per plant / species) and no stored object is a SYRUP state, operator or an (L, N) grid.
        shape = (int(self.lop.n_layers), int(self.lop.n_cells))
        syrup_objs = (day, new_day, s0, self.lop, self.boundary)
        for obj in (sess, child):
            for holder in (vars(obj), vars(obj.pop)):
                for key, val in holder.items():
                    self.assertFalse(any(val is o for o in syrup_objs), key)
                    shp = getattr(val, 'shape', None)
                    if shp is not None:
                        self.assertEqual(len(tuple(shp)), 1, f'{key} must be per-plant / per-species, got {tuple(shp)}')
                        self.assertNotEqual(tuple(shp), shape, key)
            self.assertNotIn(shape, [tuple(getattr(v, 'shape', ())) for v in vars(obj.pop).values()])

    def test_real_alternate_physical_branches_with_equal_digests_are_refused(self):
        wl = self.wl
        s0 = self.start()
        a_state, b_state = self.day_from(s0), self.day_from(s0)                      # two REAL daily transitions of ONE parent
        self.assertIsNot(a_state, b_state)
        rd = lb.read_availability(self.src, self.lop, a_state, max_bytes=1 << 30)
        with self.assertRaises(lb.BridgeResultError):
            lb.request_args(rd, self.lop, b_state)                                   # read A, aim at B
        q = self.xp.zeros((self.lop.n_layers, self.lop.n_cells), dtype=np.float64)
        q[0] = 1e-6
        with self.assertRaises(wl.LifecycleError):                                    # SYRUP itself refuses A's capability on B
            wl.request_withdrawal(self.lop, b_state, expected_parent=rd._expected, depth_m=q, identity='leaf:branch')
        nb, cb = wl.request_withdrawal(self.lop, b_state, expected_parent=wl.expect_parent(b_state), depth_m=q, identity='leaf:branch')
        with self.assertRaises(lb.BridgeResultError):                                 # bypass: a capability minted for B cannot pass as A's result
            lb.accepted_withdrawal(rd, 'leaf:branch', nb, cb, lop=self.lop, max_bytes=1 << 30)

    def test_candidate_of_another_real_withdrawal_cannot_be_mixed(self):
        wl = self.wl
        day = self.day_from(self.start())
        rd = lb.read_availability(self.src, self.lop, day, max_bytes=1 << 30)
        shape = (self.lop.n_layers, self.lop.n_cells)
        q1, q2 = self.xp.zeros(shape, dtype=np.float64), self.xp.zeros(shape, dtype=np.float64)
        q1[0], q2[0] = 1e-6, 2e-6
        args = lb.request_args(rd, self.lop, day)
        n1, c1 = wl.request_withdrawal(self.lop, day, **args, depth_m=q1, identity='leaf:same')
        n2, c2 = wl.request_withdrawal(self.lop, day, **args, depth_m=q2, identity='leaf:same')
        ok = lb.accepted_withdrawal(rd, 'leaf:same', n1, c1, lop=self.lop, max_bytes=1 << 30)
        self.assertLess(ok.actual_total_m3, lb.accepted_withdrawal(rd, 'leaf:same', n2, c2, lop=self.lop, max_bytes=1 << 30).actual_total_m3)
        with self.assertRaises(lb.BridgeResultError):                                 # successor(q) with feedback(2q), same label
            lb.accepted_withdrawal(rd, 'leaf:same', n1, c2, lop=self.lop, max_bytes=1 << 30)
        audit = wl.audit_lifecycle(self.lop, n1)
        self.assertLessEqual(abs(audit['transpiration_m3'] - ok.actual_total_m3), audit['bound_m3'])

    def test_wrong_nominal_day_run_and_ordinal_are_refused_by_the_session(self):
        s0 = self.start()
        d1 = self.day_from(s0)
        sess = self.leaf_session(s0)
        later = sess.publish(sess.stage_dry_day(sess.prepare_daily(nf.day_ctx(nf.anchor(), 0, t=False, g=False), terrain_pet=5.0)))
        plan = later.prepare_daily(nf.day_ctx(nf.anchor(), 1), terrain_pet=5.0)       # nominal day 1 expects occurrence ordinal 1
        rd = lb.read_availability(self.src, self.lop, d1, max_bytes=1 << 30)           # but this read is occurrence 0
        with self.assertRaises(ps.PlantSessionStateError):
            later.propose_withdrawal(plan, rd)


if __name__ == '__main__':
    unittest.main()
