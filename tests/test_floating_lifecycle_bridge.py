"""New floating bridge controls; author has executed none.

Pure tests import no SYRUP. Native tests require an explicit Root-selected frozen
runtime and the unchanged genuine SYRUP test-helper closure on PYTHONPATH. They
perform real native construction/day/debit/close, never a provider/owner double.
Set LEAF_FLOATING_NATIVE=1 only in Root's bounded native qualification harness;
missing operands in an enabled harness are failures, not silently skipped checks.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import pathlib
import sys
import tempfile
import unittest
import weakref

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leaf import floating_lifecycle_bridge as fb  # noqa: E402
from leaf import lifecycle_bridge as lb, plant_session as ps  # noqa: E402

CAP = 1 << 29


class MetadataTests(unittest.TestCase):
    def test_unbound_source_and_connection_refuse_without_native_import(self):
        before = {n for n in sys.modules if n == 'maple_syrup' or n.startswith('maple_syrup.')}
        for source in (None, object(), object.__new__(fb.FloatingSource)):
            with self.subTest(source=type(source).__name__), self.assertRaises(fb.FloatingSourceError):
                fb.connect(source, object(), object(), day_binding=object())
        for connection in (None, object(), object.__new__(fb.FloatingConnection)):
            with self.subTest(connection=type(connection).__name__), self.assertRaises(fb.FloatingCursorError):
                fb.begin_day(connection, object(), leaf_resource_cursor=None)
        self.assertEqual(before, {n for n in sys.modules if n == 'maple_syrup' or n.startswith('maple_syrup.')})

    def test_source_digest_and_commit_validation_precede_native_import(self):
        before = {n for n in sys.modules if n == 'maple_syrup' or n.startswith('maple_syrup.')}
        good = dict(expected_syrup_digest='a' * 64, expected_syrup_commit='b' * 40,
                    expected_maple_commit='c' * 40, max_bytes=CAP)
        for key, value in (('expected_syrup_digest', 'a' * 63), ('expected_syrup_digest', True),
                           ('expected_syrup_commit', 'B' * 40), ('expected_maple_commit', 'unissued')):
            with self.subTest(key=key, value=value), self.assertRaises(fb.FloatingSourceError):
                fb.bind_source(object(), **{**good, key: value})
        with self.assertRaises(lb.BridgeAdmissionError):
            fb.bind_source(object(), **{**good, 'max_bytes': 1})
        self.assertEqual(before, {n for n in sys.modules if n == 'maple_syrup' or n.startswith('maple_syrup.')})

    def test_final_source_recheck_requires_admission_before_native_lookup(self):
        before = {n for n in sys.modules if n == 'maple_syrup' or n.startswith('maple_syrup.')}
        source = object.__new__(fb.FloatingSource)  # negative metadata only; no authority
        with self.assertRaises(TypeError):
            source.recheck()
        for cap, reserve in ((fb.COLD_SOURCE_ALLOWANCE_BYTES - 1, 0),
                             (fb.COLD_SOURCE_ALLOWANCE_BYTES, 1)):
            with self.subTest(cap=cap, reserve=reserve), self.assertRaises(lb.BridgeAdmissionError):
                source.recheck(max_bytes=cap, reserve_bytes=reserve)
        with self.assertRaises(fb.FloatingSourceError):
            source.recheck(max_bytes=fb.COLD_SOURCE_ALLOWANCE_BYTES)
        self.assertEqual(before, {n for n in sys.modules if n == 'maple_syrup' or n.startswith('maple_syrup.')})

    def test_bridge_records_cannot_be_directly_constructed(self):
        for kind in (fb.FloatingSource, fb.FloatingConnection):
            with self.subTest(kind=kind.__name__), self.assertRaises(fb.FloatingBridgeError):
                kind()

    def test_n15_value_read_cannot_gain_floating_authority(self):
        ident = lb.ResourceIdentity('a' * 64, 'b' * 64, 0, 0, 0, 0, 0.0)
        read = lb.AvailabilityRead(ident, np.zeros((2, 3)), 1.0, 2, 3, 'numpy')
        with self.assertRaises(fb.FloatingCursorError):
            fb.request_depth_m(read, np.zeros(6), max_bytes=CAP)

    def test_finite_array_validation_refuses_wrong_namespace_shape_dtype_and_values(self):
        fb._array(np.zeros((2, 3)), 'actual', (2, 3))
        for bad in (np.zeros((2, 3), dtype=np.float32), np.zeros((2, 4)),
                    np.zeros((2, 6))[:, ::2], [[0.0] * 3] * 2,
                    np.full((2, 3), -1.0), np.full((2, 3), np.nan), np.full((2, 3), np.inf)):
            with self.subTest(kind=type(bad).__name__), self.assertRaises(fb.FloatingBridgeError):
                fb._array(bad, 'actual', (2, 3))

    def test_conversion_and_validation_workspace_is_admitted_together(self):
        for op in ('read', 'request', 'accept'):
            native = lb.estimate_conversion_bytes(3, 17, operation=op)
            bridge = fb.estimate_bridge_bytes(3, 17, operation=op)
            self.assertGreater(bridge['total_bytes'], native['total_bytes'])
            with self.assertRaises(lb.BridgeAdmissionError):
                lb._admit(bridge['total_bytes'], bridge['total_bytes'] - 1, 0, op)
        with self.assertRaises(lb.BridgeError):
            fb.estimate_bridge_bytes(3, 17, operation='unknown')


@unittest.skipUnless(os.environ.get('LEAF_FLOATING_NATIVE') == '1',
                     'Root-selected genuine floating Native runtime is not enabled')
class NativeFloatingTests(unittest.TestCase):
    """Real native factories with unchanged existing small fixture laws.

    Root must pin bridge_builders, fpond_builders and their genuine Source closure;
    these are ordinary Source helpers, not saved simulation data. The configured
    source profile must be the actual new current profile, not the old N15 name.
    """

    @classmethod
    def setUpClass(cls):
        from maple_syrup import floating_daily as fd, floating_event_daily_bridge as br
        from maple_syrup import native_case_profile as ncp, recharge as rc, soil_evaporation as se
        import bridge_builders as bb
        # Existing LEAF fixture makes only catalog/plant/config metadata here.
        sys.path.insert(0, str(ROOT / 'tests'))
        import native_leaf_fixture as nf
        cls.fd, cls.br, cls.rc, cls.bb, cls.nf = fd, br, rc, bb, nf
        cls.partition = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
        cls.profile = ncp.verify_source_profile(os.environ['LEAF_FLOATING_BUNDLE'],
            expected_syrup_digest=os.environ['LEAF_FLOATING_SYRUP_SHA256'],
            profile=os.environ['LEAF_FLOATING_NATIVE_PROFILE'])
        cls.source = fb.bind_source(cls.profile,
            expected_syrup_digest=os.environ['LEAF_FLOATING_SYRUP_SHA256'],
            expected_syrup_commit=os.environ['LEAF_FLOATING_SYRUP_COMMIT'],
            expected_maple_commit=os.environ['LEAF_FLOATING_MAPLE_COMMIT'], max_bytes=CAP)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.lineage, self.boundary, self.factory = self.bb.dry_boundary(pathlib.Path(self.tmp.name), 'numpy', False)
        self.binding = ps.DayBinding(self.bb.RUN, '2027-06-01')
        self.connection = fb.connect(self.source, self.lineage, self.boundary, day_binding=self.binding)
        # Native column and surface define the genuine fixture's row-major shape/area.
        ny, nx = self.boundary.surface_depth_m.shape
        dx = float(np.sqrt(self.boundary.root.column.cell_area_m2))
        self.assertEqual(dx * dx, self.boundary.root.column.cell_area_m2)
        # Explicit authored fixture layer count, matching the real native column;
        # the old nf.make_catalog helper does not rewrite doc['root_layers'].
        with (ROOT / 'examples' / 'phase1' / 'species.json').open(encoding='utf-8') as fh:
            doc = json.load(fh)
        self.nf.simple_edit(doc)
        L = self.boundary.root.column.n_layers
        doc['root_layers'] = L
        for species in doc['species'].values():
            species.setdefault('roots', {})['layer_fractions'] = [1.0 / L] * L
        cat = self.nf.traits.SpeciesCatalog(doc)
        base = self.nf.make_state(cat, nx=nx, ny=ny, dx=dx,
                                  plants=[(1, 0.5 * dx, 0.5 * dx, 0.1)])
        cfg = self.nf.make_config(cat, cadence=1)
        self.session = ps.PlantSession.start(base, cfg, anchor=self.nf.anchor(),
            day_binding=self.binding, initial_resource=self.connection.resource_cursor)

    def open_day(self):
        opening = self.br.open_day(self.lineage, self.boundary, partition=self.partition,
            run_digest=self.bb.RUN, available_bytes_fn=self.bb.AVAIL)
        plan = self.rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=self.bb.RUN,
                                        ordinal=self.connection.cursor.next_ordinal)
        n = opening.ctx.n_cells
        result = self.br.day_action(opening, plan=plan, pet_terrain_m_s=np.zeros(n),
            cover=np.zeros(n), available_bytes_fn=self.bb.AVAIL)
        self.assertEqual(result.outcome, 'interevent')
        fb.begin_day(self.connection, opening, leaf_resource_cursor=self.session.resource_cursor)
        return result.state

    def prepare(self, parent):
        ordinal = self.connection.cursor.next_ordinal
        nominal = self.nf.anchor() + dt.timedelta(days=ordinal)
        plan = self.session.prepare_daily(ps.DayContext(nominal, 20.0, True, True),
                                          terrain_pet=np.full(parent.ctx.n_cells, 2.0))
        read = fb.read_availability(self.connection, parent,
            leaf_resource_cursor=self.session.resource_cursor, max_bytes=CAP, available_bytes_fn=self.bb.AVAIL)
        proposal = self.session.propose_withdrawal(plan, read)
        depth = fb.request_depth_m(read, proposal.proposed_m3, max_bytes=CAP)
        return read, proposal, depth

    def debit(self, parent, read, depth):
        return self.fd.request_floating_withdrawal(parent, request_m=depth,
            **fb.request_args(read, self.connection, parent), available_bytes_fn=self.bb.AVAIL)

    def commit(self, parent, staged):
        closed = self.fd.close_floating_day(parent, expected_parent=self.fd.expect_parent(parent),
                                           available_bytes_fn=self.bb.AVAIL).state
        self.boundary = self.br.hand_back(self.lineage, closed)
        self.session = self.session.publish(staged)
        return fb.commit_day(self.connection, parent, closed, self.boundary, session=self.session)

    def test_actual_feedback_zero_time_history_and_second_day_native_sequence_reset(self):
        for ordinal in range(2):
            parent = self.open_day()
            before = parent.water.water_m.copy()
            read, proposal, depth = self.prepare(parent)
            outcome = self.debit(parent, read, depth)
            self.assertEqual((outcome.sequence, outcome.state.request_sequence), (0, 1))
            accepted = fb.accepted_withdrawal(self.connection, proposal, outcome, max_bytes=CAP)
            self.assertIs(outcome.state.history, parent.history)
            self.assertIs(outcome.state.surface, parent.surface)
            self.assertEqual(outcome.state.t_s, parent.t_s)
            np.testing.assert_array_equal(parent.water.water_m, before)
            np.testing.assert_array_equal(accepted.actual_m3, outcome.actual_m * read.cell_area_m2)
            staged = self.session.stage_day(proposal, accepted)
            self.assertAlmostEqual(staged.feedback.totals['delivered_m3'], outcome.actual_m3)
            cursor = self.commit(outcome.state, staged)
            self.assertEqual((cursor.next_ordinal, cursor.requests_consumed), (ordinal + 1, ordinal + 1))
            with self.assertRaises(fb.FloatingCursorError):
                fb.commit_day(self.connection, outcome.state, outcome.state, self.boundary, session=self.session)

    def test_explicit_no_t_day_preserves_request_cursor_then_next_t_day_advances(self):
        parent = self.open_day()
        old = self.session.resource_cursor
        plan = self.session.prepare_daily(ps.DayContext(self.nf.anchor(), 20.0, False, False), terrain_pet=0.0)
        cursor = self.commit(parent, self.session.stage_dry_day(plan))
        self.assertEqual((cursor.next_ordinal, cursor.requests_consumed), (1, 0))
        self.assertEqual(self.session.resource_cursor, old)
        parent = self.open_day()
        read, proposal, depth = self.prepare(parent)
        self.assertGreater(read.identity.hydro_count, old.hydro_count)
        outcome = self.debit(parent, read, depth)
        accepted = fb.accepted_withdrawal(self.connection, proposal, outcome, max_bytes=CAP)
        cursor = self.commit(outcome.state, self.session.stage_day(proposal, accepted))
        self.assertEqual((cursor.next_ordinal, cursor.requests_consumed), (2, 1))

    def test_unhandled_native_request_is_refused_even_when_zero_volume(self):
        parent = self.open_day()
        outcome = self.fd.request_floating_withdrawal(parent, expected_parent=self.fd.expect_parent(parent),
            expected_sequence=0, request_m=np.zeros((parent.ctx.n_layers, parent.ctx.n_cells)),
            available_bytes_fn=self.bb.AVAIL)
        with self.assertRaisesRegex(fb.FloatingCursorError, 'unhandled'):
            fb.read_availability(self.connection, outcome.state, leaf_resource_cursor=self.session.resource_cursor,
                                 max_bytes=CAP, available_bytes_fn=self.bb.AVAIL)

    def test_native_exact_parent_and_genuine_outcome_proof_refuses_replacement(self):
        opening = self.br.open_day(self.lineage, self.boundary, partition=self.partition,
            run_digest=self.bb.RUN, available_bytes_fn=self.bb.AVAIL)
        rain = self.rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=self.bb.RUN, ordinal=0)
        kw = dict(plan=rain, pet_terrain_m_s=np.zeros(opening.ctx.n_cells),
                  cover=np.zeros(opening.ctx.n_cells), available_bytes_fn=self.bb.AVAIL)
        parent = self.br.day_action(opening, **kw).state
        alternate = self.br.day_action(opening, **kw).state
        self.assertIsNot(parent, alternate)
        self.assertEqual(parent.digest, alternate.digest)  # genuine equal-digest branches
        fb.begin_day(self.connection, opening, leaf_resource_cursor=self.session.resource_cursor)
        read, proposal, depth = self.prepare(parent)
        outcome = self.debit(parent, read, depth)
        wrong_parent = self.fd.request_floating_withdrawal(alternate,
            expected_parent=self.fd.expect_parent(alternate), expected_sequence=0,
            request_m=depth, available_bytes_fn=self.bb.AVAIL)
        with self.assertRaises(self.fd.DailyRefusal):
            fb.accepted_withdrawal(self.connection, proposal, wrong_parent, max_bytes=CAP)
        bad = dataclasses.replace(outcome, actual_m=outcome.actual_m.copy())
        with self.assertRaises(self.fd.DailyRefusal):  # Native refusal, no mocked provider/proof.
            fb.accepted_withdrawal(self.connection, proposal, bad, max_bytes=CAP)
        self.assertIsNone(self.connection.pending_feedback)
        accepted = fb.accepted_withdrawal(self.connection, proposal, outcome, max_bytes=CAP)
        with self.assertRaises(fb.FloatingCursorError):
            fb.accepted_withdrawal(self.connection, proposal, outcome, max_bytes=CAP)
        self.assertIs(self.connection.pending_feedback, accepted)
        cursor = self.commit(outcome.state, self.session.stage_day(proposal, accepted))
        self.assertEqual(cursor.requests_consumed, 1)
    def test_captured_proof_dependencies_refuse_rebinding_then_recover(self):
        mods = fb._source(self.source)['modules']
        fd, lc = mods['floating_daily'], mods['floating_event_lifecycle']
        rc, ft, st = mods['recharge'], mods['floating_transpiration'], mods['subsurface_transpiration']
        dependencies = ((fd, '_PRODUCER'), (fd, '_WITHDRAWALS'), (fd, '_MINTED'), (fd, '_CAPS'),
                        (lc, '_HOLD_BINDING'), (lc, '_ROOTS'), (lc, '_RECORDS'),
                        (rc, 'verify_plan'), (rc.DailyCursor, 'check_for_consume'),
                        (ft, 'estimate_floating_withdrawal_memory'), (st, '_PACKET'),
                        (st, '_fused_numpy'), (fd, 'rc'), (ft, 'st'))
        original_report = self.source.report()
        for owner, name in dependencies:
            with self.subTest(owner=owner.__name__, name=name):
                original = getattr(owner, name)
                try:
                    setattr(owner, name, object())  # no call into altered Native objects
                    with self.assertRaises(fb.FloatingSourceError):
                        self.source.report()
                finally:
                    setattr(owner, name, original)
                self.assertEqual(self.source.report(), original_report)
        self.source.recheck(max_bytes=CAP)

    def test_bridge_releases_proposal_outcome_and_request_after_acceptance(self):
        parent = self.open_day()
        read, proposal, depth = self.prepare(parent)
        outcome = self.debit(parent, read, depth)
        accepted = fb.accepted_withdrawal(self.connection, proposal, outcome, max_bytes=CAP)
        staged = self.session.stage_day(proposal, accepted)
        child = outcome.state
        day = fb._connection(self.connection)['day']
        self.assertIs(day['postdebit_state'], child)
        self.assertIsInstance(fb._read_proof(read)['parent'], weakref.ReferenceType)
        self.assertIs(fb._read_proof(read)['parent'](), parent)
        self.assertIsNone(fb._read_proof(read)['request'])
        self.assertEqual(accepted.actual_m3.nbytes + accepted.unmet_m3.nbytes,
                         16 * read.n_layers * read.n_cells)
        objects = (read, proposal, outcome, depth, read.supply_m3,
                   outcome.actual_m, outcome.unmet_m, outcome.scale)
        references = tuple(weakref.ref(obj) for obj in objects)
        del objects, read, proposal, outcome, depth
        self.assertTrue(all(ref() is None for ref in references))
        self.assertIs(self.connection.pending_feedback, accepted)
        cursor = self.commit(child, staged)
        self.assertEqual(cursor.requests_consumed, 1)
        self.assertIsNone(self.connection.pending_feedback)



if __name__ == '__main__':
    unittest.main()
