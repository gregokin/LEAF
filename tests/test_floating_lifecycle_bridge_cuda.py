"""Dense CUDA floating bridge controls; the author executed NONE of them (no GPU/CuPy).

MetadataTests import no SYRUP and no CuPy. NativeCudaTests run only in Root's bounded
GPU qualification harness (LEAF_FLOATING_NATIVE_CUDA=1) with:
  LEAF_FLOATING_BUNDLE            materialized MAPLE bundle of the explicit current CUDA source capability
  LEAF_FLOATING_NATIVE_PROFILE    that profile's literal name (device_runtime_capability True; Root pins it)
  LEAF_FLOATING_SYRUP_SHA256 / LEAF_FLOATING_SYRUP_COMMIT / LEAF_FLOATING_MAPLE_COMMIT   explicit expected Source values
  LEAF_FLOATING_SYRUP_TESTS       the SYRUP `tests` directory: the genuine GPU-definition builders of
                                  `phase2_floating_event_daily_bridge_cuda/test_gpu_definitions.py` (`Side`, `dry_kwargs`)
                                  construct the real CUDA dry boundary through the genuine lifecycle; no double here
  LEAF_GPU_DEVICE                 CUDA device id (default 0)
Missing operands in an enabled harness are failures, not skips. Negative fault
injections are labelled; no fake positive physical result class exists here.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.util
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
from leaf import floating_lifecycle_bridge_cuda as fbc  # noqa: E402
from leaf import lifecycle_bridge as lb, plant_session as ps  # noqa: E402

CAP = 1 << 29
NATIVE = {n for n in sys.modules if n == 'maple_syrup' or n.startswith('maple_syrup.') or n == 'cupy'}


def _native_now():
    return {n for n in sys.modules if n == 'maple_syrup' or n.startswith('maple_syrup.') or n == 'cupy'}


class MetadataTests(unittest.TestCase):
    def test_unbound_records_refuse_without_native_or_cupy_import(self):
        before = _native_now()
        pools = fbc.CudaPools(CAP, lambda: CAP, CAP, lambda: CAP)
        for source in (None, object(), object.__new__(fbc.CudaFloatingSource), object.__new__(fb.FloatingSource)):
            with self.subTest(source=type(source).__name__), self.assertRaises(fb.FloatingSourceError):
                fbc.connect(source, object(), object(), day_binding=object())
        for connection in (None, object(), object.__new__(fbc.CudaFloatingConnection), object.__new__(fb.FloatingConnection)):
            with self.subTest(connection=type(connection).__name__), self.assertRaises(fb.FloatingCursorError):
                fbc.begin_day(connection, object(), leaf_resource_cursor=None)
            with self.subTest(connection=type(connection).__name__), self.assertRaises(fb.FloatingCursorError):
                fbc.read_availability(connection, object(), leaf_resource_cursor=None, pools=pools)
        self.assertEqual(before, _native_now())

    def test_cpu_module_refuses_cuda_records_and_keeps_its_strict_default(self):
        with self.assertRaises(fb.FloatingSourceError):
            fb.connect(object.__new__(fbc.CudaFloatingSource), object(), object(), day_binding=object())
        with self.assertRaises(fb.FloatingCursorError):
            fb.begin_day(object.__new__(fbc.CudaFloatingConnection), object(), leaf_resource_cursor=None)
        self.assertEqual(fb.PROFILE, 'leaf.floating_lifecycle_bridge.cpu.v1')
        self.assertEqual((fb._CPU.name, fb._CPU.device_runtime_capability), ('numpy', False))
        self.assertEqual((fbc._CUDA.name, fbc._CUDA.device_runtime_capability), ('cuda', True))
        self.assertNotEqual(fb.PROFILE, fbc.PROFILE)
        self.assertEqual(fb._CPU.roster.module_names, fb._MODULE_NAMES)  # the CPU roster is unchanged

    def test_source_inputs_validate_before_native_import(self):
        before = _native_now()
        good = dict(expected_syrup_digest='a' * 64, expected_syrup_commit='b' * 40, expected_maple_commit='c' * 40, max_bytes=CAP)
        for key, value in (('expected_syrup_digest', 'a' * 63), ('expected_syrup_commit', 'B' * 40), ('expected_maple_commit', None)):
            with self.subTest(key=key), self.assertRaises(fb.FloatingSourceError):
                fbc.bind_source(object(), **{**good, key: value})
        with self.assertRaises(lb.BridgeAdmissionError):
            fbc.bind_source(object(), **{**good, 'max_bytes': 1})
        self.assertEqual(before, _native_now())

    def test_pools_require_real_readers_and_explicit_finite_caps(self):
        ok = fbc.CudaPools(1 << 20, lambda: 1, 1 << 20, lambda: 1, device_reserve_bytes=0, host_reserve_bytes=5)
        self.assertEqual((ok.device_max_bytes, ok.host_reserve_bytes), (1 << 20, 5))
        for kw in (dict(device_max_bytes=0), dict(host_max_bytes=-1), dict(device_max_bytes=1.5), dict(host_max_bytes=True),
                   dict(device_reserve_bytes=-1), dict(host_reserve_bytes=2.0)):
            with self.subTest(kw=kw), self.assertRaises(lb.BridgeError):
                fbc.CudaPools(**{**dict(device_max_bytes=CAP, device_available_bytes_fn=lambda: 1, host_max_bytes=CAP, host_available_bytes_fn=lambda: 1), **kw})
        for kw in (dict(device_available_bytes_fn=None), dict(host_available_bytes_fn=CAP)):
            with self.subTest(kw=kw), self.assertRaises(fb.FloatingBridgeError):
                fbc.CudaPools(**{**dict(device_max_bytes=CAP, device_available_bytes_fn=lambda: 1, host_max_bytes=CAP, host_available_bytes_fn=lambda: 1), **kw})
        with self.assertRaises(fb.FloatingBridgeError):  # a dict/tuple is not the explicit two-pool record
            fbc.read_availability(object(), object(), leaf_resource_cursor=None, pools={'max_bytes': CAP})
        with self.assertRaises(TypeError):  # the CPU keyword shape is not silently accepted
            fbc.read_availability(object(), object(), leaf_resource_cursor=None, max_bytes=CAP, available_bytes_fn=lambda: CAP)

    def test_device_and_host_estimates_are_separate_and_never_summed(self):
        for op in ('read', 'request', 'accept'):
            small, big = fbc.estimate_bridge_bytes(3, 17, operation=op), fbc.estimate_bridge_bytes(3, 170000, operation=op)
            self.assertEqual(small['device_fresh_bytes'], lb.estimate_conversion_bytes(3, 17, operation=op)['fresh_bytes'])
            self.assertGreater(big['device_bytes'], small['device_bytes'])
            self.assertEqual(big['host_bytes'], small['host_bytes'])  # scalar packets only: no L*N host term
            self.assertNotIn('total_bytes', small)
            with self.assertRaises(lb.BridgeAdmissionError):
                lb._admit(small['host_bytes'], small['host_bytes'] - 1, 0, op)
        with self.assertRaises(lb.BridgeError):
            fbc.estimate_bridge_bytes(3, 17, operation='unknown')

    def test_records_cannot_be_constructed_and_api_shape_matches_cpu(self):
        for kind in (fbc.CudaFloatingSource, fbc.CudaFloatingConnection):
            with self.subTest(kind=kind.__name__), self.assertRaises(fb.FloatingBridgeError):
                kind()
        functions = {'bind_source', 'connect', 'begin_day', 'estimate_bridge_bytes', 'read_availability', 'request_args',
                     'request_depth_m', 'accepted_withdrawal', 'commit_day'}
        self.assertTrue(functions <= set(fb.__all__) and functions <= set(fbc.__all__))

    def test_n15_value_read_cannot_gain_cuda_authority(self):
        ident = lb.ResourceIdentity('a' * 64, 'b' * 64, 0, 0, 0, 0, 0.0)
        read = lb.AvailabilityRead(ident, np.zeros((2, 3)), 1.0, 2, 3, 'cuda')
        with self.assertRaises(fb.FloatingCursorError):
            fbc.request_depth_m(read, np.zeros(6), pools=fbc.CudaPools(CAP, lambda: CAP, CAP, lambda: CAP))

    def test_cuda_roster_extends_the_cpu_roster_with_owning_cuda_modules(self):
        r = fbc._ROSTER
        self.assertTrue(set(fb._MODULE_NAMES) < set(r.module_names))
        for name in ('floating_daily_cuda', 'floating_transpiration_cuda', 'floating_soil_layers_cuda', 'floating_event_daily_bridge_cuda'):
            self.assertIn(name, r.module_names)
        self.assertIn('withdraw', r.function_names['floating_transpiration_cuda'])
        self.assertIn('storage_depth', r.function_names['floating_soil_layers_cuda'])
        self.assertIn(('floating_daily', 'fdc', 'floating_daily_cuda'), r.module_aliases)
        self.assertEqual(r.function_names['floating_daily'], fb._FUNCTION_NAMES['floating_daily'])


@unittest.skipUnless(os.environ.get('LEAF_FLOATING_NATIVE_CUDA') == '1',
                     'Root-selected genuine CUDA floating Native runtime is not enabled')
class NativeCudaTests(unittest.TestCase):
    """Real native CUDA factories (genuine lifecycle dry boundary) with the unchanged small fixture laws."""

    @classmethod
    def setUpClass(cls):
        import cupy as cp
        from maple_syrup import floating_daily as fd, floating_event_daily_bridge as br
        from maple_syrup import native_case_profile as ncp, recharge as rc, soil_evaporation as se, subsurface as ss
        from maple.core import backend as mb
        tests = pathlib.Path(os.environ['LEAF_FLOATING_SYRUP_TESTS'])
        for sub in ('phase2_floating_event_daily_bridge_cuda', 'phase2_floating_event_daily_bridge', 'phase2_floating_event_lifecycle_cuda',
                    'phase2_floating_event_lifecycle', 'phase2_floating_event_refresh_cuda', 'phase2_floating_event_refresh', 'phase2_floating_daily',
                    'phase2_floating_checkpoint', 'phase2_floating_pond', 'phase2_floating_checkpoint_cuda', 'phase2_floating_context_checkpoint',
                    'phase2_native_floating_soil', 'phase2_compact_soil_profile', 'pond_event', 'pond_windowed'):
            sys.path.insert(0, str(tests / sub))
        spec = importlib.util.spec_from_file_location('leaf_syrup_gpu_bridge_defs', tests / 'phase2_floating_event_daily_bridge_cuda' / 'test_gpu_definitions.py')
        gd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gd)
        sys.path.insert(0, str(ROOT / 'tests'))
        import native_leaf_fixture as nf
        cls.cp, cls.fd, cls.br, cls.rc, cls.gd, cls.nf, cls.mb = cp, fd, br, rc, gd, nf, mb
        cls.dev = int(os.environ.get('LEAF_GPU_DEVICE', '0'))
        cls.host_reader = staticmethod(ss.default_host_available_bytes)
        cls.partition = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
        cls.profile = ncp.verify_source_profile(os.environ['LEAF_FLOATING_BUNDLE'], expected_syrup_digest=os.environ['LEAF_FLOATING_SYRUP_SHA256'],
                                                profile=os.environ['LEAF_FLOATING_NATIVE_PROFILE'])
        assert cls.profile.device_runtime_capability is True
        kw = dict(expected_syrup_digest=os.environ['LEAF_FLOATING_SYRUP_SHA256'], expected_syrup_commit=os.environ['LEAF_FLOATING_SYRUP_COMMIT'],
                  expected_maple_commit=os.environ['LEAF_FLOATING_MAPLE_COMMIT'], max_bytes=CAP)
        with unittest.TestCase().assertRaises(fb.FloatingSourceError):  # the old public CPU factory still requires capability False
            fb.bind_source(cls.profile, **kw)
        cls.source = fbc.bind_source(cls.profile, **kw)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.side = self.gd.Side(pathlib.Path(self.tmp.name), self.gd.dry_kwargs(), True)
        self.boundary = self.side.boundary()
        self.lineage = self.side.lin
        self.dev_reader = self.side.rdr
        self.hk = dict(available_bytes_fn=self.dev_reader, host_available_bytes_fn=self.host_reader)
        self.pools = fbc.CudaPools(CAP, self.dev_reader, CAP, self.host_reader)
        self.binding = ps.DayBinding(self.gd.RUN, '2027-06-01')
        self.connection = fbc.connect(self.source, self.lineage, self.boundary, day_binding=self.binding)
        column = self.boundary.root.column
        ny, nx = self.boundary.surface_depth_m.shape
        dx = float(np.sqrt(column.cell_area_m2))
        self.assertEqual(dx * dx, column.cell_area_m2)
        with (ROOT / 'examples' / 'phase1' / 'species.json').open(encoding='utf-8') as fh:
            doc = json.load(fh)
        self.nf.simple_edit(doc)
        L = column.n_layers
        doc['root_layers'] = L
        for species in doc['species'].values():
            species.setdefault('roots', {})['layer_fractions'] = [1.0 / L] * L
        cat = self.nf.traits.SpeciesCatalog(doc)
        base = self.nf.make_state(cat, nx=nx, ny=ny, dx=dx, plants=[(1, 0.5 * dx, 0.5 * dx, 0.1)])
        cfg = self.nf.make_config(cat, cadence=1, backend='cuda', device=self.dev)
        self.session = ps.PlantSession.start(base, cfg, anchor=self.nf.anchor(), day_binding=self.binding,
                                             initial_resource=self.connection.resource_cursor)

    def d2h(self):
        return int(self.mb.read_transfer_counters().device_to_host_bytes)

    def open_day(self):
        opening = self.br.open_day(self.lineage, self.boundary, partition=self.partition, run_digest=self.gd.RUN, **self.hk)
        plan = self.rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=self.gd.RUN, ordinal=self.connection.cursor.next_ordinal)
        n = opening.ctx.n_cells
        result = self.br.day_action(opening, plan=plan, pet_terrain_m_s=self.cp.zeros(n), cover=self.cp.zeros(n), **self.hk)
        self.assertEqual(result.outcome, 'interevent')
        fbc.begin_day(self.connection, opening, leaf_resource_cursor=self.session.resource_cursor)
        return result.state

    def prepare(self, parent):
        nominal = self.nf.anchor() + dt.timedelta(days=self.connection.cursor.next_ordinal)
        plan = self.session.prepare_daily(ps.DayContext(nominal, 20.0, True, True), terrain_pet=self.cp.full(parent.ctx.n_cells, 2.0))
        t0 = self.d2h()
        read = fbc.read_availability(self.connection, parent, leaf_resource_cursor=self.session.resource_cursor, pools=self.pools)
        self.assertLessEqual(self.d2h() - t0, 256)  # bounded scalar packets only; no (L, N) download
        self.assertIs(type(read.supply_m3), self.cp.ndarray)
        self.assertEqual((read.backend, int(read.supply_m3.device.id)), ('cuda', self.dev))
        proposal = self.session.propose_withdrawal(plan, read)
        t0 = self.d2h()
        depth = fbc.request_depth_m(read, proposal.proposed_m3, pools=self.pools)
        self.assertLessEqual(self.d2h() - t0, 256)
        self.assertIs(type(depth), self.cp.ndarray)
        return read, proposal, depth

    def debit(self, parent, read, depth):
        return self.fd.request_floating_withdrawal(parent, request_m=depth, **fbc.request_args(read, self.connection, parent), **self.hk)

    def accept(self, proposal, outcome):
        t0 = self.d2h()
        accepted = fbc.accepted_withdrawal(self.connection, proposal, outcome, pools=self.pools)
        self.assertLessEqual(self.d2h() - t0, 512)
        self.assertIs(type(accepted.actual_m3), self.cp.ndarray)
        return accepted

    def commit(self, parent, staged):
        closed = self.fd.close_floating_day(parent, expected_parent=self.fd.expect_parent(parent), **self.hk).state
        self.boundary = self.br.hand_back(self.lineage, closed, **self.hk)
        self.session = self.session.publish(staged)
        return fbc.commit_day(self.connection, parent, closed, self.boundary, session=self.session)

    def test_resident_feedback_two_days_native_sequence_reset_and_bridge_counters(self):
        cp = self.cp
        for ordinal in range(2):
            parent = self.open_day()
            before = cp.array(parent.water.water_m, copy=True)  # test-side device copy; no host mirror
            read, proposal, depth = self.prepare(parent)
            outcome = self.debit(parent, read, depth)
            self.assertEqual((outcome.sequence, outcome.state.request_sequence), (0, 1))
            accepted = self.accept(proposal, outcome)
            self.assertIs(outcome.state.history, parent.history)
            self.assertEqual(outcome.state.t_s, parent.t_s)
            self.assertTrue(bool(cp.array_equal(parent.water.water_m, before)))
            self.assertTrue(bool(cp.array_equal(accepted.actual_m3, outcome.actual_m * read.cell_area_m2)))
            staged = self.session.stage_day(proposal, accepted)
            self.assertAlmostEqual(staged.feedback.totals['delivered_m3'], outcome.actual_m3)
            cursor = self.commit(outcome.state, staged)
            self.assertEqual((cursor.next_ordinal, cursor.requests_consumed), (ordinal + 1, ordinal + 1))
            with self.assertRaises(fb.FloatingCursorError):  # replay of the committed day
                fbc.commit_day(self.connection, outcome.state, outcome.state, self.boundary, session=self.session)

    def test_explicit_no_t_day_then_t_day_advances(self):
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
        cursor = self.commit(outcome.state, self.session.stage_day(proposal, self.accept(proposal, outcome)))
        self.assertEqual((cursor.next_ordinal, cursor.requests_consumed), (2, 1))

    def test_unhandled_native_request_refused_even_when_zero(self):
        parent = self.open_day()
        outcome = self.fd.request_floating_withdrawal(parent, expected_parent=self.fd.expect_parent(parent), expected_sequence=0,
                                                      request_m=self.cp.zeros((parent.ctx.n_layers, parent.ctx.n_cells)), **self.hk)
        with self.assertRaisesRegex(fb.FloatingCursorError, 'unhandled'):
            fbc.read_availability(self.connection, outcome.state, leaf_resource_cursor=self.session.resource_cursor, pools=self.pools)

    def test_wrong_parent_and_fault_injected_outcome_refused_then_replay_refused(self):
        opening = self.br.open_day(self.lineage, self.boundary, partition=self.partition, run_digest=self.gd.RUN, **self.hk)
        rain = self.rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=self.gd.RUN, ordinal=0)
        n = opening.ctx.n_cells
        kw = dict(plan=rain, pet_terrain_m_s=self.cp.zeros(n), cover=self.cp.zeros(n), **self.hk)
        parent = self.br.day_action(opening, **kw).state
        alternate = self.br.day_action(opening, **kw).state
        self.assertEqual(parent.digest, alternate.digest)
        fbc.begin_day(self.connection, opening, leaf_resource_cursor=self.session.resource_cursor)
        read, proposal, depth = self.prepare(parent)
        outcome = self.debit(parent, read, depth)
        wrong = self.fd.request_floating_withdrawal(alternate, expected_parent=self.fd.expect_parent(alternate), expected_sequence=0, request_m=depth, **self.hk)
        with self.assertRaises(self.fd.DailyRefusal):  # genuine equal-digest other branch
            fbc.accepted_withdrawal(self.connection, proposal, wrong, pools=self.pools)
        bad = dataclasses.replace(outcome, actual_m=self.cp.array(outcome.actual_m, copy=True))  # NEGATIVE FAULT INJECTION
        with self.assertRaises(self.fd.DailyRefusal):
            fbc.accepted_withdrawal(self.connection, proposal, bad, pools=self.pools)
        self.assertIsNone(self.connection.pending_feedback)
        accepted = self.accept(proposal, outcome)
        with self.assertRaises(fb.FloatingCursorError):
            fbc.accepted_withdrawal(self.connection, proposal, outcome, pools=self.pools)
        self.assertIs(self.connection.pending_feedback, accepted)
        self.assertEqual(self.commit(outcome.state, self.session.stage_day(proposal, accepted)).requests_consumed, 1)

    def test_wrong_backend_device_and_host_array_refusals(self):
        parent = self.open_day()
        read, proposal, depth = self.prepare(parent)
        host = np.asarray(proposal.proposed_m3.get())  # TEST-ONLY explicit download
        with self.assertRaises(fb.FloatingBridgeError):  # a host proposal has no device authority
            fbc.request_depth_m(read, host, pools=self.pools)
        with self.assertRaises(fb.FloatingCursorError):  # the CPU module refuses the CUDA read/connection
            fb.request_depth_m(read, host, max_bytes=CAP)
        cpu = self.gd.Side(pathlib.Path(self.tmp.name) / 'cpu', self.gd.dry_kwargs(), False)
        cpu_boundary = cpu.boundary()
        with self.assertRaises(fb.FloatingBridgeError):  # a NumPy boundary is refused by the CUDA bridge
            fbc.connect(self.source, cpu.lin, cpu_boundary, day_binding=ps.DayBinding(self.gd.RUN, '2027-06-01'))
        if self.cp.cuda.runtime.getDeviceCount() >= 2:
            other = 1 if self.dev == 0 else 0
            with self.cp.cuda.Device(other), self.assertRaises(fb.FloatingBridgeError):  # gate before any reader
                fbc.read_availability(self.connection, parent, leaf_resource_cursor=self.session.resource_cursor, pools=self.pools)

    def test_separate_pool_admission_refuses_before_any_array_and_retry_works(self):
        parent = self.open_day()
        cur = self.session.resource_cursor
        with self.assertRaises(lb.BridgeAdmissionError):  # LEAF host cap, pure arithmetic
            fbc.read_availability(self.connection, parent, leaf_resource_cursor=cur, pools=fbc.CudaPools(CAP, self.dev_reader, 1, self.host_reader))
        with self.assertRaises(MemoryError):  # native device budget
            fbc.read_availability(self.connection, parent, leaf_resource_cursor=cur, pools=fbc.CudaPools(1, self.dev_reader, CAP, self.host_reader))
        with self.assertRaises(MemoryError):  # an empty device pool is never covered by host bytes
            fbc.read_availability(self.connection, parent, leaf_resource_cursor=cur, pools=fbc.CudaPools(CAP, lambda: 0, CAP, lambda: CAP))
        with self.assertRaises(MemoryError):  # an empty host pool is never covered by device bytes
            fbc.read_availability(self.connection, parent, leaf_resource_cursor=cur, pools=fbc.CudaPools(CAP, lambda: CAP, CAP, lambda: 0))
        self.assertIsNone(fb._connection(self.connection)['day']['read'])
        read, proposal, depth = self.prepare(parent)
        outcome = self.debit(parent, read, depth)
        self.assertEqual(self.commit(outcome.state, self.session.stage_day(proposal, self.accept(proposal, outcome))).requests_consumed, 1)

    def test_stale_cursor_and_generation_refusals(self):
        parent = self.open_day()
        old = self.session.resource_cursor
        read, proposal, depth = self.prepare(parent)
        outcome = self.debit(parent, read, depth)
        staged = self.session.stage_day(proposal, self.accept(proposal, outcome))
        self.commit(outcome.state, staged)
        opening = self.br.open_day(self.lineage, self.boundary, partition=self.partition, **self.hk)
        with self.assertRaises(fb.FloatingCursorError):  # stale LEAF cursor
            fbc.begin_day(self.connection, opening, leaf_resource_cursor=old)
        fbc.begin_day(self.connection, opening, leaf_resource_cursor=self.session.resource_cursor)
        from maple_syrup import floating_event_lifecycle as lc
        with self.assertRaises((fb.FloatingCursorError, lc.LifecycleError)):  # the committed day's interevent is a stale generation
            fbc.read_availability(self.connection, outcome.state, leaf_resource_cursor=self.session.resource_cursor, pools=self.pools)

    def test_bridge_releases_read_proposal_outcome_after_acceptance(self):
        parent = self.open_day()
        read, proposal, depth = self.prepare(parent)
        outcome = self.debit(parent, read, depth)
        accepted = self.accept(proposal, outcome)
        staged = self.session.stage_day(proposal, accepted)
        child, key = outcome.state, id(read)
        self.assertIsNone(fb._read_proof(read)['request'])
        refs = tuple(weakref.ref(o) for o in (read, proposal, outcome))
        del read, proposal, outcome, depth
        self.assertTrue(all(r() is None for r in refs))
        self.assertNotIn(key, fb._READS)
        self.assertIs(self.connection.pending_feedback, accepted)
        self.assertEqual(self.commit(child, staged).requests_consumed, 1)
        self.assertIsNone(self.connection.pending_feedback)

    def test_cuda_proof_dependencies_refuse_rebinding_then_recover(self):
        mods = fb._source(self.source)['modules']
        deps = ((mods['floating_daily_cuda'], 'gate'), (mods['floating_daily_cuda'], 'admit'), (mods['floating_transpiration_cuda'], 'withdraw'),
                (mods['floating_soil_layers_cuda'], 'storage_depth'), (mods['floating_event_daily_bridge_cuda'], 'scalar'),
                (mods['floating_daily'], 'fdc'), (mods['subsurface_transpiration'], '_CUDA_BODY'))
        original = self.source.report()
        self.assertEqual((original['backend'], original['device_runtime_capability']), ('cuda', True))
        for owner, name in deps:
            with self.subTest(owner=owner.__name__, name=name):
                saved = getattr(owner, name)
                try:
                    setattr(owner, name, object())
                    with self.assertRaises(fb.FloatingSourceError):
                        self.source.report()
                finally:
                    setattr(owner, name, saved)
                self.assertEqual(self.source.report(), original)
        self.source.recheck(max_bytes=CAP)


if __name__ == '__main__':
    unittest.main()
