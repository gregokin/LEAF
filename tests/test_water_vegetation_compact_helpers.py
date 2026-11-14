'''leaf.water_vegetation public PURE compact helpers (H1 `compact_content_sha256`, H2 species / stream helpers, H3 `rederive_fields_from_compact`) CPU tests.
Run from the candidate root:  python -m unittest tests.test_water_vegetation_compact_helpers -v
Golden content hashes were produced by the FROZEN pre-refactor module (the identical fixtures) so a refactor that changed one byte of the V3 hash is caught.'''
from __future__ import annotations

import gc
import hashlib
import json
import pathlib
import tracemalloc
import unittest

import numpy as np

from leaf import water_vegetation as wv
from tests import compact_helper_fixtures as fx
from tests.test_water_vegetation import BASELINE_27, ROOT

BIG = fx.BIG
GOLDEN = {
    'masked3': 'sha256:edae06322b26eb6bc7a2408917dacdcd081837577b3a4ac32fc89ebba8f4bb59',
    'periodic_origin': 'sha256:74472ee51bed658de837235c3d086f4c3c954b33a788118c8fce2be93a7fe8fc',
    'session': 'sha256:483f78ccfd5af8d6a1ef2603e4ce920e4ce10da2a279fcd713ec44c0b8f4b28e',
    'empty': 'sha256:63d50fb3e340edb41aeb8e421fa3fc6d75629becae509d47172276eb4d774315',
    'dead_only': 'sha256:6b3ae66514a0e738b3829ee2d01b61ebd9319547b7d95382af1d65fb63e89fd8',
    'zero_weights': 'sha256:e2538ec24587342cf3b8af8c542d1afbf701eebf8001d16a4a60b69aa29c0f10',
}


STREAM_GOLDEN = {
 "masked3": {
  "ids": [
   "grass"
  ],
  "weights": "e7baeb7909609738a047447dae0128d423ae95be4d0b6b00db6f542b86b2d2f2",
  "arrays": [
   "c0b0bd3c3e49bf8dc5976409b40a8446d16b67a4fcbf0a67249d51c2a3088b9f"
  ]
 },
 "periodic_origin": {
  "ids": [
   "grass",
   "shrub"
  ],
  "weights": "3a83f87afe58573ed9c9c4a767af10b824ad593979dbbd48ea3fbdf8a207781c",
  "arrays": [
   "7f6933dbf38d4261966c9c5790f3930427c185fa3b7642d8bdb91d331c6fd791",
   "4e260c601c5795c5a88f109de4bf63a2f1515d0a36e06524aa804e1b70b5b4fd"
  ]
 },
 "session": {
  "ids": [
   "grass",
   "shrub"
  ],
  "weights": "f192dc7d8f8e51b5782248598724a065ac9cf30d5ae200fd886c79b0940c2fc6",
  "arrays": [
   "98cd0edcff70ebdb435664324bbf7429c695b602d49d4439b93f6b9a7d386fdf",
   "3dc463a76fc170607c07b104c3cb531362ce7d6e10c1a34e0c0f370aeae08ce8"
  ]
 },
 "empty": {
  "ids": [
   "grass",
   "shrub"
  ],
  "weights": "124fc60dac41dc47baad781dd0007eeb48fafe636020e7aba99bb0ab9611bd9b",
  "arrays": [
   "67042dfda5683aead81b6055d19c4dba238341f9dd82f49c0e7cc0c19c5f10d1",
   "67042dfda5683aead81b6055d19c4dba238341f9dd82f49c0e7cc0c19c5f10d1"
  ]
 },
 "dead_only": {
  "ids": [
   "grass",
   "shrub"
  ],
  "weights": "124fc60dac41dc47baad781dd0007eeb48fafe636020e7aba99bb0ab9611bd9b",
  "arrays": [
   "67042dfda5683aead81b6055d19c4dba238341f9dd82f49c0e7cc0c19c5f10d1",
   "67042dfda5683aead81b6055d19c4dba238341f9dd82f49c0e7cc0c19c5f10d1"
  ]
 },
 "zero_weights": {
  "ids": [
   "shrub"
  ],
  "weights": "676015682e17135f9985a2b8b443669306e5bebe6f736a50a9555817d68f301c",
  "arrays": [
   "a6e2bdbd8299ef4a6cfd5baf85da144cbe1a412245a8169653a42d3d14fae356"
  ]
 }
}  # produced by the FROZEN C01 module (Leaf28 f45978c1…) on the identical fixtures


def has_numba():
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


def independent_hash(ex):
    '''Re-derived from the EXPORT alone with hashlib only (no module hash function).'''
    h = hashlib.sha256()
    h.update(json.dumps({'schema': ex['schema'], 'version': ex['version'], 'species_ids': list(ex['species_ids']), 'shape': list(ex['shape']), 'cell_area_m2': float(ex['cell_area_m2']).hex(),
                         'n_cells': ex['n_cells'], 'n_entries': ex['n_entries'], 'n_plants': ex['n_plants'], 'root_layers': ex['root_layers']}, sort_keys=True).encode())
    for k in ex['field_order']:
        a = np.ascontiguousarray(ex['arrays'][k])
        h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + hashlib.sha256(a.tobytes()).digest())
    return 'sha256:' + h.hexdigest()


def mutable(ex, **over):
    '''A plain mutable deep copy of an export (every array copied) with optional key overrides, to build corrupted records without touching the sealed projection.'''
    out = {k: v for k, v in ex.items() if k not in ('arrays', 'identity')}
    out['arrays'] = {k: np.array(a) for k, a in ex['arrays'].items()}
    out.update(over)
    return out


def fingerprint(ex):
    return {k: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for k, a in ex['arrays'].items()}


class GoldenHash(unittest.TestCase):
    def test_h1_equals_the_frozen_pre_refactor_hash_the_producer_hash_and_an_independent_derivation_for_every_fixture(self):
        for name in fx.NAMES:
            with self.subTest(name):
                p, _ = fx.build(name, wv)
                ex = p.export_compact()
                self.assertEqual(p.identity['content_sha256'], GOLDEN[name])                          # unchanged by the refactor (initial state AND PlantSession)
                self.assertEqual(wv.compact_content_sha256(ex, max_bytes=BIG), GOLDEN[name])
                self.assertEqual(independent_hash(ex), GOLDEN[name])
        self.assertEqual(wv.VERSION, 3)

    def test_the_source_identity_is_not_part_of_the_content_hash(self):
        p, _ = fx.build('masked3', wv)
        ex = p.export_compact()
        a = mutable(ex)                                                                               # no identity key at all
        b = mutable(ex, identity={'package': {'digest': 'sha256:' + '0' * 64}})
        self.assertEqual(wv.compact_content_sha256(a, max_bytes=BIG), wv.compact_content_sha256(b, max_bytes=BIG))

    def test_a_recorded_hash_must_equal_the_recomputed_one(self):
        p, _ = fx.build('masked3', wv)
        ex = mutable(p.export_compact())
        self.assertEqual(wv.compact_content_sha256(ex, max_bytes=BIG), GOLDEN['masked3'])
        ex['content_sha256'] = 'sha256:' + '1' * 64
        with self.assertRaises(wv.CompactContentError):
            wv.compact_content_sha256(ex, max_bytes=BIG)

    def test_the_export_mapping_is_accepted_directly_without_copy(self):
        p, _ = fx.build('masked3', wv)
        before = tracemalloc.is_tracing()
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        wv.compact_content_sha256(p.export_compact(), max_bytes=BIG)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertLessEqual(peak, wv.estimate_compact_validation_bytes(p.n_cells, p.n_entries, 2, 3, len(p.omitted_ids), max_id_chars=5))
        self.assertFalse(before)


class Corruption(unittest.TestCase):
    '''Every malformed field is refused, nothing is repaired, and the input records are never mutated.'''

    def setUp(self):
        self.p, _ = fx.build('masked3', wv)
        self.ex = self.p.export_compact()

    def refuses(self, rec, what, fn=None):
        fn = fn or (lambda r: wv.compact_content_sha256(r, max_bytes=BIG))
        snap = fingerprint(rec) if isinstance(rec, dict) else None
        with self.assertRaises(wv.CompactContentError, msg=what):
            fn(rec)
        if snap is not None:
            self.assertEqual(fingerprint(rec), snap, what)                                            # the refused input is unchanged

    def test_structure_metadata_is_refused(self):
        ex = self.ex
        m = mutable(ex)
        m['extra'] = 1
        self.refuses(m, 'unknown key')
        m = mutable(ex)
        del m['n_plants']
        self.refuses(m, 'missing key')
        self.refuses(mutable(ex, version=2), 'old version')
        self.refuses(mutable(ex, version=True), 'bool version')
        self.refuses(mutable(ex, schema='other'), 'schema')
        self.refuses(mutable(ex, field_order=tuple(reversed(ex['field_order']))), 'field order')
        self.refuses(mutable(ex, species_ids=('grass', 'grass')), 'duplicate species')
        self.refuses(mutable(ex, species_ids=()), 'empty species')
        self.refuses(mutable(ex, n_cells=True), 'bool count')
        self.refuses(mutable(ex, n_cells=99), 'cell count')
        self.refuses(mutable(ex, n_entries=ex['n_entries'] + 1), 'entry count')
        self.refuses(mutable(ex, n_plants=-1), 'negative plants')
        self.refuses(mutable(ex, root_layers=0), 'layers')
        self.refuses(mutable(ex, shape=(5, 5)), 'shape product')
        self.refuses(mutable(ex, shape=(10,)), 'shape rank')
        self.refuses(mutable(ex, cell_area_m2=0.0), 'area zero')
        self.refuses(mutable(ex, cell_area_m2=float('nan')), 'area nan')
        self.refuses(mutable(ex, cell_area_m2=1), 'area int')
        self.refuses(mutable(ex, content_sha256='sha256:XYZ'), 'bad recorded hash format')
        self.refuses(mutable(ex, n_plants=0), 'entries without plants')
        self.refuses(object(), 'not a mapping')

    def test_array_dtype_shape_layout_and_extra_fields_are_refused(self):
        ex = self.ex
        for k in ex['field_order']:
            m = mutable(ex)
            m['arrays'][k] = m['arrays'][k].astype(np.float32 if m['arrays'][k].dtype.kind == 'f' else np.int16)
            self.refuses(m, f'dtype {k}')
            m = mutable(ex)
            a = m['arrays'][k]
            m['arrays'][k] = np.zeros((a.shape[0] + 1,) + a.shape[1:], dtype=a.dtype)
            self.refuses(m, f'shape {k}')
            m = mutable(ex)
            del m['arrays'][k]
            self.refuses(m, f'missing array {k}')
        m = mutable(ex)
        m['arrays']['junk'] = np.zeros(1)
        self.refuses(m, 'extra array')
        m = mutable(ex)
        m['arrays']['splash_cover'] = np.asfortranarray(np.zeros((10, 10)))[:, 0]                        # non-contiguous view
        self.refuses(m, 'noncontiguous')
        m = mutable(ex)
        m['arrays']['splash_cover'] = list(m['arrays']['splash_cover'])
        self.refuses(m, 'not an ndarray')

    def test_value_constraints_are_refused(self):
        ex = self.ex
        cases = {
            'nan cover': ('splash_cover', lambda a: a.__setitem__(3, np.nan)),
            'cover above one': ('shading_cover', lambda a: a.__setitem__(3, 1.5)),
            'negative cover': ('splash_cover', lambda a: a.__setitem__(3, -0.1)),
            'inf r1': ('matrix_root_index', lambda a: a.__setitem__(3, np.inf)),
            'negative r1': ('matrix_root_index', lambda a: a.__setitem__(3, -1e-9)),
            'offsets start': ('cell_offsets', lambda a: a.__setitem__(0, 1)),
            'offsets end': ('cell_offsets', lambda a: a.__setitem__(-1, a[-1] + 1)),
            'offsets decrease': ('cell_offsets', lambda a: a.__setitem__(50, a[50] - 5)),
            'entry cell out of range': ('entry_cell', lambda a: a.__setitem__(0, 100)),
            'entry cell negative': ('entry_cell', lambda a: a.__setitem__(0, -1)),
            'entry cell vs offsets': ('entry_cell', lambda a: a.__setitem__(1, a[1] + 1)),
            'species out of range': ('entry_species', lambda a: a.__setitem__(0, 2)),
            'species negative': ('entry_species', lambda a: a.__setitem__(0, -1)),
            'live above all': ('entry_area_live_m2', lambda a: a.__setitem__(0, a[0] + 1.0)),
            'nan area': ('entry_area_all_m2', lambda a: a.__setitem__(0, np.nan)),
            'negative area': ('entry_area_live_m2', lambda a: a.__setitem__(0, -1.0)),
            'negative root': ('root', lambda a: a.__setitem__((0, 0), -0.1)),
            'nan root': ('root', lambda a: a.__setitem__((0, 1), np.nan)),
            'lower mismatch': ('lower', lambda a: a.__setitem__(0, a[0] + 0.125)),
            'negative weight': ('weights', lambda a: a.__setitem__((0, 0), -1.0)),
            'nan weight': ('weights', lambda a: a.__setitem__((0, 1), np.nan)),
            'has_w two': ('has_w', lambda a: a.__setitem__(0, 2)),
            'duplicate omitted ids': ('omitted_ids', lambda a: a.__setitem__(slice(None), 7)),
        }
        for what, (k, edit) in cases.items():
            m = mutable(ex)
            if m['arrays'][k].size < 2 and what.startswith('duplicate'):
                continue
            edit(m['arrays'][k])
            self.refuses(m, what)

    def test_order_inside_a_cell_must_be_strictly_increasing(self):
        p, _ = fx.build('zero_weights', wv)                                                           # two species share a cell
        ex = p.export_compact()
        a = ex['arrays']
        off = np.asarray(a['cell_offsets'])
        c = int(np.flatnonzero(np.diff(off) == 2)[0])
        m = mutable(ex)
        i = int(off[c])
        m['arrays']['entry_species'][i], m['arrays']['entry_species'][i + 1] = m['arrays']['entry_species'][i + 1], m['arrays']['entry_species'][i]
        self.refuses(m, 'species order inside a cell')
        m = mutable(ex)
        m['arrays']['entry_species'][i + 1] = m['arrays']['entry_species'][i]
        self.refuses(m, 'duplicate (cell, species)')

    def test_a_participation_flag_less_species_must_have_zero_weights(self):
        m = mutable(self.ex)
        m['arrays']['has_w'][0] = 0
        self.refuses(m, 'weights without participation')

    def test_tampered_content_with_a_matching_recorded_hash_of_the_ORIGINAL_is_refused(self):
        m = mutable(self.ex)                                                                          # the export carries its own content hash
        i = int(np.argmax(m['arrays']['entry_area_live_m2']))
        self.assertGreater(float(m['arrays']['entry_area_live_m2'][i]), 0.0)
        m['arrays']['entry_area_live_m2'][i] *= 0.5
        self.refuses(m, 'tampered area')
        m = mutable(self.ex)
        m['arrays']['weights'][0, 2] += 1.0                                                           # a valid-looking raw weight change is still a different hash
        self.refuses(m, 'tampered weight')

    def test_rederive_and_streams_refuse_malformed_input_too(self):
        m = mutable(self.ex, version=2)
        for fn in (lambda r: wv.rederive_fields_from_compact(r, max_bytes=BIG), lambda r: wv.species_cover_from_compact(r, 'grass', max_bytes=BIG),
                   lambda r: wv.species_contribution_from_compact(r, 'grass', max_bytes=BIG), lambda r: wv.preferential_inputs_from_compact(r, max_bytes=BIG)):
            self.refuses(m, 'version', fn)
        m = mutable(self.ex)
        m['arrays']['entry_cell'][0] = -1
        self.refuses(m, 'negative index in rederive', lambda r: wv.rederive_fields_from_compact(r, max_bytes=BIG))


class Streams(unittest.TestCase):
    def test_every_species_stream_and_cover_is_byte_identical_to_the_projection(self):
        for name in fx.NAMES:
            with self.subTest(name):
                p, _ = fx.build(name, wv)
                ex = p.export_compact()
                ids0, w0, it0 = p.preferential_inputs(max_bytes=BIG)
                ids1, w1, it1 = wv.preferential_inputs_from_compact(ex, max_bytes=BIG)
                self.assertEqual(ids0, ids1)
                np.testing.assert_array_equal(w0, w1)
                self.assertEqual(w0.tobytes(), w1.tobytes())
                a0, a1 = list(it0), list(it1)
                self.assertEqual(len(a0), len(a1))
                for x, y in zip(a0, a1):
                    self.assertEqual(x.tobytes(), y.tobytes())
                    self.assertTrue(x.flags.writeable and not np.shares_memory(y, ex['arrays']['entry_area_live_m2']))
                for sid in p.species_ids:
                    for elig in ('live', 'all'):
                        self.assertEqual(p.species_cell_cover(sid, elig, max_bytes=BIG).tobytes(), wv.species_cover_from_compact(ex, sid, elig, max_bytes=BIG).tobytes())
                    i = p.species_ids.index(sid)
                    want = p.species_cell_cover(sid, 'live', max_bytes=BIG) * float(ex['arrays']['lower'][i])
                    got = wv.species_contribution_from_compact(ex, sid, max_bytes=BIG)
                    np.testing.assert_allclose(got, want, rtol=1e-15, atol=0.0)

    def test_the_stream_content_cases_zero_lower_fraction_masked_periodic_and_no_active(self):
        p, _ = fx.build('zero_weights', wv)                                                           # grass: all-zero weights (disabled); shrub enabled
        ids, w, it = wv.preferential_inputs_from_compact(p.export_compact(), max_bytes=BIG)
        self.assertEqual(ids, ('shrub',))
        np.testing.assert_array_equal(w, [[0.0, 2.0, 1.0]])
        out = list(it)
        self.assertEqual(len(out), 1)
        p, _ = fx.build('empty', wv)
        ids, w, it = wv.preferential_inputs_from_compact(p.export_compact(), max_bytes=BIG)
        self.assertEqual(len(list(it)), len(ids))
        self.assertEqual(float(np.abs(wv.species_cover_from_compact(p.export_compact(), 'grass', max_bytes=BIG)).max()), 0.0)
        p, _ = fx.build('dead_only', wv)                                                              # dead standing: no live contribution, positive 'all' cover
        ex = p.export_compact()
        self.assertEqual(float(wv.species_cover_from_compact(ex, 'grass', 'live', max_bytes=BIG).max()), 0.0)
        self.assertGreater(float(wv.species_cover_from_compact(ex, 'grass', 'all', max_bytes=BIG).max()), 0.0)

    def test_quota_exact_and_one_below_and_reader_and_previous_held_term(self):
        p, _ = fx.build('periodic_origin', wv)
        ex = p.export_compact()
        N, M = p.n_cells, p.n_entries
        e0 = wv.estimate_stream_bytes(N, M, 2, 4)
        e1 = wv.estimate_stream_bytes(N, M, 2, 4, previous_held=True)
        _, _, it = wv.preferential_inputs_from_compact(ex, max_bytes=e1['per_next_bytes'])
        a, b = next(it), next(it)                                                                     # previous + new fit exactly
        del a, b
        _, _, it = wv.preferential_inputs_from_compact(ex, max_bytes=e0['per_next_bytes'])
        a = next(it)
        with self.assertRaises(wv.VegetationBudgetError):
            next(it)                                                                                  # `a` is still held: the previous array is counted again
        del a
        with self.assertRaises(wv.VegetationBudgetError):
            wv.preferential_inputs_from_compact(ex, max_bytes=e0['weights_bytes'] - 1)
        _, _, it = wv.preferential_inputs_from_compact(ex, max_bytes=e0['per_next_bytes'] - 1)
        with self.assertRaises(wv.VegetationBudgetError):
            next(it)
        _, _, it = wv.preferential_inputs_from_compact(ex, max_bytes=e0['per_next_bytes'] + 100, reserve_bytes=101)          # reserve counted once: 101 + need > cap
        with self.assertRaises(wv.VegetationBudgetError):
            next(it)
        _, _, it = wv.preferential_inputs_from_compact(ex, max_bytes=e0['per_next_bytes'] + 100, reserve_bytes=100)
        next(it)
        with self.assertRaises(wv.VegetationBudgetError):
            wv.preferential_inputs_from_compact(ex, max_bytes=BIG, available_bytes_fn=lambda: 10)       # the weights stage refuses at the call
        wv.preferential_inputs_from_compact(ex, max_bytes=BIG, available_bytes_fn=lambda: 10 ** 9)
        with self.assertRaises(wv.VegetationBudgetError):
            wv.preferential_inputs_from_compact(ex, max_bytes=BIG, available_bytes_fn=lambda: 10)
        need = wv.estimate_stream_bytes(N, M, 0, 4)['species_cover_bytes']
        wv.species_cover_from_compact(ex, 'grass', max_bytes=need)
        with self.assertRaises(wv.VegetationBudgetError):
            wv.species_cover_from_compact(ex, 'grass', max_bytes=need - 1)

    def test_unknown_species_and_eligibility_are_refused(self):
        ex = fx.build('masked3', wv)[0].export_compact()
        with self.assertRaises(wv.VegetationProjectionError):
            wv.species_cover_from_compact(ex, 'tree', max_bytes=BIG)
        with self.assertRaises(wv.VegetationProjectionError):
            wv.species_cover_from_compact(ex, 'grass', 'dead', max_bytes=BIG)
        with self.assertRaises(wv.VegetationProjectionError):
            wv.species_contribution_from_compact(ex, 'tree', max_bytes=BIG)


class Rederive(unittest.TestCase):
    BACKENDS = ('numpy', 'numba')

    def setUp(self):
        if not has_numba():
            self.skipTest('numba is not importable')

    def test_all_fields_are_bit_identical_to_the_producer_on_both_cpu_backends(self):
        for name in fx.NAMES:
            p, _ = fx.build(name, wv)
            ex = p.export_compact()
            for b in self.BACKENDS:
                with self.subTest(name, backend=b):
                    r = wv.rederive_fields_from_compact(ex, backend=b, max_bytes=BIG)
                    for k in ('splash_cover', 'shading_cover', 'matrix_root_index'):
                        self.assertEqual(getattr(r, k).tobytes(), np.asarray(getattr(p, k)).tobytes(), k)
                        self.assertTrue(getattr(r, k).flags.writeable and not np.shares_memory(getattr(r, k), ex['arrays'][k]))
                    self.assertFalse(r.active_mask_checked)

    def test_numba_equals_numpy_and_the_result_is_a_derived_receipt_not_a_projection(self):
        p, _ = fx.build('periodic_origin', wv)
        ex = p.export_compact()
        a, b = (wv.rederive_fields_from_compact(ex, backend=x, max_bytes=BIG) for x in self.BACKENDS)
        for k in ('splash_cover', 'shading_cover', 'matrix_root_index'):
            self.assertEqual(getattr(a, k).tobytes(), getattr(b, k).tobytes())
        self.assertIs(type(a), wv.RederivedFields)
        self.assertFalse(a.genuine_projection)
        self.assertNotIsInstance(a, wv.VegetationProjection)
        with self.assertRaises(AttributeError):
            a.backend = 'x'
        with self.assertRaises(AttributeError):
            a.extra = 1

    def test_cuda_and_unknown_backends_are_refused_before_anything(self):
        ex = fx.build('masked3', wv)[0].export_compact()
        for b in ('cuda', 'gpu', None):
            with self.assertRaises(wv.VegetationBackendError):
                wv.rederive_fields_from_compact(ex, backend=b, max_bytes=BIG)
            with self.assertRaises(wv.VegetationBackendError):
                wv.estimate_rederive_bytes(100, 2, backend=b)

    def test_changed_cover_fields_in_the_record_are_NOT_used_by_the_rederivation(self):
        p, _ = fx.build('masked3', wv)
        m = mutable(p.export_compact())
        want = wv.rederive_fields_from_compact(m, max_bytes=BIG)
        m['arrays']['splash_cover'][:] = 0.5                                                          # stored covers are caller-compared values, never inputs
        got = wv.rederive_fields_from_compact(m, max_bytes=BIG)
        self.assertEqual(want.splash_cover.tobytes(), got.splash_cover.tobytes())
        self.assertNotEqual(m['arrays']['splash_cover'].tobytes(), got.splash_cover.tobytes())         # so the caller's comparison detects the tamper

    def test_active_mask_needs_its_known_sha_and_entries_must_be_active(self):
        p, mask = fx.build('masked3', wv)
        ex = p.export_compact()
        sha = p.identity['active_mask_sha256']
        r = wv.rederive_fields_from_compact(ex, active_mask=mask, expected_active_mask_sha256=sha, max_bytes=BIG)
        self.assertTrue(r.active_mask_checked)
        with self.assertRaises(wv.VegetationProjectionError):
            wv.rederive_fields_from_compact(ex, active_mask=mask, max_bytes=BIG)                       # a mask without its sha
        with self.assertRaises(wv.VegetationProjectionError):
            wv.rederive_fields_from_compact(ex, expected_active_mask_sha256=sha, max_bytes=BIG)        # a sha without the mask
        bad = mask.copy()
        bad[5, 5] = False
        with self.assertRaises(wv.StaleProjectionError):
            wv.rederive_fields_from_compact(ex, active_mask=bad, expected_active_mask_sha256=sha, max_bytes=BIG)
        with self.assertRaises(wv.VegetationProjectionError):
            wv.rederive_fields_from_compact(ex, active_mask=mask.astype(np.uint8), expected_active_mask_sha256=sha, max_bytes=BIG)
        with self.assertRaises(wv.VegetationProjectionError):
            wv.rederive_fields_from_compact(ex, active_mask=np.ones((5, 20), dtype=bool), expected_active_mask_sha256=sha, max_bytes=BIG)
        none = np.zeros(mask.shape, dtype=bool)                                                       # every entry lies in an inactive cell of THIS (hash-consistent) mask
        sha_none = hashlib.sha256(none.reshape(-1).view(np.uint8).tobytes()).hexdigest()
        with self.assertRaises(wv.CompactContentError):
            wv.rederive_fields_from_compact(ex, active_mask=none, expected_active_mask_sha256=sha_none, max_bytes=BIG)

    def test_the_mask_is_never_guessed_the_held_surface_may_predate_the_current_bed(self):
        p, _ = fx.build('masked3', wv)
        r = wv.rederive_fields_from_compact(p.export_compact(), max_bytes=BIG)                        # no mask: nothing about masks or surfaces is checked or inferred
        self.assertFalse(r.active_mask_checked)
        import inspect
        self.assertNotIn('surface', inspect.signature(wv.rederive_fields_from_compact).parameters)


class Budgets(unittest.TestCase):
    def measure(self, fn):
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        out = fn()
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        return out, peak

    def dense(self):
        from tests.test_water_vegetation import catalog, plants_state
        cat = catalog(layers=3, grass=[0, 1, 1], shrub=[0, 1, 0], fractions={'grass': [0.5, 0.3, 0.2], 'shrub': [0.2, 0.3, 0.5]})
        rng = np.random.default_rng(3)
        pl = [(i, 'grass' if i % 2 else 'shrub', float(rng.uniform(0, 100)), float(rng.uniform(0, 100)), float(rng.uniform(0.6, 1.4)), 'live') for i in range(1, 12000)]
        st = plants_state(cat, pl, nx=100, ny=100)
        return wv.project_from_initial_state(st, cat, np.zeros(st.domain.shape), max_edges=1 << 20, max_bytes=1 << 30)

    def test_declared_validation_and_rederive_peaks_bound_the_measured_helper_allocation_on_a_dense_large_case(self):
        p = self.dense()
        ex = p.export_compact()
        N, M = p.n_cells, p.n_entries
        _, peak = self.measure(lambda: wv.compact_content_sha256(ex, max_bytes=BIG))
        self.assertLessEqual(peak, wv.estimate_compact_validation_bytes(N, M, 2, 3, len(p.omitted_ids)))
        from tests.test_water_vegetation import catalog, plants_state
        cat = catalog(layers=3, grass=[0, 1, 1], shrub=[0, 1, 0])
        big = wv.project_from_initial_state(plants_state(cat, [(1, 'grass', 500.5, 500.5, 2.0, 'live')], nx=1000, ny=1000), cat, np.zeros((1000, 1000)), max_edges=1 << 16, max_bytes=1 << 30)
        bex = big.export_compact()
        _, bpeak = self.measure(lambda: wv.compact_content_sha256(bex, max_bytes=BIG))
        self.assertLessEqual(bpeak, wv.estimate_compact_validation_bytes(10 ** 6, big.n_entries, 2, 3, 0))
        self.assertLess(bpeak, 8 * 10 ** 6 // 2)                                                      # N = 1e6: the temporaries are chunk-bounded, far below one (N,) float array
        wv.rederive_fields_from_compact(ex, max_bytes=BIG, backend='numpy')
        _, peak_np = self.measure(lambda: wv.rederive_fields_from_compact(ex, max_bytes=BIG, backend='numpy'))
        est = wv.estimate_rederive_bytes(N, 2, backend='numpy')
        self.assertLessEqual(peak_np, est['peak_bytes'])
        self.assertGreaterEqual(peak_np, est['outputs_bytes'])                                        # the outputs are really allocated: not a tautology
        if has_numba():
            wv.rederive_fields_from_compact(ex, max_bytes=BIG, backend='numba')                       # warm (a compile is not steady-state work)
            _, peak_nb = self.measure(lambda: wv.rederive_fields_from_compact(ex, max_bytes=BIG, backend='numba'))
            est = wv.estimate_rederive_bytes(N, 2, backend='numba')
            self.assertLessEqual(peak_nb, max(est['peak_bytes'], wv.estimate_compact_validation_bytes(N, M, 2, 3, len(p.omitted_ids))))      # the call's validation stage is the larger one here
            self.assertGreaterEqual(peak_nb, est['outputs_bytes'])

    def test_no_population_or_geometry_is_materialised_the_helpers_hold_only_o_n_and_o_m_results(self):
        p = self.dense()
        ex = p.export_compact()
        _, peak = self.measure(lambda: wv.preferential_inputs_from_compact(ex, max_bytes=BIG)[1])
        self.assertLess(peak, 16384)                                                                  # only the raw (n, L) weights (+ Python objects) before the first next(): no O(N) array
        _, _, it = wv.preferential_inputs_from_compact(ex, max_bytes=BIG)
        _, peak = self.measure(lambda: next(it))
        est = wv.estimate_stream_bytes(p.n_cells, p.n_entries, 2, 3)
        self.assertLessEqual(peak, est['result_bytes'] + est['scratch_bytes'])

    def test_rederive_quota_exact_need_minus_one_reader_and_reserve_refuse_before_any_allocation(self):
        p = self.dense()
        ex = p.export_compact()
        N, S = p.n_cells, 2
        for backend in ('numpy', 'numba') if has_numba() else ('numpy',):
            need = max(wv.estimate_rederive_bytes(N, S, backend=backend)['peak_bytes'], wv.estimate_compact_validation_bytes(N, p.n_entries, S, 3, len(p.omitted_ids)))
            wv.rederive_fields_from_compact(ex, backend=backend, max_bytes=need)
            for kw in ({'max_bytes': need - 1}, {'max_bytes': need, 'reserve_bytes': 1}, {'max_bytes': BIG, 'available_bytes_fn': (lambda v=need - 1: v)}):
                with self.subTest(backend=backend, kw=sorted(kw)):
                    gc.collect()
                    tracemalloc.start()
                    base = tracemalloc.get_traced_memory()[0]
                    tracemalloc.reset_peak()
                    with self.assertRaises(wv.VegetationBudgetError):
                        wv.rederive_fields_from_compact(ex, backend=backend, **kw)
                    refused = tracemalloc.get_traced_memory()[1] - base
                    tracemalloc.stop()
                    self.assertLess(refused, 16 * 1024)                                                # no O(N) array was allocated before the refusal
        need_v = wv.estimate_compact_validation_bytes(N, p.n_entries, S, 3, len(p.omitted_ids), max_id_chars=max(map(len, ex['species_ids'])))      # the H1 header term
        wv.compact_content_sha256(ex, max_bytes=need_v)
        with self.assertRaises(wv.VegetationBudgetError):
            wv.compact_content_sha256(ex, max_bytes=need_v - 1)
        with self.assertRaises(wv.VegetationBudgetError):
            wv.compact_content_sha256(ex, max_bytes=need_v, available_bytes_fn=lambda: need_v - 1)
        with self.assertRaises(wv.VegetationProjectionError):
            wv.compact_content_sha256(ex, max_bytes=True)
        with self.assertRaises(TypeError):
            wv.compact_content_sha256(ex)                                                              # the budget is mandatory

    def test_models_scale_with_their_own_dimensions_only(self):
        a = wv.estimate_compact_validation_bytes(10 ** 6, 10 ** 6, 2, 3)
        b = wv.estimate_compact_validation_bytes(10 ** 9, 10 ** 9, 2, 3)
        self.assertEqual(a, b)                                                                         # chunk-bounded, independent of N and M
        self.assertEqual(wv.estimate_compact_validation_bytes(10, 10, 2, 3, 100) - wv.estimate_compact_validation_bytes(10, 10, 2, 3, 0), 2400)
        self.assertEqual(wv.estimate_compact_validation_bytes(10, 10, 12, 3) - wv.estimate_compact_validation_bytes(10, 10, 2, 3), 12 * (12 - 2) + (4 * 12 * 3 + 48 * 12) - (4 * 2 * 3 + 48 * 2))      # S enters as the 12 S sort + the row chunk
        self.assertEqual(wv.estimate_rederive_bytes(200, 3, backend='numpy')['outputs_bytes'], 4800)


class Contracts(unittest.TestCase):
    def test_the_projection_stream_and_cover_methods_share_the_module_arithmetic(self):
        import inspect
        for fn in (wv.VegetationProjection._cover, wv.VegetationProjection.preferential_inputs):
            src = inspect.getsource(fn)
            self.assertTrue('_species_cover' in src or '_preferential_stream' in src)
        self.assertIn('_content_digest', inspect.getsource(wv._build))
        self.assertIn('_content_digest', inspect.getsource(wv.compact_content_sha256))
        self.assertIn('_reduce_numpy', inspect.getsource(wv.rederive_fields_from_compact))

    def test_every_helper_is_exported_and_documents_cold_versus_per_call_checks(self):
        for n in ('compact_content_sha256', 'species_cover_from_compact', 'species_contribution_from_compact', 'preferential_inputs_from_compact', 'rederive_fields_from_compact',
                  'estimate_compact_validation_bytes', 'estimate_rederive_bytes', 'CompactContentError', 'RederivedFields'):
            self.assertIn(n, wv.__all__)
        src = pathlib.Path(wv.__file__).read_text()
        self.assertIn('COLD (once per loaded record)', src)
        self.assertIn('PER CALL', src)
        self.assertIn('NOT a content-generation hash', src)

    def test_no_projection_class_or_injection_factory_exists_and_the_projection_stays_sealed(self):
        with self.assertRaises(wv.VegetationProjectionError):
            wv.VegetationProjection()
        with self.assertRaises(TypeError):
            type('X', (wv.VegetationProjection,), {})
        names = [n for n in dir(wv) if 'from_compact' in n or 'rehydrat' in n.lower() or 'restore' in n.lower()]
        self.assertEqual(sorted(names), ['preferential_inputs_from_compact', 'rederive_fields_from_compact', 'species_contribution_from_compact', 'species_cover_from_compact'])

    def test_the_inputs_are_never_mutated_by_any_helper(self):
        p, mask = fx.build('periodic_origin', wv)
        ex = p.export_compact()
        before = fingerprint(ex)
        wv.compact_content_sha256(ex, max_bytes=BIG)
        wv.rederive_fields_from_compact(ex, max_bytes=BIG, active_mask=mask, expected_active_mask_sha256=p.identity['active_mask_sha256'])
        list(wv.preferential_inputs_from_compact(ex, max_bytes=BIG)[2])
        wv.species_cover_from_compact(ex, 'grass', max_bytes=BIG)
        self.assertEqual(fingerprint(ex), before)
        self.assertEqual(p.identity['content_sha256'], GOLDEN['periodic_origin'])

    def test_the_27_preexisting_leaf_modules_are_byte_identical(self):
        for name, want in BASELINE_27.items():
            self.assertEqual(hashlib.sha256((ROOT / 'leaf' / name).read_bytes()).hexdigest(), want, name)

    def test_the_external_bench_helper_record_is_unchanged_and_the_package_digest_counts_28_files(self):
        rec = wv.package_source_record()
        self.assertEqual(rec['file_count'], len(BASELINE_27) + 1)
        ext = wv.external_source_record()
        self.assertIn('benchmarks.spatial_mapping', json.dumps(ext, default=str))


class VersionStrictness(unittest.TestCase):
    """Correction01 defect 1: the version is a closed integer; float / string / bool / malformed values never reach the content hash or any output."""

    def setUp(self):
        self.p, _ = fx.build('masked3', wv)
        self.ex = self.p.export_compact()

    def calls(self):
        return (lambda r: wv.compact_content_sha256(r, max_bytes=BIG), lambda r: wv.rederive_fields_from_compact(r, max_bytes=BIG),
                lambda r: wv.species_cover_from_compact(r, 'grass', max_bytes=BIG), lambda r: wv.species_contribution_from_compact(r, 'grass', max_bytes=BIG),
                lambda r: wv.preferential_inputs_from_compact(r, max_bytes=BIG))

    def test_non_integer_versions_are_refused_by_all_public_helpers(self):
        for bad in (3.0, np.float64(3.0), '3', b'3', True, np.bool_(True), 3.5, None, [3], (3,), 2, 4, -3, 0):
            rec = mutable(self.ex, version=bad)
            for fn in self.calls():
                with self.subTest(version=repr(bad)), self.assertRaises(wv.CompactContentError):
                    fn(rec)

    def test_a_python_or_numpy_integer_three_is_the_declared_metadata_contract(self):
        for ok in (3, np.int64(3), np.int32(3)):
            self.assertEqual(wv.compact_content_sha256(mutable(self.ex, version=ok), max_bytes=BIG), GOLDEN['masked3'])
        self.assertEqual(wv.rederive_fields_from_compact(mutable(self.ex, version=np.int64(3)), max_bytes=BIG).n_cells, 100)


class MetadataBudget(unittest.TestCase):
    """Correction01 defect 2: nothing O(S) is allocated before the caller's admission; all inputs are created OUTSIDE the tracer."""

    S = 100000

    @classmethod
    def setUpClass(cls):
        p, _ = fx.build('empty', wv)
        d = dict(p.export_compact())
        d.pop('content_sha256')
        d.pop('identity')
        d['species_ids'] = [f's{i:06d}' for i in range(cls.S)]
        a = dict(d['arrays'])
        a.update(root=np.tile(np.array([1.0, 0.0]), (cls.S, 1)), lower=np.zeros(cls.S), weights=np.zeros((cls.S, 2)), has_w=np.zeros(cls.S, dtype=np.uint8))
        d['arrays'] = a
        d['root_layers'] = 2
        cls.d = d
        w = np.zeros((cls.S, 2))
        w[:, 1] = 1.0
        cls.enabled = dict(d, arrays=dict(a, weights=w, has_w=np.ones(cls.S, dtype=np.uint8)))

    def refused_peak(self, fn):
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        try:
            with self.assertRaises(wv.VegetationBudgetError):
                fn()
            return tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()

    def test_the_roots_probe_case_s_100000_refuses_without_an_owned_s_sized_allocation(self):
        for name, fn in (('h1 cap', lambda: wv.compact_content_sha256(self.d, max_bytes=1)),
                         ('h1 reader', lambda: wv.compact_content_sha256(self.d, max_bytes=BIG, available_bytes_fn=lambda: 1)),
                         ('h3 cap', lambda: wv.rederive_fields_from_compact(self.d, max_bytes=1)),
                         ('h3 reader', lambda: wv.rederive_fields_from_compact(self.d, max_bytes=BIG, available_bytes_fn=lambda: 1)),
                         ('h2 stream', lambda: wv.preferential_inputs_from_compact(self.enabled, max_bytes=1)),
                         ('h2 stream reader', lambda: wv.preferential_inputs_from_compact(self.enabled, max_bytes=BIG, available_bytes_fn=lambda: 1)),
                         ('h2 cover', lambda: wv.species_cover_from_compact(self.d, 's000007', max_bytes=1)),
                         ('h2 contribution', lambda: wv.species_contribution_from_compact(self.d, 's000007', max_bytes=1))):
            with self.subTest(name):
                self.assertLess(self.refused_peak(fn), 16 * 1024, name)                                  # the old code owned 6.29 MB before refusing (root probe)

    def test_the_exact_cap_passes_and_one_below_refuses_with_the_literal_s_terms(self):
        need = wv.estimate_compact_validation_bytes(100, 0, self.S, 2, 0, max_id_chars=7)
        self.assertGreater(need, 12 * self.S)
        self.assertEqual(wv.compact_content_sha256(self.d, max_bytes=need), wv.compact_content_sha256(self.d, max_bytes=need + 12345))
        self.assertLess(self.refused_peak(lambda: wv.compact_content_sha256(self.d, max_bytes=need - 1)), 16 * 1024)

    def test_declared_validation_peak_bounds_the_measured_allocation_at_large_s(self):
        need = wv.estimate_compact_validation_bytes(100, 0, self.S, 2, 0, max_id_chars=7)
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        wv.compact_content_sha256(self.d, max_bytes=need)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertLessEqual(peak, need)
        self.assertGreaterEqual(peak, 8 * self.S)                                                      # the uniqueness sort is really there (not a tautology)
        self.assertLess(peak, 32 * self.S)                                                             # and no longer the old 6.3 MB-class S x L expansion

    def test_the_enabled_species_index_is_admitted_before_it_is_built(self):
        n = self.S
        est = wv.estimate_stream_bytes(100, 0, n, 2)
        self.assertLess(self.refused_peak(lambda: wv.preferential_inputs_from_compact(self.enabled, max_bytes=est['weights_stage_bytes'] - 1)), 16 * 1024)
        self.assertLess(self.refused_peak(lambda: wv.preferential_inputs_from_compact(self.enabled, max_bytes=BIG, available_bytes_fn=lambda: est['weights_stage_bytes'] - 1)), 16 * 1024)
        self.assertLess(self.refused_peak(lambda: wv.preferential_inputs_from_compact(self.enabled, max_bytes=est['weights_stage_bytes'], reserve_bytes=1)), 16 * 1024)
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        ids, w, it = wv.preferential_inputs_from_compact(self.enabled, max_bytes=est['weights_stage_bytes'])
        peak = tracemalloc.get_traced_memory()[1] - base
        resident = tracemalloc.get_traced_memory()[0] - base
        tracemalloc.stop()
        self.assertEqual((len(ids), w.shape), (n, (n, 2)))
        self.assertLessEqual(peak, est['weights_stage_bytes'])                                       # the literal weights-stage peak bounds the owned data AND metadata
        self.assertGreaterEqual(peak, est['weights_bytes'] + est['index_bytes'] + est['ids_bytes'])     # not a tautology: index + weights + tuple really exist
        self.assertLessEqual(resident, est['weights_bytes'] + est['retained_metadata_bytes'])         # what stays live is the payload + the retained index / ids
        del it

    def test_the_shared_private_stream_called_like_the_live_projection_at_s_100000_refuses_the_old_weights_only_budget(self):
        """Root's h2 probe: the shared stream called exactly as `VegetationProjection` does, S=1e5, L=2, N=100, M=0; the old declared weights-only budget (1.6 MB) is no longer accepted."""
        n, N = self.S, 100
        ids = self.enabled['species_ids']
        weights, lower, has = self.enabled['arrays']['weights'], self.enabled['arrays']['lower'], self.enabled['arrays']['has_w']
        cover = lambda i, elig, scale: np.zeros(N)
        est = wv.estimate_stream_bytes(N, 0, n, 2)
        self.assertEqual(est['weights_bytes'], 1600000)
        self.assertLess(self.refused_peak(lambda: wv._preferential_stream(wv._quota(1600000, 0, None), N, 0, ids, weights, lower, has, cover)), 16 * 1024)
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        out = wv._preferential_stream(wv._quota(est['weights_stage_bytes'], 0, None), N, 0, ids, weights, lower, has, cover)
        peak = tracemalloc.get_traced_memory()[1] - base
        cur = tracemalloc.get_traced_memory()[0] - base
        tracemalloc.stop()
        self.assertLessEqual(peak, est['weights_stage_bytes'])
        self.assertLessEqual(cur, est['weights_bytes'] + est['retained_metadata_bytes'])
        self.assertGreater(peak, 7 * 10 ** 5 * 4)                                                      # ~ 8 n L + 8 n + 8 n (+ transient list) really allocated: 3.2 MB+
        it = out[2]
        peaks = []
        gc.collect()
        tracemalloc.start()
        b2 = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        first = next(it)
        peaks.append(tracemalloc.get_traced_memory()[1] - b2)
        tracemalloc.stop()
        self.assertLessEqual(peaks[0], est['result_bytes'] + est['scratch_bytes'] + est['weights_bytes'])
        del first, out

    def test_live_projection_and_compact_use_the_same_stream_model_and_admission(self):
        p, _ = fx.build('periodic_origin', wv)
        ex = p.export_compact()
        self.assertEqual(p.stream_estimate(), wv.estimate_stream_bytes(p.n_cells, p.n_entries, 2, 4))
        for held in (False, True):
            self.assertEqual(p.stream_estimate(previous_held=held)['per_next_bytes'], wv.estimate_stream_bytes(p.n_cells, p.n_entries, 2, 4, previous_held=held)['per_next_bytes'])
        est = p.stream_estimate()
        for src in (p, None):
            call = (lambda cap, s=src: s.preferential_inputs(max_bytes=cap)) if src is not None else (lambda cap: wv.preferential_inputs_from_compact(ex, max_bytes=cap))
            with self.assertRaises(wv.VegetationBudgetError):
                call(est['weights_stage_bytes'] - 1)
            _, _, it = call(est['per_next_bytes'])
            next(it)
            _, _, it = call(est['per_next_bytes'] - 1)
            with self.assertRaises(wv.VegetationBudgetError):
                next(it)

    def test_live_projection_stream_peak_is_bounded_by_the_same_model_at_large_s_equivalent_counts(self):
        p, _ = fx.build('periodic_origin', wv)
        est = p.stream_estimate()
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        ids, w, it = p.preferential_inputs(max_bytes=BIG)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertLessEqual(peak, est['weights_stage_bytes'])
        del ids, w, it

    def test_the_projection_estimate_and_species_cover_allocate_no_s_sized_index(self):
        p, _ = fx.build('masked3', wv)
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        p.stream_estimate()
        p.species_cell_cover('grass', max_bytes=BIG)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertLessEqual(peak, 8 * p.n_cells + p.n_entries * 33 + wv.STREAM_FIXED_BYTES)                           # the cover result + its scratch only; the count loop owns nothing
        import inspect
        self.assertNotIn('_enabled_index', inspect.getsource(wv.VegetationProjection.stream_estimate))
        self.assertNotIn('_enabled_index', inspect.getsource(wv.VegetationProjection.species_cell_cover))

    def test_the_stream_values_are_bit_identical_to_the_frozen_c01_module_live_and_compact(self):
        for name in fx.NAMES:
            p, _ = fx.build(name, wv)
            for label, (ids, w, it) in (('live', p.preferential_inputs(max_bytes=BIG)), ('compact', wv.preferential_inputs_from_compact(p.export_compact(), max_bytes=BIG))):
                with self.subTest(fixture=name, source=label):
                    want = STREAM_GOLDEN[name]
                    self.assertEqual(list(ids), want['ids'])
                    self.assertEqual(hashlib.sha256(w.tobytes()).hexdigest(), want['weights'])
                    self.assertEqual([hashlib.sha256(a.tobytes()).hexdigest() for a in it], want['arrays'])

    def test_hostile_key_counts_are_refused_before_any_set_is_built(self):
        p, _ = fx.build('masked3', wv)
        rec = mutable(p.export_compact())
        rec.update({f'junk{i}': 0 for i in range(200000)})
        self.assertLess(self.refused_peak_any(lambda: wv.compact_content_sha256(rec, max_bytes=BIG)), 16 * 1024)
        rec = mutable(p.export_compact())
        rec['arrays'].update({f'junk{i}': 0 for i in range(200000)})
        self.assertLess(self.refused_peak_any(lambda: wv.compact_content_sha256(rec, max_bytes=BIG)), 16 * 1024)

    def refused_peak_any(self, fn):
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        try:
            with self.assertRaises(wv.CompactContentError):
                fn()
            return tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()

    def test_duplicate_ids_and_bad_ids_are_refused_cold_and_without_repair(self):
        rec = dict(self.d, species_ids=['a'] * 5 + [f'x{i}' for i in range(self.S - 5)])
        with self.assertRaises(wv.CompactContentError):
            wv.compact_content_sha256(rec, max_bytes=BIG)
        with self.assertRaises(wv.CompactContentError):
            wv.rederive_fields_from_compact(rec, max_bytes=BIG)
        for bad in ([''] + ['y'] * 2, [1, 2, 3], 'abc', ()):
            p, _ = fx.build('masked3', wv)
            with self.assertRaises(wv.CompactContentError):
                wv.compact_content_sha256(mutable(p.export_compact(), species_ids=bad), max_bytes=BIG)


class HeaderStreaming(unittest.TestCase):
    """Correction01 audit: the streamed header is byte-identical to the canonical JSON and its transient is the modelled few copies of ONE id."""

    IDS = (['grass', 'shrub'], ['a"b', 'c\\d', 'tab\there', 'uni\u00e9', 'astral\U0001F600x', '\u0000nul', 'x' * 300], ['only'], ['\u4e2d\u6587', '\ud7ff'])

    def reference(self, ids, shape=(7, 9), area=2.25, n=63, m=11, plants=5, layers=3):
        return json.dumps({'schema': wv.SCHEMA, 'version': wv.VERSION, 'species_ids': list(ids), 'shape': list(shape), 'cell_area_m2': float(area).hex(), 'n_cells': n,
                           'n_entries': m, 'n_plants': plants, 'root_layers': layers}, sort_keys=True).encode()

    def test_the_pieces_concatenate_to_the_exact_canonical_json_bytes(self):
        for ids in self.IDS:
            with self.subTest(ids=[i[:8] for i in ids]):
                got = b''.join(wv._content_header(ids, (7, 9), 2.25, 63, 11, 5, 3))
                self.assertEqual(got, self.reference(ids))

    def test_a_long_identifier_hash_equals_the_reference_and_the_transient_is_one_id_not_all(self):
        p, _ = fx.build('masked3', wv)
        ex = p.export_compact()
        ids = [('long' + str(i)).ljust(3000, 'z') for i in range(2)]
        rec = mutable(ex, species_ids=ids)
        rec.pop('content_sha256')
        want = hashlib.sha256()
        want.update(self.reference(ids, shape=ex['shape'], area=ex['cell_area_m2'], n=ex['n_cells'], m=ex['n_entries'], plants=ex['n_plants'], layers=ex['root_layers']))
        for k in ex['field_order']:
            a = np.ascontiguousarray(rec['arrays'][k])
            want.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + hashlib.sha256(a.tobytes()).digest())
        self.assertEqual(wv.compact_content_sha256(rec, max_bytes=BIG), 'sha256:' + want.hexdigest())
        # many long ids: the streamed header never holds them all (only the 12 S sort + a few copies of the longest)
        S = 3000
        ids = [f'{i:05d}'.ljust(2000, 'q') for i in range(S)]
        a = dict(ex['arrays'])
        a.update(root=np.tile(np.array([1.0, 0.0, 0.0]), (S, 1)), lower=np.zeros(S), weights=np.zeros((S, 3)), has_w=np.zeros(S, dtype=np.uint8))
        big = dict(ex, species_ids=ids, arrays=a, n_entries=0, n_plants=0)
        big.pop('content_sha256')
        big.pop('identity')
        a.update(cell_offsets=np.zeros(101, dtype=np.int64), entry_cell=np.zeros(0, dtype=np.int64), entry_species=np.zeros(0, dtype=np.int32), entry_area_all_m2=np.zeros(0),
                 entry_area_live_m2=np.zeros(0))
        need = wv.estimate_compact_validation_bytes(100, 0, S, 3, len(a['omitted_ids']), max_id_chars=2000)
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        wv.compact_content_sha256(big, max_bytes=need)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        self.assertLessEqual(peak, need)
        self.assertLess(peak, S * 2000 // 10)                                                          # nowhere near one copy of all the identifier bytes (6 MB)

    def test_the_header_term_is_in_the_model(self):
        a = wv.estimate_compact_validation_bytes(10, 0, 2, 3, 0, max_id_chars=0)
        b = wv.estimate_compact_validation_bytes(10, 0, 2, 3, 0, max_id_chars=1000)
        self.assertEqual(b - a, 3 * 12 * 1000)


if __name__ == '__main__':
    unittest.main()
