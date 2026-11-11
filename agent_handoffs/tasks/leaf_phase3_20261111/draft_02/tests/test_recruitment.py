'''LEAF Phase 3 recruitment tests (unittest; NumPy only, no native SYRUP needed): law parsing, keyed randomness, seed
trials per ecological patch, clonal daughters (ledger, donation rules, placement), by-ID row-order invariance, signed IDs,
explicit limits and budgets. ALL numbers are UNCALIBRATED fixtures (examples/phase3/species.json).
Written WITHOUT the ability to execute them (file tools only): Codex records the real runs.

Run from the candidate root:  python -m unittest tests.test_recruitment -v
'''
from __future__ import annotations

import json
import math
import pathlib
import unittest

import numpy as np

from leaf import demography, recruitment as rec, state, traits

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / 'examples' / 'phase3' / 'species.json'
BIG = 1 << 30
LIM = dict(max_new_plants=100000, max_total_records=1000000, max_attempts=1000000)


def load_catalog(modify=None):
    with open(CATALOG_PATH, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    if modify is not None:
        modify(doc)
    return traits.SpeciesCatalog(doc)


def live_arrays(catalog, rows):
    '''rows: (id, species, x, y, B, E, age).'''
    idx = {s: i for i, s in enumerate(catalog.species_ids)}
    return dict(live_ids=np.array([r[0] for r in rows], dtype=np.int64),
                live_species=np.array([idx[r[1]] for r in rows], dtype=np.int32),
                live_x=np.array([r[2] for r in rows], dtype=np.float64),
                live_y=np.array([r[3] for r in rows], dtype=np.float64),
                live_B=np.array([r[4] for r in rows], dtype=np.float64),
                live_E=np.array([r[5] for r in rows], dtype=np.float64),
                live_age_days=np.array([r[6] for r in rows], dtype=np.float64))


def run(catalog, domain, rows, *, patch=4.0, seed=7, counter=0, max_id='auto', n_total=None, **over):
    table = rec.compile_species(catalog)
    la = live_arrays(catalog, rows)
    if max_id == 'auto':
        max_id = int(la['live_ids'].max()) if rows else None
    kw = dict(LIM)
    kw.update(over)
    return rec.evaluate(table, rec.PatchConfig(patch), domain, np.full(domain.shape, 100.0), max_existing_id=max_id,
                        n_total_records=len(rows) if n_total is None else n_total, seed=seed, counter=counter,
                        max_bytes=BIG, **la, **kw)


class Law(unittest.TestCase):
    def test_parse_inactive_and_active(self):
        cat = load_catalog()
        t = rec.compile_species(cat)
        self.assertTrue(t.any_active and t.any_seed and t.any_clonal)
        self.assertEqual(sorted(t.active_species()), ['forb', 'grass', 'shrub'])
        self.assertEqual(t.coefficients('grass')['seed_on'], 0)       # clonal only
        self.assertEqual(t.coefficients('shrub')['cl_on'], 0)         # seed only
        self.assertEqual(t.coefficients('forb')['seed_on'], 1)
        self.assertEqual(t.coefficients('forb')['cl_on'], 1)
        with self.assertRaises(rec.ReproductionError):
            t.coefficients('moss_inactive')

    def test_all_inactive_holds_no_array(self):
        def off(doc):
            for sp in doc['species'].values():
                sp['reproduction'] = {'law': 'unimplemented'}
        t = rec.compile_species(load_catalog(off))
        self.assertFalse(t.any_active)
        self.assertEqual(t.resident_bytes(), 0)
        self.assertIsNone(t._f)

    def test_null_section_is_inactive(self):
        def off(doc):
            for sp in doc['species'].values():
                sp['reproduction'] = None
        self.assertFalse(rec.compile_species(load_catalog(off)).any_active)

    def test_unknown_and_malformed_refuse(self):
        def bad_law(doc):
            doc['species']['grass']['reproduction']['law'] = 'something_else_v9'

        def bad_prob(doc):
            doc['species']['shrub']['reproduction']['traits']['seed']['establishment_probability'] = 1.5

        def bad_frac(doc):
            doc['species']['grass']['reproduction']['traits']['clonal']['max_donation_fraction'] = 1.0

        def bad_key(doc):
            doc['species']['grass']['reproduction']['traits']['clonal']['extra'] = 1

        def bad_int(doc):
            doc['species']['shrub']['reproduction']['traits']['seed']['max_recruits_per_patch'] = 2.5

        def bad_bool(doc):
            doc['species']['shrub']['reproduction']['traits']['seed']['max_recruits_per_patch'] = True

        def bad_agb(doc):
            doc['species']['forb']['reproduction']['traits']['seed']['initial_agb_dry_kg'] = 0.0

        for mod in (bad_law, bad_prob, bad_frac, bad_key, bad_int, bad_bool, bad_agb):
            with self.assertRaises(rec.ReproductionError, msg=mod.__name__):
                rec.compile_species(load_catalog(mod))

    def test_budget_refused_before_table(self):
        with self.assertRaises(rec.ReproductionBudgetError):
            rec.compile_species(load_catalog(), max_bytes=1)


class Keyed(unittest.TestCase):
    def test_vector_matches_python_reference(self):
        a = np.array([-(2 ** 63), -1, 0, 1, 2 ** 63 - 1], dtype=np.int64)
        b = np.array([0, 1, 2, 2 ** 32 + 5, 7], dtype=np.uint64)
        key = rec.stream_key(99, 3, rec.STREAM_SEED_X)
        got = rec._u(key, a.astype(np.uint64), b)
        for k in range(a.shape[0]):
            self.assertEqual(float(got[k]), rec.keyed_uniform_py(99, 3, rec.STREAM_SEED_X, int(a[k]), int(b[k])))
        self.assertTrue(np.all((got >= 0.0) & (got < 1.0)))

    def test_streams_are_domain_separated_from_demography(self):
        keys = {rec.stream_key(5, 0, s) for s in rec.STREAMS}
        keys |= {demography.stream_key(5, 0, demography.STREAM_AGE), demography.stream_key(5, 0, demography.STREAM_STRESS)}
        self.assertEqual(len(keys), len(rec.STREAMS) + 2)
        # the phase 2 golden hash is unchanged by this module
        self.assertEqual(demography.keyed_uniform_py(5, 0, demography.STREAM_AGE, 17),
                         float(demography.keyed_uniform(5, 0, demography.STREAM_AGE, np.array([17], dtype=np.int64))[0]))

    def test_strict_seed_counter(self):
        with self.assertRaises(rec.ReproductionError):
            rec.stream_key(True, 0, rec.STREAM_SEED_X)
        with self.assertRaises(rec.ReproductionError):
            rec.stream_key(1, -1, rec.STREAM_SEED_X)
        with self.assertRaises(rec.ReproductionError):
            rec.stream_key(1, 0, 1)             # a demography stream is not a reproduction stream


class Seeds(unittest.TestCase):
    def setUp(self):
        self.cat = load_catalog()
        self.dom = state.Domain(8, 8, 1.0)

    def test_zero_live_zero_records_seed_establishment(self):
        res = run(self.cat, self.dom, [], max_id=None, n_total=0)
        d = res['diagnostics']
        b = res['births']
        self.assertGreater(d['n_seed_recruits'], 0)          # 4 patches x 2 seed species, p = 0.6 / 0.3: fixed seed gives some
        self.assertEqual(d['n_births'], d['n_seed_recruits'])
        np.testing.assert_array_equal(b['ids'], np.arange(1, d['n_births'] + 1))
        self.assertTrue(np.all(self.dom.contains(b['x_m'], b['y_m'])))
        self.assertTrue(np.all(b['establishment_ground_m'] == 100.0))
        self.assertTrue(np.all(b['seasonal_agb_dry_kg'] <= b['agb_dry_kg']))
        led = d['ledger']
        self.assertEqual(led['old_total_dry_agb_kg'], 0.0)
        self.assertAlmostEqual(led['external_seed_agb_kg'], float(b['agb_dry_kg'].sum()), places=12)
        self.assertTrue(led['closed'])
        self.assertFalse(b['is_clonal'].any())
        # clonal-only species never appear as seeds
        grass = self.cat.species_ids.index('grass')
        self.assertFalse(np.any(b['species_index'] == grass))

    def test_deterministic_and_counter_dependent(self):
        a = run(self.cat, self.dom, [], max_id=None, n_total=0)
        b = run(self.cat, self.dom, [], max_id=None, n_total=0)
        np.testing.assert_array_equal(a['births']['x_m'], b['births']['x_m'])
        diffs = [run(self.cat, self.dom, [], max_id=None, n_total=0, counter=c)['diagnostics']['n_births'] for c in range(1, 6)]
        pos = [run(self.cat, self.dom, [], max_id=None, n_total=0, counter=c)['births']['x_m'].tolist() for c in range(0, 4)]
        self.assertGreater(len({json.dumps(p) for p in pos}), 1)
        self.assertTrue(all(isinstance(n, int) for n in diffs))

    def test_positions_use_clipped_boundary_patch_area(self):
        dom = state.Domain(5, 5, 1.0)              # 5 m domain, 4 m patches: boundary patches are 1 m wide
        res = run(self.cat, dom, [], max_id=None, n_total=0, patch=4.0, seed=11)
        self.assertTrue(np.all(dom.contains(res['births']['x_m'], res['births']['y_m'])))
        self.assertEqual(rec.PatchConfig(4.0).n_patches(dom), 4)
        x0, x1, y0, y1 = rec.patch_geometry(rec.PatchConfig(4.0), dom, np.arange(4, dtype=np.int64))
        np.testing.assert_allclose((x1 - x0) * (y1 - y0), [16.0, 4.0, 4.0, 1.0])

    def test_patch_invariance_across_hydraulic_resolution(self):
        coarse, fine = state.Domain(8, 8, 1.0), state.Domain(16, 16, 0.5)
        a = run(self.cat, coarse, [], max_id=None, n_total=0, seed=21)
        b = run(self.cat, fine, [], max_id=None, n_total=0, seed=21)
        for k in ('ids', 'species_index', 'x_m', 'y_m', 'agb_dry_kg'):
            np.testing.assert_array_equal(a['births'][k], b['births'][k], err_msg=k)

    def test_patch_required_for_seed_law(self):
        table = rec.compile_species(self.cat)
        with self.assertRaises(rec.ReproductionError):
            rec.preflight_bytes(table, None, self.dom, 0)

    def test_patch_config_strict(self):
        for bad in (0.0, -1.0, float('nan'), float('inf'), True, '4', None):
            with self.assertRaises(rec.ReproductionError, msg=repr(bad)):
                rec.PatchConfig(bad)


class Clonal(unittest.TestCase):
    ROWS = [(10, 'grass', 2.0, 2.0, 0.5, 0.1, 800.0), (11, 'grass', 5.0, 5.0, 0.4, 0.0, 900.0),
            (12, 'forb', 3.5, 6.0, 0.2, 0.05, 400.0), (13, 'shrub', 6.0, 2.0, 5.0, 0.0, 2000.0),
            (14, 'moss_inactive', 1.0, 6.5, 0.04, 0.0, 300.0), (15, 'grass', 6.0, 6.0, 0.1, 0.0, 800.0),
            (16, 'grass', 4.0, 1.0, 0.5, 0.0, 100.0)]

    def setUp(self):
        def always(doc):
            for s in ('grass', 'forb'):
                doc['species'][s]['reproduction']['traits']['clonal']['probability'] = 1.0
            doc['species']['shrub']['reproduction']['traits']['seed']['establishment_probability'] = 0.0
            doc['species']['forb']['reproduction']['traits']['seed']['establishment_probability'] = 0.0
        self.cat = load_catalog(always)
        self.dom = state.Domain(8, 8, 1.0)
        self.cl = self.cat.species_ids

    def test_ledger_donation_and_eligibility(self):
        res = run(self.cat, self.dom, self.ROWS, max_id=16)
        d, b = res['diagnostics'], res['births']
        self.assertGreater(d['n_daughters_placed'], 0)
        self.assertTrue(d['ledger']['closed'])
        self.assertEqual(d['n_seed_recruits'], 0)
        ids = np.array([r[0] for r in self.ROWS])
        la = live_arrays(self.cat, self.ROWS)
        donors = {int(la['live_ids'][r]): k for r, k in zip(res['parent_rows'], range(len(res['parent_rows'])))}
        for pid, k in donors.items():
            n_d = int(np.count_nonzero(b['parent_id'] == pid))
            self.assertGreater(n_d, 0)
            row = int(np.flatnonzero(ids == pid)[0])
            agb = 0.05 if self.ROWS[row][1] == 'grass' else 0.02
            self.assertAlmostEqual(res['parent_B_new'][k], self.ROWS[row][4] - n_d * agb, places=12)
            self.assertGreater(res['parent_B_new'][k], 0.0)
            self.assertLessEqual(res['parent_E_new'][k], res['parent_B_new'][k])
        # ineligible: shrub / inactive species, too young (id 16: age 100 < 365), too small (id 15: B 0.1 < 0.2)
        for pid in (13, 14, 15, 16):
            self.assertNotIn(pid, donors)
            self.assertFalse(np.any(b['parent_id'] == pid))
        # newborns: fresh unique IDs above the existing maximum, parent species, age 0 (appended by the session)
        self.assertTrue(np.all(b['ids'] > 16))
        self.assertEqual(len(set(b['ids'].tolist())), b['ids'].shape[0])
        self.assertTrue(b['is_clonal'].all())
        self.assertTrue(np.all(self.dom.contains(b['x_m'], b['y_m'])))
        sp_of = {r[0]: r[1] for r in self.ROWS}
        for k in range(b['ids'].shape[0]):
            self.assertEqual(self.cl[b['species_index'][k]], sp_of[int(b['parent_id'][k])])

    def test_daughter_inside_dispersal_radius(self):
        res = run(self.cat, self.dom, self.ROWS, max_id=16)
        b = res['births']
        pos = {r[0]: (r[2], r[3], r[1]) for r in self.ROWS}
        for k in range(b['ids'].shape[0]):
            px, py, sp = pos[int(b['parent_id'][k])]
            R = 0.6 if sp == 'grass' else 0.4
            self.assertLessEqual(math.hypot(b['x_m'][k] - px, b['y_m'][k] - py), R * (1 + 1e-12))

    def test_rejected_placements_cause_no_donation(self):
        def tiny_attempts(doc):
            c = doc['species']['grass']['reproduction']['traits']['clonal']
            c['probability'] = 1.0
            c['max_attempts_per_daughter'] = 1
            c['dispersal_radius_m'] = 6.0                  # about half of the single draws leave the 8 m domain
            doc['species']['forb']['reproduction']['traits']['clonal']['probability'] = 0.0
            doc['species']['forb']['reproduction']['traits']['seed']['establishment_probability'] = 0.0
            doc['species']['shrub']['reproduction']['traits']['seed']['establishment_probability'] = 0.0
        cat = load_catalog(tiny_attempts)
        rows = [(i, 'grass', 4.0, 4.0, 0.5, 0.1, 800.0) for i in range(100, 160)]
        res = run(cat, self.dom, rows, max_id=159)
        d = res['diagnostics']
        self.assertGreater(d['n_daughters_rejected'], 0)
        self.assertGreater(d['n_daughters_placed'], 0)
        self.assertEqual(d['n_daughters_requested'], d['n_daughters_placed'] + d['n_daughters_rejected'])
        total_donated = float(np.sum(0.5 - res['parent_B_new']))
        self.assertAlmostEqual(total_donated, 0.05 * d['n_daughters_placed'], places=10)
        self.assertEqual(int(res['parent_rows'].shape[0]), len({int(p) for p in res['births']['parent_id']}))

    def test_row_permutation_gives_same_by_id_results(self):
        res = run(self.cat, self.dom, self.ROWS, max_id=16)
        perm = [self.ROWS[i] for i in (6, 2, 4, 0, 5, 3, 1)]
        res2 = run(self.cat, self.dom, perm, max_id=16)
        for k in ('ids', 'species_index', 'x_m', 'y_m', 'agb_dry_kg', 'seasonal_agb_dry_kg', 'parent_id'):
            np.testing.assert_array_equal(res['births'][k], res2['births'][k], err_msg=k)
        la, la2 = live_arrays(self.cat, self.ROWS), live_arrays(self.cat, perm)
        m1 = dict(zip(la['live_ids'][res['parent_rows']].tolist(), res['parent_B_new'].tolist()))
        m2 = dict(zip(la2['live_ids'][res2['parent_rows']].tolist(), res2['parent_B_new'].tolist()))
        self.assertEqual(m1, m2)

    def test_signed_ids_and_int64_exhaustion(self):
        rows = [(-(2 ** 63), 'grass', 2.0, 2.0, 0.5, 0.1, 800.0), (0, 'grass', 5.0, 5.0, 0.4, 0.0, 900.0),
                (-7, 'forb', 3.5, 6.0, 0.2, 0.05, 400.0)]
        res = run(self.cat, self.dom, rows, max_id=0)
        self.assertEqual(int(res['births']['ids'][0]), 1)             # above max(existing, 0)
        # existing max-int64 refuses ONLY when births need IDs
        rows2 = rows + [(2 ** 63 - 1, 'moss_inactive', 1.0, 1.0, 0.04, 0.0, 10.0)]
        with self.assertRaises(rec.ReproductionLimitError):
            run(self.cat, self.dom, rows2, max_id=2 ** 63 - 1)

        def none(doc):
            for s in ('grass', 'forb'):
                doc['species'][s]['reproduction']['traits']['clonal']['probability'] = 0.0
            for s in ('shrub', 'forb'):
                doc['species'][s]['reproduction']['traits']['seed']['establishment_probability'] = 0.0
        res3 = run(load_catalog(none), self.dom, rows2, max_id=2 ** 63 - 1)
        self.assertEqual(res3['diagnostics']['n_births'], 0)
        self.assertIsNone(res3['births'])

    def test_limits_and_budget_refuse_without_truncation(self):
        with self.assertRaises(rec.ReproductionLimitError):
            run(self.cat, self.dom, self.ROWS, max_id=16, max_new_plants=1)
        with self.assertRaises(rec.ReproductionLimitError):
            run(self.cat, self.dom, self.ROWS, max_id=16, max_total_records=len(self.ROWS))
        with self.assertRaises(rec.ReproductionLimitError):
            run(self.cat, self.dom, self.ROWS, max_id=16, max_attempts=0)
        table = rec.compile_species(self.cat)
        la = live_arrays(self.cat, self.ROWS)
        with self.assertRaises(rec.ReproductionBudgetError):
            rec.evaluate(table, rec.PatchConfig(4.0), self.dom, np.full(self.dom.shape, 100.0), max_existing_id=16,
                         n_total_records=7, seed=7, counter=0, max_bytes=100, **la, **LIM)
        for bad in (True, 1.5, -1, '3', None):
            with self.assertRaises(rec.ReproductionError, msg=repr(bad)):
                rec.check_limits(bad, 1, 1)

    def test_counts_bounded_by_configuration(self):
        for c in range(6):
            res = run(self.cat, self.dom, self.ROWS, max_id=16, counter=c)
            la = live_arrays(self.cat, self.ROWS)
            for pid in set(res['births']['parent_id'].tolist()) - {-1}:
                row = [r for r in self.ROWS if r[0] == pid][0]
                cap = 2 if row[1] == 'grass' else 3
                self.assertLessEqual(int(np.count_nonzero(res['births']['parent_id'] == pid)), cap)
            self.assertEqual(la['live_ids'].shape[0], len(self.ROWS))


if __name__ == '__main__':
    unittest.main()
