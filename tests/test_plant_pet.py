'''Tests for leaf.plant_pet (reused qualified C1 canopy PET) and the `growth.demand_per_plant` sibling next to the unchanged scalar
`growth.demand`. NOT RUN by the author. Run from the candidate root: python -m unittest tests.test_plant_pet -v'''
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leaf import growth, plant_pet as pp, state  # noqa: E402
import native_leaf_fixture as nf  # noqa: E402

try:
    import numba  # noqa: F401
    HAVE_NUMBA = True
except Exception:   # pragma: no cover
    HAVE_NUMBA = False

BIG = 1 << 26


def one_plant_map(builder='numpy', mask=None, r=0.5, x=2.0, y=2.0):
    dom = state.Domain(4, 4, 1.0, origin_x_m=0.0, origin_y_m=0.0)
    ids = np.array([7], dtype=np.int64)
    return dom, pp.build_canopy_map(dom, ids, np.array([x]), np.array([y]), np.array([r]), max_edges=1000, max_bytes=BIG,
                                    builder=builder, valid_cell_mask=mask)


class GeometryTests(unittest.TestCase):
    def test_exact_circle_rectangle_area(self):
        self.assertAlmostEqual(pp.circle_rectangle_area(1.0, -2, 2, -2, 2), math.pi, places=12)
        self.assertAlmostEqual(pp.circle_rectangle_area(1.0, 0, 2, -2, 2), math.pi / 2, places=12)
        self.assertEqual(pp.circle_rectangle_area(1.0, 2, 3, 2, 3), 0.0)

    def test_corner_plant_touches_four_cells_with_equal_area_weights(self):
        dom, cm = one_plant_map()
        a = cm.host_arrays()
        self.assertEqual(sorted(a['cells'].tolist()), [5, 6, 9, 10])
        np.testing.assert_allclose(a['areas_m2'], math.pi * 0.25 / 4.0, rtol=1e-12)
        np.testing.assert_allclose(a['area_total_m2'], [math.pi * 0.25], rtol=1e-12)


class PetTests(unittest.TestCase):
    def test_area_weighted_mean_not_a_cell_count_or_root_weight(self):
        dom, cm = one_plant_map(x=2.3, y=2.0, r=0.6)
        field = np.arange(16, dtype=np.float64)
        out = pp.canopy_pet(cm, field, max_bytes=BIG)['pet_mm_per_day']
        a = cm.host_arrays()
        expect = float((a['areas_m2'] * field[a['cells']]).sum() / a['area_total_m2'][0])
        np.testing.assert_allclose(out, [expect], rtol=1e-12)
        self.assertNotAlmostEqual(float(out[0]), float(field[a['cells']].mean()), places=6)   # unequal areas: not the plain mean

    def test_uniform_field_is_reproduced_exactly(self):
        dom, cm = one_plant_map(x=2.3, y=1.7, r=0.9)
        out = pp.canopy_pet(cm, np.full(16, 4.321), max_bytes=BIG)['pet_mm_per_day']
        self.assertEqual(float(out[0]), 4.321)

    def test_zero_valid_support_refuses_and_there_is_no_site_mean_fallback(self):
        dom = state.Domain(4, 4, 1.0, origin_x_m=0.0, origin_y_m=0.0)
        mask = np.ones(16, dtype=np.bool_)
        mask[[5, 6, 9, 10]] = False                                                  # every cell under the canopy is masked out
        with self.assertRaises(pp.PlantPetError) as cm:
            pp.build_canopy_map(dom, np.array([7], np.int64), np.array([2.0]), np.array([2.0]), np.array([0.5]), max_edges=1000,
                                max_bytes=BIG, valid_cell_mask=mask)
        self.assertIn('ZERO', str(cm.exception))

    def test_invalid_pet_values_and_wrong_namespace_are_refused(self):
        dom, cm = one_plant_map()
        bad = np.full(16, 1.0)
        bad[5] = -1.0
        with self.assertRaises(pp.PlantPetError):
            pp.canopy_pet(cm, bad, max_bytes=BIG)
        with self.assertRaises(pp.PlantPetError):
            pp.canopy_pet(cm, np.full(16, np.nan), max_bytes=BIG)
        with self.assertRaises(pp.PlantPetError):
            pp.canopy_pet(cm, np.full(16, 1.0, dtype=np.float32), max_bytes=BIG)
        with self.assertRaises(pp.PlantPetBudgetError):
            pp.canopy_pet(cm, np.full(16, 1.0), max_bytes=8)

    @unittest.skipUnless(HAVE_NUMBA, 'numba not installed')
    def test_compiled_builder_and_backend_match_numpy(self):
        dom, cn = one_plant_map(builder='numpy', x=2.3, y=1.7, r=0.9)
        _, cc = one_plant_map(builder='compiled', x=2.3, y=1.7, r=0.9)
        field = np.linspace(1.0, 9.0, 16)
        a = pp.canopy_pet(cn, field, max_bytes=BIG)['pet_mm_per_day']
        b = pp.canopy_pet(cc, field, max_bytes=BIG)['pet_mm_per_day']
        np.testing.assert_allclose(a, b, rtol=1e-12)


class PerPlantDemandTests(unittest.TestCase):
    def setUp(self):
        self.cat = nf.make_catalog()
        self.table = growth.compile_species(self.cat, 'numpy', nf.BIG)
        self.B = np.array([0.1, 0.2, 0.3])
        self.area = np.array([0.5, 1.0, 2.0])
        self.sidx = np.zeros(3, dtype=np.int32)
        self.live = np.ones(3, dtype=np.bool_)

    def call(self, pet):
        return growth.demand_per_plant(self.table, B_kg=self.B, canopy_area_m2=self.area, species_index=self.sidx, live=self.live,
                                       month=6, temperature_c=20.0, pet_mm_per_day_per_plant=pet, dt_days=1.0, max_bytes=BIG)

    def test_uniform_array_equals_scalar_demand_bit_for_bit_and_scalar_signature_is_unchanged(self):
        s = growth.demand(self.table, B_kg=self.B, canopy_area_m2=self.area, species_index=self.sidx, live=self.live, month=6,
                          temperature_c=20.0, pet_mm_per_day=5.0, dt_days=1.0, max_bytes=BIG)
        p = self.call(np.full(3, 5.0))
        np.testing.assert_array_equal(s['requested_m3'], p['requested_m3'])
        self.assertAlmostEqual(float(s['requested_m3'][0]), 0.5 * 5.0 * 0.001 * 0.5 * 0.5, places=15)   # hand calculation, plant 0

    def test_each_plant_uses_its_own_pet_and_own_canopy_area(self):
        d = self.call(np.array([2.0, 4.0, 8.0]))['requested_m3']
        base = self.call(np.full(3, 1.0))['requested_m3']
        np.testing.assert_allclose(d, base * np.array([2.0, 4.0, 8.0]), rtol=1e-14)
        np.testing.assert_allclose(base / self.area, np.full(3, base[0] / self.area[0]), rtol=1e-14)

    def test_invalid_live_pet_refused_dead_plant_not_read(self):
        with self.assertRaises(growth.GrowthDataError):
            self.call(np.array([1.0, -1.0, 1.0]))
        live = self.live.copy()
        live[1] = False
        r = growth.demand_per_plant(self.table, B_kg=self.B, canopy_area_m2=self.area, species_index=self.sidx, live=live,
                                    month=6, temperature_c=20.0, pet_mm_per_day_per_plant=np.array([1.0, -1.0, 1.0]),
                                    dt_days=1.0, max_bytes=BIG)
        self.assertEqual(float(r['requested_m3'][1]), 0.0)

    def test_memory_operation_and_admission(self):
        m = growth.memory_estimate('demand_per_plant', 1000, self.table.n_species, 'numpy')
        self.assertGreater(m['total_bytes'], growth.memory_estimate('demand', 1000, self.table.n_species, 'numpy')['total_bytes'])
        with self.assertRaises(growth.GrowthBudgetError):
            growth.demand_per_plant(self.table, B_kg=self.B, canopy_area_m2=self.area, species_index=self.sidx, live=self.live,
                                    month=6, temperature_c=20.0, pet_mm_per_day_per_plant=np.full(3, 1.0), dt_days=1.0,
                                    max_bytes=16)

    @unittest.skipUnless(HAVE_NUMBA, 'numba not installed')
    def test_numba_matches_numpy(self):
        t2 = growth.compile_species(self.cat, 'numba', nf.BIG)
        pet = np.array([2.0, 4.0, 8.0])
        a = self.call(pet)['requested_m3']
        b = growth.demand_per_plant(t2, B_kg=self.B, canopy_area_m2=self.area, species_index=self.sidx, live=self.live, month=6,
                                    temperature_c=20.0, pet_mm_per_day_per_plant=pet, dt_days=1.0, max_bytes=BIG)['requested_m3']
        np.testing.assert_allclose(a, b, rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
