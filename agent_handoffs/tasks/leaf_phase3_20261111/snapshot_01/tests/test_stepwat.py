'''Acceptance tests for the opt-in STEP-style aboveground growth adaptation (leaf.stepwat) in the vegetation
Simulation and the coupled native-water driver. Trait values below are TEST FIXTURES, not calibrations.'''
import math
import unittest
from unittest import mock

import numpy as np

from benchmarks import daily_water_growth as dwg
from benchmarks import spatial_mapping as sm
from leaf import stepwat
from leaf.coupling import CoupledSimulation

try:
    from maple_syrup import subsurface as ss
    NATIVE = True
except ImportError:  # pragma: no cover
    NATIVE = False

# ---- fixture helpers (same shapes as tests/test_coupling.py, which is read-only)
THETA_R, THICK = 0.05, 0.1
RES_M = THETA_R * THICK
WET = 0.02  # theta 0.2: 0.015 m available per layer/cell above residual
PI = math.pi

# FIXTURE trait rows (labelled; arbitrary bounded values, no scientific calibration)
GRASS = {'mature_biomass_kg': 10.0, 'intrinsic_rate_per_year': 0.8, 'biomass_support_kg_per_m3': 20.0,
         'live_fraction': 0.5, 'canopy_biomass_density_kg_m2': 1.0, 'potential_transpiration_m_per_day': 0.004,
         'extra_growth_fraction': 0.25, 'monthly_activity': [1.0] * 12}
SHRUB = {'mature_biomass_kg': 40.0, 'intrinsic_rate_per_year': 0.3, 'biomass_support_kg_per_m3': 8.0,
         'live_fraction': 0.4, 'canopy_biomass_density_kg_m2': 2.0, 'potential_transpiration_m_per_day': 0.003,
         'extra_growth_fraction': 0.1, 'monthly_activity': [0.5] * 12}
T1 = {'grass': GRASS}


def make_case(**over):
    case = {'name': 'stepwat', 'nx': 1, 'ny': 1, 'dx': 10.0, 'grass_count': 1, 'shrub_count': 0,
            'grass_diameter': 2.0, 'shrub_diameter': 4.0, 'root_multiplier': 1.0, 'pattern': 'uniform',
            'seed': 1, 'layers': 1, 'max_edges': 100000, 'max_bytes': 50_000_000,
            'root_layer_fractions': {'grass': [1.0]}}
    case.update(over)
    return case


def make_inputs(xs, ys, radii, mult=1.0, pid=None):
    cr = np.asarray(radii, dtype=np.float64)
    inp = {'x': np.asarray(xs, np.float64), 'y': np.asarray(ys, np.float64), 'canopy_radius': cr,
           'root_radius': mult * cr, 'ids': np.arange(len(cr), dtype=np.int64)}
    if pid is not None:
        inp['root_profile_id'] = pid
    return inp


def xp_for(backend):
    return sm.cp if backend == 'cuda' else np


def make_column(case, backend='numpy', active=None, diffusivity=0.0, available=None, reserve_bytes=0):
    L, N = case['layers'], case['nx'] * case['ny']
    layers = ss.soil_layers(np.linspace(0.0, THICK * L, L + 1), theta_residual=THETA_R, theta_field_capacity=0.25,
                            theta_saturation=0.40, diffusivity_m2_s=diffusivity, theta_evaporation_stress=0.10)
    mask = np.ones(N, bool) if active is None else np.asarray(active, bool)
    mask = xp_for(backend).asarray(mask)
    fn = None if available is None else (lambda: available['bytes'])
    return ss.prepare_column(layers, mask, case['dx'] ** 2, backend=backend, reserve_bytes=reserve_bytes,
                             available_bytes_fn=fn)


def build(case, inputs, traits, water=None, backend='numpy', active=None, diffusivity=0.0, available=None):
    col = make_column(case, backend, active, diffusivity, available=available)
    L, N = case['layers'], case['nx'] * case['ny']
    w = np.full((L, N), WET) if water is None else np.asarray(water, dtype=np.float64)
    state = ss.column_state(col, col.xp.asarray(w))
    return CoupledSimulation(case, col, state, inputs=inputs, stepwat_traits=traits), col


def forcing(sim):
    xp = sim.xp
    return xp.zeros((sim.L, sim.N), dtype=xp.float64), xp.zeros(sim.N, dtype=xp.float64)


def host(a):
    return np.asarray(a.get() if hasattr(a, 'get') else a)


def veg_sim(case, inputs, traits, backend='cpu'):
    '''Vegetation-only Simulation in STEP mode driven directly through commit_daily_uptake (no native water).'''
    return dwg.Simulation(case, backend, inputs=inputs, vegetation_only=True, stepwat_traits=traits)


# ---- pure python reference of the numerical contract
def ref_period(B, E, U, tr, tint, act_cur, act_next, dt, mult, year_end):
    persistent = B - E
    m = tr['mature_biomass_kg']
    R = persistent / m
    frac = act_cur * dt / 365.0
    required = B * tr['live_fraction'] * frac
    available = tr['biomass_support_kg_per_m3'] * U
    water = min(1.0, available / required) if required > 0.0 else 0.0
    ng = tr['intrinsic_rate_per_year'] * 0.95 * tint * water * persistent * (1.0 - R)
    ng = min(max(ng, 0.0), max(m - B, 0.0))
    surplus = max(available - required, 0.0)
    mean_tf = min(tint / frac, 1.0) if frac > 0.0 else 0.0
    xg = min(tr['extra_growth_fraction'] * surplus * mean_tf, max(m - B - ng, 0.0))
    loss, e_new = (E + xg, 0.0) if year_end else (0.0, E + xg)
    b_new = persistent + ng + e_new
    c = math.sqrt(b_new / (PI * tr['canopy_biomass_density_kg_m2']))
    Dp = PI * c * c * tr['potential_transpiration_m_per_day'] * tr['live_fraction'] * act_next
    return {'B': b_new, 'E': e_new, 'gain': ng + xg, 'loss': loss, 'normal': ng, 'extra': xg, 'canopy': c,
            'root': mult * c, 'Dp': Dp}


def ref_run(tr, B0, days, u_of_day, tf_of_day, mult=1.0):
    '''Daily-constant uptake/temperature reference over the growth-boundary rule.'''
    act = tr['monthly_activity']
    st = {'B': B0, 'E': 0.0, 'U': 0.0, 'tint': 0.0, 'start': 0, 'gain': 0.0, 'loss': 0.0, 'extra': 0.0,
          'normal': 0.0, 'updates': 0, 'Dp': PI * (B0 / (PI * tr['canopy_biomass_density_kg_m2']))
          * tr['potential_transpiration_m_per_day'] * tr['live_fraction'] * act[0]}
    for d in range(1, days + 1):
        a_cur = act[stepwat.month_of_day(st['start'])]
        st['U'] += u_of_day(d - 1)
        st['tint'] += tf_of_day(d - 1) * a_cur / 365.0
        if stepwat.is_growth_boundary(d):
            r = ref_period(st['B'], st['E'], st['U'], tr, st['tint'], a_cur, act[stepwat.month_of_day(d)],
                           d - st['start'], mult, d % 365 == 0)
            st.update(B=r['B'], E=r['E'], Dp=r['Dp'], U=0.0, tint=0.0, start=d)
            st['gain'] += r['gain']
            st['loss'] += r['loss']
            st['extra'] += r['extra']
            st['normal'] += r['normal']
            st['updates'] += 1
    return st


def dp0(tr, r, month=0):
    return PI * r * r * tr['potential_transpiration_m_per_day'] * tr['live_fraction'] * tr['monthly_activity'][month]


class PureHelperTests(unittest.TestCase):
    def test_calendar_helpers(self):
        for day, m in ((0, 0), (30, 0), (31, 1), (58, 1), (59, 2), (334, 11), (364, 11), (365, 0), (396, 1)):
            self.assertEqual(stepwat.month_of_day(day), m, day)
        self.assertFalse(stepwat.is_growth_boundary(0))
        self.assertTrue(stepwat.is_growth_boundary(7))
        self.assertFalse(stepwat.is_growth_boundary(8))
        self.assertTrue(stepwat.is_growth_boundary(365))  # partial period (365 % 7 == 1)
        self.assertEqual(stepwat.expected_growth_updates(364), 52)
        self.assertEqual(stepwat.expected_growth_updates(365), 53)
        self.assertEqual(stepwat.expected_growth_updates(366), 53)
        self.assertEqual(stepwat.expected_growth_updates(2555), 365 + 6)  # 7th year-end coincides with a week
        self.assertEqual(stepwat.state_bytes_estimate(3, 2), 8 * (2 * 3 + 23 * 2))
        self.assertEqual(stepwat.host_mirror_bytes(2, False), 0)
        self.assertEqual(stepwat.host_mirror_bytes(2, True), 8 * 19 * 2)

    def test_trait_validation_pure(self):
        vt = stepwat.validate_traits({'grass': GRASS, 'shrub': SHRUB}, ['grass', 'shrub'])
        self.assertEqual(vt.S, 2)
        self.assertEqual(vt.host_arrays()['monthly_activity'].shape, (2, 12))
        self.assertEqual(vt.as_report()['shrub']['mature_biomass_kg'], 40.0)
        bad = [None, {}, [GRASS], {'grass': GRASS, 'shrub': SHRUB}, {'oak': GRASS}, {'grass': 'x'},
               {'grass': {k: v for k, v in GRASS.items() if k != 'live_fraction'}},
               {'grass': dict(GRASS, wue=1.0)}, {'grass': dict(GRASS, mature_biomass_kg=0.0)},
               {'grass': dict(GRASS, mature_biomass_kg=float('nan'))}, {'grass': dict(GRASS, mature_biomass_kg=True)},
               {'grass': dict(GRASS, intrinsic_rate_per_year=1.5)}, {'grass': dict(GRASS, live_fraction=-0.1)},
               {'grass': dict(GRASS, biomass_support_kg_per_m3=0.0)}, {'grass': dict(GRASS, canopy_biomass_density_kg_m2=0.0)},
               {'grass': dict(GRASS, potential_transpiration_m_per_day=-1e-3)},
               {'grass': dict(GRASS, extra_growth_fraction=1.01)}, {'grass': dict(GRASS, monthly_activity=[1.0] * 11)},
               {'grass': dict(GRASS, monthly_activity=[1.0] * 11 + [2.0])},
               {'grass': dict(GRASS, monthly_activity=[1.0] * 11 + [float('inf')])},
               {'grass': dict(GRASS, monthly_activity='111111111111')}]
        for t in bad:
            with self.assertRaises(ValueError, msg=repr(t)):
                stepwat.validate_traits(t, ['grass'])
        tfv = stepwat.validate_temperature_factor(0.5, 2)
        np.testing.assert_array_equal(tfv, [0.5, 0.5])
        np.testing.assert_array_equal(stepwat.validate_temperature_factor([0.0, 1.0], 2), [0.0, 1.0])
        np.testing.assert_array_equal(stepwat.validate_temperature_factor(np.array([0.25, 1]), 2), [0.25, 1.0])
        self.assertEqual(stepwat.validate_temperature_factor(np.float32(0.5), 2).dtype, np.float64)

        class FakeDevice:  # device-like value: any transfer attempt must be visible as a failure
            shape, ndim, dtype = (2,), 1, np.dtype(np.float64)
            __cuda_array_interface__ = {'shape': (2,), 'typestr': '<f8', 'version': 3}

            def get(self):
                raise AssertionError('.get() transfer attempted on a device-like temperature factor')

        class FakeDeviceNoCAI:
            shape = (2,)

            def get(self):
                raise AssertionError('.get() transfer attempted')

        big = np.zeros((4096, 4096))  # wrongly shaped: must be refused on shape alone, never converted
        nested = [[0.5] * 1000, [0.5] * 1000]  # malformed nested input of the right length
        bad = (None, True, 1.5, -0.1, float('nan'), 'x', b'xx', [0.5], [0.5, 0.5, 0.5], np.array([[0.5, 0.5]]),
               np.array([True, False]), np.array(['a', 'b']), {'a': 1}, FakeDevice(), FakeDeviceNoCAI(), big,
               nested, [0.5, [0.5]], [0.5, True], [0.5, 'x'], [0.5, float('inf')], (0.5, None), range(3), object())
        with mock.patch.object(stepwat.np, 'asarray', side_effect=AssertionError('bulk conversion')), \
                mock.patch.object(stepwat.np, 'ascontiguousarray', side_effect=AssertionError('bulk conversion')), \
                mock.patch.object(stepwat.np, 'array', side_effect=AssertionError('bulk conversion')), \
                mock.patch.object(stepwat.np, 'full', side_effect=AssertionError('bulk conversion')):
            for t in bad:
                with self.assertRaises(ValueError, msg=repr(type(t))):
                    stepwat.validate_temperature_factor(t, 2)
        if sm.cp is not None and dwg.gpu_available():  # a real device array is refused (host-only interface)
            with self.assertRaises(ValueError):
                stepwat.validate_temperature_factor(sm.cp.asarray([0.5, 0.5]), 2)


class VegetationOnlyStepTests(unittest.TestCase):
    '''Direct commit_daily_uptake driving (fast; no native water): analytical equations and clocks.'''

    def _sim(self, tr=GRASS, r=1.0, **over):
        case = make_case(**over)
        return veg_sim(case, make_inputs([5.0], [5.0], [r], pid=[0]), {'grass': tr})

    def test_initial_state_and_mode_guards(self):
        sim = self._sim()
        self.assertIsNotNone(sim.stepwat)
        np.testing.assert_allclose(sim.B, [PI], rtol=1e-15)
        np.testing.assert_array_equal(sim.stepwat.E, [0.0])
        np.testing.assert_allclose(sim.Dp, [dp0(GRASS, 1.0)], rtol=1e-15)
        self.assertEqual(sim.counters()['pending_days'], 0)
        self.assertTrue(sim.counters()['stepwat'])
        self.assertEqual(sim.counters()['growth_updates'], 0)
        inp = make_inputs([5.0], [5.0], [1.0], pid=[0])
        with self.assertRaises(ValueError):  # synthetic standalone water refused in STEP mode
            dwg.Simulation(make_case(), 'cpu', inputs=inp, stepwat_traits=T1)
        with self.assertRaises(ValueError):  # explicit profile ids required (no species inferred from size)
            dwg.Simulation(make_case(), 'cpu', inputs=make_inputs([5.0], [5.0], [1.0]), vegetation_only=True,
                           stepwat_traits=T1)
        with self.assertRaises(ValueError):  # generated population is size-classified: refused
            dwg.Simulation(make_case(), 'cpu', vegetation_only=True, stepwat_traits=T1)
        with mock.patch.object(dwg.sm, 'build_root_graph_cpu', wraps=dwg.sm.build_root_graph_cpu) as b:
            with self.assertRaises(ValueError):  # B0 = 4 pi > mature 10: refused before any graph/water day
                self._sim(r=2.0)
        b.assert_not_called()
        # legacy vegetation-only simulation refuses a temperature factor and exposes no STEP state
        leg = dwg.Simulation(make_case(), 'cpu', inputs=inp, vegetation_only=True)
        self.assertIsNone(leg.stepwat)
        with self.assertRaises(ValueError):
            leg.commit_daily_uptake(np.zeros(1), temperature_factor=1.0)
        self.assertEqual(leg.day, 0)
        self.assertNotIn('stepwat', leg.counters())
        self.assertNotIn('stepwat_state_bytes', leg.last_ledger)
        np.testing.assert_allclose(leg.B, [PI * dwg.FIXTURE['biomass_density_kg_m2']])

    def test_invalid_traits_refused_before_materialization(self):
        inp = make_inputs([5.0], [5.0], [1.0], pid=[0])
        for t in ({'shrub': GRASS}, {'grass': dict(GRASS, live_fraction=2.0)}, {'grass': GRASS, 'x': GRASS}):
            with mock.patch.object(dwg.sm, 'validate_inputs') as val, \
                    mock.patch.object(dwg, 'build_root_profile_table') as tab:
                with self.assertRaises(ValueError, msg=repr(t)):
                    veg_sim(make_case(), inp, t)
            val.assert_not_called()
            tab.assert_not_called()

    def test_wet_week_matches_equations(self):
        sim = self._sim()
        u = 1.5 * dp0(GRASS, 1.0)
        for _ in range(7):
            sim.commit_daily_uptake(np.full(1, u), temperature_factor=1.0)
        ref = ref_period(PI, 0.0, 7 * u, GRASS, 7.0 / 365.0, 1.0, 1.0, 7, 1.0, False)
        self.assertGreater(ref['extra'], 0.0)
        self.assertGreater(ref['normal'], 0.0)
        for k, a in (('B', sim.B), ('E', sim.stepwat.E), ('gain', sim.gain_cum), ('loss', sim.loss_cum),
                     ('canopy', sim.canopy), ('root', sim.root), ('Dp', sim.Dp)):
            np.testing.assert_allclose(a, [ref[k]], rtol=1e-9, atol=1e-15, err_msg=k)
        self.assertEqual(float(sim.loss_cum[0]), 0.0)
        np.testing.assert_allclose(sim.B, sim.B0 + sim.gain_cum - sim.loss_cum, rtol=1e-12)
        c = sim.counters()
        self.assertEqual((c['weeks_done'], c['growth_updates'], c['maps_built'], c['pending_days']), (1, 1, 2, 0))
        self.assertEqual(float(sim.uptake_week[0]), 0.0)
        np.testing.assert_array_equal(sim.stepwat.temp_integral, [0.0])
        ps = sim.plant_summary()
        self.assertTrue(ps['ok'], ps['checks'])
        rep = ps['stepwat']
        self.assertEqual(rep['model'], stepwat.MODEL_ID)
        tot = rep['domain_totals_kg']
        np.testing.assert_allclose(tot['extra_gain'], ref['extra'], rtol=1e-9)
        np.testing.assert_allclose(tot['normal_gain'], ref['normal'], rtol=1e-9)
        np.testing.assert_allclose(tot['seasonal'], ref['E'], rtol=1e-9)
        np.testing.assert_allclose(tot['persistent'], ref['B'] - ref['E'], rtol=1e-9)
        self.assertIn('domain sum', tot['unit'])
        self.assertIn('per individual', rep['per_individual_kg']['unit'])
        self.assertEqual(rep['per_individual_kg']['population'], 1)
        np.testing.assert_allclose(rep['per_individual_kg']['B_mean'], ref['B'], rtol=1e-9)
        self.assertIn('DOMAIN SUM', rep['biomass_unit'])
        self.assertIn('DOMAIN SUM', ps['biomass']['unit'])
        self.assertEqual(ps['biomass']['scope'], 'domain_sum')
        self.assertEqual(rep['traits']['per_profile']['grass']['mature_biomass_kg'], 10.0)
        self.assertIn('intrinsic_rate_per_year', rep['traits']['definitions'])
        self.assertIn('rgroup_Grow', rep['label'])
        self.assertEqual(ps['biomass']['model'], stepwat.MODEL_ID)
        self.assertNotIn('wue', ps['biomass'])

    def test_limited_zero_and_temperature(self):
        tr = dict(GRASS, biomass_support_kg_per_m3=2.0)  # water-limited: available < required, no surplus
        sim = self._sim(tr)
        u = 0.5 * dp0(tr, 1.0)
        sim.commit_daily_uptake(np.full(1, u), temperature_factor=1.0)
        for _ in range(6):
            sim.commit_daily_uptake(np.zeros(1), temperature_factor=1.0)
        ref = ref_period(PI, 0.0, u, tr, 7.0 / 365.0, 1.0, 1.0, 7, 1.0, False)
        required = PI * 0.5 * 7.0 / 365.0
        self.assertLess(2.0 * u, required)  # genuinely limited
        self.assertEqual(ref['extra'], 0.0)
        np.testing.assert_allclose(sim.B, [ref['B']], rtol=1e-9)
        np.testing.assert_allclose(sim.gain_cum, [ref['normal']], rtol=1e-9)
        np.testing.assert_allclose(ref['normal'], 0.8 * 0.95 * (7.0 / 365.0) * (2.0 * u / required) * PI * (1 - PI / 10.0),
                                   rtol=1e-12)
        # zero uptake: nothing grows, demand unchanged
        sim = self._sim()
        for _ in range(7):
            sim.commit_daily_uptake(np.zeros(1), temperature_factor=1.0)
        np.testing.assert_array_equal(sim.B, [PI])
        np.testing.assert_array_equal(sim.gain_cum, [0.0])
        np.testing.assert_allclose(sim.Dp, [dp0(GRASS, 1.0)], rtol=1e-15)
        self.assertEqual(sim.counters()['growth_updates'], 1)
        # temperature_factor 0 with plenty of water: no normal and no extra gain
        sim = self._sim()
        for _ in range(7):
            sim.commit_daily_uptake(np.full(1, 1.5 * dp0(GRASS, 1.0)), temperature_factor=0.0)
        np.testing.assert_array_equal(sim.B, [PI])
        np.testing.assert_array_equal(sim.stepwat.E, [0.0])
        self.assertGreater(float(sim.uptake_cum[0]), 0.0)  # uptake continues; temperature only modifies growth
        # temperature_factor 0.5 per-profile array form
        sim = self._sim()
        u = 1.5 * dp0(GRASS, 1.0)
        for _ in range(7):
            sim.commit_daily_uptake(np.full(1, u), temperature_factor=np.array([0.5]))
        ref = ref_period(PI, 0.0, 7 * u, GRASS, 0.5 * 7.0 / 365.0, 1.0, 1.0, 7, 1.0, False)
        np.testing.assert_allclose(sim.B, [ref['B']], rtol=1e-9)
        np.testing.assert_allclose(sim.stepwat.E, [ref['E']], rtol=1e-9)

    def test_dormancy_month_switch(self):
        tr = dict(GRASS, monthly_activity=[0.0] + [1.0] * 11)  # January dormant
        sim = self._sim(tr)
        np.testing.assert_array_equal(sim.Dp, [0.0])
        for d in range(28):
            sim.commit_daily_uptake(np.zeros(1), temperature_factor=1.0)
            np.testing.assert_array_equal(sim.stepwat.temp_integral, [0.0])  # activity 0: no accumulation
        np.testing.assert_array_equal(sim.Dp, [0.0])  # day 28 is still January
        self.assertEqual(sim.stepwat.month_cur, 0)
        for _ in range(7):
            sim.commit_daily_uptake(np.zeros(1), temperature_factor=1.0)
        self.assertEqual(sim.stepwat.month_cur, 1)  # period starting day 35 selects February
        np.testing.assert_allclose(sim.Dp, [dp0(tr, 1.0, month=1)], rtol=1e-12)
        np.testing.assert_array_equal(sim.B, [PI])  # no gain without an active period
        sim.commit_daily_uptake(np.zeros(1), temperature_factor=1.0)
        np.testing.assert_allclose(sim.stepwat.temp_integral, [1.0 / 365.0], rtol=1e-15)
        self.assertEqual(sim.counters()['pending_days'], 1)

    def test_mature_bound_caps_gains(self):
        tr = dict(GRASS, biomass_support_kg_per_m3=1e4)
        r = 1.7
        sim = self._sim(tr, r=r)
        B0 = PI * r * r
        self.assertLess(B0, 10.0)
        u = 1.5 * dp0(tr, r)
        for _ in range(7):
            sim.commit_daily_uptake(np.full(1, u), temperature_factor=1.0)
        ref = ref_period(B0, 0.0, 7 * u, tr, 7.0 / 365.0, 1.0, 1.0, 7, 1.0, False)
        np.testing.assert_allclose(sim.B, [10.0], rtol=1e-12)
        np.testing.assert_allclose(sim.B, [ref['B']], rtol=1e-12)
        np.testing.assert_allclose(sim.gain_cum, [10.0 - B0], rtol=1e-12)  # only the ACTUAL (capped) gain is recorded
        E1 = float(sim.stepwat.E[0])
        self.assertGreater(E1, 0.0)
        for _ in range(7):
            sim.commit_daily_uptake(np.full(1, u), temperature_factor=1.0)
        np.testing.assert_allclose(sim.B, [10.0], rtol=1e-12)  # no room: no gains at mature size
        np.testing.assert_allclose(sim.stepwat.E, [E1], rtol=1e-12)
        np.testing.assert_allclose(sim.gain_cum, [10.0 - B0], rtol=1e-12)
        self.assertTrue(sim.plant_summary()['ok'])

    def test_year_end_partial_period_and_closure(self):
        sim = self._sim()
        u = 0.002
        for d in range(366):
            sim.commit_daily_uptake(np.full(1, u), temperature_factor=1.0)
            if d + 1 == 365:
                c = sim.counters()
                self.assertEqual((c['weeks_done'], c['growth_updates'], c['year_end_updates'], c['maps_built'],
                                  c['pending_days']), (52, 53, 1, 54, 0))
                np.testing.assert_array_equal(sim.stepwat.E, [0.0])  # all seasonal mass removed at year-end
                self.assertEqual(float(sim.uptake_week[0]), 0.0)
                tot = sim.plant_summary()['stepwat']['domain_totals_kg']
                np.testing.assert_allclose(tot['seasonal_loss'], tot['extra_gain'], rtol=1e-9)
                np.testing.assert_allclose(sim.loss_cum, [tot['extra_gain']], rtol=1e-9)
                self.assertGreater(float(sim.loss_cum[0]), 0.0)
        ref = ref_run(GRASS, PI, 366, lambda d: u, lambda d: 1.0)
        self.assertEqual(ref['updates'], 53)
        np.testing.assert_allclose(sim.B, [ref['B']], rtol=1e-9)
        np.testing.assert_allclose(sim.stepwat.E, [ref['E']], rtol=1e-9)
        np.testing.assert_allclose(sim.gain_cum, [ref['gain']], rtol=1e-9)
        np.testing.assert_allclose(sim.loss_cum, [ref['loss']], rtol=1e-9)
        np.testing.assert_allclose(sim.Dp, [ref['Dp']], rtol=1e-9)
        np.testing.assert_allclose(sim.B, sim.B0 + sim.gain_cum - sim.loss_cum, rtol=1e-12)
        c = sim.counters()
        self.assertEqual((c['weeks_done'], c['growth_updates'], c['pending_days'], c['period_start_day']), (52, 53, 1, 365))
        np.testing.assert_allclose(sim.uptake_week, [u])
        np.testing.assert_allclose(sim.stepwat.temp_integral, [1.0 / 365.0])
        ps = sim.plant_summary()
        self.assertTrue(ps['ok'], ps['checks'])
        self.assertEqual(ps['stepwat']['period']['pending_days'], 1)

    def test_year_end_refusal_then_retry_once(self):
        a, b = self._sim(), self._sim()
        u = 0.002
        for _ in range(364):
            a.commit_daily_uptake(np.full(1, u), temperature_factor=0.7)
            b.commit_daily_uptake(np.full(1, u), temperature_factor=0.7)
        a.commit_daily_uptake(np.full(1, u), temperature_factor=0.7)
        forced = sm.AdmissionRejected('forced', 'forced refusal', {})
        with mock.patch.object(dwg.sm, 'build_root_graph_cpu', side_effect=forced):
            with self.assertRaises(sm.AdmissionRejected):
                b.commit_daily_uptake(np.full(1, u), temperature_factor=0.7)
        self.assertTrue(b.weekly_pending)
        self.assertEqual((b.day, b.weeks_done, b.stepwat.growth_updates), (365, 52, 52))
        np.testing.assert_allclose(b.uptake_week, [u])  # the accepted day's uptake is retained
        np.testing.assert_allclose(b.stepwat.temp_integral, [0.7 / 365.0])  # temperature state retained
        self.assertGreater(float(b.stepwat.E[0]), 0.0)  # seasonal mass not yet removed
        with self.assertRaises(sm.AdmissionRejected):
            b.commit_daily_uptake(np.full(1, u), temperature_factor=0.7)  # latched
        b.retry_weekly_update()
        self.assertFalse(b.weekly_pending)
        for n in ('B', 'Dp', 'gain_cum', 'loss_cum', 'uptake_week', 'uptake_cum'):
            np.testing.assert_allclose(getattr(b, n), getattr(a, n), rtol=1e-12, err_msg=n)
        np.testing.assert_array_equal(b.stepwat.E, [0.0])
        np.testing.assert_array_equal(b.stepwat.temp_integral, [0.0])
        self.assertEqual(b.counters(), a.counters())
        self.assertEqual(b.stepwat.year_end_updates, 1)
        with self.assertRaises(ValueError):  # nothing pending: the retry applied exactly once
            b.retry_weekly_update()

    def test_memory_ledger_counts_growth_state(self):
        case = make_case(layers=2, grass_count=2, root_layer_fractions={'grass': [0.6, 0.4], 'shrub': [0.3, 0.7]})
        inp = make_inputs([5.0, 5.0], [5.0, 5.0], [1.0, 1.0], pid=[0, 1])
        leg = dwg.Simulation(case, 'cpu', inputs=inp, vegetation_only=True)
        st = veg_sim(case, inp, {'grass': GRASS, 'shrub': SHRUB})
        P, S = 2, 2
        add = stepwat.state_bytes_estimate(P, S)
        self.assertEqual(st.live_bytes() - leg.live_bytes(), add)
        self.assertEqual(st.stepwat.live_bytes(), add)
        self.assertEqual(st.last_ledger['stepwat_state_bytes'], add)
        self.assertEqual(st.last_ledger['stepwat_host_mirror_bytes'], 0)
        self.assertEqual(st.last_ledger['total_bytes'] - leg.last_ledger['total_bytes'], add)
        names = {r['name'] for r in st.array_records()}
        for n in ('stepwat.E', 'stepwat._Ec', 'stepwat.monthly_activity', 'stepwat.temperature_integral'):
            self.assertIn(n, names)
        self.assertEqual(sum(r['nbytes'] for r in st.array_records() if r['name'].startswith('stepwat.')), add)
        self.assertNotIn('stepwat.E', {r['name'] for r in leg.array_records()})
        # init guard: the legacy ledger exactly fits, the STEP ledger does not -> refused before any graph build
        tight = dict(case, max_bytes=leg.last_ledger['total_bytes'])
        dwg.Simulation(tight, 'cpu', inputs=inp, vegetation_only=True)
        with mock.patch.object(dwg.sm, 'build_root_graph_cpu', wraps=dwg.sm.build_root_graph_cpu) as b:
            with self.assertRaises(sm.AdmissionRejected):
                veg_sim(tight, inp, {'grass': GRASS, 'shrub': SHRUB})
        b.assert_not_called()


@unittest.skipUnless(NATIVE, 'maple_syrup native package / transpiration branch not importable')
class CoupledStepTests(unittest.TestCase):
    def test_wet_week_native_water_matches_equations(self):
        sim, _ = build(make_case(), make_inputs([5.0], [5.0], [1.0], pid=[0]), T1)
        v = sim.vegetation
        D = dp0(GRASS, 1.0)
        np.testing.assert_allclose(host(v.Dp), [D], rtol=1e-15)
        added, pet = forcing(sim)
        for _ in range(7):
            res = sim.step_day(added, pet, temperature_factor=1.0)
        np.testing.assert_allclose(res.transpiration_budget.actual_m3, D, rtol=1e-9)
        ref = ref_period(PI, 0.0, 7 * D, GRASS, 7.0 / 365.0, 1.0, 1.0, 7, 1.0, False)
        for k, n in (('B', 'B'), ('canopy', 'canopy'), ('root', 'root'), ('Dp', 'Dp'), ('gain', 'gain_cum')):
            np.testing.assert_allclose(host(getattr(v, n)), [ref[k]], rtol=1e-9, err_msg=k)
        np.testing.assert_allclose(host(v.stepwat.E), [ref['E']], rtol=1e-9)
        s = sim.summary()
        self.assertTrue(s['ok'], s['checks'])
        self.assertEqual(s['stepwat']['model'], stepwat.MODEL_ID)
        self.assertEqual(s['calendar']['vegetation']['growth_updates'], 1)
        self.assertEqual(s['biomass']['model'], stepwat.MODEL_ID)
        np.testing.assert_allclose(s['demand']['plant_actual'], 7 * D, rtol=1e-9)
        self.assertIn('stepwat_bytes', s['memory'])
        self.assertEqual(s['memory']['stepwat_bytes'], v.stepwat.live_bytes())
        # legacy coupled driver: no STEP keys, temperature factor refused before the native step
        leg, _ = build(make_case(), make_inputs([5.0], [5.0], [1.0], pid=[0]), None)
        self.assertIsNone(leg.vegetation.stepwat)
        with mock.patch.object(leg._st, 'step_with_transpiration') as native:
            with self.assertRaises(ValueError):
                leg.step_day(*forcing(leg), temperature_factor=1.0)
        native.assert_not_called()
        ls = leg.summary()
        self.assertNotIn('stepwat', ls)
        self.assertNotIn('stepwat_bytes', ls['memory'])
        self.assertNotIn('stepwat', ls['calendar']['vegetation'])

    def test_grass_shrub_traits_all_layer_profiles(self):
        frac = {'grass': [0.6, 0.4], 'shrub': [0.3, 0.7]}
        case = make_case(layers=2, grass_count=2, root_layer_fractions=frac)
        traits = {'grass': GRASS, 'shrub': SHRUB}
        sim, _ = build(case, make_inputs([5.0, 5.0], [5.0, 5.0], [1.0, 1.0], pid=[0, 1]), traits)
        v = sim.vegetation
        Dg, Ds = dp0(GRASS, 1.0), dp0(SHRUB, 1.0)
        np.testing.assert_allclose(host(v.B), [PI, 2 * PI], rtol=1e-15)
        np.testing.assert_allclose(host(v.Dp), [Dg, Ds], rtol=1e-15)
        tf = np.array([1.0, 0.5])  # per-profile temperature factor
        for _ in range(7):
            sim.step_day(*forcing(sim), temperature_factor=tf)
        rg = ref_period(PI, 0.0, 7 * Dg, GRASS, 7.0 / 365.0, 1.0, 1.0, 7, 1.0, False)
        rs = ref_period(2 * PI, 0.0, 7 * Ds, SHRUB, 0.5 * 0.5 * 7.0 / 365.0, 0.5, 0.5, 7, 1.0, False)
        np.testing.assert_allclose(host(v.uptake_cum), [7 * Dg, 7 * Ds], rtol=1e-9)
        np.testing.assert_allclose(host(v.B), [rg['B'], rs['B']], rtol=1e-9)
        np.testing.assert_allclose(host(v.stepwat.E), [rg['E'], rs['E']], rtol=1e-9)
        np.testing.assert_allclose(host(v.Dp), [rg['Dp'], rs['Dp']], rtol=1e-9)
        self.assertNotAlmostEqual(rg['gain'] / PI, rs['gain'] / (2 * PI))  # different traits, different growth
        tab = host(v.root_layer_fractions)
        np.testing.assert_allclose(tab, [frac['grass'], frac['shrub']])  # vertical profiles unchanged
        s = sim.summary()
        self.assertTrue(s['ok'], s['checks'])
        self.assertEqual(s['stepwat']['traits']['per_profile']['shrub']['live_fraction'], 0.4)

    def test_invalid_forcing_refused_before_any_native_or_plant_change(self):
        sim, _ = build(make_case(), make_inputs([5.0], [5.0], [1.0], pid=[0]), T1)
        added, pet = forcing(sim)

        class FakeDevice:  # device-like forcing: must be refused without any .get() transfer
            shape = (1,)
            __cuda_array_interface__ = {'shape': (1,), 'typestr': '<f8', 'version': 3}

            def get(self):
                raise AssertionError('.get() transfer attempted')

        v = sim.vegetation
        state0 = sim.state
        for bad in (None, 1.5, -0.1, float('nan'), True, [0.5, 0.5], 'x', np.array([[0.5]]), FakeDevice(),
                    [[0.5]], np.zeros((2048, 2048))):
            with mock.patch.object(sim._st, 'step_with_transpiration', wraps=sim._st.step_with_transpiration) as nat, \
                    mock.patch.object(v.stepwat, 'accumulate_temperature', wraps=v.stepwat.accumulate_temperature) as acc, \
                    mock.patch.object(stepwat.np, 'asarray', side_effect=AssertionError('bulk conversion')), \
                    mock.patch.object(stepwat.np, 'ascontiguousarray', side_effect=AssertionError('bulk conversion')):
                with self.assertRaises(ValueError, msg=repr(bad)):
                    sim.step_day(added, pet, temperature_factor=bad)
            nat.assert_not_called()
            acc.assert_not_called()
        self.assertIs(sim.state, state0)
        self.assertEqual((sim.days, v.day, sim.state.t_s), (0, 0, 0.0))
        np.testing.assert_array_equal(host(sim.state.water_m), WET)
        np.testing.assert_array_equal(host(v.uptake_cum), [0.0])
        np.testing.assert_array_equal(host(v.uptake_week), [0.0])
        np.testing.assert_array_equal(host(v.stepwat.temp_integral), [0.0])
        self.assertIsNone(sim.last_step)
        self.assertEqual(sim.cum['transpiration'], 0.0)
        self.assertEqual(sim.step_days([(added, pet, 0.5), (added, pet, np.array([1.0]))]), 2)
        np.testing.assert_allclose(host(sim.vegetation.stepwat.temp_integral), [1.5 / 365.0], rtol=1e-12)
        with self.assertRaises(ValueError):
            sim.step_days([(added, pet)])  # STEP mode: a bare pair carries no temperature factor
        self.assertEqual(sim.days, 2)
        # invalid traits at construction: refused before the vegetation Simulation is even created
        col = make_column(make_case())
        state = ss.column_state(col, np.full((1, 1), WET))
        with mock.patch.object(dwg, 'Simulation') as veg_cls:
            for t in ({'grass': dict(GRASS, intrinsic_rate_per_year=2.0)}, {'shrub': GRASS}, {}):
                with self.assertRaises(ValueError, msg=repr(t)):
                    CoupledSimulation(make_case(), col, state, inputs=make_inputs([5.0], [5.0], [1.0], pid=[0]),
                                      stepwat_traits=t)
        veg_cls.assert_not_called()

    def test_cell_areas_and_inactive_mask_with_evaporation(self):
        for dx in (1.0, 0.5):
            case = make_case(nx=2, ny=2, dx=dx)
            r = 0.6 * dx  # root circle covers all four cells equally; cell 3 inactive
            inp = make_inputs([dx], [dx], [r], pid=[0])
            sim, _ = build(case, inp, T1, active=[True, True, True, False])
            v = sim.vegetation
            D = dp0(GRASS, r)
            np.testing.assert_allclose(host(v.Dp), [D], rtol=1e-15)
            added, pet = forcing(sim)
            pet = pet + 1e-8  # positive soil evaporation on active cells, separate from transpiration
            pet[3] = 0.0
            for _ in range(7):
                sim.step_day(added, pet, temperature_factor=1.0)
            U = 7 * 0.75 * D
            np.testing.assert_allclose(host(v.uptake_cum), [U], rtol=1e-9)
            ref = ref_period(PI * r * r, 0.0, U, GRASS, 7.0 / 365.0, 1.0, 1.0, 7, 1.0, False)
            np.testing.assert_allclose(host(v.B), [ref['B']], rtol=1e-9)
            np.testing.assert_allclose(host(v.stepwat.E), [ref['E']], rtol=1e-9)
            np.testing.assert_allclose(host(v.Dp), [ref['Dp']], rtol=1e-9)
            s = sim.summary()
            self.assertTrue(s['ok'], s['checks'])
            w = s['water']
            self.assertGreater(w['evaporation'], 0.0)
            np.testing.assert_allclose(w['transpiration'], U, rtol=1e-9)
            np.testing.assert_allclose(s['demand']['plant_actual'], U, rtol=1e-9)
            np.testing.assert_allclose(s['demand']['unmapped_inactive'], 7 * 0.25 * D, rtol=1e-9)
            np.testing.assert_allclose(w['initial_storage'] - w['current_storage'] + w['added'],
                                       w['evaporation'] + w['transpiration'] + w['drainage'] + w['surface_return'],
                                       rtol=1e-9, atol=w['bound'])
            self.assertGreater(w['transpiration'], 0.0)  # E and T are separate, both positive and distinct
            self.assertNotAlmostEqual(w['evaporation'], w['transpiration'])
            np.testing.assert_array_equal(host(sim.state.water_m)[:, 3], WET)
            self.assertEqual(s['mask']['cell_area_m2'], dx * dx)

    def test_split_run_equivalence_with_hydrology(self):
        frac = {'grass': [0.6, 0.4], 'shrub': [0.3, 0.7]}
        case = make_case(nx=4, ny=4, dx=0.5, grass_count=2, layers=2, max_edges=10000, root_layer_fractions=frac)
        inp = make_inputs([0.8, 1.3], [1.0, 1.1], [0.5, 0.4], pid=[0, 1])
        traits = {'grass': GRASS, 'shrub': SHRUB}
        water = np.vstack([np.full(16, 0.030), np.full(16, 0.030)])
        a, _ = build(case, inp, traits, water, diffusivity=1e-9, active=[True] * 15 + [False])
        b, _ = build(case, inp, traits, water, diffusivity=1e-9, active=[True] * 15 + [False])
        rng = np.random.default_rng(2)
        days = []
        for _ in range(16):
            add = rng.uniform(0, 0.004, (2, 16)) * np.array([[1.0], [0.0]])
            add[:, 15] = 0.0
            pet = rng.uniform(0, 2e-8, 16)
            pet[15] = 0.0
            days.append((add, pet, float(rng.uniform(0.0, 1.0))))
        a.step_days(days)
        b.step_days(days[:9])
        b.step_days(days[9:])
        np.testing.assert_allclose(host(b.state.water_m), host(a.state.water_m), rtol=1e-12)
        for n in ('B', 'Dp', 'canopy', 'uptake_cum', 'uptake_week', 'gain_cum'):
            np.testing.assert_allclose(host(getattr(b.vegetation, n)), host(getattr(a.vegetation, n)), rtol=1e-12)
        np.testing.assert_allclose(host(b.vegetation.stepwat.E), host(a.vegetation.stepwat.E), rtol=1e-12)
        np.testing.assert_allclose(host(b.vegetation.stepwat.temp_integral), host(a.vegetation.stepwat.temp_integral), rtol=1e-12)
        sa, sb = a.summary(), b.summary()
        self.assertEqual(sa['calendar'], sb['calendar'])
        self.assertEqual(sa['calendar']['vegetation']['growth_updates'], 2)
        self.assertEqual(sa['calendar']['vegetation']['pending_days'], 2)
        self.assertTrue(sa['ok'], sa['checks'])
        self.assertGreater(sa['water']['evaporation'], 0.0)
        self.assertGreater(sa['water']['transpiration'], 0.0)
        self.assertGreater(float(host(a.vegetation.stepwat.E).sum()), 0.0)

    def test_weekly_refusal_latches_then_retry_matches_twin(self):
        inp = make_inputs([5.0], [5.0], [1.0], pid=[0])
        a, _ = build(make_case(), inp, T1)
        b, _ = build(make_case(), inp, T1)
        for _ in range(6):
            a.step_day(*forcing(a), temperature_factor=0.8)
            b.step_day(*forcing(b), temperature_factor=0.8)
        a.step_day(*forcing(a), temperature_factor=0.8)
        v = b.vegetation
        forced = sm.AdmissionRejected('forced', 'forced refusal', {})
        with mock.patch.object(dwg.sm, 'build_root_graph_cpu', side_effect=forced):
            with self.assertRaises(sm.AdmissionRejected):
                b.step_day(*forcing(b), temperature_factor=0.8)
        self.assertTrue(b.weekly_pending)
        self.assertEqual((b.days, v.day, v.weeks_done, v.stepwat.growth_updates), (7, 7, 0, 0))
        np.testing.assert_allclose(host(v.uptake_week), [7 * dp0(GRASS, 1.0)], rtol=1e-9)
        np.testing.assert_allclose(host(v.stepwat.temp_integral), [0.8 * 7 / 365.0], rtol=1e-12)
        np.testing.assert_array_equal(host(v.B), [PI])
        np.testing.assert_array_equal(host(v.stepwat.E), [0.0])
        self.assertEqual(v.counters()['pending_days'], 7)
        with self.assertRaises(sm.AdmissionRejected):
            b.step_day(*forcing(b), temperature_factor=0.8)
        self.assertEqual(b.days, 7)
        b.retry_weekly_update()
        self.assertFalse(b.weekly_pending)
        for n in ('B', 'Dp', 'canopy', 'gain_cum', 'uptake_cum', 'uptake_week'):
            np.testing.assert_allclose(host(getattr(v, n)), host(getattr(a.vegetation, n)), rtol=1e-12, err_msg=n)
        np.testing.assert_allclose(host(v.stepwat.E), host(a.vegetation.stepwat.E), rtol=1e-12)
        np.testing.assert_array_equal(host(v.stepwat.temp_integral), [0.0])
        self.assertEqual(v.counters(), a.vegetation.counters())
        self.assertEqual(v.counters()['pending_days'], 0)
        with self.assertRaises(ValueError):
            b.retry_weekly_update()
        b.step_day(*forcing(b), temperature_factor=0.8)
        self.assertTrue(b.summary()['ok'])

    def test_weekly_old_new_peak_counts_growth_state(self):
        inp = make_inputs([5.0], [5.0], [1.0], pid=[0])
        sw_bytes = stepwat.state_bytes_estimate(1, 1)
        ledgers = []
        real = CoupledSimulation._vegetation_admission

        def rec(self_, ledger):  # class-level wrapper: the Simulation binds THIS callback during construction
            ledgers.append(dict(ledger))
            return real(self_, ledger)

        with mock.patch.object(CoupledSimulation, '_vegetation_admission', rec):
            a, _ = build(make_case(), inp, T1)
            self.assertEqual([l['stage'] for l in ledgers], ['init_scalar_bound', 'init_bound', 'init_generated'])
            for _ in range(7):
                a.step_day(*forcing(a), temperature_factor=1.0)
        stages = {l['stage']: l for l in ledgers}
        self.assertEqual([l['stage'] for l in ledgers][3:], ['rebuild_bound', 'rebuild_actual'])
        init_total = stages['init_generated']['total_bytes']
        self.assertEqual(a.vegetation.last_ledger['stage'], 'rebuild_actual')
        for l in ledgers:
            self.assertEqual(l['stepwat_state_bytes'], sw_bytes)
            self.assertEqual(l['total_bytes'], l['state_bytes'] + l['old_graph_bytes'] + l['new_graph_bytes']
                             + l['scratch_allowance_bytes'] + l['host_mirror_bytes'] + l['external_reserve_bytes']
                             + l['stepwat_state_bytes'] + l['stepwat_host_mirror_bytes'])
        T_bound = stages['rebuild_bound']['total_bytes']
        self.assertGreaterEqual(a.memory_report()['peak_required_bytes'], stages['rebuild_actual']['total_bytes'])
        avail = {'bytes': 1 << 30}
        b, _ = build(make_case(), inp, T1, available=avail)
        for _ in range(6):
            b.step_day(*forcing(b), temperature_factor=1.0)
        avail['bytes'] = T_bound - sw_bytes  # a legacy-sized peak would fit; the STEP peak must not
        with mock.patch.object(dwg.sm, 'build_root_graph_cpu', wraps=dwg.sm.build_root_graph_cpu) as builder:
            with self.assertRaises(sm.AdmissionRejected) as cm:
                b.step_day(*forcing(b), temperature_factor=1.0)
        builder.assert_not_called()
        self.assertEqual(cm.exception.stage, 'coupled.vegetation_admission')
        self.assertEqual(cm.exception.estimate['peak_bytes'], T_bound)
        self.assertTrue(b.weekly_pending)
        self.assertEqual(b.days, 7)
        avail['bytes'] = T_bound
        b.retry_weekly_update()
        self.assertFalse(b.weekly_pending)
        np.testing.assert_allclose(host(b.vegetation.B), host(a.vegetation.B), rtol=1e-12)
        np.testing.assert_allclose(host(b.vegetation.stepwat.E), host(a.vegetation.stepwat.E), rtol=1e-12)
        # construction guard: native availability one byte below the STEP init ledger refuses construction
        avail2 = {'bytes': init_total - 1}
        col = make_column(make_case(), available=avail2)
        state = ss.column_state(col, np.full((1, 1), WET))
        with self.assertRaises(sm.AdmissionRejected):
            CoupledSimulation(make_case(), col, state, inputs=inp, stepwat_traits=T1)


@unittest.skipUnless(NATIVE and dwg.gpu_available(), 'CuPy device with CC >= 6.0 not available')
class GpuStepTests(unittest.TestCase):
    def test_cpu_gpu_parity_and_device_residency(self):
        frac = {'grass': [0.6, 0.3, 0.1], 'shrub': [0.2, 0.3, 0.5]}
        case = make_case(nx=6, ny=6, dx=0.5, grass_count=3, shrub_count=2, grass_diameter=0.6, shrub_diameter=1.2,
                         root_multiplier=2.0, layers=3, max_edges=100000, root_layer_fractions=frac)
        rng = np.random.default_rng(7)
        inp = make_inputs(rng.uniform(0.5, 2.5, 5), rng.uniform(0.5, 2.5, 5), [0.3, 0.3, 0.3, 0.6, 0.6], mult=2.0,
                          pid=[0, 0, 0, 1, 1])
        traits = {'grass': GRASS, 'shrub': SHRUB}
        water = rng.uniform(RES_M, 0.03, (3, 36))
        active = [True] * 34 + [False, True]
        days = []
        for _ in range(9):
            add = rng.uniform(0, 0.002, (3, 36)) * np.array([[1.0], [0.0], [0.0]])
            add[:, 34] = 0.0
            pet = rng.uniform(0, 2e-8, 36)
            pet[34] = 0.0
            days.append((add, pet, rng.uniform(0.0, 1.0, 2)))
        g, _ = build(case, inp, traits, water, backend='cuda', active=active, diffusivity=1e-9)
        c, _ = build(case, inp, traits, water, backend='numba', active=active, diffusivity=1e-9)
        cp = sm.cp
        c.step_days(days)
        g.step_days([(cp.asarray(a), cp.asarray(p), t) for a, p, t in days])
        np.testing.assert_allclose(host(g.state.water_m), host(c.state.water_m), rtol=1e-9, atol=1e-15)
        for n in ('B', 'Dp', 'canopy', 'root', 'uptake_cum', 'uptake_week', 'gain_cum', 'loss_cum'):
            np.testing.assert_allclose(host(getattr(g.vegetation, n)), host(getattr(c.vegetation, n)), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(host(g.vegetation.stepwat.E), host(c.vegetation.stepwat.E), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(host(g.vegetation.stepwat.temp_integral), host(c.vegetation.stepwat.temp_integral),
                                   rtol=1e-12)
        self.assertEqual(g.vegetation.counters(), c.vegetation.counters())
        sg = g.summary()
        self.assertTrue(sg['ok'], sg['checks'])
        self.assertIs(type(g.vegetation.stepwat.E), cp.ndarray)
        recs = {r['name']: r for r in g.vegetation.array_records()}
        self.assertEqual(recs['stepwat.E']['where'], 'device')
        self.assertEqual(recs['stepwat.host.monthly_activity']['where'], 'host')
        self.assertEqual(sg['memory']['stepwat_host_mirror_bytes'], stepwat.host_mirror_bytes(2, True))
        self.assertEqual(g.vegetation.last_ledger['stepwat_host_mirror_bytes'], stepwat.host_mirror_bytes(2, True))
        self.assertEqual(g.vegetation.live_bytes() - dwg.state_bytes_estimate(5, 36, 3, n_profiles=2, soil_maps=1),
                         stepwat.state_bytes_estimate(5, 2))


if __name__ == '__main__':
    unittest.main()
