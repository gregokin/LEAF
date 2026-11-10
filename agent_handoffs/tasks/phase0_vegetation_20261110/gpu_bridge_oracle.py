"""Small real-CUDA Phase 0 state capture/restore/CSV check; no hydrology scenario."""
import copy
import json
import math
import tempfile
from pathlib import Path
import numpy as np
import cupy as cp
from generation_oracles import make_catalog
from leaf.traits import SpeciesCatalog, STEPWAT_GROWTH_LAW
from leaf.state import Domain, PlantState, constant_surface, write_plant_csv, read_plant_csv
from leaf.runtime import legacy_inputs_from_state, restore_seasonal_mass, capture_snapshot
from benchmarks.daily_water_growth import Simulation


def main():
    row = {'mature_biomass_kg': 1.0, 'intrinsic_rate_per_year': 0.5,
           'biomass_support_kg_per_m3': 10.0, 'live_fraction': 0.5,
           'canopy_biomass_density_kg_m2': 1.0, 'potential_transpiration_m_per_day': 0.003,
           'extra_growth_fraction': 0.25, 'monthly_activity': [1.0]*12}
    document = make_catalog().document
    for species in document['species'].values():
        species['allometry']['diameter'] = {'a': 2/math.sqrt(math.pi), 'b': 0.5}
        species['growth'] = {'law': STEPWAT_GROWTH_LAW, 'traits': copy.deepcopy(row)}
    catalog = SpeciesCatalog(document); domain = Domain(2, 2, 1.0)
    records = []
    for pid, sid, xy, B, E in [(17, 'BOER', 0.5, 0.01, 0.001), (42, 'PROS', 1.5, 0.04, 0.002)]:
        records.append({'plant_id': pid, 'species': sid, 'x_m': xy, 'y_m': xy,
                        'establishment_ground_m': 'SURFACE', 'agb_dry_kg': B,
                        'seasonal_agb_dry_kg': E, 'age_days': 100.0, 'status': 'live'})
    population = PlantState.from_records(catalog, domain, records, surface=constant_surface(domain, 100.0))
    traits = catalog.stepwat_traits()
    bridge = legacy_inputs_from_state(population, catalog, stepwat_traits=traits, seasonal_policy='restore')
    case = {'name': 'phase0-gpu-bridge', 'nx': 2, 'ny': 2, 'dx': 1.0,
            'grass_count': 1, 'shrub_count': 1, 'grass_diameter': 0.1,
            'shrub_diameter': 0.2, 'root_multiplier': 4.0, 'pattern': 'uniform',
            'seed': 91, 'layers': 2, 'max_edges': 10000, 'max_bytes': 50_000_000,
            'root_layer_fractions': bridge['root_layer_fractions']}
    case.update(bridge['case_fragment'])
    sim = Simulation(case, 'gpu', inputs=bridge['inputs'], vegetation_only=True, stepwat_traits=traits)
    assert restore_seasonal_mass(population, sim) == 2
    np.testing.assert_array_equal(cp.asnumpy(sim.stepwat.E), population.seasonal_agb_dry_kg)
    sim.commit_daily_uptake(cp.zeros(2, dtype=cp.float64), temperature_factor=0.7)
    population.advance_time(1.0)
    saved = capture_snapshot(population, sim, in_place=False)
    np.testing.assert_allclose(saved.agb_dry_kg, cp.asnumpy(sim.B), rtol=0, atol=0)
    np.testing.assert_array_equal(saved.seasonal_agb_dry_kg, cp.asnumpy(sim.stepwat.E))
    np.testing.assert_array_equal(saved.age_days, [101.0, 101.0])
    before = cp.asnumpy(sim.stepwat.E)
    try:
        restore_seasonal_mass(saved, sim)
        raise AssertionError('Midrun seasonal restore was accepted')
    except ValueError:
        pass
    np.testing.assert_array_equal(cp.asnumpy(sim.stepwat.E), before)
    with tempfile.TemporaryDirectory(prefix='leaf-phase0-gpu-', dir='/tmp') as tmp:
        path = Path(tmp)/'vegetation.csv'; write_plant_csv(saved, path)
        replay = read_plant_csv(path, catalog)
        for key in saved.ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(saved, key), getattr(replay, key))
    print(json.dumps({'pass': True, 'physical_gpu': 1, 'logical_gpu': cp.cuda.Device().id,
                      'compute_capability': cp.cuda.Device().compute_capability,
                      'plants': 2, 'accepted_zero_uptake_fixture_days': sim.day,
                      'fresh_restore': True, 'midrun_restore_refused': True,
                      'B_E_capture_by_id': True, 'exact_csv_replay': True,
                      'hydrology_constructed': False, 'calibrated_parameters': False}, indent=2))


if __name__ == '__main__':
    main()
