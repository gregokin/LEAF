"""Independent cover/count/exclusion checks and bounded Phase 0 CPU capacity."""
import argparse
import copy
import json
import math
import resource
import time
import numpy as np
from independent_reference import poisson_count_prior, union_cover_random_points, assert_nonoverlap_small
from review_probes_01 import catalog
from leaf.traits import SpeciesCatalog
from leaf.state import Domain, constant_surface
from leaf.initialization import generate_population


def make_catalog():
    document = catalog().document
    shrub = copy.deepcopy(document['species']['BOER'])
    shrub.update(functional_type='shrub',
                 allometry={'model': 'power_law_v1', 'diameter': {'a': 2.0, 'b': 0.5},
                            'height': {'a': 1.5, 'b': 0.3}},
                 roots={'lateral_radius_multiplier': 4.0, 'layer_fractions': [0.25, 0.75]})
    document['species']['PROS'] = shrub
    return SpeciesCatalog(document)


def config(nx, *, overlap, seed, grass_cover=0.06):
    return {'domain': Domain(nx, nx, 0.25), 'seed': seed, 'overlap': overlap,
            'cover_tolerance': 0.005, 'cover_subsamples': 4, 'max_iterations': 12,
            'species': {'BOER': {'target_cover': grass_cover, 'diameter_mean_m': 0.15,
                                  'diameter_std_m': 0.05, 'initial_age_days': 100.0},
                        'PROS': {'target_cover': 0.05, 'diameter_mean_m': 2.0,
                                  'diameter_std_m': 0.3,
                                  'initial_age_days': {'distribution': 'uniform',
                                                       'min_days': 300.0, 'max_days': 3000.0}}}}


def check_case(nx, overlap, seed, *, probes):
    c = make_catalog(); cfg = config(nx, overlap=overlap, seed=seed)
    dom = cfg['domain']; ground = constant_surface(dom, 4.25)
    start = time.perf_counter()
    state, report = generate_population(c, cfg, surface=ground)
    duration = time.perf_counter()-start
    diameters = c.diameter_from_biomass(state.species_index, state.agb_dry_kg)
    stats = []
    bounds = (dom.origin_x_m, dom.origin_y_m, dom.width_m, dom.height_m)
    for sp in report['species']:
        sid = sp['species']; wanted = cfg['species'][sid]
        selected = state.species_index == c.index(sid)
        if overlap == 'allowed':
            prior = poisson_count_prior(dom.area_m2, wanted['target_cover'],
                                        wanted['diameter_mean_m'], wanted['diameter_std_m'])
            assert abs(sp['initial_count_estimate']-prior) <= 0.51
        estimate = union_cover_random_points(state.x_m[selected], state.y_m[selected],
                                             0.5*diameters[selected], bounds,
                                             samples=probes, seed=9123+c.index(sid))
        tolerance = cfg['cover_tolerance'] + 4*estimate['standard_error']
        assert abs(estimate['fraction']-wanted['target_cover']) <= tolerance, (sid, estimate, sp)
        assert abs(sp['measured_union_cover']-estimate['fraction']) <= tolerance, (sid, estimate, sp)
        stats.append({'species': sid, 'count': int(selected.sum()), 'target': wanted['target_cover'],
                      'reported_union': sp['measured_union_cover'], 'independent_union': estimate})
    clearance = None
    if overlap == 'disallowed':
        clearance = assert_nonoverlap_small(state.x_m, state.y_m, 0.5*diameters)
    assert np.array_equal(state.establishment_ground_m, np.full(len(state), 4.25))
    assert not np.any(state.establishment_unresolved)
    assert np.allclose(c.biomass_from_diameter(state.species_index, diameters), state.agb_dry_kg,
                       rtol=2e-15, atol=0)
    state_bytes = sum(getattr(state, k).nbytes for k in state.ARRAY_NAMES)
    return {'grid': [nx, nx], 'dx_m': 0.25, 'overlap': overlap, 'plants': len(state),
            'generation_wall_seconds': duration, 'plant_array_bytes': state_bytes,
            'cover_measurement': report['cover_measurement'], 'species': stats,
            'minimum_nonoverlap_clearance_m': clearance,
            'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--capacity', action='store_true'); args = p.parse_args()
    cases = [check_case(128, 'allowed', 4401, probes=250_000),
             check_case(128, 'disallowed', 7751, probes=250_000)]
    if args.capacity:
        cases.append(check_case(512, 'allowed', 20261110, probes=150_000))
    print(json.dumps({'pass': True, 'cases': cases}, indent=2))


if __name__ == '__main__':
    main()
