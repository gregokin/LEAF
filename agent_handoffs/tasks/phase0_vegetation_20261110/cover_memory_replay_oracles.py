"""Independent verification of separate raster/scratch budgets and saved config replay."""
import json
import tracemalloc
import numpy as np
from generation_oracles import make_catalog, config
from leaf.state import Domain, constant_surface
from leaf.canopy import measure_cover, ResourceBoundError
from leaf.initialization import generate_population


def main():
    result = {}
    domain = Domain(512, 512, 1.0)
    raster_cap = 2*512*512
    tracemalloc.start()
    try:
        measure_cover(domain, [256.0], [256.0], [400.0], subsamples=1,
                      max_cover_cells=512*512, max_cover_bytes=raster_cap, max_working_bytes=1)
        good = False
    except ResourceBoundError:
        _, peak = tracemalloc.get_traced_memory()
        good = peak < 65536
    finally:
        tracemalloc.stop()
    result['scratch_preflight'] = {'pass': good}

    scratch_cap = 1_000_000
    tracemalloc.start()
    measured = measure_cover(domain, [256.0, 256.0], [256.0, 256.0], [400.0, 400.0],
        subsamples=1, max_cover_cells=512*512, max_cover_bytes=raster_cap, max_working_bytes=scratch_cap)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    admitted = measured['raster_bytes']+measured['working_bytes']+measured.get('index_bytes', 0)
    result['working_bound'] = {'pass': peak <= admitted+65536,
        'traced_peak_bytes': peak, 'admitted_array_bytes': admitted,
        'python_metadata_allowance_bytes': 65536,
        'raster_cap': raster_cap, 'scratch_cap': scratch_cap,
        'total_union_fraction': measured['total_union_fraction']}
    assert measured['total_union_fraction'] == 1.0

    catalog = make_catalog(); cfg = config(128, overlap='allowed', seed=4422)
    ground = constant_surface(cfg['domain'], 7.0)
    first, report = generate_population(catalog, cfg, surface=ground)
    try:
        saved_config = json.loads(json.dumps(report['generation_config']))
        second, _ = generate_population(catalog, saved_config, surface=ground)
        same = all(np.array_equal(getattr(first, key), getattr(second, key)) for key in first.ARRAY_NAMES)
        result['report_config_replays_exactly'] = {'pass': same}
    except (ValueError, KeyError, TypeError) as exc:
        result['report_config_replays_exactly'] = {'pass': False, 'error': str(exc)}
    print(json.dumps(result, indent=2))
    return 0 if all(item['pass'] for item in result.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
