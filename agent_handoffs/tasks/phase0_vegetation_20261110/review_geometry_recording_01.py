"""Independent geometry, ownership and snapshot transaction probes."""
import json
import tempfile
import tracemalloc
from pathlib import Path
import numpy as np
from review_probes_01 import catalog, state
from leaf.canopy import maple_vegetation_view, measure_cover
from leaf.recording import SnapshotWriter
from leaf.state import Domain, constant_surface


def main():
    out = {}
    c = catalog()
    s = state(c)
    with tempfile.TemporaryDirectory() as tmp:
        w = SnapshotWriter(str(Path(tmp)/'missing'/'plants_{index}.csv'), cadence_days=7.0)
        before = w.manifest()
        try:
            w.write_initial(s)
            out['initial_write_failure_rollback'] = {'pass': False, 'observed': 'unexpected success'}
        except (OSError, ValueError):
            out['initial_write_failure_rollback'] = {'pass': w.manifest() == before,
                'before_next_due': before['next_due_days'], 'after_next_due': w.next_due_days}

    ground = constant_surface(s.domain, 10.0)  # anchor=3m, height=0.2m: fully buried
    try:
        view = maple_vegetation_view(s, c, current_surface=ground)
        height = view['fields']['height_m']
        out['buried_canopy_safe_default'] = {'pass': len(height) == 0,
            'observed': {'exported_heights': height.tolist(), 'n_unsupported': view['meta']['n_unsupported']}}
    except ValueError as exc:
        out['buried_canopy_safe_default'] = {'pass': True, 'error': str(exc)}

    digest = c.hash
    original = c.diameter_from_biomass('BOER', 1.0)
    public = c.document
    public['species']['BOER']['allometry']['diameter']['a'] = 100.0
    out['catalog_fingerprint_ownership'] = {
        'pass': c.canonical_text().find('100.0') < 0 or c.hash != digest,
        'observed': {'hash_unchanged': c.hash == digest, 'diameter_cached': c.diameter_from_biomass('BOER', 1.0),
                     'original_diameter': original, 'document_a': c.document['species']['BOER']['allometry']['diameter']['a']}}

    dom = Domain(512, 512, 1.0)
    cap = 2*512*512
    tracemalloc.start()
    result = measure_cover(dom, [256.0], [256.0], [400.0], subsamples=1,
                           max_cover_cells=512*512, max_cover_bytes=cap)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out['cover_working_memory_scope'] = {'declared_raster_bytes': result['raster_bytes'],
        'max_cover_bytes': cap, 'traced_peak_bytes': peak,
        'observation': 'Check admission includes the large float footprint temporary, not just boolean rasters.'}
    print(json.dumps(out, indent=2))
    return 0 if all(v.get('pass', True) for v in out.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
