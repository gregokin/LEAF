"""Post-correction geometry/recording oracles, allowing legitimate immutable APIs."""
import json
import tempfile
import tracemalloc
from pathlib import Path
import numpy as np
from review_probes_01 import catalog, state
from leaf.canopy import maple_vegetation_view, measure_cover, ResourceBoundError
from leaf.recording import SnapshotWriter
from leaf.state import Domain, constant_surface


def main():
    out = {}; c = catalog(); s = state(c)
    with tempfile.TemporaryDirectory() as tmp:
        existing = Path(tmp)/'plants_0.csv'; existing.write_bytes(b'preserve this file')
        writer = SnapshotWriter(str(Path(tmp)/'plants_{index}.csv'), cadence_days=7.0)
        before = writer.manifest()
        try:
            writer.write_initial(s)
            good = False
        except (OSError, ValueError):
            good = writer.manifest() == before and existing.read_bytes() == b'preserve this file'
        out['initial_collision_transaction'] = {'pass': good}

    try:
        buried = maple_vegetation_view(s, c, current_surface=constant_surface(s.domain, 10.0))
        good = len(buried['fields']['height_m']) == 0
    except ValueError:
        good = True
    out['fully_buried_safe_default'] = {'pass': good}
    partial = maple_vegetation_view(s, c, current_surface=constant_surface(s.domain, 3.05))
    out['partial_burial_preserves_top'] = {
        'pass': bool(np.allclose(partial['fields']['height_m'], [0.15], rtol=0, atol=2e-14))}
    normal = maple_vegetation_view(s, c, current_surface=constant_surface(s.domain, 3.0))
    out['initial_native_geometry'] = {'pass': bool(np.allclose(normal['fields']['height_m'], [0.2], rtol=0, atol=1e-15)
                                                 and np.allclose(normal['fields']['semi_major_m'], [0.075]))}

    digest, canonical = c.hash, c.canonical_text()
    try:
        public = c.document; public['species']['BOER']['allometry']['diameter']['a'] = 100.0
    except (TypeError, AttributeError):
        pass
    out['catalog_fingerprint_ownership'] = {'pass': c.hash == digest and c.canonical_text() == canonical
                                                   and c.diameter_from_biomass('BOER', 1.0) == 1.5}

    dom = Domain(512, 512, 1.0); cap = 2*512*512
    tracemalloc.start()
    try:
        result = measure_cover(dom, [256.0], [256.0], [400.0], subsamples=1,
                               max_cover_cells=512*512, max_cover_bytes=cap)
        _, peak = tracemalloc.get_traced_memory()
        good = peak <= cap+65536
        action = 'measured within the declared working budget'
    except ResourceBoundError:
        _, peak = tracemalloc.get_traced_memory()
        good = peak <= 65536  # reject before the large raster allocations
        action = 'refused before allocation because scratch does not fit'
    finally:
        tracemalloc.stop()
    out['tight_cover_memory_admission'] = {'pass': good, 'action': action,
        'traced_peak_bytes': peak, 'max_cover_bytes': cap,
        'python_metadata_allowance_bytes': 65536}
    result = measure_cover(dom, [256.0], [256.0], [400.0], subsamples=1,
                           max_cover_cells=512*512)  # existing default byte cap, unchanged
    out['large_circle_cover'] = {'pass': result['total_union_fraction'] == 1.0}
    print(json.dumps(out, indent=2))
    return 0 if all(v['pass'] for v in out.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
