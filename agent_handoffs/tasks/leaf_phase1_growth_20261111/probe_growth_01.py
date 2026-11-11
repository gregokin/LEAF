"""Independent finite-ledger boundary probe; numerical fixture, not a model run."""
import copy
import json
import math
from pathlib import Path

import numpy as np

from leaf import growth, traits


def main():
    document = traits.parse_strict_json((Path.cwd() / 'examples/phase1/species.json').read_text())
    document['species']['grass']['growth']['traits']['mature_biomass_kg'] = 1e308
    rows = []
    for backend in ('numpy', 'numba'):
        table = growth.compile_species(traits.SpeciesCatalog(copy.deepcopy(document)), backend, 10**7)
        try:
            answer = growth.grow(
                table, B_kg=np.array([1e308]), E_kg=np.array([0.]),
                species_index=np.array([0], dtype=np.int32), live=np.array([True]),
                requested_m3=np.array([0.]), delivered_m3=np.array([0.]),
                activity_integral=np.array([.01, .01]),
                temperature_integral=np.array([.01, .01]), year_end=False, max_bytes=10**7)
        except growth.GrowthDataError as exc:
            rows.append({'backend': backend, 'rejected': True, 'reason': str(exc)})
        else:
            ledger = answer['ledger']
            bad = any(not math.isfinite(v) for v in ledger.values()
                      if isinstance(v, (float, np.floating)))
            rows.append({'backend': backend, 'rejected': False, 'ledger': ledger,
                         'nonfinite_ledger_reported_closed': bad and ledger['population_closed']})
    print(json.dumps({'cases': rows}, indent=2))
    return 1 if any(row.get('nonfinite_ledger_reported_closed') for row in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
