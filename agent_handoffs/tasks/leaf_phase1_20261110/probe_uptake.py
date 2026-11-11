"""Codex numerical/admission probes against an explicitly frozen module.

Exit success means the probes completed, not that the candidate was accepted.
Individual observed defects are recorded in JSON. No production code is edited.
"""
from pathlib import Path
import importlib.util
import json
import sys
import tracemalloc

import numpy as np

from scalar_oracle import CASES, reference


def load(path):
    name = 'leaf._phase1_uptake_review'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def case_arrays(case):
    resource_names = list(case['available'])
    resource_index = {r: i for i, r in enumerate(resource_names)}
    ids, offsets, resources, weights, demand = [], [0], [], [], []
    for plant in case['plants']:
        ids.append(plant['id'])
        demand.append(float(plant['demand']))
        for r, w in plant['roots'].items():
            resources.append(resource_index[r])
            from fractions import Fraction
            weights.append(float(Fraction(w)))
        offsets.append(len(resources))
    return (np.array(ids, dtype=np.int64), np.array(offsets, dtype=np.int64),
            np.array(resources, dtype=np.int64), np.array(weights, dtype=np.float64),
            np.array(demand, dtype=np.float64),
            np.array([case['available'][r] for r in resource_names], dtype=np.float64))


def oracles(module, backend):
    checked = []
    for name, case in CASES.items():
        ids, offsets, resources, weights, demand, available = case_arrays(case)
        root_map = module.RootMap.from_arrays(ids, offsets, resources, weights,
                                              n_layers=1, n_cells=len(available),
                                              backend=backend, max_edges=1000, max_bytes=10**8)
        expected = reference(**case)
        plan = module.propose(root_map, demand, available)
        accepted = np.array([float(v) for v in expected['accepted'].values()])
        feedback = module.accept(plan, accepted)
        for attr in ('delivered', 'unmet', 'satisfaction'):
            np.testing.assert_allclose(getattr(feedback, attr),
                                       [float(p[attr]) for p in expected['plants']],
                                       rtol=2e-14, atol=0)
        np.testing.assert_allclose(plan.request,
                                   [float(v) for v in expected['combined_request'].values()],
                                   rtol=2e-14, atol=0)
        assert feedback.closure['closed']
        checked.append(name)
    return checked


def nonfinite(module):
    m = module.RootMap.from_arrays(np.array([1, 2]), np.array([0, 1, 2]),
                                   np.array([0, 1]), np.ones(2), n_layers=1, n_cells=2,
                                   backend='numpy', max_edges=100, max_bytes=10**8)
    try:
        plan = module.propose(m, np.full(2, 1e308), np.full(2, 1e308))
        feedback = module.accept(plan, plan.proposed)
    except (ValueError, FloatingPointError) as exc:
        return {'rejected': True, 'error': str(exc)}
    return {'rejected': False, 'defect_reproduced': True,
            'plan_demand_total': plan.summary['demand_total_m3'],
            'closure': feedback.closure}


def admission(module):
    p = 200_000
    ids, offsets, resources, weights = np.arange(p), np.arange(p+1), np.zeros(p, dtype=np.int64), np.ones(p)
    m = module.RootMap.from_arrays(ids, offsets, resources, weights, n_layers=1, n_cells=1,
                                   backend='numba', max_edges=p, max_bytes=10**9)
    demand = np.full(p, 0.001)
    plan = module.propose(m, demand, np.array([500.0]))
    # Warm all JIT functions on a tiny input before measuring any allocator work.
    tiny = module.RootMap.from_arrays(np.array([1]), np.array([0, 1]), np.array([0]), np.array([1.]),
                                      n_layers=1, n_cells=1, backend='numba', max_edges=1, max_bytes=10**6)
    module.propose(tiny, np.array([0.001]), np.array([0.001])).accept_all()
    est = module.estimate_feedback_bytes(p, p, 1, 'numba')
    resident = m.resident_bytes + plan.summary['admission']['owned_bytes']
    cap = resident + est['total_bytes']
    tracemalloc.start()
    feedback = module.accept(plan, plan.proposed, max_bytes=cap)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    feedback_result = {'declared_operation_bytes': est['total_bytes'],
                       'traced_operation_peak_bytes': peak,
                       'traced_retained_operation_bytes': current,
                       'resident_bytes': resident, 'cap_bytes': cap,
                       'declared_admission_accepted': True,
                       'defect_reproduced': peak > est['total_bytes'] + 100_000,
                       'method': 'tracemalloc after tiny numba warmup; existing map/plan/caller arrays excluded; '
                                 '100 kB allowance only distinguishes Python bookkeeping from array-scale error'}
    del feedback
    tracemalloc.start()
    try:
        m.with_backend('numba', max_bytes=0)
    except module.AdmissionError:
        pass
    else:
        raise AssertionError('zero budget should reject')
    _, transfer_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    sequence = [list(range(p)), list(range(p+1)), [0]*p, [1.0]*p]
    tracemalloc.start()
    try:
        module.RootMap.from_arrays(*sequence, n_layers=1, n_cells=1,
                                  backend='numpy', max_edges=0, max_bytes=0)
    except module.AdmissionError:
        pass
    else:
        raise AssertionError('zero budget should reject')
    _, sequence_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'feedback': feedback_result,
            'with_backend_zero_cap': {'traced_peak_bytes': transfer_peak,
                                      'defect_reproduced': transfer_peak > 100_000},
            'sequence_input_zero_cap': {'traced_peak_bytes': sequence_peak,
                                        'defect_reproduced': sequence_peak > 100_000}}


def mixed_bool_ids(module):
    try:
        module.RootMap.from_arrays([True, 2], [0, 1, 2], [0, 0], [1.0, 1.0],
                                  n_layers=1, n_cells=1, backend='numpy', max_edges=2, max_bytes=10**6)
    except ValueError:
        return {'rejected': True}
    return {'rejected': False, 'defect_reproduced': True}


def main():
    module = load(Path(sys.argv[1]).resolve())
    results = {'scope': 'interim independent probes; not phase acceptance',
               'numpy_oracles_passed': oracles(module, 'numpy'),
               'numba_oracles_passed': oracles(module, 'numba'),
               'nonfinite_aggregate': nonfinite(module),
               'admission': admission(module),
               'mixed_bool_ids': mixed_bool_ids(module)}
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
