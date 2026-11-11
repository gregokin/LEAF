"""Require actual CUDA and check saved allocator math/guards on the selected GPU."""
import json
from unittest import mock

import cupy as cp
import numpy as np

from leaf import uptake
from probe_uptake import case_arrays
from scalar_oracle import CASES, reference


def main():
    status = uptake.backend_status('cuda')
    if not status['available']:
        raise RuntimeError(f'Explicit CUDA check cannot skip: {status}')
    checked = []
    for name, case in CASES.items():
        ids, offsets, resources, weights, demand, available = case_arrays(case)
        m = uptake.RootMap.from_arrays(ids, offsets, resources, weights, n_layers=1, n_cells=len(available),
                                       backend='cuda', max_edges=1000, max_bytes=10**8)
        plan = uptake.propose(m, cp.asarray(demand), cp.asarray(available))
        expected = reference(**case)
        accepted = cp.asarray([float(v) for v in expected['accepted'].values()])
        feedback = uptake.accept(plan, accepted)
        for attr in ('delivered', 'unmet', 'satisfaction'):
            np.testing.assert_allclose(getattr(feedback, attr).get(),
                                       [float(p[attr]) for p in expected['plants']], rtol=2e-13, atol=0)
        assert feedback.closure['closed']
        checked.append(name)
    ids = np.array([1])
    offsets = np.array([0, 2])
    resources = np.array([0, 1])
    weights = np.array([0.5, 0.5])
    m = uptake.RootMap.from_arrays(ids, offsets, resources, weights, n_layers=2, n_cells=1,
                                   backend='cuda', max_edges=10, max_bytes=10**7)
    d, a = cp.array([100.0]), cp.array([[25.0], [200.0]])
    plan = uptake.propose(m, d, a)
    feedback = plan.accept_all()
    assert float(feedback.delivered[0]) == 75.0
    refused = []
    # Change only the reported current-device identity. No other device is
    # selected, initialized or allocated on for these negative guard checks.
    with mock.patch.object(uptake, '_current_device', return_value=m.device_id+1):
        for name, call in [('proposal', lambda: uptake.propose(m, d, a)),
                           ('feedback', lambda: uptake.accept(plan, plan.proposed))]:
            try:
                call()
            except uptake.UptakeError:
                refused.append(name)
            else:
                raise AssertionError(f'Missing current-device guard: {name}')
    try:
        uptake.accept(plan, plan.proposed * 1.01)
    except uptake.UptakeError:
        refused.append('actual-excess')
    else:
        raise AssertionError('Material accepted excess not refused')
    # Device-native construction exercises validation/sort/reduction path too.
    gm = uptake.RootMap.from_arrays(cp.asarray(ids), cp.asarray(offsets), cp.asarray(resources), cp.asarray(weights),
                                    n_layers=2, n_cells=1, backend='cuda', max_edges=10, max_bytes=10**7)
    assert float(uptake.propose(gm, d, a).accept_all().delivered[0]) == 75.0
    cp.cuda.Device().synchronize()
    print(json.dumps({'scope': 'partial allocator CUDA core only; no full phase acceptance',
                      'actual_backend': status, 'oracle_cases_passed': checked,
                      'layered_user_case_passed': True, 'device_native_map_passed': True,
                      'guards_refused': refused, 'passed': True}, indent=2))


if __name__ == '__main__':
    main()
