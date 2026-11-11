"""Independent checks for the saved allocator correction; not task acceptance."""
import json
import warnings
from unittest import mock

import numpy as np

from leaf import uptake
from probe_uptake import oracles


def main():
    checked = {backend: oracles(uptake, backend) for backend in ('numpy', 'numba')}
    refusals = []
    for backend in ('numpy', 'numba'):
        # Construction only: no kernel ever receives these malformed offsets.
        try:
            uptake.RootMap.from_arrays(
                np.arange(4), np.array([0, np.iinfo(np.int64).max, np.iinfo(np.int64).min+1, -1, 0]),
                np.array([], dtype=np.int64), np.array([], dtype=np.float64),
                n_layers=1, n_cells=1, backend=backend, max_edges=0, max_bytes=10000)
        except uptake.UptakeError:
            refusals.append(f'{backend}:offset-overflow')
        else:
            raise AssertionError('Malformed offsets accepted')
        for resources in ([0, 0], [0, 1]):
            m = uptake.RootMap.from_arrays(np.array([1, 2]), np.array([0, 1, 2]),
                                           np.array(resources), np.ones(2), n_layers=1, n_cells=2,
                                           backend=backend, max_edges=10, max_bytes=10**7)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                try:
                    uptake.propose(m, np.full(2, 1e308), np.full(2, 1e308))
                except uptake.UptakeError:
                    refusals.append(f'{backend}:overflow:{resources}')
                else:
                    raise AssertionError('Overflowing plan returned')
    m = uptake.RootMap.from_arrays(np.array([1]), np.array([0, 1]), np.array([0]), np.array([1.0]),
                                   n_layers=1, n_cells=1, backend='numpy', max_edges=10, max_bytes=10**7)
    with mock.patch.object(uptake.RootMap, 'host_arrays', side_effect=AssertionError('Copy before admission')) as copied:
        try:
            m.with_backend('numpy', max_bytes=0)
        except uptake.AdmissionError:
            refusals.append('conversion-zero-budget-before-copy')
        else:
            raise AssertionError('Zero-budget conversion accepted')
        copied.assert_not_called()
    for data in ([True, 2], [1, 2]):
        try:
            uptake.RootMap.from_arrays(data, np.array([0, 1, 2]), np.array([0, 0]), np.ones(2),
                                       n_layers=1, n_cells=1, backend='numpy', max_edges=2, max_bytes=0)
        except uptake.UptakeError:
            refusals.append(f'untyped-input:{data}')
        else:
            raise AssertionError('Untyped input accepted before zero-budget refusal')
    plan = uptake.propose(m, np.array([0.1]), np.array([1.0]))
    native = np.nextafter(plan.proposed, np.inf)
    feedback = uptake.accept(plan, native)
    np.testing.assert_array_equal(feedback.accepted, native)
    assert feedback.closure['closed']
    assert feedback.summary['n_roundoff_excess'] == 1
    print(json.dumps({'scope': 'saved uptake correction only; roots/tests/docs incomplete',
                      'oracle_cases_passed': checked, 'refusals_passed': refusals,
                      'native_accepted_roundoff_preserved': True,
                      'passed': True}, indent=2))


if __name__ == '__main__':
    main()
