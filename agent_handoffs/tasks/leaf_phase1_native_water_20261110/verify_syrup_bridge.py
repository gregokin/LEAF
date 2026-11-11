"""Codex tests of the two saved Claude modules composed for review only.

All water inventories/debits use actual native SYRUP. No dated growth driver,
transport event, ecological parameterization or phase acceptance is implied.
"""
from dataclasses import replace
import json
import sys
from unittest import mock

import numpy as np

from leaf import uptake
from leaf import syrup_water as bridge
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st


def host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


def fixture(map_backend, native_backend, *, demand, offsets, resources, weights, available,
            area=1000.0, active=None, available_bytes=None):
    if native_backend == 'cuda':
        import cupy as xp
    else:
        xp = np
    offered = np.asarray(available, dtype=np.float64)
    layers, cells = offered.shape
    soil = ss.soil_layers(np.arange(layers+1, dtype=np.float64), theta_residual=0.05,
                          theta_field_capacity=0.30, theta_saturation=0.60,
                          diffusivity_m2_s=0.0, theta_evaporation_stress=0.10)
    mask = np.ones(cells, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    col = ss.prepare_column(soil, xp.asarray(mask), area, backend=native_backend,
                            available_bytes_fn=None if available_bytes is None else lambda: available_bytes['value'])
    state = ss.column_state(col, xp.asarray(0.05 + offered / area), t_s=123.25)
    op = st.prepare_transpiration(col)
    demand = xp.asarray(demand, dtype=xp.float64)
    root_map = uptake.RootMap.from_arrays(
        np.arange(len(demand), dtype=np.int64), np.asarray(offsets, dtype=np.int64),
        np.asarray(resources, dtype=np.int64), np.asarray(weights, dtype=np.float64),
        n_layers=layers, n_cells=cells, backend=map_backend, max_edges=10**6, max_bytes=10**8)
    return root_map, demand, op, state


BASE = dict(demand=[100.0], offsets=[0, 2], resources=[0, 1], weights=[0.5, 0.5], available=[[25.0], [200.0]])


def run_case(map_backend, native_backend, case, expected):
    m, d, op, state = fixture(map_backend, native_backend, **case)
    before, dbefore = host(state.water_m).copy(), host(d).copy()
    real = st.withdraw_transpiration
    with mock.patch.object(st, 'withdraw_transpiration', wraps=real) as debit:
        result = bridge.withdraw_plants(m, d, op, state, max_bytes=10**8)
        assert debit.call_count == 1
        assert debit.call_args.args[0] is op and debit.call_args.args[1] is state
    np.testing.assert_allclose(host(result.delivered_m3), expected, rtol=5e-13, atol=0)
    np.testing.assert_array_equal(host(result.feedback.accepted), host(result.accepted_volume_m3).ravel())
    np.testing.assert_array_equal(host(result.accepted_volume_m3), host(result.native.actual_m) * op.column.cell_area_m2)
    np.testing.assert_array_equal(host(state.water_m), before)
    np.testing.assert_array_equal(host(d), dbefore)
    assert result.native_state.t_s == state.t_s == 123.25
    assert result.conservation['closed'] and result.conservation['requested_matches_proposal']
    assert abs(result.native.budget.residual_m3) <= result.native.budget.bound_m3
    return {'map_backend': map_backend, 'native_backend': native_backend,
            'delivered_m3': host(result.delivered_m3).tolist(),
            'native_actual_m3': result.native.budget.actual_m3,
            'water_residual_m3': result.conservation['delivered_minus_native_actual_m3'],
            'memory_total_bytes': result.memory['total_bytes'], 'native_calls': 1}


def guards(map_backend, native_backend):
    m, d, op, state = fixture(map_backend, native_backend, **BASE)
    before = host(state.water_m).copy()
    passed = []
    with mock.patch.object(st, 'withdraw_transpiration', wraps=st.withdraw_transpiration) as debit, \
            mock.patch.object(bridge, '_project', side_effect=AssertionError('Projection before admission')) as project:
        try:
            bridge.withdraw_plants(m, d, op, state, max_bytes=0)
        except bridge.SyrupWaterAdmissionError:
            passed.append('zero-budget-before-projection-or-debit')
        else:
            raise AssertionError('Zero budget accepted')
        debit.assert_not_called()
        project.assert_not_called()
    with mock.patch.object(st, 'withdraw_transpiration', wraps=st.withdraw_transpiration) as debit:
        try:
            bridge.withdraw_plants(m, d[:0], op, state, max_bytes=10**8)
        except bridge.BindingError:
            passed.append('bad-demand-shape-before-debit')
        else:
            raise AssertionError('Bad shape accepted')
        debit.assert_not_called()
    with mock.patch.object(st, 'withdraw_transpiration', wraps=st.withdraw_transpiration) as debit, \
            mock.patch.object(uptake, 'accept', side_effect=RuntimeError('injected gather failure')):
        try:
            bridge.withdraw_plants(m, d, op, state, max_bytes=10**8)
        except RuntimeError as exc:
            assert str(exc) == 'injected gather failure'
            passed.append('gather-failure-after-pure-native-stage')
        else:
            raise AssertionError('Injected failure ignored')
        assert debit.call_count == 1
    np.testing.assert_array_equal(host(state.water_m), before)
    assert state.t_s == 123.25
    # A narrow TEST acceptance cap still uses the real native water debit once.
    # It is not an added production law. Budget request/unmet fields describe
    # the full proposal, while physical actual/storage come from the real call.
    real = st.withdraw_transpiration
    def half_acceptance(operator, old, requested, **kwargs):
        accepted = real(operator, old, requested * 0.5, **kwargs)
        full = float(host(requested).sum()) * operator.column.cell_area_m2
        budget = replace(accepted.budget, requested_m3=full,
                         unmet_m3=full-accepted.budget.actual_m3, request_residual_m3=0.0)
        return replace(accepted, unmet_m=requested-accepted.actual_m,
                       scale=accepted.actual_m/requested, budget=budget)
    with mock.patch.object(st, 'withdraw_transpiration', side_effect=half_acceptance) as debit:
        result = bridge.withdraw_plants(m, d, op, state, max_bytes=10**8)
        assert debit.call_count == 1
    np.testing.assert_allclose(host(result.delivered_m3), [37.5], rtol=5e-13, atol=0)
    np.testing.assert_allclose(host(result.satisfaction), [0.375], rtol=5e-13, atol=0)
    np.testing.assert_array_equal(host(state.water_m), before)
    passed.append('native-shortfall-uses-actual-one-real-debit')
    return passed


def main():
    mode = sys.argv[1]
    combinations = [('cuda', 'cuda')] if mode == 'cuda' else [
        ('numpy', 'numpy'), ('numba', 'numba'), ('numpy', 'numba'), ('numba', 'numpy')]
    cases = [('user', BASE, [75.0]),
             ('competition', dict(demand=[100., 50.], offsets=[0, 2, 3], resources=[0, 1, 0],
                                  weights=[.5, .5, 1.], available=[[25.], [200.]]), [62.5, 12.5]),
             ('different_layers', dict(demand=[100., 100.], offsets=[0, 2, 4], resources=[0, 1, 0, 1],
                                       weights=[.8, .2, .2, .8], available=[[0.], [100.]]), [20., 80.]),
             ('inactive', dict(demand=[200.], offsets=[0, 2], resources=[0, 1], weights=[.5, .5],
                               available=[[150., 150.]], active=[False, True]), [100.]),
             ('empty', dict(demand=[], offsets=[0], resources=[], weights=[], available=[[25.], [200.]]), [])]
    results, guard_results = [], {}
    for mb, nb in combinations:
        for name, case, expected in cases:
            results.append(dict(case=name, **run_case(mb, nb, case, expected)))
        for area in (0.0625, 1.0, 2.3):
            case = dict(demand=[1e-4], offsets=[0, 1], resources=[0], weights=[1.],
                        available=[[0.015*area]], area=area)
            results.append(dict(case=f'area_{area}', **run_case(mb, nb, case, [1e-4])))
        guard_results[f'{mb}/{nb}'] = guards(mb, nb)
    rng = np.random.default_rng(17)
    demand = rng.uniform(1e-6, 1e-3, 32)
    case = dict(demand=demand, offsets=list(range(33)), resources=list(range(32)),
                weights=[1.]*32, available=np.full((1, 32), .01), area=.09)
    mb, nb = combinations[0]
    results.append(dict(case='nonbinary_area_exact_native_feedback', **run_case(mb, nb, case, demand)))
    print(json.dumps({'scope': 'review composition of interrupted modules; no phase acceptance',
                      'native_provenance': bridge.native_provenance(verify=True),
                      'case_count': len(results), 'results': results, 'guards': guard_results,
                      'passed': True}, indent=2))


if __name__ == '__main__':
    main()
