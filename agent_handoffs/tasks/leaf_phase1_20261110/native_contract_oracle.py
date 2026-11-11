"""Independent small-fixture check of the new allocator with actual native SYRUP.

No growth or hydrology time integration is run. Toy demands isolate the water
contract; they are not ecological parameter estimates. SYRUP alone constructs
the fresh water inventory. The availability projection reads the native
operator's residual floor; it introduces no independently configurable floor.
"""
import json

import numpy as np

from leaf import uptake
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st


def column_state(layers, cells, area, available_volume, backend, active=None):
    soil = ss.soil_layers(np.arange(layers + 1, dtype=np.float64), theta_residual=0.05,
                          theta_field_capacity=0.30, theta_saturation=0.60,
                          diffusivity_m2_s=0.0, theta_evaporation_stress=0.10)
    mask = np.ones(cells, dtype=bool) if active is None else np.array(active, dtype=bool)
    column = ss.prepare_column(soil, mask, area, backend=backend)
    state = ss.column_state(column, 0.05 + np.array(available_volume, dtype=np.float64) / area,
                            t_s=123.25)
    return column, state, st.prepare_transpiration(column)


def execute(backend, *, ids, offsets, resources, weights, demand, available,
            area, expected, active=None):
    available = np.array(available, dtype=np.float64)
    layers, cells = available.shape
    column, state, operator = column_state(layers, cells, area, available, backend, active)
    before = state.water_m.copy()
    m = uptake.RootMap.from_arrays(np.array(ids, dtype=np.int64), np.array(offsets, dtype=np.int64),
                                   np.array(resources, dtype=np.int64), np.array(weights, dtype=np.float64),
                                   n_layers=layers, n_cells=cells, backend=backend,
                                   max_edges=10000, max_bytes=10**8)
    # This is a read-only projection of the native inventory/floor, not a second stock.
    offered = np.maximum(state.water_m - operator.residual_xp[:, None], 0.0) * area
    offered[:, ~np.asarray(column.active_u8, dtype=bool)] = 0.0
    plan = uptake.propose(m, np.array(demand, dtype=np.float64), offered)
    native = st.withdraw_transpiration(operator, state, plan.proposed_grid() / area)
    actual_volume = native.actual_m * area
    feedback = uptake.accept(plan, actual_volume)
    np.testing.assert_allclose(feedback.delivered, expected, rtol=5e-14, atol=0)
    np.testing.assert_allclose(feedback.delivered.sum(), native.budget.actual_m3, rtol=5e-14, atol=0)
    np.testing.assert_array_equal(state.water_m, before)
    assert native.state.t_s == state.t_s == 123.25
    assert abs(native.budget.residual_m3) <= native.budget.bound_m3
    return {'delivered_m3': feedback.delivered.tolist(),
            'satisfaction': feedback.satisfaction.tolist(),
            'native_actual_m3': native.budget.actual_m3,
            'native_water_residual_m3': native.budget.residual_m3,
            'native_water_bound_m3': native.budget.bound_m3,
            'leaf_water_residual_m3': feedback.closure['residual_m3'],
            'preserves_exact_native_feedback': bool(np.array_equal(feedback.accepted, actual_volume.ravel())),
            'native_clock_unchanged': native.state.t_s == state.t_s,
            'withdrawable_remaining_m3': (np.maximum(native.state.water_m - operator.residual_xp[:, None], 0.0)
                                           * area).tolist()}


def main():
    results = {}
    for backend in ('numpy', 'numba'):
        results[backend] = {
            'user_two_layers': execute(backend, ids=[1], offsets=[0, 2], resources=[0, 1],
                                      weights=[0.5, 0.5], demand=[100.0], available=[[25.0], [200.0]],
                                      area=1000.0, expected=[75.0]),
            'competition_different_roots': execute(backend, ids=[1, 9], offsets=[0, 2, 3],
                                                   resources=[0, 1, 0], weights=[0.5, 0.5, 1.0],
                                                   demand=[100.0, 50.0], available=[[25.0], [200.0]],
                                                   area=1000.0, expected=[62.5, 12.5]),
            'shallow_vs_deep': execute(backend, ids=[1, 9], offsets=[0, 2, 4], resources=[0, 1, 0, 1],
                                      weights=[0.8, 0.2, 0.2, 0.8], demand=[100.0, 100.0],
                                      available=[[0.0], [100.0]], area=1000.0, expected=[20.0, 80.0]),
            'inactive_cell_no_redistribution': execute(backend, ids=[1], offsets=[0, 2], resources=[0, 1],
                                                      weights=[0.5, 0.5], demand=[200.0],
                                                      available=[[150.0, 150.0]], area=1000.0,
                                                      expected=[100.0], active=[False, True]),
        }
        for area in (0.0625, 1.0, 2.3):
            results[backend][f'area_{area}'] = execute(
                backend, ids=[7], offsets=[0, 1], resources=[0], weights=[1.0], demand=[1e-4],
                available=[[0.015 * area]], area=area, expected=[1e-4])
    # Non-binary area exposes real native depth<->volume roundoff. Keeping the
    # exact accepted vector matters even when the change is only one ulp.
    count = 32
    rng = np.random.default_rng(17)
    demand = rng.uniform(1e-6, 1e-3, count)
    results['roundoff_feedback'] = execute(
        'numba', ids=list(range(count)), offsets=list(range(count + 1)), resources=list(range(count)),
        weights=[1.0] * count, demand=demand,
        available=np.full((1, count), 0.01), area=0.09, expected=demand)
    print(json.dumps({'scope': 'actual native pure-withdrawal fixtures; no dated growth driver or transport run',
                      'results': results,
                      'all_numeric_water_oracles_passed': True,
                      'roundoff_feedback_defect_reproduced': not results['roundoff_feedback']['preserves_exact_native_feedback']},
                     indent=2))


if __name__ == '__main__':
    main()
