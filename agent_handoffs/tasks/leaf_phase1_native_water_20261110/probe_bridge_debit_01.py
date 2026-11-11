"""Independently test whether a forged fresh result can claim T without a debit."""
from dataclasses import replace
import json
from unittest import mock

import numpy as np

from leaf import syrup_water as bridge
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st
from verify_syrup_bridge import BASE, fixture


def main():
    root_map, demand, operator, old = fixture('numba', 'numba', **BASE)
    real = st.withdraw_transpiration

    def missing_debit(op, state, request, **kwargs):
        native = real(op, state, request, **kwargs)
        unchanged_copy = ss.SoilColumnState(state.water_m.copy(), state.t_s)
        return replace(native, state=unchanged_copy)

    with mock.patch.object(st, 'withdraw_transpiration', side_effect=missing_debit) as call:
        try:
            result = bridge.withdraw_plants(root_map, demand, operator, old, max_bytes=10**8)
        except bridge.SyrupWaterError as exc:
            print(json.dumps({'rejected': True, 'native_calls': call.call_count,
                              'reason': str(exc)}, indent=2))
            return 0
    removed = float(np.sum(old.water_m - result.native_state.water_m)) * operator.column.cell_area_m2
    delivered = float(np.sum(result.delivered_m3))
    print(json.dumps({'rejected': False, 'native_calls': call.call_count,
                      'reported_closed': result.conservation['closed'],
                      'actual_water_removed_m3': removed, 'delivered_to_plants_m3': delivered,
                      'mass_conservation_failure': removed != delivered}, indent=2))
    return 1 if removed != delivered else 0


if __name__ == '__main__':
    raise SystemExit(main())
