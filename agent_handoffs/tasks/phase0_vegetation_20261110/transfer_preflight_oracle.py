"""A malformed device vector must fail on metadata before any host transfer."""
import json
from types import SimpleNamespace
import numpy as np
from leaf.runtime import read_driver_arrays


class DeviceVector:
    def __init__(self, shape, *, value=0.01):
        self.shape = shape
        self.dtype = np.dtype('float64')
        self.calls = 0
        self.value = value

    def get(self):
        self.calls += 1
        if self.shape != (1,):
            raise AssertionError('Unexpected large device-to-host transfer')
        return np.array([self.value])


def driver(B, E=None):
    return SimpleNamespace(host_inputs={'ids': np.array([17], np.int64)}, B=B,
        host_profile_ids=np.array([0], np.int32), profile_names=['BOER'],
        stepwat=None if E is None else SimpleNamespace(E=E), day=0)


def rejected_before_transfer(sim, bad):
    try:
        read_driver_arrays(sim)
    except (ValueError, TypeError, AttributeError):
        return bad.calls == 0
    except AssertionError:
        return False
    return False


def main():
    badB = DeviceVector((3, 2_000_000))
    badE = DeviceVector((3, 2_000_000))
    out = {
        'bad_B_shape_before_get': rejected_before_transfer(driver(badB), badB),
        'bad_E_shape_before_get': rejected_before_transfer(driver(np.array([0.02]), badE), badE),
    }
    B, E = DeviceVector((1,), value=0.02), DeviceVector((1,), value=0.001)
    actual = read_driver_arrays(driver(B, E))
    out['valid_vectors_transfer_once'] = bool(B.calls == 1 and E.calls == 1 and actual['B'][0] == 0.02 and actual['E'][0] == 0.001)
    print(json.dumps(out, indent=2))
    return 0 if all(out.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
