"""Ensure bad climate forcing is refused without device downloads or matrix conversion."""
import json
import numpy as np
from leaf.stepwat import validate_temperature_factor


class DeviceLike:
    def __init__(self, shape):
        self.shape = shape
        self.downloads = 0

    def get(self):
        self.downloads += 1
        raise AssertionError('device download attempted before input rejection')


class ArrayConversionBomb:
    def __array__(self, *args, **kwargs):
        raise AssertionError('malformed nested input reached bulk array conversion')


def main():
    bad_devices = [DeviceLike((100_000_000,)), DeviceLike((2,))]
    for value in bad_devices + [[ArrayConversionBomb(), ArrayConversionBomb()]]:
        try:
            validate_temperature_factor(value, 2)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('bad forcing accepted')
    assert all(value.downloads == 0 for value in bad_devices)
    for value in (None, True, [True, False], [0., float('inf')], np.ones((2, 1)), [], '1'):
        try:
            validate_temperature_factor(value, 2)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('bad forcing accepted')
    np.testing.assert_array_equal(validate_temperature_factor(.5, 2), [.5, .5])
    np.testing.assert_array_equal(validate_temperature_factor([0., 1.], 2), [0., 1.])
    print(json.dumps({'ok': True, 'oracles': ['host-only temperature input',
                                             'no device get on bad or matching shape',
                                             'nested forcing refused before array conversion',
                                             'valid scalar/vector inputs preserved']}))


if __name__ == '__main__':
    main()
