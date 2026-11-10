"""Run the independent acceptance oracles against installed LEAF sources only."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

from leaf import stepwat

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    assert Path(stepwat.__file__).resolve() == ROOT / 'leaf/stepwat.py', stepwat.__file__
    original = sys.argv[:]
    names = ['independent_kernels', 'independent_coupling', 'independent_validation']
    for name in names:
        path = TASK / (name + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.argv = [str(path)] + (['--gpu'] if args.gpu and name != 'independent_validation' else [])
        module.main()
    sys.argv = original
    print(json.dumps({'ok': True, 'installed_source': str(Path(stepwat.__file__).resolve()),
                      'backend': 'cuda' if args.gpu else 'cpu', 'independent_suites': names}))


if __name__ == '__main__':
    main()
