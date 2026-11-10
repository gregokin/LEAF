"""Run unchanged legacy tests plus candidate Phase 0 tests against the frozen candidate."""
from pathlib import Path
import sys
import unittest


def main():
    candidate=Path(sys.argv[1]).resolve()
    legacy=Path('/home/okin/leaf/tests')
    suite=unittest.TestSuite()
    suite.addTests(unittest.TestLoader().discover(str(legacy),pattern='test_[cds]*.py',top_level_dir=str(legacy)))
    suite.addTests(unittest.TestLoader().discover(str(candidate/'tests'),pattern='test_phase0.py',top_level_dir=str(candidate/'tests')))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
