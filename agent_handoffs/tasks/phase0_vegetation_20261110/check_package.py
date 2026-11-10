"""Build/inspect a source distribution using existing tools; never install/fetch.

Neither model venv contains setuptools or wheel. The existing system Python
has setuptools 68.1.2 but no wheel. Exercise its real sdist backend, then test
the unpacked package using the NumPy-capable model interpreter. Wheel building
is reported unverified rather than changing the declared backend for this host.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tarfile


def main():
    source = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix='leaf-phase0-package-', dir='/tmp') as temp:
        work = Path(temp)/'source'; work.mkdir()
        for name in ('leaf', 'benchmarks', 'docs'):
            shutil.copytree(source/name, work/name, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(source/'pyproject.toml', work/'pyproject.toml')
        if (source/'README.md').is_file():
            shutil.copy2(source/'README.md', work/'README.md')
        dist = Path(temp)/'dist'; dist.mkdir()
        env = dict(os.environ); env.update(PYTHONPATH=str(work), CUDA_VISIBLE_DEVICES='')
        builder = '/usr/bin/python3'
        build = subprocess.run([builder, '-c',
            'import setuptools.build_meta; setuptools.build_meta.build_sdist('+repr(str(dist))+')'],
            cwd=work, env=env, capture_output=True, text=True, timeout=90)
        if build.returncode:
            print(build.stdout); print(build.stderr, file=sys.stderr)
            raise RuntimeError('Source-distribution build failed; no dependency installation was attempted')
        archives = list(dist.glob('*.tar.gz')); assert len(archives) == 1
        artifact = archives[0]; extracted = Path(temp)/'extracted'
        with tarfile.open(artifact) as archive:
            names = archive.getnames(); archive.extractall(extracted, filter='data')
        package_root = next(extracted.glob('*/leaf/state.py')).parent.parent
        assert (package_root/'leaf/traits.py').is_file()
        assert (package_root/'benchmarks/daily_water_growth.py').is_file()
        assert not any('/tasks/' in name or '/public_evidence/' in name for name in names)
        metadata_path = package_root/'PKG-INFO'
        metadata = metadata_path.read_text()
        assert 'Landscape Ecohydrology with Agent-based Flora' in metadata
        # Empty cwd and extracted artifact first: checkout imports cannot pass.
        cwd = Path(temp)/'empty'; cwd.mkdir()
        env['PYTHONPATH'] = str(package_root)
        code = '''import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'numba','cupy','maple_syrup','maple','benchmarks'}:
   raise AssertionError('unwanted core import: '+fullname)
sys.meta_path.insert(0,Block())
import leaf,leaf.state,leaf.traits,leaf.initialization,leaf.canopy,leaf.recording,leaf.events,leaf.runtime
from leaf.__main__ import build_parser
assert 'extracted' in leaf.__file__
print(json.dumps({'package':leaf.__file__,'core_imports':True,'cli_parser':bool(build_parser())}))
'''
        core = subprocess.run([sys.executable, '-c', code], cwd=cwd, env=env,
                              capture_output=True, text=True, timeout=30)
        assert core.returncode == 0, core.stderr
        legacy = subprocess.run([sys.executable, '-c',
            'from leaf import CoupledSimulation,CoupledPhaseError,stepwat; '
            'import benchmarks.daily_water_growth; '
            'assert "extracted" in benchmarks.daily_water_growth.__file__; print("legacy imports passed")'],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=30)
        assert legacy.returncode == 0, legacy.stderr
        print(json.dumps({'pass': True, 'source': str(source), 'builder': builder, 'build_kind': 'sdist',
                          'artifact': artifact.name, 'artifact_bytes': artifact.stat().st_size,
                          'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
                          'core': json.loads(core.stdout), 'legacy': legacy.stdout.strip(),
                          'installed_packages': False, 'fetched_dependencies': False,
                          'wheel_built': False,
                          'wheel_unverified_reason': 'Both model environments lack setuptools/wheel; existing system '
                                                     'setuptools can build sdists but wheel is not installed.'}, indent=2))


if __name__ == '__main__':
    main()
