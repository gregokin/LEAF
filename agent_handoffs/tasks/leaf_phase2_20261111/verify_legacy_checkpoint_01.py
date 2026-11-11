"""An actual Phase1 example checkpoint must fail clearly before a Phase2 restore."""
import hashlib
import json
from pathlib import Path

from leaf import checkpoint, traits
from leaf.interevent import IntereventSession
from verify_native_biology_02 import make


def main():
    path = Path('/tmp/leaf-phase1-final-example/checkpoint')
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['schema_version'] == 1
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
    session, _, operator, _ = make('numba')
    catalog = traits.SpeciesCatalog.from_file('examples/phase1/species.json')
    reports = []
    for permit_source_change in [False, True]:
        try:
            IntereventSession.load(path, catalog, operator=operator, backend='numba',
                                    max_edges=1_000_000, max_bytes=100_000_000,
                                    allow_source_change=permit_source_change)
        except checkpoint.CheckpointError as exc:
            message = str(exc)
            assert 'version 1' in message and 'version 2' in message, message
            reports.append({'allow_source_change': permit_source_change, 'old_version_refused': True,
                            'message': message})
        else:
            raise AssertionError('unqualified Phase1 format was silently migrated')
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
    assert before == after
    print(json.dumps({'actual_phase1_checkpoint_untouched': True, 'checks': reports}, indent=2))


if __name__ == '__main__':
    main()
