"""Exercise the installed user-facing Phase 0 commands in temporary output space."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile


def main():
    root = Path('/home/okin/leaf')
    species = str(root/'examples/phase0/species.json')
    with tempfile.TemporaryDirectory(prefix='leaf-phase0-cli-', dir='/tmp') as temp:
        out = Path(temp)
        commands = [
            ['validate-species', '--species', species],
            ['init', '--species', species, '--plants', str(root/'examples/phase0/plants.csv'),
             '--nx', '20', '--ny', '20', '--dx-m', '1', '--surface-constant', '100',
             '--out', str(out/'listed.csv')],
            ['generate', '--species', species, '--config', str(root/'examples/phase0/generation.json'),
             '--surface-constant', '100', '--out', str(out/'generated.csv'),
             '--report', str(out/'generation.json')],
            ['inspect', '--species', species, '--plants', str(out/'generated.csv'),
             '--surface-constant', '100', '--cover-subsamples', '4'],
            ['export-canopy', '--species', species, '--plants', str(out/'generated.csv'),
             '--surface-constant', '100', '--out', str(out/'view.json')],
        ]
        results = []
        for arguments in commands:
            run = subprocess.run([sys.executable, '-m', 'leaf', *arguments],
                                 cwd=root, env=dict(os.environ), capture_output=True, text=True, timeout=30)
            results.append({'command': arguments[0], 'returncode': run.returncode})
            if run.returncode:
                print(run.stdout); print(run.stderr, file=sys.stderr)
                raise AssertionError(arguments[0])
        from leaf.state import read_plant_csv
        from leaf.traits import SpeciesCatalog
        catalog = SpeciesCatalog.from_file(species)
        listed = read_plant_csv(out/'listed.csv', catalog)
        population = read_plant_csv(out/'generated.csv', catalog)
        assert listed.n == 6 and not listed.establishment_unresolved.any()
        view = json.loads((out/'view.json').read_text())
        assert view['meta']['native_maple_valid'] and view['meta']['n_objects'] == population.n
        print(json.dumps({'pass': True, 'commands': results, 'listed_plants': listed.n,
                          'generated_plants': population.n, 'temporary_outputs': True}, indent=2))


if __name__ == '__main__':
    main()
