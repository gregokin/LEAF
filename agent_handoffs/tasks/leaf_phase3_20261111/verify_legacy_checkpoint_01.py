"""Independent refusal of actual accepted Phase1/Phase2 checkpoints without editing them."""
import hashlib
import json
from pathlib import Path
from leaf import checkpoint, traits

def identities(root):
    return {a.name:hashlib.sha256(a.read_bytes()).hexdigest() for a in root.iterdir() if a.is_file()}

def main():
    catalog=traits.SpeciesCatalog(json.loads(Path('examples/phase2/species.json').read_text()))
    reports=[]
    for name in ['leaf-phase1-final-example','leaf-phase2-example05']:
        root=Path('/tmp')/name/'checkpoint'
        before=identities(root)
        original=json.loads((root/'restart.json').read_text())
        assert original['schema_version'] in (1,2)
        for override in [False,True]:
            try:
                checkpoint.load_session(root,catalog,operator=None,backend='numba',max_edges=1000,
                                        max_bytes=100_000_000,allow_source_change=override)
            except checkpoint.CheckpointError as exc:
                message=str(exc)
                assert 'version' in message and '3' in message,(name,message)
                reports.append({'version':original['schema_version'],'allow_source_change':override,
                                'refused':True,'message':message})
            else:raise AssertionError('Unqualified checkpoint migration accepted')
        assert identities(root)==before,'legacy fixture edited'
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
