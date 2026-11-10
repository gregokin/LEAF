"""Verify completed writer scope and freeze a stable review tree."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys

TASK = Path(__file__).resolve().parent


def identity(path):
    b=path.read_bytes()
    return {'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def main():
    rev=int(sys.argv[1]);candidate=TASK/'candidate'
    launch=json.loads((TASK/f'launch_{rev:02d}.json').read_text())
    exit_record=json.loads((TASK/f'exit_{rev:02d}.json').read_text())
    assert exit_record['returncode']==0 and exit_record['child_reaped']
    baseline=json.loads((TASK/'baseline.json').read_text())['candidate_initial']
    allowed=set(launch['write_files']);tools=[];results=[]
    for line in (TASK/f'claude_{rev:02d}.stdout.jsonl').read_text().splitlines():
        entry=json.loads(line)
        if entry.get('type')=='result':results.append(entry)
        for block in entry.get('message',{}).get('content',[]):
            if isinstance(block,dict) and block.get('type')=='tool_use':
                name=block['name'];arg=block.get('input',{});file=Path(arg.get('file_path',arg.get('path','')))
                assert name in ['Read','Edit','Write'],name
                if name in ['Edit','Write']:
                    assert file.is_relative_to(candidate) and str(file.relative_to(candidate)) in allowed,str(file)
                else:
                    assert file.is_relative_to(candidate) or file==TASK/f'prompt_{rev:02d}.md',str(file)
                tools.append({'tool':name,'file':str(file)})
    assert len(results)==1 and results[0].get('is_error') is False and results[0].get('subtype')=='success'
    manifest={str(p.relative_to(candidate)):identity(p) for p in candidate.rglob('*') if p.is_file()}
    changes=[rel for rel in set(manifest)|set(baseline) if manifest.get(rel)!=baseline.get(rel)]
    assert not set(changes)-allowed,changes
    frozen=TASK/f'revision_{rev:02d}';assert not frozen.exists()
    shutil.copytree(candidate,frozen)
    (TASK/f'revision_{rev:02d}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (TASK/f'scope_revision_{rev:02d}.json').write_text(json.dumps({
        'at_utc':datetime.now(timezone.utc).isoformat(),'writer_changes':sorted(changes),
        'scope_pass':True,'tools':tools,'structured_result':{
            'subtype':results[0]['subtype'],'is_error':results[0]['is_error'],
            'session_id':results[0].get('session_id')},'child_reaped':True},indent=2)+'\n')
    print(json.dumps({'pass':True,'frozen':str(frozen),'files':len(manifest),'writer_changes':sorted(changes)}))


if __name__=='__main__':
    main()
