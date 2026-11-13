"""Codex-only immutable candidate capture and Claude scope audit."""
from pathlib import Path
import hashlib,json,shutil,sys
TASK=Path(__file__).resolve().parent
cfg=json.loads((TASK/'session.json').read_text());candidate=Path(cfg['cwd']).resolve()
rev=int(sys.argv[1]);name=f'review_{rev:02d}'
exitrec=json.loads((TASK/f'exit_{rev:02d}.json').read_text())
assert exitrec['child_reaped'] and not exitrec['timed_out'] and exitrec['returncode']==0,exitrec
allowed=set(cfg['write_files']);reads=[];writes=[];errors=[];results=[]
for s in (TASK/f'claude_{rev:02d}.stdout.jsonl').read_text().splitlines():
 o=json.loads(s)
 if o.get('type')=='result':
  results.append({k:v for k,v in o.items() if k not in ('usage','modelUsage','permission_denials')})
  if o.get('is_error') or o.get('subtype')!='success':errors.append(o)
 for c in o.get('message',{}).get('content',[]):
  if c.get('type')=='tool_use':
   tool=c['name'];inp=c.get('input',{})
   if '__unparsedToolInput' in inp:
    errors.append({'type':'malformed_tool_input_not_executed','tool':tool,'input':inp})
    continue
   path=Path(inp.get('file_path','')).resolve()
   assert tool in ('Read','Edit','Write'),(tool,inp)
   if tool=='Read':
    okay=path.is_relative_to(candidate) or str(path) in cfg['read_roots'] or path==TASK/f'prompt_{rev:02d}.md'
    assert okay,('read outside scope',str(path));reads.append(str(path))
   else:
    assert path.is_relative_to(candidate),('write outside candidate',path)
    rel=str(path.relative_to(candidate));assert rel in allowed,('unapproved write',rel)
    writes.append(rel)
  if c.get('type')=='tool_result' and c.get('is_error'):errors.append(c)
assert len(results)==1,results
# Tool errors are preserved for classification even if author recovers; final result must still be success.
changed=[]; unexpected=[]
root=TASK.parents[2]
for p in candidate.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(candidate);base=root/rel
 if not base.is_file() or p.read_bytes()!=base.read_bytes():
  changed.append(str(rel))
  if str(rel) not in allowed:unexpected.append(str(rel))
assert not unexpected,unexpected
snap=TASK/'public_evidence'/name
shutil.copytree(candidate,snap)
manifest={str(p.relative_to(snap)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'n_bytes':p.stat().st_size} for p in sorted(snap.rglob('*')) if p.is_file()}
(TASK/f'{name}.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
out={'revision':rev,'exit':exitrec,'structured_result':results[0],'tool_error_events':errors,'read_calls':len(reads),'write_calls':len(writes),'written_files':sorted(set(writes)),'changed_files':sorted(changed),'scope_pass':True,'frozen_tree':str(snap),'n_files':len(manifest)}
(TASK/f'{name}.scope.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('structured_result','tool_error_events')},indent=2))
