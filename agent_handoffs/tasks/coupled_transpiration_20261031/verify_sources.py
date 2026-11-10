"""Read-only verification of the qualified local source lock; no auto-update or fallback."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[3];lockpath=root/'dependencies/syrup-subsurface.lock.json'
lock=json.loads(lockpath.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checked=0
for key in ['leaf','syrup']:
 r=lock[key];base=Path(r['root'])
 for rel,v in r['runtime_files'].items():
  assert sha(base/rel)==v['sha256'],str(base/rel);checked+=1
for key in ['syrup','maple']:
 r=lock[key]
 for cmd,field in [(['git','rev-parse','HEAD'],'head'),(['git','branch','--show-current'],'branch')]:
  assert subprocess.check_output(cmd,cwd=r['root'],text=True).strip()==r[field],(key,field)
r=lock['maple'];m=(lockpath.parent/r['runtime_manifest']).resolve();assert sha(m)==r['runtime_manifest_sha256']
for rel,v in json.loads(m.read_text())['files'].items():
 assert sha(Path(r['root'])/rel)==v,rel;checked+=1
print(json.dumps({'status':'pass','runtime_files_checked':checked,'syrup_branch':lock['syrup']['branch'],'exact_source_identity':True}))
