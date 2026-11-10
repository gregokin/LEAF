"""Install only independently qualified exact bytes; never replace existing upstream files."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
TASK=Path(__file__).resolve().parent;ROOT=TASK.parents[2]
mode=sys.argv[1];assert mode in ('native','leaf')
assert not (TASK/f'install_{mode}.json').exists()
base=json.loads((TASK/'baseline.json').read_text());cand=TASK/'candidate'
manifest=json.loads((TASK/'candidate_after_05.json').read_text())
for rel,v in manifest.items():assert hashlib.sha256((cand/rel).read_bytes()).hexdigest()==v['sha256'],rel
for d in ['/home/okin/SYRUP','/home/okin/MAPLE']:
 for cmd,key in [(['git','rev-parse','HEAD'],'head'),(['git','branch','--show-current'],'branch')]:
  assert subprocess.check_output(cmd,cwd=d,text=True)==base['repositories'][d][key]['stdout'],(d,key)
for name,v in base['files'].items():
 if name.startswith(('/home/okin/SYRUP/','/home/okin/MAPLE/')):
  assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==v['sha256'],name
if mode=='native':
 pairs=[('native/maple_syrup/subsurface_transpiration.py',Path('/home/okin/SYRUP/src/maple_syrup/subsurface_transpiration.py')),
        ('native/test_transpiration.py',Path('/home/okin/SYRUP/tests/subsurface/test_transpiration.py'))]
 for rel,dest in pairs:assert not dest.exists(),f'preserve existing {dest}'
else:
 pairs=[(rel,ROOT/rel) for rel in ['benchmarks/daily_water_growth.py','leaf/__init__.py','leaf/coupling.py','tests/test_coupling.py','docs/coupled-transpiration.md']]
 for rel,dest in pairs:
  if str(dest) in base['files']:assert hashlib.sha256(dest.read_bytes()).hexdigest()==base['files'][str(dest)]['sha256'],dest
  else:assert not dest.exists(),dest
written=[]
for rel,dest in pairs:
 dest.parent.mkdir(parents=True,exist_ok=True)
 data=(cand/rel).read_bytes()
 with dest.open('xb' if mode=='native' or not dest.exists() else 'wb') as f:
  f.write(data);f.flush();os.fsync(f.fileno())
 assert dest.read_bytes()==data
 written.append({'source':str(cand/rel),'target':str(dest),'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
record={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':mode,'qualified_revision':5,'files':written,'existing_upstream_files_modified':False}
(TASK/f'install_{mode}.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
