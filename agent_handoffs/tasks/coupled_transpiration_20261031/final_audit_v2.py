"""Independent final scope/source/result/process audit; read-only except its evidence JSON."""
from pathlib import Path
import hashlib,json,subprocess
T=Path(__file__).resolve().parent;R=T.parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((T/'baseline.json').read_text());checked=[]
for name,v in b['files'].items():
 if name.startswith(('/home/okin/SYRUP/','/home/okin/MAPLE/')) or name in [str(R/'AGENTS.md'),str(R/'claude.md'),str(R/'agent_handoffs/AGENTS.md'),str(R/'benchmarks/spatial_mapping.py'),str(R/'tests/test_daily_water_growth.py'),str(R/'tests/test_spatial_mapping.py')]:
  assert sha(Path(name))==v['sha256'],name;checked.append(name)
for n in ['benchmarks/daily_water_growth.py','leaf/coupling.py','leaf/__init__.py','tests/test_coupling.py']:
 assert (R/n).read_bytes()==(T/'candidate'/n).read_bytes(),n
for f in json.loads((T/'install_native.json').read_text())['files']:assert sha(Path(f['target']))==f['sha256']
allowed={'benchmarks/daily_water_growth.py','leaf/__init__.py','leaf/coupling.py','tests/test_coupling.py','native/maple_syrup/subsurface_transpiration.py','native/test_transpiration.py','docs/coupled-transpiration.md'}
counts={};candidate=(T/'candidate').resolve()
for rev in range(1,6):
 rows=[json.loads(s) for s in (T/f'claude_{rev:02d}.stdout.jsonl').read_text().splitlines()]
 for row in rows:
  message=row.get('message',{})
  content=message.get('content',[]) if isinstance(message,dict) else []
  if not isinstance(content,list):continue
  for block in content:
   if block.get('type')!='tool_use':continue
   name=block['name'];counts[name]=counts.get(name,0)+1;assert name in ('Read','Edit','Write')
   f=Path(block['input']['file_path']);f=f if f.is_absolute() else candidate/f
   rel=str(f.resolve().relative_to(candidate))
   if name in ('Edit','Write'):assert rel in allowed,(rev,rel)
 if rev>1:assert rows[-1]['type']=='result' and rows[-1].get('subtype')=='success' and rows[-1].get('is_error') is False
 end=json.loads((T/f'exit_{rev:02d}.json').read_text());assert end['child_reaped']
 if rev>1:assert end['returncode']==0
receipts=[]
for launch in T.glob('*.launch.json'):
 end=T/(launch.name.replace('.launch.json','.exit.json'));assert end.is_file(),launch.name
 r=json.loads(end.read_text());assert r['child_reaped'] and not r['timed_out'],end.name;receipts.append({'name':launch.name,'returncode':r['returncode']})
profiles=[]
for name in ['performance_cpu128.json']+['performance_gpu_'+c['name']+'.json' for c in json.loads((T/'performance_cases.json').read_text())]:
 r=json.loads((T/name).read_text());assert r['status']=='ok' and r['days']==365 and r['summary']['ok'] and all(r['summary']['checks'].values()),name
 c=r['counters'];assert c['day']==365 and c['weeks_done']==52 and c['maps_built']==53 and c['demand_map_builds']==53 and c['population']==r['case']['grass_count']+r['case']['shrub_count'] and c['recharge_events'] is None
 assert r['summary']['memory']['peak_required_bytes']<=r['case']['max_bytes']
 if r['backend']=='cuda':
  m=r['memory'];assert m['cupy_live_at_end_bytes']==m['unknown_free_events']==0
  assert m['cupy_live_rounded_peak_bytes']<=m['allocator_policy']['pool_limit_bytes']<=r['case']['max_bytes']
 for path,v in r['source_sha256'].items():assert sha(Path(path))==v,path
 profiles.append({'name':name,'status':'pass','days':365,'all_checks':True,'population':c['population']})
assert json.loads((T/'installed_gpu_oracles.exit.json').read_text())['returncode']==0
assert json.loads((T/'usage_example.exit.json').read_text())['returncode']==0
assert json.loads((T/'final_source_identity.exit.json').read_text())['returncode']==0
assert json.loads((T/'candidate_cpu_final.exit.json').read_text())['returncode']==0
assert json.loads((T/'candidate_gpu_final.exit.json').read_text())['returncode']==0
summary={'status':'pass','protected_original_files_checked':len(checked),'claude_tool_scope':counts,'all_claude_children_reaped':True,'verification_children':receipts,'profiles':profiles,'code_matches_reviewed_candidate':True,'original_upstreams_unchanged':True,'root_AGENTS_unchanged':True}
(T/'final_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ('verification_children','profiles')},indent=2))
