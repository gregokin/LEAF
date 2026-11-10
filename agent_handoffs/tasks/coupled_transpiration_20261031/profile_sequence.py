"""Sequential bounded GPU qualification/profiles; each child has split streams and receipts."""
from pathlib import Path
import json,subprocess,sys
p=Path(__file__).resolve().parent;py=sys.executable
q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
(p/'gpu_allocation_before.json').write_text(json.dumps({'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'selection':'allocated physical GPU0 only; GPU1 not used'},indent=2)+'\n')
runs=[('installed_gpu_oracles',[py,'-B',str(p/'installed_verification.py'),'gpu'])]
for case in json.loads((p/'performance_cases.json').read_text()):
 name='performance_gpu_'+case['name']
 runs.append((name,[py,'-B',str(p/'performance.py'),'--backend','cuda','--case',case['name'],'--days','365','--output',str(p/(name+'.json'))]))
for name,cmd in runs:
 subprocess.run([py,'-B',str(p/'run_process.py'),'--name',name,'--timeout','180','--cwd','/home/okin/leaf','--',*cmd],check=True)
q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
(p/'gpu_allocation_after.json').write_text(json.dumps({'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr},indent=2)+'\n')
print('all sequential GPU checks/profiles completed',flush=True)
