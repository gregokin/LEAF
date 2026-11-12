"""Audit archived Claude tool calls against the bounded candidate scope."""
from pathlib import Path
import json

TASK = Path(__file__).resolve().parent
cfg = json.loads((TASK/'session.json').read_text())
root = Path(cfg['cwd']).resolve()
write_paths = {str((root/p).resolve()) for p in cfg['write_files']}
violations=[]; counts={}; results=[]
for log in sorted(TASK.glob('claude_*.stdout.jsonl')):
    for line in log.read_text().splitlines():
        obj=json.loads(line)
        if obj.get('type')=='result':
            results.append(dict(file=log.name,subtype=obj.get('subtype'),is_error=obj.get('is_error'),result=str(obj.get('result',''))[:1500]))
        if obj.get('type')!='assistant': continue
        for block in obj.get('message',{}).get('content',[]):
            if block.get('type')!='tool_use': continue
            name=block.get('name'); data=block.get('input',{}); path=data.get('file_path')
            counts[name]=counts.get(name,0)+1
            if name not in ('Read','Edit','Write'):
                violations.append(dict(tool=name,reason='tool outside allowed set')); continue
            if not path:
                violations.append(dict(tool=name,reason='no file path')); continue
            target=Path(path)
            if not target.is_absolute(): target=root/target
            target=target.resolve()
            if name=='Read':
                if not target.is_relative_to(root): violations.append(dict(tool=name,path=str(target),reason='read outside candidate'))
            elif str(target) not in write_paths:
                violations.append(dict(tool=name,path=str(target),reason='write outside exact list'))
report=dict(tool_counts=counts,violations=violations,results=results,pass_scope=not violations)
(TASK/'scope_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(1 if violations else 0)
