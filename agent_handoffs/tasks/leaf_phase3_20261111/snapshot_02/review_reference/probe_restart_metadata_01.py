"""Independent strict reproduction-state metadata guard tests with valid recomputed file hashes."""
import json,shutil,tempfile
from pathlib import Path
from leaf import checkpoint
from probe_recruitment_01 import raw_catalog
from review_tools import make,step_day

session,cat,op,xp=make(raw_catalog(),[],backend='numba')
step_day(session,xp)
failures=[];reports=[]
with tempfile.TemporaryDirectory(prefix='leaf-phase3-metadata-') as temp:
    root=Path(temp);good=session.save(root/'good')
    for name,changes in {'counter_uint64_overflow':{'n_reproduction':2**64},
                         'counter_more_than_handled_years':{'n_reproduction':1},
                         'opportunity_before_start_year':{'last_opportunity_year':2022},
                         'invented_last_record':{'last_reproduction':{'year':2023}}}.items():
        target=root/name;shutil.copytree(good,target)
        manifest=json.loads((target/'restart.json').read_text());manifest['phase3'].update(changes)
        body={k:v for k,v in manifest.items() if k!='content_hash'}
        manifest['content_hash']=checkpoint._sha(checkpoint._canonical(body))
        (target/'restart.json').write_text(json.dumps(manifest,indent=2)+'\n')
        try:checkpoint.load_session(target,cat,operator=op,backend='numba',max_edges=100_000,max_bytes=150_000_000)
        except checkpoint.CheckpointError as exc:reports.append({'check':name,'refused':True,'message':str(exc)})
        except Exception as exc:failures.append({'check':name,'secondary_exception':type(exc).__name__,'message':str(exc)})
        else:failures.append({'check':name,'incorrectly_accepted':True})
print(json.dumps({'reports':reports,'failures':failures},indent=2))
raise SystemExit(int(bool(failures)))
