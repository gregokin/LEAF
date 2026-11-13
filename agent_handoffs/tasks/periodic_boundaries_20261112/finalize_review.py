"""Codex-only final evidence aggregation; refuses missing/failed checks or changed frozen files."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,shutil,subprocess
P=Path(__file__).resolve().parent;ROOT=P.parents[2];SNAP=P/'public_evidence/review_04'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads((P/'baseline.json').read_text());cfg=json.loads((P/'session.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==base['head']
control={'agent_handoffs/current_prompt.md','agent_handoffs/orchestration_state.md','agent_handoffs/codex_review.md','agent_handoffs/verification.md'}
for rel,rec in base['files'].items():
 if rel not in control:assert sha(ROOT/rel)==rec['sha256'],('unexpected baseline drift',rel)
manifest=json.loads((P/'review_04.manifest.json').read_text())
for rel,rec in manifest.items():assert sha(SNAP/rel)==rec['sha256'],('frozen tree changed',rel)
names=['cpu','cuda','aliases','mixed_endpoints','canopy_union','golden','cover_memory','recruitment_memory','geometry_numpy','geometry_numba','geometry_cuda','native_tests_cpu','native_tests_cuda','native_pipeline_cpu','native_pipeline_cuda','example','standalone_cpu_v2','standalone_cuda_v2']
checks={}
for name in names:
 stem='review04_'+name;ex=json.loads((P/f'{stem}.exit.json').read_text());assert ex['returncode']==0 and ex['child_reaped'],(name,ex)
 checks[name]={'command':f'{stem}.command.json','stdout':f'{stem}.stdout.log','stderr':f'{stem}.stderr.log','exit':f'{stem}.exit.json','elapsed_seconds':ex['elapsed_seconds']}
 if name in ('cpu','cuda','native_tests_cpu','native_tests_cuda'):
  s=(P/f'{stem}.stderr.log').read_text();m=re.search(r'Ran (\d+) tests? in ([\d.]+)s',s);assert m;sk=re.search(r'OK \(skipped=(\d+)\)',s)
  checks[name].update(tests=int(m[1]),skipped=int(sk[1]) if sk else 0)
  checks[name]['passed']=checks[name]['tests']-checks[name]['skipped']
source_files={str(p.relative_to(SNAP)):{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(SNAP.glob('leaf/*.py'))+sorted(SNAP.glob('benchmarks/*.py'))}
installed=[f for f in cfg['write_files'] if f!='periodic_report.md'];installed_files={f:{'sha256':sha(SNAP/f),'size_bytes':(SNAP/f).stat().st_size} for f in installed}
oldlock=ROOT/'dependencies/masy-daily-plant-interface.lock.json';old=json.loads(oldlock.read_text())
water={}
for be in ('cpu','cuda'):
 d=json.loads((P/f'public_evidence/review04_native_pipeline_{be}/qualification.json').read_text())
 assert d['source']==old['selected_syrup']['package_sha256'] and d['native_maple_package']==old['native_maple']['package_sha256']
 water[be]={'model_days':d['model_days'],'physical_seconds':d['physical_seconds'],'one_hydro_and_one_T_each_day':d['one_hydro_and_one_T_each_day'],'max_closure_residual_m3':max(abs(r['closure_residual_m3']) for r in d['rows']),'max_closure_bound_m3':max(r['closure_bound_m3'] for r in d['rows']),'growth_updates':d['final_counters']['growth_updates'],'rollback':d['late_geometry_rollback'],'pending_restart_without_water_replay':d['pending_after_debit_resume_without_water_replay'],'closed_restart':d['closed_boundary_restart'],'source_sha256':d['source'],'tracemalloc_peak_bytes':d['tracemalloc_peak_bytes']}
 if be=='cuda':water[be]['cuda_pool_live_peak_bytes']=d['cuda_pool_live_peak_bytes']
# No unqualified cross-backend restart promise: numerical biological outputs only.
import numpy as np
a=json.loads((P/'public_evidence/review04_native_pipeline_cpu/qualification.json').read_text());b=json.loads((P/'public_evidence/review04_native_pipeline_cuda/qualification.json').read_text())
parity={}
for k in ('final_dry_agb_kg','final_seasonal_dry_agb_kg','final_age_days'):
 np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-14);parity[k]=float(np.max(abs(np.array(a[k])-np.array(b[k]))))
casefiles={r:rec for r,rec in base['files'].items() if r.startswith('cases/')};locks={r:rec for r,rec in base['files'].items() if r.startswith('dependencies/')}
now=datetime.now(timezone.utc).isoformat()
q={'accepted_at_utc':now,'task':'periodic_boundaries_20261112','verdict':'accept','revision':4,'baseline_head':base['head'],'baseline_dirty':base['status_porcelain_z'],'scope_audits':[f'review_{i:02d}.scope.json' for i in range(1,5)],'frozen_manifest':'review_04.manifest.json','checks':checks,'installed_files':installed_files,'source_files':source_files,'native_water':water,'native_biological_cpu_cuda_max_abs_difference':parity,'cover_memory':json.loads((P/'review04_cover_memory.stdout.log').read_text()),'recruitment_memory':json.loads((P/'review04_recruitment_memory.stdout.log').read_text()),'preservation':{'case_files_byte_identical':len(casefiles),'old_lock_files_byte_identical':len(locks),'AGENTS_md_unchanged':True,'existing_tests_unchanged':True,'growth_uptake_laws_unchanged':True,'upstream_sources_written':False,'case_generation_or_model_runs':False,'scope_of_execution':'only small synthetic/interface fixtures, never the user trial; one extra synthetic canopy memory check uses1M bitmap midpoints'},'limits':['square cells for roots/PET','periodicity via Domain or JSON, not CLI dimension flags','source hashes changed; older restart requires explicit compatibility handling','MAPLE adapter still needs native boundary flags passed through its leaf_domain helper; no upstream file edited','tested mechanics; no ecological validation, transport acceptance or domain-scale performance extrapolation'],'preserved_failures':['review_01.md','review_02.md','review_03.md','standalone_harness_correction.json','orchestration_errors.json']}
(P/'qualification.json').write_text(json.dumps(q,indent=2)+'\n')
shutil.copyfile(SNAP/'periodic_report.md',P/'report_04.md')
(P/'preservation.json').write_text(json.dumps({'checked_at_utc':now,'case_files':casefiles,'historical_locks':locks,'AGENTS_md':base['files']['AGENTS.md'],'phase3_resumption':'phase3_preservation.md'},indent=2)+'\n')
(P/'final_review.md').write_text('''# Independent acceptance: periodic canopies and roots

**Accept revision4**, exact source in `review_04.manifest.json` and `qualification.json`. Claude is the sole implementation writer; Codex independently reviewed, executed, corrected and accepted. All four Claude launches were reaped; actual file access/write scope passed. Recovered tool errors and rejected earlier candidates are preserved, never counted as acceptance.

## Result

One immutable per-axis `Domain.periodic` drives full-domain generation, canopy exclusion/union cover, exact canopy PET support, unique layered root footprints and daughter wrapping. Species root fractions and the demand/allocation/growth laws are retained. Physical water still belongs solely to SYRUP. One ID/record per plant, exact upper aliases, mixed axes/origins, bounded sparse arrays, source-bound checkpoints and the plain CSV plus companion convention are preserved.

The shared exact circle–rectangle integrator also corrects a pre-existing zero-breakpoint tangency error (a circle tangent to all four cell sides previously returned the entire square). Random default reference outputs remain byte-identical; that degenerate numerical case intentionally changes to the analytical circle area.

## Executed on the completed frozen source

- CPU:730 tests,677 passed,53 skipped. CUDA:730 tests,722 passed,8 skipped. CPU-only environment skips CUDA; CUDA suite skips the unavailable-path test, two omitted case helpers and five native tests. The five actual native tests are separately configured and pass5/5 on EACH CPU/CUDA. No configured failure was skipped.
- Independent geometry:640 root cases,32 canopy PET cases and4 sampled covers on EACH NumPy/Numba/CUDA, including roots wider than the domain, unique cell-layer resources and opposite-seam water. 48 decimal upper-alias and32 mixed-boundary endpoint cases pass. 80 independent adaptive-quadrature canopy-cell union cases pass. Legacy seeded plant/root/canopy hashes match exactly.
- Native Phase1.5: two eight-day6x4x2-layer CPU/CUDA harnesses use the genuine source guards. Each has8 daily hydrology updates,8 accepted zero-time transpiration debits and691200 physical seconds; weekly growth, failure rollback and pending/closed restart without replay pass, with residual inside the native bound. Biological CPU/CUDA outputs agree under the recorded FP64 contract.
- Historical c76 standalone driver: periodic eight-day continuous vs day3 checkpoint/resume numerical state matches on CPU/CUDA; wrong boundary declarations are refused and the native source guard stays enabled. The first Codex harness indexed the restored calendar twice and was correctly refused; corrected harness and original failure remain recorded.
- Cover peak after mask-lifetime correction:6,246,595 traced bytes versus6,291,472 declared array bytes in the1M-midpoint synthetic fixture (plus32KiB allowed Python metadata tolerance). Before correction it used6,705,429. Caps unchanged. Synthetic25,079-daughter placement remains below its existing9.09MB conservative estimate and closes dry-biomass bookkeeping.
- The new small periodic CLI example generated576 plants in scratch output; no trial case was regenerated or advanced.

Commands, separate streams, reaped exits, actual source identities and numerical/memory scopes are in `qualification.json`. Cases, catalogs, existing tests, AGENTS.md and historical locks remain unchanged. Real native qualification selects published SYRUP a5b8b183/package82e9c011 with MAPLE9f466d2/packagefbc504bb; the dirty live MAPLE checkout is read-only handoff context, not test input. No full coupled transport or ecological validation is claimed.

## Handoff / remaining work

`developer_handoff.md` gives actual public calls and the current MAPLE adapter step: propagate GeometrySpec periodic flags through `leaf_domain`, remove the2m wrapper inset and inner-cover correction only when separately regenerating. MAPLE's canopy/shelter wrapping already works. Square-cell restriction and explicit JSON/Domain declaration remain. Old checkpoints require explicit source compatibility handling; no historical lock was rebound. Phase3 scientific work stays paused at `phase3_preservation.md`.

Codex may now install exact accepted source, run installed-tree checks and commit/push under the standing authorization. This acceptance does not authorize trial regeneration, calibration, upstream edits or new ecological laws.
''')
print(json.dumps({'verdict':'accept','revision':4,'installed_paths':installed,'checks_passed':len(checks),'case_files_preserved':len(casefiles),'old_locks_preserved':len(locks)},indent=2))
