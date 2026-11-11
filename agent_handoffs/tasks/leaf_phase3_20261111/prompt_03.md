# Phase3 revision3 — correct omitted rejection-work peak record

Same task/session a5c6d72f-e529-49ec-beea-b376caaa8814; R2 completed/reaped,
scope/tool errors0. Automatic user-authorized source-sharing/correction loop.
Sole implementation writer, ordinary Read/Edit/Write only; no execution/Bash/agents/
network/settings/Git/install/outside reads. Read current prompt and candidate/**.
Write ONLY leaf/interevent.py, tests/test_interevent.py, docs/phase3-reproduction.md,
report.md. ALL numerical modules, traits, fixtures and other tests stay read-only.
Author checks NOT RUN. Preserve all source guards, limits, native state and physics.

R2 final CPU525 tests (44skips) and actual CUDA525 (1skip) pass. Independent native
partial-year metrics/overrides/newborn roots/restart, strict metadata, guard probes,
old-format refusals, newborn hazard and season flush checks all pass. Compiled CPU
example passes. Actual successful-birth CUDA memory profile:100000records/50000live,
50000daughters+64seedlings,575k edges, birth0.514s, declared138461632B; nextday actual
array lower34565976B<=declared36395608B. Caps unchanged.

One remaining measured memory-reporting defect:
review_reference/snapshot02_reject_many_memory.stdout.log and profile_recruitment_02.py
show100000records/25000live,62403 requested daughters, ALL rejected,0births.
The pure evaluate estimator correctly admits its larger candidate work, BUT the
session's memory receipt records ONLY reproduction_preflight (trial stage), since
no `_rebind` occurs. Simultaneous NumPy allocation-domain bytes + resident host arrays
+ current hooked CUDA arrays reach24018366B, exceeding declared20432608B by3585758B.
The pure estimate itself is28398624B, so byte cap enforcement was not bypassed; the
missing session stage makes the published peak false. No Python-object overhead is
included in that measured array lower bound. Hook observed zero unknown frees/end live.

Fix minimally: record the FULL evaluate stage estimate in session's `mem` on every
accepted call, including n_new==0, using the exact same global reserve + gather + full
pure scratch estimate already admitted by evaluate. e.g. a reproduction_evaluate
_check_stage after the pure result (before publish/rebind). Preserve all existing
pre-allocation admissions inside evaluate; do not just inflate a constant, raise a cap,
truncate work/state or change science/parameters. The memory peak must cover the same
actual work when there are successful births AND when all placements fail.

Add a focused regression with requested candidates but zero births, showing evaluate
stage is present and global peak includes its full admitted estimate; valid canonical
state/counters/AGB/E/native water unchanged except the accepted opportunity. Update
memory documentation/report, honestly NOT RUN. Codex will repeat actual rejection
profile, successful profile and final CPU/CUDA checks. Leave uncommitted.
