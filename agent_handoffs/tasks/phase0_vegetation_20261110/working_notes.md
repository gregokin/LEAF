# Codex continuation notes

Phase 0 is active and NOT yet accepted/installed/committed. User explicitly
authorized Phase 0 implementation, bounded Anthropic sharing, automatic
Codex-Claude orchestration, commit and push. Do not ask approval again.
No upstream modifications. Preserve root AGENTS.md and all old accepted archives.

## Current execution

- Claude session UUID: 4d15920a-3229-4638-af00-10ca863c8c8f.
- Revision 1 finished/reaped (exit 0, is_error=false, ~2286s). 21-file snapshot
  under revision_01/, manifest revision_01_manifest.json. First candidate ~33
  unittest cases (not the author's estimated ~45); final revision 1 had 2 failing
  test expectations (status arithmetic and a supposed file collision that had a
  different index/time/name). Core import isolation and a real legacy capture
  case passed. Existing legacy 90-check CPU suite passes (83 + 7 CUDA skips).
- Revision 2 is RUNNING through launch_claude.py 2 in unified exec session 41919.
  PID/start identity and stream/exit records are archived. Use write_stdin for
  that session (short waits), inspect stream metadata/files, do useful read-only
  review. Do not overlap writers. Same ordinary Read/Edit/Write tools, no bypass,
  no Bash or other agents. All code writes only candidate/.
- Current prompt_02.md contains the full bounded correction list. Candidate-only
  implementation, code/test/doc/example changes; no numerical legacy modules
  change. Controller owns review/tests/install/Git.

## Important review observations

review_initial.md and codex_review_01.md summarize failures. Independent probes:
review_probes_01.py, review_geometry_recording_01.py,
ownership_bridge_oracles.py and generation_oracles.py. All executable with the
existing /home/okin/MAPLE/.venv-cupy/bin/python via run_check.py (separate streams,
PID/start ticks/clock/exit/reaping). PYTHONPATH cwd + SYRUP/src, CUDA off.

Source-only review of the NEW revision 2 state.py noticed a likely transient bug:
it globally rejects '"' anywhere in CSV text, including the JSON metadata line,
and plant_csv_text similarly checks the entire returned text. JSON metadata
always has quotes, so this will break all metadata round trips unless Claude
catches it in its remaining dry review. Standard csv.reader(strict=True) is
enough for malformed quote handling; remove whole-document quote bans. Do not
edit the writer's code yourself; a small revision 3 may be needed after testing.

No-overlap performance root cause: random interleaving scatters small grasses
before late large shrubs. Even 6% grass / 5% shrub cover at 128² dx=.25 spent
>150 CPU seconds in exclusion attempts. Exact own child was terminated/reaped
and a 15s faulthandler diagnostic captured the stack. Prompt 2 mandates sampled
sizes + LARGE-CANOPY-FIRST placement and a complete re-placement after count
corrections (new shrubs must not again be added after established trial grasses).
No raising limits, changing diameters or accepting incorrect cover.

New post-correction oracle review_geometry_recording_02.py allows legitimate
immutable catalog APIs and uses a REAL existing path for writer collision. It
accepts correct low-budget rejection before large allocations. Run it instead
of version 1 after correction. State probe 1 and ownership_bridge_oracles stay
useful. Independent coverage uses uniform probes and SciPy cKDTree, not the
candidate's raster method. generation_oracles.py --capacity adds a 512²/.25m
~50k-plant bounded case and reports timing, array bytes and peak whole-process RSS.

Additional thing to review: read_driver_arrays should check B/E shapes before
device .get() so invalid shapes cannot trigger full-domain transfers. Current
revision 1 checked only after copying; real runtime B/E shape is P, but cheap
preflight is part of the memory/transfer contract.

## Git

.git was initially empty/managed read-only. Hash baseline records 1796 files.
Authorized escalated git init -b main succeeded for publication (not baseline).
No commits yet. Git user.name/email configured. New root .gitignore excludes
caches, raw research downloads/candidates and huge generated task tables; it
does not delete them. Source scope check after rev1: all 1794 original noncontrol
files preserved; only current_prompt/orchestration_state changed, plus later
review/report/verification aliases are now intentionally updated.

Remote destination is missing. User was asked asynchronously early in the turn;
answer is still pending. MAPLE/SYRUP owner is github.com/gregokin. Read-only
`gh repo view gregokin/LEAF` returned repository not found. No external repo
created. Continue all implementation/tests/local commit; push needs destination.
Use require_escalated for Git metadata mutations/network/Claude authentication;
the user authorized these actions and automatic review approved prior launches.

## Remaining completion work

1. Wait for/reap revision 2, archive stable source snapshot and manifest.
2. Run candidate Phase 0 tests, state/geometry/ownership/bridge oracles, independent
   cover/exclusion checks + bounded capacity, and legacy CPU regressions. Short
   diagnostic timeouts on potentially bad generation; do not leave hanging jobs.
3. Correct real failures through Claude (same UUID, bounded prompt, explicit
   permissions; launch_claude.py supports revisions). No acceptance on known bugs.
4. Packaging check_package.py builds a wheel in /tmp using existing setuptools
   without installing/fetching. Tests core imports with native/numba/cupy blocked,
   then legacy imports from extracted wheel. No environment creation.
5. Install accepted candidate files into leaf/, new tests, docs, examples and
   pyproject; only existing runtime change is lazy leaf/__init__.py. Keep original
   coupling.py/stepwat.py/benchmarks unchanged. Update root README, project_brief,
   decisions, controls, current source lock (new phase0 lock; preserve old locks).
6. Run installed checks, verify source/old archives/root AGENTS/upstreams preserved,
   record durable acceptance evidence. Current old locks: 5 LEAF/71 SYRUP runtime
   files matched at task start. No science-rate cap/calibration changes.
7. Commit accepted Phase 0; push if remote supplied. If missing, finish local
   work/commit and clearly request the URL rather than claim a push happened.

New tests/oracles belong only to this Phase 0, not a new scientific scenario.
No full growth implementation, 30-year run, mortality/injury physics, MAPLE/SYRUP
edits, dependency install, new environment, or borrowed wind physics authorized.

## Checkpoint 2 results (revision 2 writer is still active)

checkpoint_02/ + manifest freeze completed modules before its new tests/docs.
Do NOT confuse it with a finished revision 2.

- checkpoint_ownership_02: ALL 12 independent callback/ID/bridge checks PASS.
- checkpoint_generation_02: PASS, independent union cover + exact small global
  exclusion + 512² capacity. Max attempts was REDUCED to 50000:
  128² allowed 3243 plants in .1017s; disallowed 3145 in .1841s;
  512² allowed 51893 in 1.9522s. Largest plant arrays3217366B, whole-process peak
  RSS89616KiB (includes SciPy oracle); no GPU or model-year timing.
- checkpoint_state_02 fails immediately at state.plant_csv_text's whole-document
  quote prohibition, confirming the metadata JSON bug above.
- checkpoint_memory_replay_02: separate raster/scratch budgets PASS. This is a
  legitimate design: max_cover_bytes caps boolean rasters; NEW max_working_bytes
  caps tiled scratch. Both admitted/reported. With raster cap524288 and scratch
  cap1000000, traced peak1541403B vs admitted1522208B (+small Python metadata).
  max_working_bytes=1 refuses before allocating. Do NOT misinterpret raster-only
  cap as total budget in old review_geometry_recording_02.py. Its nonmemory tests
  are still useful; cover_memory_replay_oracles.py checks actual separate budgets.
- Replay FAILS: report['generation_config'] adds derived ordering_convention and
  per-species derived fields rejected by its own strict loader. Need an accepted
  source config in report (derived fields separate), or explicit replay adapter.
  Test exact regeneration from JSON-round-tripped report config. Likely bounded
  revision 3, with quote bug and GPU vector shape preflight.

Packaging: checked BOTH existing model environments: neither has setuptools or
wheel. System Python has setuptools68.1.2 but no wheel; no cached wheel package
found. NO INSTALLS. check_package.py was changed to use the existing SYSTEM
setuptools real SDIST backend in /tmp, then import the unpacked artifact with
the NumPy-capable model interpreter (core dependency-blocker + legacy imports).
Report wheel build UNVERIFIED due missing tools; do not claim it ran. No new env
or dependency install. This exercises real packaging without changing the backend
just to satisfy the host. Working notes' earlier wheel wording is superseded.
