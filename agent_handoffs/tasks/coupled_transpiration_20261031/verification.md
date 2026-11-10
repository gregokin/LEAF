# Independent verification — coupled_transpiration_20261031

Verdict: **pass**, revision5. Codex executed checks independently of Claude's report,
then installed the exact qualified code bytes. Actual SYRUP branch:
`feature/subsurface-hydrology`, HEAD `deb60c29834a4aad0bf8cb4cfbb611fc28ebe090` plus
uncommitted sources. MAPLE HEAD `963b06ad1cf0f3e2c420a7d7fb50939e3bd43b7d`.
The [source lock](../../../../dependencies/syrup-subsurface.lock.json) records dirty runtime identities.

## Passing checks

| Evidence stem | Result |
|---|---|
| `native_baseline_cpu` | Original native suite153 passed/23 skipped. |
| `leaf_baseline_cpu` | Original47 benchmark tests OK/3 GPU skips. |
| `candidate_cpu_final` |232 passed/31 GPU-related skips, including original native and LEAF suites plus new tests. |
| `candidate_gpu_final` |261 passed/2 intentional unavailable-GPU-path skips; real GTX1080Ti on allocated GPU0. |
| `independent_cpu_final` | NumPy and Numba analytical uptake/biomass at areas1 and0.0625m², mixed masks, split4+5vs9, strict dtype and memory-order guards. |
| `independent_limits_final` | Native reserve and full-ledger admission before allocation; weekly refusal/retry retains accepted day/uptake and geometry, no duplicate uptake. |
| `installed_gpu_oracles` | Actual installed module origins plus all independent CUDA analytic/mask/area/biomass/split checks pass; no skip path. |
| `legacy_compatibility_04` | All initialized live arrays, root graphs, counters and full summaries bit-identical to preserved default over23days/three weeks; benchmark code unchanged from revision4 to installed revision5. |
| `usage_example` | Exact documented example executes against installed native source. |
| `final_source_identity` |461 runtime file hashes and actual branch/HEADs match the lock. |
| `final_audit.json` |95 protected originals unchanged; all Claude tool scopes valid; code equals reviewed candidate; all child processes reaped. |

Each run has `<stem>.launch.json` (exact command/env/cwd, PID/start identity, timestamp,
source hashes), separate `.stdout.log`/`.stderr.log`, and `.exit.json` (status/reaping/time).
No acceptance result is inferred from an author claim. CPU/GPU test invocations were:

```
python -B -m pytest -q -p no:cacheprovider -c /dev/null \
  tests native/test_transpiration.py /home/okin/SYRUP/tests/subsurface
```

The real existing `/home/okin/MAPLE/.venv-cupy/bin/python` was used. Candidate CWD and
`PYTHONPATH=runtime:candidate:/home/okin/SYRUP/src` are expanded absolutely in receipts.
CPU hides CUDA; GPU selects UUID `GPU-5c8bd3d4-2885-82c6-a0f0-ad2b7f87b871` only.
Task-local Numba/CuPy/CUDA caches, bytecode disabled, CPU thread counts1. Runtime overlay
symlinked every original native module; it did not copy or replace hydrology.
The two real GPU skips are tests of deliberately unavailable GPU paths, not missing
implementation coverage. The only suite warning is the intentional ndarray-subclass
rejection fixture using deprecated `np.matrix`.

## Coupled capacity measurements

`performance_cpu128.json` and five `performance_gpu_*.json` artifacts each contain365
daily steps/52weekly updates, explicit native intake/E/drainage/T and actual-uptake-driven
biomass, all audits passing. GPU populations3717,36305,36305,59483,547588 were retained.
CPU128² took1.404s; GPU512²/3layers/59483plants took4.207s, measured live peak135.38MiB,
sampled reserved pool439.7MiB. Dense hectare547588plants:3.155s,224.09MiB live peak.
All GPU hooks ended with0live bytes/0unknown frees. Cap1.5GB was never raised.

These are one-year fixture observations, not a measured30-year coupled run. First-use
compilation and periodic reports are included; GPU0 carried another job at18–22% before/
after samples. Startup/setup/RSS/pool/live/estimated peaks are distinct in the artifacts
and [guide](../../../../docs/coupled-transpiration.md). No claim of clean steady-state scaling,
full MASY throughput or calibrated ecological response follows from these timings.

## Review corrections and failed attempts

| Attempt | Classification and disposition |
|---|---|
| Claude revision1 |`tool_error` / `interrupted`: single-slash absolute permission matcher denied candidate writes. Identity-checked termination/reap143; no candidate edits. Double-slash absolute rules fixed matching for the SAME paths; no bypass/broader scope. |
| revision2 CPU |`check_failed`:66 passed/5 skipped/2 analytic fixture failures (nonzero diffusion). Native/constructor/gather reproductions independently exposed masked storage, admission-order, negative reserve and metadata defects; corrected in revision3. |
| revision3 CPU |`check_failed`:73 passed/7 skipped/3 new fixture failures; independent NumPy/Numba oracles already pass. Weekly zero-availability bypass reproduced; device and scratch gaps reviewed, corrected in revision4. |
| revision3 GPU |`check_failed`:4 failed/5 passed/1 skip; NVRTC bare INFINITY undefined. First diagnostic math.h hypothesis failed (header unavailable); intrinsic substitution in process memory passed9/1, explicitly diagnostic only. Actual source fixed and fully requalified in revision5. |
| revision4 CPU |`check_failed`:76 passed/7 skipped/1 wet-bottom drainage fixture failure. Two reserve/peak gaps independently reproduced. All corrected in revision5. |
| `legacy_compatibility_03` |`tool_error` in independent harness: pre-task API has no live_names method. Preserved script/log; v2 uses its existing _LIVE_NAMES after initialized weekly buffers. `legacy_compatibility_04` passes unchanged checks. |
| first final-audit harness |`tool_error`: some Claude events contain string messages; parser expected dict. Preserved original and error note; `final_audit_v2.py` handles event forms and passes every scope/source/result/process check. |

No usage/auth/network limit or automatic approval rejection occurred in this task.
The historical bootstrap rejection was not retried with its private payload. Claude
received only original generic benchmark files and numeric API facts; no upstream
source, scientific brief or data. Full launch commands/session UUID and split streams
are archived for all five revisions. A read-only independent agent reviewed native and
coupled contracts; its final disposition is in `independent_design_review_final.md`.

## Installation and scope

`install_native.json` records only NEW SYRUP source/test files, created with exclusive
new-file opens after target-absence, branch/HEAD, original-hash and frozen-candidate checks.
`install_leaf.json` records the reviewed LEAF module/benchmark/test and author guide.
Codex assembled final documentation and the source lock afterward; runtime code is unchanged.
No original upstream file or dirty change was overwritten. Root AGENTS.md hash remains
`90a8b6217553480283efd0dcba514306c5472809b35b1d5a0e80e3009e48ef31`.
No Git commit/push/branch change, package installation or new environment occurred.

The qualified scope is daily interevent column/plant coupling, not storm/pond/MASY
scheduling, shared-surface texture/depth remapping, restart serialization or calibrated
ecological laws. The underlying benchmark's WUE/turnover/allometry remain fixtures.
