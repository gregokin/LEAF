# Independent verification — STEP-style growth, revision3

Final result: installed-tree checks pass. This qualifies the bounded implementation
and accounting, not a calibrated ecosystem model or exact STEPWAT2 reproduction.

| Check | Installed result | Evidence |
|---|---|---|
| Full CPU suite |90tests:83passed,7CUDA-dependent skips |installed_cpu_final.* |
| Full CUDA-enabled suite |90tests:89passed,1intentional unavailable-GPU-path skip |installed_gpu_final.* |
| Independent scalar kernels |7cases each, Numba/CUDA: wet, limited, zero water, cold, dormant, mature cap, year-end |installed_oracles_cpu/gpu.* |
| Independent native coupling |Two cell areas, separate E/T, actual-uptake-to-biomass, day365 partial period, failed year-end then single retry, next partial week |same oracle streams |
| Forcing validation |Host-only; no device download or malformed-matrix conversion; invalid forcing rejected before physical mutation |same oracle streams |
| Source/write scope |No violations; only assigned candidate reads/writes; original tests and geometry preserved |scope_installed_final.json |
| Source/docs |Root AGENTS unchanged;5LEAF/71SYRUP runtime identities,386current MAPLE sources,88local links checked |source_and_document_checks_final.json |

Exact commands, working directories, environment, source hashes, PID/start identity
and reaped exits are in each *.launch.json/*.exit.json; raw stdout and stderr are
separate. All real CUDA work selected GPU0 UUID
GPU-5c8bd3d4-2885-82c6-a0f0-ad2b7f87b871. GPU1 was not used. Source lock:
../../../dependencies/stepwat-growth.lock.json. Existing environment versions are
in environment_final.json. No packages or new environment were created.

The analytical scripts and expected values were independently written by Codex.
They verify actual installed imports, not an uninstalled overlay. Complete
candidate Python files are byte-identical to the installed files. Codex corrected
publication-only documentation commands after Claude exited; see
publication_documentation_changes.json. No numerical source was edited by Codex.

## Review/correction history and errors

- Initial restricted launch: repeated structured api_retry/error unknown; managed
  execution interrupted (tool exit130). Specific connection cause unconfirmed,
  child reaping not claimed. Exact-session host preflight later found no original
  writer remaining. Classification: interrupted.
- First network escalation rejected before launch for external project-data
  transfer. User explicitly authorized the bounded transfer and reconfirmed
  automatic approval; actual subsequent dispatches were approved. Historical
  rejection retained in approval_rejection_02.json; no workaround was used.
- Revision2 delivered the six candidate files. A nonessential allowed-directory
  Read found docs/coupling.md absent (tool_error). The writer's unfinished report
  was identity-verified and interrupted for concrete review feedback; launcher
  reaped it (exit143). Partial work and raw streams preserved.
- Snapshot CPU:82passed,7skipped,1test-fixture failure. The fixture patched an
  already-bound callback and missed construction ledgers. Independent kernels and
  water coupling already passed. A separate oracle reproduced the unbounded
  temperature-input conversion (forcing_reproduction_02.*). Classification:
  check_failed; no failed candidate was accepted.
- Revision3 resumed the exact session, corrected host-only validation, unit labels
  and the memory test without changing equations/limits. Structured success,
  exit0/reaped. Final CPU/CUDA and independent checks passed.
- An import-only diagnostic without a writable Numba cache raised “cannot cache
  function '_fused_numba': no locator available”. Explicit NUMBA_CACHE_DIR under
  /tmp resolved this environmental issue without modifying upstream sources.

## Bounds and remaining science

Added FP64 array storage is8*(2P+23S) compute bytes plus8*19S GPU host-trait
mirror bytes. For547,588individuals and2profiles this is8.36MiB additional compute
arrays, not full model peak memory. Candidate buffers, native water and old+new
root graphs remain in the admission ledgers; no caps were raised or plants
removed. Memory guard tests exercise init, full rebuild peaks, refusal and retry.
No new RSS/allocator/scaling or30-year performance measurement is claimed.

Weekly Euler timing, proportional spatial competition, explicit canopy-demand
traits and supplied temperature modifiers are documented LEAF adaptations.
Species parameters require calibration. Demography, event injury, moving-surface
remapping, restart serialization and full MASY storm scheduling remain future
scope. Small automated integration fixtures were run; no new field scenario or
long-horizon benchmark was run. No upstream source edits, commits or pushes.
