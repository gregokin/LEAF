# Phase 0 independent verification

Reviewer/executor: Codex. Date: 2026-11-10. Result: **pass, revision 3 installed**.

## Reproduction and primary records

Existing interpreter: `/home/okin/MAPLE/.venv-cupy/bin/python`. No installs/new
environment. Each check uses `run_check.py NAME ABSOLUTE_CWD COMMAND...`; its
`NAME.launch.json` records exact argv, cwd, PID/start ticks, timestamp and env.
Separate `NAME.stdout.log`, `NAME.stderr.log` and `NAME.exit.json` record results,
elapsed time and child reaping. No writer/test child is left running.

CPU environment: `PYTHONPATH=/home/okin/leaf:/home/okin/SYRUP/src`,
`NUMBA_CACHE_DIR=/tmp/leaf-phase0-numba`, `CUPY_CACHE_DIR=/tmp/leaf-phase0-cupy`,
`PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=''`.
Real CUDA checks select physical GPU 1 (`LEAF_PHASE0_GPU=1`, logical device 0).
Sandbox CUDA access failed with `cudaErrorInsufficientDriver`; the bounded
authorized checks outside that sandbox used the existing driver successfully.
No driver/package change was made. GPU 1 was idle when selected.

| Check / log prefix | Invocation or invariant | Result |
|---|---|---|
| `installed_cpu` | `python -m unittest discover -s tests -v` | 134 total; 127 pass, 7 CUDA skips; 2.577s unittest time |
| `installed_gpu` | Same command, physical GPU 1 | 134 total; 133 pass, 1 intentional skip; 4.014s unittest time |
| `revision03_phase0_cpu` | `python -m unittest discover -s tests -p test_phase0.py -v` on frozen revision | All 44 pass |
| `revision03_state` | `review_probes_01.py` | FP64 replay; anchor declaration; malformed CSV; overflow rollback pass |
| `revision03_geometry` | `review_geometry_recording_01.py` | Failed-write rollback; buried canopy safe default; catalog ownership pass |
| `revision03_ownership` | `ownership_bridge_oracles.py` | 12 callback/ownership/ID/profile/allometry/E guards pass |
| `revision03_memory_replay` | `cover_memory_replay_oracles.py` | Scratch admission, bounded temporary memory, saved-config exact replay pass |
| `revision03_transfer_retry` | `transfer_preflight_oracle.py` | Invalid B/E shape rejected before `.get()`; valid vectors transferred once |
| `revision03_capacity` | `generation_oracles.py --capacity`, explicitly bounded to 50,000 attempts | Independent union probes and exclusion clearance pass |
| `revision03_gpu_bridge` | `gpu_bridge_oracle.py` | Real CUDA B/E capture by ID, fresh restore, midrun refusal, exact CSV replay pass |
| `revision03_maple_abi` | `maple_abi_oracle.py` | Actual MAPLE object validator; nonzero-origin coordinates and ID sidecar pass |
| `revision03_package` | `check_package.py ABSOLUTE_REVISION_03` | Real sdist build; unpacked core/CLI import isolation and legacy imports pass |
| `installed_cli` | `cli_smoke.py` | validate-species, init, generate, inspect, export-canopy pass |

Checks on the frozen revision use that absolute directory in PYTHONPATH. Installed
files are byte-identical. Package build used existing system setuptools 68.1.2;
the model interpreters lack build tools. No wheel was built, fetched or installed.
The two-plant GPU bridge fixture advanced one zero-uptake day, constructed no
hydrology scenario, and claimed no calibration or performance result.

## Capacity and memory interpretation

| Grid / placement | Plants | Generation wall seconds | Plant-array bytes |
|---|---:|---:|---:|
| 128², overlap allowed | 3,243 | 0.1004 | 201,066 |
| 128², overlap excluded | 3,145 | 0.1694 | 194,990 |
| 512², overlap allowed | 51,893 | 1.6350 | 3,217,366 |

Fixture spacing is 0.25 m; grass diameter 0.15±0.03 m, shrub 2.0±0.3 m;
target covers 0.06/0.05. Independent uniform probes/SciPy cKDTree check cover;
minimum tested exclusion clearance is 0.0002958 m. Sampling error/tolerance
are explicit; the 512² exclusion case was not measured.

512² cover uses an 8,388,608-byte raster pair and 1,042,432 admitted scratch
bytes, plus population indices. Full oracle-process peak RSS is 89,584 KiB,
including SciPy and interpreter. The separate memory oracle traced 1,540,909
bytes against 1,522,208 admitted raster/scratch array bytes plus a documented
65,536-byte Python metadata allowance. No P×grid or P² storage is introduced.
The early geometry probe's memory text is observational; the dedicated memory
oracle is authoritative. These are initialization measurements, not growth speed,
full-model memory or GPU memory measurements.

## Corrections and error disposition

Raw failures remain preserved. `candidate_phase0_01*`, `candidate_phase0_02`,
`early_state_probes`, `early_geometry_recording`, `ownership_bridge_01`,
`checkpoint_state_02`, `checkpoint_memory_replay_02`, `transfer_preflight_02`
are **check_failed**, corrected and superseded by the passing revision-3 checks.
`early_generation` was **interrupted** after an excessive rejection-placement
runtime; its child was reaped. `diagnostic_generation` is **check_failed** from
the bounded diagnostic timeout. Large-canopy-first placement corrected the cause.

`revision03_transfer` is **check_failed in the Codex harness**: NumPy bool could
not be JSON serialized; explicit bool conversion fixed reporting.
`revision03_transfer_fixed` is **tool_error**: a relative cwd produced an invalid
relative PYTHONPATH. The corrected absolute-cwd retry passed. The sandbox GPU
skip attempt is **tool_error**, not CUDA verification; actual CUDA passed.
There was no auth/network/rate-limit failure in the three Claude revisions;
structured `is_error:false`, success and reaped exit 0 are archived.

## Preservation, identity and remaining scope

`installation.json`, `scope_final.json`, `claude_tool_audit.json`, the immutable
baseline and revision manifests establish scope. `finalize_identity.py` verified
all 1,796 original files against the allowed-change set, exact installed hashes,
root AGENTS.md, historical locks/accepted archives and legacy numerical sources.
All 71 locked SYRUP sources and the inspected MAPLE interface/instruction hashes
match. Current upstream Git status and 412 MAPLE Python sources are pinned in
`final_upstream_git.json`, `maple_runtime_identity_current.json` and the new
`dependencies/phase0-vegetation.lock.json`. Upstream work by others is preserved;
this task wrote no upstream source. Earlier locks remain historical unchanged.

No full ecological/native scenario, calibration, new injury law, upstream
integration or 30-year run was performed. Vegetation replay is not full coupled
restart. Commit is authorized after acceptance; push requires the user's remote
URL, requested asynchronously and still unavailable at this record.
