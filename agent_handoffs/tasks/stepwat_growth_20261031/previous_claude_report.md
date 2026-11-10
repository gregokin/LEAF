# Claude final author report

Task coupled_transpiration_20261031 revision5; author did not execute tests.
Independent results are in [verification](verification.md).

Everything in this revision is complete; no further tool calls are needed. Final report:

## Revision 5 report (unexecuted; Codex runs qualification)

**Literal CUDA edit** (`native/maple_syrup/subsurface_transpiration.py`, `_CUDA_SRC`): added
`const double positive_inf = __longlong_as_double(0x7ff0000000000000LL);` at the top of the per-cell body and replaced both `INFINITY` uses — `mn = positive_inf` and `qv == positive_inf`. No headers, build options, physics, validation or CPU paths changed.

**1. Column reserve under `case.max_bytes`** (`leaf/coupling.py`): `column_reserve_bytes` (strictly validated) is now part of the vegetation external reserve at construction (`native_reserve`) and in `_native_reserve_bytes()`, so every vegetation ledger (init zero-edge/edge-bound/generated, weekly, retry) and the cheap scalar pre-check include it once; the exact trigger (cap 1e6 B, reserve 1e8 B, high availability) is refused before `Simulation` is constructed. Not double-added elsewhere; the column is never mutated.

**2. Full-peak admission** (`_vegetation_admission`): the fully reserved ledger's `total_bytes` is the single simultaneous-live peak, compared with BOTH `case.max_bytes` and a strict fresh availability reading; stale ledgers (external reserve below the current native reserve incl. retained diagnostics) are refused; the peak is published only on admission. Callback timing (before init/weekly/retry allocations) and latch semantics unchanged. Tests: `test_column_reserve_counted_under_case_max_bytes`, `test_vegetation_admission_uses_full_ledger_peak` (availability at the old smaller native bound and at `total-1` → refused with `peak_bytes == total`; at `total` → admitted; `max_bytes = total-1` → refused; stale ledger → refused; refused peaks not published).

**3. Device binding checks**: new `_check_device(stage)` (current device == column device == vegetation bound device, plus `operator.check_device()` once prepared) at `step_day` entry (before request-buffer work), `retry_weekly_update` entry (before candidate buffers / weekly RawKernels) and in the admission callback (covers construction, where the operator does not yet exist). GPU test `test_gpu_coupled_entry_points_check_bound_device` simulates a binding mismatch via the recorded bound id (no other physical GPU), asserting retry/step/callback refusal with the GPU builder and native step spies uncalled and the request buffer untouched, then recovery; construction-time mismatch via a fake current device is refused before `Simulation`.

**4. Fixture**: `test_hydrology_budget_split_run_matches_single_run` now starts BOTH layers at 0.030 m (θ 0.30, above field capacity, below saturation) and asserts positive bottom drainage directly; mask, diffusion, intake, E, T and split-run comparisons retained; no production or tolerance changes.

Docs updated (`docs/coupled-transpiration.md`: reserve composition, single-peak admission against both bounds, device-binding checks).
