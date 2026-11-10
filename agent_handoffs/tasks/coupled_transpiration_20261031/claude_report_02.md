All authorized files are written. Nothing further is needed; here is the report.

## Report — bounded coupled-transpiration implementation (unexecuted)

**Files written/changed (all uncommitted, only the permitted set):**
- `benchmarks/daily_water_growth.py` — additive edits only; `spatial_mapping.py` and original tests untouched.
- `native/maple_syrup/subsurface_transpiration.py` (new), `native/test_transpiration.py` (new)
- `leaf/__init__.py`, `leaf/coupling.py` (new), `tests/test_coupling.py` (new)
- `docs/coupled-transpiration.md` (new)

**Native module contract delivered**
- `prepare_transpiration(column, *, extra_reserve_bytes=0)`: `_check_column`, residual/saturation copies, scalar admission before parameter copy + (1,1) warm-up, Numba `njit(cache=True)` / CUDA `RawKernel` compile; CUDA device + `preflight` checked.
- `withdraw_transpiration(op, state, requested_water_m)`: law exactly as specified; time preserved; fresh `SoilColumnState`; `actual_m/unmet_m/scale`; `TranspirationBudget` (area multiplied once; residual and request residual vs `volume_roundoff_bound_m3`); no clipping; validation on all cells via `_check_state`/`_check_dynamic` plus one packet (water finite/≥0/≤ sat+tol per layer, request finite/≥0/zero on inactive); one result packet; admission before output allocation.
- `step_with_transpiration(...)`: hydro `ss.step` then withdrawal; dt=0 identity requires zero T; combined admission (state + forcing + live hydro outputs + T work + column/caller reserves); `CoupledBudget` with distinct E and T; intermediate water not retained.

**Vegetation reuse**: `Simulation(..., vegetation_only=False, external_reserve_bytes=0)` (defaults unchanged); veg-only skips W/scale/removed/overflow/W0/cap/rech/period; `step_days`/`summary` refuse in that mode; `plant_summary`, `live_names`, mode-aware ledgers (`soil_maps`, `external_reserve_bytes`), compiled `_gather_candidate_cpu` and CUDA `gather_candidate` (fresh [P] candidate), `gather_uptake_candidate`, `commit_daily_uptake` (reuses the transactional weekly update and latch/retry).

**Coupled driver**: `CoupledSimulation(case, column, state, inputs=None)`; strict N/L/area/namespace/device checks; `step_day`/`step_days` daily-only, phase refusal (`CoupledPhaseError`), request = R/area zeroed on inactive cells (no renormalization), candidate finiteness and plant-vs-native volume closure checked before publishing; whole-day rollback on native/gather failure; weekly refusal latch with `retry_weekly_update`; `summary()`/`memory_report()` stream scalars.

**Limitations / assumptions to verify (I could not execute anything)**
- `soil_layers` assumed to accept scalar theta parameters and a positive tiny diffusivity (1e-15) in tests.
- `estimate_step_memory` fields `host_bytes`/`device_bytes` and `StepBudget` field names used as given in the contract.
- Native storage definition assumed consistent between `storage_m3` and `StepBudget` (closure is independent of inactive-water handling, but verify).
- `maple_syrup.subsurface_transpiration` import falls back to the repository `native/` tree only if not installed alongside.
- CUDA tests skip when unavailable; no timings or pass counts are claimed.
