# Coupled native soil water + individual-plant growth (daily)

Implemented and independently qualified against SYRUP's actual `feature/subsurface-hydrology`
checkout. SYRUP owns the physical soil-water state and transpiration subtraction; LEAF owns
individual plants, species-specific root fractions and actual-uptake-driven weekly growth.
The standalone benchmark retains its existing numerical behavior.

The later [STEP-style growth option](stepwat-growth.md) adds species-specific
weekly/year-end aboveground growth. Its [current source lock](../dependencies/stepwat-growth.lock.json)
pins the installed checkout; the original coupling lock and measurements below
retain their historical source identities and fixture growth rule.

The [source lock](../dependencies/syrup-subsurface.lock.json) records HEADs and all dirty runtime
identities. SYRUP HEAD is `deb60c29834a4aad0bf8cb4cfbb611fc28ebe090`; the hydrology and new
transpiration files are uncommitted, so that HEAD alone cannot reproduce this integration. MAPLE
HEAD is `963b06ad1cf0f3e2c420a7d7fb50939e3bd43b7d`. No existing upstream source was modified.
Qualification evidence and exact commands are under
[the task record](../agent_handoffs/tasks/coupled_transpiration_20261031/verification.md).

## Components

| File | Role |
|---|---|
| [SYRUP transpiration module](../../SYRUP/src/maple_syrup/subsurface_transpiration.py) | Explicit transpiration withdrawal operator for the native column (`maple_syrup.subsurface`). NumPy reference, Numba and CUDA RawKernel. |
| `benchmarks/daily_water_growth.py` | Vegetation model. New opt-in `Simulation(..., vegetation_only=True, external_reserve_bytes=...)`, compiled fresh-candidate gather, `commit_daily_uptake`, `plant_summary`. |
| `leaf/coupling.py` | `CoupledSimulation(case, column, state, inputs=None)`: the daily coupled driver. |
| [SYRUP native tests](../../SYRUP/tests/subsurface/test_transpiration.py), `tests/test_coupling.py` | Focused tests (CUDA tests skip when unavailable). |

Imports require the real SYRUP source package: `from maple_syrup import subsurface` and
`from maple_syrup import subsurface_transpiration`. The new module is installed alongside the
column on its branch. Qualification first used a symlink overlay to the same original native
sources, then verified the installed import paths. There is no path fallback; a missing extension
raises a clear `ImportError`. Vegetation
code imports the native package lazily (inside `CoupledSimulation.__init__`), so `benchmarks` and
`leaf` import without `maple_syrup`; the coupled driver requires it plus `maple_syrup.conservation`
(`volume_roundoff_bound_m3`) and uses `maple.core.backend.to_host` for the single GPU report packet
when available.

## Native withdrawal (`subsurface_transpiration`)

* Order of every public operation: metadata/time checks with the pinned native helpers
  (`_check_column/_check_state/_check_dynamic/_check_dt`, no allocation) → strict scalar guards →
  scalar memory admission → allocation → fused compiled kernel → ONE host packet → reject or publish.
* `prepare_transpiration(column, *, extra_reserve_bytes=0)` — admits (host + compute-side residual and
  saturation-bound vectors, (1,1) warm-up) BEFORE building even the small per-layer vectors, then
  compiles the Numba/CUDA kernel and warms it up. The active mask is `column.active_u8` (reused by the
  kernel; never duplicated or downloaded).
* `withdraw_transpiration(op, state, requested_water_m[L,N])` — pure. Per layer/cell:
  `available = max(w - residual_m[l], 0)`, `actual = min(requested, available)`,
  `water' = w - actual`, `unmet = requested - actual`, `scale = actual/requested` (0 for zero demand).
  Time preserved. The fused kernel (NumPy reference; Numba layer-outer/cell-inner loop; CUDA thread
  per cell looping layers) writes the four `[L,N]` outputs plus `[8,N]` per-cell rows: active-only
  sums of water before/after, actual, unmet, requested; min water after (all cells); validity flags
  (bit 1 water not finite/negative/above `saturation_m + capacity_tolerance_m`, bit 2 request not
  finite/negative, bit 4 positive request on an inactive cell — all cells, inactive included); max
  request. Eight row reductions are stacked into ONE host packet; any flag rejects without publishing.
  `TranspirationBudget` volumes are ACTIVE-cell depth sums × `cell_area_m2` (once), the same
  definition as native `storage_m3`/`StepBudget`; `residual = after - before + actual`,
  `request_residual = requested - actual - unmet`, bound from `volume_roundoff_bound_m3`. No clipping.
* `step_with_transpiration(op, state, dt_s, *, added_water_m, potential_evaporation_m_s,
  requested_transpiration_m, extra_reserve_bytes=0)` — metadata checks, combined admission, native
  `step` (advances `t_s` once; its own native packet), fused withdrawal (time preserved; one T packet):
  two host reads per normal day. `dt=0` is an identity and rejects any positive requested T (flagged
  from the same packet, even when the column is dry). `CoupledBudget` uses the true endpoints
  (`storage_before` = native hydro before, `storage_after` = active storage of the final state) and the
  distinct intake/return/drainage/E/T terms: `residual = after - before - (added - return - drainage -
  E - T)` with bound `volume_roundoff_bound_m3(hydro n_terms + T n_terms, before + added + requested)`;
  `intermediate_mismatch_m3` (hydro after vs T before of the same intermediate state) is also bounded.
  The intermediate water is not retained.
* Memory: `estimate_transpiration_memory(L, N, *, backend, reserve_bytes, extra_reserve_bytes)` fields:
  `parameter_bytes` (2·8L persistent vectors), `state_bytes`, `input_bytes`, `output_bytes` (4·8LN),
  `row_bytes` (8·8N), `scratch_bytes` (conservative: NumPy reference 6 FP64 `[L,N]` equivalents; all
  backends one FP64 N-row of reduction workspace + three simultaneously live packets — the eight 0-d
  reduction results, the stacked compute-side packet and the host packet), `packet_bytes`,
  `host/device_bytes`, `required_bytes` (admitted side + reserves),
  `total_bytes` (host + device + reserves). Sizes must be non-boolean integers ≥ 1; reserves, extra
  reserves and `available_bytes_fn()` readings non-boolean integers ≥ 0 (a negative extra reserve can
  never bypass admission). `estimate_combined_memory(column, *, extra_reserve_bytes, query_available)`
  adds the native hydro estimate and the original state once. Refusal raises `SubsurfaceMemoryError`
  without mutation; budgets are never raised. GPU: arrays stay on the column's device; current device
  and `preflight` are checked before work; no per-layer host checks, no full-field downloads.

Units: depth in m over the already defined interval; the operator never scales a rate.

## Vegetation changes (defaults unchanged)

* `Simulation(case, backend, inputs=None, *, vegetation_only=False, external_reserve_bytes=0)`.
  In vegetation-only mode `W, scale, removed_cum, overflow_cum, W0, cap, rech, period` are `None`;
  `R`, traits, roots, weekly uptake/growth stay. `step_days()` and `summary()` refuse; use
  `plant_summary()`, `live_names()`, `counters()` (`recharge_events=None`).
* Ledgers: `state_bytes_estimate(..., soil_maps=)` and `admission_ledger(..., soil_maps=,
  external_reserve_bytes=)`; every init/weekly admission counts the external reserve under the
  original `max_bytes`/`max_edges`.
* `gather_uptake_candidate(scale, out)` — compiled gather-only twin of the standalone uptake
  (`_gather_candidate_cpu` / CUDA `gather_candidate`), fresh `[P]` candidate, no accumulation. Both
  arrays are validated BEFORE launch: exact backend ndarray type (no subclass/list/implicit
  conversion), float64, C-contiguous, expected shape, and on the GPU: the simulation's BOUND
  construction device (`Simulation.device_id`, recorded at init) must equal the current device, the
  argument's device and the resident arrays' device (authoritative `Dp`).
* `Simulation(..., admission_check=None)` — optional `callable(ledger)` invoked with every fully
  reserved ledger (init zero-edge/edge-bound/generated, weekly `rebuild_bound`/`rebuild_actual`, and
  retries) after the `max_bytes` check and before any materialization or graph build; raising refuses
  (inside weekly work it latches exactly like any other rebuild refusal). Standalone default: none.
* `commit_daily_uptake(candidate)` — external-water path only (`vegetation_only=True`; standalone
  simulations refuse). Trusted-caller semantics: the coupled driver has already checked finiteness,
  sign and closure against the native water; this method validates mode and array metadata, then
  accumulates `uptake_week/uptake_cum`, advances `day` and runs the existing transactional weekly
  update at day % 7 == 0 (latch/`retry_weekly_update()` unchanged).
* `counters()` gains `vegetation_only: True` only in water-free mode; the historical standalone dict
  is unchanged.

## Coupled driver (`leaf.coupling.CoupledSimulation`)

Construction checks: `column.n_cells == nx*ny` (row-major `j*nx+i`), `column.n_layers == layers`,
`cell_area_m2 == dx²` (4 ulp), namespace/backend alignment (`cuda`→GPU vegetation on the same CuPy
device; `numpy`/`numba`→CPU), strict `extra_reserve_bytes` (non-boolean integer ≥ 0). No
copies/transfers/resampling. Root profile rows are the native layers top first.

Admission order at construction (nothing of size P or N is allocated before the guards pass):
1. cheap scalar lower bound — native side (soil state + hydro + T arrays/parameters, `column.reserve_bytes`,
   extra reserve) + coupling buffers + vegetation state/scratch/mirrors with ZERO edges — checked against
   the case `max_bytes` and against `column.available_bytes_fn()` (`AdmissionRejected`);
2. the vegetation `Simulation` (its own zero-edge / edge-bound / generated ledgers all include the native
   reserve under `max_bytes`);
3. full native admission with the actual vegetation arrays and graph, then `prepare_transpiration`.
The vegetation external reserve (counted by every vegetation ledger under `case.max_bytes`) is: native
side (hydro + T arrays/parameters) + soil state + coupling buffers + operator parameters +
`column.reserve_bytes` + caller `extra_reserve_bytes` + retained diagnostics. It is counted once, there.
The vegetation `Simulation` is constructed with `admission_check=CoupledSimulation._vegetation_admission`:
every fully reserved vegetation ledger (init bounds, weekly rebuild bound and actual old+new CSR, retries)
IS the single simultaneous-live peak; that same `total_bytes` is compared with BOTH `case.max_bytes` and a
fresh strict `available_bytes_fn()` reading (a ledger whose external reserve is stale is refused too). A
refusal raises `AdmissionRejected('coupled.vegetation_admission')` BEFORE any graph allocation and, inside
weekly work, latches (`weekly_pending`, `retry_weekly_update()`), never re-applying the accepted day's
uptake; the peak is published only on admission. The column reserve/cap is never mutated.
Device binding (checks only, no switching): `step_day`, `retry_weekly_update` and the admission callback
first verify that the current CUDA device equals the column device and the vegetation's bound
construction device, and run the operator device/preflight check once it exists.
After each accepted day the vegetation external reserve is refreshed from pure metadata (native side
fixed at construction + that day's retained diagnostics) BEFORE `commit_daily_uptake`, so weekly work
counts them; availability is queried only at construction, inside daily native admission (before any
publication) and inside weekly work — never after a committed day outside that boundary.
`memory_report()['peak_required_bytes']` is the ADMITTED-ESTIMATE peak: the maximum over construction
ledgers, daily native combined admissions and weekly rebuild ledgers (old+new CSR, retries) each with the
native requirement and column reserve; `vegetation_rebuild_peak_bytes` is the vegetation-side subset.
Neither is an observed allocator peak.

Per day (`step_day(added_water_m[L,N], potential_evaporation_m_s[N], *, phase='interevent')`,
`dt = 86400 s`; `step_days(records)` accepts pairs or objects with those attributes, `dt_s` must be
86400): request depth = `R / area` zeroed on inactive cells (no renormalization), previous diagnostics
released, native `step_with_transpiration`, compiled gather of the per-plant actual candidate, checks
(finite, ≥ 0, `sum(candidate) == native actual_m3` within the MAPLE bound) **before** publishing; then
soil state/counters/cumulative scalars are published and `commit_daily_uptake` runs (weekly refresh at
7 accepted days). Failures:

* native/admission/gather failure → nothing published (prior soil state, plant accumulators, counters
  unchanged; no partial day);
* refused WEEKLY rebuild → the day and its uptake are already accepted and latched; `weekly_pending`
  is True, further `step_day` calls are refused until `retry_weekly_update()` succeeds (same semantics
  as the standalone `retry_weekly_update`).

Phases other than `interevent` (wind/rain/water event/storm) raise `CoupledPhaseError` before work.
No synthetic recharge: forcing is mandatory. Native `t_s` may start anywhere; `days` counts accepted
intervals since construction and `t_s` advances by exactly 86400 per day.

`summary()` streams scalars only: cumulative added/return/drainage/E/T (+unmet), native initial and
current storage (`storage_m3`), water residual vs accumulated bound, potential vs mapped vs
unmapped(inactive) demand, native vs plant actual flux closure, biomass budget, calendar (native time
vs accepted days), mask/profile/backend metadata, memory scopes (`memory_report()`: vegetation ledger,
buffers, retained diagnostics, operator/prepare bytes, last combined native estimate, peak estimate —
estimates, not allocator peaks).

## Simplified soil profile and deferred surface remapping

For the current phase, use soil texture to inform soil-water properties within
the simplified layered representation. Continue the native daily water budget
and plant uptake without explicitly tracking how erosion or deposition changes
the material, geometry, stored water or root access of each soil layer.

**Future implementation:** define how changes to the shared MASY surface and
texture update layer properties and boundaries, account conservatively for water
in added or removed material, and update root access. Preserve each plant's ground
elevation at establishment when evaluating burial or exposure. This detailed
remapping is deferred by user direction and is not required for the current phase.
Weekly root-map updates caused by plant growth remain part of the current model.

## Limitations

* Daily-only interface; no partial-day calendar, no storm/MASY event integration, no texture change,
  no production ecology. Fixture ecology remains uncalibrated.
* Splitting is first order (hydro then T within the day).
* Memory admission compares estimates with a reading of `available_bytes_fn`; it is not a
  reservation/pool bound.
* Serialization is not provided; split runs on the same live objects are equivalent.
* Budgets follow the native ACTIVE-cell storage definition; inactive water is preserved exactly and
  excluded from every volume.
* Test fixtures: isolated uptake oracles use `diffusivity_m2_s=0` (no redistribution); separate
  budget/parity tests use positive diffusivity with intake, evaporation, drainage and inactive cells.
  CUDA tests probe hardware availability only (`cupy.cuda.runtime.getDeviceCount()` /
  `dwg.gpu_available()`); construction or implementation errors on an available device fail.
* Claude authored the bounded candidate without execution tools. Codex independently reviewed,
  corrected, tested and installed the qualified bytes; the archived author report is not test evidence.

## Use and verification

Run from `/home/okin/leaf` with the existing environment and the recorded source branch:

```sh
/home/okin/MAPLE/.venv-cupy/bin/python -B \
  agent_handoffs/tasks/coupled_transpiration_20261031/verify_sources.py
PYTHONPATH="$PWD:/home/okin/SYRUP/src" CUDA_VISIBLE_DEVICES="" \
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/leaf-numba-cache \
/home/okin/MAPLE/.venv-cupy/bin/python -B -m pytest -q \
  -p no:cacheprovider -c /dev/null tests /home/okin/SYRUP/tests/subsurface
```

For CUDA correctness, select allocated physical GPU0 explicitly with
`CUDA_VISIBLE_DEVICES=GPU-5c8bd3d4-2885-82c6-a0f0-ad2b7f87b871` in the test command.
The CPU run passed232 tests (31 GPU-related skips); the real CUDA run passed261
(two intentionally unavailable-GPU-path skips). No implementation failure is treated as a skip.
Independent analytical oracles additionally cover unequal species fractions, cell areas1 and0.0625m²,
inactive storage, weekly biomass, split stepping, full reserve admission and latched retry without
duplicate water use. The old default remains bit-identical over23days/three weekly updates.

Example below uses explicit zero soil intake and potential soil evaporation. For real forcing,
`added_water_m` is **accepted soil intake depth**, not a second rainfall/infiltration calculation.
Pass an existing native column/state instead when available; publish `sim.state` as the next
authoritative soil state. Native states are immutable results, not in-place water arrays.

```python
import numpy as np
from maple_syrup import subsurface as ss
from leaf import CoupledSimulation

case = dict(name="example", nx=8, ny=8, dx=0.25, grass_count=10, shrub_count=1,
            grass_diameter=0.15, shrub_diameter=2.0, root_multiplier=3.0,
            pattern="uniform", seed=1, layers=3, max_edges=100000, max_bytes=100000000)
layers = ss.soil_layers([0, 0.1, 0.3, 1.0], theta_residual=0.05,
                       theta_field_capacity=0.25, theta_saturation=0.45,
                       diffusivity_m2_s=2e-7, theta_evaporation_stress=0.20)
column = ss.prepare_column(layers, np.ones(64, dtype=bool), 0.25**2, backend="numba")
soil_state = ss.uniform_state(column, [0.15, 0.15, 0.15])
sim = CoupledSimulation(case, column, soil_state)
added_water_m = np.zeros((3, 64), dtype=np.float64)
soil_evaporation_m_s = np.full(64, 3e-8, dtype=np.float64)
for day in range(7):
    sim.step_day(added_water_m, soil_evaporation_m_s, phase="interevent")
soil_state = sim.state
assert sim.summary()["ok"]
```

To supply field plants, pass the existing `inputs` table (`x`, `y`, `canopy_radius`,
`root_radius`, stable `ids`, optional `root_profile_id`) and named
`case["root_layer_fractions"]`; the [profile guide](species-root-profiles.md) defines the schema.
Request/gather weights use these same per-species fractions. Fractional soil water is shared
proportionally within each cell/layer; there is no wetter-layer compensation.

## Measured capacity and timing

Each fresh process stepped365 explicit days with52 weekly growth/root rebuilds (53 total maps),
retaining the complete population. Initial soil theta0.15,10mm accepted soil intake to the top layer
every14days, potential soil evaporation3e-8m/s; no storm/rainfall/wind solver ran. All cumulative
water/plant-flux/biomass/calendar checks passed. Sizes/counts reuse the Jornada cover scenarios
from the earlier capacity task, with0.25m cells.

| Backend | Grid / layers / root radius | Plants | One model year, s | Peak live GPU, MiB | Pool reserved, MiB | Host RSS, MiB |
|---|---|---:|---:|---:|---:|---:|
| numba | 128² / 3L / 3× | 3,717 | 1.40 | — | — | 477.8 |
| cuda | 128² / 3L / 3× | 3,717 | 4.98 | 7.4 | 9.5 | 753.1 |
| cuda | 400² / 3L / 3× | 36,305 | 1.98 | 72.6 | 92.7 | 561.0 |
| cuda | 400² / 2L / 5× | 36,305 | 2.64 | 70.3 | 252.1 | 560.7 |
| cuda | 512² / 3L / 5× | 59,483 | 4.21 | 135.4 | 439.7 | 562.1 |
| cuda | 400² / 3L / 3× | 547,588 | 3.15 | 224.1 | 476.5 | 620.4 |

GPU: GTX1080Ti, allocated physical GPU0; another job used about603MiB and18–22% GPU at the
before/after samples. These timings include periodic reports and first-use compilation during the
loop; the128² case also incurred3.48s setup. Other GPU setups were0.08–0.40s, with0.42–0.52s
tiny warm-up. CPU warm-up was1.49s. They are bounded one-year observations, not a clean steady-state
comparison, measured30-year run, or a full MASY throughput estimate. Growth can change graph costs.

Live GPU peaks are allocation-hook measurements including rounded requested blocks and transient
old/new graphs. Pool reserved values are samples; host RSS is whole-process peak, including Python,
JIT and libraries. CUDA contexts/other allocators/jobs are outside the live-array hook. The fixed
1.5GB case/pool cap was not raised; no population/state was discarded. All five GPU profiles ended
with0 tracked live bytes and0 unknown frees. Scalar admission estimates remain conservative
working-array/reserve estimates, separate from allocator measurements. Exact startup, loop, budgets,
counters, source hashes and process receipts are in the task's `performance_*.json` artifacts.

The [profiling script](../agent_handoffs/tasks/coupled_transpiration_20261031/performance.py) and
[case manifest](../agent_handoffs/tasks/coupled_transpiration_20261031/performance_cases.json) reproduce
these bounded fixtures with `--backend numba|cuda --case NAME --days 365 --output NEW_PATH`
Existing output paths are refused.
