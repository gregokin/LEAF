# daily-water-growth: bounded long-horizon water/growth benchmark

Generic, synthetic workload layered on `benchmarks/spatial_mapping.py` (imported as `sm`, unmodified; its
CPU/GPU root-graph builders, admission, transfer and warm-up are reused, not duplicated). Everything
ecological below is a **benchmark fixture**, not calibrated ecology.

Status: independently reviewed and executed by the project orchestrator (unit suites including the
executed CPU/GPU parity check, and complete 30-year runs with conservation and calendar checks).
Evidence, pool-budget measurement policy and measured results: `../docs/daily-water-30year.md`.
The author of this module did not execute it; nothing here reports timing values.

## Fixture rules

- Soil `W[L, N]` FP64 m3: 0.15 m plant-available storage per column split equally over `L` layers, half
  full at start; capacity per layer cell `0.15/L*dx^2`.
- Recharge 0.010 m every 14 days from day 0, split equally over layers, clipped to capacity; clipped
  volume accumulated per cell as overflow. No evaporation, drainage or transport.
- Plants: `B0 = canopy area * 1 kg/m2`; `Dp = 0.003 m/day * canopy area`; root radius =
  `root_multiplier * canopy radius`; horizontal weights from `sm` (uniform, sum 1), vertical `1/L`.
  Grass/shrub trait by nearest initial diameter (ties: first `grass_count` indices are grass).
- Demand cache `R[l, j] = sum_p Dp*w_pj/L`, built at start and after each weekly growth only
  (compiled CPU loop / FP64 `atomicAdd` kernel). Constant within a week.
- Daily: recharge, `accepted_j = min(W_j, R_j)`, `W -= accepted`, `scale_j = accepted/R_j` (0 if
  `R=0`), cellwise cumulative removal; each plant gathers `sum_j Dp*w/L*scale_j` (weekly + cumulative).
  No overdraw, no phantom water, no wetter-layer compensation, no plant omitted.
- Weekly (continuous clock, after day 7k): `gain = 1000*WUE*weekly uptake` (WUE 0.002 grass, 0.0015
  shrub), `turnover = B*(1-exp(-0.003*7))`, `B = B - turnover + gain`, `canopy = sqrt(B/pi)`,
  `root = multiplier*canopy`, `Dp` updated; weekly uptake reset. No caps, no plant removal.
- Growth is transactional: candidate arrays -> byte admission (live state + old graph + new graph
  bound) -> `sm.check_admission` -> rebuild CSR via `sm` builder -> old+new peak check -> swap/commit
  -> rebuild `R`. Rejection raises `sm.AdmissionRejected`; prior state is preserved and reported.
- Calendar: default 30*365 = 10950 days -> 1564 weekly updates, 2 pending days (uptake retained,
  never flushed), 1565 maps (`calendar(days)`).

## Usage

```
python -m benchmarks.daily_water_growth --cases-json cases.json --output out.json \
    --backend cpu|gpu|both [--years 30] [--days N] [--report-every 365] [--isolate]
```
API: `run_case(case, backend, days=None, years=None, report_every=365, return_arrays=True,
inputs=None, memory_hook=None, progress=None)`; `Simulation(case, backend, inputs=None).step_days(n)`,
`.summary()`, `.export_arrays()`, `.retry_weekly_update()`; `calendar(days)`; `gpu_available()`;
`initial_edge_bound`, `admission_ledger`, `scratch_allowance`. Pool profiling uses an external CuPy
MemoryHook (`malloc_postprocess`/`free_postprocess(**kwargs)`) passed as `memory_hook`; `run_case`
enters it exactly once around warm-up, initialization, loop, exports and cleanup (its `report()` is
attached under `memory.pool_hook` if present); no built-in hook.

Admission ledger: every decision (init bound, generated, weekly bound, weekly actual) sums state,
old graph, new graph (per-trait bbox bound pre-generation; actual edges after build), a documented
scratch allowance `64*P + 8*L*N + 64 KiB` and host mirrors, against `max_bytes`; rejections raise
`sm.AdmissionRejected(stage, reason, ledger)`. A refused weekly update latches the simulation at the
boundary (uptake retained, committed state unchanged); only `retry_weekly_update()` may continue.
GPU startup runs a tiny 7-day simulation to exercise lazy CuPy paths before timing.

## Output per case

`status` (ok | rejected | audit_failed | error | gpu_unavailable), validated case, environment/versions/
device, `timing` (startup/cache load, initialization, daily-block, weekly-update, audit/report,
integration = loop minus audit, end-to-end = init + loop, run_case, since-module-import),
`throughput` (model days/wall s, model years/wall min, simulated s/wall s), `counters`, `audit`
(raw residuals, tolerances 1e-9 relative + 1e-12 absolute, per-check flags), `memory` (live array
dtype/shape/nbytes, estimated admission peak (state + old + new graph + scratch allowance + mirrors;
not an observed allocator peak), admission ledger, scratch allowance, host mirrors,
ru_maxrss whole-process peak, sampled CuPy pool, external hook report if supplied), yearly `progress`
JSON lines (also printed to stderr). CLI omits arrays. Exit
code 0 only if every result is `ok`, else 2. `--isolate` runs each case/backend in a subprocess for
per-case RSS. No measured speedups are claimed anywhere.

Validation cadence: candidate weight normalization, positive counts and finite positive radii/biomass
are checked on every map rebuild (initial and each weekly growth) before commit; full mass-balance,
flux-equality, calendar and root/canopy ratio reports are produced at each annual report boundary
and at the end of the run.

Timing: the `since_module_import` timer is only a proxy for whole-process wall time; the independent
wrapper records the complete process time (and per-case RSS under `--isolate`).

Memory: the default CuPy memory pool keeps cached/reserved blocks that can greatly exceed the live
arrays listed in the ledger; the sampled pool numbers are reservations, not live state. The
pool-budget measurement policy and measured results are in `../docs/daily-water-30year.md`.

## Tests

`python -m unittest tests.test_daily_water_growth`: single-plant depletion, two-plant competition,
dry cells, recharge/overflow budget, weekly growth budgets/radii, 7/8/14-day counters and pending
uptake, weekly (not daily) demand-map rebuilds, split-run equivalence, memory refusal before
generation, invalid scalars, CPU/GPU parity (skipped without a CC>=6 device; tolerance, not bitwise).

## Limitations

- Synthetic ecology: storage, recharge, demand, WUE and turnover are fixtures chosen for a bounded,
  auditable workload; results say nothing about real vegetation or soils.
- Uniform root weights per intersected cell and uniform `1/L` layer weights; no wetter-layer
  compensation, no evaporation, drainage, transport or storm hydrology.
- CPU daily kernels are serial `njit` loops (no `prange`); they are a standalone reference, not a
  tuned multicore baseline.
- Weekly growth uses NumPy/CuPy pointwise operations on preallocated candidate arrays rather than
  custom kernels; the daily hot paths are compiled CPU loops / RawModule GPU kernels.
- No production restart or checkpoint framework: `Simulation.step_days` split runs are equivalent,
  but state is not serialized.
- Admission peaks are ledger estimates (state + old/new graph + scratch allowance + mirrors), not
  observed allocator peaks; measured peaks come from the independent profiler in the linked report.
