# daily-water-growth: bounded long-horizon water/growth benchmark

Generic, synthetic workload layered on `benchmarks/spatial_mapping.py` (imported as `sm`, unmodified; its
CPU/GPU root-graph builders, admission, transfer and warm-up are reused, not duplicated). Everything
ecological below is a **benchmark fixture**, not calibrated ecology.

Status: the author of this module did not execute it; nothing here reports timing values. Independent
verification and measurements are recorded by the project orchestrator: the current species-root-profile
version (0.2.0) is covered by the follow-up `../docs/species-root-profiles.md`; the historical 0.1.0
equal-layer results (unit suites, CPU/GPU parity, complete 30-year runs with conservation and calendar
checks) are in `../docs/daily-water-30year.md`.

## Fixture rules

- Soil `W[L, N]` FP64 m3: 0.15 m plant-available storage per column split equally over `L` layers, half
  full at start; capacity per layer cell `0.15/L*dx^2`.
- Recharge 0.010 m every 14 days from day 0, split equally over layers, clipped to capacity; clipped
  volume accumulated per cell as overflow. No evaporation, drainage or transport.
- Plants: `B0 = canopy area * 1 kg/m2`; `Dp = 0.003 m/day * canopy area`; root radius =
  `root_multiplier * canopy radius`; horizontal weights from `sm` (uniform, sum 1); vertical weights
  `f[s, l]` from the plant's root profile row (see "Root layer profiles"; `sm`'s legacy uniform
  vertical metadata is ignored). Grass/shrub trait (WUE) by nearest initial diameter (ties: first
  `grass_count` indices are grass).
- Demand cache `R[l, j] = sum_p Dp*w_pj*f[id_p, l]`, scattered directly into the resident `[L, N]`
  cache at start and after each weekly growth only (compiled CPU loop / FP64 `atomicAdd` kernel);
  no global `Rh[N]` map, no per-species rasters, no per-edge-by-layer tensors. Constant within a week.
- Daily: recharge, `accepted_lj = min(W_lj, R_lj)`, `W -= accepted`, `scale_lj = accepted/R_lj` (0 if
  `R=0`), cellwise cumulative removal; each plant gathers exactly its requested pieces times the same
  scale, `sum_j sum_l Dp*w_pj*f[id_p,l]*scale_lj` (weekly + cumulative). Soil caps competition; no
  overdraw, no phantom water, no wetter-layer compensation, no double rainfall, no plant omitted.
- Weekly (continuous clock, after day 7k): `gain = 1000*WUE*weekly uptake` (WUE 0.002 grass, 0.0015
  shrub), `turnover = B*(1-exp(-0.003*7))`, `B = B - turnover + gain`, `canopy = sqrt(B/pi)`,
  `root = multiplier*canopy`, `Dp` updated; weekly uptake reset. No caps, no plant removal.
- Growth is transactional: candidate arrays -> byte admission (live state + old graph + new graph
  bound) -> `sm.check_admission` -> rebuild CSR via `sm` builder -> old+new peak check -> swap/commit
  -> rebuild `R`. Rejection raises `sm.AdmissionRejected`; prior state is preserved and reported.
- Calendar: default 30*365 = 10950 days -> 1564 weekly updates, 2 pending days (uptake retained,
  never flushed), 1565 maps (`calendar(days)`).

## Root layer profiles (0.2.0)

`case['root_layer_fractions']` is an optional insertion-ordered mapping `profile name -> [L fractions]`,
top layer first. Every entry must be a finite positive number (no bool/NaN/inf/zero/negative), rows
must have exactly `L` entries and sum to 1 within `1e-12`; invalid rows are refused before any
generation, allocation or transfer and are never normalized silently. Defaults (configurable,
UNCALIBRATED fixtures): `L=3` grass `[0.70,0.20,0.10]`, shrub `[0.15,0.25,0.60]`; `L=2` grass
`[0.80,0.20]`, shrub `[0.25,0.75]`; `L=1` both `[1]`; other `L` linear descending (grass) / ascending
(shrub) weights normalized to one (table bytes checked against `max_bytes` before construction).

```json
{"name": "three_profiles", "nx": 400, "ny": 400, "dx": 0.25, "grass_count": 100, "shrub_count": 20,
 "grass_diameter": 0.5, "shrub_diameter": 1.5, "root_multiplier": 3.0, "layers": 3,
 "max_edges": 10000000, "max_bytes": 1500000000,
 "root_layer_fractions": {"grass": [0.70, 0.20, 0.10], "shrub": [0.15, 0.25, 0.60], "deep": [0.05, 0.15, 0.80]}}
```
Profile IDs index the named table in insertion order (`grass`=0, `shrub`=1, `deep`=2 above). Generated
plants take the `grass`/`shrub` row of their initial class; a custom mapping must name a class row only
if that class is actually present (a grass-only case may supply just `grass`), unless explicit IDs are
given. Order of operations at initialization: pure row/ID checks on caller-owned data (no copies) ->
aggregate admission preflight (state + graph bound + scratch + profile mirrors against `max_bytes`) ->
materialization of the small FP64 table / int32 IDs -> geometry generation (only `sm.CASE_FIELDS` are
passed to `sm.generate_inputs`) -> transfer. On GPU the profile IDs/table are copied in a separately
timed, once-synchronized step recorded as `profile_transfer_record` (`4*P + 8*S*L` bytes;
`profile_wall_s` times only this copy) next to the `sm.to_device` record (which has its own
geometry-copy timing); `transfer_summary` composes inputs (`40*P`), profiles and the WUE trait vector
(`trait_bytes = 8*P`, included only in total `initialization_s`) into `total_setup_h2d_bytes =
52*P + 8*S*L`. Before any P-sized edge-bound temporaries are computed, a scalar zero-edge ledger
(`init_scalar_bound`) refuses hopeless cases outright. No daily transfers or readbacks. Callers may pass `inputs['root_profile_id']` (integer, not bool/float, shape
`[P]`, `0 <= id < S`, validated before conversion/allocation); explicit IDs are authoritative, so three
or more named profiles/species can be used without inferring species from canopy sizes (WUE still
follows the grass/shrub size class). State: contiguous FP64 `root_layer_fractions[S, L]` and int32
`root_profile_ids[P]`, fixed across weekly growth (only canopy/root radius, root map and `Dp` change
weekly); exported in `arrays`, `result['root_profiles']` (names, table, bytes, source, assignment)
and `audit['root_profiles']`. Fractions are potential-uptake weights per layer, not volumetric root
density and not guaranteed actual uptake: a dry layer yields nothing to the plants rooted there.
CPU `njit` loops and GPU RawModule kernels implement the same request/gather; conservation holds as
before (uptake = removal). Audits (report boundaries) check table positivity/normalization, ID range
and that the requested total `sum R` equals `sum Dp` within `1e-8` relative.

Version note: 0.2.0 removes the `Rh[N]` map and adds `4*P + 8*S*L` bytes of IDs/table; the memory
ledger and daily kernels differ from 0.1.0, so earlier timings/peaks are not comparable. No speed
figures are claimed here; an independent follow-up measures the new 30-year cases.

## Usage

```
python -m benchmarks.daily_water_growth --cases-json cases.json --output out.json \
    --backend cpu|gpu|both [--years 30] [--days N] [--report-every 365] [--isolate]
```
API: `run_case(case, backend, days=None, years=None, report_every=365, return_arrays=True,
inputs=None, memory_hook=None, progress=None)`; `Simulation(case, backend, inputs=None).step_days(n)`,
`.summary()`, `.export_arrays()`, `.retry_weekly_update()`; `calendar(days)`; `gpu_available()`;
`initial_edge_bound`, `admission_ledger(..., n_profiles=2)`, `state_bytes_estimate(P, N, L, n_profiles=2)`,
`scratch_allowance`, `default_root_layer_fractions(L)`, `validate_root_layer_fractions(mapping, L)`,
`resolve_root_profile_names(case)`, `build_root_profile_table(case, names)`, `check_profile_ids(ids, P, S)`. Pool profiling uses an external CuPy
MemoryHook (`malloc_postprocess`/`free_postprocess(**kwargs)`) passed as `memory_hook`; `run_case`
enters it exactly once around warm-up, initialization, loop, exports and cleanup (its `report()` is
attached under `memory.pool_hook` if present); no built-in hook.

Admission ledger: every decision (init bound, generated, weekly bound, weekly actual) sums state,
old graph, new graph (per-trait bbox bound pre-generation; actual edges after build), a documented
scratch allowance `64*P + 8*L*N + S*L + 32*S + 64 KiB` (the `S*L` booleans and up to four overlapping
FP64 row-sum/subtract/abs temporaries of size `S` are the profile audit temporaries) and host mirrors
(incl. int32 IDs and the FP64 table), against `max_bytes`; rejections raise
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
pool-budget measurement policy and measured results for the current version are in
`../docs/species-root-profiles.md`; the historical 0.1.0 equal-layer results remain in
`../docs/daily-water-30year.md`.

## Tests

`python -m unittest tests.test_daily_water_growth`: single-plant depletion, two-plant competition,
dry cells, recharge/overflow budget, weekly growth budgets/radii, 7/8/14-day counters and pending
uptake, weekly (not daily) demand-map rebuilds, split-run equivalence, memory refusal before
generation, invalid scalars, CPU/GPU parity (skipped without a CC>=6 device; tolerance, not bitwise);
root profiles: default per-layer request sums, opposite profiles with a dry top/bottom layer
(analytical uptake from remaining water), all-dry/finite competing supply/weekly growth with a third
explicit profile, explicit uniform control against the equal-layer reference, invalid rows/IDs refused
before generation/transfer, live-ledger formula equality, CPU/GPU parity with heterogeneous water.

## Limitations

- Synthetic ecology: storage, recharge, demand, WUE and turnover are fixtures chosen for a bounded,
  auditable workload; results say nothing about real vegetation or soils.
- Uniform horizontal root weights per intersected cell; vertical weights are fixed per-profile
  fractions (no plasticity, no wetter-layer compensation), no evaporation, drainage, transport or
  storm hydrology. Default profiles are uncalibrated fixtures; for very large `L` the linear default
  rows are normalized in floating point and may fall outside the `1e-12` input tolerance.
- CPU daily kernels are serial `njit` loops (no `prange`); they are a standalone reference, not a
  tuned multicore baseline.
- Weekly growth uses NumPy/CuPy pointwise operations on preallocated candidate arrays rather than
  custom kernels; the daily hot paths are compiled CPU loops / RawModule GPU kernels.
- No production restart or checkpoint framework: `Simulation.step_days` split runs are equivalent,
  but state is not serialized.
- Admission peaks are ledger estimates (state + old/new graph + scratch allowance + mirrors), not
  observed allocator peaks; measured peaks come from the independent profiler in the linked reports
  (`../docs/species-root-profiles.md` for the current version, `../docs/daily-water-30year.md` for 0.1.0).
