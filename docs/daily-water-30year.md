# Daily water uptake and weekly growth: 30-year benchmark

> Historical benchmark: these measurements describe the equal-layer v0.1 implementation.
> The [species-profile follow-up](species-root-profiles.md) records the subsequent
> implementation and measurements. Original source is preserved in the
> [task baseline](../agent_handoffs/tasks/species_root_profiles_20261031/baseline_sources/).

This isolated LEAF benchmark measures daily water accounting with spatial
individual plants and weekly biomass, root-geometry and root-map updates.
It does not run MAPLE, SYRUP or MASY. The purpose is to measure the proposed
workload before selecting scientific growth and hydrology formulations.

## Workload and fixtures

Every completed long run must execute 10,950 daily steps (30 fixed 365-day
years), 1,564 weekly growth updates and 1,565 root maps including the initial
map. The last two days of uptake remain pending for the next growth update.
Year boundaries do not reset the weekly clock. No daily field trajectories
are stored, and no individuals are dropped to reduce the workload.

The [Jornada capacity assessment](jornada-capacity.md) supplies the population
fixtures: approximately 36,305 individuals per hectare for its June example
(15 cm grasses and 2 m shrubs), and 547,588 for a constructed dense-grass bound
(10 cm grasses). These are cover-derived geometric equivalents, not a plant
census; the dense bound is not a measured joint maximum. A 512 × 512 grid at
0.25 m covers 128 × 128 m and has 59,483 individuals at the June densities.
Root radii use the user's working range of three to five times canopy radius.

## Deliberately simple accounting

Water is stored in cubic metres in two or three layers. Total available
capacity is 150 mm, split equally between layers and initially half full.
Synthetic recharge adds 10 mm every 14 days starting on day zero; overflow
is recorded separately. Thus one hectare starts with 750 m³ and receives
78,300 m³ through 783 recharge pulses over the 30-year run. This recharge
fixture is not a rainfall, infiltration or drainage solver.

Potential daily uptake is 3 mm times each plant's canopy area. A sparse map
distributes that demand uniformly across intersected root cells and equally
among layers. Within a week, geometry and potential demand stay fixed, so
requested cell uptake is cached. Each day, available water limits uptake;
plants gather their accepted shares using the same cell acceptance factors.

Weekly growth converts actual uptake to biomass with a synthetic water-use
efficiency of 0.002 kg dry matter/kg water for grass and 0.0015 for shrubs.
Explicit turnover uses a 0.003/day rate. Canopy area is biomass divided by
1 kg/m²; root radius follows the chosen multiplier. These are performance
fixtures, not calibrated ecological parameters. Biomass, radii, root maps and
demand must update together after admission succeeds.

The run omits evaporation, a drainage solver, climate, recruitment, mortality,
event injury, canopy export and sediment transport. Future integration must
obtain soil properties from the shared MASY surface. Uniform layer weighting
and uniform horizontal root weights are additional simplifying assumptions.

## Measurement and verification

Numba supplies compiled CPU loops; CuPy custom kernels supply the GPU path.
GPU daily state remains on the device. Weekly map admission and sparse audit
reports introduce synchronization; daily full-state transfers are excluded
from the algorithm. Root connections are stored horizontally, with vertical
weights factored by layer rather than duplicating the graph per layer.

Report integration wall time, daily-block time, weekly growth/mapping time,
setup and total process time separately. Integration must include the weekly
rebuilds and their admissions. Report persistent bytes, the observed live
CuPy allocation peak including rebuild temporaries, reserved pool memory and
host process peak RSS separately. CuPy pool instrumentation excludes CUDA
context memory and other allocators; RSS includes Python, libraries and JIT.

Independent checks compare plant uptake with grid water removal, close water
and biomass budgets, verify calendar counts, and compare small CPU/GPU cases.
Resource limits remain 10 million horizontal connections and 1.5 billion
accounted bytes per large fixture; rejected runs must be identified explicitly.

## Results: complete 30-year runs

All rows below are actual 10,950-step runs, not extrapolations. Each case ran in a fresh process. Hardware: one NVIDIA GTX 1080 Ti (11 GiB listed memory), and an Intel Core i9-7900X CPU reference using one compiled thread. Existing Python 3.12.3, NumPy 2.5.3, Numba 0.67.0 and CuPy 14.2.0 were used.

The GPU table uses the existing 1,500,000,000-byte budget for the CuPy pool as well as the array/scratch admission checks. **Live peak and reserved peak are different measurements:** the first includes all observed temporary pool allocations; the second includes cached free blocks. Pool limits/counters exclude CUDA context memory and other allocators. Whole-process time includes imports, startup, initialization, integration, annual audits and output; loop time includes all weekly growth/maps but excludes annual audits.

| Scenario (0.25 m cells) | Plants | 30-year loop | Whole process | Observed live peak | Sampled reserved peak |
|---|---:|---:|---:|---:|---:|
| 1 ha, 400², 3 layers, roots 3× | 36,305 | 23.8 s | 25.4 s | 50.3 MiB | 1426.9 MiB |
| 1 ha, 400², 3 layers, roots 5× | 36,305 | 52.8 s | 54.4 s | 83.2 MiB | 1429.8 MiB |
| 1 ha, 400², 2 layers, roots 5× | 36,305 | 43.2 s | 44.8 s | 77.1 MiB | 1429.9 MiB |
| 1.6384 ha, 512², 3 layers, roots 5× | 59,483 | 95.2 s | 96.8 s | 136.6 MiB | 1428.3 MiB |
| Dense grass, 1 ha, 400², 3 layers, roots 3× | 547,588 | 27.8 s | 29.7 s | 197.8 MiB | 548.8 MiB |
| Dense grass, 1 ha, 400², 3 layers, roots 5× | 547,588 | 51.5 s | 53.4 s | 254.8 MiB | 820.5 MiB |

GPU host-process RSS peaked at 488–541 MiB. RSS includes Python, libraries, JIT and host mirrors; add it separately to device allocation estimates. These are single-run measurements on the existing workstation, suitable as capacity estimates rather than statistical performance bounds.

### Wall time versus model time

The daily and weekly columns are accumulated synchronized wall times over all 30 model years. Weekly time includes growth, invariants, admission, root-map replacement and demand-cache rebuilding.

| Scenario | Daily blocks | Weekly updates | Annual audits | Wall/model day | Model years/wall minute |
|---|---:|---:|---:|---:|---:|
| 1 ha, 400², 3 layers, roots 3× | 14.6 s | 9.1 s | 0.10 s | 2.17 ms | 75.8 |
| 1 ha, 400², 3 layers, roots 5× | 38.1 s | 14.7 s | 0.10 s | 4.82 ms | 34.1 |
| 1 ha, 400², 2 layers, roots 5× | 28.6 s | 14.6 s | 0.12 s | 3.94 ms | 41.7 |
| 1.6384 ha, 512², 3 layers, roots 5× | 73.5 s | 21.6 s | 0.15 s | 8.69 ms | 18.9 |
| Dense grass, 1 ha, 400², 3 layers, roots 3× | 19.4 s | 8.5 s | 0.10 s | 2.54 ms | 64.7 |
| Dense grass, 1 ha, 400², 3 layers, roots 5× | 38.9 s | 12.6 s | 0.12 s | 4.70 ms | 34.9 |

### Compiled CPU reference

CPU reference runs use identical inputs, numerical fixtures and the full continuous calendar. The speed ratio compares the measured CPU loop with the GPU loop under the pool budget.

| Scenario | 30-year loop | Whole process | Host RSS peak | CPU/GPU wall ratio |
|---|---:|---:|---:|---:|
| 1 ha, 400², 3 layers, roots 3× | 139.6 s | 140.9 s | 385.7 MiB | 5.9× |
| 1 ha, 400², 3 layers, roots 5× | 263.5 s | 264.7 s | 438.7 MiB | 5.0× |

### Allocator behavior matters

The original runs used CuPy's default unlimited pool cache. Growing root graphs left many differently sized free blocks cached. Low live-array memory therefore did not establish a small device reservation. Those original results are preserved separately. The repeat changes only cache retention through `MemoryPool.set_limit(size=case['max_bytes'])`; it refuses an inherited lower nonzero limit. No individual, water, biomass, connection or pending uptake is discarded, and the declared cap is unchanged.

| Scenario | Default loop | Limited-pool loop | Default reserved counter | Limited reserved counter |
|---|---:|---:|---:|---:|
| 1 ha, 400², 3 layers, roots 3× | 23.7 s | 23.8 s | 3015.1 MiB | 1426.9 MiB |
| 1 ha, 400², 3 layers, roots 5× | 52.7 s | 52.8 s | 7666.0 MiB | 1429.8 MiB |
| 1 ha, 400², 2 layers, roots 5× | 43.3 s | 43.2 s | 7659.9 MiB | 1429.9 MiB |
| 1.6384 ha, 512², 3 layers, roots 5× | 150.4 s | 95.2 s | 12467.3 MiB | 1428.3 MiB |
| Dense grass, 1 ha, 400², 3 layers, roots 3× | 28.1 s | 27.8 s | 548.8 MiB | 548.8 MiB |
| Dense grass, 1 ha, 400², 3 layers, roots 5× | 51.8 s | 51.5 s | 820.5 MiB | 820.5 MiB |

These are CuPy allocator counters, not a trace of resident VRAM. In particular, the default 512² run's counter exceeded the card's listed memory on this WSL system; it must not be interpreted as a physical VRAM measurement. The repeat enforces the pool's allocation budget. Runtime comparisons include normal run-to-run and host-load variation. Production coupling should explicitly budget persistent state and reuse scratch/graph buffers or manage unused cache at phase boundaries.

### Memory scaling

With `N` horizontal cells, `L` layers, `P` plants and `H` horizontal root connections, this prototype's persistent arrays occupy approximately `40*L*N + 8*N + 184*P + 12*H + 8` bytes, before host mirrors. The five layered FP64 fields include water, demand, acceptance scale and two cumulative accounting fields. Plant/candidate buffers and original inputs are included. Graph replacement briefly retains old and new maps; the admission ledger additionally reserves scratch and host mirrors. Those estimates are separate from the measured allocation peak in the table. Factorized vertical weights avoid storing a separate connection list per layer. Memory follows current state and graph size, not 30 years of field histories.

These figures cover this uptake/growth workload. Shared MASY soil properties, richer root/allometric traits, individual demography, canopy export and persistent MAPLE/SYRUP storage need their own additions. The measurements support feasibility for the specified grids and populations; they do not qualify a scientific ecohydrology model or establish full coupled-model runtime.

## Verification and provenance

Fourteen complete runs (six default GPU, six limited-pool GPU and two CPU) represent 420 model years. Every run retained its full population, completed the required calendar and passed all 30 annual audits. Independent checks reconcile array byte ledgers, source/wrapper hashes, launch/exit receipts, final water/biomass budgets and full-run CPU/GPU scalar agreement. Default versus limited-pool repeats match final water/biomass/flux totals and graph/calendar counters within FP64 tolerances; bitwise identity is not required. Small CPU/GPU tests compare individual arrays as well.

Largest absolute final residuals across all runs: water `1.3e-11 m³`, plant/grid uptake `1.46e-11 m³`, biomass `3.91e-11 kg`. No residual corrections or balancing fluxes were applied. The final 36-test suite passed in CPU-only and real-GPU executions; expected unavailable-backend skips are recorded in their logs.

[Verified results](../agent_handoffs/tasks/daily_water_30year_20261030/result_verification.json), [case fixtures](../agent_handoffs/tasks/daily_water_30year_20261030/benchmark_cases.json), [runtime identity](../agent_handoffs/tasks/daily_water_30year_20261030/runtime_identity.json), [frozen runtime source hashes](../agent_handoffs/tasks/daily_water_30year_20261030/source_after_correction_04.json), [bounded profiler](../agent_handoffs/tasks/daily_water_30year_20261030/profile_daily_bounded.py), [raw result directory](../benchmarks/results/daily_water_30year_20261030/). The task directory preserves Claude session records, exact prompts, corrections, stdout/stderr and per-run process identities/exits. Claude authored the benchmark; Codex reviewed, corrected through the handoff loop, independently executed and verified it. Root AGENTS.md and upstream models were preserved.

## Reproduce

From `~/leaf`, use the existing environment. Select an idle GPU UUID for the GPU command. The bounded profiler preserves the case's byte cap and does not install dependencies.

```sh
../MAPLE/.venv-cupy/bin/python -m unittest discover -s tests -v
../MAPLE/.venv-cupy/bin/python -m benchmarks.daily_water_growth \
  --cases-json agent_handoffs/tasks/daily_water_30year_20261030/case_sets/june_hectare_3layers_roots3.json \
  --output /tmp/leaf-30year-cpu.json --backend cpu --years 30
CUDA_VISIBLE_DEVICES=YOUR_IDLE_GPU_UUID ../MAPLE/.venv-cupy/bin/python \
  agent_handoffs/tasks/daily_water_30year_20261030/profile_daily_bounded.py \
  --cases-json agent_handoffs/tasks/daily_water_30year_20261030/case_sets/june_hectare_3layers_roots5.json \
  --output /tmp/leaf-30year-gpu.json --backend gpu --years 30
```
