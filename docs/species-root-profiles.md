# Species-specific root-layer profiles

The daily-water/weekly-growth benchmark now uses configurable vertical rooting
profiles for individual plants on CPU and GPU. Grasses can favor surface water,
shrubs can favor deeper water, and every profile retains access to every layer.
The sparse horizontal root graph and finite-water accounting are preserved.

## Configuration and update order

Fractions are ordered from top to bottom:

| Layers | Grass | Shrub |
|---|---|---|
| Two | 80%, 20% | 25%, 75% |
| Three | 70%, 20%, 10% | 15%, 25%, 60% |

These defaults are illustrative benchmark parameters, not calibrated species
measurements. [Jornada uptake candidates](candidate-uptake-parameters.md) now
record observed grass/mesquite depth partitioning and fitted profiles as inactive
potential parameters. Conversion needs physical layer boundaries and a policy
for uptake below the modeled bottom. Set `root_layer_fractions` in a case to
override the current defaults:

```json
"root_layer_fractions": {
  "grass": [0.70, 0.20, 0.10],
  "shrub": [0.15, 0.25, 0.60],
  "deep_species": [0.05, 0.15, 0.80]
}
```

Each row must contain exactly one finite, strictly positive number per layer
and sum to one within `1e-12`. Invalid rows are rejected rather than silently
normalized. Thus zero roots in a layer are deliberately excluded from this
initial contract.

For generated populations, grass/shrub profiles follow the initial plant class.
The Python API also accepts `inputs["root_profile_id"]`, a one-dimensional
integer array with one ID per plant. IDs index the named table in insertion
order: 0, 1 and 2 in the example. Explicit IDs are authoritative and support
additional species independently of canopy size. Grass-only or shrub-only
fixtures need only their own named profile when explicit IDs are absent.

Profile membership and vertical fractions stay fixed during growth. Every
seven model days, actual water uptake updates biomass; canopy/root radii, the
horizontal root connections and the cell/layer demand cache are rebuilt.
Water limitation and plant uptake are computed daily. The weekly clock continues
across year boundaries; a 30-year run has 10,950 days, 1,564 growth updates,
1,565 maps including initialization, and two days of pending uptake.

## Water allocation and memory

For plant `p`, connected cell `j` and layer `l`, potential demand is
`Dp[p] * horizontal_weight[p,j] * fraction[profile_id[p],l]`.
Requests are summed into the existing layer/cell demand map. Each day the
available water caps total uptake in each layer/cell; plants gather exactly
their requested shares times that same acceptance factor. The request and gather
use identical profile weights, preserving plant/grid flux agreement.

A dry layer supplies nothing. The fixture does not shift unmet demand to wetter
layers. Fractions weight potential uptake; they are not volumetric root density,
which would also require layer thickness and rooted volume.

The implementation stores one int32 ID per plant and an FP64 `S × L` profile
table. It creates neither species-by-grid rasters nor edge-by-layer tensors, and
removes the former horizontal demand raster. At fixed graph size, the persistent
change from v0.1 is `4P + 8SL - 8N` bytes. Exact persistent array storage including
one horizontal graph is:

```text
40LN + 188P + 8SL + 12H + 8 bytes
N = grid cells; P = plants; L = layers; S = profiles; H = horizontal connections
```

Host mirrors, simultaneous old/new graphs, scratch, allocator caches and runtime
libraries require additional memory. Admission includes profile IDs/table and
bounded profile-audit scratch. A scalar lower-bound guard rejects impossible
states before population-sized edge-bound temporaries are allocated. Weekly
map replacement remains transactional; no population is discarded on refusal.

GPU state remains resident during daily uptake. Initial H2D bytes comprise five
geometry arrays, profile IDs/table and WUE traits: `52P + 8SL`. The profile copy
has a separate synchronized timing record. Geometry-copy timing and total
initialization are reported separately; WUE-copy time is included in the latter.
Weekly checks and report boundaries still introduce synchronization.

## Verification and measurements

Final CPU and actual-GPU suites each pass 47 tests, with respectively three
and one expected backend skips. Independent analytical checks cover dry upper
and lower layers, finite competing supply, all-dry cells, an explicit third
profile and a uniform reference. Invalid inputs, early admission refusal,
weekly growth, split-run equivalence and heterogeneous-water CPU/GPU parity are
also tested. The completed 30-year CPU/GPU reference agrees on calendar counts,
water and biomass within the recorded floating-point tolerances.

All rows below are complete 10,950-day runs, performed in fresh processes.
They retain every plant and perform every scheduled weekly rebuild. Populations
come from the [Jornada capacity fixtures](jornada-capacity.md): 36,305 geometric
individual equivalents per hectare in the June example, 59,483 over the larger
512² domain, and a constructed dense-grass bound of 547,588 per hectare. These
are capacity fixtures rather than a census or a measured joint cover maximum.

Hardware/environment match the historical benchmark: one selected idle NVIDIA
GTX 1080 Ti, Intel Core i9-7900X CPU reference using one compiled thread, Python
3.12.3, NumPy 2.5.3, Numba 0.67.0 and CuPy 14.2.0. No environment or package was
created. Numerical source and the measurement wrapper stayed frozen.

| Scenario (0.25 m cells) | Plants | Layers | Root radius | 30-year loop | Whole process | Live peak | Reserved peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 ha, 400² | 36,305 | 3 | 3.0× canopy | 61.7 s | 63.5 s | 46.6 MiB | 1429.2 MiB |
| 1 ha, 400² | 36,305 | 3 | 5.0× canopy | 123.1 s | 124.8 s | 75.3 MiB | 1425.4 MiB |
| 1 ha, 400² | 36,305 | 2 | 5.0× canopy | 86.6 s | 88.1 s | 70.4 MiB | 1421.6 MiB |
| 1.6384 ha, 512² | 59,483 | 3 | 5.0× canopy | 167.0 s | 168.6 s | 124.0 MiB | 1428.7 MiB |
| Dense grass, 1 ha, 400² | 547,588 | 3 | 3.0× canopy | 127.3 s | 129.2 s | 194.1 MiB | 352.1 MiB |
| Dense grass, 1 ha, 400² | 547,588 | 3 | 5.0× canopy | 261.5 s | 263.4 s | 244.8 MiB | 495.6 MiB |

Loop time includes daily accounting and weekly growth, admission, graph
replacement and demand-map scattering; annual audits are excluded. Whole-process
time includes imports, startup, initialization, audits and output. Each figure
is one measured run on this workstation, not an extrapolation or a performance
guarantee.

| Scenario | Daily blocks | Weekly updates | Wall/model day | Model years/wall minute |
|---|---:|---:|---:|---:|
| 1 ha, 400², 3 layers, 3.0× roots | 40.7 s | 21.0 s | 5.64 ms | 29.2 |
| 1 ha, 400², 3 layers, 5.0× roots | 90.2 s | 32.9 s | 11.25 ms | 14.6 |
| 1 ha, 400², 2 layers, 5.0× roots | 60.5 s | 26.1 s | 7.91 ms | 20.8 |
| 1.6384 ha, 512², 3 layers, 5.0× roots | 119.8 s | 47.1 s | 15.25 ms | 10.8 |
| Dense grass, 1 ha, 400², 3 layers, 3.0× roots | 97.3 s | 30.0 s | 11.63 ms | 14.1 |
| Dense grass, 1 ha, 400², 3 layers, 5.0× roots | 203.8 s | 57.6 s | 23.88 ms | 6.9 |

The one-thread CPU reference for the first row took 128.4 s
of integration and 129.6 s for the whole process, with
381.7 MiB peak host RSS. Its GPU integration was
2.08× faster in these runs. GPU host
processes peaked at 487.9–680.0 MiB RSS;
RSS includes Python, libraries, compilation and host mirrors.

The equal-layer compatibility control also ran all 30 years. It matches the
historical v0.1 calendar, final connection count and water/biomass totals within
tolerance, and took 24.2 s with
49.3 MiB live GPU peak. This is close to the historical 23.8 s equal-layer result. The weighted cases
are slower in these measurements and follow different water-competition,
growth and graph trajectories. Earlier equal-layer times should therefore not
be used as estimates for the weighted configurations. These single runs do not
identify the cause of the runtime difference or isolate a stable percentage
implementation overhead; kernel and work-distribution profiling would be needed.

All eight full runs pass annual and final conservation/profile/calendar checks.
Maximum absolute final residuals were 8.98e-12 m³ for the
water budget, 2.91e-11 m³ for plant/grid uptake agreement,
and 2.91e-11 kg for biomass. No population was truncated,
no state was discarded and no resource limit was raised.

### Memory measurement policy

The same 10-million-connection and 1,500,000,000-byte limits remain in force.
The latter also bounds the CuPy pool. **Live peak** comes from malloc/free hooks
and includes observed temporaries; **reserved peak** is sampled after weekly
updates and includes cached free blocks. These are distinct pool measurements,
excluding the CUDA context and other allocators; they are not a trace of physical
VRAM residency. Pool reservations can remain much larger than live arrays.
Host RSS and the conservative admission ledger are reported separately. All
hook allocations were released at process end, with no unknown free events.

### Reproduction and evidence

Use the existing environment and select an idle GPU explicitly. This repeats
the first measured case with the original memory cap:

```sh
CUDA_VISIBLE_DEVICES=GPU-b02858c1-29c3-9ea9-33b1-831c33f8cf75 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
NUMBA_CACHE_DIR=/tmp/leaf-root-numba-cache CUPY_CACHE_DIR=/tmp/leaf-root-cupy-cache \
CUDA_CACHE_PATH=/tmp/leaf-root-cuda-cache \
../MAPLE/.venv-cupy/bin/python -B \
  agent_handoffs/tasks/species_root_profiles_20261031/profile_daily_bounded.py \
  --cases-json agent_handoffs/tasks/species_root_profiles_20261031/case_sets/june_hectare_3layers_roots3.json \
  --output /tmp/leaf-species-roots.json --backend gpu --years 30 --report-every 365

../MAPLE/.venv-cupy/bin/python -B -m unittest discover -s tests -v
```

Raw outputs are in the [result directory](../benchmarks/results/species_root_profiles_20261031/).
The [independent result verification](../agent_handoffs/tasks/species_root_profiles_20261031/result_verification.json)
contains exact timing/memory values and comparison outcomes. The
[task evidence](../agent_handoffs/tasks/species_root_profiles_20261031/) records
immutable baseline snapshots, exact source hashes, the real Claude session,
corrections, separate stdout/stderr and process launch/exit receipts. The
[measurement manifest](../agent_handoffs/tasks/species_root_profiles_20261031/measurement_source_manifest.json)
identifies the executed numerical sources and wrapper.

## Scope

This remains an isolated synthetic performance benchmark. Root fractions,
recharge, storage, demand, WUE, turnover and allometry are fixtures. Explicit
profile IDs control rooting; WUE still uses the benchmark's two initial canopy
size classes. No species parameter library, adaptive rooting, evaporation,
drainage, climate, recruitment, mortality, serialized restart or coupled
MAPLE/SYRUP/MASY event solver is added.

The [benchmark guide](../benchmarks/daily-water-growth.md) documents the API and
commands. The [historical report](daily-water-30year.md) retains the equal-layer
v0.1 measurements; its source snapshots and results remain preserved.
