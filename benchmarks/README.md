# Sparse circle-to-grid geometry benchmark (`benchmarks/spatial_mapping.py`)

A standalone synthetic numerical benchmark. It is **not** an ecological, hydrological or physical model:
there is no wind, water, growth or mortality solver, and no scientific allocation law is encoded.

## What it measures

1. **Root graph build** - each plant (circle, `root_radius`) is mapped to every square cell it
   *positively* overlaps (closest-point distance strictly less than the radius; tangent-only contact is
   not an edge). Candidate cells are the clipped bounding box of the circle, so there is no
   all-plants-by-all-cells search. The result is a plant-major CSR graph: `int64` offsets/counts,
   `int32` cell indices (guarded by `nx*ny <= 2^31-1`), `float64` weights. Weights are **uniform** over
   each plant's intersected cells and sum to 1; they are **not area weighted**. Sub-cell roots still map
   (at least one cell per plant inside the domain) even when no cell centre lies inside the circle.
   Vertical weights are stored factorized as a uniform `1/L` (a **separability assumption**); no dense
   plant x layer x cell tensor is formed. Any positive layer count is supported.
2. **Layered reduction** - a synthetic unit demand per plant is scattered equally over its edges and
   `L` layers into `float64[L, ncells]`. CPU: Numba `njit(cache=True)`; GPU: custom FP64 `atomicAdd`
   (compute capability >= 6 required) with explicit zero-init. This measures traversal/contention only.
   The total allocated demand is audited against the plant count with explicit tolerances.
3. **Canopy projection** - circle cover sampled with fixed 8x8 sub-cell midpoints per bounding-box cell,
   accumulated and capped at 1. This is coverage only: not an exact union and not area conserving.
   Plants missed entirely by sampling (small circles in coarse cells) are counted and reported, never
   silently discarded. The root predicate does not depend on canopy sampling.

## Admission

Before any large allocation a byte estimate (inputs + graph + grid outputs) is checked against
`max_bytes`, and after the count pass the edge total is checked against `max_edges` and the estimate
is re-checked. A rejected case is reported as `status: rejected` and never truncated; other cases
continue. No resource limits are raised and there are no hidden population caps.

## CLI

```
python benchmarks/spatial_mapping.py --cases-json cases.json --output result.json --backend cpu|gpu|both --repeats 5
```

`cases.json` is a list of objects; `layers` defaults to 2, all other fields are required:

```
[{"name": "small", "nx": 200, "ny": 200, "dx": 0.5,
  "grass_count": 20000, "shrub_count": 2000, "grass_diameter": 0.3, "shrub_diameter": 2.0,
  "root_multiplier": 1.5, "pattern": "clumped", "seed": 1, "layers": 3,
  "max_edges": 50000000, "max_bytes": 2000000000}]
```

`pattern` is `uniform` or `clumped`. Clumped is a deliberate synthetic contention fixture: at most 8
seeded gaussian cluster centres with sigma = 0.02 * min(width, height), clipped to the domain boundary
(`cluster_count` and `cluster_sigma` are echoed in the result metadata). Neither pattern is an
ecological spatial pattern. Host input is generated once per case and shared by CPU and GPU. Boundary clipping is
allowed; root weights renormalize over retained cells. Cases are validated upfront; invalid cases are
recorded with `status: invalid` (exit code 2). Audit/parity failures or exceptions give exit code 1.
The cheap pre-count byte guard runs before any host array generation or device transfer. If a
requested GPU is unavailable the case is `partial` (CPU record retained) or `unavailable` and the
exit code is 3; such cases are never reported as passed.

## Output

Each case records `case_wall_seconds` (host generation/validation plus all requested backends including
startup), which is distinct from kernel timings; single-thread CPU vs GPU kernel medians are a kernel
comparison, not a scientific speedup. Per backend: startup costs (Numba warm-up, CUDA context init,
kernel compile possibly served from the CuPy cache, and a once-per-process fixed small-fixture
H2D->build->reduction->canopy->D2H GPU warm-up, none of which is a true cold start) reported separately;
H2D and D2H transfer times; synchronized whole-build wall time; warmed repeated reduction and canopy
timings (raw repeats + median; CPU `perf_counter`, GPU CUDA events with synchronize); dtype/nbytes
ledger; `ru_maxrss` labelled as a cumulative process peak; CuPy pool used/total sampled at stage
boundaries (not a true peak). The graph stays resident on the device during timed repeats and
validation copies are outside kernel timing. If the GPU is unavailable the exact exception is recorded
and the backend is marked `unavailable`; it is never reported as passed. CPU/GPU agreement is checked
with tolerances (`rtol=1e-9, atol=1e-12` for fields, exact for CSR structure) because atomics are
order-nondeterministic.

## Tests

```
python -m unittest discover -s tests -v
```

Tests are small geometry oracles (corner/centre sub-cell roots, tangent exclusion, boundary clipping,
shared cells, CSR normalization, layered sums, deterministic generation, capacity refusal, canopy
aliasing). The CPU/GPU parity test skips explicitly when no capable device exists.

## Scope limitations

See `SCOPE_LIMITATIONS` in the module; every result JSON embeds the same list. CPU loops are
single-threaded compiled code; the benchmark does not claim exact area conservation of sampled cover,
physical water competition, periodic geometry, or bitwise CPU/GPU equality.
