# LEAF Phase 1, bounded part A — candidate report (Claude, implementation writer)

Scope honoured: writes limited to `leaf/uptake.py`, `leaf/roots.py`, `leaf/forcing.py`, `leaf/traits.py`,
`leaf/__init__.py`, `tests/test_uptake.py`, `tests/test_roots.py`, `tests/test_forcing.py`,
`docs/phase1-water-contract.md`, `report.md`. Nothing in `coupling` / `stepwat` / `runtime` / `canopy` /
`state` / `benchmarks` / existing tests / fixtures / `pyproject.toml` was touched. No growth law, no temperature
response, no driver. Tools used: Read / Edit / Write only.

## Files and behaviour

| file | what |
|---|---|
| `leaf/uptake.py` (new) | `RootMap.from_arrays` (validated immutable plant-major CSR, int64 IDs/offsets/resources, FP64 weights, general non-separable weights, inactive rows, duplicate/bool/float/overflow refusal before cast, owned frozen copies), `propose` → `UptakePlan` (Q, proposed = min(A, Q), f), `accept` → `UptakeFeedback` (clipped accepted, scale, delivered, unmet, S, closure), explicit memory admission (`estimate_map_bytes` / `estimate_plan_bytes` / `estimate_feedback_bytes` / `admit`, `AdmissionError` with scoped estimate), three backends (`numpy` reference, `numba` kernels, `cuda` RawModule with FP64 atomics), `backend_status`, `BackendUnavailableError`, `closure_tolerance_m3`, `acceptance_tolerance_rel`. Plain-Python kernels `_k_*` are the single authoritative per-edge definition (compiled by numba, mirrored in CUDA C, vectorised by NumPy). |
| `leaf/roots.py` (new) | `build_root_map(state, catalog, *, max_edges, max_bytes, backend, builder, statuses, layer_fraction_overrides)`: shared allometry via `canopy.derive_geometry`; horizontal law = existing compiled circle-overlap CSR (`benchmarks.spatial_mapping.build_root_graph_cpu`, numba) or arithmetic-identical NumPy reference, **labelled** as the simple uniform-overlap geometric approximation; species layer fractions or ID-keyed overrides; layer-major expansion admitted before allocation; non-square domains refused before allocation; selected statuses mapped consistently (zero-radius → inactive row). |
| `leaf/forcing.py` (new, stdlib only) | `TimestepRecord` (fixed-offset aware start/end, required temperature, optional PET mm/day, process kind gate), exact `Fraction` seconds/days, midnight and boundary splitting, `check_contiguous`, `records_from_weather_rows` (`date` YYYY-MM-DD, `tmean_c`), `parse_date`. |
| `leaf/traits.py` (edit) | `SpeciesCatalog.from_files(paths)`, `load_catalog` accepting one path or a sequence, provenance attributes `sources` / `species_sources` (+ `metadata()` keys `sources`, `species_sources`, `provenance_note`), constructor keywords `sources=None, species_sources=None`. Existing behaviour, validation, hash and `from_file` identity unchanged. |
| `leaf/__init__.py` (edit) | lazy exports `uptake`, `roots`, `forcing` (+ docstring); `import leaf` still imports nothing beyond NumPy. |
| `tests/test_uptake.py` (new) | required 75 % example, proportional multi-plant sharing (not equal litres, order independent), non-separable multi-cell/layer oracle, zero demand / zero supply / inactive rows / empty population, masked `A = 0` cell without renormalisation (residual unused), partial acceptance + tolerance (material excess refused, ulp-level counted), invalid arrays/metadata with input preservation and ownership, memory refusal before allocation (edge bound, `n_cells = 1e12`, byte bound, op-level budgets), random many-overlap closure vs. an independent dense oracle at FP64 tolerance, reordering invariance, plain-Python kernels vs. NumPy path, numba parity (skips only if numba unavailable), CUDA parity / no-implicit-transfer / device validation (skips only if CUDA unavailable). |
| `tests/test_roots.py` (new) | NumPy builder vs. brute-force overlap geometry (clipping, normalisation, law label), species fractions and individual overrides (+ invalid overrides), statuses / inactive rows / empty selection, non-square refusal, expanded-edge admission before allocation (stages), masked cell through uptake with fixed weights, custom general map, compiled-builder parity (skip only without numba); **multi-file catalog** tests (`TestMultiFileCatalog`: order-independent hash equal to the single-file fixture identity, provenance, `load_catalog` sequence, state compatibility, duplicates/incomplete/root_layers/units refusals, catalog_id rule). Placed here because the allowed test files are the three listed. |
| `tests/test_forcing.py` (new) | record validation (naive / DST-style zones / offset mismatch / kinds / gates / temperature / PET), exact fractions and partial days, leap day and year-boundary midnight splits, growth-boundary splits, contiguity (gap / overlap / reorder / offset change), weather-row conversion (leap date, padding, gaps, duplicates, required temperature, PET column). |
| `docs/phase1-water-contract.md` (new) | contract, API, units, backends, memory scope, root approximation label, forcing records, multi-file loading, native-authority boundary. |

## Key decisions and limits (for Codex review)

* Tolerances: `WEIGHT_SUM_TOL = traits.LAYER_FRACTION_TOL = 1e-9` (builder maps inherit the catalog's layer-sum
  tolerance, so a tighter value would refuse valid catalogs); acceptance excess `eps * (max edges per resource + 2)`;
  closure `eps * (2E + R + P + 8) * total`; delivered ≤ `D * (1 + 1e-9 + eps * (max edges per plant + 4))` with
  `unmet` / `S` clamped at that level only. All proportional to eps or the catalog tolerance; none enlarged.
* Resource indices are int64 (no int32 cell guard inside uptake; `roots` inherits the benchmark Grid's int32
  `nx*ny` guard for the compiled and numpy builders through `sm.Grid` only on the compiled path — the numpy
  builder has no such guard, R is bounded by admission instead).
* `cuda` maps may be loaded from host arrays (the one explicit transfer, validated on host first) or from CuPy
  arrays (validated on device by scalar reductions and P/R-sized temporaries). Plan / feedback inputs must be CuPy
  arrays; no implicit transfers, no fallback. GPU atomics make Q order-nondeterministic; parity is tolerance based.
* The map admission counts an `8R` int64 bincount scratch (resource edge counts, needed for the acceptance
  tolerance); for very large `L*N` with a small `max_bytes` this is what refuses first (by design, before allocation).
* `roots` horizontal builder: the compiled path's own admission estimate (`benchmarks.spatial_mapping.estimate_bytes`)
  includes a `(layers+1)*N*8` grid term that this part does not allocate; it is reported verbatim and may refuse
  slightly earlier than our own estimate. The NumPy builder uses an O(P) Python loop with vectorised cell tests
  (reference / fallback; production path is the compiled builder). Default `builder='compiled'` raises
  `BackendUnavailableError` without numba (explicit choice, no silent fallback).
* `from_files`: with several files `catalog_id` / `notes` are kept only when identical in every file; the
  Phase 0 fixture split into per-species files (same `catalog_id`/`notes`) merges to the **same hash** as the
  original file. `traits.metadata()` gained three keys; the existing `source` key is unchanged.
* `forcing` requires `datetime.timezone` (fixed offset) tzinfo; zoneinfo zones are refused by design.
* numba kernels are created by `njit(cache=True)(plain_function)` at first use; CUDA kernels compiled with
  `-fmad=false` to match CPU arithmetic (same convention as the existing benchmark).

## Tests: NOT RUN

I could not execute anything (Read/Edit/Write only). **All tests are reported as NOT RUN**: `tests/test_uptake.py`,
`tests/test_roots.py`, `tests/test_forcing.py`, and the existing `tests/test_phase0.py` (expected unchanged;
`traits.metadata()` now carries three additional keys which could matter only if an existing test compares the
whole dict). Nothing was installed, imported or compiled. I desk-checked the tests against the implementations
and fixed four defects found that way (exception types in `uptake`/`roots`, one forcing assertion). Residual risks
I could not verify without execution: numba acceptance of read-only (frozen) NumPy inputs in the kernels, CuPy
`repeat`/`bincount(weights=...)`/`unique` behaviour on the device validation path, exact bitwise agreement of the
NumPy builder with the compiled `_overlaps` predicate at tangent cases (the parity test asserts equality), and the
benchmark helper's admission estimate interplay with small `max_bytes` in `test_compiled_builder_parity`.
No tool, auth, network or rate failures occurred.
