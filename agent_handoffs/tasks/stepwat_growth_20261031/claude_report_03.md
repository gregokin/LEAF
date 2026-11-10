Revision 3 corrections are complete; nothing further is needed from the files, and no tool requests remain. Report follows.

## Revision 3 report — task stepwat_growth_20261031 (writer: Claude)

Scope honoured: only the three listed corrections; equations, trait schema, fixed root fractions, native water authority, weekly/year-end transaction, memory accounting and legacy defaults untouched. `leaf/coupling.py`, `leaf/__init__.py`, legacy tests and `spatial_mapping.py` were not modified in this revision.

### Changed files

**`leaf/stepwat.py`**
- `validate_temperature_factor` is now strictly host-only and shape-first:
  - NumPy arrays: `ndim`/`shape[0] == S` checked, then dtype kind, *before* any conversion; valid arrays still go through one `ascontiguousarray(float64)` of S floats (identical numerics for valid input).
  - Non-NumPy values exposing `.get()`, `__cuda_array_interface__` or `.device` are refused with `ValueError` **without calling `.get()`** (no transfer).
  - Host sequences: `len == S` checked before touching elements, then every element validated as a real scalar (`_unit`) before a single `np.array` of S floats — nested/oversized/malformed input never reaches a bulk conversion. Scalars unchanged (`np.full`).
- Report: `totals` → `domain_totals_kg` (explicit `unit: kg, domain sum over all individuals`), new `per_individual_kg` block (population, B_min/B_max/B_mean, E_max/E_mean); `biomass_unit` states that arrays are kg per individual and totals are domain sums.

**`benchmarks/daily_water_growth.py`** — STEP-mode `plant_summary()['biomass']`: `unit` now says domain sum explicitly (only `min` per individual), plus `scope: 'domain_sum'`, `population`, `per_individual_mean`. Legacy (non-STEP) output unchanged.

**`tests/test_stepwat.py`**
- Trait/forcing pure test: accepts scalar/sequence/NumPy `[S]` forms; rejects an expanded bad set including fake device-likes whose `.get()` raises `AssertionError`, a 4096×4096 wrongly shaped array, right-length nested lists, mixed/bool/None elements, `range`, `object()`. All bad cases run with `np.asarray/ascontiguousarray/array/full` patched to raise, proving no bulk conversion; a real CuPy array is refused when a GPU is present.
- Coupled bad-forcing test: adds the fake device-like, nested and large-matrix inputs; asserts native step and `accumulate_temperature` not called, state object identity preserved, counters/water/uptake/temperature integral untouched, with NumPy conversions patched to raise.
- `test_weekly_old_new_peak_counts_growth_state`: now patches `CoupledSimulation._vegetation_admission` at class level **around construction and the week**, so the Simulation binds the recording callback; asserts the exact stage sequence `init_scalar_bound, init_bound, init_generated, rebuild_bound, rebuild_actual`, takes `init_total` from the captured `init_generated` ledger, and retains all old+new-peak, before-builder, exact-boundary (`T_bound - sw_bytes` refused / `T_bound` admitted) and retry assertions. No limits raised, no physics altered.
- Report-key tests updated to `domain_totals_kg` / `per_individual_kg` / `scope`.

**`docs/stepwat-growth.md`** — host-only forcing contract; a "Units of reported values" section separating per-individual arrays from domain-sum summaries; test commands using the existing interpreter with `NUMBA_CACHE_DIR=/tmp/leaf_numba_cache`.

### Limitations / not done
- Nothing was executed; no claim that tests pass. Codex runs them.
- The CuPy-array refusal test does not patch `cupy.ndarray.get` (extension type); non-transfer is proven via the fake device-like objects instead.
- The weekly-peak test relies on the recorder being bound at construction; the fixture asserts the exact five-stage sequence, so any future added ledger stage would need that list updated.
- Changes remain uncommitted; no Git metadata created.
