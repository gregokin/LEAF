# Report: LEAF Phase 1 pure species demand and growth kernels (candidate)

Author: Claude (sole implementation writer, isolated candidate). Date 2026-11-11. Uncommitted.
**Nothing was executed.** I only had Read/Edit/Write; every test, import and compilation below is **NOT RUN**.

## Revision 2 (finite bounds, units, complete admission, backend naming) – ALL NOT RUN

Writes only to the same five files. Revision-1 Codex results (CPU 49 pass, real CUDA parity pass) are Codex's
independent checks, not mine; I executed nothing in this revision either.

1. **Non-finite ledger bounds.** Every backend (NumPy, Numba, CUDA) now treats a non-finite per-plant ledger
   reference, residual or tolerance as `bad_output`, so e.g. `mature = B = 1e308` (tolerance `16*eps*2e308 = inf`)
   is refused with `GrowthDataError` instead of certified. On the host the population scale/residual/tolerance must
   be finite and `|residual| <= tolerance`, otherwise `GrowthDataError` (no `population_closed=False` return). The
   normal plan is short-circuited to exactly 0 when `B-E >= mature` before the `(1 - (B-E)/mature)` product
   (equal to the former `max(0, ...)` for finite values; tolerances unchanged, no clipping of plans). Consequence
   (documented): live plants whose mass sums overflow the bound (about B > 1e307) are now rejected, which replaces
   the earlier success case in `test_overflow_refused`. Regression: `TestNonFiniteLedgerAllBackends` runs on NumPy,
   Numba and CUDA when available.
2. **Admission of all allocating helpers.** `species_response` and `integral_increments` now require `max_bytes`
   (optional `extra_reserve_bytes`) and admit new `memory_estimate` operations `'response'` (3 output + 9 scratch
   S-vectors) and `'integrals'` (2 output + 17 scratch S-vectors) plus the live tables BEFORE any array is created.
   Public `temperature_factor` is scalar-only (arrays refused, returns float); the vector math is the private
   `_tf_vector`, used after admission. `demand` forcing-temporary estimate raised from 48 S to 128 S bytes to cover
   response + rate arithmetic. Tests include a trap (`_response` patched to fail) proving refusal precedes work.
3. **Time basis renames (names only, no numerical change).** Trait `biomass_support_kg_year_per_m3`,
   `extra_growth_rate_per_year` (still [0, 1]); outputs `required/requested/actual_support_kg_year`,
   `maintenance_deficit_kg_year`; packet slot `sum_maintenance_deficit_kg_year`. Fixture, tests and docs updated.
   Old law/module/study records untouched; no earlier fit is activated.
4. **Backend naming.** Canonical `'cuda'` (`BACKENDS = ('numpy','numba','cuda')`); `'cupy'` accepted only as an alias
   canonicalised on entry (compile_species, memory functions). Device guards unchanged (exact device, no switching).

Remaining risks: syntax/typing errors (nothing run); the new 72 S / 136 S / 128 S scratch constants are my counts,
not measured; kernels' behaviour for B ≳ 1e307 is now refusal rather than success; the `'cupy'` alias is optional and
may be dropped.

(The sections below describe revision 1 and still apply except where superseded above; old names such as
`required_support_kg` in them are historical.)

## Files written (exactly the allowed set)

- `leaf/growth.py` – new pure module (law `stepwat_requested_support_v1`): catalog compilation, temperature /
  activity helpers, two demand laws, requested-water growth kernel, itemized memory model; NumPy reference,
  Numba kernels and fused CUDA RawKernels.
- `tests/test_growth.py` – unittests (list below).
- `examples/phase1/species.json` – complete uncalibrated two-species FIXTURE catalog (shallow grass, canopy_pet_v1,
  multiplier 3, layers [0.85, 0.15]; deep shrub, biomass_specific_pet_v1, multiplier 5, layers [0.25, 0.75]).
- `docs/phase1-growth.md` – equations, units, API, memory, limits.
- `report.md` – this file.

No other file was modified (old sources/tests/fixtures, `leaf/__init__.py`, roots/allocator/native bridge untouched).

## Source context actually read

`leaf/canopy.py`, `leaf/traits.py`, `leaf/state.py` (first 150 lines), `leaf/__init__.py`, `leaf/stepwat.py`,
`docs/stepwat-growth.md`, `examples/phase0/species.json`. Guessed paths that did not exist (honest tool errors, no
content assumed): `leaf/species.py`, `tests/test_traits.py`, `leaf/coupling.py`, `docs/phase0.md`,
`tests/test_stepwat.py`, `tests/test_canopy.py`. No saved terrain/weather payload existed and none was invented.

## Design decisions to review

- Parameters are read from `growth.traits` and `water_use.traits` inside each species record (the generic Phase 0
  validator already preserves arbitrary finite sections; no catalog schema change was needed). Unknown keys refuse.
- `species_index` is `int32` (matches the older kernel precedent), `live` bool, all float64; exact dtypes/types,
  1-D C-contiguous, no implicit conversion/copy. The CUDA path rejects NumPy arrays and other-device arrays.
- Demand is split into an S-sized host "rate" `k[s]` (all forcing math in float64 on the host) and a per-plant
  multiply in the kernel, so all backends do the identical per-plant arithmetic.
- Tolerance choices (documented, easily adjusted): over-delivery `U <= D*(1+1e-9)` (relative; user bound
  <= 1e-6); per-plant ledger bound `16*eps*(B+Bnew+gains+loss)`; `Aint <= 7/365*(1+1e-12)`. I did not have the
  allocator's actual tolerance definition (relative vs absolute), only the statement "1e-9"; Codex should confirm.
- Validation lives in the kernels and is reported through a packet; on failure outputs are discarded and nothing is
  returned (inputs are never mutated). The grow outputs are views of one `[12, P]` buffer (96 B/plant).
- Numba kernels are defined lazily inside a factory without `cache=True` (a separate cached module was not an
  allowed file), so the first call pays JIT time.
- CUDA kernels use `--fmad=false`, a grid-stride loop with at most 256 blocks, block shared-memory tree reductions and
  a host reduction of the `[blocks, K]` partial packet (deterministic, bounded).
- Device guard: `_current_device_id()` / `_array_device_id()` / `_cupy()` are small module functions so tests can
  mock the guard without a GPU.
- `live_fraction == 0` with the biomass-specific law can produce demand and surplus extra growth (required = 0);
  documented, not special-cased.

## Tests written – ALL NOT RUN

`tests/test_growth.py` (unittest; Numba/CUDA tests skip when unavailable):
- hand calculation (full plan, 75 % delivery = 0.75 of full not 0.5625, Fplan < 1, dry / zero demand, dormancy,
  year-end loss + ledger, above-mature, cap to mature, per-species differences, inactive plants, over-delivery
  tolerance, aggregate packet);
- temperature cardinal points, monthly activity by actual month, invalid month/temperature, integral increments;
- both demand laws and units, arbitrary canopy area, no second live-fraction, zero PET/dormancy/dead plants,
  invalid forcing, invalid content and overflow, dtype/shape/type refusal, inputs unchanged on failure;
- compile errors: rate 295 / > 1, missing/unknown/unimplemented/old laws, missing/unknown/invalid traits, Phase 0
  catalog rejected here but still loadable, backend/device arguments, budget refusal before parsing;
- admission: refusal before any work, exact byte threshold, itemized estimate, real-allocation (tracemalloc) vs
  estimate;
- NumPy/Numba parity, real CUDA parity (skipped without CuPy/device), mocked current-device and array-device guards.

## Known risks / things most likely to need a correction (because nothing was run)

1. Syntax/typing errors in `growth.py` or the test file are possible; no interpreter was available.
2. Numba typing (e.g. `int32` index unified with literal `0`) and CUDA/NVRTC compilation of `_CUDA_SRC` are
   unverified; `fin()` is used instead of `isfinite` deliberately.
3. The tracemalloc-vs-estimate test and the near-overflow cases in `test_overflow_refused` (subnormal arithmetic)
   may be fragile and might need loosening.
4. `ndarray.device` exists in NumPy 2; the CUDA test helper `to_host` relies on `cp.asnumpy` accepting NumPy arrays.
5. Memory byte constants (128 / 512 B per chunk element) are my count of live NumPy temporaries, not measured.
6. Real CPU/GPU memory, timing and numerical qualification are left to Codex.

## Tool errors

Only the "file does not exist"/EISDIR read errors listed above. No write errors.
