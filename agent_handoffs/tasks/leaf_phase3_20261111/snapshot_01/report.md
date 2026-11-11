# LEAF Phase 3 candidate report — simple seed establishment and nearby vegetative daughter plants

Task leaf_phase3_20261111, baseline 5e887e66415f0522771ea789654b8345226d2496. Implementation by Claude in the isolated
candidate; left uncommitted. **ALL checks are NOT RUN** (no execution tool was available): nothing below has been imported,
compiled, tested or run; Codex must execute everything. Expect that untested code may contain defects.

## Files written (exactly the authorised set)

`leaf/recruitment.py` (new), `leaf/interevent.py`, `leaf/checkpoint.py`, `leaf/demography.py`, `leaf/__init__.py`,
`tests/test_recruitment.py` (new), `tests/test_interevent.py`, `tests/test_demography.py`, `examples/phase3/species.json` (new),
`examples/phase3/run_recruitment.py` (new), `docs/phase3-reproduction.md` (new), `docs/maple-leaf-integration.md`, `README.md`,
`report.md`. Nothing else was touched (no native bridge, allocator, roots, growth, canonical CSV / PlantState, species parser).

## New APIs

* `leaf.recruitment`: `compile_species(catalog, *, max_bytes=None, extra_reserve_bytes=0)` -> `CompiledReproduction`;
  `PatchConfig(patch_size_m)`; `check_limits`; `stream_key` / `keyed_uniform_py`; `patch_geometry`; `memory_estimate`;
  `preflight_bytes`; `evaluate(table, patch, domain, current_surface, *, live_ids, live_species, live_age_days, live_B, live_E,
  live_x, live_y, max_existing_id, n_total_records, seed, counter, max_new_plants, max_total_records, max_attempts, max_bytes,
  extra_reserve_bytes=0)` (pure); errors `ReproductionError`, `ReproductionBudgetError`, `ReproductionLimitError`.
* `IntereventSession(..., reproduction_patch_size_m=None)`; `apply_reproduction(year, current_surface, *, max_new_plants,
  max_total_records, max_attempts, interval_id=None)`; `decline_reproduction_opportunity(year, *, interval_id=None)`;
  read-only `reproduction_active`, `reproduction_patch_size_m`, `n_reproduction_opportunities`, `reproduction_last_year`,
  `n_total_records`, `phase3_state()`; `describe()['phase3']`.
* Checkpoint `SCHEMA_VERSION = 3` with a `phase3` block (no new arrays); `leaf.recruitment` added to the source hashes.

## Reuse / adapters

Phase 2 `_rebind` is generalised (not duplicated) with `births=`; `_materialize(births=)` appends rows to the canonical state;
`Domain.contains/cell_indices/check_field`, `PlantState`, the shared allometry / canopy (`_build_geometry`), the shared root
builder, `_publish`, `_prepare_ids`, the global byte ledger and the demography SplitMix64 finalizer are reused. Surviving
per-plant root overrides are kept; daughters use species roots.

## Scientific choices / deviations (all explicit and uncalibrated)

* Law numbers live in the species `reproduction` section; examples are UNCALIBRATED FIXTURES. Seeds: per-patch trial independent
  of cell size; probability per patch regardless of clipped area; external-propagule credit of dry AGB in the ledger; no water.
* Daughters: separately located ramets (user choice), funded by exact parent B / E transfer; parent stays live (fraction < 1);
  bounded rejection placement, never wrapped; canopy overlap allowed and untested.
* Opportunity = just-completed local year, strictly sequential, taken while the cursor is in year + 1; unusable years must be
  declined explicitly (a deliberate API addition, so that a missed year is never silently dropped yet the session is not stuck).
* Newborn correction in `leaf.demography`: background hazard exposure `min(period_days, age_end)` (changes Phase 2 results only
  for plants younger than the decision period). The Phase 2 RNG streams / golden hash values are untouched; new streams 11..24.
* Reproduction randomness counter = accepted opportunities; the declined opportunity consumes none.
* An existing Phase 2 test asserting checkpoint version 2 (`test_checkpoint_format_version_two_is_explicit_and_strict`) was
  updated to version 3 because the format bump is required by the task; no source guard was weakened.

## Open limits / things Codex should check first

1. Everything is NOT RUN; names / imports (`traits.load_catalog`, `state.write_plant_csv`, `PlantState.from_records`,
   `cat.index`) were used as seen in existing tests / examples.
2. Implementation is a vectorised NumPy HOST reference on all backends (no Numba / CUDA eligibility or placement kernel); on cuda
   B / E are read back (counted) and the rebuilt state uploaded. Not GPU resident.
3. Byte constants (`BYTES_PER_*` in `recruitment.py`, `birth_extra` in `_rebind`) are conservative estimates, not measurements.
4. The `max_new_plants` / ID checks use the REQUESTED daughter count for the limit and the exact placed count for ID exhaustion.
5. Session tests assume the Phase 1 fixture catalog (`grass`, `shrub`) and that `make_plants` ages / sizes make grass IDs 1 and 3
   eligible; statistical assertions (seed recruits >= 4, 2..4 daughters) hold for the configured p = 1 except for the (tiny)
   chance that every placement attempt of a parent fails. Unit test `Seeds.test_zero_live_zero_records_seed_establishment`
   relies on the fixed seed giving at least one success (probability of failure ~0.6 % a priori).
6. The existing Phase 2 tests were not re-read in full; if any asserted `rebind` dict equality, the added `n_births` key or the
   newborn hazard correction may need adapting (no existing test was seen to depend on a plant younger than its period).
7. Unsupported / future: calibration, seed banks, spacing rules, per-daughter root policy, multi-year catch-up, compiled / CUDA
   placement kernels, `--backend` handling beyond the example's required flag (the example passes the selected backend only).
