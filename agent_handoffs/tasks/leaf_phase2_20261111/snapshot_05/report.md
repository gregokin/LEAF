# LEAF Phase 2 report: disturbance, biomass loss and persistent mortality state

Task `leaf_phase2_20261111`, baseline `f0d0b56854a576b35ee38bce4a8eab6dd623baf6`, session
`711c5973-64c3-45f8-af8a-379ba0e357a7`. Implemented by Claude in the isolated candidate with ordinary Read / Edit / Write
only. **Nothing was executed**: no author check (unit tests, example, import, syntax compile, Numba JIT, CUDA) was run.
Every statement below about behaviour is from desk review of the source, not from a run. Left uncommitted for Codex.

## Files written (exactly the allowed set)

| file | change |
|---|---|
| `leaf/disturbance.py` | NEW: catalog law parser, compiled species table, NumPy / Numba / CuPy kernels, event evaluation, ledger, pure `respond_state` |
| `leaf/demography.py` | NEW: age / background hazard, persistent stress counter, keyed counter-based RNG, kernels, memory estimate |
| `leaf/interevent.py` | EXTENDED: `rng_seed`, Phase 2 boundary state, `apply_surface_event`, `apply_annual_decision`, transactional `_rebind`, `time_days_at`, accessors, accumulation hook in `_growth_update`, `_Boundary` defaults, `_publish(static=)`, `_materialize(status_live=)` |
| `leaf/checkpoint.py` | EXTENDED: five Phase 2 arrays, `phase2` manifest block, strict restore validation, `rng_seed` consistency kwarg, module-hash list |
| `tests/test_disturbance.py`, `tests/test_demography.py` | NEW (pure, no native SYRUP) |
| `tests/test_interevent.py` | APPENDED Phase 2 classes (`Phase2Disturbance`, `Phase2Annual`, numba / CUDA subclasses); existing tests untouched |
| `examples/phase2/species.json`, `examples/phase2/run_disturbance.py` | NEW fixtures / bounded example |
| `docs/phase2-disturbance.md`, `README.md`, `report.md` | docs |

No canonical state / traits module, root builder, allocator, growth law, native bridge, source guard, upstream, resource limit
or environment was touched.

## Exact API

`leaf.disturbance`: `compile_species(catalog, backend, *, device=None) -> CompiledDisturbance`,
`estimate_event_bytes(n_event, n_species, backend)`, `evaluate_event(table, catalog, event, *, species_index, status, B, E,
x_m, y_m, establishment_ground_m, domain, current_surface, max_bytes, extra_reserve_bytes=0)`,
`respond_state(state, catalog, event, current_surface, *, table=None, backend='numpy', max_bytes=1<<30, require_clock=True)
-> (candidate PlantState, result)`; errors `DisturbanceError / DisturbanceBudgetError / DisturbanceBackendError`; law
`mechanical_response_v1`.

`leaf.demography`: `compile_species(catalog, backend, *, device=None)`, `decide(table, *, species_index, plant_ids,
age_end_days, stress, required_support, actual_support, planned_growth, actual_growth, period_days, seed, counter, max_bytes,
extra_reserve_bytes=0)`, `keyed_uniform(seed, counter, stream, plant_ids)`, `keyed_uniform_py`, `stream_key`,
`check_seed / check_counter`, `memory_estimate(n_live, n_species, backend)`; errors `DemographyError / ...BudgetError /
...BackendError`; law `age_background_stress_v1`.

`IntereventSession(..., rng_seed=None)` (REQUIRED, strict integer, iff any mortality law is active);
`apply_surface_event(event, current_surface, *, interval_id=None)`; `apply_annual_decision(year, *, interval_id=None)`;
`time_days_at(instant)`; read-only `rng_seed, demography_active, disturbance_active, n_annual_decisions,
n_disturbance_events, annual_due_year, annual_period_start, transport_pending, last_decision`, `stress_counters()`,
`annual_accumulators()`, `phase2_state()` (also under `describe()['phase2']`). `checkpoint.load_session(..., rng_seed=None)`.

## Decisions and deviations the reviewer should know

1. **Catalog sections.** Phase 2 laws live in the existing `mortality` and `disturbance` sections (no schema change in
   `leaf.traits`). `unimplemented` / null is inactive; unknown / malformed active laws raise at session construction.
2. **Simultaneous composition** = multiplicative survival `(1-f_b)(1-f_e)(1-f_a)`; precedence dislodgement > lethal >
   injury; lethal plants keep all B / E (no injury on top); per-cause attribution proportional to `f_k` (ULP-level
   attribution residual is reported separately from the exact `B - B_new` total).
3. **Burial / exposure increments** are derived from the current surface and the event's signed net change
   (`delta_before = delta_after - net`) because the session stores no earlier surface; `increment_basis` selects net or gross.
   This assumes the event diagnostics and the surface are consistent.
4. **Event binding** is to the LAST accepted wind / rain / recession interval only (one decision per interval; source map
   wind -> maple_wind, rain / recession -> syrup_water; `other` sources are refused for session-bound events). Binding state
   is one small dict in the boundary and in the checkpoint; no event-ID list exists. Pure `respond_state` has no
   transport binding (only the clock check).
5. **Annual decision gating.** With an active mortality law and live plants, a new-year crossing sets `annual_due` and
   `step_interevent` REFUSES until `apply_annual_decision(year)` (closed-gate pauses and transports stay allowed). This is a
   deliberate extra restriction so a new year's growth can never mix into the completed year's accumulators. It does not
   apply when no mortality law is active.
6. **Dormancy / first-year definitions.** Exempt: `age_end <= exempt_until_age_days` (an explicit species coefficient, not
   a default). Not evaluated: zero denominator (no required support / no planned growth). Both leave the counter
   unchanged; the stress response only follows a LOW evaluated year.
7. **Accumulators** are four per-live-plant float64 arrays + an int32 counter, only allocated when some species has a
   `stress` law; they are accumulated on the backend at the existing growth update from `growth.grow` outputs
   (`required_support`, `actual_support`, `normal_plan + extra_plan`, `normal_gain + extra_gain`): no readback, no
   trajectory, no new maintenance-water fraction, S not re-applied.
8. **Rebind re-anchors** the session base at the cursor (same convention as a checkpoint restore). Consequently
   `session._anchor` is not the original start after a rebind (helper code that used `_anchor` as the start, such as the
   existing test helper `session_start`, is only valid for sessions without a rebind; the Phase 2 tests step from
   `session.cursor`). The checkpoint field `original_anchor` records the current anchor.
9. **CUDA kernels are vectorised CuPy** (same code as the NumPy reference, explicit device binding, explicit uploads /
   downloads), not hand-written RawKernels; Numba kernels are serial `njit` loops. The RNG is exact integer arithmetic so
   draws are identical across backends; `pow` / `expm1` probabilities may differ in the last bit.
10. **Checkpoint**: Phase 1 checkpoints are refused (no `phase2` block / arrays; no silent migration). The auxiliary arrays
    follow the CSV live-row mapping and carry no IDs (documented limitation). `SOURCE_MODULES` now also hashes
    `demography`, `disturbance`, `events`.
11. Only live plants respond to events (standing dead plants are not removed by later events) — a documented simplification.
12. The session-level receipt for `apply_surface_event` carries the per-species / per-cause diagnostics from
    `disturbance.evaluate_event`; a zero-injury event still publishes (sequence + 1, replay guard, counter) but does not
    rebind.

## Revision 2 (corrections; see docs/phase2-disturbance.md "Revision 2 guards and admission")

1. Nominal tests (`tests/test_disturbance.py` helper `ev`, `tests/test_interevent.py` helper `event_now`) now supply an explicit
   zero dose in `kg m-1` for wind events; `nodose=True` keeps the dedicated missing-dose / flux tests. Production guards unchanged.
2. `demography.decide` validates species bounds, unique positive IDs, finite non-negative ages / metrics, counters within species
   caps and `actual_growth <= planned_growth * (1 + 1e-9)` before any kernel (backend scalar reductions; no repair).
3. `apply_surface_event`: new `event_lookup` admission stage before surface validation, and a local `_lookup_rows` that never
   populates `base._order / _sorted_ids`. New accessors `stress_counters` / `annual_accumulators` are admitted. Phase 1
   `live_rows` / `live_ids` copies remain caller-bound (documented).
4. `respond_state(..., extra_reserve_bytes=0)` admits (206 B / plant incl. the caller's original, event arrays, table) before
   resolve / compile / copy; `compile_species(..., max_bytes=None, extra_reserve_bytes=0)` in both modules admits
   `table_compile_bytes(S, backend)` first; the session passes its ledger (`initialization_phase2_tables`).
5. Full dieback: the first draft ALREADY refused `biomass_loss_fraction >= 1` (and `max_injury_fraction >= 1`) at parse, so the
   reported 'accepts 1' could not be reproduced by reading the source; I added endpoint tests (1.0 refused, 0.9999999999999999
   accepted) and documented the contract. If your probe saw acceptance, it came through another path - please send it.
6. `_pause` refuses while `annual_due` is set and refuses a transport / pause crossing more than one new year (live plants +
   active mortality law only); regression test added incl. exact restart.

Final signatures: `disturbance.compile_species(catalog, backend, *, device=None, max_bytes=None, extra_reserve_bytes=0)`,
`disturbance.respond_state(..., require_clock=True, extra_reserve_bytes=0)`, same compile signature in `demography`,
`demography.decide` unchanged (new `DemographyError` content refusals). All new / changed tests NOT RUN.

## Revision 3 (current; supersedes the Revision 2 ID statement)

`demography._validate_content` now requires plant IDs to be exact int64 and UNIQUE only; zero, negative values and both int64
endpoints are valid (canonical signed-ID contract). The earlier "unique and positive" wording in Revision 2 (item 2) and in
the Revision 2 test `zero_id` is HISTORICAL and withdrawn. All other guards (species bounds, finite non-negative metrics,
counter caps, actual <= planned growth, dtype/device) are unchanged; the modulo-2**64 hash, streams and populations are
untouched. New tests: signed IDs `[-2**63, -1, 0, 2**63-1]` vs the scalar reference and row permutations (numpy, plus numba /
CUDA classes that skip when unavailable); bool / float / duplicate IDs still refuse. NOT RUN.

Current final signatures (unchanged from Revision 2): `disturbance.compile_species(catalog, backend, *, device=None,
max_bytes=None, extra_reserve_bytes=0)`, `disturbance.respond_state(state, catalog, event, current_surface, *, table=None,
backend='numpy', max_bytes=1<<30, require_clock=True, extra_reserve_bytes=0)`, same compile signature in `demography`,
`demography.decide(table, *, species_index, plant_ids, age_end_days, stress, required_support, actual_support,
planned_growth, actual_growth, period_days, seed, counter, max_bytes, extra_reserve_bytes=0)`. Budgets: `table_compile_bytes =
8*rows*S` (x2 cuda), `STATE_POPULATION_BYTES_PER_PLANT = 206`, `event_lookup` = 24 B / total plant + 8 B / cell + 64 B / event
plant, decision 448 / 896 B per live plant, persistent 36 B per live plant (stress law only).

## Revision 4 (checkpoint format version; current)

`checkpoint.SCHEMA_VERSION` is now 2 (earlier sections that say the restart schema stays 1 are HISTORICAL). The loader checks
`type(schema_version) is int and == 2` immediately after manifest parsing (before arrays / population); the error names the
actual schema / version and the expected 2; v1 is never migrated or defaulted. CSV version unchanged. New test
`test_checkpoint_format_version_two_is_explicit_and_strict` (save v2, round trip, CSV v1, rejection of 1, 3, 2.0, True, '2',
None with re-hashed manifests and no array decoding). Only `leaf/checkpoint.py`, `tests/test_interevent.py`, the docs and this
report changed. NOT RUN.

## Revision 5 (actual-CUDA corrections; current)

1. `demography.decide` CUDA input guard now checks `isinstance(a, cupy.ndarray)` before reading `.device` (NumPy >= 2 arrays
   expose `.device`), so NumPy / foreign data fail with `DemographyError`, never an AttributeError; nothing is coerced or uploaded.
2. `Phase2CudaParity` now expects `stress is None` for the no-stress fixture (nothing allocated for it) and a new test runs an
   ACTIVE stress-law fixture on CUDA: device-resident counter / annual accumulators, an annual decision and an own-backend restart.
   Not run.

## Author checks

**ALL NOT RUN.** Specifically not run: `python -m unittest` of any file (new or existing), module import / syntax
compilation, the example, Numba JIT of the new kernels, CuPy / CUDA paths, native SYRUP provenance verification,
checkpoint round trips, any budget / fault-injection test. The new tests were written against the real APIs but their
numeric expectations were derived by hand from the documented formulas.

## Places most likely to need correction (desk-review risks)

* Numba typing of the `uint64` hash (module-level `np.uint64` constants, `np.uint64(int64)` casts) and of boolean
  short-circuits in `disturbance._numba_kernel` / `demography._numba_kernel`.
* CuPy semantics of `bincount(..., weights=)`, boolean-mask indexing and `expm1` / `power` in the CUDA path; device checks.
* `copy.copy(session)` candidate staging in `IntereventSession._rebind`: every attribute that the geometry / materialise
  helpers read must be overridden (done for `_base, _live_rows, _Pl, _has_plants, _anchor, _time0, _live_ids, _overrides,
  _sidx, _live`); a missed one would silently use the old population.
* Memory-ledger constants for the new stages (`80 * P_live`, `320 * n` event arrays, `60 * P_live` decision arrays,
  `REBIND_ARRAY_BYTES_PER_LIVE = 17`) are conservative estimates, not measurements; tests that bound budgets from receipts
  (`*_rebind_preflight`) assume the stage names used here.
* `tests/test_interevent.py` Phase 2 classes: expected values in `test_stress_accumulators_use_actual_growth_outputs_once...`
  re-implement the documented rule; dry-soil behaviour of the shrub growth ratio is data dependent (only the rule, not an
  outcome, is asserted). Floating-point exactness assumptions use binary-fraction surface values (0.0625, 0.125, 0.3125 ...).
* Existing Phase 1 tests: `_persistent` gained two zero-valued items when inactive; receipts gain
  `annual_decision_due_year` only when a mortality law is active; `describe()` gains a `phase2` key. Please confirm that no
  Phase 1 test pins the full key sets of `describe()`, `restart_arrays.npz` (now 18 arrays) or `restart.json`.

## Limitations (also in `docs/phase2-disturbance.md`)

Uncalibrated fixtures; no field data; no wind / water / burial physics and no surface remapping; aboveground dry biomass
only; one-event-per-transport-interval binding; 365-day years for hazards; annual accumulators dropped with the plant;
CUDA / Numba parity unverified; explicit LEAF adaptation, no ECOTONE / STEPWAT2 reproduction claim.
