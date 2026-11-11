# Phase 2: disturbance, biomass loss and persistent mortality state

Everything here is **unqualified and uncalibrated**. It is an explicit LEAF *aboveground-only adaptation*: there is no
exact ECOTONE or STEPWAT2 reproduction, no belowground or carbon pool, no wind or water physics, no soil / surface-layer
remapping and no fire. The species numbers in `examples/phase2/species.json` are readable **fixtures**, not field fits.
The code and the tests were written without the ability to execute anything (see `report.md`).

Modules: `leaf.disturbance` (species mechanical response to a surface event), `leaf.demography` (opt-in age /
background hazard and persistent low-growth stress, keyed randomness), `leaf.interevent` (two explicit biological
boundary APIs and the transactional population rebind), `leaf.checkpoint` (accepted Phase 2 state in complete restarts).
With no active Phase 2 law every Phase 1 path, receipt and byte count is unchanged.

## Where the numbers live

Every species coefficient is in that species' own catalog record, beside allometry, roots and growth: the `disturbance`
and `mortality` sections (preserved verbatim by `leaf.traits`, consumed here). There is no species table in any module
and no assumed field default. `law: "unimplemented"` (or a null section) is **explicitly inactive** (Phase 0 / 1
catalogs are unchanged). An unknown law name, an unknown key, a missing trait, a non-finite or out-of-range value, a
wrong type: the compile step **refuses naming the species** (it never silently disables a law). The catalog
`life_expectancy` is metadata only: **a mean is not a hard maximum**, nothing here reads it.

## Mechanical response (`disturbance.law = "mechanical_response_v1"`)

```json
"disturbance": {"law": "mechanical_response_v1", "traits": {
  "burial":   {"increment_basis": "net|gross", "scale": "m|height_fraction", "injury_threshold": x0,
               "injury_saturation": x1, "max_injury_fraction": f, "curve": "linear|smoothstep",
               "lethal_depth_m": null|>0, "lethal_height_fraction": null|>0},
  "exposure": {"increment_basis": "net|gross", "injury_threshold_m": x0, "injury_saturation_m": x1,
               "max_injury_fraction": f, "curve": "...", "lethal_depth_m": null|>0, "dislodgement_depth_m": null|>0},
  "abrasion": {"dose_unit": "kg m-1", "applies_to_sources": ["maple_wind"], "injury_threshold": x0,
               "injury_saturation": x1, "max_injury_fraction": f, "curve": "...", "lethal_dose": null|>0}}}
```

Each mechanism is `null` (explicitly off) or complete; `0 <= x0 < x1`, `0 <= f < 1`; `null` thresholds are off (never
default values). Units: depths in m, burial may be scaled by the intact allometric height, abrasion in the **declared**
dose unit.

**Geometry.** The plant keeps its FIXED establishment ground `z_est` (ground elevation at establishment, *not* the
plant height at establishment). The caller passes the ACTUAL current surface after the event (`ny, nx`); each plant
samples its containing cell: `delta_after = surface - z_est` (signed NET change since establishment, + deposition /
burial, - erosion / exposure), `burial_after = max(delta_after, 0)`, `exposure_after = max(-delta_after, 0)`.
The state before the event is `delta_before = delta_after - net_surface_delta_m`. **Net vs gross:** the event's
`deposition_depth_m` / `erosion_depth_m` are GROSS; `net_surface_delta_m` is their signed difference. Injury may use
either (`increment_basis`): `net` = the increase of absolute burial (exposure) this event caused, `gross` = the event's
deposition (erosion) depth. A deposition that only refills an erosion hole has gross > net; equal erosion and deposition
have net 0 (a `net` species is not injured, a `gross` species is).

**Increments injure, absolute depths kill.** Injury responds to the increment of THIS event only: accumulated
historical burial is never reapplied as new damage (a plant is re-evaluated only when an event names it, and a repeated
quiet event changes nothing). The ABSOLUTE post-event `burial_after` (m, or fraction of the intact height),
`exposure_after` and abrasion dose may cross explicit **lethal** or **dislodgement** thresholds.

**Abrasion dose is not flux.** `abrasion_dose` has a declared unit and `transport_flux` a different one; LEAF invents
no conversion. A species with an applicable abrasion mechanism needs an event dose whose `abrasion_unit` equals the
species `dose_unit` **exactly** (no unit conversion, no trimming); a flux-only event refuses. `applies_to_sources` lists
the event sources the dose applies to (the fixture: `maple_wind` only, so SYRUP water erodes / deposits but does not
abrade; a dose in a water event is ignored).

**Per-mechanism injury.** `f_k = max_injury_fraction * g(x)` where `g` is the clipped ramp
`clip((x - x0) / (x1 - x0), 0, 1)` (`smoothstep`: `g*g*(3 - 2g)`); `g = 0` below the threshold (zero injury).

**Simultaneous composition (bounded).** `retained = (1 - f_burial)(1 - f_exposure)(1 - f_abrasion)` in `(0, 1]`.
`B_new = B * retained`, `E_new = min(E * retained, B_new)`; the seasonal share is preserved and `0 <= E <= B`. Total loss
`B (1 - retained) < B`: it never exceeds the existing dry AGB. Precedence: dislodgement
(`exposure_after >= dislodgement_depth_m`) -> `removed`; else lethal burial / exposure / dose -> `dead_standing`
(B, E unchanged, no extra injury); else injured (`live`, composed fraction). Loss per cause is attributed in proportion
to `f_k`; dislodgement exports the whole remaining B / E (zeroed in the state; the exported kg are in the receipt).
Only LIVE plants respond (dead_standing / removed plants and species with an inactive law are counted and skipped).
X / Y, anchors, stable IDs and ages are never touched; dead / removed records remain forever.

**Ledger.** `old dry AGB = new retained dry AGB + removed dry AGB`, verified per plant within `16 eps B` (and for the
population) and reported per cause (burial, exposure, abrasion, dislodgement) and per species; the seasonal E ledger is
reported too. A ledger that does not close refuses.

## Age / background hazard and persistent stress (`mortality.law = "age_background_stress_v1"`)

```json
"mortality": {"law": "age_background_stress_v1", "traits": {
  "age_hazard": null | {"weibull_shape": >0, "weibull_scale_years": >0},
  "background_hazard_per_year": >=0,
  "stress": null | {"metric": "support_ratio|growth_ratio", "low_threshold": (0,1], "exempt_until_age_days": >=0,
                    "limit_years": int>=1, "counter_cap_years": int>=limit, "mortality_probability": [0,1],
                    "biomass_loss_fraction": [0,1)}}}
```

* **Hazard** over the decision period (365-day years): `dH = max(H(a1) - H(a0), 0) + background * period_years`,
  `H(a) = (a / 365 / scale)^shape`, `p = -expm1(-dH)`, death iff the keyed uniform `u < p` (so probability 0 never kills,
  probability 1 always kills). A death is `dead_standing` (B, E kept, age frozen). The Weibull scale is its own
  coefficient: nothing is coupled to `life_expectancy` and there is no cutoff at the mean.
* **Stress counter** (per live plant, persistent, int32). During the year the session accumulates, per plant and
  WITHOUT a new maintenance-water fraction, the growth outputs it already computes from the ACTUAL native feedback
  (`delivered_m3`; satisfaction S is applied once inside `growth.grow`, never again, and unmet demand never becomes extra
  withdrawal): required support `R`, actual support `A` (kg*year), planned growth `P` and actual growth `G` (kg). These are
  support / growth DIAGNOSTICS, not biomass-production yields. At the annual decision a plant is **exempt** (age <=
  `exempt_until_age_days`: counter unchanged), **not evaluated** (denominator 0: no required support or no planned growth,
  i.e. dormant, zero potential or at the maturity cap; counter unchanged) or evaluated: `ratio = A/R` (or `G/P`);
  `ratio < low_threshold` -> `counter = min(counter + 1, cap)`, otherwise `counter = max(counter - 1, 0)` (recovery is a
  decrement, not a reset). **Dormancy or maturity alone is never stress.** A LOW year with `counter >= limit_years` is a
  stressed year: death with `mortality_probability` (keyed), otherwise dieback of `biomass_loss_fraction` of B (E scaled
  equally). A healthy, exempt or unevaluated year never triggers the response.
* Precedents (`reference/method_notes.md`: first-year exemption, a counter that decrements, an age-dependent hazard) are
  structure only. No source constant is adopted as a default.

**Keyed randomness.** `u = keyed_uniform(seed, decision_counter, stream, plant_id)`: a SplitMix64-finalizer hash of
integers (stateless; NO global RNG), identical on NumPy / Numba / CuPy, independent of row order and species order, and
reproducible after a restart because only `(seed, counter)` are random state. The counter advances only when a decision
is accepted: rejected calls consume neither random state nor biomass nor water. An active mortality law needs an explicit
integer `rng_seed` at session construction (no default).

## Session API (`leaf.interevent.IntereventSession`)

```python
session = IntereventSession(plants, catalog, ..., rng_seed=424242)     # rng_seed required iff a mortality law is active
session.flush_growth()
session.advance_transport('wind', t0, t1)                              # accepted wind / rain / recession interval
rec = session.apply_surface_event(event, current_surface)              # right AFTER the transport interval
...
rec = session.apply_annual_decision(year)                              # right after the calendar crossed a new year
session.time_days_at(instant), session.transport_pending, session.annual_due_year, session.n_annual_decisions,
session.n_disturbance_events, session.stress_counters(), session.annual_accumulators(), session.rng_seed
```

**`apply_surface_event(event, current_surface, interval_id=None)`** runs only AFTER the event (never during growth or
transpiration). It is bound to the last accepted transport interval, at most ONE decision per interval: `event.source`
must be its source (`wind` -> `maple_wind`, `rain` / `recession` -> `syrup_water`), `time_start_days` / `time_end_days`
must equal the interval (tolerance `max(1e-9, 64 eps |t|)` days; `session.time_days_at` gives the model time of any
instant), IDs must resolve and be unique, growth must be flushed (always true after a transport). A replay, a stale
(older), future or wrong-start event, a second event for the same interval, an event after any later accepted interval,
a wrong source, a wrong unit / missing dose, an unknown ID, a malformed surface or a failed rebuild / budget raise
with the whole accepted state untouched. Native water and native time never change. A zero-injury event still records
the decision (replay guard) without rebuilding anything.

**`apply_annual_decision(year)`** is the explicit scheduler call at the local new-year boundary. After an open interval
ends on 1 January (or a transport spans a new year) and an active mortality law exists, the session records
`annual_due_year` and **`step_interevent` refuses until the decision is taken** (a closed-gate pause and transport remain
allowed; a later year end only moves the due year forward). Ordering inside a year: growth updates (including the
year-end update) -> annual decision -> the deferred seasonal turnover of the NEXT open interval. A year end *inside
transport* accrues no growth: the accumulators keep the real pending year (and are not reset or discarded) until its
decision. Pending growth must be flushed first. `year` must equal the outstanding year (a replay / wrong year refuses).
The decision covers the period since the previous decision (the session start for the first), so hazards integrate the
real elapsed time even across multi-year transports.

### Transactional population rebind

Death, dieback, injury and removal change status / B / E. The new population is built BEFORE publication:

1. the new canonical base state at the cursor (ages frozen for plants that stop being live; the base is **re-anchored**
   at the cursor exactly like a checkpoint restore, so ages and `time_days` continue without double aging);
2. the new live-row map, species index and the ID-keyed layer overrides of surviving plants;
3. common geometry through the shared allometry and roots through the shared builder (non-live plants have no roots and
   therefore request no water on the next native day; dead_standing plants keep their canopy in the shared view, removed
   plants have none);
4. new working arrays (B, E, D, U zeros, stress counter and annual accumulators subset to survivors);
5. a *shallow candidate session* carries the new static fields; ONE replacement publishes the new boundary and the new
   static fields together after the last operation that can raise.

Any failure or admission refusal leaves the old state, native water / time, clock, growth grid, deferred season,
pending diagnostics, RNG counter and decision guards intact; a valid retry is then accounted exactly once.

## Memory (global `max_bytes`, host + device, caller reserve preserved)

Every allocating path validates scalar budgets and admits BEFORE population-sized allocation. Estimates (bytes):
disturbance: `704 * n_event` (host) / `1344 * n_event` (CUDA host + device), no P-sized array; demography decision:
`448 * P_live` (host) / `896 * P_live` (CUDA) transient; persistent accepted state: `36 * P_live` (int32 counter + four
float64 annual accumulators, **only when a stress law is active**) and S-sized law tables (`160 + 96` bytes per species,
doubled on CUDA, none when inactive). A rebind stage total counts the old boundary (old map included) plus the new
canonical base (`62 * P_all`), the materialisation peak, the candidate geometry, new working arrays (about `80 * P_live`),
the event / decision arrays and the new accumulators; the root builder receives `max_bytes - everything else live`; a
`*_rebind_commit` stage counts old + new coexistence and the next interval must fit with the new map. Large populations
stay linear in plants + edges: no history, no max-capacity expansion, no plants-by-cells array. Excluded: interpreter,
JIT caches, CUDA context, allocator pools, caller-held arrays outside `reserve_bytes`. Weekly / rebind geometry is host
based (explicit, counted transfers on CUDA); the Numba / CuPy kernels are population-parallel with no Python per-plant
loop (the CUDA path is vectorised CuPy on one explicit device, not a hand-written kernel).

## Checkpoint (`leaf.checkpoint`)

`plants.csv` remains the SINGLE canonical plant-record format. The auxiliary accepted state lives in the existing
full-checkpoint arrays: `stress_counter`, `annual_required_support_kg_year`, `annual_actual_support_kg_year`,
`annual_planned_growth_kg`, `annual_actual_growth_kg` (live-sized, zero-length unless a stress law is active) and a
`phase2` manifest block (RNG seed as decimal text, decision counters, year-due flag, decision-period start, pending
transport binding, last decision record). **Limitation:** these arrays carry no plant identity of their own (they follow
the live-row mapping of the CSV) and the pending / annual decision state is not representable in the CSV. A checkpoint
written before Phase 2 is refused (no silent migration). Restore verifies hashes, seed binding (a different supplied seed
refuses), counters, counter caps, finiteness, active-law flags and binding consistency before a session exists; a
restored session makes the same keyed draws as an uninterrupted run.

**Format version.** The Phase 2 restart layout (five required arrays plus the required `phase2` manifest block) is
`restart.json` schema `leaf.interevent_restart` **`schema_version` 2**; the canonical plant CSV stays version 1. The
loader requires an actual non-boolean integer equal to 2 (2.0, `true`, `"2"`, null, 1 and 3 are refused) and checks it right
after parsing the manifest, BEFORE any array is decoded or any population is allocated; the message reports the actual
schema / version and the expected 2. Version-1 (Phase 1) checkpoints are NOT read, defaulted or migrated, and
`allow_source_change` does not authorize a format conversion; resume such runs with the pinned Phase 1 runtime. Any future
migration needs separate qualification: the CSV alone is not a complete restart (pending growth, random counters and stress
state live in the arrays / manifest).

## Tests and example

`tests/test_disturbance.py`, `tests/test_demography.py` (pure; no native SYRUP) and the Phase 2 classes appended to
`tests/test_interevent.py` (real native SYRUP, the existing Phase 0 / 1 tests are unchanged). Numba and CUDA classes skip
when the backend is unavailable. **NOT RUN by the author.** `examples/phase2/run_disturbance.py` runs one synthetic wind
event and one annual decision (needs an explicit `--backend` and `--max-bytes`; see its header for the command).

## Revision 2 guards and admission

* **Dose fixtures.** A wind event under an applicable abrasion law must carry an explicit dose in the declared unit
  (a zero dose is legitimate when only burial / erosion is exercised); the production unit / dose guards are unchanged.
* **Annual-decision input validation (before any kernel, gather or bincount).** `demography.decide` checks species
  indices in `[0, S)`, finite non-negative ages and the four annual metrics, stress counters
  `>= 0` and `<= counter_cap_years` of their species (0 for species without a stress law), unique plant IDs of exact
  int64 dtype (the canonical state's UNIQUE SIGNED int64: zero, negatives and both endpoints are valid; the keyed hash
  reads them modulo 2**64 unchanged, no abs / renumbering), and actual growth `<=`
  planned growth within `1e-9` relative (sums of satisfaction-capped gains). Actual support may exceed required support
  (surplus is real). Checks are backend reductions (scalars only); nothing is clamped or repaired; numpy, numba and CUDA
  refuse identically.
* **ID lookup.** `apply_surface_event` admits an `event_lookup` stage (24 bytes per TOTAL plant of local argsort /
  sorted-ID / mask scratch, the surface validation copy, event index arrays) BEFORE validating the surface or resolving IDs,
  and resolves IDs with a LOCAL lookup: the accepted base never receives the `PlantState.indices_of` cache (16 bytes per
  total plant that no ledger counted). Unknown / duplicate IDs refuse; the accepted state is untouched.
* **Pure entry points.** `respond_state` admits original + candidate population, copy / update / index scratch
  (`STATE_POPULATION_BYTES_PER_PLANT = 206` per plant, the caller's original included), the event arrays and, when no
  table is supplied, the table BEFORE ID resolution, compilation or any copy (`extra_reserve_bytes` keeps the caller's
  reserve). `compile_species` of both modules accept `max_bytes` / `extra_reserve_bytes` and admit
  `table_compile_bytes = 8 * rows * S` (host; x2 with the CUDA device mirror, counted in full even if an inactive catalog
  later drops the arrays) before allocating or uploading; the session passes its running ledger. Disturbance table rows
  `14 float + 6 int` (160 B / species), demography `7 + 5` (96 B / species).
* **Accessors.** `stress_counters()` and `annual_accumulators()` are admitted against the global ledger before the copy.
  The Phase 1 `live_rows` / `live_ids` copies (8 bytes per live plant) remain caller-bound copies outside the ledger, as
  documented in Phase 1.
* **Full dieback.** `biomass_loss_fraction` (and every `max_injury_fraction`) must lie in `[0, 1)`: a value of 1 is refused at
  configuration (no zero-biomass live plant, no hidden floor, no discarded mass). Complete death is expressed by
  `mortality_probability`, which yields `dead_standing` with B / E retained.
* **Outstanding annual decision.** With live plants and an active mortality law, `advance_transport` and closed-gate
  pauses are refused while `annual_due_year` is set (the original year and observation period are preserved), and a
  transport / pause may cross at most ONE new year (a longer event must be split at the new year with the decision in
  between): no year of stress is invented, relabelled or discarded during an event, and no E / T / growth accrues. The
  checkpoint preserves the outstanding decision.

## Limitations

Uncalibrated fixtures; no calibration data; no wind / water / burial physics (the event is an input); no surface
remapping or soil-layer change; aboveground dry biomass only (no root biomass, no carbon pool); only live plants respond
(a standing dead plant is not removed by later events); the CUDA path is vectorised CuPy (not RawKernel) and its parity
and the Numba parity are untested here; hazards use `pow` / `expm1`, whose last bit may differ between libraries (a
decision can differ only if a draw lies within ~1e-15 of `p`); years are 365 days for hazard rates; the annual
accumulators are per live plant and are dropped with the plant; the example event is synthetic.
