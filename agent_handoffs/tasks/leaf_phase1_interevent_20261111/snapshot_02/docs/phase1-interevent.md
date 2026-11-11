# Phase 1: dated native-water / growth session and complete restart

`leaf.interevent.IntereventSession` composes the already supplied Phase 1 pieces into ONE dated driver for
interevent periods, and `leaf.checkpoint` saves / restores its whole accepted boundary. Everything here is
**unqualified and uncalibrated**: the species numbers in `examples/phase1/species.json` are fixtures, the example
weather is synthetic, and nothing was run by the writer (see `report.md`). It is not a MASY coordinator, not storm or
wind physics, and not a full MAPLE / SYRUP / MASY scenario.

## What one accepted boundary holds

| item | owner / representation |
|---|---|
| soil water | native SYRUP `SoilColumnState` (`water_m` [L, N] m, `t_s`): the ONLY water store; LEAF never subtracts water |
| living-plant biomass B and seasonal biomass E | session arrays [live] on the backend (authoritative); `PlantState` is only an export snapshot |
| pending requested / delivered water D, U | per live plant, m3 summed over the open growth period |
| pending activity / temperature integrals A, T | per species [S], elapsed **years** weighted by activity / temperature |
| root map + canopy areas | current fixed `RootMap` (live rows) and the period's canopy area (m2, circles, from the shared allometry) |
| clock | exact fixed-offset cursor (datetime, microsecond exact), next growth boundary, deferred season flag, interval sequence, last receipt |

**Row mapping.** Working arrays and the root map hold LIVE plants only, in state row order: working row `i` is the
plant `plants.ids[live_rows[i]]` (`session.live_rows`, `session.live_ids`; fixed for the session, root-map ids are
checked at every rebuild). All plants, dead and removed included, stay recoverable from the private host base arrays:
`session.snapshot()` returns a canonical `PlantState` at the cursor (explicit device-to-host transfer, counted).
Live ages are `age at start + exact elapsed days`, so any split of a run gives the same ages and snapshots never
advance the clock. There is no daily host readback of the population.

## Executable use

```python
import datetime as dt
from maple_syrup import subsurface as ss, subsurface_transpiration as st
from leaf import forcing, interevent, traits, state

catalog = traits.load_catalog('examples/phase1/species.json')
column  = ss.prepare_column(layers, active_mask, cell_area_m2, backend='numpy')   # prepared by the caller
operator = st.prepare_transpiration(column)
native   = ss.uniform_state(column, [0.12, 0.15], 0.0)
session = interevent.IntereventSession(
    plants, catalog, operator=operator, native_state=native,
    start=dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc), backend='numpy',
    growth_cadence=dt.timedelta(days=7), max_edges=20_000_000, max_bytes=500_000_000,
    reserve_bytes=0, root_builder='compiled', layer_fraction_overrides={17: [0.1, 0.9]})
rec = forcing.TimestepRecord(start, end, temperature_c=18.0, pet_mm_per_day=5.0)
for piece in session.split_record(rec):                       # midnights, growth grid, new years
    receipt = session.step_interevent(piece, added_water_m=added, potential_evaporation_m_s=e0)
session.flush_growth()                                         # before a transport event
session.advance_transport('wind', t0, t1)                      # or ('rain', ..., native_state=owner_result)
snap = session.snapshot()                                      # PlantState at the cursor
session.save('run/checkpoint'); session = interevent.IntereventSession.load('run/checkpoint', catalog, operator=operator, backend='numpy', max_edges=..., max_bytes=...)
```

`examples/phase1/run_interevent.py` runs this with the generator, a synthetic soil and synthetic weather; it needs an
explicit `--backend` and `--max-bytes` and performs no package or environment setup. All its values are UNCALIBRATED.

### One open interval (`step_interevent`)

1. refuse unless `record.start == cursor` (no gap, overlap or replay), the offset equals the session offset, and the
   record does not cross a local midnight, a growth-grid point or a new year (`IntereventBoundaryError`; split first);
2. the REAL native `step(column, state, dt, added_water_m, potential_evaporation_m_s)` on the accepted state;
3. `growth.demand` on the EXISTING period geometry with the actual month, temperature, shared `pet_mm_per_day` and the
   active interval length;
4. the REAL `syrup_water.withdraw_plants` once on the stepped state (SYRUP accepts and debits; the actual delivery
   is gathered; stage continuity and storage closure are checked);
5. D, U and the integrals (365 / 366 day year of the record) are accumulated on the same backend;
6. at a growth boundary (grid point, new year, or `flush_growth`) `growth.grow` is applied once to the accumulated
   requested / actual water, B / E are copied out of its shared output buffer, and roots + canopy areas are rebuilt by
   `leaf.roots.build_root_map` from the shared catalog implementation (species and ID-keyed overrides preserved;
   unmet roots are never redirected to wet cells);
7. ONE replacement of the accepted boundary after every stage passed. Any failure (native, demand, bridge, growth,
   rebuild, admission) leaves the previous boundary untouched; the same record can be retried and is then accounted
   once (no double debit, no age advance, no lost pending demand).

`added_water_m` (m per layer) and `potential_evaporation_m_s` are explicit native forcing prepared for the already
split interval. **LEAF computes no PET, no infiltration and no soil-vs-plant partition of evaporation**: SYRUP owns
that partition upstream (current upstream limits combined potential soil loss and shares E before plant T); this driver
accepts native forcing prepared for that contract and does not claim full persistent storm / E adoption. Synthetic
qualification may pass an explicit `E0`.

### Growth boundaries

Growth boundaries are a FIXED GRID `start + cadence * k` (cadence <= 7 days) plus every local new year. A flush ends a
partial period without moving the grid. `boundaries()` and `split_record()` expose them. The weekly Euler envelope
(rates 0..1 per year) is qualified; historical fitted rates (e.g. 295 per year) remain inactive and unadmitted.

### Transport, closed gate and seasons

* `flush_growth()` finishes a partial period at the cursor (no-op, unpublished, if nothing is pending) so native
  consumers can export the current canopy (`export_canopy_view`).
* `advance_transport(kind, start, end, native_state=None)`: refuses nonempty pending growth; advances the calendar (so
  live ages exactly once), skips grid points crossed (`growth_boundaries_skipped`, no catch-up); no T, growth or E.
  `wind`: native water unchanged, native `t_s` advanced. `rain` / `recession`: the external native owner's resulting
  `SoilColumnState` is REQUIRED, checked (column binding, native validity, end time) and copied; LEAF runs no storm physics.
* A year end inside transport sets `season_due`; the turnover (E removed, B reduced by it, no accrual of event
  duration) is applied at the start of the next OPEN interevent interval and stored in restart.
* A CLOSED gate record is a pause with the wind semantics: no native step (the scheduler's other process owns the time),
  no T, no growth, forcing must not be passed, pending growth must be flushed first.
* No mortality or disturbance law lives here; ages advance for live plants only.

## Parameter ownership and units

All species parameters live in the complete species records of the catalog: `growth.traits`
(`mature_biomass_kg`, `intrinsic_rate_per_year` in [0, 1], `biomass_support_kg_year_per_m3`, `live_fraction`,
`extra_growth_rate_per_year`, `monthly_activity[12]`, cardinal temperatures) and `water_use.traits`; `roots`
(lateral radius multiplier, `layer_fractions`) and `allometry` (power laws, arbitrary exponents). The session adds no
species default. Units: B, E kg dry aboveground; D, U m3 per period; support kg*year; integrals years; native water m
(depth); `t_s` s; cadence `timedelta`.

## Memory and transfers

One global `max_bytes` (host + device summed, plus `reserve_bytes`) is admitted before every allocating stage:
persistent state + the stage transient from the component estimators (native `estimate_step_memory`,
`growth.memory_estimate`, `syrup_water.estimate_bridge_bytes`, the allocator plan / feedback estimates) + old / new
coexistence; components receive the remaining budget through `extra_reserve_bytes`; arrays a component already counts are
excluded and the rest of the double counting is labelled conservative. The root builder gets `max_bytes` minus all
other live bytes INCLUDING the old map; a rebuild (or a map that would make the next interval unadmissible) is
refused whole. Actual root edges are counted after the builder's bounded count pass; individuals are never dropped.
Excluded: interpreter, JIT, CUDA context, allocator pools, caller-held arrays outside `reserve_bytes`. The weekly
geometry (host allometry and the NumPy / compiled root builder) is HOST based: on CUDA B and E are copied to the host
and the new CSR back (explicit and counted). It is not claimed to be GPU resident.

## Restart (`leaf.checkpoint`)

`session.save(dir, overwrite=False)` publishes a directory atomically (temporary directory in the parent, fsync, rename;
never overwrites by default): `plants.csv` (the canonical plant CSV of ALL plants at the cursor), `restart_arrays.npz`
(native water, pending D/U/A/T, canopy areas, row mapping, the exact current root CSR, overrides; loaded with
`allow_pickle=False`) and `restart.json` (schema, exact cursor / boundary / pending days / season flag / sequence,
native fingerprint of the column, backend, species / catalog hash, module and native source hashes, last receipt,
file and per-array sha256, content hash). `IntereventSession.load(dir, catalog, operator=..., backend=..., max_edges=...,
max_bytes=...)` verifies every hash, schema, dtype, shape, finiteness and time / boundary / pending consistency,
validates the CSR through `leaf.uptake`, rebuilds roots and canopy areas and compares them exactly (`verify_roots`),
and only then returns a session; nothing is adopted on any failure. No silent migration of catalog, backend, native
column, root builder or sources (`allow_source_change` relaxes only the module-source comparison). The session is
re-anchored at the saved cursor, so ages agree with a continuous run to FP64 rounding of one addition.

Coupled-restart scope: this restores the LEAF accepted boundary and the native water / time of this column only. A full
MASY-level coupled restart (storm / wind owners, shared scheduler state, event queue, surface remapping) is not part of
this phase.

## Guards (revision 2)

* **Scalar admission first.** `max_edges`, `max_bytes`, `reserve_bytes` are strict non-boolean integers (no float
  truncation, no negative reserve) checked before anything else in the constructor and in `load`; the constructor then
  admits a metadata-derived lower bound (population copy + validation scratch + tables + native column copies + native
  state copies + reserve) BEFORE `PlantState.validate` / `copy`, and the actual live counts afterwards. Budgets are never
  raised.
* **Overrides** go through the shared strict `roots` validator after admission: integer plant IDs of live plants only
  (no bool / float / alias coercion), exactly `L` finite non-negative fractions summing to 1, stored as an immutable copy.
* **Pending water** (D, U, A, T, duration) is validated BEFORE publication: finite, non-negative, finite totals,
  `U <= D (1 + rtol)`, `T <= A`, `A <=` duration / 365, duration within the cadence. `rtol` is the allocator's
  `overdelivery_bound_rel(root_map)` plus 16 eps, passed identically to `growth.grow`. Only scalar reductions are read
  back; it uses vectorised backend operations (not a fused kernel) whose temporaries are ledgered.
* **Water closure is enforced**: the native non-T budget must be finite and self-consistent (recomputed residual inside its
  reported bound), the two native stages must be continuous, and the end-to-end residual
  `after - before - (added - return - drain - evap - T)` must lie inside the sum of the reported FP64 bounds; otherwise
  nothing is published. Nothing is balanced and no tolerance is loosened.
* **Native clock handoff** (rain / recession): `|t_in - (t0 + dt)| <= 8 eps max(|t0|, |t1|, dt)` (a few ULPs, microsecond
  scale at 1e9 s, never seconds); the calendar cursor stays an exact microsecond `datetime`.
* **Memory.** One `PlantState` materialisation (snapshot, root rebuild, checkpoint) is ledgered structurally at 157 bytes
  per plant (owned arrays 62 + constructor copies 62 + uniqueness scratch 17 + mask temporaries 16) plus 16 per live plant for
  age temporaries and, on CUDA, 16 per live plant for the device-to-host B / E copies. The accumulation stage takes the
  full bridge ledger (all buffers the still-live `PlantWithdrawal` owns) plus the new accumulators.

## What remains upstream or unqualified

No field calibration; the fixture catalog is not data. Storm / wind / recession physics, E partitioning, rational
(sub-microsecond) MASY instants (refused by type: `forcing.exact_microseconds` is the refusing conversion), surface
remapping, mortality and disturbance, custom non-separable external root maps (refused; they need an explicit compatible
rebuild policy), and the hand-off of exported canopies to native consumers are outside this phase. The native
bridge pins SYRUP revision `c76c603...` (`syrup_water.NATIVE_PIN`); `native_provenance(verify=True)` is checked once at
session construction and recorded in restart, never per interval. Tests are written against the real native API but were
NOT executed by the writer.
