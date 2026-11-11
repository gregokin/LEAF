# LEAF Phase 3 — simple seed establishment and nearby vegetative daughter plants

Status: UNQUALIFIED and UNCALIBRATED. Written without the ability to execute anything; every check is NOT RUN by the author.
This is an explicit LEAF adaptation of simple STEPWAT2-style structure, **not** a STEPWAT2 / ECOTONE reproduction, and no
number below is a field default (all example numbers are UNCALIBRATED FIXTURES in `examples/phase3/species.json`).

## 1. What is added

| item | where |
|---|---|
| species-local laws, keyed randomness, evaluation | `leaf/recruitment.py` (law `simple_seed_clonal_v1`) |
| annual boundary API, staged population path | `leaf/interevent.py` (`apply_reproduction`, `decline_reproduction_opportunity`, `_rebind(births=...)`) |
| version-3 complete checkpoint | `leaf/checkpoint.py` |
| newborn background-hazard correction | `leaf/demography.py` |
| example catalog / script | `examples/phase3/species.json`, `examples/phase3/run_recruitment.py` |

Existing `null` and `law: "unimplemented"` reproduction sections remain inactive: nothing is allocated, no RNG seed is needed
and all earlier paths, receipts and byte counts are unchanged. An unknown law, unknown key or malformed value refuses clearly.

## 2. Catalog section (all numbers beside the species' other traits)

```json
"reproduction": {"law": "simple_seed_clonal_v1", "status": "...", "notes": "...", "units": {...},
  "traits": {
    "seed":   null | {"establishment_probability": [0,1], "max_recruits_per_patch": int>=1,
                      "initial_agb_dry_kg": >0, "initial_seasonal_fraction": [0,1]},
    "clonal": null | {"min_age_days": >=0, "min_agb_dry_kg": >0, "probability": [0,1],
                      "max_daughters_per_parent": int>=1, "initial_agb_dry_kg": >0,
                      "max_donation_fraction": (0,1), "dispersal_radius_m": >0, "max_attempts_per_daughter": int>=1}}}
```

`seed` / `clonal` are independent (seed-only, clonal-only, both, neither). Units: kg dry aboveground mass, metres, days.
There is no plant-height-at-establishment field and no default value anywhere.

## 3. Seeds

* **Ecological patch** = explicit session configuration `reproduction_patch_size_m` (metres, square, tiling the domain from its
  south-west origin; east / north boundary patches are clipped to the closed domain). It is independent of the hydraulic cell
  size (the same physical domain at 1 m and 0.5 m cells gives identical patches and identical draws). There is no implicit full
  community per fine cell. Required whenever any species has a seed law.
* **Trial**: per annual opportunity, species with a seed law in sorted species order, patches in row-major order: success iff
  `u < establishment_probability` (per patch, regardless of its clipped area: an explicit assumption), then
  `n = 1 + floor(u * max_recruits_per_patch)` seedlings, positioned uniformly in the ACTUAL clipped patch rectangle.
* **Seedlings**: age 0, `B = initial_agb_dry_kg`, `E = initial_seasonal_fraction * B`, establishment ground = containing-cell
  sample of the supplied current surface (`Domain` rule), status live.
* **External-propagule assumption**: no seed bank, dispersal kernel, light / nutrient / water filter. The seedling AGB is an
  **external establishment / seed-reserve credit** in the ledger, not biomass derived from current transpiration. No rain, ET
  or hidden water withdrawal occurs; native water and time are untouched.

## 4. Vegetative daughters (user-selected distinct ramets)

* Eligible parent: LIVE, `age >= min_age_days`, `B >= min_agb_dry_kg`, `max_donation_fraction * B >= initial_agb_dry_kg`.
* Per eligible parent (keyed by parent ID): success iff `u < probability`; requested `n = 1 + floor(u * max)`, capped so the
  donation never exceeds `max_donation_fraction * B`.
* Each requested daughter draws up to `max_attempts_per_daughter` positions uniformly in a disk (`r = R sqrt(u)`) around the
  parent; the first position inside the closed, **non-periodic** domain is accepted; nothing is clipped, wrapped or forced.
* **Funding**: only accepted daughters are paid. Parent `B -= k * initial_agb_dry_kg`; parent
  `E -= k * initial_agb_dry_kg * E/B` (consistent proportional debit); each daughter starts with exactly that `B` and
  `E = initial_agb_dry_kg * E/B (<= B)`. The parent stays LIVE with `B' >= (1 - max_donation_fraction) B > 0`. No hidden
  belowground pool, no floor, no revival, no extra mortality. Failed placements cause no donation.
* Daughters: new ID, parent's species, age 0, ground sampled at their own position, same canonical `PlantState` row layout.
* Overlap: parent / daughter / neighbour canopies may overlap; this is documented, not tested and not claimed otherwise
  (no all-pairs spacing). The Phase 0 generator still supports overlap and exclusion independently.

## 5. IDs, ordering and randomness

* New IDs: `max(max existing ID, 0) + 1 ...`, no reuse; assigned seeds first (species, patch, recruit index) then daughters
  (parent ID, daughter index). Existing IDs may be zero, negative or the int64 endpoints: positivity is never imposed. Refusal
  on int64 exhaustion happens only when births actually need IDs (a max-int64 plant with zero births is fine) and before the
  birth arrays exist.
* Randomness: counter-based integer hash (the SplitMix64 finalizer of `leaf.demography`) of
  `(seed, reproduction counter, stream, a, b)`; streams 11..24 are domain-separated from the demography streams 1 and 2 (whose
  results are unchanged). `a` = patch id or parent ID, `b` = (species, recruit index) / (daughter index, attempt round).
  Results are by ID: independent of row order, species declaration order and restarts. The counter (= accepted
  opportunities) advances only when a call is accepted; rejected calls consume nothing.

## 6. Calendar, ordering and the session API

```python
session = IntereventSession(plants, catalog, ..., rng_seed=S, reproduction_patch_size_m=P)
rec = session.apply_reproduction(year, current_surface, max_new_plants=, max_total_records=, max_attempts=)
session.decline_reproduction_opportunity(year)       # explicit: the only way to pass an unusable year
```

**Opportunity** `year` = the local calendar year that just completed. Order inside a year: complete prior growth (an open
interval ending 1 January fires the year-end growth; otherwise `flush_growth()`) -> post-event damage (`apply_surface_event`,
no transport diagnostic pending) -> the due annual mortality decision (`apply_annual_decision`) -> `apply_reproduction`. New
plants request water only on the NEXT open interval. Refused (state unchanged, nothing consumed): pending growth; a pending
transport diagnostic; an unresolved annual mortality decision; a replayed / earlier year; a skipped year (years are strictly
sequential from `start.year`); a year not yet completed; a cursor beyond year + 1 (a gap of more than one year is
unsupported: no catch-up and no multiple annual chances; call `decline_reproduction_opportunity(year)` to record the missed
year explicitly, no random state is consumed). The first accepted boundary after a transport may be later than 1 January: the
opportunity is then taken at the cursor (newborn age 0 there); old annual sums are never relabelled and the partial-year
metrics of existing parents are preserved. Bareland / inactive-mortality sessions use the same gate (it is derived from the
calendar, not from plants). Repeated calls never charge two chances.

Zero-live and zero-record domains work (seed-only establishment onto an empty domain). When live plants were absent and a
mortality law is active the decision period restarts at the cursor (the empty period is not inherited).

**Newborn hazard correction (deliberate Phase 2 change)**: `leaf.demography` charges the background hazard for
`min(period_days, age_end)` instead of the whole period, so a plant is never charged for time before its birth; plants that lived
the whole period are unchanged; the Weibull age hazard already used the actual age. Newborn stress counters / annual
accumulators start at zero; existing plants' accumulators are untouched.

## 7. Staged population path and failure atomicity

`_rebind` (Phase 2) now handles status / B changes AND appended births in ONE staged path: new canonical base (existing rows
untouched, births appended with age 0), live-row map, species index, geometry via the shared allometry, roots via the shared
builder (surviving per-plant root overrides kept; daughters and seedlings use their species roots; no explicit per-daughter
policy is supported), new working arrays and fresh zero accumulators for newborns only. Nothing is published until every
stage (including the next-interval admission) passed; `_publish` replaces the boundary once. A failure leaves the old base,
B/E, native state, clock, grid, RNG counter, roots, overrides, annual sums, replay guards and receipts exactly unchanged and a
valid retry counts once.

## 8. Ledger and diagnostics (fixed-size, no per-event history)

Receipt `reproduction`: per-species lists (seed: trials, successes, failures, recruits, seed AGB; clonal: eligible, trial
successes / failures, daughters requested / placed / rejected, donor AGB), totals, attempts used, ID range, and
`ledger`: `old_total_dry_agb_kg + external_seed_agb_kg = new_total_dry_agb_kg` (closure residual with an FP64 bound of
`(32 + 4 log2 n) eps` times the total; the clonal transfer residual `parent loss - daughter gain` is checked separately and
cancels). `e_transfer_residual_kg` reports the seasonal component. Nothing per plant is retained (no parent-ID column is
persisted; the canonical state has no lineage).

## 9. Limits, memory and CPU / GPU

Explicit strict finite integers `max_new_plants`, `max_total_records`, `max_attempts` plus the session's global `max_bytes`
(`reserve_bytes` included). The REQUESTED recruits + daughters are compared with `max_new_plants` (conservative: placement
failures do not relax it); total records = existing (dead / removed included, never deleted) + requested births; attempts =
seeds + requested daughters x configured attempts. Nothing is truncated or raised. Metadata-only preflight precedes every
array (patch trials = patches x seed species; live-sized arrays), a second admission follows the actual counts before the
candidate / birth arrays exist, then the rebind stages (old + new coexistence, builder budget, next-interval fit) apply.
Work arrays are sized by the actual trials, eligible parents, requested daughters and births (prefix sums over counts); no
Python per-plant loop (only a loop over the configured attempt rounds), no plants x grid matrix, no history.

Implementation: a vectorised NumPy HOST reference on every backend (no compiled eligibility / placement kernel yet). On cuda the
session reads B / E back to the host and uploads the rebuilt state: these host transfers are counted in the ledger; the phase
is **not GPU resident**. The rebuilt roots use the selected builder (`compiled` Numba or `numpy`).

## 10. Checkpoint compatibility boundary

Complete checkpoints are **schema version 3**: `restart.json` gains a `phase3` block (flags, patch size, accepted-opportunity
counter, last handled year, last compact record) and the module hashes include `leaf.recruitment`; no new array exists (newborns
are ordinary `plants.csv` rows; next IDs derive from the retained records). The canonical plant CSV stays version 1. Version 1
and 2 checkpoints are REFUSED with a clear message and are never migrated or defaulted, even with `allow_source_change=True`.
A different catalog (hash) or flag mismatch is refused.

## 11. Developer notes (MAPLE / SYRUP; neither model is edited)

See `docs/maple-leaf-integration.md` section 0: re-export the canopy after the boundary, key everything by stable ID, newborns
request water from the next interval, canopies may overlap, native water is unchanged by the call.

## 12. Open limits

Calibration and field recruitment data; seed banks and dispersal kernels; light / nutrient / water filters on establishment;
surface remapping; spacing / non-overlap rules; a compiled or CUDA eligibility / placement kernel; a supported per-daughter
root-profile policy; multi-year catch-up (deliberately unsupported); persisted lineage.
