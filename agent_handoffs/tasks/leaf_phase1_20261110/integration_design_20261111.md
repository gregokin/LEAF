# Phase 1 integration acceptance specification — continuation

This is Codex's review/design specification, not implemented behavior or phase
acceptance. Approved science choices are recorded in decisions_20261111.md.
Candidate writers remain isolated. Do not overwrite earlier design/history.

## Responsibilities and transaction

The vegetation session composes actual native SYRUP water with LEAF demand,
uptake, growth and roots. It must not copy water physics. Non-transpiration
hydrology is a separate explicit native stage, never hidden evaporation or a
second LEAF water store. Shared scheduler forcing supplies the actual date,
temperature and PET; SYRUP supplies its evaporation forcing/partition. In
particular, do not replace the upstream-approved combined-loss-limited E share
with a fresh canopy-times-PET evaporation law in LEAF.

One accepted boundary contains the native water state, authoritative plant
biomass/seasonal mass and pending integrals, the root map, exact calendar cursor
and replay receipt. Stage every candidate first; publish with one boundary
replacement only after native acceptance, plant gathering, growth and any due
root rebuild all succeed. A failed stage preserves the previous complete
boundary. Retrying a rejected interval cannot double-debit or advance age.

The pure native bridge is not itself an exactly-once scheduler. The session must
reject interval replay/overlap/gaps and stale staged water. Full MASY transport
adoption remains an upstream integration qualification, not something established
by toy interevent tests. Tests must use the actual native API, not fake subtraction.

## Dates, gates and growth periods

Use offset-aware Gregorian instants, split at midnight, weekly geometry/growth
boundaries and new years. Preserve or refuse MASY rational submicrosecond time
before mutation; the current neutral forcing object supports microseconds only.
Track elapsed calendar time separately from active biological integrals. Every
wind/rain/recession interval advances time and live-plant age, with no T, growth
or catch-up for its duration. A closed biological gate likewise contributes no
water demand or growth integral.

Flush a partial active growth period before entering transport so consumers see
the appropriate canopy geometry. Keep the next calendar growth boundary explicit.
Any seasonal turnover deferred because a boundary fell inside transport must be
recorded and applied at the next allowed biological boundary, without attributing
transport time to growth. Do not invent water use for deferred time.

## Plant and backend state

The existing PlantState CSV remains the human-readable initial/snapshot/final
format, with stable IDs, species, XY, establishment ground, dry AGB, seasonal mass,
age and status. GPU biomass/pending uptake stay device resident between outputs
and geometry updates. A host CSV table is a synchronized snapshot, not a competing
biomass authority. Account for live ages without a daily GPU population readback.

Weekly rebuilds use the ONE canopy/catalog allometry implementation. Its current
host geometry path requires explicit bounded population transfers; measure and
report them. Root layers and individual overrides survive rebuild/restart. Hold
old and candidate root maps plus their scratch in the global admission budget.
General custom nonseparable maps require an explicit replacement/rebuild policy;
do not silently replace them with species-separable default roots.

Daily T runs against native residual/mask availability, without compensatory
uptake. The growth kernel plans using requested-water support and applies actual
delivered/requested once. Actual support/maintenance deficit remains diagnostic
for Phase 2. No field calibration or direct WUE yield is implied by fixture traits.

## Restart and evidence

A plant CSV alone is not a restart. A complete session envelope must bind actual
native water and time, plant state, pending D/U and species activity/temperature
integrals, exact cursor and next growth/season boundary, root graph and overrides,
trait/source identities, backend and accepted receipt. Serialize only current
state, not daily histories; reject inconsistent hashes/shapes/times before adoption.
Atomic output must not expose a partially written restart.

Independent tests: leap/partial/calendar boundaries; the user's 75% uptake and
75% planned growth; native shortfall applied once; shallow/deep species behavior;
shared canopy dimensions; transport gates and age; replay refusal; native/gather/
growth/root failure preservation and retry; split-run restart equivalence;
CPU/CUDA semantic parity and truthful stage/global admission including transfer
buffers. Production-scale timing/memory claims require explicit measurements of
the new engine, not reuse of old benchmark results.
