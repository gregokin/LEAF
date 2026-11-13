# Independent review and acceptance matrix

This is a review checklist, not an acceptance receipt. No requirement is passed
by an author report alone.

- Source pairing: LEAF baseline 9fd4bbb, explicitly reused C1 PET/calendar at
  52f39fe, new connection to published SYRUP a5b8b18 / N15 82e9c011 and actual
  native MAPLE 9f bundle fbc504bb. Preserve historical locks and guards.
- Public daily sequence has no plant-side hydrology, physical debit or clock
  advance; fresh native availability and both area conversions are explicit.
- Bind plan, actual feedback, biological candidate and native instance/receipt
  to one nominal date, accepted biology generation and resource generation.
  Equal-scalar alternate branches, stale/replayed/wrong feedback must refuse.
- Preserve exact demand/growth equations and zero-demand policy. PET uses sparse
  canopy overlap; root allocation uses normalized roots independently, with no
  compensation or priority. Check shallow/deep, competition and inactive cells.
- Nominal live-plant age/calendar and scheduled biological duration are supplied
  explicitly; native seconds never become nominal dates. Events earn no later
  E/T/growth credit. Independent T/growth flags and pre-disabled/event flushes.
- Staging, late growth/root/canopy failure and discard preserve the accepted
  plant state and counters. Exactly-once publication and explicit external
  ownership of any already-produced physical candidate.
- Versioned biological checkpoint saves pending requested/actual uptake,
  integrals, root/canopy identities and relevant counters; preserves plain plant
  CSV plus companion. Pending-withdrawal resume must not repeat a debit. No soil
  inventory is serialized/restored by LEAF; external lifecycle binding required.
- Required seasonal, annual demography/reproduction and surface-event boundaries
  are exposed and never silently skipped. Existing laws/parameters preserved;
  no scientific extension masquerades as an interface implementation.
- Memory preflight includes accepted/candidate coexistence and all retained
  plan/feedback arrays. No dense plant-by-cell tensors or routine P/E-sized
  device-to-host transfers; cold geometry/checkpoint transfers declared.
- Tests: NumPy/Numba CPU, actual CUDA, daily/weekly growth, rollback, bad/stale
  identities, restart, scalar-demand regression and old standalone source/format
  guards. Report unavailable paths separately, never make a substitute claim.
- Actual native 4x6 harness: one daily hydrology operation, one zero-time accepted
  T request, actual delivery equals physical debit, closure inside native bound,
  no extra native time/hydrology count. CPU/CUDA agree under existing tolerances.
- Entire trial case, old reproduction code/tests, parameters and historical
  accepted evidence remain byte-identical. No upstream writes or installs.
- Developer handoff uses complete real public signatures, arrays, units,
  ownership, resume stage identities and the preserved Phase 3 resumption point.
