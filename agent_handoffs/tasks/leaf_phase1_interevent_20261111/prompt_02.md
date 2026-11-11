# Phase 1 session revision2 — independent guard and memory corrections

Resume exact session ecdc127b-6348-4df5-8c50-87404f7afa8e. Preserve saved D1;
do not restart. User authorizes automatic relevant source sharing/review/corrections
and whole-phase publication by Codex. Sole writer in THIS isolated candidate;
Codex executes/reviews/integrates/publishes. This remains Phase1 only.

READ only this prompt and candidate/**; SOURCE_INDEX.txt lists all files. Ordinary
Read/Edit/Write only: no execution/Bash/agents/network/settings/installs/Git or
outside/archive reads. WRITES exactly leaf/interevent.py, leaf/checkpoint.py,
tests/test_interevent.py, docs/phase1-interevent.md, examples/phase1/run_interevent.py,
README.md and report.md. Component modules remain read-only. No new physics or
species-parameter changes, no source-guard relaxation or upstream modifications.

## Baseline and actual independent verification

D1 exited0/reaped, structured success, scope violations0/toolerrors0. Its allowed
changes were composed with final A4/B4/C3 outside the candidate. Full350CPU
tests pass (330pass/20GPU skips); all73 session/restart tests pass on actual
selected CUDA. An independent10-day actual-native fixture crosses February29,
restarts midweek after day3 and reproduces every plant field/native water exactly.
These are Codex's executed checks; your checks remain NOT RUN.

The candidate's READ-ONLY component tests are refreshed to B4/C3 (production
sources unchanged) and reviewer source supplied before this revision, in a NEW
baseline_revision_02.json. Do not attribute those Codex context copies to yourself.
Exact synthetic reviewer files, read FILES not their directory:
review_reference/verify_session_guards_02.py, review_reference/early_session_smoke.py,
review_reference/profile_snapshot_01.py. They use REAL native APIs and only
synthetic data, no saved terrain/weather. Review notes below describe actual
reproductions, not conjecture. If your inspected code differs, report exact proof.

## Confirmed corrections

1. **Scalar/global admission before copying.** __init__ validates/copies the whole
   PlantState BEFORE parsing max_bytes. With max_bytes=0, Codex wrapped the real
   PlantState.copy and observed a population copy before refusal. Preflight scalar
   arguments and an appropriate metadata-derived lower bound before validate/copy,
   P-sized selection or transfer. Then admit actual live counts/candidate scratch.
   Validate load's max_edges/max_bytes/reserve types/ranges before payload work too;
   no bool/float truncation or negative reserve reducing an admission requirement.
   Keep invalid input and old state untouched, no automatically raised budgets.

2. **Do not coerce root overrides into different plants.** Current _setup uses
   int(key)/float(value) before the strict shared validator. Independently accepted:
   {True:[.5,.5]} becomes plant1, {1.9:[.5,.5]} becomes plant1;
   {1:[.85,.15],1.9:[.1,.9]} silently collides and overwrites plant1;
   {1:[True,False]} becomes [1.,0.]. Refuse these before coercion. Reuse the
   strict shared validator or equivalent scalar checks AFTER memory pre-admission;
   preserve IDs/types, exact L, finite nonnegative unit-sum rows and no mutable
   caller alias. Normal valid individual overrides must still survive rebuild/restart.

3. **Pending accumulation must be finite before publication.** Actual-native
   one-plant fixture with finite daily D=1e308, B1kg, q1e308m3/kg/day, reference
   PET1, all-month activity1, Topt25C, roots[.5,.5] and ordinary finite soil stock
   succeeds on day1. Day2 publishes pending_requested=[inf] and advances cursor.
   No growth boundary was due, so growth.grow did not catch it. Validate ALL new
   D/U/A/T values, totals/relations and pending duration BEFORE publication.
   Prefer fused compiled accumulation+diagnostic packet on Numba/CUDA, with only
   fixed/small packets read back; no daily population readback or Python P loops.
   Count its outputs/scratch/packets in the global ledger. A failure after pure
   native staging preserves the ENTIRE old boundary; a valid retry accounts once.
   Pass the allocator's actual overdelivery_bound_rel(root_map) consistently to
   growth and pending validation (plus justified accumulation roundoff), not a
   narrower unrelated tolerance that could reject its honest accepted feedback.

4. **Enforce end-to-end water closure.** Codex wrapped ONE REAL native ss.step,
   perturbing only reported evaporation_m3 by .001. The bridge physical T still
   closes, native stage continuity passes, but the session returns
   residual_within_tolerance=False, residual .000999999999999952 and PUBLISHES
   that boundary. Check finite total residual/tolerance and REQUIRE closure before
   publication, not merely a diagnostic flag. Reject malformed inconsistent native
   budgets; never balance water or loosen the physical FP64 policy. This is a
   fault-injection contract test, not a claim real ss.step naturally produced it.

5. **Native time tolerance must reflect FP64 precision.** With native start1e9s,
   daily rain handoff ending1e9+86400.5 is accepted despite a .5second clock error.
   TIME_TOL_REL1e-9 permits ~1second there. Use a documented justified FP64/ULP
   bound tied to the actual interval/native clock, with microsecond calendar
   precision; no arbitrary tolerance that grows to seconds over30years. Preserve
   exact cursor and refuse unsupported rational instants, no rounding them away.
   Real native step endpoint calculations and split-run restart must still pass.
   Full storm adapters with other clock policies require separate qualification.

6. **Snapshot/global memory ledger is too low on CUDA.** At P100000, exact
   observed simultaneous ADDITIONAL host array-data lower bound is14800000B,
   declared14000000B (global31900906 with resident17900906). Confirmed omission
   >=800000B, not Python overhead, pool rounding or tracer metadata. The wrapper
   calls REAL NumPy unique and inspects actual constructor/materializer arrays
   and returned unique output while all are live. CPU lower bound13200000 fits
   its14000000 declaration; CUDA additionally holds the B/E host copies. Count
   full PlantState materialization/validation scratch AND device-to-host buffers,
   on snapshot and every other caller of _materialize, not only the final owned
   population. Do not insert the fixture's800000 as padding; derive a structural
   per-array bound from the actual shared state implementation. No cap increase.

7. **Audit accumulation/global coexistence.** The returned wd still owns request
   depth, accepted-volume and availability buffers, native outputs, plan/feedback
   while D_new/U_new are allocated. Current manual accumulate estimate appears
   to omit the two LN depth/volume conversion buffers. This is a desk inventory
   concern, not yet a measured global-peak defect. Use the full bridge ownership/
   estimates/reserves plus new accumulators rather than reconstructing an incomplete
   subset; count old/new S integrals, season carry, native states, root graphs and
   all host/device transfers. Audit stages truthfully; conservative double-counts
   are fine if labelled. Codex is profiling the frozen old tree independently.

## Tests/docs/report

Add meaningful regressions for the actual cases, failure boundary preservation,
valid retry, native call counts and corrected snapshot admission before work.
Retain every successful nominal test and actual restart/transport semantics.
No need to re-prove the already passing science by rewriting component modules.
Keep native provenance/source guards fail closed by default; do not replace them
with warnings merely because another upstream branch may later change.

Save code incrementally. Update report with actual final signatures/units/memory
scope, any tool errors and ALL tests NOT RUN; execution belongs to Codex. Leave
uncommitted. Codex will freeze and independently rerun CPU/CUDA/native/guard/
restart/global-memory checks, then correct again if needed. No whole-phase
acceptance or Phase2 implementation until this integration is accepted.
