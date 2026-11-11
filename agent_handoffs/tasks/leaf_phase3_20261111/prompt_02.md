# Phase3 revision2 — bounded independent-review corrections

Same task leaf_phase3_20261111, exact session a5c6d72f-e529-49ec-beea-b376caaa8814.
R1 completed/reaped, scope/tool errors0; snapshot_01 preserved. User authorizes this
automatic correction loop and relevant Anthropic source sharing; no new science.
You remain sole implementation writer. Ordinary Read/Edit/Write ONLY; no execution,
Bash/agents/network/settings/Git/installs/outside reads. Read this prompt and candidate/**.
New review_reference/** holds bounded synthetic independent probes and literal results.
Original baseline/prompt/snapshot immutable, baseline_revision_02 records refreshed context.
Write ONLY:
leaf/recruitment.py
leaf/interevent.py
leaf/checkpoint.py
leaf/demography.py
tests/test_recruitment.py
tests/test_interevent.py
examples/phase3/run_recruitment.py
docs/phase3-reproduction.md
docs/maple-leaf-integration.md
README.md
report.md
All other files read-only. Author checks remain NOT RUN. Save incrementally, uncommitted.

## Executed evidence

Full CPU509 (39 CUDA skips) and actual selected-GPU509 (one multi-device skip) both
fail ONLY Law.test_parse_inactive_and_active: coefficients('moss_inactive') returns
zeros rather than refusing an inactive species. Everything else passed. Independent
actual-native recruitment/fault rollback/exact restart cases pass on Numba AND CUDA,
including initially zero records. Partial-year annual metric preservation and surviving
individual root overrides/newborn species profiles pass on Numba. Newborn hazard golden
values pass. Actual legacy versions1/2 refuse before decoding, with/without override.
These passes do NOT excuse the following failures.

## 1. Standalone API guards (probe_recruitment_01.py)

- Duplicate live_ids accepted; require unique canonical int64 IDs (ANY sign incl0).
- max_existing_id9 accepted with live ID10, leading to newborn ID collision. Refuse
  global metadata inconsistent with provided live IDs.
- n_total_records0 accepted with three live rows. Require total>=live and coherent empty
  max-ID metadata. Existing IDs/dead records are never renumbered/deleted.
- Wrong patch type{} gives AttributeError; require PatchConfig (or None only where
  permitted) and raise ReproductionError before allocation.
- Metadata-before-allocation: don't call np.ndim/shape on arbitrary list-like live_ids
  before confirming typed ndarray; those functions can materialize unadmitted arrays.
- Validate/count intermediate sums/products with safe integer arithmetic so work limits
  cannot wrap in int64; huge/tiny patch configuration must refuse clearly before arange.
- coefficients must refuse a species whose seed/clonal pathways are BOTH inactive,
  matching its existing test. Do NOT weaken the expectation/test.

## 2. Real prebirth seasonal-loss defect

probe_deferred_season_01.py starts with ZERO plants, closed gate across Dec31->Jan1,
then recruits four shrubs with B=.02/E=.01 each. Birth is AFTER the new year, yet the
NEXT open interval applies deferred_season from BEFORE birth to all recruits: total B
falls .08->.04 with seasonal loss .04. This is biologically out of order, despite the
algebraic biomass ledger closing. New tissue must not pay a prior-season loss.

Use an explicit shared, admitted, transactional prior-season-finalization boundary
before births. A small `flush_deferred_season` API (or carefully documented extension of
existing flush) can reuse _growth_update(year_end=True), with NO new water/time/ET,
NO growth-grid reset, no catch-up and full failure rollback. It must also safely clear an
empty-population deferred flag. Require it before apply_reproduction if a deferred season
remains; make the error actionable. Document/test the scheduler order: pending growth,
post-transport damage, deferred old-season finalization, due mortality, then reproduction.
Preserve existing inactive Phase0/1/2 behavior and tests; prefer a new explicit API if
extending flush would change accepted contracts. Do NOT silently zero newborn E or discard
old seasonal state, add a hidden biomass floor, or exempt all old plants from true turnover.
If another minimal solution is better, explain and prove no prebirth loss and rollback.
The independent old probe may now refuse reproduction until explicit finalization: that is
acceptable; Codex will update the checker to the documented API and retain original failure.

## 3. Strict restart-state semantics (probe_restart_metadata_01.py)

All four invalid-but-content-hash-consistent manifests were accepted:
- n_reproduction=2**64;
- n_reproduction1 although no opportunity was handled (lastyear=startyear-1);
- last_opportunity_year2022 before startyear2024;
- invented last_reproduction={'year':2023} before any handled opportunity.

Validate bounded RNG counter, original startyear-1<=lastyear<=cursor.year-1,
accepted count<=number of sequential handled years, and coherent required last-record
structure/counters/cursor/sequence/declined flags. Initial state has count0 and no last
record; handled opportunities have a matching lastyear and a structurally valid record.
Explicitly declined years do NOT consume RNG, so accepted count may be less than handled
count. Refuse before returning/adopting a session, keep strict source/native/root/type
checks; no silent default/migration. Keep draft format3 (not yet published).

## Acceptance

Add focused regressions for these defects, including empty AND live pre-season cases,
zero E/no-op, donor E transfer, retry/failure exactness and restart with the new ordering.
Preserve existing suites/physics/parameters/caps/native authority. Update docs/report with
exact API, limits and all author checks NOT RUN. Codex independently reruns final tests,
real native probes and actual memory profiling before publication.
