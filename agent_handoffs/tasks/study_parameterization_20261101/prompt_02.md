# Bounded correction: study_parameterization_20261101 revision 2

Resume the exact authorized candidate session. The first process reached its
900-second execution timeout and was reaped (exit 143); automatic collaboration
remains enabled. Preserve all code/tests/README already written. Your four allowed
write files and four public metadata read paths are unchanged. All governing
scope and public-study contract instructions from revision 1 remain in force.
Codex still owns source retrieval, orchestration, review and execution. You own
only candidate/ and have Read/Write/Edit tools, no shell or tests execution.

Finish report.md (honestly no tests executed by you), implement the bounded
corrections below, and leave all other files untouched. Do not redesign the
analysis, change runtime limits or substitute aggregate observations for raw data.
The original code and tests are preserved in Codex review_snapshot_01; do not edit it.

# Initial independent Codex review — correct

Review snapshot 01 hashes identify the inspected source. Execution of the
synthetic tests with existing pytest: 22 pass, 1 fails. A first invocation used
unittest discovery and ran zero tests; that exit 5 is preserved, not a pass.

## Required corrections before real-data fitting

1. **Nonmeasurement placeholders become false water boundaries.** All 19,889
   raw records with water amount = weight before = weight after = -1 are image
   snapshot placeholders (the author's script excludes water amount = -1).
   `load_weights` currently marks them successful because -1 is finite. A real
   110 g -> 105 g loss with 10 g added to end at 115 g is 5 g. Inserting a -1
   image record makes the current last-day boundary produce -106 g instead.
   See independent_sentinel_check.json. Retain placeholder audit/counts, but
   omit these nonmeasurements from physical interval and boundary chains.
   Other genuinely incomplete measurements must remain explicit. Do not solve
   this by dropping every day containing an image snapshot. Add the independent
   3-row physical counterexample as a regression test, including correct 5 g
   balance and two real weighing boundaries.
2. **The test mismatch is at an exact threshold.** The supplied fixture has
   100 g actual irrigation and 95 g dispenser amount; code flags strictly >5 g,
   but the assertion expects a mismatch at 5 g. Retain the stated >5 semantics;
   use a >5 g fixture (e.g. 94 g dispenser) and update its dispenser-only expected
   sum. Physical loss remains 120 g. Add a boundary assertion if helpful.
3. **Use the published water-artifact exclusions as a documented sensitivity.**
   The archived author's water script excludes nonpositive daily losses,
   losses >200 g on DAP 17, 32 and 33, and dry-treatment DAP32 losses >145 g.
   Current generic 600 g daily cutoff silently accepts these study-known
   artifacts. Implement explicit, counted study-QC reasons and compare with a
   retained-flagged sensitivity. Never clip measured losses, hide exclusions or
   treat a flagged 300 g reading as calibrated transpiration. Date labels must
   match the source's local-calendar DAP convention.
4. **Keep parameter estimation separate from withheld observations.** The
   trait-support prior, T/ET sensitivities and genotype bootstrap for candidates
   must use TRAIN genotypes; keep all/heldout summaries descriptive. Currently
   pooled-all bootstrap is the only bootstrap. State that interval bootstrap
   conditions on the fitted image calibration unless that calibration is also
   refitted. Do not present fixed-allometry intervals as complete uncertainty.
5. **Evaluate the growth replay against actual withheld harvests where dates
   match.** The current endpoint metric is against image-derived mass only;
   separate allometry holdout metrics do not test growth-predicted harvest mass.
   Join predicted trajectory endpoint to withheld `dry_wt` only when the endpoint
   DAP equals that plant's last-image/harvest proxy DAP. Report eligibility and
   exclusions, retain image-shape metrics under an explicit label, and do not
   use heldout harvests in fitting or capacity selection. Last image as harvest
   date is the source's sampling proxy, not a verified exact destructive date.

## Interpretation to preserve

The mesquite aggregation/formulas agree with independent scalar calculations.
Retain all conditional labels: 38/76 Pa separate; two-day endpoint water does
not supply 39-day cumulative T; AGB increment allocation is not standing AGB
fraction. The large fitted rate is a normal-only seedling diagnostic. Seasonal
growth and adult size cannot be identified by these short observations.
Do not change a simulator, source lock, default or runtime rate bound.

Clarification of item 3: use the explicitly counted published data-quality policy
as the primary analysis, and retain a looser flagged-data summary as sensitivity.
Nonmeasurement -1 image records are not genuinely failed physical measurements;
exclude them from physical chains, retain their raw counts, and connect the
surrounding actual weighings. For other invalid/incomplete measurements, preserve
explicit reasons and refuse extrapolation across missing physical daily intervals.
At minimum the 3-row real -> image-placeholder -> real example must conserve 5 g
and avoid making an image placeholder a daily boundary.

Only adopt candidate values from TRAIN observations and TRAIN bootstrap. Pooled
and heldout summaries may remain clearly descriptive. The graph/figure code
should keep image-derived endpoints distinct from true withheld harvest mass.
