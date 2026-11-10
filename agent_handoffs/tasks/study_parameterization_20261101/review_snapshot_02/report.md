# Implementation report — study_parameterization_20261101, revision 2

Implementer: Claude (candidate/ only). Orchestrator/reviewer: Codex.

## Execution status (honest statement)

* I had **no shell** in either session. I did **not** run `study_parameters.py`,
  `pytest`, or any import check. Every statement below about behaviour is from
  reading the code, not from observed output. Syntax or runtime errors may exist;
  Codex executes the CLI and tests independently.
* Revision 1 session was reaped at the 900 s timeout (exit 143) after the code and
  tests were written. My context shows no README/report written then, but the Write
  tool reported `README.md` as "updated" rather than "created" in this session; if a
  revision 1 README existed, it has been replaced by the complete revision 2 README
  (Codex's review_snapshot_01 preserves any prior version). `report.md` is new. Codex's independent run of the revision 1 tests reported 22 pass,
  1 fail (the exact-threshold dispenser fixture); the earlier unittest-discovery
  invocation that ran zero tests (exit 5) is not a pass.
* No raw data tables were read through the model context; only the four permitted
  metadata files in `../public_evidence/` plus this prompt. `../baseline.json` and
  the review snapshot were not read (not on the permitted list).
* No simulator, source lock, default, or runtime `r_year <= 1` bound was changed.

## Revision 2 corrections (all implemented in `study_parameters.py` / tests)

1. **Image placeholder rows.** `load_weights` now marks rows with
   `weight before = weight after = water amount = -1` as `placeholder` (counted) and
   rows with any single `-1` as `sentinel_partial` (genuinely incomplete, stays in
   the chain and blocks it). `physical_records()` removes placeholders before interval
   reconstruction and boundary formation, so the real weighings around a placeholder
   connect directly and a placeholder can never be a daily boundary. Regression test
   `test_image_placeholder_rows_do_not_form_boundaries` encodes the independent 3-row
   counterexample (110 → placeholder → 105/115, +10 g): one interval, one boundary
   (DAP 20→21), balance 5 g, closure 0, boundary weights 110/115, placeholder count 1.
   Days containing an image snapshot are **not** dropped.
2. **Dispenser mismatch threshold.** Semantics retained as strictly `> 5 g`. Closure
   fixture now uses 94 g dispenser vs 100 g actual irrigation (physical loss still
   120 g; dispenser sum 94 g). New boundary test: exactly 5 g is not flagged, 5.1 g is.
3. **Published study QC as primary.** `study_qc_reasons()` implements the archived
   policy with counted reasons: `qc_nonpositive_daily_loss`,
   `qc_dap{17,32,33}_loss_gt_200g`, `qc_dry_dap32_loss_gt_145g`. `daily_boundary_balances`
   emits `complete_daily` (primary: complete 1-day chain + QC pass) and
   `complete_daily_sensitivity` (retained-flagged: complete 1-day chain, 0 ≤ loss ≤ 600 g,
   QC-flagged days kept unclipped). The generic 600 g cutoff is now used **only** by
   the sensitivity selection. Audit reports per-reason counts, primary vs sensitivity
   counts and the excluded loss total. Losses are never clipped.
   **Unresolved convention flag:** I could not read the author's script, so the DAP a
   daily loss is labelled with is a parameter: `--qc-day-label end` (default; the day
   of the weighing that registers the loss, i.e. the period's `dap_end`) or `start`.
   Codex should verify which matches the source's local-calendar convention; the test
   covers both labels (DAP 17 = 2014-01-30).
4. **TRAIN-only candidates.** `candidate_rate_block()` computes pooled q, C_balance,
   production ratio, by-treatment, T/ET sensitivities and the genotype bootstrap from
   TRAIN periods only; `candidate_fragments.json` nests these under
   `water_rate_and_support.candidate_train`. All-plant, heldout, retained-flagged and
   multi-day summaries sit under `descriptive`. The block carries an explicit note that
   the bootstrap conditions on the single fitted image calibration (k is not refitted
   per resample) and is therefore not complete uncertainty. STEP's "water balance C for
   comparison" now uses the TRAIN pooled value. Test
   `test_candidate_block_ignores_heldout_periods` perturbs heldout rows and asserts no
   candidate number moves.
5. **Harvest replay evaluation.** `evaluate_step_against_harvest()` joins each withheld
   trajectory's replay endpoint to the plant's own `dry_wt` only when `end_dap` equals
   the plant's last-image DAP (reported as the source's sampling proxy, not a verified
   destructive date); eligibility counts (no harvest row, missing dry_wt, missing last
   image DAP, DAP mismatch, eligible) are emitted for bounded and unbounded fits, with
   metrics in kg and per-genotype medians. Heldout harvests are not used in fitting or
   M-scenario selection (selection remains TRAIN SSE). Image-derived endpoint metrics
   are retained under `heldout_image_derived_shape_*`. The figure draws image-derived
   endpoints as filled markers and harvest matches as hollow circles. Because the
   STEP trajectory ends at or before DAP 33 while many last images may be later, I
   expect a sizeable `n_endpoint_dap_mismatch`; this is reported, not hidden.

## What the code does (unchanged design, summarised)

Calendar-day DAP from naive timestamps; exact-barcode genotype/treatment parsing;
SHA256-mod-5 genotype holdout; zoom calibration `a·exp(b·camera_zoom)` to all table
rows; SV areas averaged per barcode/day with out-of-bounds counts and an in-bounds
sensitivity; zero-intercept dry-mass coefficient on TRAIN only with heldout metrics
and the three harvest IDs lacking water joins listed (`harvest_ids_without_weight_join`,
not fixed); interval and daily-boundary water accounting with closure checks; window
DAP 17–33 with ≤ 3-day interpolation and no extrapolation; q / C_balance / production
ratio with units checked by test; STEP diagnostic with bounded vs unbounded r,
M scenarios from TRAIN, profiles over r and C; Peterson pot aggregation with
conditional labels; outputs refuse existing directories.

## Checks I could not perform

* Whether the real tables parse as expected (timestamp formats, `completed` spelling,
  `in_bounds` values, zoom labels) — guarded with coercion and audit counts.
* Whether `scipy.optimize.least_squares` converges from every start on real data; the
  best-of-starts SSE is used and bound hits are reported (`r_at_upper_bound`, `C_at_bound`).
* Numerical values of any candidate fragment. None are reported here because none
  were computed by me.
* Whether `n_eligible` for the harvest replay is non-zero on the real data.

## Known limitations / flags for review

* Mass at a period end uses the image of that calendar day; imaging and weighing
  times within the day differ (hours), not corrected.
* Empty pots are identified only as barcodes failing the genotype pattern; reported
  as an evaporation reference, never subtracted.
* The unbounded fitted rate is a normal-only seedling diagnostic; seasonal growth,
  adult size (M), canopy density, turnover, phenology and rooting layers are not
  identified by these observations.
* Peterson figure readings carry reading resolution, not sampling uncertainty.
