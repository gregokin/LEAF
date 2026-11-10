# Public-study candidate parameter analysis contract

Implement a small reproducible Python analysis using existing NumPy, pandas,
SciPy and Matplotlib. This is an isolated study analysis; no simulator or field
defaults are being changed. Never invent raw observations or independently
identified traits. No shell/network execution is assigned to the implementer.

## Inputs (all public)

Feldman et al. 2018: DOI 10.1104/pp.18.00146; data DOI 10.5281/zenodo.1117062;
software/data tag v1.0, 8c88886b7f6ce3e0453eaba74ccadb0f319d3b1d.
The scalar image table retains every row of the original 590 MB CSV and omits
only long landmark-list columns. Input paths will be supplied as CLI arguments:

- `--images`: projected CSV, including timestamp, frame, lifter, cartag, zoom
  (`z1` etc), plantbarcode, camera (`SV` or `TV`), area, hull-area, in_bounds.
- `--weights`: original CSV columns `plant barcode`, `timestamp`, `weight before`,
  `weight after`, `water amount`, `completed`, `car tag`. Weights are grams.
- `--harvest`: plantbarcode, fresh_wt, sat_wt, dry_wt (grams), cartag.
- `--zoom`: tab-separated calibration file, zoom, camera, rel_area and geometry.
- `--peterson`: JSON with approximate published figure readings/units/limits.
- `--out`: a new output directory. Preserve existing results; refuse overwrites.

## Required analysis

1. Planting date is 2014-01-13, timestamps lack timezone; derive local calendar
   days consistently, not machine-local POSIX time. Genotype can be parsed as
   `Dr<number>A[AB]...`, parents `Dp1`/`Dp2`; AA = wet, AB = dry. Use exact barcodes;
   retain/report the three harvest IDs lacking water joins rather than fix them.
   Declare deterministic genotype holdout with SHA256 (e.g. hash modulo 5 == 0),
   keeping all replicates, treatments and dates of a genotype together.
2. Normalize side-view pixel area for lens zoom. Match the author's basic area
   calibration: camera_zoom = 1 + 5*(raw_zoom-1)/5999; fit `a*exp(b*camera_zoom)`
   to all tabulated relative areas using nonlinear least squares (not absolute
   cm2); divide image `area` by predicted relative area. Average valid positive
   SV areas across rotations per barcode/day. Preserve/report out-of-bounds
   images and sensitivity; do not silently use zero/negative/sentinel areas.
3. Join final-day SV area to shoot dry harvest mass by barcode. Fit a positive
   zero-intercept dry-mass/area coefficient on TRAIN genotypes only; evaluate
   heldout dry-mass predictions. Do not anchor each trajectory to its own final
   harvest. Use this one calibration for all earlier area-derived mass estimates.
   Distinguish image-derived trajectories from independent harvest measurements.
4. Reconstruct measured pot mass loss at each consecutive successful weighing:
   `loss_g = previous_weight_after - current_weight_before`. Actual interval
   duration matters; separate irrigation mass `current_after-current_before`
   from reported dispenser water. Derive daily boundary water balances between
   consecutive last-weighings (sum actual irrigation mass minus storage change),
   which must equal the sum of interval losses. Mark incomplete/negative/unusual
   records explicitly, with counts and reasons. Do not clip negative losses or
   use dispenser amount as transpiration. No extrapolation across missing days.
   Keep enough audit totals to verify the mass balance before fitting.
   g water -> m3 is 1e-6. Empty pots, if identifiable, are evaporation controls,
   never plants. The paper warns that subtracting empty-pot losses can overcorrect.
5. Primary study window: DAP 17 through 33. Form per-plant daily water rates and
   dry-mass estimates, with interpolation only between neighboring observations
   (maximum 3-day image gap). No extrapolation. Select complete intervals and
   disclose exclusions; do not require the model to explain QC artifacts.
   Pot water loss is ET; use explicit T fractions of ET [1, .9, .8] as conditional
   sensitivities, not measured transpiration. Avoid claiming an ET->T calibration.
6. Estimate study-supported biomass-normalized water rate q = U/(B*days), units
   m3/kg/day, with B at matching times. Compute the annual balance support prior
   `C_balance = integral(B dt)/(365*sum(U))`, kg/m3; this is a support threshold,
   **not** new biomass per water. Also report the empirical production ratio
   delta_B/sum(U) under its own name, with correct common intervals.
   Summarize individual variation/treatment and bootstrap uncertainty using
   genotype as the resampling unit. Fixed seed; bounded resamples (100-200).
7. Fit a diagnostic STEP-style normal-growth rule to TRAIN trajectories, using
   actual period water, 7-day updates and a final partial period:
   `required = B*dt/365`; `available = C*U`; `wf = min(1,available/required)`;
   `gain = .95*r_year*(dt/365)*wf*B*(1-B/M)`; cap gain at M-B. No seasonal extra.
   Activity=temperature=live fraction=1 is an explicit experimental assumption.
   Use positive bounds for C and r, and fixed M scenarios from TRAIN mass only,
   e.g. 2x, 4x, 10x maximum training endpoint mass. Don't label M as mature mass
   inferred from a seedling harvest. Select scenarios on TRAIN only. Report heldout
   trajectory/endpoint error with genotype separation. A small profile of C versus
   r or equivalent diagnostic must expose tradeoffs instead of a falsely precise
   unique full parameter vector. Report water balance C and fitted C separately.
   Existing runtime accepts r_year<=1; explicitly compare a bounded-r fit with
   a positive unbounded diagnostic fit. Do NOT hide a rate-cap failure by using
   seasonal extra biomass, changing observations or claiming runtime compatibility.
8. Peterson & Neofotis 2004: each source pot had two seedlings; the user selects
   one aggregate modeled plant per pot. Source figure 5B reports per-seedling
   water over LAST TWO DAYS. Aggregate = source water*2; divide by two days for
   daily rate. Relative growth and mass/water ratios do not double. No 39-day
   cumulative water series or absolute aboveground mass was found; do not invent it.
   Published figure 2 whole-plant RGR and root/stem/leaf component RGRs are intensive.
   Figure 4C is log time-averaged transpiration productivity (g/g/day), defined
   from two-day water masses and the 39-day growth interval, not cumulative WUE.
   Under the paper's log-mean growth-analysis identities, a conditional total-mass
   balance support proxy is `2*1000*exp(ln_Eprod)/(365*RGR)` kg/m3.
   AGB proxy may multiply by `1-root_CRGR/RGR` ONLY under an explicit stable
   allocation approximation; this increment allocation is not observed standing
   aboveground fraction. Store both the total-mass value and conditional AGB proxy.
   Separate 38 Pa baseline and 76 Pa CO2 sensitivity, and low/high water treatments.
   Figure reading resolution is NOT sampling uncertainty or an individual CI.
   Compare 39-day observed growth factor to maximum normal growth with r_year=1,
   and calculate the minimum weekly-Euler equivalent rate (far below mature size,
   water=temperature=activity=1) needed to reach that factor. Label whole-plant
   growth versus conditional AGB assumption. Absolute mature mass, canopy density,
   seasonal turnover, annual phenology and rooting layers remain unidentified.

## Deliverables

Write only `study_parameters.py`, `test_study_parameters.py`, `README.md`, and
`report.md` in candidate/. A CLI should emit machine-readable JSON candidate
fragments and diagnostics, normalized period/plant tables and a compact PNG of
heldout biomass predictions and rate/support identifiability or growth-cap
comparison. No public-data defaults or private local source imports.

Make the key water-accounting, genotype-split, trait-fit and pot-aggregation
functions directly testable with small synthetic inputs. Tests should include
irrigation/storage closure, missing/negative interval handling, split leakage,
kg/m3/day conversion, pair aggregation preserving intensive ratios, and rate-cap
incompatibility; don't mirror the implementation as the oracle. The orchestrator
will execute analysis/tests and independently review results. Report unexecuted
checks honestly. Explicitly flag any method/data issue you cannot resolve.
