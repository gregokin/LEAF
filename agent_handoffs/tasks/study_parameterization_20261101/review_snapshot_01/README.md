# study_parameterization_20261101 — candidate analysis (revision 1)

Bounded, reproducible public-study analysis of Feldman et al. 2018 (Setaria RIL
phenotyping; DOI 10.1104/pp.18.00146, data DOI 10.5281/zenodo.1117062) plus the
Peterson & Neofotis 2004 figure readings. It produces *candidate parameter
fragments* with audit totals. It changes no simulator or field defaults.

Files in this directory (the only files owned by the implementer):

| file | purpose |
|---|---|
| `study_parameters.py` | analysis library + CLI |
| `test_study_parameters.py` | pytest unit tests on small synthetic inputs |
| `README.md` | this file |
| `report.md` | implementer's honest status report (checks **not** executed by the implementer) |

Dependencies: Python 3.9+, NumPy, pandas, SciPy, Matplotlib (Agg backend), pytest for tests.

## Running

All inputs are CLI arguments; there are no built-in data paths.

```
python study_parameters.py \
  --images   <setaria_image_scalars.csv> \
  --weights  <setaria_snapshots.csv> \
  --harvest  <setaria_harvest.csv> \
  --zoom     <zoom_calibration_data.txt> \
  --peterson <peterson_figure_readings.json> \
  --out      <new_output_dir>
```

Optional: `--seed` (20261101), `--n-boot` (200; contract range 100–200),
`--holdout-modulus 5 --holdout-remainder 0`, `--window-start 17 --window-end 33`,
`--max-image-gap 3`, `--large-daily-loss-g 600`, `--dispenser-tol-g 5`.

`--out` must not exist; an existing directory is refused (exit code 2) so earlier
results are never overwritten.

Tests: `python -m pytest test_study_parameters.py`

## Outputs (written to `--out`)

| file | content |
|---|---|
| `candidate_fragments.json` | machine-readable candidate parameters, assumptions, metrics, Peterson block |
| `diagnostics.json` | image-area audit, water-accounting audit (closure totals, flag counts), STEP profiles |
| `daily_sv_area.csv` | per barcode/day zoom-normalised SV area (all-image and in-bounds-only means, counts) |
| `harvest_join.csv` | independent harvest masses joined to final-day area, split labels, predictions |
| `weighing_intervals.csv` | every consecutive-weighing interval: loss, irrigation mass, dispenser amount, flags |
| `daily_balances.csv` | boundary water balances between consecutive last weighings with closure residuals |
| `plant_periods.csv` | normalised per-plant daily periods in the window (U, B at both ends, sources) |
| `plant_summary.csv` | per-plant q, C_balance, production ratio |
| `genotype_treatment_summary.csv` | genotype × treatment means |
| `heldout_step_bounded.csv`, `heldout_step_unbounded.csv` | heldout STEP endpoint/trajectory errors |
| `study_parameters.png` | heldout dry-mass prediction, r-profile identifiability, heldout growth-cap comparison |

## Method summary (what the code does)

1. **Calendar days / genotypes / holdout.** Naive timestamps are parsed with
   `strptime` and converted to calendar dates (no POSIX/timezone conversion);
   DAP = date − 2014‑01‑13. Barcodes must match `^(Dp[12]|Dr\d+)(AA|AB)\d+$`
   exactly (AA = wet, AB = dry). Holdout: `int(sha256(genotype),16) % 5 == 0`,
   applied to the genotype so all replicates, treatments and days travel together.
   Harvest IDs without a weights or images join are listed, not repaired.
2. **Zoom normalisation.** `camera_zoom = 1 + 5*(raw_zoom−1)/5999`;
   `a·exp(b·camera_zoom)` is fit by `curve_fit` to all 35 tabulated relative areas;
   SV `area` is divided by the predicted relative area. Valid (finite, > 0) SV areas
   are averaged across rotation frames per barcode/day. Out-of-bounds images are
   counted; an in-bounds-only mean is carried as a sensitivity. Zero/negative
   areas are counted and never used.
3. **Dry-mass coefficient.** Each harvested barcode's last imaging day is joined
   to its harvest `dry_wt`; `k` (g per normalised pixel) is a zero-intercept
   least-squares fit on TRAIN genotypes only; heldout predictions are scored
   (RMSE, MAE, R², median relative error, per genotype). All trajectory masses are
   `k × area` with this single `k` — never anchored to a plant's own harvest.
4. **Water accounting.** For consecutive weighings of a barcode:
   `loss = prev.weight_after − cur.weight_before`, `irrigation = cur.after − cur.before`,
   dispenser `water amount` kept separately, actual `dt`. Flags: incomplete
   records, non-positive duration, negative loss (kept unclipped), unusual large
   loss, negative irrigation, dispenser mismatch. Daily boundary balances between
   consecutive last successful weighings: `Σ irrigation − ΔW_after`, compared to
   the sum of interval losses (`closure_residual_g`, must be 0 for complete
   chains). A period is `complete_daily` only when the chain has no unsuccessful
   record, spans exactly one calendar day and `0 ≤ loss ≤ large_daily_loss_g`.
   Barcodes failing the genotype pattern are reported as possible empty-pot
   evaporation references and are never used as plants or subtracted.
5. **Window DAP 17–33.** Image-derived mass is interpolated only between
   neighbouring observations ≤ 3 days apart; no extrapolation. Periods require a
   complete daily balance and mass at both ends. Exclusion counts are reported.
6. **Rates.** `q = ΣU / ∫B dt` (m³/kg/day, trapezoidal B), the support prior
   `C_balance = ∫B dt / (365 ΣU)` (kg/m³ — a support threshold, not biomass per
   water), and the empirical `production_ratio = ΣΔB / ΣU` over the same periods.
   Pooled, by treatment, by split, per plant (quantiles), genotype-unit bootstrap
   (fixed seed, 200 resamples), T/ET fractions {1, .9, .8} as conditional
   sensitivities, and a sensitivity including measured ≤ 3-day complete chains.
7. **STEP-style diagnostic.** 7-day blocks (final partial) over the longest
   complete-water run per plant; `required = B·dt/365`, `available = C·U`,
   `wf = min(1, available/required)`, `gain = 0.95·r·(dt/365)·wf·B·(1−B/M)` capped
   at `M−B`, activity = temperature = live fraction = 1. M scenarios = 2×, 4×, 10×
   the maximum TRAIN endpoint mass (not a mature mass). Bounded (`r ≤ 1`, the
   runtime limit) and unbounded positive fits are both reported, scenario chosen
   on TRAIN SSE, heldout trajectory/endpoint error by genotype, plus profiles of
   SSE vs r (C optimised) and vs C (r optimised). Fitted C and water-balance C are
   reported separately. A rate-cap diagnostic reports the median observed growth
   factor, the r = 1 maximum and the minimum weekly-Euler rate needed.
8. **Peterson & Neofotis 2004.** Per-seedling two-day water × 2 = pot aggregate;
   ÷ 2 days for the daily rate. RGR and transpiration productivity are intensive.
   Total-mass support proxy `2·1000·exp(ln E)/(365·RGR)` kg/m³; conditional AGB
   proxy × `(1 − root_CRGR/RGR)` under an explicit stable-allocation assumption.
   38 Pa baseline and 76 Pa sensitivity kept separate. Growth factor `exp(RGR·39)`
   versus the r = 1 weekly-Euler maximum and the minimum rate needed.

## Caveats

See `report.md`. Nothing in this directory was executed by the implementer; the
orchestrator runs the CLI and tests independently.
