# study_parameterization_20261101 — candidate analysis (revision 2)

Bounded, reproducible public-study analysis producing *candidate* parameter
fragments from Feldman et al. 2018 (Setaria RIL phenotyping; DOI
10.1104/pp.18.00146, data 10.5281/zenodo.1117062) and approximate figure
readings of Peterson & Neofotis 2004 (DOI 10.1007/s00442-004-1688-y). No
simulator, source lock, default or runtime rate bound is changed by this code.

Files (the only ones owned by the implementer):

| file | purpose |
|---|---|
| `study_parameters.py` | library of small testable functions + CLI |
| `test_study_parameters.py` | synthetic-input unit tests (pytest), no public data needed |
| `README.md` | this file |
| `report.md` | honest implementation report (what was and was not executed) |

## Running

```
python study_parameters.py \
  --images  <setaria_image_scalars.csv> \
  --weights <setaria_snapshots.csv> \
  --harvest <setaria_harvest.csv> \
  --zoom    <zoom_calibration_data.txt> \
  --peterson <peterson_figure_readings.json> \
  --out     <NEW directory>          # refused if it already exists
  [--seed 20261101] [--n-boot 200] [--holdout-modulus 5] [--holdout-remainder 0]
  [--window-start 17] [--window-end 33] [--max-image-gap 3]
  [--qc-day-label end|start] [--large-daily-loss-g 600] [--dispenser-tol-g 5]

python -m pytest test_study_parameters.py
```

All input paths are mandatory; there are no public-data defaults or private imports.

## Outputs written to `--out`

* `candidate_fragments.json` — machine-readable candidate values, assumptions, exclusions.
* `diagnostics.json` — image/water audit totals, STEP profiles, exclusion counts.
* `daily_sv_area.csv` — per barcode/DAP zoom-normalised side-view area (mean over rotations), out-of-bounds counts, in-bounds-only mean.
* `harvest_join.csv` — harvest rows joined to last-image area, split flags, predicted dry mass.
* `weighing_intervals.csv` — every consecutive physical weighing interval with `loss_g`, `irrigation_g`, `dispenser_g`, `usable`, `reasons`.
* `daily_balances.csv` — boundary balances between consecutive last successful weighings of a day, closure residuals, study-QC reasons, primary/sensitivity selections.
* `plant_periods.csv`, `plant_periods_sensitivity_retained_flagged.csv` — normalised per-plant daily periods (U in m3, image-derived B in kg at both ends).
* `plant_summary.csv`, `genotype_treatment_summary.csv` — per-plant and per-genotype rates.
* `heldout_step_{bounded,unbounded}.csv` — image-derived shape endpoints for withheld plants.
* `heldout_harvest_replay_{bounded,unbounded}.csv` — growth replay endpoint vs withheld harvest dry mass (date-matched only).
* `study_parameters.png` — heldout dry-mass prediction, rate profile, heldout endpoint comparison (image-derived filled, harvest hollow).

## Method summary (see the contract for the governing text)

1. **Calendar days**: naive timestamps → calendar date → DAP from 2014-01-13 (no POSIX/local-zone conversion). Genotype from exact barcode `Dr<n>A[AB]…` / `Dp1|Dp2`; AA = wet, AB = dry. Holdout: `sha256(genotype) % 5 == 0`, all replicates/treatments/days of a genotype together.
2. **Zoom**: `camera_zoom = 1 + 5*(raw_zoom-1)/5999`; `a*exp(b*camera_zoom)` fitted by nonlinear least squares to all tabulated relative areas; image area divided by predicted relative area; valid positive SV areas averaged across rotations per barcode/day. Out-of-bounds images are counted and an in-bounds-only sensitivity carried.
3. **Dry mass**: last-image-day SV area joined to harvest `dry_wt`; one positive zero-intercept coefficient fitted on TRAIN genotypes; heldout metrics reported; the same coefficient converts all earlier areas. Trajectories are never anchored to their own harvest.
4. **Water**: `loss_g = previous weight_after − current weight_before` on consecutive *physical* weighings; irrigation mass `after − before` kept separate from the dispenser amount. Image-snapshot placeholder rows (before = after = water amount = −1) are non-measurements: counted, excluded from chains, surrounding weighings connected. Other incomplete records stay explicit and block their chain (no extrapolation). Daily boundary balances between consecutive last weighings must close on interval sums (`closure_residual_g`). Negative losses are flagged, never clipped.
   * **Primary selection** = complete 1-day chain passing the published study QC policy (nonpositive daily losses; > 200 g on DAP 17/32/33; dry-treatment DAP 32 > 145 g), each counted by reason.
   * **Sensitivity** = complete 1-day chain with 0 ≤ loss ≤ 600 g (study-QC-flagged days retained, unclipped).
   * `--qc-day-label` sets whether a daily loss is labelled by the weighing day (`end`, default) or the previous day; verify against the source convention.
5. **Window DAP 17–33**: mass interpolated only between neighbouring images ≤ 3 days apart; no extrapolation. T = f·ET for f ∈ {1, 0.9, 0.8} is a conditional sensitivity, not a calibration.
6. **Rates**: `q = ΣU/∫B dt` (m3/kg/day); `C_balance = ∫B dt/(365 ΣU)` (kg/m3, a support threshold, not biomass per water); production ratio `ΔB/ΣU` over the same periods. **Candidate values come from TRAIN genotypes only** (pooled, by treatment, T fractions, genotype bootstrap with fixed seed, 100–200 resamples). All-plant and heldout summaries are descriptive. Bootstrap conditions on the single fitted image calibration.
7. **STEP-style diagnostic**: 7-day blocks with actual block water, final partial block; `wf = min(1, C·U/(B·dt/365))`, `gain = 0.95 r (dt/365) wf B (1−B/M)` capped at `M−B`; M scenarios 2×/4×/10× the maximum TRAIN endpoint mass, chosen on TRAIN only. Bounded (`r_year ≤ 1`, the runtime limit) and positive unbounded fits are compared; profiles of SSE over fixed r (C optimised) and fixed C (r optimised) expose the trade-off. Heldout evaluation reports (a) image-derived shape metrics and (b) replay endpoint vs withheld harvest dry mass only when the endpoint DAP equals the plant's last-image DAP (the source's sampling proxy for harvest date).
8. **Peterson & Neofotis 2004**: one modeled plant per two-seedling pot; two-day water ×2 (÷2 for a daily rate); RGR, transpiration productivity and support proxies are intensive and unchanged; `2·1000·exp(lnE)/(365·RGR)` kg/m3 total-mass proxy and a conditional AGB proxy (`×(1 − root_CRGR/RGR)`, increment allocation, not standing fraction); 38 Pa and 76 Pa kept separate; growth factor over 39 days compared with the r_year = 1 weekly-Euler cap and the minimum rate needed.
