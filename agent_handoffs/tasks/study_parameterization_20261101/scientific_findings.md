# Scientific findings before candidate execution

Codex source inspection; these are constraints and conditional priors, not a
completed fit or field calibration.

## Observation units and missing information

- Feldman has 70,985 image rows (all retained in scalar projection), 87,368
  raw snapshot rows, and 179 harvests. Raw weights include 19,889 `-1` placeholder
  rows; positive-weight filtering leaves 67,479 records and 342 negative mass-loss
  intervals requiring QC. No negative loss may be silently clipped to zero.
- Side-view area can calibrate shoot dry mass. Relative lens calibration is
  not independently a circular-canopy footprint or field allometry. Biomass-
  normalized water rate identifies a demand/density ratio, not both traits.
- Peterson figure 5B is the LAST TWO DAYS, not cumulative 39-day water use.
  Double its extensive per-seedling quantities to represent one two-seedling pot.
  Relative rates and biomass/water coefficients stay unchanged.
- Mesquite absolute initial/final aboveground masses and the complete two-day
  water series are not tabulated in the inspected paper. Do not reconstruct
  fictitious observations from treatment means.

## Support, production and growth rate

Installed STEP-style convention: required = B*live*activity*dt/365;
available = C*actual_transpiration. A contemporaneous resource-balance prior is
`C_balance = integral(B dt)/(365*sum(T))` under unit live/activity factors.
This is a PR=1 threshold. It is different from delta_B/T and from endpoint
standing mass divided by water over an arbitrary experimental window.
It is also different from claiming that an experiment identifies an annual
STEPWAT2 resource coefficient independently of season length and live fraction.

Peterson's log-mean growth-analysis identity gives a total-mass proxy
`C = 2*1000*exp(ln_Eprod)/(365*RGR)` because water was measured over two days.
Multiplying by the aboveground INCREMENT share (1-root_CRGR/RGR) assumes this
also approximates standing aboveground allocation. That assumption is explicit;
the increment share is not itself an observed standing mass fraction.
Under it, ambient-CO2 AGB support is approximately 0.262–0.266 kg/m3 and
biomass-normalized daily water use approximately 0.0103–0.0105 m3/kg/day.
Elevated CO2 is a separate sensitivity, not pooled with baseline parameters.

The measured whole-plant RGR of about 0.097–0.105/day implies 44–60x mass in
39 days. With any initial aboveground fraction <=1, the reported aboveground
increment partition still implies at least approximately 30–43x AGB growth
(treatment-mean/figure-reading approximations). Stable allocation is not needed
to establish this large incompatibility with the current normal-growth cap.

Normal-growth weekly Euler with r_year<=1 gives only about 1.106x in 39 days,
even with full water, temperature, activity and negligible mature-size limitation.
Matching whole-plant interval factors in that update would require at least
approximately 52–58/year, or continuous-time equivalents 37–40/year. These are
diagnostic seedling rates, not annual adult shrub traits. Seasonal extra could
inflate total biomass, but forcing almost all shrub gain into a component lost
at year-end would be an unsupported interpretation of these observations.

The public pinned `ST_params.c` assigns its input `irate` directly in
`_species_init`; no [0,1] upper-bound check occurs there. This does not establish
that arbitrary large annual STEP coefficients are scientifically valid. The
current adapter's [0,1] cap and annual-to-weekly interpretation need explicit
evaluation, not attribution to a demonstrated bound in that source parser.

## Scope of resulting candidates

Normal-only fits are diagnostics. Short experiments do not identify the split
between persistent and seasonal mass, adult mature size, annual phenology,
turnover, roots or field canopy density. Holdout harvest evaluation is more
independent than comparison against image-derived biomass using the same
allometry. Measured water used as replay forcing does not independently validate
a soil-water solver or the demand model. Pot ET needs explicit E/T assumptions.
Candidates remain inactive; no runtime equation/bound/default is changed here.
