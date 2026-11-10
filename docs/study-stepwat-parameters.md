# Initial study-based STEP-style parameters

Derived 2026-11-01. [Machine-readable candidates](../parameters/candidates/study-stepwat.json)
are **inactive** and contain study estimates, conditional fits and missing traits.
The mesquite experimental pot represents **one modeled plant containing two
biological seedlings**, as requested. No runtime defaults or equations changed.

## Grass: a joint fit to the weekly growth rule

The [Feldman et al. Setaria experiment](https://doi.org/10.1104/pp.18.00146)
provides repeated images, pot weighing and endpoint shoot dry mass. Analysis
uses DAP 17–33. A training-only image calibration supplies earlier mass; whole
genotypes are withheld from both calibration and growth fitting. Plant identities
and physical weighing boundaries are preserved. The
[public archive](https://doi.org/10.5281/zenodo.1117062) is pinned by checksum.

These parameters were **jointly fitted** with live/activity/temperature factors
fixed at one and seasonal-extra growth fixed at zero:

| Quantity | Initial value | Interpretation |
| --- | ---: | --- |
| `biomass_support_kg_per_m3` | **0.01488** | Joint-fit resource coefficient, assuming pot ET approximates T |
| `intrinsic_rate_per_year` | **295.1** | Diagnostic seven-day Euler rate; exceeds the current validator's limit |
| Capacity `M` | **0.01759 kg** | Assumed scenario: twice the largest training update-boundary mass; adult mature mass is unmeasured |
| `0.95 × rate × support` | **4.172** | Gain/water coefficient in the resource-limited branch, before size limitation |

Use the rate and support coefficient together. A separate empirical
**balance prior** is `C_balance = integral(B dt)/(365 sum(U)) = 0.08335 kg/m³`.
It sets the PR=1 burden/water threshold under the adopted unit live/activity
factors. It is not the fitted coefficient above. The observed biomass-normalized
pot loss is **0.03287 m³/kg/day**, or **32.9 L/kg/day**. Using actual weighing
durations instead of nominal daily boundaries gives 0.08499 kg/m³ and
0.03224 m³/kg/day. The observed gain/water ratio is **3.666 kg/m³**; it is a
production diagnostic, rather than an interchangeable resource coefficient.

For illustrative constant `T/ET` fractions of 1.0, 0.9 and 0.8, joint-fit support
is respectively 0.01488, 0.01654 and 0.01860 kg/m³. Scaling support inversely
preserves `C × U`; these are partition assumptions, not measured evaporation
fractions. No empty-pot subtraction was imposed.

### Checks and uncertainty

The fit uses **958 training trajectories** and **177 withheld trajectories**.
Of the withheld plants, **34** have a dry harvest matching the endpoint's
last-image-day sampling proxy. Their biomass predictions have **RMSE 1.52 g,
MAE 0.93 g, R² 0.70**, and median absolute relative error **25.7%**. With the
current `rate <= 1` bound, median harvest underprediction is approximately 90%.
The separate image-calibration check uses 35 withheld harvests, with RMSE 1.61 g.
These tests use measured pot loss as forcing; they do not independently validate
transpiration demand or native soil hydrology.

Adult capacity remains weakly constrained: 2×, 4× and 10× capacity scenarios
change training SSE by less than 2%. Similar fits include `(rate, support)`
approximately `(251, 0.0181)` and `(398, 0.0106)`; these are profile alternatives,
not confidence limits. A 100-resample training-genotype bootstrap places the
balance prior at approximately 0.0818–0.0849 kg/m³ **conditional on the fixed
image calibration**. It excludes allometry, ET partition and model-form uncertainty.

## Mesquite: conditional priors from the published figures

[Peterson & Neofotis (2004)](https://doi.org/10.1007/s00442-004-1688-y) reports
whole-plant and component growth rates, transpiration productivity and final
two-day water use. Original figure pixels were inspected; values below are
approximate treatment-level readings. Absolute initial/final aboveground masses
and a complete water series were not recovered from the paper.

| CO₂ partial pressure | Water treatment | Conditional AGB support proxy, kg/m³ | Final daily T per aggregate pot, mL |
| --- | --- | ---: | ---: |
| 38 Pa | Low | **0.262** | **15.9** |
| 38 Pa | High | **0.266** | **20.6** |
| 76 Pa | Low | 0.411 | 11.0 |
| 76 Pa | High | 0.548 | 10.5 |

The ambient starting prior is therefore approximately **0.264 kg/m³**, with
conditional biomass-normalized T of **10.3–10.5 L/kg/day**. Elevated CO₂ remains
a separate sensitivity. These are balance proxies, not a joint shrub growth fit.

Derivation: the paper's log-mean identity gives total-mass support
`2 × 1000 × exp(ln Eprod)/(365 × RGR)`. The two-day observation window accounts
for the factor two. The AGB proxy multiplies by `1 − root_CRGR/RGR`, **assuming
increment allocation approximates standing aboveground allocation**. That
fraction was not independently observed as standing AGB allocation.

Doubling reported per-seedling biomass and water recovers one pot total.
Relative growth and mass/water coefficients remain unchanged. Figure 5 water
describes the **last two days**, not cumulative 39-day consumption. Figure-reading
resolution is recorded separately from statistical uncertainty.

## Compatibility and use

The installed adapter accepts `intrinsic_rate_per_year <= 1`. Mesquite's reported
whole-plant growth corresponds to approximately **52–58/year** in the diagnostic
weekly Euler clock even before mature-size or water limitation. Grass fitting
also requires a substantially larger rate. An actual validator probe rejects
these values. The normal-only experiment uses seven-day blocks and a final
partial observation update; it does not reproduce the native calendar's commit
schedule. Review the growth time scale and observation timing before activation.
Treating shrub gain as seasonal mass removed at year-end would require evidence.
The rate cap is a local adapter choice: the inspected pinned
[STEPWAT2 input parser](https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/ST_params.c#L1265)
assigns `irate` directly, without a `[0,1]` check in that routine. This observation
does not establish scientific validity for arbitrary large annual coefficients.

These are seedling parameters: Setaria is a grass proxy, and velvet mesquite
differs from mature Jornada honey mesquite. Adult mass, field canopy allometry,
live/leaf fraction, annual phenology, turnover and root distributions remain
unidentified. For known canopy density `rho` and adequate water,
`q_T = potential_transpiration_depth / rho × live_fraction × activity`; observed
`q_T` alone does not identify both canopy density and potential depth.

Reproduce the analysis using the existing environment and
[analysis commands](../analysis/study_parameterization/README.md).
[Review, source identities and independent checks](../agent_handoffs/tasks/study_parameterization_20261101/README.md)
record 29 passing tests, exact water-balance agreement and independent scalar
replays. No field calibration, native hydrology/transport run, upstream edit,
installation or new environment was performed.
