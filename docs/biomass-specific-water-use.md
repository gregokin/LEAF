# Daily water use per aboveground **dry** biomass

Literature comparison, 2026-11-01. All numerical rates below use **kilograms of
aboveground dry biomass**, excluding tissue water and roots. They describe
water loss per standing mass, rather than water needed to produce a kilogram of
new biomass. These are observations and descriptive calculations; no LEAF
parameter candidate or runtime default was changed.

## Quantity and conversion

For a daily transpired volume `T_day`, use
`q_T = T_day / B_AG_dry`, in **L kg⁻¹ day⁻¹**. A reported mL/g/day has the same
numerical value as L/kg/day. Converting grams of water to litres assumes water
density approximately 1 kg/L.

The inverse has units kg·day/m³ when `q_T` is first expressed in m³/kg/day.
An annual reference period gives the equivalent kg·year/m³ support quantity.
The fitted growth-support coefficient is a different statistical estimate from
this direct inverse; do not mix coefficients from the two calculations.

## Additional numerical evidence

| Plants and conditions | L/kg aboveground **dry** biomass/day | Basis |
| --- | ---: | --- |
| Savanna grasses, relatively moist soil | **23.4**, IQR **14.9–26.7** | Complete-day transpiration / harvested shoot dry mass; median across 30 individual means |
| Savanna woody seedlings, relatively moist soil | **13.8**, IQR **9.6–19.9** | Same calculation; 35 individual means |
| Same grass dataset, very dry soil | **2.65**, IQR **1.82–3.70** | 21 individual means |
| Same woody-seedling dataset, very dry soil | **2.22**, IQR **1.74–2.81** | 12 individual means |
| Maize, covered soil pots | **18.8** | Published final-day T / shoot dry mass |
| Soybean, covered soil pots | **25.5** | Same calculation |
| Sunflower, covered soil pots | **36.3** | Same calculation |
| Zucchini, covered soil pots | **30.8** | Same calculation |

The first four rows are **new calculations from the released data**, rather than
numbers quoted from the paper's plots. Source:
[Belovitch et al. (2023)](https://doi.org/10.1111/nph.18909), with
[data and analysis](https://github.com/mbelovitch/NPH-MS-2023-42507/tree/862abe68c7a9f1eb6397c36516bca35c9d445298).
Pots were sealed and followed through a growth-chamber dry-down. The relatively
moist category has daily mean soil water potential above −0.5 MPa; the very dry
category is at or below −1.5 MPa. All 24 hours, including night, are summed.
One end-of-dry-down shoot dry mass is used per plant: daily mass was not measured.
IQR describes variation among individuals, not a confidence interval or a
universal species range. This is a descriptive reanalysis, not the paper's
mixed-model comparison on a leaf-mass basis.

The covered-soil rows come from **Table 2** of
[Murano, Otani & Seike (2010)](https://doi.org/10.1002/etc.288).
The greenhouse was approximately 25°C with roughly 14-hour natural days;
soil was covered to suppress evaporation. The soil contained residual dieldrin,
so these are experimental comparisons, not clean field calibrations. For maize,
`192 mL/day ÷ 10.2 g dry shoots = 18.8 L/kg/day`. Ratios use treatment means;
individual ratio uncertainty cannot be inferred from the marginal standard
errors alone. This experiment used several plants per pot for some species;
pot totals in both numerator and denominator preserve the ratio.

Their separate **hydroponic Table 4** gives maize **26.8**, soybean **26.7**,
and sunflower **75.4 L/kg dry shoots/day**. Different growth durations and
conditions prevent interpreting this as an isolated soil-versus-hydroponics
treatment effect. All fourteen table-derived ratios and source means are
archived in the [research record](../agent_handoffs/tasks/biomass_specific_water_use_20261101/README.md).

## Interpretation for LEAF

These observations place our earlier Setaria **32.9 L/kg dry AGB/day** and
conditional mesquite **10.3–10.5** in a plausible order of magnitude for young,
actively transpiring plants. Exact comparisons remain limited: Setaria uses pot
ET and a time-varying inferred mass, mesquite uses a conditional AGB allocation,
and the new savanna calculation uses a fixed harvest mass. See the
[initial study assessment](study-stepwat-parameters.md).

The grass-to-woody ratio is about **1.7** for the moist-bin medians and **1.2**
for the very-dry-bin medians. These are descriptive ratios from different
subsets, not paired treatment effects. The earlier approximately threefold
comparison is experiment-specific. These sources do not establish a fixed
grass/shrub multiplier or an adult Jornada coefficient. Total dry biomass can
include substantial structural wood; leaf fraction, age, atmospheric demand
and soil-water availability need explicit treatment when transferring values.

## Useful studies requiring a different normalization

- [He et al. (2018), perennial bunchgrasses](https://doi.org/10.15302/J-FASE-2017175):
  verified shoot dry masses and mean daily water use over 30/60 days. Shoots
  include the subterranean crown. Dividing experiment-average water use by
  final shoot mass would produce an endpoint proxy, with a different mass basis.
- [El Habti et al. (2020), wheat](https://doi.org/10.3389/fpls.2020.568693):
  daily water use normalized to final vegetative biomass, excluding grains.
  That denominator and its timing differ from current total AGB; no rate from
  its figures is entered as an independently verified whole-shoot dry-mass value.
- [Ahmad et al. (2024), perennial ryegrass](https://doi.org/10.1002/pld3.70013):
  explicitly measures AGB dried at 70°C for three days and leaf mass fraction.
  Its whole-plant transpiration estimate scales morning leaf-gas-exchange
  measurements, while daily water balances quantify ET. A morning snapshot
  cannot be multiplied by 24 hours to obtain measured daily transpiration.
- [Renninger et al. (2021), young poplars](https://doi.org/10.3389/ffgc.2021.704799):
  seasonal sap flow and harvested dry woody biomass are available; woody mass,
  seasonal growth and complete shoot mass must be distinguished before daily
  normalization. No daily coefficient is invented from seasonal WUE.
- [O'Keefe et al. (2020), prairie grasses and shrubs](https://doi.org/10.1029/2019JG005446):
  valuable continuous field sap flow, reported with area-based scaling; a
  matched aboveground dry-mass denominator is needed for this comparison.

## Data checks

The archived savanna table has 21,311 rows. Two plant IDs have ambiguous duplicate
hour records; one also has an inconsistent supplied shoot normalization. Their
raw rows are preserved and both IDs excluded from this descriptive summary.
Recomputing from raw water and shoot mass leaves **759 complete plant-days**.
Negative hourly increments are retained without clipping. Species-code counts
in the released table differ from the paper's stated species counts, so they
are not presented as independently confirmed biological species counts.

Independent raw-data aggregation reproduces all eligible days and eight
group/moisture summaries; fourteen table-ratio unit conversions also agree.
Sources, exact revision, calculations and QC tables are in the
[durable evidence](../agent_handoffs/tasks/biomass_specific_water_use_20261101/README.md).
