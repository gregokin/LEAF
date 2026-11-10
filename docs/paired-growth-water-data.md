# Individual growth paired with water use

Survey: 2026-11-01. The target is **growth of identified plants and water use over
the same interval**. The earlier [Jornada inventory](validation-data.md) supplies
important field checks, but did not establish that joint measurement.

Subsequent user-authorized work has produced
[initial STEP-style study candidates](study-stepwat-parameters.md). That analysis
uses each mesquite pot as one aggregate plant, joins Feldman images/weights to
withheld dry harvests, and records the current growth-rate compatibility issue.
The source-discovery descriptions below remain the background for that analysis.

## Best leads

| Source | Paired measurements | Access verified here | Fit to LEAF |
| --- | --- | --- | --- |
| **Feldman et al. (2018), Setaria** | Individual pot weighing, repeated plant images, endpoint shoot dry mass | Weight/harvest tables downloaded; **176 exact plant-ID joins** | Best immediately accessible grass test; repeated biomass is image-estimated |
| **Peterson & Neofotis (2004), Biosphere 2 mesquite** | Water consumption every two days, initial/final leaf, stem and root dry mass | Full paper inspected; raw observation archive not established | Close woody-plant experiment, but water is shared by two seedlings/pot |
| **Ravi et al. (2022), Biosphere 2 grasses** | Repeated leaf physiology and soil moisture, endpoint aboveground mass and survival | Paper inspected; linked Figshare listing could not be retrieved | Useful physiology/competition/mortality checks; paired whole-plant ET unverified |
| **Sunny et al. (2024), dry-forest saplings** | Repeated pot water use and interval growth, four species | Public file schemas verified; XLSX retrieval returned HTTP 403 | Good woody functional-type test if IDs and mass definitions qualify |
| **Matzner et al. (2024), milkweeds** | Endpoint biomass and cumulative transpiration in drought treatments | Public schema verified; XLSX retrieval returned HTTP 403 | Integrated mass–water check; perennial forbs rather than target grasses/shrubs |

### Public grass measurements: Setaria

[Feldman et al.](https://doi.org/10.1104/pp.18.00146) used individual C4 grasses
under contrasting water supplies. Pots were weighed two–three times/day and
images acquired every other day. Dry shoot mass was measured at final harvest;
earlier biomass is inferred from images, with daily values involving smoothing.
Pot water loss includes soil evaporation. The paper starts its main analysis
at 17 days after planting to reduce that problem, and notes that increasing
plant mass affected the imposed water setpoints.

The [v1.0 software/data snapshot](https://doi.org/10.5281/zenodo.1340752) resolves
to revision `8c88886b7f6ce3e0453eaba74ccadb0f319d3b1d`. Inspection found
87,368 weight records and 179 harvest rows: **176 barcodes match exactly**
(91 wet, 85 dry); three unmatched IDs need reconciliation. Harvest columns
include `fresh_wt`, `sat_wt`, `dry_wt`; weight records contain timestamp,
pre/post weights and irrigation. Raw weights span 20 January–17 February 2014.
These are raw counts, before QC. The additional
[image-feature/analysis archive](https://doi.org/10.5281/zenodo.1117062) is public
(216 MB ZIP; CC BY 4.0); metadata were retrieved, but its contents were not.

**Proposed use:** reconstruct daily water loss with irrigation and storage
changes, then compare final dry aboveground mass and weekly image-derived
increments. Audit evaporation, missing records, actual treatment weights,
initial biomass, image calibration and dates before fitting. A final harvest
alone cannot measure each week's dry-mass gain. Do not count later analyses of
the same experiment as independent validation.

### Biosphere 2 mesquite: the closest species lead

[Peterson & Neofotis](https://doi.org/10.1007/s00442-004-1688-y) grew
**Prosopis velutina** seedlings under two CO₂ and two water treatments. A
constant-head watering system supplied water; reservoir losses were recorded
every two days and corrected using plant-free evaporation controls. Initial
and final dry masses distinguish leaves, stems and roots over a 39-day growth
interval. Importantly, **two seedlings occupied each pot**; reported water and
mass were divided by two, giving a pot average rather than independent fluxes
for each plant. The initial harvest used removed seedlings, not repeated
destructive measurements of the surviving individuals.
[Public author paper at NASA](https://www.giss.nasa.gov/pubs/docs/2004/2004_Peterson_pe02200i.pdf)

**Proposed use:** evaluate two-plant pot totals, aggregate modeled transpiration
to two-day intervals, and compare leaf-plus-stem dry mass. This tests mesquite
seedling growth/water coupling; it does not directly validate adult honey
mesquite or weekly growth trajectories. No public raw-data repository was
verified in this search; the inspected paper is not a ready observation table.

### Ravi and Breshears: relevant, with a measurement gap

[Ravi et al.](https://doi.org/10.1111/gcb.15919), with Breshears and colleagues,
compared invasive buffelgrass and native tanglehead under drought, warming and
mixed/single-species communities at Biosphere 2. Measurements include roughly
weekly leaf photosynthesis, respiration and water potential, frequent shallow
soil moisture, endpoint aboveground dry mass and next-season survival. Reliable
biomass was unavailable for drought plants because material disintegrated and
litter was lost. Several plants shared each container. The published methods
do not describe integrated individual transpiration or weighing-based ET.

The paper explicitly links a
[Figshare archive](https://figshare.com/s/7cf2cd975decff746bb3). Here that link
returned an empty HTTP 202 response, so its files could not be inspected; a
public-title API search did not identify the matching record. This is an access
limitation, not evidence that the data do not exist. Raw instrument files might
contain additional leaf gas-exchange fields, but leaf transpiration cannot be
scaled to whole-plant daily use without leaf area and temporal integration.

**Proposed use:** temperature response, competition and drought survival checks.
Qualify the linked files before treating this as joint growth–ET validation.

### Other public experiments

[Sunny et al.'s archive](https://doi.org/10.5061/dryad.qbzkh18t0) describes 96
saplings of four seasonally dry tropical tree species, with four light/water
treatments over 50 days. Its `sunny2024_water_uptake.xlsx` reports whole-plant
water use on 14 measurement days; `sunny2024_traits.xlsx` includes growth and
allocation traits. Growth intervals differ from the full water-use interval.
The displayed schema does not establish absolute repeated aboveground mass.
Exact plant-ID joins, evaporation correction and usable biomass columns remain
to be checked in the files.

[Matzner et al.'s archive](https://doi.org/10.5061/dryad.0cfxpnwbx) lists
`2019_Milkweed_GH_Drought.xlsx` with biomass, root mass, average and total
transpiration, and drought treatments. This can support an integrated mass–water
comparison once biomass definitions and starting mass qualify. Separate
2020/2021 leaf gas-exchange observations are not simultaneous with the 2019
growth experiment. Neither archive was ingested because downloads returned 403.

## Recommended validation sequence

1. **Start with the verified Setaria joins.** Reserve whole genotypes or
   individuals before fitting, keeping all dates of an individual together.
   Treat image-derived mass and the harvest calibration as linked evidence.
2. **Add the mesquite pot experiment if raw records become available.** Use the
   correct two-seedling observation unit and preserve treatment CO₂ and climate.
3. **Test water and growth jointly.** Check daily/interval transpiration bias,
   cumulative water, final dry aboveground mass and weekly increments where
   observed. Never repair a biomass fit by allowing excessive water uptake.
4. **Then test field transfer at Jornada.** Pot studies restrict roots and lack
   field-scale root overlap, surface heterogeneity and event damage. Independent
   field biomass, storage and ET remain necessary.

For total pot ET, compare modeled **evaporation plus transpiration**; for
evaporation-corrected plant water use, compare transpiration. Endpoint standing
mass needs an initial mass or a suitable initial-sample estimate to become
growth. For LEAF's STEP-style option, empirical biomass/water ratios constrain
outcomes; they are not directly interchangeable with its biomass-support
coefficient. Estimate support, size limitation and losses jointly under an
explicit observation model. No calibration or model run was performed.

Public files, retrieval failures, ID checks and preservation checks are recorded
in the [research task](../agent_handoffs/tasks/paired_growth_et_survey_20261101/README.md).
