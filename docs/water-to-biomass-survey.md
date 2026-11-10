# Water-to-biomass approaches for LEAF grasses and shrubs

Survey: 2026-10-31. Scientific assessment only; no growth law or parameter set is selected
or implemented here. The existing qualified native water coupling remains unchanged.
Detailed surface/soil/root remapping stays deferred by user direction.

Subsequent implementation: the user selected a STEP-style version, now described
in the [growth guide](stepwat-growth.md). It uses biomass-support pressure and
logistic size growth with actual native water uptake. The direct WUE-production
adapter below remains a surveyed alternative; its equations are not the selected
runtime growth rule. Accepted original survey evidence remains unchanged.

## Current scope — total aboveground biomass

Following the survey, the user selected **total aboveground biomass per individual**
as the growth state and observational target because more data are available for it.
This supersedes the initial recommendation to require separate leaf, wood, root
and reserve pools. The surveyed models' descriptions below remain source findings;
their full tissue accounting is not the current LEAF design.

Use an aboveground dry-matter convention matched to the observations, with the
treatment of live versus standing-dead material specified during calibration.
Species/PFT allometry can relate that biomass to canopy dimensions, root extent
and uptake capacity without an explicit belowground biomass state. Keep the
existing species-specific vertical root fractions. Seasonal leaf activity can
modify demand and growth without requiring separately tracked tissue masses.

## Assessment

Retain actual transpiration as the water constraint, and develop the one-pool
benchmark into an aboveground biomass model only through a separately agreed
ecological implementation. ECOTONE provides individual water-driven growth ideas;
AquaCrop provides an aboveground-production precedent. Williams–Albertson and
GRASSMIND inform climate, phenology and loss rules without requiring adoption of
their tissue pools. This is a proposed small LEAF module, not exact reproduction
of any model.

The important distinction is between water supporting production and water being the
material source of dry biomass. Photosynthetic carbon enters from the atmosphere.
Account for water withdrawal in SYRUP and aboveground production and losses in LEAF
with consistent units. A complete whole-plant carbon budget is outside this scope;
belowground allocation and respiratory costs must be represented consistently in
the chosen production coefficient rather than ignored or deducted twice.

## What the models actually do

| Model | Water-to-growth connection | Timing and state | Fit for this project |
|---|---|---|---|
| Tietjen/EcoHyD | Soil moisture and root-weighted competition limit logistic expansion. | Biweekly grass/shrub **cover**; seasonal mortality. | Useful seasonality/competition reference; a biomass and individual model would be new. |
| Supplied ECOTONE ecology | Allocated transpired water, inverse WUE and maintenance determine biomass increments. | Annual individuals; above/belowground and perennial/annual tissues. | Closest local individual precedent, but annual resource redistribution needs adaptation. |
| STEPWAT2 | Transpiration becomes supported biomass; resource sufficiency modifies size growth. | Annual relative size, persistent and seasonal production. | Good demography/size-limitation reference, less direct for a daily uptake-to-carbon ledger. |
| GRASSMIND 2.0 | Potential photosynthesis sets water demand; water limits actual production before respiration/allocation. | Daily individuals and shoot/root/reproduction pools. | Strong grass physiology reference; explicit LEAF positions/root footprints remain our architecture. |
| Williams–Albertson 2005 | Transpiration and climate-sensitive WUE support carbon gain; respiration and decay are separate. | Daily grass/wood leaf and structural pools. | Particularly close to the proposed water-driven, climate-aware production model. |
| Caylor 2004–2006 | Canopy structure, water stress and spatial overlap affect productivity/resources. | Daily patch production or water/stress analyses, depending on paper. | Guides light/canopy coupling; does not establish a ready individual biomass/demography module. |
| AquaCrop | Daily transpiration drives climate-normalized aboveground biomass productivity. | Daily homogeneous crop field. | Useful efficiency normalization; perennial shrubs/individual grass demography require another model. |

### Tietjen: cover growth, not a biomass yield

EcoHyD uses preceding 14-day moisture availability above a vegetation-type wilting
point and competitive uptake weights in a logistic cover-growth term. Wet-season
growth and end-season mortality are distinct. Its “transpiration efficiency” climate
experiments do not supply a direct kg-biomass-per-kg-transpiration equation.
[Tietjen et al. 2010, model description and plant growth](https://www.biodiversity-plants.de/downloads/Tietjen_etal_2009_Ecohydro.pdf).

The local [MAHLERAN growth adaptation](../../MAHLERAN/src/Subroutines_Vegetation/growth.f90)
implements these cover calculations and scales rates by `dt_vegetation/365`.
Reuse the scheduling idea and stress-response questions. Converting cover to biomass
and allocating it to individual shrubs/grasses would require new allometry and
accounting; it would not recover the proposed mass ledger automatically.

### ECOTONE: direct individual water-to-biomass conversion

[Peters 2002](https://www.ars.usda.gov/research/publications/publication/?seqNo115=125438)
describes annual plant dynamics coupled to daily soil water. The supplied snapshot
has later adaptations, so the following findings refer specifically to its active
[gro_new.f](</home/okin/Ecotone Wind 9/Ecotone Wind 9/Src/gro_new.f>), called from
[grow.f](</home/okin/Ecotone Wind 9/Ecotone Wind 9/Src/grow.f>).

For an individual of mass `B`, the code constructs an optimal increment from
`GRORAT*(B_full-B)`. `WUECNST` means **g water/g biomass**, the inverse of a
biomass-per-water efficiency. Maintenance is represented as a water-equivalent
allowance `B*WUECNST*MTNMODF`; water remaining after maintenance and resource
redistribution is divided by `WUECNST` to obtain new biomass.
That maintenance allowance belongs inside the growth calculation; it must not
be withdrawn from SYRUP again as a second physical water loss.

Species traits split production above/belowground (`PRATIO`) and among annual/
perennial aboveground tissues (`APRATIO1–3`, depending on growth conditions).
Root additions follow species layer distributions and coarse/fine fractions.
Under insufficient maintenance resources, the active code holds shrub increments
at zero but allows grass shrinkage. These are substantive ecological choices,
not generic numerical requirements.

**Reuse:** the trait organization, inverse-efficiency conversion and distinction
between persistent structure and replaceable tissues. **Adapt:** annual coefficients,
maintenance, loss rules and growth timing. The routine redistributes already
allocated water resources between plants. LEAF must not reassign physical uptake
after SYRUP has withdrawn it and attributed it to specific plants. Its nitrogen/light
comments are not evidence that those factors constrain this active growth formula.
Neither this snapshot nor the paper is a qualified weekly drop-in.

### STEPWAT2: supported biomass and resource-modified growth

The [2018 paper](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.2394)
separates perennial size growth from seasonal biomass. Monthly layer/functional-group
transpiration supports annual resources; growth responds to resource sufficiency,
species growth potential, size and temperature class. Resource allocation prioritizes
larger individuals.

Its reported biomass is **total aboveground biomass**, consistent with the user's
chosen observational target. In the pinned code, individual relative size times
species mature biomass gives individual mass; species totals also include seasonal
extra growth. This is simpler than a detailed tissue carbon budget, while still
distinguishing perennial size from biomass removed at season's end.

At pinned revision `003cb9b4adbf13d64a39d6d62fd95280adac27fc`,
[`sxw_resource.c`](https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/sxw_resource.c#L106)
multiplies transpiration resources by a biomass-support coefficient `_bvt`.
[`rgroup_Grow`](https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/ST_resgroups.c#L519)
uses, for normal growth, `rate=gmod*intrinsic_rate*(1-relative_size)` and
`size_increment=rate*relative_size`; `gmod` decreases with required/available
resource ratio and is modified by temperature. Extra resources can produce
additional seasonal growth. These are not a universal net carbon yield per litre.

**Reuse:** finite-size growth, seasonal/persistent distinction and clonal recovery
ideas. **Adapt:** annual timing and resource allocation; preserve the actual LEAF
uptake distribution if retaining its current competition rule. “Added transpiration”
is ecological resource accounting, not additional physical water available to LEAF.
The inspected resource allocator also substitutes a positive resource value in
one zero-resource case; that is not a physical uptake rule to copy into LEAF.
Calling the annual growth routine 52 times would change the model even if a rate
were divided by 52. New pinned growth sources are archived in this survey task.

### GRASSMIND: a practical physiology reference for grasses

The [author manual](https://formind.org/wp-content/uploads/2020/05/Grassmind_ModelDescription.pdf),
sections 3.7–3.14, links light-dependent potential production to demand through
`water_demand=temperature_modifier*potential_GPP/WUE`. Water competition limits
production; maintenance and growth respiration precede allocation. Green shoot,
senescent shoot, roots and reproduction have distinct accounting, with separate
leaf/root turnover and species shoot–root allocation.

This gives a useful demand-first template: estimate what active foliage could
produce, request the associated water, and use the native actual uptake to limit
production. Importing all original water-stress factors on top of native uptake
could count the same water limitation twice.

The [2020 grass study](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.3205)
uses daily individuals without explicit positions within a patch and a European
grass parameterization. Use its concepts, not its species coefficients or patch
hydrology as Jornada defaults. LEAF retains its existing spatial individuals.

### Williams–Albertson: especially relevant to shrubs plus grasses

[Williams and Albertson 2005](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2004WR003750)
connect daily transpiration to CO₂ uptake through climate-sensitive WUE. Modeled
nonleaf respiration depends on biomass, temperature and moisture; production is
allocated to leaf and structural pools with distinct decay/leaf-flush behavior.
Their C₄ grass and C₃ wood differ in efficiency and other traits. This is a strong
published basis for separating fast foliage dynamics from persistent shrub structure.

It is a functional-cover model, so transferring equations to spatial individuals is
an adaptation. Its deeper reservoir gives wood exclusive access; retain LEAF's
specified grass/shrub roots in **all** layers instead. Its WUE is expressed using
CO₂ mass, which requires conversion before using kgC or dry biomass. This source
is not justification for giving every shrub and grass the same turnover coefficient.

### Caylor: preserve the canopy–productivity connection

[Caylor et al. 2004](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2486.2003.00704.x)
and [Caylor–Shugart 2004](https://doi.org/10.1023/B:LAND.0000030450.11302.c2)
connect heterogeneous canopy/leaf structure with patch productivity; the latter
uses a daily productivity model. [The 2005 study](https://link.springer.com/article/10.1007/s10021-004-0027-9)
separates under-canopy/intercanopy water and surface components.
[The 2006 study](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2005WR004683)
uses canopy/root-overlap statistics to describe water/stress heterogeneity.

For LEAF, these support keeping active leaf area distinct from physical canopy
size and accounting for shrub effects on grass light/water conditions. Wet soil
alone should not generate unlimited shaded-grass production. Detailed 2004/2005
physiology equations were not verified from accessible full methods; the four
papers must not be treated as one inspected demographic codebase.

### AquaCrop: normalize efficiency for evaporative demand

[FAO's daily scheme](https://www.fao.org/aquacrop/overview/calculation-scheme/en)
separates transpiration from evaporation and uses climate-normalized productivity
for aboveground biomass. A primary
[AquaCrop evaluation, equation 1](https://gmd.copernicus.org/articles/14/7309/2021/),
expresses the relationship as `B=WP* sum(Tr/ETo)` with consistent area/mass units.

This offers a simple climate correction, but `WP*` is not a raw kg/kg efficiency,
and aboveground yield does not include new roots. Its uniform-field crop framework
is not a perennial shrub model. Native potential **soil evaporation** is not the
reference evapotranspiration `ETo`; do not substitute one for the other. Adopt the
normalization only with compatible forcing and explicit individual unit conversion.

## Proposed aboveground growth adapter

The current [growth kernel](../benchmarks/daily_water_growth.py) already consumes
actual daily plant uptake once per week. Its deliberately synthetic rule is

`B_new = B_old*exp(-0.003*7) + 1000*efficiency_s*sum(actual_uptake_m3)`.

Grass/shrub efficiencies are 0.002/0.0015 kg biomass per kg water; the mass pool and
turnover are shared in form, with constant area-based potential demand. This is a
qualified accounting/performance fixture, not a calibrated aboveground growth model.
Its generic `B` has not yet been qualified as observed aboveground dry mass.
The current species root profiles determine uptake; they do not supply production,
respiration, phenology, tissue allocation or species-specific hydraulic stress traits.

Recommended next design, **not implemented by this survey**:

1. **Declare aboveground production units.** Use kg aboveground dry matter per
   plant and an efficiency with numerator kg new aboveground dry matter. A
   whole-plant or gross-carbon efficiency needs an explicit conversion before
   use here; ECOTONE's aboveground allocation fraction cannot simply be omitted.
   Specify which allocation and respiratory costs the fitted coefficient includes.
   Standing biomass and new production are different observations; calibration
   against stock changes must also account for biomass losses.
2. **Resolve daily climate and activity.** Species/PFT temperature and phenology
   regulate active leaf area, production capacity and demand. Compare a constant
   transpiration efficiency control with a humidity/evaporative-demand correction;
   radiation or a defensible potential-growth ceiling constrains favorable wet
   periods. C₃/C₄ class alone is not a complete species parameterization.
3. **Keep the native water authority.** Demand reaches SYRUP through existing
   roots; actual per-plant transpiration constrains production. Exclude soil
   evaporation. A production limitation does not create water or authorize
   reassigning already accepted uptake to another plant.
4. **Update total aboveground biomass weekly.** Add accumulated aboveground
   production and subtract explicitly modeled losses using species/PFT rules.
   Relate biomass to canopy and root geometry through species allometry; keep
   current vertical root fractions initially. Root biomass and separate tissue
   pools are not required. Total mass alone does not specify green leaf area:
   a phenology/activity fraction can provide that distinction for demand.
5. **Accumulate daily production before the weekly update.** If efficiency changes
   with weather, accumulate `sum(efficiency_day*uptake_day)`, not mean efficiency
   times total weekly uptake. Keep respiration/turnover time units explicit.
   Record aboveground losses consistently with the live/standing-dead convention,
   using an aggregate loss/export ledger if needed rather than mandatory tissue
   pools. Mortality and event damage remain separately scoped processes.

One possible **aboveground dry-matter** accounting form for plant `i` is

`production_AG_i = sum_day(efficiency_AG_species(day)*1000*actual_uptake_i_m3(day))`

`B_AG_i_new = B_AG_i_old + production_AG_i - losses_AG_i`.

Here `efficiency_AG` has units kg aboveground dry matter/kg transpired water;
`1000` converts m³ water to kg. These are proposed accounting equations, not a
selected growth/loss law or quotations from a surveyed model. Climate corrections,
growth ceilings and loss coefficients remain to be selected. Costs already
included in efficiency must not be subtracted again. Root allocation is implicit
in the aboveground efficiency; no whole-plant carbon closure is claimed. Regrowth
and seasonal activity need compatible empirical rules before demographic use.

## CPU/GPU consequences

Daily production and weekly biomass updates are regular per-plant operations,
suitable for compiled CPU and GPU kernels with structure-of-arrays state. They do
not require a plant×cell biomass array or a daily history. The existing sparse root
coupling remains reusable. This is an architectural inference, not a runtime
benchmark of any candidate model.

| FP64 array storage only | 59,483 plants | 547,588 plants |
|---|---:|---:|
| One aboveground biomass array (`8*P` bytes) | 0.45 MiB | 4.18 MiB |
| One daily-to-weekly production accumulator (`8*P`) | 0.45 MiB | 4.18 MiB |

These are total storage for those arrays, not increments over all current arrays
or complete peak memory. The current fixture already has a biomass array; giving
it an aboveground interpretation does not itself add a new array. Transactional
candidates, uptake accumulation, traits and scratch still add storage. No new
precision/resource-limit assumptions or runtime measurements are implied.

Daily weather-dependent demand can require a more frequent **demand-cache** update
than the weekly **root-geometry** rebuild. Keep those operations separate and
measure any extra sparse scatter. Detailed canopy radiation can also add spatial
work; it should be a separately bounded phase rather than presumed free. No full
model was compiled, run or ranked by runtime in this survey.

## Recommended bounded phases and decisions

1. Match total aboveground biomass units and live/dead conventions to observations;
   define species traits, temperature/phenology and aggregate production/loss rules.
   Calibrate canopy/root allometry separately from aboveground efficiency for the
   intended Jornada grasses and mesquite. Do not import unrelated model defaults.
2. Qualify a fixed-individual growth adapter against the existing native water
   budgets: wet/dry layers, zero uptake, dormant/warm periods, biomass losses and
   daily-versus-weekly integration. Compare constant and climate-aware efficiency.
3. Add empirical grass regrowth and shrub phenology with biomass-to-demand/geometry
   feedbacks, then recruitment and mortality under their own agreed scope.
   Benchmark the implemented growth rules with fixed population and full
   memory/water-accounting checks before larger demographic runs. More tissue
   states require a separate scientific reason and are not a prerequisite.

The current design target is **one total aboveground biomass state per individual**,
with transpiration-based production, species/PFT allometry and climate/phenology.
Select constant versus climate-corrected aboveground efficiency and compatible loss
rules before implementation. Surface/soil/root remapping remains explicitly deferred.

## Evidence and access limits

Local source hashes/HEADs, pinned public growth files and the full reference list
are in [the survey record](../agent_handoffs/tasks/water_biomass_survey_20261031/README.md).
That accepted archive retains the original survey and multi-pool recommendation;
the current document reflects the user's subsequent aboveground-only decision.
Original code/papers/adapters are distinguished above. The supplied ECOTONE source
is an adapted snapshot, not proof of exact Peters 2002 behavior; its wind component
and `Src/copy` backups were excluded. The original ECOTONE PDF was inaccessible;
Caylor 2004/2005 detailed methods remain unverified. This limits claims of exact
model reproduction, not the directly inspected local growth equations.
