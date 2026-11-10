# Corrected public-only report: spatial individual-plant demography coupled to multilayer soil water, and generic CPU/GPU consequences

**Evidence status.** Three evidence classes are used: (V) verified by me earlier from public code or open text; (R) reviewer-supplied primary-source pointers and summaries, accepted as corrections but *not* re-inspected here (no tools enabled); (I) inference. A list of corrected and withdrawn claims is at the end.

## 1. Tietjen, Zehe & Jeltsch 2009 (WRR) and Tietjen et al. 2010 (Ecohydrology)

(V) The 2009 Crossref abstract describes a generic, process-based, spatially explicit two-layer soil moisture model tested at Israeli sites and emphasizing intra-annual rainfall variability. The 2010 OpenAlex abstract describes daily two-layer soil moisture with shrubs and grasses as two competing functional types applied to a Namibian savanna, including a CO₂ effect on plant water-use efficiency.

(R) The original texts (2009 full text; 2010 author PDF, methods pp. 2–3) specify 5 m cells; hourly rainfall/runoff sub-steps in the hydrology; two cover types (grass, shrub) per cell; biweekly wet-season logistic cover growth driven by the preceding 14-day mean layer water availability and root competition; and seasonal mortality and dispersal. My earlier report inferred the original method from the 2022 EcoHyD descendant (Irob et al. 2022, V: 30 × 30 cells of 5 × 5 m, three meta-PFTs, biweekly wet-season vegetation step, R scripts on figshare); that inference is withdrawn in favour of the original-source description above.

(I) Classification: a cover-fraction, cell-based ecohydrological model, not individual-based. Reusable methods: two-layer budget with PFT-specific root fractions; hourly surface/daily soil/biweekly vegetation/seasonal demography time-scale separation; logistic cover growth as a function of lagged layer water availability. **Not** to be attributed to the original equations: a transpiration-to-individual-biomass WUE mass balance (the CO₂/WUE effect is a growth modifier on cover, per the abstract, not a tracked carbon ledger). Missing for an individual model: positions, allometry, roots crossing cells, size structure; terrain burial/erosion not found.

## 2. STEPWAT2 revision 003cb9b4 with SOILWAT2 b05f410

All items (V) from the pinned revisions; current master may differ.

**Individuals.** `indiv_st` stores id, age, `relsize` (size relative to a full-sized individual), `res_required`, `res_avail`, `pr`, and `Next/Prev` list pointers; individuals form a doubly linked list per species, `calloc`-allocated and freed on death. There are no x/y, canopy or per-individual root-geometry fields. The plot is 1.0 m² (`plot.in`), and densities/seed rates in #/m² are converted to #/plot. Biomass = `relsize × mature_biomass`.

**Water→resource coupling.** Each demographic year `SXW_Run_SOILWAT` converts resource-group biomass to SOILWAT2 vegetation inputs and runs a full daily year; monthly transpiration by layer is collected and smoothed over a multi-year window. Group allocation accumulates relative active roots (root profile × phenology, rescaled annually by biomass) times transpiration per layer-month. Within a group individuals get water in proportion to size; `pr = res_required/res_avail` divides the growth modifier when > 1; growth is logistic in `relsize`; surplus becomes temporary “extra growth” removed at year end. (I) This is a biomass-ratio limitation, not a carbon/WUE budget.

**Time scales and persistence.** The demographic timestep is annual (species may be annual or perennial; annuals use a seedbank with beta-distributed establishment, perennials a fecundity-based `Species_NumEstablish`; clonal `max_vegunits` permits regrowth rather than new individuals). SOILWAT2 is daily. Non-gridded runs carry soil water across years. In gridded mode each cell is a full plot whose globals are swapped by `load_cell/unload_cell`; at this revision `load_cell` re-initialises SOILWAT2 weather, flow, soil temperature and soil water each year, with an in-code WARNING that a cell should ideally continue its previous state and a FIXME tied to SOILWAT2 re-entrancy. Cells interact via seed dispersal and colonization; `ST_seedDispersal.c` iterates source cells and candidate recipient cells with an exponential distance kernel and uniform draws (I: pairwise over cells as read; scaling with N not measured). RNG: seven streams designed to be unique per iteration/year/cell. Types: `RealF`=float, `RealD`=double, tolerance macros `EQ/GT/LT`; `MAX_LAYERS`=25, `MAX_TRANSP_REGIONS`=4 (other counts in SW_Defines.h left unverified).

**SOILWAT2 flow.** (V, summary-level) per-type potential transpiration followed by per-type `remove_from_soil`, stored as `transpiration[k][i]`. Whether type order affects results was not line-verified and is excluded from recommendations.

**Palmquist et al. 2018** (V abstract): individuals compete for multilayer soil water via temporal matching of water with active root distributions by depth; CoMSES lists Zenodo code.

(I) Reusable: root × phenology × layer-transpiration weighting; transpiration smoothing; per-cell RNG stream design; plot-area scaling. Adaptations: arrays instead of linked lists, positions/footprints, a re-entrant per-cell water model, roots over several cells. Missing: canopy overlap, lateral flow, carbon/WUE, terrain.

## 3. ECOTONE (Peters 2002)

(V) The USDA ARS record describes an individual-plant-based model of recruitment, growth and mortality in which temporal and spatial partitioning of soil water drove grass–shrub coexistence. (R/public record) annual demography with daily multilayer soil water. (I) Conceptual ancestor of STEPWAT2 (gap plot, daily water, species root distributions, staged recruitment). **Unknown in public-only terms:** plot size (conflicting snippets), layer count, the rule dividing layer water among individuals, growth/allocation form, positions. My earlier statement that ECOTONE uses biomass-ratio water limitation is withdrawn: no inspected public code supports it, and implementations can differ.

## 4. Caylor 2004 (GCB), Caylor & Shugart 2004 (LE), Caylor et al. 2005 (Ecosystems), Caylor et al. 2006 (WRR)

(V) GCB 2004: principal components of measured patch structure parameterize an ecophysiology productivity model across four sites (879–365 mm). LE 2004 (search-level): a daily productivity model applied to horizontal structure distributions from field 3-D canopy data. Ecosystems 2005 (search/EIS-level): daily soil-moisture bucket, stochastic storms, Priestley–Taylor ET, five surface components (canopy; grass and bare soil under/between canopies); canopies reduce under-canopy stress mid-gradient. (V Crossref + R full text) WRR 2006: tree centres as a 2-D Poisson process with circular canopy radius r, circular root radius R, ratio a_t = R/r, and local counts of overlapping roots/canopies; the stress/uptake optimality implies roots become **less laterally extended** (not shallower) with rainfall.

(I) These are structure-driven productivity and under/between-canopy stress analyses; the public abstracts/texts I read do not describe recruitment–growth–mortality loops, but abstracts alone cannot prove every implementation was static, and whether they share one code base is undocumented (2003 dissertation restricted). Caylor 2006 **does** provide a lateral root/canopy-overlap geometry precedent (statistical landscape formulation, not a tracked demographic simulator); my earlier “missing lateral root geometry” claim is corrected. Reusable: Poisson-canopy initial conditions and overlap statistics; under/between-canopy ET partitioning.

## 5. Additional close precedents

**Hi-sAFe (R; Dupraz et al. 2019, MDPI; code github.com/hisafe/hisafe).** Daily individual-tree/crop competition with 3-D soil voxels, tree roots overlapping voxels, demand/supply-based water uptake. (I) Closest direct precedent for roots crossing horizontal resource cells with conservative competition; agroforestry-specific and not a simple GPU-ready transplant.

**TROLL 4.0 (V open text).** Explicit 3-D individuals; layered soil with coarse horizontal voxels (e.g. 25 m × 25 m); root depth 0.35·dbh^0.54 with exponential fine-root profile; conductance-weighted uptake; daily water balance; allometric growth from wood NPP; seed bank per 1 m² cell; stress/starvation/treefall mortality; C++ via rcontroll (README states 3.1.7). Roots are explicitly confined to the column under the stem. (I) Strong allometry/carbon/demography precedent; weaker than Hi-sAFe on horizontal root sharing.

**GRASSMIND 2.0 (R; UFZ method PDF).** Daily physiology with WUE-based resource/demographic methods for grass individuals competing within 1 m patches **without** explicit within-patch locations. UFZ states v3.0 is under testing and not compared with v2.0; do not transfer v2 validation. (I) Useful as a grass demography/WUE comparator, not as a positional precedent.

**Holdo 2013 (V open).** Horizontally implicit; eight layers to 70 cm; Beta root profiles; Michaelis–Menten uptake by layer moisture and root mass; daily; code in Source S1. Compact per-layer competition kernel. SEIB-DGVM and TreeGrass remain unverified leads.

(I) WUE/carbon accounting: TROLL, GRASSMIND, Holdo (assimilation→biomass); Tietjen cover equations use water-driven logistic growth with a WUE modifier; STEPWAT2 uses biomass-ratio limitation. PFT/species: SOILWAT2 vegetation types vs STEPWAT2 groups/species; Tietjen two cover types; TROLL species traits. Vegetative spread: STEPWAT2 clonal regrowth; Tietjen grass dispersal. Terrain burial/erosion: not found in any inspected source.

## 6. Generic array/GPU design — hypotheses only

Symbols: N cells, L layers, P plants, E nonzero plant–(cell,layer) entries; 32-bit float/index 4 B, double 8 B. No benchmarks; no cost ranking is asserted.

- **Layout.** Linked records (STEPWAT2 style) defeat coalescing; NVIDIA’s best-practices guide prioritizes unit-stride warp access. A structure-of-arrays with stable IDs, a live mask and periodic compaction is the standard alternative (I).
- **Sparse incidence.** Footprints form a P × (N·L) sparse matrix. One shared weight/index array of E entries plus a transpose permutation (E indices) avoids duplicating weights into two CSR views; memory ≈ E·(4 idx + 4 weight + 4 perm) + (P+1)·4 + (N·L+1)·4. Formats and trade-offs follow Bell & Garland.
- **Conservative uptake.** A demand pass (plant-major), a per-(cell,layer) segmented reduction giving scale = min(1, supply/demand), and a gather pass yield uptake sums that equal removal in exact arithmetic (Hi-sAFe-style demand/supply, R). Floating-point residuals remain and must be audited; fixed-order segmented reductions aid run-to-run reproducibility but do not guarantee bitwise CPU/GPU identity. CUDA atomics are interference-free read-modify-writes and `atomicAdd` supports float/double, but their accumulation order is unspecified.
- **Graph refresh.** Incidence must be rebuilt whenever plant or soil geometry changes (recruitment, growth, death, terrain change), not necessarily once per year; count → exclusive scan → fill, with transpose by sort or second count. Which phase dominates (daily N·L water, irregular root links, spatial rebuild/sorting, canopy mapping, or host–device transfers) is unknown without measurement.
- **Persistent vs scratch.** Persistent: soil water N·L (double, as SOILWAT2), plant SoA, incidence, RNG counters. Scratch: per-(cell,layer), per-plant, per-entry temporaries, preallocated.
- **Widths.** 32-bit indices suffice while E, P, N·L < 2³¹; size the E index type separately. Float for weights needs a numerical check against long-run drift and small-flux cancellation; keep the water balance in double unless tested.
- **RNG/restart.** Counter-based generators (Random123/Philox) are stateless functions of (key, counter) and support scheduling-independent reproducibility only with an explicit key/counter/draw-index contract and complete checkpointing of counters and all persistent arrays.
- **CPU/GPU split.** Branchy annual demography can stay on the CPU with device-resident soil state if transfers (P × attribute bytes) are bounded; pairwise dispersal should be bounded by a kernel radius before scaling N (I).

## Corrections and unverified items from the original report

Corrected: (1) Tietjen 2009/2010 methods now from original texts (5 m cells, hourly rainfall/runoff, biweekly logistic cover growth, seasonal mortality/dispersal), no WUE mass balance attributed. (2) Caylor 2006 recognized as a lateral root/canopy overlap geometry precedent; “shallower roots” → less laterally extended; static-structure classification softened to “not described in abstracts.” (3) ECOTONE biomass-ratio claim withdrawn. (4) Hi-sAFe added as the strongest roots-crossing-cells precedent; GRASSMIND added as a non-positional WUE comparator. (5) “Daily water is dominant,” “float is adequate,” annual-only rebuilds, bitwise reproducibility and exact conservation statements downgraded to hypotheses with qualifications. (6) O(N²) dispersal, SOILWAT type-order effects and NVEGTYPES removed from recommendations. (7) “Plants are annual” restated as an annual demographic timestep.

Left unverified: ECOTONE plot size/layers/allocation rule; SOILWAT2 vegetation-type count and order effects; SEIB-DGVM and TreeGrass details; whether Caylor studies share one implementation; TROLL simultaneous vs sequential column update; current STEPWAT2/SOILWAT2 master behaviour.
