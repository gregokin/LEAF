# Spatial individual vegetation coupled to gridded soil water

Assessment date: 2026-10-30. Status: architecture recommendation for discussion; no scientific implementation or model benchmark. Numerical examples are capacity arithmetic, not measured allocations or runtimes.

## Recommendation

The proposed hybrid is feasible in principle at 128²–512² hydraulic cells and 0.25–2 m spacing. Retain plants as spatial individuals, keep water on a layered grid, and connect them through sparse root footprints. A plant community should not be instantiated independently in every hydraulic cell: individual positions and root/canopy extents can cross cell boundaries.

Recommend a small, explicit ecological core drawing on ECOTONE and grass-model methods, rather than adopting an entire gap-model driver. Use Caylor for spatial structure and Hi-sAFe for the precedent of shared soil voxels. Evaluate a conservative layered-column water scheme in SYRUP separately from the demographic choice. STEPWAT2 remains valuable if its species demography and disturbance behavior are scientific priorities, but it does not supply the required spatial or continuous-water interface unchanged.

The principal capacity variable is the number of plant–soil connections. The soil grid itself is modest. Whether graph construction, daily uptake, canopy projection, or transfers controls runtime remains unmeasured. Mutually exclusive transport and growth phases allow scratch reuse, while their persistent states still coexist.

## Agreed design context

- Store plant identity, X/Y, **ground elevation at establishment**, biomass, age, species and functional type. Current plant height follows allometry; establishment plant height is not required.
- Root systems and canopies may span cells; subcell positions matter even when cells are large.
- The shared MASY landscape supplies authoritative ground and soil conditions, including evolving texture. Rooting/water layers are distinct from sediment voxels and MAPLE's active layer.
- Wind transport, water transport, and biological water use/growth occupy separate intervals. Between events, regularly update transpiration, evaporation and drainage; accumulate actual uptake for a coarser biomass update, provisionally weekly.
- Burial, erosion/root exposure or dislodgement, and abrasion have species-specific responses, including differences within a functional type. Include temperature, age and stochastic mortality; defer fire.
- Soil moisture initially has no effect on wind transport. Reserve a versioned read-only moisture interface for later qualification.

Interpret the user's “1–2 M” as metres of cell spacing. Plant density and the definition of a grass individual remain unspecified; the population examples below do not select them.

## Scientific precedents and limits

| Model or study | Relevant precedent | What must change or be added |
| --- | --- | --- |
| Tietjen 2009/2010 | Two-layer moisture/ET and separated hydrological, biweekly and seasonal updates. | The vegetation is grass/shrub fractional cover with moisture-driven logistic growth, not spatial individuals or a transpiration-to-biomass ledger. |
| STEPWAT2/SOILWAT2 | Species and functional groups, individual age/size, establishment and disturbance; daily multilayer water supports annual demography through monthly resources. | Add subcell positions and crossing footprints; adapt globals, weather, persistent water and calendar. Annual routines cannot simply become weekly routines. |
| Supplied ECOTONE ecology | Individual biomass/age/species, layered roots and a direct water-to-biomass conversion in the active growth code. | Portable ecological baseline, spatial X/Y, independent grid water, bounded arrays and new time integration. |
| Caylor studies | Spatial canopy structure affects productivity and water stress; root and canopy footprints need not have the same radius. | Methods for spatial coupling, rather than a demonstrated complete demographic/code package for this project. |
| Hi-sAFe | Individual tree roots occupy multiple soil voxels; plants compete for water within shared voxels. | Its hydraulic-potential competition and agroforestry framework are more complex than the proposed minimal model; no GPU qualification established here. |
| GRASSMIND 2.0 | Individual biomass compartments, allometry, daily physiology and WUE-based water demand. | Individuals have no explicit positions within a patch; spatial footprints would be new. |

Tietjen's original methods provide a useful small-state reference, but their runoff must not duplicate SYRUP hydraulics. The 2010 model uses preceding 14-day water availability for cover growth; attributing the proposed individual WUE budget to it would be inaccurate. MAHLERAN contains adaptations, including storage clamping, so equation and budget checks are needed before reuse. [Tietjen 2009](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006589), [Tietjen 2010 author PDF](https://www.biodiversity-plants.de/downloads/Tietjen_etal_2009_Ecohydro.pdf).

The [STEPWAT2 paper](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.2394) supports the demographic/resource separation. At the inspected revision, linked individual records have no X/Y or individual canopy/root footprint. The gridded loader explicitly reinitializes hydrological state rather than restoring the cell's previous year, and its weather interface expects generated weather. “Added transpiration” in resource allocation must remain separate from physical water. These are revision-specific integration constraints, not evidence against the ecological methods. See archived [loader](../../SYRUP/agent_handoffs/tasks/vegetation_model_comparison_20261030/stepwat_source/ST_grid.c), [weather interface](../../SYRUP/agent_handoffs/tasks/vegetation_model_comparison_20261030/stepwat_source/sxw_soilwat.c), and [resource allocation](../../SYRUP/agent_handoffs/tasks/vegetation_model_comparison_20261030/stepwat_source/sxw_resource.c).

The Caylor papers address complementary questions. The two 2004 studies connect measured canopy/vertical structure to productivity; the 2005 study distinguishes under-canopy and intercanopy water/stress. The 2006 model explicitly represents overlapping lateral root and canopy circles. Its landscape statistics are a strong footprint precedent, but do not establish a ready individual growth/recruitment driver. Full methods were available for 2006; the other three were checked at publisher abstract level, so detailed shared-code claims remain unverified. [Caylor et al. 2004](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2486.2003.00704.x), [Caylor and Shugart 2004](https://link.springer.com/article/10.1023/B:LAND.0000030450.11302.c2), [Caylor et al. 2005](https://link.springer.com/article/10.1007/s10021-004-0027-9), [Caylor et al. 2006](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2005WR004683).

Hi-sAFe is the closest additional precedent for plants sharing horizontally and vertically distributed soil resources. Its water competition uses plant/soil potentials and root transport properties; the proportional cap proposed below is a simpler candidate law, **not a reproduction of Hi-sAFe**. Its published architecture uses Java and STICS, so methods are more immediately relevant than wholesale adoption. [Hi-sAFe description](https://www.mdpi.com/2071-1050/11/8/2293), [authors' source repository](https://github.com/hisafe/hisafe).

GRASSMIND 2.0 derives geometry from biomass and uses a water/carbon framework, but treats within-patch overlap implicitly. It also approximates generative and vegetative recruitment through a common investment, which would not by itself meet a requirement to distinguish seed-only and vegetative strategies. Its methods are useful comparators rather than a spatial drop-in. [Model description](https://formind.org/wp-content/uploads/2020/05/Grassmind_ModelDescription.pdf).

MAESPA is another individual-canopy/physiology comparator, but its documented soil-water state is stand-averaged, so it does not supply independent grid-cell root sharing. [MAESPA paper](https://gmd.copernicus.org/articles/5/919/2012/index.html), [manual](https://maespa.github.io/manual.html).

## What the local infrastructure already supplies

| Existing component | Reusable behavior | Current boundary |
| --- | --- | --- |
| [MAPLE vegetation types](../../MAPLE/src/maple/core/types/vegetation.py) | Subcell object centres, elliptical footprints, height, porosity and drag; five derived physical rasters. | Objects lack ecological identity/state. Current native wind input is rasterized; preserving individual coordinates does not make wind resolve every subcell arrangement. |
| [MAPLE rasterizer](../../MAPLE/src/maple/core/types/vegetation.py) | Defined projection/overlap semantics suitable as a reference. | Host NumPy plant loop with footprint sampling; recurring ecological publication needs timing and possibly compiled/GPU implementation. |
| [MASY vegetation boundary](../../MASY/src/masy/landscape_vegetation.py) | Version/hash checks, transactions, receipts and staging estimates. | Current profile accepts host FP64 rasters, rejects object plants without stable IDs, and refuses nonempty water-exchange channels. |
| [MASY dense resources](../../MASY/src/masy/dense_device_resources.py) | Device/stream ownership and resource admission patterns. | Soil-water validation currently describes one 2D FP64 array, not layered ecological state. |
| [MASY scheduling](../../MASY/src/masy/dense_schedule.py) and [clock](../../MASY/src/masy/clock.py) | Ordered native events and precise native clock handling. | Relative wind/rain schedule lacks ecological/calendar intervals. Add an adapter while preserving native timing semantics. |
| [SYRUP infiltration](../../SYRUP/src/maple_syrup/infiltration.py) | Array backend, validation and explicit source/destination water accounting. | Single soil store per cell; layered intake, persistent moisture and ET need a revised contract. |
| [SYRUP pond finalization](../../SYRUP/src/maple_syrup/pond_event.py) and [complete-event finalization](../../SYRUP/src/maple_syrup/complete_event.py) | Existing reset profiles and named removal accounting. | Finalization can explicitly remove both soil and surface water. Continuous ecology needs a separately versioned retained-soil profile. |

Keep existing qualified behavior independently usable. Do not reinterpret reset removals as ET, recover discarded water after finalization, or silently change old checkpoints. Retaining subsurface water and handling residual surface ponding are separate decisions. Previous small-domain/manual vegetation or bounded four-event results do not qualify dynamic ecology at 512².

ECOTONE's active [plant record](</home/okin/Ecotone Wind 9/Ecotone Wind 9/Src/eco_varbl.inc>) has biomass, roots, age and species. Its added `x` field is a scaled gap variable, not an inspected geographic X/Y pair. In [gro_new.f](</home/okin/Ecotone Wind 9/Ecotone Wind 9/Src/gro_new.f>), `WUECNST` is water mass per biomass mass: remaining allocated water after maintenance/redistribution is divided by it to form biomass. This is directly relevant, with inverse units to the usual biomass-per-water WUE. [swater.f](</home/okin/Ecotone Wind 9/Ecotone Wind 9/Src/swater.f>) apportions layer transpiration through active roots, but its soil-type-indexed resources are not independent persistent hydraulic-cell stores.

This snapshot uses DEC/Intel Fortran extensions and fixed capacities. Two declared `70000 × 20 × 73` REAL root-resource arrays alone imply about 780 MiB if REAL occupies four bytes; that is declaration arithmetic, not measured memory. The annual driver calls the soil driver per plot: soil-type indexing does **not** imply one cheap daily column per soil type. No portable build, isolated ecological baseline or runtime ranking was established. Active growth is principally water-driven; prescribed monthly nitrogen bookkeeping does not demonstrate dynamic nitrogen limitation. Ignore ECOTONE's wind component entirely. [Original ecological reference](https://doi.org/10.1016/S0304-3800(01)00460-4).

## Proposed state and ownership

Use a structure of arrays for live plants, with stable IDs independent of compacted array slots. Store species/PFT indices once and look up traits; avoid repeating trait tables for every individual. Keep biomass compartments only where the selected growth/allometry requires them. Optional per-layer root biomass adds O(P L), rather than a historical resource array for every possible plant.

Store layered water and hydraulic parameters in arrays accessible to both compiled CPU and GPU kernels. A layer-major `[L, N]` layout permits adjacent GPU threads to process adjacent columns, with a short compiled layer loop. A conservative bucket/finite-volume column is an appropriate minimal candidate; choose retention, conductivity, evaporation and drainage equations explicitly before claiming a published-model match. Four/eight layers below are examples, not a selected discretization. Lateral subsurface flow, capillary redistribution and freeze–thaw should be included only through an explicit scientific scope.

Keep one canonical population and publish derived, versioned physical views. The plant table and native rasters must not be independently editable vegetation authorities. A new schema must bind population, projected vegetation, soil state, surface identity, climate cursor, clock and random state. Existing manual raster-delta edits need a defined reconciliation policy in this mode.

Shared surface texture and soil depth determine hydraulic parameters through an explicit mapping; sediment classes alone do not automatically specify retention curves or rooting conditions. Erosion/deposition can change pore capacity and layer geometry. Remap retained water conservatively and name any displaced water/runoff/drainage. Clipping water to a new capacity without accounting is unacceptable.

## Sparse roots and daily competition

Represent only nonzero plant–(cell, layer) connections. Distinguish each plant's normalized spatial root allocation from the relative competitive shares of all roots in a soil element; they are not interchangeable fractions. PFT/species allometry can generate a bounded horizontal footprint and vertical root distribution, modified by local soil depth and the chosen rooting law.

At the beginning of a quiet period, construct or validate the mapping. Rebuild affected footprints after root growth, recruitment, death or relevant surface/layer changes. Geometry may remain cached between those changes while daily moisture/activity factors vary. A once-per-year rebuild is insufficient; unconditional daily reconstruction may also waste work.

A possible minimal allocator is:

1. Compute plant transpiration demand `D[p]`, normalized root weights `w[p,j]` and moisture response `f[p,j]`.
2. Request `r[p,j] = D[p] × w[p,j] × f[p,j]`; ensure total requests do not exceed the plant demand.
3. Reduce requests per soil element: `R[j] = sum_p r[p,j]`. Set `a[j] = min(1, available[j]/R[j])`, with a defined zero-demand case.
4. Accept `F[p,j] = a[j] × r[p,j]`. Remove that same accepted flux from soil and accumulate it as actual plant transpiration.

This candidate avoids shared-store overdraw and arbitrary plant iteration priority. It requires a scientific decision about redistribution of unmet demand to wetter layers/cells, root-density competition and soil-potential effects. It is not established as equivalent to ECOTONE, STEPWAT2 or Hi-sAFe. Soil evaporation competes for the same finite stores and needs an explicit order or joint allocation rule. A reference PET value is not universally an upper bound on all canopy ET; storage and explicitly selected demand laws supply the physical constraints.

Use separate reduction/gather passes, rather than racing plant kernels that read and decrement a shared cell. Track floating residuals in both soil and plant sums. Start with FP64 water accounting, consistent with native interfaces; qualify other precisions rather than assuming them adequate. Fixed reduction ordering aids repeatability but does not guarantee bitwise CPU/GPU identity. CuPy's documented `scatter_add` type list does not include FP64, so a NumPy scatter translation needs an explicitly qualified reduction/kernel path. [CuPy scatter documentation](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scatter_add.html).

## Biomass, recruitment, disturbance and time

Convert **actual transpiration**, not soil evaporation or potential demand, into growth resources. Specify whether WUE means carbon or dry biomass per water, then account for carbon fraction, maintenance, turnover and root/shoot allocation consistently. Water-limited growth still needs seasonal/temperature or energy constraints; plentiful water alone should not imply unrestricted winter production. Weekly biomass updates should consume accumulated daily uptake exactly once. Replacing annual gap-model updates with weekly calls changes rates and feedbacks and requires validation.

Initially, simple recruitment can use species seedbank counts by ecological patch or cell, germination/establishment probabilities, and sampled subcell locations. Vegetative recruitment can create nearby offspring with stable identities and parent links where biologically appropriate. Seedbank counts do not require one agent per seed. Define whether grass individuals are genets, ramets or clumps before sizing populations; replacing selected individuals with cohorts would change the model. Distinguish clonal regrowth of an existing plant from recruitment of a new one.

Use establishment ground elevation as a reference for burial/exposure, with current root and stem geometry. A single height difference cannot encode burial followed by exposure, cumulative abrasion, or root damage. Retain bounded exposure/damage accumulators when required by the selected species laws. Define how partial burial changes exposed canopy geometry and where dead biomass goes; do not assume mortality immediately removes all physical shelter. Frost mortality based on air temperature is distinct from soil freezing/snow hydrology, whose water accounting needs a separate decision.

One physical calendar must include event durations, partial days and year/season boundaries. Split quiet intervals at daily and biomass-update boundaries without resetting native clocks or inventing zero-forcing transport events. ET/growth remains disabled during events as requested; calendar age and forcing position still advance. State explicitly whether event-time subsurface redistribution belongs to the storm solver or is omitted under the initial approximation. After transport, reconcile terrain/texture, apply event injury, update mappings and publish the next valid state. Carry weekly uptake across interruptions; do not discard a partial week.

Use plant-ID/calendar/process/draw-index random streams with an explicit algorithm. Checkpoint water, plants, RNG counters, climate/calendar cursors, pending uptake, damage and demographic accumulators, plus parameter/surface identities. Derived maps may be rebuilt if that preserves the defined restart behavior. Reordering/compacting plants must not silently change stochastic outcomes.

## Memory at the requested scale

Let `N` be hydraulic cells, `L` water layers, `P` live plants, `k` mean horizontal root cells per plant, and `E = P k L` expanded root connections. For a rough interior circular footprint, `k ≈ π r² / dx²`; actual cell intersections, clipping and heterogeneous roots determine the true count.

An illustrative allocation uses eight FP64 fields per soil element, 128 bytes per live plant, five native FP64 rasters, and 28 bytes per root connection: two 32-bit indices plus FP64 weight, one reverse permutation, and one FP64 flux buffer. CSR offsets add `4(NL + P + 2)` bytes. These choices are a transparent estimate, not a finished storage design:

`bytes ≈ 64NL + 128P + 40N + 28E + 4(NL + P + 2)`.

| Grid | L | Live plants | k | Expanded connections | Illustrative total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128² | 4 | 10,000 | 4 | 0.16 million | 10.4 MiB |
| 256² | 4 | 50,000 | 12 | 2.4 million | 89.9 MiB |
| 512² | 4 | 100,000 | 12 | 4.8 million | 218.8 MiB |
| 512² | 8 | 100,000 | 50 | 40 million | 1.20 GiB |
| 512² | 8 | 1,000,000 | 50 | 400 million | 10.70 GiB |

Excludes sediment/transport state, staging copies, graph rebuild/sort workspace, CUDA context/pools, host/checkpoint mirrors, and plant state beyond the assumed record. One FP64 water array at 512² × 8 is only 16 MiB. A dense 100,000-plant × 512²-cell × 8-layer FP64 tensor would be about **1.53 TiB** and must not be constructed. Full arithmetic is in [memory estimates](../agent_handoffs/tasks/individual_vegetation_feasibility_20261030/memory_estimates.json).

The inspected machine exposes four GTX 1080 Ti devices with 11 GiB each; this is not a pooled 44 GiB allocation. The largest illustrative case leaves insufficient room for shared transport state and scratch on one device. The 1.20 GiB case is a plausible candidate for measurement, not a demonstrated whole-model fit. [Read-only device metadata](../agent_handoffs/tasks/individual_vegetation_feasibility_20261030/gpu_metadata.json).

At fixed physical area, refining 1 m to 0.25 m increases both cell count and approximate footprint intersections by sixteen. At fixed 512² cells, it instead shrinks the domain: area is 1.6384 ha at 0.25 m, 26.2144 ha at 1 m, and 104.8576 ha at 2 m. Population examples therefore must be interpreted with density and domain area together.

If scientifically valid, separate horizontal incidence `P k` from per-plant vertical weights `P L`; avoid materializing the L-fold expansion. Soil heterogeneity can prevent simple separability. Alternatives are exact tiled/layer-wise evaluation and recomputing cheap weights. Share weights between plant- and soil-oriented views using a permutation. Admit actual nonzeros and populations before mutation, guarding index/offset widths; do not allocate maximal communities everywhere or truncate plants to fit.

## Runtime and GPU strategy

Daily column work scales with `N L`, uptake/reductions with expanded `E`, and individual updates with `P`. Footprint construction and canopy projection scale with actual intersections plus indexing/sorting work. No `P × N` search is necessary: use bounded spatial bins or footprint bounding boxes and count/scan/fill. Clumping and a few large plants can make average-only estimates misleading.

Benchmark compiled CPU and GPU paths using the same laws. GPU-resident water/uptake with CPU demographic updates is legitimate, as is a compiled CPU quiet-period baseline with bounded transfers at phase boundaries. Avoid Python per-cell/per-plant hot loops. Transfer sufficient summaries for weekly demography, rather than the full root graph every day. Cache device state and use dirty regions where semantics permit.

Phase separation permits a budget of `persistent shared state + max(wind work, water work, ecology work/maps) + publication/rebuild peak` if ecological maps are released during events. Maps kept for reuse become persistent instead. Allocator pools can retain released blocks; measure reserved as well as live memory. Timing must include synchronized GPU events, startup/JIT separately from steady state, transfers, mapping, publication and complete simulated periods. [CuPy performance](https://docs.cupy.dev/en/stable/user_guide/performance.html), [memory management](https://docs.cupy.dev/en/stable/user_guide/memory.html).

## Reuse classification and bounded phases

**Preserve and reuse:** MAPLE/SYRUP transport physics and shared bed; native clock semantics; vegetation raster definitions; boundary versioning and resource-admission patterns. Published ecological equations may be reusable once their exact scope and units are selected. A compiled CPU ecological reference remains a valid option to qualify.

**Adapters/engineering:** plant SoA and stable IDs; sparse root mapping; geometry projection; layered persistent state/schema; re-entrant interfaces; calendar and climate input; portable non-wind ECOTONE extraction; restart and memory admission. Reference comparisons must separate layout-only changes from altered model equations.

**New or changed scientific/numerical work:** spatial competition and compensatory uptake; selected WUE/growth time basis; trait/allometry parameterization; texture-to-hydraulic mapping and conservative evolving-layer remapping; seeds/clonal establishment; species burial/exposure/abrasion/frost responses. A SYRUP/ecological hybrid should be described and benchmarked as such, without exact-publication reproduction claims.

Proposed phases, each requiring separately agreed implementation scope:

1. **Fix scientific contracts and reference cases.** Define grass individual, biomass/WUE units, rooting and water competition, initial PFT/species and climate/calendar. Select a small non-wind reference to qualify; do not first port an entire annual driver.
2. **Measure geometry and capacity.** A bounded compiled CPU/GPU prototype for mapping and canopy projection at 128², 256² and 512², with 0.25/1/2 m spacing, stated plant densities/radii, clumped and dispersed patterns, and changing footprints. Measure startup, steady state, peak CPU/GPU reserved memory and transfer costs. This is the recommended first coding assignment after discussion.
3. **Persistent water with fixed plants.** Add the retained-water/calendar contract and selected layered laws. Verify single-column/domain budgets, dry/wet extremes, texture/capacity changes, event crossings, partial days and uninterrupted versus restarted behavior.
4. **Individual uptake and biomass.** Add conservative sharing, weekly accumulation, allometry and mapping invalidation. Check overlapping roots, dry-layer demand compensation, units, biomass accounting and CPU/GPU tolerances.
5. **Demography and event injury.** Add selected seed/clonal processes and species damage laws; then qualify native sequential wind/water/ecology at the requested scales with source-bound resource and restart evidence.

## Source identities and verification limits

| Checkout | Actual inspected HEAD |
| --- | --- |
| MAPLE | `963b06ad1cf0f3e2c420a7d7fb50939e3bd43b7d` |
| SYRUP | `deb60c29834a4aad0bf8cb4cfbb611fc28ebe090` |
| MASY | `68936bfae87db5243912ea5fc69195c113883bd0` |
| MAHLERAN | `305bd95d32123f13708be2f9a88e42ddd45d6f28` |
| Supplied ECOTONE | No usable Git metadata; active-source hashes recorded. |

SYRUP has dirty and untracked source; other workspaces contain existing evidence/artifacts. HEAD alone is insufficient. The SYRUP qualification lock names MAPLE `39bcf79e1e67b6ed960f78a923ddfedf1d173e68`, not the live MAPLE HEAD above. Its accepted scope is not a qualification of the inspected live trio for continuous ecology. See [source manifest](../agent_handoffs/tasks/individual_vegetation_feasibility_20261030/source_provenance_before.json) and the [qualification lock](../../SYRUP/dependencies/maple_shared_storage/qualification.lock.json).

The archived STEPWAT2 selection is revision `003cb9b4adbf13d64a39d6d62fd95280adac27fc`, SOILWAT2 submodule `b05f410a9e96a5f26857bd37247fe7152e9861d8`; it is not a complete runnable checkout. All 12 archived file hashes and 13 selected active ECOTONE ecological source hashes match the prior manifests. [Comparison evidence](../agent_handoffs/tasks/individual_vegetation_feasibility_20261030/archived_source_verification.json).

Claude supplied a bounded public-source review and correction; Codex independently checked primary literature and local non-wind code, rejected unsupported performance and model-equivalence claims, and integrated this recommendation. Private local source/design was not supplied to Claude. No ecological model was compiled/run, no GPU benchmark was performed, and no upstream source, environment or dependency was changed. The research supports feasibility and the proposed measurements; it does not establish an accepted scientific parameterization or measured speed ranking.
