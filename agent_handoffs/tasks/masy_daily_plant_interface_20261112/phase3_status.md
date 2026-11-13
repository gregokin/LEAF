# Phase 3 resumption report

The interface priority is complete and independently accepted. Resume Phase 3 from its existing mechanics, not by rebuilding Phases 1–3. No new reproductive science, parameterization, trial run or plant generation occurred during this task.

## Already implemented

The opt-in species-local `simple_seed_clonal_v1` law supports seed establishment and vegetative reproduction independently. Annual opportunities use explicit counters and deterministic keyed randomness. New individuals have their own stable IDs, age zero and establishment-ground elevation sampled at their locations.

Seed establishment uses an explicit ecological patch grid, independent of hydraulic cell resolution. Seedling biomass is an external seed/establishment credit. There is currently no seed bank, dispersal kernel or moisture/temperature filter for establishment.

Vegetative reproduction creates nearby daughter ramets. Accepted daughters receive dry aboveground biomass transferred from their parent; failed placements cost no biomass. Parent/daughter mass closes, survivors retain state, and newborns request water from the next open interval. Canopy overlap during reproduction is currently allowed. The annual placement evaluator remains NumPy on CPU, with declared bounded cold transfers for CUDA sessions; it has not been rewritten as a GPU demographic kernel.

The new water-free session exposes these same mechanics through `stage_reproduction` / `stage_decline_reproduction`, after growth, disturbance, deferred-season and annual-mortality boundaries. Biological checkpoints preserve counters, RNG seed, patch configuration, pending growth and geometry. Independent tests include annual opportunities, births and save/restore.

## Scenario and outstanding science

The Jornada scenario is unchanged: BOER uses vegetative reproduction; PROGLA uses seed establishment. The selected general policy remains Sporobolus by seed, black/blue grama vegetatively. Existing BOER probability0.1 per eligible parent/year, PROGLA probability0.05 per patch/year, seedling/daughter sizes, eligibility, spacing and establishment timing are provisional assumptions or explicitly labeled proxies, not calibrated field rates.

Next Phase3 work is to justify recruitment probabilities, ecological patch size, eligible ages/biomass, daughter placement and establishment timing using observations; decide whether seed availability and water-conditioned establishment are needed; and validate recruit counts/survival. That is scientific parameterization and validation beyond the tested mechanics. The trial case has not been run, regenerated or recalibrated.
