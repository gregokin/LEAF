# User-supplied LEAF project outline

Source: user message on 2026-10-30. This is a scope/context record, not a newly verified assessment. Current authorization is handoff setup only. The user requested reading this outline before further substantive work.

## Purpose and immediate assessment gate

Work in `/home/okin/leaf`. Evaluate and develop vegetation and persistent soil-water processes for eventual coupling with MAPLE-SYRUP. The next substantive task is to compare Tietjen, STEPWAT2, and the supplied ECOTONE ecological core and recommend an implementation architecture. Present the assessment and proposed bounded phases before full implementation. Implementation, commits, pushes, and model runs require separately agreed scope.

MAPLE handles aeolian sediment transport. SYRUP handles water erosion and sediment transport using actual MAPLE sediment storage. MASY coordinates alternating wind and water events; wind and water transport do not occur simultaneously. Vegetation should respond to soil water, climate, establishment, competition and mortality, eventually landscape changes, and supply consistent physical properties to MAPLE and SYRUP.

Reuse scientific code where practical. Do not copy MAPLE wind physics, SYRUP sediment physics, or ECOTONE/WEMO wind physics into LEAF. Keep improvements to the existing models independently usable.

## Locations, provenance, and environments

- LEAF: `/home/okin/leaf`.
- MAPLE: `/home/okin/MAPLE`.
- SYRUP: `/home/okin/SYRUP`.
- MASY: `/home/okin/MASY`.
- MAHLERAN: `/home/okin/MAHLERAN`.

Read each repository's AGENTS.md before working there. Preserve dirty changes, simulations, and other agents' work. Prefer isolated development in LEAF; do not modify upstream projects as routine setup. Establish actual HEADs, dirty-source identities, and dependency locks before relying on implementations. Historical assessments used SYRUP HEAD `deb60c298` plus uncommitted changes and a qualified MAPLE39 dependency; these are historical identities, not current checkout claims.

Useful SYRUP instructions/documentation: `AGENTS.md`, `claude.md`, `docs/conservative_water_usage.md`, and `docs/maple39_shared_storage_qualification.md`.

Existing Python environments are `/home/okin/MAPLE/.venv` and `/home/okin/MAPLE/.venv-cupy`. Check them and project instructions before creating an environment or installing packages.

## Option A: Tietjen

- Tietjen, Zehe & Jeltsch (2009), "Simulating plant water availability in dry lands under climate change: a generic model of two soil layers," Water Resources Research 45, W01418: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006589
- Tietjen et al. (2010), "Effects of climate change on the coupled dynamics of water and vegetation in drylands," Ecohydrology 3, 226-237: https://doi.org/10.1002/eco.70
- Author PDF: https://www.biodiversity-plants.de/downloads/Tietjen_etal_2009_Ecohydro.pdf

The 2009 model supplies two-layer soil-water and ET processes. The 2010 model adds dynamic grass/shrub fractional cover, water competition, biweekly growth, and seasonal mortality/dispersal. It does not track individual age and size as gap models do.

Adapted MAHLERAN references: `src/Subroutines_Interstorm/calc_inf.f90`, `calc_et.f90`, `calc_sm.f90`, and `src/Subroutines_Vegetation/`. Do not assume the adaptations exactly reproduce the papers or copy questionable behavior blindly.

Previous assessment: `/home/okin/SYRUP/agent_handoffs/tasks/tietjen_feasibility_20261028/`, especially `review_workspace/assessment.md` and `root_review_disposition.md`.

## Option B: STEPWAT2

- Repository: https://github.com/DrylandEcology/STEPWAT2
- Palmquist et al. (2018), "STEPWAT2: An individual-based model for exploring the impact of climate and disturbance on dryland plant communities": https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.2394
- Water model: https://github.com/DrylandEcology/SOILWAT2

STEPWAT2 combines individual-plant demography, species/functional groups, and daily multilayer SOILWAT2 hydrology. Monthly transpiration by layer and plant group supplies resources for annual vegetation updates.

A selected source archive, not a runnable checkout, is under `/home/okin/SYRUP/agent_handoffs/tasks/vegetation_model_comparison_20261030/stepwat_source/`, with `stepwat_source_manifest.json` alongside it. Inspected STEPWAT2 revision: `003cb9b4adbf13d64a39d6d62fd95280adac27fc`; pinned SOILWAT2 submodule: `b05f410a9e96a5f26857bd37247fe7152e9861d8`.

Provided findings at those revisions: C with process-global state and linked individuals; grid cells load into global state; gridded driver reinitializes hydrological state on cell loading rather than retaining each cell's previous-year water; weather interface expects generated weather; "added transpiration" is ecological resource accounting, not a physical water input. Demographic reuse is attractive, but continuous storm coupling is not an existing plug-in interface. Recheck findings for any different revision.

## Option C: supplied ECOTONE ecological model

User-owned source: `/home/okin/Ecotone Wind 9/Ecotone Wind 9`. Read its `AGENTS.md` before work. IGNORE ITS WIND COMPONENT: do not port, adopt, or evaluate WEMO wind physics. MAPLE owns wind transport.

Ecological/soil-water sources in `Src/`: `ecotone.f`, `soilwater.f`, `soildrive.f`, `grow.f`, `gro_new.f`, `add_abiotics.f`, `swater.f`, `calcar.f`, `snitro.f`, `eco_param.inc`, `eco_varbl.inc`, `grow.inc`, and `linklists.inc`. Inputs/documentation: `Inputs/`, `Inputs/SoilIn/`, `Ecotone Subroutines.doc`, and `Ecotone Arrays.doc`. `Src/copy` contains backups, not the active implementation.

Reference: Peters (2002), "Plant species dominance at a grassland-shrubland ecotone: an individual-based gap dynamics model of herbaceous and woody species," Ecological Modelling 152, 5-32: https://doi.org/10.1016/S0304-3800(01)00460-4

Provided snapshot findings: fixed-form Fortran77 style with DEC/Intel STRUCTURE/RECORD and DFPORT; supplied Intel/Windows build projects, no tested portable build; annual ecology, daily water, monthly or five-day plant-water resources; individuals with biomass, age, species and layered roots; shared soil-type-indexed water/resource arrays, without an established independent persistent column per hydraulic cell. Active `gro_new.f` is principally water-driven; nitrogen is prescribed monthly and retained in bookkeeping, and old nitrogen/light comments need not describe active growth. Large fixed-capacity arrays need memory work. A wind-off switch does not qualify a runnable isolated ecology baseline.

## Existing comparative assessment and limits

Read `/home/okin/SYRUP/agent_handoffs/tasks/vegetation_model_comparison_20261030/`, especially `assessment.md`, `root_review_disposition.json`, `review_initial.md`, `review_ecotone.md`, and `ecotone_source_manifest.json`. Read final sections and root disposition; earlier sections predate discovery of local ECOTONE source.

Rejected inference: soil-type-indexed storage does NOT establish only one daily water column calculation per soil type; the annual driver calls the soil driver for every plot. No bottleneck or runtime ranking was measured. Neither STEPWAT2 nor ECOTONE was compiled or run for that assessment.

## CPU, GPU, and memory requirements

Evaluate CPU and GPU architecture from the beginning. Avoid Python loops for translated numerical kernels; prefer compiled CPU kernels and GPU-suitable arrays. Reusing compiled CPU ecology between GPU storms is legitimate and should be benchmarked before deciding to rewrite all demography for GPU.

Avoid repeated transfers, hidden synchronization, and full-domain copying. Allocate by actual populations and active work, not maximum populations or trajectories for every cell; stream/accumulate seasonal statistics. Preserve individuals when that is the chosen science: cohorts or aggregate cover change the model. Separate ecological patch/resource resolution from hydraulic resolution where justified, rather than creating an entire community per arbitrarily fine hydraulic cell.

Measure startup, steady-state time, peak CPU/GPU memory, transfer costs, and scaling. Preserve science, conservation, and equivalent restart; do not raise limits or discard state to pass checks.

## Common scientific and coupling requirements

Use one authoritative water budget and one authoritative vegetation state. Current SYRUP finalization can reset surface/soil water; continuous ecohydrology must preserve subsurface moisture across storms/dry periods, explicitly accounting for ET and drainage. Never run independent SYRUP and SOILWAT rainfall/infiltration on the same water. Assign owners for intake, storage, ET, drainage, runoff/runon. Water/root layers are distinct from MAPLE sediment active layers/voxels.

Define storm/daily/seasonal/annual order; persistent state, RNG and restart; root/demand to layer mappings; ecological patch to hydraulic cell mappings; biomass/cover to geometry and surface properties; and eventual erosion/deposition effects on depth, roots, burial and exposure. SYRUP hydraulics with changed ecology or water is a hybrid: document departures and benchmark, without claiming exact published-model reproduction.

## Provisional conclusions to verify, not a selection

Tietjen appears the smallest-state native CPU/GPU route if grass/shrub cover is sufficient. STEPWAT2 offers species demography/disturbance and reusable C with significant calendar/water-state integration. Supplied ECOTONE ecology is a credible individual-based alternative with portability and layout work and reusable readable routines. Local source availability removes an objection to ECOTONE; do not rank it lower for lack of source. Science needs determine the choice.

First verify conclusions and required ecological detail; propose bounded phases and distinguish unchanged reusable code, adapters, and new physics/numerics. Do not begin that assessment as part of documentation setup. Follow Codex-Claude bounded assignments, one writer per scope, independent review, and durable records. Establish local AGENTS.md, claude.md, and agent_handoffs consistently. Implementation, commits, pushes, and model runs remain outside the currently agreed setup scope.
