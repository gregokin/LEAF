# LEAF — Landscape Ecohydrology with Agent-based Flora

LEAF will evaluate vegetation and persistent soil-water processes for eventual
coupling with [MAPLE](../MAPLE), [SYRUP](../SYRUP), and [MASY](../MASY). MAPLE
supplies wind transport, SYRUP supplies water and sediment processes, and MASY
coordinates alternating events.

LEAF owns plant state. Species-specific allometry connects **aboveground dry
biomass** to shared canopy height and width for MAPLE and SYRUP. Porosity
is initially constant, unused by LEAF ecology but used by native MAPLE wind. The
[canopy coupling requirement](agent_handoffs/project_brief.md#canopy-allometry-and-maple-coupling--2026-11-08)
also defines feedback for vegetation damage and burial. Native SYRUP moisture
coupling is implemented. LEAF now publishes MAPLE-compatible canopy objects and
accepts event diagnostics; MAPLE / SYRUP adoption and calibrated ecological laws remain future work (uncalibrated
species injury and mortality responses are available from Phase 2 below).

## Phase 0: initialize and exchange vegetation

The [Phase 0 guide](docs/phase0-vegetation.md) documents the implemented species
JSON catalog and canonical plant CSV. The same CSV format serves initial input,
user-cadenced snapshots and final output. It records stable IDs, species, X/Y,
ground elevation at establishment, total aboveground **dry** biomass in kg,
seasonal biomass, age in days and status. Initial `SURFACE` anchors resolve once
against the supplied topography and are then saved as numeric elevations.

**Plant files carry no embedded metadata (project convention).** A LEAF plant CSV
is plain: the nine-column header on line 1 and nine fields on every data row, with
no `#` metadata line, JSON or comments — in every file LEAF writes (CLI `init` /
`generate`, snapshots, final output, checkpoint `plants.csv`, preview tables).
File metadata (time, domain, species-catalog identity, row count, units, initial /
`SURFACE` declaration, provenance and the CSV's SHA-256) lives in a separate,
human-readable companion JSON `<stem>.metadata.json` beside the CSV
(`initial_plants.csv` → `initial_plants.metadata.json`, `plants.csv` →
`plants.metadata.json`). Readers load and verify the companion automatically and
refuse a CSV that does not match it; a handwritten plain list needs no companion
(the caller supplies the domain). Old files with an embedded `# {json}` first line
can still be read, but nothing writes that form. Details: the
[Phase 0 guide](docs/phase0-vegetation.md#23-plant-csv-and-companion-metadata).

Generate multiple species from target cover and mean/standard-deviation canopy
diameter, with seeded random placement and optional canopy exclusion. Cover is
per-species union ground cover on a finite domain, measured at a declared sampling
resolution and tolerance. Size draws are positive lognormal; exclusion places
larger canopies first and refuses infeasible requests within explicit resource bounds. Initial ages are drawn
independently and uniformly from 0 to each species' nominal life expectancy (read from its catalog record; the default
when `initial_age_days` is omitted, selector `{"distribution": "uniform_to_life_expectancy"}`); this is an initial
sampling range, not a death age, and a species with `life_expectancy: null` must be given explicit ages
([details](docs/phase0-vegetation.md#4-generator-leafinitialization)).

The [MAPLE developer handoff](docs/maple-leaf-integration.md) describes shared
loading, placement and allometry calls for prescribed vegetation, without requiring
growth or hydrology. It includes stable-ID event sockets and safe handling of
buried or elevated canopies. No MAPLE or SYRUP source was changed for Phase 0.
Example allometry and ecological traits are uncalibrated fixtures.

The [shared canopy interface](docs/shared-canopy.md), `leaf.canopy.canopy_view`,
provides one derived representation for both consumers: locations, dimensions,
porosity, identities and burial/exposure diagnostics. The existing
`maple_vegetation_view` is a thin adapter over it, retaining MAPLE's native
format restrictions. SYRUP can consume the common view directly; upstream
adoption and model-specific physics remain future work. Allometry and surface
geometry have one implementation, so updates reach both consumers together.

Run from this checkout with the existing environment (NumPy alone is sufficient
for the Phase 0 core):

```sh
../MAPLE/.venv-cupy/bin/python -m leaf validate-species --species examples/phase0/species.json
../MAPLE/.venv-cupy/bin/python -m leaf generate \
  --species examples/phase0/species.json --config examples/phase0/generation.json \
  --surface-constant 100 --out /tmp/leaf-initial.csv --report /tmp/leaf-initial-report.json
../MAPLE/.venv-cupy/bin/python -m leaf export-canopy \
  --species examples/phase0/species.json --plants /tmp/leaf-initial.csv \
  --surface-constant 100 --out /tmp/leaf-canopies.json
```

Source is in `leaf/`, tests in `tests/`, runnable fixtures in `examples/phase0/`,
research tools in `analysis/` and `benchmarks/`, and source identities in
`dependencies/`. `pyproject.toml` declares the package and optional growth/GPU
dependencies; no installation is needed to use this checkout.

The [individual vegetation feasibility assessment](docs/individual-vegetation-feasibility.md)
compares model precedents, inspects the current integration interfaces, and
recommends sparse connections between spatial plants and layered soil water.
It includes capacity estimates for 128²–512² grids and proposed bounded phases.
This is an architecture recommendation for discussion; no calibrated ecological
model has been selected. Subsequent bounded implementations
and their qualifications are described below.

The [Jornada capacity follow-up](docs/jornada-capacity.md) uses 10–20 cm grass
clumps, 2 m mesquite crowns and two/three soil layers. It includes literature
cover estimates, hectare-scale populations and measured results from the
authorized [synthetic geometry benchmark](benchmarks/README.md). This prototype
implements root mapping, a synthetic layered reduction and sampled canopy cover;
it does not implement vegetation growth or a water solver.

The separately authorized [30-year water/growth benchmark](docs/daily-water-30year.md)
adds simple daily uptake from finite soil stores and weekly biomass/root-map
updates. It measures compiled CPU and GPU implementations with two/three
layers and the cover-derived Jornada populations. Its recharge, growth and
turnover rules are synthetic performance fixtures rather than a calibrated
ecological model.

The [species-root-profile extension](docs/species-root-profiles.md) adds named,
configurable layer fractions and explicit per-plant profile IDs. Grasses favor
surface layers and shrubs deeper layers in the illustrative defaults; all
profiles retain roots in every layer. Daily uptake and weekly geometry updates
use matching weights on CPU and GPU.

The [coupled transpiration driver](docs/coupled-transpiration.md) now connects
those individuals to the real layered soil state on SYRUP's
`feature/subsurface-hydrology` branch. SYRUP removes actual daily transpiration
and retains evaporation/drainage accounting; LEAF uses that actual uptake for
weekly growth and root-map updates, without its own soil store. NumPy, Numba
and CUDA paths are independently qualified. Storm/MASY scheduling, evolving
soil remapping and calibrated ecology remain separate work.

The [current shared-canopy source lock](dependencies/canopy-view.lock.json) records
the installed LEAF code and qualified upstream identities. The earlier
[Phase 0 lock](dependencies/phase0-vegetation.lock.json),
[growth lock](dependencies/stepwat-growth.lock.json) and
[water-coupling lock](dependencies/syrup-subsurface.lock.json) remain historical
evidence; a Git HEAD alone is insufficient for these uncommitted sources.

The [water-to-biomass survey](docs/water-to-biomass-survey.md) compares ECOTONE,
STEPWAT2, Tietjen, GRASSMIND, Caylor, daily savanna carbon accounting and AquaCrop.
The user selected total aboveground biomass per individual as the growth target.
The [STEP-style growth mode](docs/stepwat-growth.md) is now available through
`CoupledSimulation(..., stepwat_traits=...)`. Actual transpiration supplies
biomass-support resources for species-specific, size-limited growth. Updates run
weekly and at year-end, with an internal seasonal aboveground component removed
at year-end. Named traits, monthly activity and explicit daily temperature
modifiers control growth and demand; roots retain their species layer profiles.
CPU and GPU kernels use bounded array storage. This is a weekly adaptation with
explicit fixture or calibrated traits; no field calibration is supplied. Existing
benchmark defaults still use their synthetic growth rule.

The [validation-data survey](docs/validation-data.md) identifies Jornada seasonal
biomass, harvest allometry, layered soil moisture, species uptake-depth tracer
data and NEON observations. It records verified metadata, access limits and a
proposed independent validation sequence; no field calibration or model run has
been performed.

The [paired growth and water-use survey](docs/paired-growth-water-data.md)
addresses individual validation directly, including public Setaria pot-weight
and dry-harvest records with 176 verified plant-ID matches, and Biosphere 2
grass/mesquite experiments with their measurement limits. The
[Jornada uptake candidates](docs/candidate-uptake-parameters.md) preserve reported
grass/shrub profiles and uncertainty as potential parameters, without changing
runtime defaults.

The [initial STEP-style study parameterization](docs/study-stepwat-parameters.md)
now supplies inactive grass fit candidates and conditional mesquite priors.
It includes genotype holdout checks against actual dry harvests and treats each
two-seedling mesquite pot as one modeled plant. The fitting exposes a mismatch
between rapid seedling growth and the current adapter's rate limit; no runtime
bound or default was changed. Reproduction commands are in
[the analysis guide](analysis/study_parameterization/README.md).

The [biomass-specific water-use comparison](docs/biomass-specific-water-use.md)
adds literature observations in litres per kilogram of **aboveground dry
biomass** per day, with explicit transpiration/ET and biomass-timing distinctions.
It includes checked savanna data and published covered-pot measurements.

## Phase 1: dated native-water / growth session and restart

The [interevent session guide](docs/phase1-interevent.md) documents `leaf.interevent.IntereventSession`: one accepted
boundary of native SYRUP soil water, living-plant biomass, pending requested / delivered water and activity /
temperature integrals, the current fixed root map and an exact fixed-offset calendar cursor. Each explicit interval
runs the real native non-transpiration step, LEAF demand on the existing period geometry, one real native plant
withdrawal and, at growth boundaries (cadence <= 7 days, new years, explicit flush), one `leaf.growth.grow` update with
a root / canopy rebuild, then publishes everything by a single replacement; any failure leaves the previous boundary
intact. Transport gates (wind, externally supplied rain / recession results, closed gate) advance calendar and ages
without transpiration, growth or catch-up, and a year end inside an event is applied at the next open interval.
`leaf.checkpoint` saves and restores the complete boundary (canonical plain `plants.csv` with its companion
`plants.metadata.json`, plus typed arrays, exact roots, hashes, atomic directory publication; layout version 4). A global byte ledger is admitted before every allocating stage. The runnable
`examples/phase1/run_interevent.py` needs an explicit backend and budget and is UNCALIBRATED; no field calibration,
storm / wind physics or full MASY coupling is claimed. The Phase 1 implementation was independently reviewed and
verified in its own task records under `agent_handoffs/tasks/`. The plain-CSV / layout-version-4 change and species-linked
uniform initial ages passed independent verification: 531 CPU tests passed, with 45 CUDA skips. See the
[current review and evidence](agent_handoffs/tasks/plain_plant_csv_20261111/final_review.md).

## Phase 2: disturbance, biomass loss and persistent mortality state

The [Phase 2 guide](docs/phase2-disturbance.md) documents two explicit biological boundary APIs of the same session.
`apply_surface_event(event, current_surface)` runs only AFTER an accepted wind / rain / recession transport interval and
turns one stable-ID `SurfaceEventDiagnostics` plus the ACTUAL current surface into species-specific injury, standing death
or dislodgement of live plants (`leaf.disturbance`; burial, root exposure and abrasion; signed NET change distinguished
from gross erosion / deposition; abrasion DOSE in its exactly declared unit, never converted from flux; the establishment
ground is the fixed reference; increments injure, absolute depths cross lethal / dislodgement thresholds; a bounded
multiplicative composition and a dry-AGB ledger per cause). `apply_annual_decision(year)` is the explicit new-year
scheduler call for the opt-in age / background hazard and the persistent low-growth counter (`leaf.demography`; keyed,
row-order independent randomness from an explicit seed; actual vs required support kept distinct from production yield;
dormancy or maturity alone is never stress; no life-expectancy cutoff). Death and removal are published by a transactional
population rebind (new live rows, geometry and roots before publication; growth grid, deferred season, ages, native water
and RNG state preserved on any failure), and the accepted Phase 2 state (stress counters, annual accumulators, seed,
counters, replay guards) is part of the complete checkpoint. Every species coefficient lives in the species' catalog record
(`mortality`, `disturbance`); `examples/phase2/species.json` holds UNCALIBRATED fixtures, and
`examples/phase2/run_disturbance.py` is a bounded synthetic example. This is an explicit aboveground-only LEAF adaptation,
not an ECOTONE / STEPWAT2 reproduction; there is no wind or water physics, surface remapping, fire or calibration. The Phase 2
implementation and its tests (`tests/test_disturbance.py`, `tests/test_demography.py`, the Phase 2 classes of
`tests/test_interevent.py`) were independently reviewed and verified in their own task records under `agent_handoffs/tasks/`
(`codex_review.md`, `verification.md`); see those records for the accepted evidence.

## Phase 3: simple seed establishment and nearby vegetative daughter plants

The [Phase 3 guide](docs/phase3-reproduction.md) documents `IntereventSession.apply_reproduction(year, current_surface,
max_new_plants=, max_total_records=, max_attempts=)`, the explicit annual reproduction boundary (after growth, post-event
damage and the due annual mortality decision). Species-local opt-in laws (`leaf.recruitment`, catalog `reproduction` section,
law `simple_seed_clonal_v1`; null / `unimplemented` stays inactive) give (1) a STEP-inspired establishment trial per
ecological PATCH (an explicit fixed physical size, independent of the hydraulic cell size) and species with an external-
propagule assumption and a bounded integer number of seedlings, and (2) vegetative reproduction as separately located
DAUGHTER plants (own ID, age 0, uniform in a disk, bounded rejection, never wrapped) funded by an exact transfer of parent
dry AGB and seasonal E. Seedling AGB is an external establishment credit in a closed dry-AGB ledger; no water is added or
withdrawn. Births are published by the same transactional rebind as Phase 2 (generalised to appended births), randomness is
keyed by (seed, accepted-opportunity counter) and independent of row order, and the complete checkpoint carries the
new state (layout version 4 since the plain-CSV change; versions 1, 2 and 3 are refused, not migrated). Phase 2 background mortality now charges a newborn only the
time it was alive. `examples/phase3/species.json` holds UNCALIBRATED FIXTURES and `examples/phase3/run_recruitment.py` is a
bounded synthetic example. Limits: aboveground only, external propagules, canopy overlap allowed, host (not GPU-resident)
placement, no seed bank, no surface remapping, no calibration. The Phase 3 implementation and its tests
(`tests/test_recruitment.py`, the Phase 3 classes of `tests/test_interevent.py`, `tests/test_demography.py::NewbornExposure`)
were independently reviewed and verified in their own task records under `agent_handoffs/tasks/` (`codex_review.md`,
`verification.md`); see those records for the accepted evidence.

## Benchmark commands

Use the existing environment; no installation or new environment is required:

```sh
PYTHONPATH="$PWD:/home/okin/SYRUP/src" NUMBA_CACHE_DIR=/tmp/leaf-stepwat-numba \
  CUDA_VISIBLE_DEVICES='' \
  ../MAPLE/.venv-cupy/bin/python -m unittest discover -s tests -v
../MAPLE/.venv-cupy/bin/python benchmarks/spatial_mapping.py \
  --cases-json agent_handoffs/tasks/jornada_capacity_benchmark_20261030/benchmark_cases.json \
  --output /tmp/leaf-geometry-results.json --backend both --repeats 5
../MAPLE/.venv-cupy/bin/python -m benchmarks.daily_water_growth \
  --cases-json agent_handoffs/tasks/daily_water_30year_20261030/case_sets/june_hectare_3layers_roots3.json \
  --output /tmp/leaf-water-growth.json --backend cpu --years 30
```

Select an idle GPU explicitly with `CUDA_VISIBLE_DEVICES` before GPU execution.
The recorded run used one GTX 1080 Ti. Commands, source hashes, runtime versions
and result artifacts are linked from the follow-up. The existing contributor
guide is preserved; these are the newly established benchmark commands.
For daily uptake on GPU, use the bounded-pool reproduction command in the
30-year report. Default CuPy caching can reserve much more memory than the
live plant/soil arrays.

## Working here

Read [AGENTS.md](AGENTS.md), the
[project brief](agent_handoffs/project_brief.md),
[handoff instructions](agent_handoffs/AGENTS.md),
[orchestration state](agent_handoffs/orchestration_state.md), and
[current assignment](agent_handoffs/current_prompt.md) before acting.
[claude.md](claude.md) is Claude's entry point; `CLAUDE.md` imports it.

Codex assigns bounded tasks and independently reviews and verifies. Claude
implements and corrects, leaving changes uncommitted. Automatic orchestration
continues for requested tasks until changed. An idle record is not an assignment,
and these files do not start a background service.

See the [handoff guide](agent_handoffs/README.md) for templates and evidence, and
[upstream notes](agent_handoffs/upstreams.md) before relying on existing code.
Phase 0 was verified using the existing environments; no packages were installed.
