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
accepts event diagnostics; MAPLE adoption and ecological injury laws remain future work.

## Phase 0: initialize and exchange vegetation

The [Phase 0 guide](docs/phase0-vegetation.md) documents the implemented species
JSON catalog and canonical plant CSV. The same CSV format serves initial input,
user-cadenced snapshots and final output. It records stable IDs, species, X/Y,
ground elevation at establishment, total aboveground **dry** biomass in kg,
seasonal biomass, age in days and status. Initial `SURFACE` anchors resolve once
against the supplied topography and are then saved as numeric elevations.

Generate multiple species from target cover and mean/standard-deviation canopy
diameter, with seeded random placement and optional canopy exclusion. Cover is
per-species union ground cover on a finite domain, measured at a declared sampling
resolution and tolerance. Size draws are positive lognormal; exclusion places
larger canopies first and refuses infeasible requests within explicit resource bounds.

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
