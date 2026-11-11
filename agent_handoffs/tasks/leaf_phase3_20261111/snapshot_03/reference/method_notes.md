# Independent scientific review — Phase 1–3 preparation

Status: source inspection only; not a selected or calibrated production law.
User choices on growth normalization and disturbance responses remain pending.
No ECOTONE wind physics is adopted or evaluated here.

## Pinned STEPWAT2 evidence

Revision: `003cb9b4adbf13d64a39d6d62fd95280adac27fc`. Public primary
sources and hashes are archived under `reference_sources/`. These are selected
source files, not a runnable checkout; no STEPWAT2 build or run is claimed.

- `ST_resgroups.c`, `rgroup_PartResources` and `rgroup_Grow`: resource demand
  represents biomass support; ordinary relative-size growth has the logistic
  factor `intrinsic_rate * R * (1-R)` multiplied by `0.95 * temperature_modifier /
  max(required/available, 1)`. Seasonal surplus growth is separately tracked.
  This is not a physical split assigning a fixed fraction of transpiration to
  new biomass versus maintenance, and the support coefficient is not a WUE yield.
- Its group-level, largest-first resource allocator and nonzero dry-resource
  fallback must not be imported into LEAF. The user's local proportional-demand
  allocation and zero-delivery behavior take precedence.
- `ST_environs.c`, `_set_temp_reduction`: the initially calculated quadratic
  temperature response is overwritten by a hard-coded cool-/warm-season switch
  at mean annual temperature 9.5 C. It is incorrect to describe that revision's
  active response as the earlier quadratic. Applying this annual-temperature
  shortcut to daily scheduler temperature would be a new scientific choice.
- `ST_mortality.c`, `_slow_growth`: first-year individuals are exempt; a
  species/group-dependent slow-growth threshold accumulates slow years. Once the
  species limit is reached, the source uses a 0.368 mortality probability. A
  successful year decrements the counter by one, rather than resetting it.
  These exact constants are source findings, not calibrated LEAF defaults.
- `ST_species.c`, `Species_NumEstablish`: its simple perennial establishment
  path tests a probability and then draws an integer from 1 through a configured
  maximum. The dispersal-enabled path additionally requires seed presence.
  Explicit seed-bank accounting is not mandatory in its simplest path.
- `ST_resgroups.c`: clonal recovery can add seedling-size vegetative units to
  the same surviving individual. Equating this with a separately located new
  individual would be a documented adaptation requiring an explicit choice.
- `ST_seedDispersal.c`: reproductive size and plant height affect its spatial
  seed-dispersal option. A full implementation is beyond the simplest initial
  recruitment path; do not conflate recruitment probability per ecological patch
  with probability per arbitrarily fine hydraulic cell.

Primary publication: [Palmquist et al. (2018)](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.2394).
Source: [pinned ST_resgroups.c](https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/ST_resgroups.c).

## Supplied ECOTONE ecological precedent

Read the local repository instructions before inspecting the active `Src/`
files, not `Src/copy`. No compiler or ecological baseline has been qualified.

- `adjust_gro.f`: for the relevant clonal-type branch and age above one,
  sustained low growth is measured relative to a potential increment. Active
  pools include belowground biomass; LEAF's aboveground-only interpretation
  therefore needs to be identified as an adaptation.
- `mark_kill.f`: the executable slow-growth probability differs from a nearby
  historical comment. Do not copy the comment's divisor or silently transplant
  its hard-coded mortality thresholds as species-specific LEAF parameters.
- `pamort.f`: the active final age response overwrites earlier calculations:
  `1 / (max_age - min(age,max_age) + 1)**1.2`. This is a precedent for age-dependent
  mortality, not evidence that LEAF should hard-code this exact law.

## Integration constraints established by current source inspection

SYRUP's native transpiration operator owns residual-limited soil-water removal
and returns accepted withdrawals. It returns a new native state without publishing
it. LEAF must derive plant delivery from that accepted feedback, not its proposal.
Cell-area conversion between m3 and native layer depth occurs exactly once.

MASY's calendar uses exact rational elapsed seconds, fixed UTC offsets and actual
Gregorian dates. Its shared weather rows supply actual date and mean temperature;
there must be no independent January-1 origin or weather series in the new driver.
Wind, rain/recession and interevent biology are mutually exclusive. Advancing the
calendar through a transport event must not later produce catch-up transpiration.

Any coupled transaction must publish the native inventory and biological
accumulators together only after withdrawal, gather and validation succeed.
Repeated interval submission must not debit water twice. Root rebuild admission
must include simultaneously live old and candidate graphs.
