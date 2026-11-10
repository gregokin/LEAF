# Bounded implementation: STEP-style individual aboveground growth

Task ID: stepwat_growth_20261031. Revision 1. Writer: Claude; reviewer: Codex.
Work only in the isolated candidate directory containing this task's original
generic Python source copies. Baseline is ../baseline.json and ../baseline_sources/.
No Git metadata is available; do not initialize Git.

## Governing instructions and scope

Automatic dispatch/implement/review/verify/correct is authorized for this bounded
task. Exactly one implementation writer. Codex owns orchestration and acceptance;
your final message reports changes and limitations, never acceptance. Preserve
existing code/defaults. Read only candidate files and this supplied prompt. No
shell, network, other agents, external source, private scientific briefs, inputs,
upstream repositories, credentials, installs, commits, pushes or model runs.
Available tools: Read, Write, Edit only. Working directory: candidate/.
Allowed writes (exactly): leaf/stepwat.py, leaf/__init__.py, leaf/coupling.py,
benchmarks/daily_water_growth.py, tests/test_stepwat.py, docs/stepwat-growth.md.
Other files including legacy tests and spatial_mapping.py are read-only.
Report in your final message; Codex captures it durably. No execution claims;
Codex runs tests independently. On tool/auth/network/rate errors stop and retain
partial changes. Baselines/prompts, control files and historical evidence are immutable.

## Goal and scientific scope

Implement an opt-in STEP-style growth mode in the existing native-water coupling.
Total aboveground dry biomass per individual is the sole observed biomass target.
Keep a seasonal-extra component internally to preserve the source model's
distinction between persistent size growth and season-end biomass loss; no root,
leaf/wood or carbon pools. No new physical water store. Root layer fractions,
spatial individuals, native residual-limited transpiration, masks and area
conversion stay authoritative. Legacy benchmark and coupling defaults must retain
their current numerical behavior exactly when the new mode is absent.

The source method at public STEPWAT2 revision
003cb9b4adbf13d64a39d6d62fd95280adac27fc (ST_resgroups.c rgroup_Grow,
_extra_growth; ST_species.c Species_GetBiomass; sxw_resource.c) is:
individual relative size R = biomass / species mature biomass;
normal annual increment dR = intrinsic_rate * gmod * R * (1-R), with
gmod = 0.95 * temperature_modifier / max(PR, 1), PR = required/available
biomass-support resources; surplus resources yield seasonal extra biomass.
This task is a documented weekly adaptation, not a complete port or annual
trajectory reproduction. Retain spatial proportional root competition; STEP's
patch-level largest-first resource allocator is not introduced here. Never
create STEP's added-transpiration resource correction or a nonzero dry fallback.

## Required API and traits

Provide a clear opt-in keyword (e.g. stepwat_traits) accepted by CoupledSimulation
and vegetation-only Simulation. Use profile names and existing root_profile_ids
as authoritative species/PFT trait identifiers. Require one trait row per used
profile, explicit finite values, no inferred species from size in the new mode.
Reject this mode with synthetic standalone water (vegetation_only=False).
Required traits, with documented units:

- mature_biomass_kg: positive total aboveground mature mass.
- intrinsic_rate_per_year: finite in [0,1] for the bounded Euler adaptation.
- biomass_support_kg_per_m3: positive annual biomass-support coefficient (not WUE
  or new-biomass production yield).
- live_fraction: finite in [0,1].
- canopy_biomass_density_kg_m2: positive allometric scale.
- potential_transpiration_m_per_day: nonnegative canopy-area transpiration demand
  depth, explicitly distinct from native potential SOIL evaporation or reference ET.
- extra_growth_fraction: finite in [0,1], converts surplus resource to seasonal mass.
- monthly_activity: 12 explicit finite fractions in [0,1], 365-day no-leap calendar.

No scientifically calibrated defaults. Example values must be labelled fixtures.
Allow scalar or per-profile HOST temperature_factor in [0,1] on step_day(); require
it explicitly in STEP mode, validate BEFORE native state/counters change. It is a
caller-supplied modifier, not a new asserted temperature-response law. No frost,
recruitment, mortality, injury, plastic roots or detailed surface remapping.

## Numerical contract

Initial B = pi*canopy_radius^2 * species canopy_biomass_density_kg_m2;
initial seasonal mass E=0. Reject B > mature mass before accepting any water day.
At a growth-period start, select activity from that day's month. Hold it and the
physical demand cache constant until the next geometry/growth update (weekly or
year-end). Demand Dp = canopy_area * potential_transpiration_m_per_day *
live_fraction * activity. Reuse existing root scatter/gather and actual daily T.
Temperature only modifies growth; the explicit activity controls uptake dormancy.

At each accepted physical day accumulate actual per-plant uptake in existing
uptake_week, and per-profile temperature_factor*activity/365 into a small
temperature integral. Failed physical days must not accumulate anything. No
P-sized daily temperature array or plant/cell history. No demand changes until a
successful growth commit. Temperature state is retained on a refused growth rebuild.

Update growth at day%7==0 OR day%365==0, with dt_days equal to accepted days since
the last SUCCESSFUL growth commit. This handles partial periods at year-end;
365 days is not 52 complete weeks. Let

  persistent = B-E
  R = persistent / mature_biomass
  required = B * live_fraction * activity * dt_days/365
  available = biomass_support_kg_per_m3 * actual_uptake_period_m3
  water_factor = min(1, available/required) if required>0 else 0
  normal_gain = intrinsic_rate_per_year * 0.95 * temperature_integral[profile]
                * water_factor * persistent * (1-R)

Bound normal_gain by max(mature_biomass-B,0), recording actual gain only. No
turnover from the old fixture in this mode. Seasonal extra gain is

  extra_gain = min(extra_growth_fraction * max(available-required,0),
                   max(mature_biomass-B-normal_gain,0))

Require an active growth period for extra gain as well: multiply the potential
extra gain by mean temperature_factor over the active period (the temperature
integral divided by activity*dt_days/365; zero when activity==0). Thus no gains
without actual water, during dormancy or at temperature_factor==0. This explicit
weekly climate adaptation is not an unchanged STEPWAT routine.
At year-end remove ALL seasonal mass including this period's extra_gain and record
it as aboveground loss; otherwise carry E+extra_gain. New total B is persistent+
normal_gain plus remaining seasonal. Track actual normal+extra gain and seasonal
loss in existing gain_cum/loss_cum; B = B0 + gains - losses must close. Never silently
discard individuals or unrecorded mass. No whole-plant carbon closure is asserted.

New canopy = sqrt(B_new/(pi*species density)); roots = root_multiplier*canopy.
New Dp uses the NEXT period-start month's activity. Root vertical profiles stay
unchanged. New biomass, seasonal state, geometry, demand, graph, and reset of
period uptake/temperature/clocks must commit together ONLY after existing
old+new graph admissions/invariants pass. Refusal retains accepted native water
day and accumulated uptake/temperature and latches, as existing coupling does.
Explicit retry must produce the same result once, without a second water day.

## Architecture, memory, and compatibility

Add the growth code under leaf/stepwat.py, not another large implementation in
coupling.py. Reuse the existing candidate/build/admit/swap transaction. Introduce
small, explicit hooks if needed; no unrelated refactoring. CPU per-plant growth
must use a compiled Numba kernel; GPU an equivalent fused CuPy RawKernel or
ElementwiseKernel (no Python per-plant loops). Trait tables may stay S-sized;
only small climate forcing transfers are allowed. Do not copy plant vectors to
the host to grow them. Lazy imports must still work without the native package.

Count ALL added resident/candidate/trait/host-mirror/scratch arrays in initial
scalar admission, every vegetation ledger, combined native admission, retry and
live/array memory reports. Preflight before new P-sized allocations. Do not hide
added bytes in caller reserve that later set_external_reserve_bytes overwrites.
No cap increases. Keep legacy ledger outputs unchanged when new mode is absent.

Counters/audits must distinguish weekly boundaries from actual growth updates
(year-end adds a partial update). In the new mode maps_built=growth_updates+1;
pending days/uptake reflect the last successful commit, including year-end/retry.
Expose model identity, biomass units, trait definitions, persistent/seasonal totals,
period clock, resource-support interpretation and caveats in plant/coupled summaries.
Don't use legacy fixture WUE/turnover labels as the new model's parameter report.

## Acceptance tests to write (Codex executes)

Use unittest and existing fixture helpers. Meaningful analytical tests for the
above equations, wet/limited/zero uptake, mature-size bounds, grass/shrub different
traits and all-layer profiles, temperature/dormancy, explicit invalid trait/forcing
refusal before mutation, aboveground gain/loss closure, year-end partial periods,
split-run equivalence and weekly/year-end refusal then retry. Test at least two
cell areas and inactive masks with actual native water and separate E/T closure.
Memory guards must reject added growth state at init and weekly old+new peaks;
live arrays/reports must include new state. Test CPU and skip-labelled CUDA parity
using existing gpu_available gates; legacy tests remain read-only and must pass.
No full model runs; no environment or package creation. Tests in tests/test_stepwat.py.

## Documentation and report

Write docs/stepwat-growth.md with public pinned source URLs, equations, units,
how annual support coefficients and rates were adapted, daily/weekly/year-end
order, 365-day calendar, total aboveground output and seasonal subcomponent,
explicit fixture-only trait example and native API usage, CPU/GPU memory design,
spatial proportional competition versus STEP largest-first, no added-resource
water, deferred features and calibration limits. Report changed files and what
you did not run. Leave changes uncommitted.
