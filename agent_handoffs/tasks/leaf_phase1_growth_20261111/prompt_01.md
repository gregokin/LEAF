# LEAF Phase 1 — pure species demand and growth kernels

You are Claude, sole implementation writer in THIS isolated candidate. Codex
scopes, executes, independently reviews, integrates and publishes. User authorized
automatic correction loops and relevant sharing with Anthropic; on 2026-11-11
the user approved the recommended requested-water growth normalization and
species cardinal temperature response. This is one subtask, not complete Phase 1.

READ only this prompt and candidate/**. Ordinary Read/Edit/Write tools only.
No execution, Bash, agents, network, settings, installs, Git or external/archive
reads. WRITES exactly: leaf/growth.py, tests/test_growth.py,
examples/phase1/species.json, docs/phase1-growth.md, report.md. Other files are
read-only. Source context is eight selected existing LEAF source/example/doc/test
files. No actual saved terrain/weather/simulation payload is supplied. Do not
guess absent paths. Tests are NOT RUN by you; Codex executes independently.

## Task and strict boundaries

Implement a PURE new versioned aboveground demand/growth module, independent of
the older benchmark driver and native water. No imports of benchmarks, native
SYRUP, or numba/cupy until their explicit backend is used. Other writers own
roots/allocator and native bridge in separate candidates. Do not modify them.
No calendar driver, publication/restart, recruitment, disturbance, stochastic
death or root plasticity here. Do not modify old sources/tests/default fixtures.

Use NumPy reference and compiled Numba / fused CUDA kernels for production plant
loops. Explicit backend, exact CUDA device binding, no fallback or device switch.
All P-sized inputs already typed arrays on the chosen namespace. Daily work may
read back fixed-size diagnostic packets only, not population arrays. S-sized
tables/forcing may transfer explicitly. No plant allometry implementation here:
the caller supplies canopy_area_m2 derived at the geometry boundary by the ONE
existing leaf.canopy/canonical catalog implementation. Never assume B ∝ area.

## Complete species traits, strict opt-in

Compile active growth and demand tables from SpeciesCatalog complete species
records, using its sorted IDs and exact hash. All scientific parameters live in
each species record together with its existing allometry/roots/life/reproduction/
disturbance sections. No separate growth-parameter map or hard-coded species
defaults. Missing/unimplemented/unknown laws fail clearly when this module is
explicitly selected. Phase 0 reading of unimplemented species remains untouched.

Use a NEW growth law identifier, e.g. stepwat_requested_support_v1, distinct from
the old stepwat_weekly_aboveground_v0.1. Required explicit growth traits:
mature_biomass_kg >0; intrinsic_rate_per_year in [0,1] (old bounded weekly Euler
adaptation); biomass_support_kg_per_m3 >0; live_fraction in [0,1];
extra_growth_fraction in [0,1]; monthly_activity exactly 12 fractions in [0,1];
temperature_min_c < temperature_opt_c < temperature_max_c (>= absolute zero).
No silently activated study fits: earlier fitted grass rate295/year is outside
this qualified Euler range. Preserve that limitation explicitly; another solver
and requalification would be separate work, not a raised bound.

Temperature factor is 0 at/below Tmin and at/above Tmax, 1 at Topt, linear on
each side. Monthly activity is selected by actual supplied month (1..12). Missing
or impossible temperature/month is refused, not replaced with defaults. Date
splitting and exact Gregorian year fractions belong to the later scheduler.

Water demand is a declared species law, chosen explicitly with its own required
parameters in water_use. Support TWO small opt-in laws so that existing dry-AGB
specific observations can be used without assuming a universal B/area relation:

- canopy_pet_v1: D = canopy_area_m2 * PET_mm_per_day * 0.001 * dt_days *
  transpiration_coefficient * live_fraction * activity * temperature_factor.
  transpiration_coefficient >=0 is dimensionless and explicit.
- biomass_specific_pet_v1: D = B_kg_dry_AGB * demand_m3_per_kg_day *
  (PET_mm_per_day / reference_pet_mm_per_day) * dt_days * activity * temperature_factor.
  demand_m3_per_kg_day >=0; reference_pet_mm_per_day >0; both explicit. This
  coefficient is per TOTAL dry AGB; do not multiply live_fraction a second time.

Both are DOCUMENTED LEAF demand adaptations, not a claim about STEPWAT daily
physiology. Shared scheduler PET is REQUIRED, finite >=0; do not compute PET or
invent its weather. The later run must select its law/parameters explicitly.
Only live plants demand/grow. Zero PET/activity/T gives D0 and no positive gain.
Do not cap root-overlapping canopies by cell area or silently alter demand.
Example catalog: complete uncalibrated two-species fixtures, shallow grass and
deep shrub with roots in both layers and radius multipliers3–5; all other traits
remain in those same records. Label coefficients fixtures, not literature fits.

## Growth mathematics — requested-water plan, actual satisfaction ONCE

Keep total aboveground DRY mass B kg and seasonal-extra E kg (0<=E<=B), persistent
mass B-E. No whole-plant carbon closure, root biomass pool or WUE production
yield is claimed. All water quantities are VOLUME m3. The existing source
precedent is pinned STEPWAT2 rgroup_Grow/_extra_growth, relative size and biomass
support, with LEAF spatial competition replacing its patch allocator.

The caller accumulates over an actual interevent growth period, holding B/E and
geometry fixed between updates:
- D[p] = SUM requested plant water, U[p] = SUM actual accepted plant water.
- Aint[s] = SUM activity_s * dt_days / days_in_actual_calendar_year.
- Tint[s] = SUM temperature_factor_s * activity_s * dt_days / year_days.
  They exclude ALL wind/rain/recession/closed durations. Validate finite
  0<=Tint<=Aint, Aint<=7/365 for this kernel's weekly-or-shorter envelope;
  callers split longer periods. Leap-year fractions are supplied explicitly.

For a live plant, let required=B*live_fraction*Aint, requested_support=k*D,
actual_support=k*U, k=biomass_support_kg_per_m3. Its units mean annual biomass
SUPPORTED by water (kg·year/m3); k is NOT kg new biomass/m3 and must be described
with its time basis. required is the matching annual-support-equivalent prorated
to this period. No fixed physical percentage of water is assigned to maintenance.

Fplan = min(1, requested_support/required) when required>0, else0.
normal_plan = max(0, .95*r*Tint*Fplan*(B-E)*(1-(B-E)/mature)).
Cap normal_plan to max(mature-B,0); never shrink B merely because B>mature.
meanT=Tint/Aint when Aint>0 else0.
extra_plan = extra_growth_fraction * max(requested_support-required,0) * meanT,
capped to max(mature-B-normal_plan,0).
S=min(1,U/D) for D>0 else0, with only documented honest FP64 overdelivery tolerance.
normal_gain=normal_plan*S, extra_gain=extra_plan*S. Do NOT use actual support in
Fplan then multiply by S again. Cap planned components BEFORE applying S so the
user's 75% delivery supports 75% of otherwise planned growth.
On ordinary updates Enew=E+extra_gain, Bnew=(B-E)+normal_gain+Enew.
At a year-end update lose E+extra_gain (seasonal loss), Enew=0 and
Bnew=(B-E)+normal_gain. Loss is intentional STEP-style seasonal bookkeeping,
not water-stress mortality. Dead/removed plants are unchanged, no invented loss.

Return clear per-plant planned/realized normal and extra gain, Bnew/Enew,
seasonal_loss, satisfaction, required/requested/actual support and
maintenance_deficit=max(required-actual_support,0). Include an aboveground ledger
Bnew == B + gains - seasonal_loss within explicit FP64 bound. Do not clip actual
U. Validate U<=D within a documented small roundoff bound compatible with the
allocator's root-row tolerance1e-9; report any boundary clamping. D0 requires U0.
All relevant intermediate arrays and aggregate packets must be finite; reject
overflow or invalid content without mutating inputs. No hidden balancing fluxes.

## Suggested public surface / memory

Keep it small and clearly documented: compile_species(catalog, backend, max_bytes),
temperature/activity helper, demand(... month,temperature_c,pet_mm_per_day,dt_days...),
grow(...requested_m3,delivered_m3,activity_integral,temperature_integral,year_end...).
Name signatures explicitly and keep them consistent across tests/docs. Caller
supplies typed B/E/species_index/live mask/canopy area, already on the backend;
avoid accidental copies of inputs and do not own another population inventory.

Every allocating public path preflights shapes/dtypes/device and explicit
max_bytes BEFORE copy, transfer or large allocation. Tables are S-sized, work
O(P+S), no graph/grid/history arrays. Provide truthful itemized memory estimates
covering input/compiled-table resident arrays, simultaneous outputs and all
validation/reduction scratch and transfer packets. Accept extra_reserve_bytes for
the later driver. Document host/device scope and interpreter/JIT/context/pool
exclusions. Conservative structural bounds are fine; fixture-specific padding or
unbounded CuPy reducer scratch is not. Prefer fused bounded diagnostic packets.

## Tests/report and acceptance

Write meaningful unittests for the independent hand calculation, 75% growth
relative to full delivery (not 56.25%), dry/zero demand/no gain, temperature
cardinal points/dormancy, both demand laws/units, arbitrary provided canopy area,
normal/extra and seasonal ledger, above-mature behavior, per-species differences,
inactive plants, rejected rates/missing law/fields, invalid typed data/overflow,
budget refusal before copies, unchanged inputs on failure. NumPy/Numba parity,
real CUDA parity when available and mocked current-device guard without touching
another GPU. No tests executing models or depending on private old benchmark
state. A test catalog may be made from the supplied complete fixture.

Save incremental work and finish report.md listing files/limits and all tests
NOT RUN; you cannot execute. Record any tool errors honestly. Leave uncommitted.
Codex will freeze and independently review/test the stable tree, qualify actual
CPU/GPU memory and numerical cases, and dispatch bounded corrections if needed.
