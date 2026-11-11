# Phase 1 integration review draft

This is a review specification, not implemented behavior or phase acceptance.
Part A is the active bounded writer scope. Growth normalization and disturbance
response choices remain pending user answers; no dependent implementation is
authorized by this draft alone.

## Native water and transaction boundary

The new dated driver should compose the existing native SYRUP column and
transpiration operator with the new stateless LEAF allocator. There is one native
water inventory. Any availability view must be tied to the exact native state,
operator, active mask and residual parameters that SYRUP will use for acceptance.
The adapter must not expose an independently configurable LEAF withdrawal floor.

Within one interevent segment:

1. Validate forcing, cursor, identities, backend and full memory admission.
2. Stage native non-transpiration hydrology through its existing pure API, with
   separate evaporation accounting. Obtain the native withdrawable-water view
   after that stage, retaining root weights in inactive/unavailable resources.
3. Compute plant demands and the proposed volume per native resource. Convert
   m3 to native layer-depth units once, with the actual native cell area.
4. Ask native SYRUP to accept and debit the staged withdrawal exactly once.
5. Gather actual plant delivery from native accepted volumes, validate water
   closure, and stage biological accumulators and any due growth/root update.
6. Publish both native state and biological state/cursor only after all staging
   succeeds. A rejected transaction leaves the prior accepted boundary intact.

An interval receipt/cursor must reject replay. Pure staging is not publication;
re-evaluating a rejected transaction must neither double-debit water nor advance
age, accumulated stress or the calendar. Explicit transport advance may update
age/time, but may not accrue transpiration or retrospectively run biology.

## Calendar and shared forcing

Use actual fixed-offset Gregorian dates and exact elapsed intervals. MASY keeps
submicrosecond rational remainders alongside datetimes; a bridge must preserve
them or explicitly refuse unsupported precision before mutation, never round
them away. Split at midnight, growth boundaries and year boundaries. Partial
days use their actual durations. Date/temperature come from the same scheduler
weather records that drive SYRUP; do not create a second weather stream or PET
implementation. Actual temperature response and demand traits require explicit
species declarations and a documented scientific choice.

## Plant state, roots and common canopy

Preserve stable IDs, individual dry aboveground biomass, age and establishment
ground elevation. The common `canopy_view` remains the sole geometric contract
for both MAPLE and SYRUP; new growth may not revert to the older benchmark's
square-root biomass/diameter restriction. Reuse the catalog's actual allometry.

A weekly geometry update is a transaction, not root plasticity. Root weights
stay fixed during the uptake interval; dry cells do not redirect demand. Distinct
species layer fractions and individual overrides must survive growth and restart.
Root-builder admission includes all simultaneously live old/new graphs and
temporary geometry, layer expansion, sort/validation scratch and native state.

Daily GPU uptake must remain on the native backend. Any population transfer for
weekly allometry, snapshots or exports must be explicit and measured; do not
claim full GPU residency if the canonical host plant table forces a round trip.
General expanded CSR has a larger footprint than the older factorized benchmark;
its older timings and byte counts do not automatically qualify the new engine.

## Growth and diagnostics

For positive total demand, the user requires growth satisfaction to be
`actual_transpiration / requested_transpiration`, with zero-demand behavior
explicit. Do not apply that deficit twice through both satisfaction and a second
actual-water stress multiplier. The support coefficient remains distinct from
biomass-production yield; store requested/actual support and maintenance deficit
diagnostics needed for Phase 2 rather than inventing a fixed maintenance fraction.

Preserve older adapters, fixtures and study candidates as explicitly historical
or opt-in. Candidate fits requiring intrinsic rates above the old Euler validator
are not silently runtime-compatible. A changed integration method needs its own
qualification, not a raised bound to pass a test.

## Restart and acceptance

The human-readable plant CSV is the shared initial/snapshot/final population
format. It alone is not a coupled restart. A qualified restart also binds native
water, exact cursor and next growth boundary, pending demand/delivery/support and
temperature/activity integrals, seasonal mass, individual root overrides, source
and species hashes, selected backend and last accepted receipt. No historical
trajectory arrays are required.

Acceptance needs actual native CPU tests, actual CUDA parity where available,
water closure with area conversion, transport gating, leap/partial intervals,
duplicate-interval rejection, failure/retry preservation and split-run restart
equivalence. Full wind/rain model orchestration remains a separately qualified
MASY adapter; do not describe a narrow interevent test as a complete coupled run.
