# LEAF development phases — user authorized 2026-11-10

The user said proceed after specifying the following phases and authorized
automatic Codex–Claude orchestration, relevant source sharing, and commit/push
after each independently verified phase, then automatic continuation.

1. **Growth and water use:** species-specific STEP-consistent size-dependent
   aboveground dry biomass growth, scheduler date/temperature demand, unified
   per-species definitions, daily actual withdrawals and coarser growth. LEAF
   computes proportional cell/layer competition from normalized roots; SYRUP
   validates/accounts accepted water once. S = total delivered / total requested
   applies to planned growth, no minimum-layer stress or compensation. User
   example D100, half roots in two layers, supply25/200 gives delivery25/50,
   S0.75, remaining0/150. Maintenance/support accounting follows inspected STEP
   science; support coefficient is not a biomass production yield.
2. **Biomass loss and death:** species-specific burial/erosion/abrasion/root
   exposure/dislodgement and persistent low-growth/maintenance stress; use ground
   elevation at establishment. SYRUP water transports erode/deposit; MAPLE wind
   also abrades. Scientific response laws/parameters remain to be resolved.
3. **Reproduction:** simple STEP/STEPWAT-style seed and vegetative recruitment,
   species-specific mechanisms and co-located traits; exact laws not yet chosen.

Preserve one plant state and common canopy_view for both consumers. Wind,
rain/recession and interevent biology never overlap; no retroactive growth/T
for transport durations. Compile CPU/GPU kernels, bound memory by actual work,
preserve individuals/conservation, existing dirty work and other jobs. No new
environment/install or broad upstream modifications as routine setup.

Ask only material unresolved science questions; continue independent authorized
work while answers are pending. No daemon/background promise is implied. Phase 1
part A can implement the uptake/root/calendar/species plumbing while the planned-
growth versus support-demand normalization question is pending asynchronously.
