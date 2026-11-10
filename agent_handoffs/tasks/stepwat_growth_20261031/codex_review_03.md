# Codex review — accept

Accept stepwat_growth_20261031 revision3 as a bounded, source-pinned implementation
of STEP-style individual aboveground growth coupled to actual native SYRUP water.
Claude authored the numerical code; Codex independently reviewed the diff, read/
write scope, annual-to-weekly resource accounting, partial year-end and retry
semantics, memory coverage and installed-source tests.

Final CPU:83passed/7CUDA skips. CUDA-enabled:89passed/1intentional unavailability
skip. Independent scalar, native E/T/area/calendar/retry and bounded-input oracles
pass on both backends. Completed candidate Python files match installed sources;
legacy tests and geometry are unchanged. Root AGENTS and historical evidence
preserved; current source lock records all dirty runtime identities.

The source's largest-first patch allocator and added-transpiration adjustment
are excluded explicitly; LEAF spatial proportional competition and actual native
withdrawal remain authoritative. Total aboveground biomass is the observable,
with an internal seasonal component. This is a weekly adaptation with uncalibrated
explicit traits, not exact STEPWAT2 trajectories or a full demographic model.
No remaining required work in this bounded assignment. See verification.md and
acceptance.json for evidence and scope.
