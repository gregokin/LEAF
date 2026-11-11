# Independent review of interrupted allocator correction

Verdict: **preserve partial work; correct before acceptance**.
`partial_02.manifest.json` freezes the saved allocator rewrite. Claude's exact
session is reaped and stopped by the explicit Fable usage limit; reset unknown.
The user preference for automatic orchestration remains enabled.

## Independently verified saved core

- Seven exact rational cases pass on each of NumPy, Numba and real CUDA:
  user75% example, local proportional competition, actual shortfall feedback,
  nonseparable roots, unavailable resources, zero/empty roots and populations.
- Offset-overflow construction is now refused without running an invalid map.
  Scatter/aggregate overflow, pre-copy zero-budget conversion and untyped input
  guards pass. Native accepted ulp-level differences are retained unchanged.
- Real CUDA additionally passes the explicit two-layer example, device-native
  construction and current-device refusal tests without allocating on another GPU.
- Actual native SYRUP numerical fixtures pass and preserve exact accepted feedback,
  the input state and native clock. Area conversion is applied once each way.
- Four CPU allocation workloads and five CUDA allocation workloads find no
  array-scale underestimate for the allocator's measured paths. CUDA malloc/free
  hooks count requested bytes separately from pool rounding; every case ends with
  zero tracked bytes and zero unknown frees. This is bounded qualification of
  selected cases, not every grid/population or a 30-year growth run.

Evidence prefixes: `partial_02_uptake_oracles`, `partial_02_uptake_cuda`,
`native_contract_02`, `memory_profile_02_numba`, `memory_profile_02_cuda`.
All launch/exit records and separate raw streams are retained.

## Required continuation

Only leaf/uptake.py was written in revision 2. Root builder still calls the removed
private `_take_ownership` argument: its targeted suite has11 tests, 5 passed / 6 errors.
Integrate the intended trusted ownership helper and finish aggregate root-builder
admission, including shared geometry/overrides/horizontal and expanded maps.
Update new regression tests/diagnostics, forcing precision documentation and
date parsing, canonical multi-file metadata comparison, CUDA excess-input test,
and the implementer's report. Do not alter old fixtures/drivers/defaults.

The existing report.md is stale revision 1; no completed revision 2 report exists.
No saved module has been installed in the live package, committed or pushed.
Phase 1 still needs the chosen growth/temperature laws, co-located active species
traits, dated scheduler integration, transactional publication and restart
qualification. Pending science is listed in pending_science.md.
