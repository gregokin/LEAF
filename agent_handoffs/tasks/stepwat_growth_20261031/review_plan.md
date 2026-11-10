# Independent acceptance plan

- Review the completed candidate diff and every source read/write against the
  six-file implementation scope. Preserve spatial_mapping.py and legacy tests.
- Compare native water and candidate aboveground mass to analytical oracles for
  wet, limited, zero uptake, temperature, dormancy, mature size and species traits.
- Exercise day364→365→371 with partial periods, seasonal gain/loss closure,
  month changes, split calls and failed growth rebuild followed by explicit retry.
- Independently refuse invalid forcing before water changes; reject extra state
  at init and fully reserved old+new graph peaks before allocation.
- Verify actual native water withdrawal equals individual uptake, with inactive
  cells and two cell areas, and separate evaporation/transpiration/drainage budgets.
- Run full existing and new CPU tests, then real CUDA correctness tests on the
  explicitly selected allocated GPU0; no long scenarios or performance claims.
- Inspect all live/candidate/trait/scratch memory, CPU compiled kernels and fused
  GPU equations; no host plant-vector growth path or plant×cell history.
- Install reviewed files only, rerun final appropriate checks on installed tree,
  write a new growth source lock rather than silently rebinding historical locks,
  and record independent acceptance with source and upstream-preservation checks.

The earlier MAPLE manifest differs in closed_windowed_commit.py; current hashes
are recorded separately. Native SYRUP modules remain hash-identical. Preserve
upstream changes and qualify only the actual current pair used in the checks.
