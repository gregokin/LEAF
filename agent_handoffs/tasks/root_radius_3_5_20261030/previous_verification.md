# Verification

Task: `jornada_capacity_benchmark_20261030`.

- [Final acceptance](tasks/jornada_capacity_benchmark_20261030/final_acceptance.json)
  checks source hashes, invariants, arithmetic, recorded execution and links.
- [20-test receipt](tasks/jornada_capacity_benchmark_20261030/tests_02.exit.json)
  and [raw output](tasks/jornada_capacity_benchmark_20261030/tests_02.stderr.log):
  exit 0, all tests passed including actual GPU parity, no skips.
- [Benchmark launch](tasks/jornada_capacity_benchmark_20261030/benchmark_01.launch.json)
  and [exit](tasks/jornada_capacity_benchmark_20261030/benchmark_01.exit.json):
  all 11 cases passed, both backends available, no rejection or partial case.
  [Raw results](../benchmarks/results/jornada_20261030/profiled_results.json)
  retain repeats, ledgers, demand audits, parity and memory observations.
- The source SHA256 in unit tests and benchmark results matches the completed
  tree. CuPy hook accounting ends with zero live bytes and zero unknown frees.
  Peak means observed rounded pool allocations; context/other allocators excluded.
- [Runtime identity](tasks/jornada_capacity_benchmark_20261030/runtime_identity.json)
  records the existing interpreter, imported origins and version pins. No install.
- [Capacity inputs](tasks/jornada_capacity_benchmark_20261030/cover_sources.json),
  [216 arithmetic cases](tasks/jornada_capacity_benchmark_20261030/capacity_estimates.json)
  and [independent review](tasks/jornada_capacity_benchmark_20261030/independent_capacity_review.md)
  distinguish cover definitions, inferred counts and unmeasured root extents.
- [Upstream preservation](tasks/jornada_capacity_benchmark_20261030/upstream_preservation.json):
  all 513 manifested MAPLE/SYRUP/MASY paths and HEADs unchanged. SYRUP/MASY Git
  statuses unchanged. [MAPLE concurrent status](tasks/jornada_capacity_benchmark_20261030/maple_concurrent_status_change.json)
  gained one untracked image; it was neither opened nor modified by this task.
- Root AGENTS.md matches the baseline. Closed task files were not modified.

Initial draft tests passed but review identified defects; they were corrected
before final measurement. The CPU-only sandbox test explicitly skipped GPU with
an unavailable-driver error; real GPU tests and preflight succeeded through the
approved execution boundary. This was not a rate limit. No final GPU failure,
benchmark timeout or unresolved review blocker remains.

No ET/drainage, competition, biomass, demography, injury, complete native export
or transport run was performed. Those are outside this prototype's qualification.
