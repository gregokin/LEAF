# Current Verification

Independent evidence is retained under
[allocator task](tasks/leaf_phase1_20261110/) and
[native-bridge task](tasks/leaf_phase1_native_water_20261110/).

- Initial completed candidate:166 CPU tests,158 passed / 8 CUDA-related skips. Real GPU
  allocator check had one invalid-test failure; no acceptance was granted.
- Saved allocator correction: seven exact cases each on NumPy, Numba and CUDA;
  offset/nonfinite/pre-copy/device/native-feedback guards pass. Actual SYRUP
  numerical fixtures pass. Four CPU/five CUDA allocation workloads find no
  array-scale underestimate for the measured allocator paths; zero CUDA hook leaks.
- The targeted root suite on that partial has11 tests, 5 passed / 6 errors because its
  companion API update was not written before the provider limit.
- Review-only native bridge composition:33 CPU / 9 CUDA cases pass, plus budget,
  shape, one-native-debit, shortfall and failure-preservation guards. Imported
  native source hashes match the pin; native time and input state stay unchanged.
- Combined CUDA memory profile: plant-heavy live requested peak19,209,617 bytes
  versus native-side ledger19,200,796 bytes, an 8,821 byte underestimate. Grid-heavy
  case fits. Zero unknown frees/retained tracked bytes; allocator rounding is
  separate. CPU trace includes bookkeeping, so its small flagged gap is
  inconclusive for algorithm arrays; its old exit 0 was not a pass verdict.

Both Claude launches ended with explicit usage errors and were reaped; reset
unknown. Primary streams, scope audits, partial manifests, reviews and profiler
source/dispositions are preserved. No full corrected-tree or Phase 1 acceptance,
new growth-law implementation, 30-year replay or transport-scenario claim.

Live source/test/example/parameter/lock hashes and root AGENTS.md remain unchanged.
No new environment/install or upstream implementation edits; no active writer or
scheduled retry. Continue from [resume.md](tasks/leaf_phase1_20261110/resume.md).
