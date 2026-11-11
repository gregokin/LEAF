# Current Verification

LEAF Phase 1 part A, revision 1 — incomplete qualification; corrections assigned.
All receipts under [task archive](tasks/leaf_phase1_20261110/).

- `snapshot_01`: 32 files, exact allowed writes only; final writer exit 0/reaped.
  Nine intermediate missing-file Read errors retained in `scope_01.json`.
- `candidate_01_cpu`: 166 tests, 158 passed / 8 CUDA-related skips.
- `candidate_01_uptake_gpu`: actual GPU 1; 13 allocator tests, one invalid-test
  failure. Compilation and preceding parity checks completed. Not GPU acceptance.
- `scalar_oracle` and `uptake_interim_probe`: seven exact cases pass on NumPy and
  Numba; probes separately reproduce nonfinite closure and admission defects.
- `offset_probe_00.json`: malformed offsets accepted at construction; numerical
  kernels were deliberately not called on that invalid map.
- `qualification_inventory_00`: current native module origins and five upstream
  source hashes match the task baseline. No upstream implementation edits.

GPU guard mistake/disposition: `gpu_selection_01_disposition.md`; future launches
use the fail-closed `run_gpu_check.py`. Root AGENTS.md, old accepted locks and live
runtime remain unchanged. No new environment/install or transport scenario.
