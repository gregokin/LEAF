# LEAF Phase 1 part A — finish interrupted corrections, revision 3

Resume exact session fd82b85e-8a07-4982-b621-b132f845eab7. Preserve the saved work;
do not restart the task. The prior revision saved only leaf/uptake.py before an
explicit Fable usage limit. The user has now approved another available Claude
model, so Codex selected the documented sonnet alias for this continuation.

You are the sole implementation writer in this candidate. Codex controls task
records, independent execution/review, installation and publication. Automatic
correction loops and relevant source sharing with Anthropic are authorized.
This is a Phase 1 infrastructure subtask, not the entire phase.

## Scope and tools

READ only this prompt and this candidate/**. Ordinary Read/Edit/Write only.
No execution, Bash, agents, network, settings, installs, Git or external reads.
WRITES exactly: leaf/uptake.py, leaf/roots.py, leaf/forcing.py, leaf/traits.py,
leaf/__init__.py, tests/test_uptake.py, tests/test_roots.py, tests/test_forcing.py,
docs/phase1-water-contract.md and report.md. All other files are read-only.
The candidate contains tests/test_phase0.py and examples/phase0/species.json;
there is no README.md or separate test_traits.py/test_canopy.py/test_state.py.
Do not guess missing paths again. No growth/native water physics here.

## Saved allocator is already independently verified

Do not rewrite the corrected allocator without an identified defect. Its saved
uptake.py passes seven exact rational cases on NumPy, Numba and real CUDA,
including the user's 75% example, proportional local competition and actual
accepted shortfall feedback. Offset overflow, computed nonfinite totals,
pre-copy budget rejection, typed input validation, exact accepted roundoff and
CUDA affinity guards pass. Four CPU/five CUDA allocation cases found no
array-scale undercount; live CUDA requested-byte counts end at zero. These are
Codex's checks, not checks you ran. Full companion-tree acceptance is still open.

Core API remains RootMap/propose/accept. Public from_arrays accepts typed arrays
only. The removed private _take_ownership argument is replaced by INTERNAL
adopt_owned_arrays(ids, offsets, resources, weights, *, n_layers, n_cells,
backend, max_edges, max_bytes, provenance=None, resident_bytes=0). Read its actual
contract. The builder may supply fresh owned contiguous host arrays only; public
caller aliases must never become mutable map state.

## Finish these bounded corrections

1. roots.py still passes _take_ownership=True and six targeted tests error.
   Integrate the trusted ownership helper. Count ALL simultaneously live root
   builder geometry (including names/object arrays), layer tables, override and
   index scratch, horizontal and expanded maps, old/new maps and transfer buffers.
   Pre-admit bounded NumPy counting scratch before any per-plant bounding-grid or
   edge arrays. Compiled counting already avoids per-edge arrays. Do not change
   root physics or silently drop plants/raise limits. Reuse shared canopy allometry.
2. traits.from_files metadata comparison uses Python equality. Notes 1 versus
   1.0, or True versus 1, can compare equal but serialize differently; reversed
   paths then change merged hashes. Compare canonical serialization or omit
   differing metadata deterministically. Preserve exact single-file hashes,
   complete species records and duplicate refusal. Add regression tests.
3. forcing.parse_date must use isoformat instead of strftime('%Y') so valid
   zero-padded years below 1000 work. Preserve fractional fixed UTC offsets
   exactly or explicitly refuse unsupported precision; never truncate metadata.
   Document that this part supports microsecond-exact datetimes, not MASY's
   rational submicrosecond instants; a future adapter must preserve or refuse them.
4. Update the NEW tests for actual allocator signatures, diagnostic names and
   typed arrays. The CUDA excess test must exceed the PROPOSAL, not multiply
   already-reduced accepted values by 1.01. Keep meaningful regressions for offset
   overflow, nonfinite scatter/totals, pre-copy admission, CUDA device guards,
   honest accepted roundoff, canonical metadata and forcing edge cases. Preserve
   old tests/fixtures/benchmarks unchanged.
5. Update docs and report.md to the actual APIs, memory/precision/roundoff scope
   and remaining limits. The current report is stale revision 1. Record all checks
   NOT RUN (you cannot execute). Revision 1 had nine ordinary missing-file Read
   errors, revision 2 had the explicit Fable limit; do not claim no tool errors.

Save the companion corrections incrementally and finish the report. Do not spend
this continuation re-proving or rewriting the already-correct allocator. Fixed
normalized roots, local demand-proportional sharing, no compensatory uptake,
actual delivered/requested feedback, compiled hot loops, exact device identity
and bounded sparse O(P+E+LN) memory remain mandatory. No daily full population
readback or implicit backend fallback. Codex independently freezes/tests/reviews
the entire completed tree. Leave code uncommitted; no Phase 1 completion claim.
