# Part A revision 4 — exported helper pre-admission correction

Resume exact session fd82b85e-8a07-4982-b621-b132f845eab7. Preserve saved revision3.
Sole writer in this candidate; automatic source sharing/correction authorized.
Codex executes/reviews/publishes. This remains a subtask, not phase completion.
READ this prompt and candidate/** only; ordinary Read/Edit/Write, no execution,
Bash, agents, network, settings, installs, Git or external/archive reads.
Allowed writes remain the same ten files, but restrict actual corrections to
leaf/roots.py, tests/test_roots.py, docs/phase1-water-contract.md and report.md
unless you identify a directly necessary defect. Do not rewrite the allocator.

## Independent verification of revision 3

Stable snapshot:181 CPU tests pass (10 GPU-related skips). All21 allocator tests
pass on real CUDA. The complete native bridge composition passes its35 tests on
real CUDA. Four NumPy allocation workloads found no array-scale undercount, so
your desk concern about the reference allocator is not a confirmed defect.
Your report's runtime claims stay NOT RUN: these are Codex's executed results.

## Confirmed correction

horizontal_edges_compiled is exported in __all__. With P100000 existing typed
float64 x/y/radius arrays, nx=ny=1, dx1, max_edges=0, max_bytes=0, it creates
np.arange(P) (800000 bytes) BEFORE rejecting in the shared helper's pre-count
admission. The shared helper also validates/sorts inputs before its own admission.
Codex intercepted np.arange and independently reproduced that allocation before
AdmissionError. A first instrumentation attempt used a MagicMock that interfered
with lazy numpy.ma loading; it was corrected to a real wrapper, and only the
second run establishes the defect. Do not claim the first harness failure was
a product failure. The input arrays precede the call and are not the allocation.

Add scalar/type/shape/budget preflight in the exported wrapper BEFORE its own
copies/IDs or shared-helper validation. Reuse the shared estimator as appropriate
and include all validation/cast/sort scratch/owned arrays. Typed inputs may be
required consistently instead of materializing sequences. Noncontiguous/cast
paths must be admitted before copies or refused before conversion. Preserve the
shared builder physics and upstream benchmark source. A zero-byte cap must
refuse before any P-sized allocation. Add a meaningful trap regression.

Also close the ledger explanation for the 16P state._ensure_index cache created
by overrides: it survives beyond roots.overrides, so hold its newly created
arrays in subsequent stages or avoid creating the persistent caller-state cache
with a locally admitted lookup. Do not claim every live component is tracked
while dropping that newly allocated resident component. Preserve scientific state
and caller-owned arrays; preexisting caller-held cache can remain documented as
external. No fixture-specific allowance or increased configured limit.

Update docs/report with the actual final scope, all tests NOT RUN and exact tool
errors if any. Save incremental corrections. No Git/install/model execution.
Codex will freeze and rerun the affected guards/root tests and broader checks
when the actual changes justify them. No new user approval is needed.
