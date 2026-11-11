# Phase2 revision3 — preserve the existing signed-ID contract

Resume exact session711c5973-64c3-45f8-af8a-379ba0e357a7, same12-file scope,
normal Read/Edit/Write only. Read this prompt and candidate/**; no execution,
outside reads, agents/settings/Git/network/install tools. No physics changes.

R2 completed/reaped with scope/tool errors0. Codex independently ran full450 CPU
tests (427pass/23GPU skips), all11 entry-point refusal probes, the compiled example
with checkpoint, and the ACTUAL mixed-live CUDA memory profile. All pass. The
cache is now0B, actual array lower18125656B <= declared19237416B. Preserve these
corrections; they are Codex executed checks, your tests remain NOT RUN.

One compatibility error comes from MY correction prompt: I incorrectly requested
positive IDs. Canonical leaf.state.PlantState permits UNIQUE SIGNED int64 IDs,
including zero, negative values and the int64 endpoints; it deliberately has no
positivity constraint. The existing uptake/root/event contracts support them too.
R2 followed my incorrect request and now rejects valid populations in decide.
This is a Codex instruction mistake, not a Claude scope failure.

Fix _validate_content and relevant docs/tests to require uniqueness and exact
int64 representation, WITHOUT restricting sign or zero. Preserve species-index,
finite annual-metric, stress-cap, dtype/device and every other guard. Keep the
current deterministic modulo2^64 ID hash: do not renumber, take abs, truncate,
drop IDs, change random streams or alter a population to accommodate a check.

Add meaningful numpy/numba/CUDA regressions for IDs[-2**63,-1,0,2**63-1], equality
to the independent scalar hash reference and row permutations; invalid bool/float/
duplicate IDs must still refuse before a kernel. Correct any new R2 test that
mistakenly treats a valid zero/negative ID as invalid. The exact synthetic reviewer
source review_reference/verify_signed_ids_01.py is now supplied in NEW
baseline_revision_03.json (Codex context copy, not your change). No need to touch
unrelated production modules. Update report with the CURRENT final signatures /
budgets, clearly mark earlier report sections as historical where they differ.

All author checks NOT RUN; save uncommitted. Codex independently reruns final
CPU/actualCUDA/native/restart/memory/ID checks, then publishes the whole Phase2
if accepted. Phase3 starts only after Phase2 push.
