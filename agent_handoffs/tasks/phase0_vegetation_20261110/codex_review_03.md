# Phase 0 review — revision 3

Verdict: **accept**. Reviewer: Codex, 2026-11-10. Accepted source:
`revision_03_manifest.json`; 16 installable files match the frozen candidate
byte-for-byte (`installation.json`). Claude was the sole implementation writer;
Codex reviewed, tested, installed and updated project/control documentation.

## Delivered behavior

- Versioned species JSON and one CSV plant schema for initial input, snapshots
  and final output: stable ID, species, subcell coordinates, establishment ground
  anchor, aboveground **dry** kg, seasonal mass, age and status.
- Strict transactional validation, catalog fingerprints, exact FP64 round trips,
  initial-only `SURFACE` resolution, elapsed-time aging and selectable output cadence.
- Seeded multiple-species initialization by union cover and mean/SD diameter,
  with optional global canopy exclusion, explicit finite boundaries, numerical
  resolution/tolerance, feasible-resource admission and reproducible input config.
- Independent NumPy-only loading/placement/allometry and MAPLE-compatible canopy
  fields with ID sidecars. Native publication checks current-surface burial and
  elevated bases; MAPLE's own validator accepted the supplied fields.
- ID-keyed erosion/deposition/abrasion/flux diagnostics and transactional species
  response socket, without introducing ecological injury laws.
- Opt-in legacy B/E capture at requested outputs, exact law/identity checks,
  pre-transfer device-vector validation and fresh-driver seasonal restoration.
- Examples, CLI, package metadata and concrete MAPLE developer handoff.

## Independent evidence

Installed suite: **134 checks, 127 passed / 7 CUDA skips on CPU; 133 passed /
1 intentional skip with physical GPU 1**. All 44 new Phase 0 tests passed.
Independent state, ownership, malformed input, canopy safety, cover/memory,
generation/replay, transfer-preflight, native MAPLE ABI and actual CUDA bridge
checks passed. Five installed CLI commands passed. A real source distribution
built and imported with heavy/native imports blocked; legacy imports also passed.
See [verification.md](verification.md) for exact commands, paths and limitations.

The 512² initialization fixture generated 51,893 plants in 1.635 seconds with
3,217,366 bytes of plant arrays. This is initialization, not model-year speed.
The process RSS includes the independent SciPy oracle; it is not live GPU memory.

## Scope and science

The 1,796-file initial baseline was checked; only listed runtime/control/project
documentation files changed. Root AGENTS.md, legacy numerical drivers, prior
accepted archives and historical locks are preserved. All 71 locked SYRUP
runtime hashes and inspected MAPLE interface/instructions match. Actual current
upstream dirty identities are newly pinned; no upstream source was written.
Claude's three successful, reaped revisions used only scoped Read/Edit/Write
tools; tool paths and structured results are in `claude_tool_audit.json`.

Revision 1 required ownership/safety/resource corrections; revision 2 required
focused serialization/replay/preflight corrections. Those verdicts remain
historical. Codex corrected a final probe's NumPy-bool JSON serialization and a
relative-PYTHONPATH invocation error; the implementation passed the corrected
independent check. No limit was raised or state discarded to pass verification.

Fixtures are uncalibrated. Exclusion is a sequential large-canopy-first hard-core
process, not an unconditional Poisson process. Sampled finite-domain cover is
not exact analytic or periodic MAPLE cover. CSV is not a coupled water/calendar/
RNG restart. Recruitment, mortality/injury laws, evolving subsurface remapping,
MAPLE adoption and full MASY scheduling remain separately scoped work. Wheel
building was not verified; no dependency/environment was installed.

The user authorized a Codex commit and push after acceptance. Local commit may
now proceed. Push awaits a destination: this workspace initially had no usable
Git metadata or remote, and the proposed GitHub name was not found by the
read-only probe. No repository creation or guessed push is authorized by a URL.
