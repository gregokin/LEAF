# Correction task plain_plant_csv_20261111 — revision 3

Resume the same candidate; preserve all partial work. Codex reviewed the completed revision 2, verified source/tool scope, installed a draft in the real repository, and ran the full independent CPU suite. It ran 561 tests in 15.964 s, with 45 expected CUDA skips and 15 ERROR entries all cascading within ONE new mutation test. All other executed tests passed. No field case or image regeneration was run. This is NOT acceptance yet.

Correct these bounded findings:

1. **Mutation-test fixture bug (mandatory)**. tests/test_plant_csv.py TestCompanionVerification.test_edited_companion_fields_are_refused corrupts metadata.schema to leaf.other, then at the start of the next mutation calls write_plant_csv(...overwrite=True). The production guard correctly REFUSES overwriting an unrelated-schema companion, so subsequent mutations cascade before reaching their assertions (FileExistsError at state.py _check_replaceable_companion). Restore the pristine fixture metadata directly in the TEMP TEST directory between mutations, or use a fresh temporary pair per case. Do NOT weaken the production unrelated-metadata guard. Preserve all corruption checks.

2. **Actual companion memory accounting (mandatory)**. checkpoint.PLANT_METADATA_LEDGER_BYTES = 3 * 16384 is admitted as conservative, but provenance size is unbounded. Replace the arbitrary guess with accounting of the actual companion JSON size (or a demonstrably conservative bound that scales with retained provenance), before the large serialization allocations. A streaming/counting JSON encoding pass is acceptable. Preserve max_bytes/reserve_bytes and all provenance; do not silently omit metadata, raise caller limits, or introduce arbitrary provenance truncation. Include a large-provenance test: sufficient budget round-trips it unchanged, tight budget refuses before files/large buffers are published. Keep existing load-size checks. Small structural control-object overhead may follow the existing documented exclusion, but the serialized text/bytes must be accounted.

3. **Preview helper scope correction (mandatory)**. I overstated an existing rerender path in the original prompt. The original script had no --rerender-from CLI; the earlier operator used render() directly. Remove the newly added rerender feature (new argument, extra functions/constants/docs), since this task is serialization only. Keep the existing initializer's declared companion output, preflight, provenance file hashes and readback verification. No new initialization, image rendering, model run or subprocess is authorized. The current existing PNG and all plant rows will be preserved by Codex's pure metadata split.

4. **Metadata consistency (small)**. csv.n_bytes is emitted but not verified. Require a non-boolean nonnegative integer and verify exact length against CSV bytes alongside SHA-256; add mutation checks. When tests intentionally re-bind edited CSV bytes to exercise row-count/SURFACE checks, refresh BOTH hash and size so failures reach the intended check.

5. **Version documentation**. You reported stale module docstrings in leaf/interevent.py and leaf/__init__.py. Additional writes to these TWO files are now allowed ONLY to correct version-4/plain-CSV/companion descriptions; no runtime/physics changes. README's 'still need an independent run' sentences for Phases 1–3 are stale from before the accepted independent historical reviews. Replace those stale statements with pointers to the corresponding existing final-review records; do not claim final current tests passed (one test failed and your corrections remain unrun).

6. **Existing preview provenance/documentation**. The active preview_provenance.json is the preserved original render record and has pre-split CSV hashes. Codex will keep it immutable and produce a separate plant_files_migration.json addendum containing current pair hashes, original provenance hash and the row/image preservation checks. Update preview/README.md accordingly: distinguish what newly executed helper outputs would contain from what this existing historical picture was made with. Do not claim that this existing picture was produced by the new helper or that its original provenance already contains companion hashes. Do not write or rerender the existing data yourself.

Report the corrections in candidate/report.md. Still Read/Edit/Write only; do not run checks. Codex will repeat the full independent CPU checks after installation. Same scope plus the two narrowly scoped docstring files, same source-sharing authority and all original exclusions. The required governing instructions and original full task are repeated below; the corrections above take precedence where they clarify scope.

# Task plain_plant_csv_20261111 — revision 1

User: "from now on, don't embed metadata in the plant file. that should be somewhere else." Then "ok, commit and push when done."
You are Claude, sole implementation writer in the CANDIDATE tree /home/okin/leaf/agent_handoffs/tasks/plain_plant_csv_20261111/candidate. Codex will independently review, run checks, install accepted changes, convert the existing preview without regenerating plants, and commit/push. You MUST NOT commit/push/run anything yourself. No subagents.

## Task and requirements
Implement metadata-free plant CSV outputs throughout LEAF: first line is the nine-column plant header, every data row has nine fields, no embedded JSON or comments, including init/generate CLI, ordinary snapshots/final outputs, checkpoint plants.csv, and the preview helper. Keep all plant data/science/units identical. This is serialization only.
Use separate human-readable companion JSON for saved-file metadata, consistently named <stem>.metadata.json, e.g. initial_plants.metadata.json or plants.metadata.json. Retain necessary time, domain, species catalog identity, initial/SURFACE safety, row counts, units and provenance checks. File readers should automatically load companions and verify they match the CSV (include a CSV SHA-256 binding so mismatched partial writes cannot be silently used). In-memory serializer always returns plain CSV; allow explicit metadata dict for in-memory parsing if useful. Backward READ compatibility with embedded-metadata legacy inputs is acceptable; no code path may WRITE embedded metadata, even opt-in.
Maintain atomic/no-clobber behavior as far as possible for a two-file pair: validate and serialize both before writing, preflight both, avoid overwriting unrelated metadata, clean up newly created outputs on failure, refuse mismatched/stale pairs on reads. Do not claim two renames are atomic; document detected-interruption behavior if needed.
Checkpoint metadata may live in companion JSON or restart.json, but normal read_plant_csv on checkpoint plants.csv should remain convenient. Preserve complete restart fidelity, verification hashes, memory limits, transactional publication, SURFACE rejection for non-initial snapshots, and test safety. If changing restart layout/version, explicitly document compatibility. Prefer minimal clear version handling over silent reinterpretation.
Update CLI generate report/output collisions to cover metadata paths too. Preserve failure rollback behavior for optional report. Preserve canonical CSV columns and dry-mass meaning. Update docs and focused regression tests that exercise real CSV + companion roundtrip, edited or missing/mismatched metadata, plain initial input with caller context, legacy reads, SURFACE and catalog/time/grid guards, no-clobber and injected write failure, CLI generate/inspect, snapshot and complete restart. Adapt old tests honestly to the new storage location instead of weakening validation.
The existing case's preview plant table is generated time-zero data. DO NOT regenerate/reorder/alter/delete records or images. Codex will split its first metadata line into the new companion after installing the writer. Your preview helper/docs must support the new file layout and provenance file hashes. Its rerender path must read correctly.
Persist the user's no-embedded-metadata convention clearly in current documentation. Root AGENTS.md is intentionally preserved per earlier user instruction and MUST NOT be changed.

## Baseline
Actual HEAD 1e468e1410992430e0aa5045ca44637b382ac0e9; immutable full actual-source and existing dirty-file identities are in ../baseline.json (provided by Codex; you need not read outside the candidate). Existing trial-case/image changes are uncommitted and must be preserved. No packages or environment changes.

## Source scope (read/share)
Only candidate leaf/, tests/, docs/, examples/, dependencies/, cases/ and candidate README.md, pyproject.toml; this prompt and the embedded governing instructions below. All other repository task histories, scientific briefs, upstream project files, secrets/settings and other directories are excluded. Candidate includes copies only. Binary terrain/image assets are deliberately omitted; do not create or run them.

## Tools and allowed writes
Read, Edit, Write ONLY. No Bash, execution, network tool, tool discovery, agents, settings, Git or installs. Read only approved candidate files. Write only these candidate-relative files:
- leaf/state.py
- leaf/__main__.py
- leaf/checkpoint.py
- leaf/recording.py
- tests/test_phase0.py
- tests/test_interevent.py
- tests/test_plant_csv.py
- README.md
- docs/phase0-vegetation.md
- docs/phase1-interevent.md
- docs/phase2-disturbance.md
- docs/phase3-reproduction.md
- docs/shared-canopy.md
- docs/maple-leaf-integration.md
- cases/jornada_grass05_shrub10_256/README.md
- cases/jornada_grass05_shrub10_256/preview/README.md
- cases/jornada_grass05_shrub10_256/preview/render_preview.py
- report.md
No orchestration control files or baseline/prompt edits. If another file needs changing, report it to Codex. Scope failures are failures.

## Acceptance and reporting
Codex will run the CPU unittest suite in the existing MAPLE environment with current native SYRUP on PYTHONPATH (no field/model runs or benchmark). Independently verify exact same current preview plant row bytes/count and image hash before/after metadata split, strict CSV nine fields, matching companion context and hashes, and source scope. Do not mark tests passed unless actually run (you are NOT authorized execution). Produce report.md with changes, design/compatibility choice, suggested exact checks, limitations and work not done. Stop when done or on tool/network/auth/rate-limit error; preserve partial work.

## Governing Codex–Claude instructions (verbatim)
# Orchestration instructions

Location: `agent_handoffs/`. Links are relative to this directory. The file map is in [README.md](README.md); files listed there but absent are supplied locally by the orchestrator.

## Roles

- **Codex (orchestrator / reviewer):** reads local project context, supplies the approved bounded context in each task, writes tasks into [current_prompt.md](current_prompt.md), maintains [orchestration_state.md](orchestration_state.md), independently reviews and verifies, and records the verdict in [codex_review.md](codex_review.md) with evidence in [verification.md](verification.md).
- **Claude (implementer):** executes exactly the current bounded task within its write and source scope, corrects on review, reports in [claude_report.md](claude_report.md), and leaves changes uncommitted.

## Automatic collaboration mode

- "Automatic" means the user has pre-approved the dispatch → implement → review → verify → correct loop for the bounded tasks the user requested, until the user changes that preference. Record the preference and its date in `orchestration_state.md`.
- These documents create no daemon, scheduler, or live process. Every launch is explicit and foreground, bounded by one task.
- Automatic mode never widens permissions, tool scope, source scope, or the set of tasks.
- An execution error (tool, auth, network, rate limit) pauses the current task; it does not switch off the user's preference. Only the user changes the preference. Codex records the error and the pending state and continues within its authority.

## Dispatch preconditions

- Check the installed `claude --help` and actual files/processes before a new launch. Use waits of at most 60 seconds and report meaningful progress while active.
- Dispatch requires: a readable `orchestration_state.md`, a bounded prompt (inline or an explicit pointer to an archived prompt under `tasks/<task_id>/`), and the governing instructions available to the session (auto-discovered, or embedded in the prompt for safe-mode launches).
- If any of these is missing, dispatch stops pending orchestration. Nobody starts guessed work.

## Baseline

- Each task records the actual Git HEAD and existing dirty/untracked source identities, plus relevant dependency locks. Where Git metadata is unavailable, use `tasks/<task_id>/baseline.json` with actual relative paths, sizes, and content hashes. A HEAD alone does not describe dirty runtime inputs.
- A manifest replaces missing Git metadata. Never fabricate history, commits, or hashes. Never initialize Git to obtain a baseline.
- Baselines and dispatched prompts are immutable once dispatched.

## Scope and acceptance

- Each task lists: allowed writes, exclusions, source scope (files/directories that may be read or shared), excluded payloads, allowed tools and directories, and acceptance criteria checkable by a command or a named output.
- A task may explicitly assign documentation writes under `agent_handoffs/`. Unless the exact task allows it, preserve orchestration control files (`orchestration_state.md`, `current_prompt.md`, `codex_review.md`, `verification.md`), immutable baselines and prompts, and historical task evidence.
- Writes outside the allowed list, or reads outside the source scope, are task failures even if the result is correct.
- Acceptance is recorded by Codex in `codex_review.md`. `verification.md` holds the evidence; it is not acceptance by itself.

## One writer per implementation scope

- Exactly one agent holds write rights to a given scope at a time; Codex records holder and scope in `orchestration_state.md`.
- Documentation writes explicitly granted by a task are part of that task's scope and do not violate this rule.
- Read-only review, verification, and diffing may run concurrently with implementation. Overlapping write scopes are serialised.

## Process and session identity

- A real launch is evidenced by at least: the session UUID; the exact command and working directory; the PID together with a start identity (for example process start ticks from `/proc/<pid>/stat` on Linux, or the platform equivalent); start timestamp with timezone; raw stdout and stderr; exit status and how the process was reaped, with end timestamp.
- Formats are flexible: JSON launch/exit records (e.g. `launch.json`, `exit.json`) and numbered artifacts are acceptable. No particular filename or `baseline/` directory is mandated.
- PID alone is insufficient. Before signalling or terminating a process, verify that PID and start identity match the record.
- Session UUIDs are generated by the orchestrator and passed explicitly so exact sessions can be resumed deliberately. Never describe a launch that did not happen.

## Output capture

- Capture stdout and stderr to separate files; never merge them. Record the exit status separately.
- Preserve literal error text in evidence. Raw streams plus exit record are primary; summaries are secondary.

## Structured errors

- The structured JSON output is authoritative. `is_error: true`, an error subtype, an empty or refused result, or an explicit auth/network/limit message is a failure even if the exit code is 0.
- A non-zero exit with usable partial output is an interruption, not success.
- Classify every failure as one of `scope`, `check_failed`, `tool_error`, `auth`, `network`, `rate_limit`, `interrupted`, `other` and record it in `verification.md`.

## Rate limits versus other errors

- Record `rate_limit` when the output explicitly reports a usage or rate limit, even if no reliable reset time is given.
- Copy the reset time and timezone verbatim when supplied; otherwise record `reset: unknown`. Never guess, round, or invent a reset.
- A derived resume time may be kept in a separately labelled field only when computed from a reliable literal reset.
- DNS, auth, network, and test/check failures are never classified as usage limits.
- On a rate limit: preserve state and partial work, use short interruptible waits, do not retry indefinitely, and hand control back to orchestration.

## Continuation and resumption

- Active orchestration may resume the exact recorded session (by UUID) within its granted authority; this is continuation of the same task, not a new task.
- A promise of unattended resumption is allowed only when a live process or scheduler has been verified and recorded. These documents create none, so no such promise is made by default.
- Do not prohibit continuation the user has approved, and do not silently disable the preference after an execution error.

## Interruptions and partial work

- Keep partial changes on disk, uncommitted. Do not revert, stash, or clean up without an explicit task to do so.
- Record in `claude_report.md` what was done, what was not, the literal error text, and the session UUID.

## Codex review and verification loop

1. Read `claude_report.md` and the diff against the baseline (`git diff` if Git exists, otherwise a `baseline.json` comparison).
2. Check write scope and source-scope compliance.
3. Run final acceptance checks on the completed tree or a stable immutable snapshot. Re-run each acceptance check independently; record commands, output locations, and pass/fail in `verification.md`. Do not copy results from the report.
4. Record the verdict in `codex_review.md`: `accept`, `correct` (bounded correction list), or `stop` (reason). Acceptance exists only when written here.
5. On `correct`, dispatch a revision (same task ID, incremented revision). On `accept`, update `orchestration_state.md` and set `current_prompt.md` to idle. Committing, if any, needs separately granted task authority.

## Archival evidence

- `tasks/<task_id>/` keeps the dispatched prompt, baseline reference or `baseline.json`, launch/exit records, raw stdout and stderr, report, review, and verification.
- Durable decisions go to [decisions.md](decisions.md); external references and dependencies to [upstreams.md](upstreams.md).
- Task evidence is append-only once the task is accepted or stopped.

## Permissions and setup

- Normal permissions only. No permission-bypass flags, no elevated privileges, no blanket approvals beyond the tool list a task grants.
- Within granted scope and execution controls, no additional per-action human approval is required.
- `Bash`, if ever needed, is granted per task with the exact commands listed.
- Git init, commits, pushes, dependency installs, and model runs occur only under separately granted task authority; setup grants none.

