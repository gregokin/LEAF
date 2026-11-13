# Bounded implementation: MASY daily plant interface, revision 1

Task ID masy_daily_plant_interface_20261112. You are Claude, sole implementation writer. Codex independently reviews, executes checks and installs. User explicitly authorized automatic Claude/Anthropic sharing, this interface, small tests, then Codex commit/push. Further Phase 3 work is PAUSED. Existing phases 0-3 mechanics and the case parameterization must be preserved, not restarted.

## Identity / scope
CWD /home/okin/leaf/agent_handoffs/tasks/masy_daily_plant_interface_20261112/candidate; baseline LEAF 9fd4bbbe2faae907922ed7194f633caa1da3624f (clean at task start). Your candidate has the source/docs/tests/examples only, no trial case.
Read scope: candidate tree, /home/okin/leaf/agent_handoffs/tasks/masy_daily_plant_interface_20261112/public_evidence (explicit immutable SYRUP N15 and LEAF C1 references), and baseline.json, syrup_selection.json, qualified_leaf_reference.json in /home/okin/leaf/agent_handoffs/tasks/masy_daily_plant_interface_20261112. No other files, private sources, settings, credentials, upstream checkouts or old task directories.
Allowed writes ONLY these candidate-relative files:
- leaf/plant_pet.py
- leaf/biocalendar.py
- leaf/growth.py
- leaf/plant_session.py
- leaf/plant_checkpoint.py
- leaf/lifecycle_bridge.py
- tests/test_plant_session.py
- tests/test_plant_checkpoint.py
- tests/test_lifecycle_bridge.py
- tests/test_plant_pet.py
- tests/test_biocalendar.py
- tests/native_leaf_fixture.py
- docs/masy-daily-plant-interface.md
- examples/masy_daily_plants.py
- interface_report.md
Preserve every other file. No changes to interevent.py, checkpoint.py, syrup_water.py, uptake.py, roots.py, canopy.py, traits/state, recruitment/demography/disturbance, old tests, historical locks, trial case or upstreams. Additional necessary scope must be reported to Codex; do not guess permissions.
Allowed tools Read/Edit/Write only, no Bash, execution, network, agents, installs or Git. Codex runs verification. Report tests as NOT RUN by you. All output implementation stays uncommitted.

## Reuse selected sources before implementing
Read existing leaf/growth.py, roots.py, uptake.py, canopy.py, syrup_water.py, interevent.py, checkpoint.py, state.py, forcing.py, traits.py, demography.py, disturbance.py, recruitment.py and their docs.
Qualified C1 LEAF at 52f39fe06166a91e768740b19d3c26d4ffceaf6e is supplied in public_evidence/leaf_qualified/leaf. It already implements canopy PET (plant_pet.py), biology calendar (biocalendar.py), growth.demand_per_plant extension preserving demand laws, and delegation of a few calendar functions in interevent.py. Reuse plant_pet.py, biocalendar.py and growth.py from this reference, initially exact copies; DO NOT replace interevent.py (preserve main's standalone driver). If changes to these reference extensions are necessary, document/justify narrowly, and preserve existing growth law/scalar demand signatures.
SYRUP snapshot public_evidence/syrup_phase15 is actual published feature/subsurface-daily-recharge a5b8b18358e0d014fe13c9f48d00eb582710655a, 114-source-file package 82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107. Read its docs/subsurface_hydrology/leaf_water_contract.md, phase15_qualification.md, phase15_source_manifest.json, phase15_acceptance.json, and actual src/maple_syrup/daily_water.py, lifecycle_availability.py, water_lifecycle.py, recharge.py interfaces. The current live SYRUP checkout is older c76c603 plus dirty work; never use/rebind it for this new connection. Old LEAF syrup_water pin c76c603 remains intact for standalone drivers.
Read public native fixture/example in SYRUP docs/manuals/examples/native_daily_recharge/daily_recharge_example.py and tests/subsurface to write a small real-native integration harness. Its sibling native_leaf_calendar_cli case data may not yet be in snapshot; report exact requirements for Codex to supply the fixture. Never fake a lifecycle state or replace physics merely to make a test pass.

## Binding architecture / public operations
LEAF owns plants, roots, potential demand, allocation, biological accumulators and growth. SYRUP owns sole physical inventory and EVERY water update. MASY supplies all forcing/temporal controls from scenario.csv, sequences calls and accepts/checkpoints combined state. The new API MUST NOT call any hydrology/evaporation/standalone debit or maintain/copy authoritative water. Do NOT hide MASY sequence inside IntereventSession.
Provide a public water-free biological session/state with immutable accepted parent and staged candidates (names consistent with repo). Reuse existing leaf.growth, uptake, roots, canopy, biocalendar, plant_pet, biology laws. Thin lifecycle_bridge may READ SYRUP availability/identity and validate actual withdrawal results; all physical daily/withdrawal calls stay visible at MASY call site. In particular the plant interface must not call syrup_water.withdraw_plants or native standalone withdraw_transpiration. A lifecycle availability read is pure and can be used in bridge.
Public operations must cover:
1 prepare daily demand from plant state, nominal fixed-offset date/context, T, shared terrain PET, independent transpiration/growth flags, allowed process kind;
2 propose root-weighted withdrawals against FRESH SYRUP availability;
3 consume actual accepted resource withdrawals, produce per-plant actual/unmet/satisfaction;
4 stage accumulation and scheduled growth without accepted-parent mutation; 5 publish/discard candidate with anti-stale/anti-duplicate identity;
6 export accepted root/canopy information through public methods; 7 biological-only save/restore with explicit external SYRUP bindings, no restored water;
8 expose required season/annual/mortality/reproduction/disturbance boundaries and existing counters, preserving existing mechanics. Do not duplicate or invent ecological laws. A new wrapper need not implement a full event scheduler or full coupled checkpoint, but must clearly expose genuine unsupported boundaries rather than silently skip due decisions. If delegating to existing public annual evaluation functions, retain their semantics and counters. Record any scope/scientific question promptly in report; do independent work meanwhile.
Use explicit typed value objects/capabilities with validated IDs. Tie nominal day + biology sequence + plant/root generation + actual SYRUP parent INSTANCE/digest/revision/request sequence. Equal scalar alternate physical branches must be refused. Feed actual SYRUP result and receipt/identity; a stale plan, wrong parent, wrong result, duplicate publication/debit identity must refuse before mutation. Give MASY a serializable stage identity to resume without replay. Physical water candidate may exist externally after withdrawal; LEAF failure keeps accepted biology intact, with enough saved pending plan/feedback/stage metadata to retry biology without issuing another withdrawal. Do not retain unbounded receipts/history.
Serialization must preserve accepted plants and pending demand/actual, growth integrals, geometry identities/maps, relevant biological counters, and any staged uptake needed for exactly-once resume. Canonical PLAIN nine-column plants.csv plus companion metadata, versioned separate biological codec; leave standalone checkpoint version 4 and old locks untouched. Restore external lifecycle reference via explicit bindings and renewed live capabilities, not a saved raw pointer or new water store. Reject corruption, wrong catalog/source/geometry/day/resource parent and stale/duplicate replay; allow existing explicit source-compatibility policies only when justified.

## Scientific invariants
Root CSR normalized, no species priority, local proportional competition, no compensatory redistribution from dry cells/layers. Inactive resources give zero supply, retain root weights. Actual accepted total U/D summed across a plant drives existing growth, zero demand S=0. Native availability floor/mask only, no LEAF substitute. Volume-depth conversion ONCE each direction: supply depth * native cell_area_m2, proposal / same area, accepted depth * same area. Canopy PET is area-weighted terrain mean from existing qualified sparse plant_pet (domain/mask provenance, zero support refuses), NOT root-weighted PET; separate canopy/root geometry. Preserve scalar demand too, no new ecological law or catalog defaults. Independent flags: T true/growth false = real water debit, no new D/U/activity/T/time growth credit; earned credit retained and explicit flush before disabled/transport per biocalendar. No E/T/growth during wind/rain/recession, no retrospective event-duration credit. Prescribed SYRUP hydrology-cover/species-infiltration maps stay prescribed (Phase2.5 deferred).
Biological dates and age derive nominal calendar/explicit elapsed context, NEVER accumulated native storm/wind seconds. MASY has no independent LEAF weather/scheduling config. Daily biology and supported growth cadence <=7 days; expose splits/flush/year boundaries. Calendar age includes transport according to existing live-age rule, but native physics seconds do not determine nominal date; document exact bindings and unresolved choices.

## Required visible daily call sequence in developer handoff/example
MASY completes nominal day's wind/precip treatment -> SYRUP dw.daily_water_action or finish_daily_storm ONCE (small rain, redistribution, drainage, E) -> fresh lifecycle_availability.layer_availability_m -> LEAF demand + propose -> wl.request_withdrawal(lop, parent, expected_parent=wl.expect_parent(parent), depth_m=..., identity=..., retained_feedback_bytes=...) ONCE -> LEAF actual feedback + stage growth -> MASY accepts paired water/biology/scheduler. wl.request_withdrawal is ZERO time: unchanged t_s and hydro_count, T request count +1. No extra daily operation/clock advance hidden anywhere.
Document exact real signatures, shapes (L,N resources, P plants, CSR E), units, ownership, backend/device restrictions and explicit bridge source selection/guards (new qualification rather than old pin replacement). Future full MASY integration/acceptance not claimed.

## Memory / backends
numpy reference, compiled numba CPU and existing cuda paths on current device; no silent fallback, no dense P x N or P x L x N tensors, no routine P/E-sized host/device readbacks. Sparse canopy/roots and per-plant arrays. Stage-sum admission before allocation, old/candidate coexistence, no raised bounds/dropped plants/state. Reuse existing memory estimates where possible; document cold weekly root/canopy rebuild host transfers, checkpoint transfers, scalar diagnostics and exclusions. CPU/GPU parity under existing numerical tolerances; available paths actually measured by Codex later.

## Checkable deliverables / acceptance
- implementation + focused unittest tests daily/7day cadence, zero demand, dry/competing/multilayer/inactive cells, native actual shortfalls, independent flags, stale/duplicate/wrong-parent refusals, candidate discard and failure atomicity including growth/root rebuild, biological save/load incl pending stages/counters/explicit binding; no case runs;
- small real SYRUP Phase1.5 CPU/CUDA harness proving daily hydrology count once and T once, water closure, unchanged physical clock at withdrawal, no copied/fake inventory;
- scalar growth/PET and standalone behavior regression; core imports optional packages lazily; old source guards/formats retained;
- docs/masy-daily-plant-interface.md complete public call sequence usable without private LEAF fields, concise scope/limits/resumption point;
- interface_report.md exact modified files, API summary, known limitations, tests NOT RUN by author, source reference and any questions.

## Governing instructions (embedded for safe-mode launch)
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

# Claude entry point

This file governs sessions that auto-discover it. Safe-mode launches do not read it; they rely on the governing instructions embedded in the archived task prompt, which must be consistent with this file and [agent_handoffs/AGENTS.md](agent_handoffs/AGENTS.md).

## Read order

1. [AGENTS.md](AGENTS.md) — shared conventions.
2. [agent_handoffs/AGENTS.md](agent_handoffs/AGENTS.md) — orchestration rules.
3. [agent_handoffs/orchestration_state.md](agent_handoffs/orchestration_state.md) — mode, baseline reference, scope holder, open items.
4. [agent_handoffs/current_prompt.md](agent_handoffs/current_prompt.md) — the current bounded task, or an explicit pointer to an archived one.
5. [agent_handoffs/project_brief.md](agent_handoffs/project_brief.md) — reference link only. Read it only if the current task's source scope lists it.

If any of items 1–4 is missing, empty, or unreadable: stop, report "dispatch incomplete — pending orchestration", and do no guessed work.

## Context and source scope

- Codex reads local project context and supplies the bounded, approved subset inside the task prompt. Claude reads only the files and directories the task's source scope authorizes.
- No generic authorization covers scientific briefs, data, or other payloads. Not listed in the task's source scope means not readable for that task.
- Within the granted scope and the execution controls already in place, act without seeking per-action approval.

## What counts as an assignment

- `current_prompt.md` assigns work when it contains either a filled bounded prompt (task ID, baseline, allowed writes, source scope, acceptance) or an explicit reference to an archived bounded prompt under `agent_handoffs/tasks/<task_id>/`. A pointer need not duplicate every field; the archived prompt is authoritative.
- Idle markers, unfilled copies of [agent_handoffs/templates/task_prompt.md](agent_handoffs/templates/task_prompt.md), and historical prompts are not assignments.
- Work only on the current bounded task.

## Working rules (summary)

- Implement or correct within the allowed writes; excluded paths and excluded payloads stay untouched.
- Leave changes uncommitted. Commits, pushes, dependency installs, and model runs happen only under separately granted task authority; setup grants none.
- Report in `agent_handoffs/claude_report.md`, shaped like [agent_handoffs/templates/task_report.md](agent_handoffs/templates/task_report.md). Never mark a check or acceptance as passed unless it actually ran and passed.
- Use short, interruptible waits. On interruption, preserve partial changes and record the exact session details.

Full rules: [agent_handoffs/AGENTS.md](agent_handoffs/AGENTS.md).


The candidate's root AGENTS.md is a preserved historical scaffold. The task above and embedded current instructions govern actual scope. Root control files outside candidate are Codex-owned. Your report is candidate/interface_report.md, not root agent_handoffs/claude_report.md. Work autonomously within this boundary; no user confirmation needed.
