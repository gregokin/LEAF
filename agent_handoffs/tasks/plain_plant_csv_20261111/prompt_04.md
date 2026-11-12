# Task plain_plant_csv_20261111 — revision 4, new user steering: uniform initial ages

The user added during this task: "The plant generator should generate plants with a range of ages, uniformy distributed from 0 to the life expectancy". Implement this as an addition to the metadata task. Preserve the completed revision-3 metadata code. Codex has installed it in the real tree and is independently running its CPU suite; you are sole writer ONLY in the candidate tree. No checks may be executed by you (Read/Edit/Write only). No subagents, network tools, model run, initialization, image or Git.

## Age requirement
- For a normally generated population, independently draw each individual's age from a continuous UNIFORM distribution from zero to THAT species' nominal life expectancy, using the existing seeded RNG and VECTORISED sampling. Keep existing random diameter/placement laws, cover acceptance and memory limits; do not invent an age-biomass relation.
- Read the upper bound from SpeciesCatalog.life_expectancy_days(sid), i.e. that species record's life_expectancy.mean/unit. Use the existing days/year convention. Do not duplicate life-expectancy constants in the generator/config or infer it from allometry or a Weibull scale.
- Make this the DEFAULT when initial_age_days is omitted. Provide an explicit self-describing selector such as {"distribution":"uniform_to_life_expectancy"} for the normal generation files. Reuse/extend the existing uniform sampler; existing explicit constant/numeric/bounded-uniform age controls may remain for reproducible legacy fixtures or controlled experiments. Do not change handwritten plant ages, loaded snapshots/restarts or age-zero recruits.
- If the selected species has no usable life expectancy, give a clear validation error rather than guessing an upper bound or silently using zero. Preserve explicit-age legacy configs even with life_expectancy null.
- Ensure validate/config_record/generation_report round-trip and deterministic replay still work: record resolved age bounds and their catalog origin in suitable report metadata, which belongs OUTSIDE the plant CSV. If preserving only resolved uniform bounds in replay config, explain/report the original rule separately so it is not lost.
- Update shipped examples/phase0/generation.json and the Jornada trial generation.json to use the species-linked age policy. Trial existing species means are BOER=15 years and PROGLA=80 years (provisional author traits); read them from the catalog, no new life-expectancy estimates. Document 0..15 and 0..80 year initial ranges and the distinction between this initial sampling bound and the existing probabilistic mortality law (NOT a hard death age).
- New meaningful tests: two species with different lifespan bounds/units; missing/null lifespan fails for default/selector but explicit legacy ages still work; fixed-seed reproducibility, bounded nonconstant age samples compatible with the uniform law; config/report replay; CSV age records/load and newborn behavior unchanged. A small synthetic generator test is suitable; do not generate the full actual case, run growth/water/events, render, install or benchmark.

## Preserve existing preview and catalog
The existing preview was already generated with constant ages 730 d grass / 1825 d shrubs. The user has not requested a new initialization. Keep its CSV rows, image, surface, original preview_provenance.json, initialization report and new companion UNTOUCHED. Clearly label it in case/preview docs as the earlier initialization, not a realization of the new uniform-age rule. New generation config is for the next explicitly requested initialization.
Do NOT modify the trial species.json: its complete catalog hash is bound to the saved preview. Its legacy "METADATA ONLY" lifespan note meant that it was not a deterministic age cutoff; explain in PARAMETERS/docs that the nominal value now also supplies initial-age sampling. You MAY update leaf/traits.py's general documentation of that field to describe the new usage (no mortality law change).

## Scope
Approved read/share scope is the SAME candidate leaf/, tests/, docs/, examples/, cases/, dependencies/, README.md and pyproject.toml plus this prompt. No parent/root live files, historical task evidence, upstreams or secrets. The same governing instructions below apply.
The allowed implementation writes FOR THIS revision are ONLY:
- leaf/initialization.py
- leaf/traits.py
- tests/test_initial_ages.py
- tests/test_phase0.py
- examples/phase0/generation.json
- docs/phase0-vegetation.md
- README.md
- cases/jornada_grass05_shrub10_256/generation.json
- cases/jornada_grass05_shrub10_256/PARAMETERS.md
- cases/jornada_grass05_shrub10_256/SOURCES.md
- cases/jornada_grass05_shrub10_256/README.md
- cases/jornada_grass05_shrub10_256/preview/README.md
- report.md

Do not change state/checkpoint/recording or the preview renderer in this revision unless you report an unresolved defect to Codex first. Update report.md with exact changes, design decisions, unrun checks, limitations. Leave everything uncommitted. Codex independently tests, reviews, installs and commits/pushes after acceptance.

The immutable revision baseline is recorded by Codex in baseline_revision_04.json outside the candidate (you need not read it). Do not rewrite any baseline, prompt or controls.

## Governing instructions (verbatim)
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
