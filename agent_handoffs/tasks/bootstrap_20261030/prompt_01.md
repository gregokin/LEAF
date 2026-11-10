# Bootstrap leaf automatic Codex-Claude handoffs

## Objective and authority

The user says: "In this project we're going to investigate a vegetation model that can be run with ~/maple and ~/syrup (and their combined ~/masy). You will work with claude, who will code with your review, as in those others. Set up agent_handoffs here so you can do automatic orchestration with claude."

Implement only the documentation setup in `/home/okin/leaf`. Codex is the orchestrator and independent reviewer; Claude implements bounded assignments and corrections, leaving changes uncommitted. Automatic preference applies until changed to requested leaf tasks. It does not authorize self-directed model development or a persistent service.

## Baseline and reference findings

Read the existing root `AGENTS.md` but preserve it byte-for-byte. There is no usable Git metadata or application code. The pre-edit manifest and AGENTS copy are in `agent_handoffs/tasks/bootstrap_20261030/`.

The reference repositories actually use uppercase paths: `/home/okin/MAPLE` (wind), `/home/okin/SYRUP` (water), and `/home/okin/MASY` (combined native wind/water wrapper). Codex already inspected their workflow instructions. Their protocol is: bounded current prompt; exact baseline; archived per-task prompts/reports/review/verification; ordinary Claude CLI permissions; explicit tool and directory scope; separate raw stdout/stderr; exact session and PID/start identity; exit/reaping record; one writer per tree; Codex independently inspects actual changes and verifies before acceptance. Do not read the upstream repositories or personal configuration for this task.

## Writable files

Create only:

- `README.md`: concise project purpose, actual upstream paths and workflow entry points. State vegetation model/scientific choices are still open and no model integration is implemented.
- `CLAUDE.md`: entry point directing Claude to root AGENTS, handoff instructions, orchestration state and current prompt. Idle/template prompts are not assignments; leave work uncommitted.
- `agent_handoffs/AGENTS.md`: actionable workflow instructions for work coordinated here, including Codex/Claude responsibilities, bounded until-changed preference, one writer, independent verification, ordinary permissions, baseline and evidence, interruption and resume behavior.
- `agent_handoffs/README.md`: map durable files and the step-by-step handoff/correction/acceptance cycle. Include a CLI example verified against options below, scoped to a task, with separate log capture and an explicit UUID session. Say an example is not a full runner and launches need recorded process identity/exit and inspection of JSON errors. Do not include bypass flags.
- `agent_handoffs/decisions.md`: user-authorized scope, roles, no chosen vegetation formulation, actual uppercase upstream locations, and questions to resolve only before relevant scientific work.
- `agent_handoffs/upstreams.md`: upstream roles, source ownership and future verification needs; no claimed compatibility or test acceptance. Prefer adapters and actual upstream APIs, no copying/forking or altering wind/water physics as setup.
- `agent_handoffs/templates/task_prompt.md`: reusable bounded assignment with task/baseline, requirements, allowed writes, exclusions, acceptance, checks, deliverables and stop conditions.
- `agent_handoffs/templates/task_report.md`: changed behavior/files, exact executed checks/results, unresolved findings/limits, and uncommitted status.
- `agent_handoffs/claude_report.md`: your actual bootstrap completion report. Identify file inspection versus checks actually run. You have file tools only; Codex will run structural checks.
- `agent_handoffs/tasks/bootstrap_20261030/claude_report.md`: archive the same completion report.

Do not edit any other files. In particular Codex owns `current_prompt.md`, `orchestration_state.md`, `codex_review.md`, `verification.md`, launch records and baseline files. Do not mark Codex review or acceptance complete.

## Requirements

Keep the setup concise and consistent with the existing contributor guide. Keep implementation and task evidence under the documented paths. Preserve historical evidence and partial work. Distinguish authorized automatic preference from an actually running process. On a reported usage limit, record the literal error/reset/timezone; do not guess reset times or misclassify auth/network/permission failures. Resume only the exact recorded session after checking files/processes, no ambiguous latest-session continuation and no indefinite retries. Record blocked tool actions honestly.

No Git initialization/commit/push, environment creation/install, simulation, GPU allocation, upstream changes, scientific parameterization or growth/mortality/ET/uptake law is part of this setup. Source/design collaboration is authorized; do not share actual simulation payloads or credentials as routine source review.

The locally installed `claude --help` was checked. Supported relevant flags are `--print`, `--output-format json`, `--session-id <uuid>`, `--resume <id>`, `--tools <tools...>`, `--allowedTools <tools...>`, `--permission-mode dontAsk`, `--permission-prompts none`, and `--add-dir`. Ordinary permissions with an explicit allowed-tool list are available. For documentation examples use quoted paths, no shell eval, and narrowly allowed tools; CLI examples must leave tool permission scope explicit. Claude implementations that need execution should have task-specific Bash commands allowed by Codex, not blanket unrestricted Bash. Do not prescribe a model or claim unmeasured performance/compatibility.

## Acceptance and report

All requested files exist, links/paths resolve within the workspace or accurately name the three upstreams, mode is consistent, templates are unmistakably templates, and no guide claims a live service or completed science. Preserve root AGENTS and all Codex-owned evidence. End by reporting created files and remaining limitations. If a tool is denied, report it; do not attempt permission bypass.
