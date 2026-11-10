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
