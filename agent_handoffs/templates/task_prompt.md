# TASK PROMPT TEMPLATE — NOT AN ASSIGNMENT

> This is a template. A prompt with angle-bracket placeholders or status `UNASSIGNED` assigns nothing. A filled copy archived under `tasks/<task_id>/prompt.md` is immutable once dispatched; `current_prompt.md` may hold it inline or simply point to it. Rules: [../AGENTS.md](../AGENTS.md).

Status: UNASSIGNED
Task ID: <task_id>
Revision: <0 for first dispatch; n for correction n>
Dispatched by: <Codex>
Date: <YYYY-MM-DD, timezone>

## Governing instructions

<For safe-mode launches, embed or quote verbatim the applicable sections of ../AGENTS.md here, since auto-discovered files are not loaded. Otherwise: "auto-discovered via claude.md".>

## Task

<One paragraph. What to produce or fix. Bounded; one implementation scope.>

## Baseline

<Actual HEAD plus dirty/untracked source manifest and relevant dependency locks, or tasks/<task_id>/baseline.json with actual paths, sizes, hashes when Git is unavailable. Include any pre-existing changes to preserve.>

## Requirements

- <Requirement 1>
- <Requirement 2>

## Source scope (may be read or shared)

- <file or directory>
- <file or directory>

Include governing instructions and control files here if Claude must open them; safe-mode prompts may embed approved instructions instead. Nothing outside this list is readable for this task. Codex has already read the project context and supplied what is needed.

## Excluded payloads

- <data, briefs, or files that must not be read, copied, or quoted>

## Allowed tools and directories

- Tools: Read, Glob, Grep, Write, Edit.
- Bash: <none, or the exact commands permitted>.
- Working directories: <list>.

## Allowed writes

- <path or glob>
- agent_handoffs/claude_report.md
- agent_handoffs/tasks/<task_id>/ (report artifacts only)
- <documentation under agent_handoffs/ only if explicitly listed here>

## Exclusions

- <paths that must not change>
- Orchestration control files, immutable baselines/prompts, and historical evidence, unless listed above.

## Acceptance criteria

- <Observable outcome, checkable by a command or a named output>

## Checks to run

- <exact command or procedure> — expected: <result>

## Deliverables

- Changed files, left uncommitted.
- `agent_handoffs/claude_report.md` following [task_report.md](task_report.md).

## Stop conditions

- All acceptance criteria met and all checks run; or
- A check fails and cannot be fixed within allowed writes; or
- A needed change or read falls outside the granted scope; or
- A tool, auth, network, or rate-limit error occurs (record the literal text, preserve partial work, use short interruptible waits, do not retry indefinitely).

## Version control and setup

Leave all changes uncommitted. No Git init, commit, push, stash, revert, dependency install, or model run unless this task grants it explicitly: <none granted>.
