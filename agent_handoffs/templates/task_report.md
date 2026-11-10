# TASK REPORT TEMPLATE — NOT A COMPLETED REPORT

> This is a template. Angle-bracket placeholders mean the field was not filled in. `NOT RUN` and `NOT VERIFIED` entries are not evidence of completion. Acceptance is recorded only by Codex in `codex_review.md`. Rules: [../AGENTS.md](../AGENTS.md).

Task ID: <task_id>
Revision: <n>
Session UUID: <explicit UUID passed to the CLI, or "not launched via CLI">
Process identity: <PID + start identity if known to the implementer; otherwise "recorded by orchestrator">
Reported by: Claude
Date: <YYYY-MM-DD, timezone>
Status: <complete | partial | blocked> — defaults to `partial` until every check below has an actual result.

## Changes

| File | Change | In allowed writes? |
|---|---|---|
| <path> | <what changed> | <yes / no> |

## Source scope compliance

- Files read: <list>
- Any read outside the task's source scope: <none / list and explain>

## Checks run and results

Fill in only checks actually executed. Leave others as `NOT RUN`.

| Check | Command / procedure | Result | Evidence |
|---|---|---|---|
| <check> | <exact command> | <pass / fail / NOT RUN> | <path to output or quoted line> |

## Acceptance criteria (implementer's view only)

| Criterion | View | Note |
|---|---|---|
| <criterion> | <met / not met / NOT VERIFIED> | <why> |

## Unresolved findings and limits

- <Anything that failed, was skipped, or is uncertain>
- <Constraints that prevented a check or change>

## Interruption / session details

- Interrupted: <yes / no>
- If yes: step reached, what remains, literal error text (verbatim), classification (`scope` / `check_failed` / `tool_error` / `auth` / `network` / `rate_limit` / `interrupted` / `other`).
- Rate limit: <"not reported", or "reported" with reset time and timezone verbatim if supplied, otherwise "reset: unknown">. DNS/auth/network/test failures are not rate limits.
- Derived resume time (optional, clearly derived): <only if computed from a reliable literal reset; otherwise "none">
- Partial work preserved on disk: <yes / no>
- Resumption: by active orchestration resuming this session UUID within its authority; no unattended resumption is promised unless a live scheduler has been verified and recorded.

## Version control and setup status

- All changes left uncommitted: <yes / no>
- Git init / commit / push / dependency install / model run performed: <none, or list with the task authority that granted it>
