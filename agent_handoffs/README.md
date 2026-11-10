# agent_handoffs

Shared handoff area for the Codex ↔ Claude workflow. Rules: [AGENTS.md](AGENTS.md). Claude entry point for auto-discovering sessions: [../claude.md](../claude.md).

## Files

| File | Owner | Purpose |
|---|---|---|
| [orchestration_state.md](orchestration_state.md) | Codex | Mode (automatic/manual) with date, baseline reference, scope holder, pending errors, open items. |
| [current_prompt.md](current_prompt.md) | Codex | The single current bounded prompt, an explicit pointer to an archived prompt in `tasks/<task_id>/`, or an idle marker. Templates and history are not assignments. |
| [claude_report.md](claude_report.md) | Claude | Report for the current task, shaped like [templates/task_report.md](templates/task_report.md). |
| [codex_review.md](codex_review.md) | Codex | Verdict (accept / correct / stop). Acceptance is recorded here. |
| [verification.md](verification.md) | Codex | Independent re-run of checks: commands, output locations, error classification. Evidence, not acceptance. |
| [decisions.md](decisions.md) | Codex (Claude proposes) | Durable decisions and rationale. |
| [upstreams.md](upstreams.md) | Codex | External sources, references, dependencies. |
| [project_brief.md](project_brief.md) | User / Codex | Project scope and constraints. Read by Claude only when a task's source scope lists it. |
| [templates/](templates/) | — | Unassigned prompt and report templates. |
| `tasks/<task_id>/` | Both (append-only after close) | Immutable prompt, `baseline.json` or commit reference, launch/exit records (JSON or numbered artifacts), raw stdout/stderr, report, review, verification. |

Files listed here but absent are supplied locally by the orchestrator. Missing required state or prompt stops dispatch pending orchestration.

## Cycle

1. **Dispatch** — Codex records the baseline, writes the bounded prompt (with source scope and, for safe-mode launches, the governing instructions) to `tasks/<task_id>/prompt.md`, points `current_prompt.md` at it, assigns the write scope in `orchestration_state.md`.
2. **Implement** — Claude works within allowed writes and source scope, leaves changes uncommitted, writes `claude_report.md`.
3. **Review** — Codex reads report and diff, checks write and read scope. Read-only review may overlap implementation.
4. **Verify** — Codex re-runs each check independently into `verification.md`.
5. **Verdict** — Codex writes `codex_review.md`. `correct` loops to step 1 with a revision; `accept` sets the prompt idle; `stop` records why. Execution errors pause the task without changing the user's mode preference.

Evidence for every real launch: exact command and cwd, session UUID, PID plus start identity, start/end timestamps with timezone, raw stdout and stderr files, exit status and reaping record. Without these the launch is unverified.

## CLI example (illustrative only)

An **example**, not a complete runner. Verify every flag against the installed CLI (`claude --help`) before use; names and behaviour vary by version.

```bash
# EXAMPLE ONLY — not a complete runner. Verify flags with: claude --help
TASK_ID="t-0001"
EVID="agent_handoffs/tasks/${TASK_ID}/attempt-01"
PROMPT="agent_handoffs/tasks/${TASK_ID}/prompt.md"   # immutable archived prompt: governing instructions + read scope inside
SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
mkdir "${EVID}" || exit 1  # a new attempt directory; preserve earlier evidence
printf '%s\n' "${SESSION_ID}" > "${EVID}/session_id"

claude --print --output-format json \
  --safe-mode \
  --strict-mcp-config \
  --tools "Read,Glob,Grep,Write,Edit" \
  --allowedTools "Read,Glob,Grep,Write,Edit" \
  --permission-mode dontAsk \
  --permission-prompts none \
  --session-id "${SESSION_ID}" \
  < "${PROMPT}" \
  > "${EVID}/stdout.json" \
  2> "${EVID}/stderr.log"
printf '%s\n' "$?" > "${EVID}/exit_code"
```

Notes:

- `--tools` restricts which tools exist in the session; `--allowedTools` pre-approves their use. Set both to the same list. `--strict-mcp-config` ignores MCP servers not explicitly configured.
- `--safe-mode` prevents unintended customization, hooks, and automatic context loading. It skips auto-discovered `CLAUDE.md`/`claude.md`, so the archived prompt must explicitly carry the governing instructions (or verbatim excerpts of [AGENTS.md](AGENTS.md)) and the approved read scope.
- The prompt is passed by stdin redirection from the immutable archived file, not by command substitution in argv.
- The UUID comes from Python's standard library; `uuidgen` is not assumed.
- Not shown and still required for a real launch: capture of PID plus start identity, start/end timestamps, and an exit/reaping record (see [AGENTS.md](AGENTS.md#process-and-session-identity)).
- `Bash` is not granted. If a later task needs it, grant it in that task's prompt with exact commands.
- Inspect `stdout.json` for structured errors before trusting the exit code.

## Daily water/growth performance evidence

`tasks/daily_water_30year_20261030/` preserves the bounded Claude assignment,
source baselines, revisions, process records, independent verification and
14actual30-year runs. See [results](../docs/daily-water-30year.md) for measured
CPU/GPU wall time, live/pooled memory, conservation and reproduction commands.

## Species-root-profile extension

`tasks/species_root_profiles_20261031/` records the real Claude implementation
session, bounded revisions, immutable baselines, independent analytical and
CPU/GPU checks, source identities and updated performance evidence. The
[profile follow-up](../docs/species-root-profiles.md) describes configuration,
fixed vertical traits, daily/weekly update order and the memory ledger. Earlier
equal-layer measurements and accepted evidence remain historical records.

## Native transpiration coupling

[tasks/coupled_transpiration_20261031/](tasks/coupled_transpiration_20261031/) records
the actual source branch, scoped Claude session/revisions, independent reviews and
reproductions, CPU/GPU tests, new-file installation guards and six bounded coupled
capacity cases. The [guide](../docs/coupled-transpiration.md) gives installed API
usage, source-lock verification, units/order, measured memory/timing and remaining
storm/MASY/remapping limitations. Acceptance is in the task's Codex review.

## STEP-style aboveground growth

[tasks/stepwat_growth_20261031/](tasks/stepwat_growth_20261031/) records the bounded
Claude implementation, explicit transfer authorization, correction review,
independent scalar/native-water/calendar oracles and CPU/CUDA checks. The
[guide](../docs/stepwat-growth.md) documents the opt-in trait schema, weekly
adaptation, seasonal aboveground component and calibration limits. Current
runtime identities are in [the growth lock](../dependencies/stepwat-growth.lock.json).
