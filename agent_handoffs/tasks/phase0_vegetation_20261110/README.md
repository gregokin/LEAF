# Phase 0 vegetation — accepted revision 3

Start with [independent acceptance](codex_review_03.md) and
[verification](verification.md). The installed implementation and examples are
documented in [the Phase 0 guide](../../../docs/phase0-vegetation.md) and
[MAPLE handoff](../../../docs/maple-leaf-integration.md).

- `baseline.json`: immutable original 1,796-file identity before dispatch.
- `prompt_01.md`, `prompt_02.md`, `prompt_03.md`: bounded assignments/corrections.
- `launch_01.json` through `exit_03.json`: exact Claude session/process records;
  separate raw streams remain locally available. `claude_tool_audit.json` records
  scoped Read/Edit/Write paths and structured success for all three revisions.
- `revision_01/`–`revision_03/` and matching manifests: frozen review trees.
- `codex_review_01.md`, `codex_review_02.md`: historical correction verdicts.
- `installation.json`, `scope_final.json`: installed identity and preservation.
- `source_identity.json`, `upstream_interface.json`, `final_upstream_git.json`,
  `maple_runtime_identity_current.json`: actual dirty upstream identities.
- `*.launch.json`, separate `*.stdout.log` / `*.stderr.log`, `*.exit.json`:
  independent check commands/results/reaping. See verification for dispositions.

Codex's oracle scripts remain here for reproduction. Use absolute cwd paths with
`run_check.py`; select an available physical GPU explicitly for CUDA checks.
Neither this index nor the handoff controls assign new implementation work.
