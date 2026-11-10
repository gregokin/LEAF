# Bootstrap Verification

Task: `bootstrap_20261030`. Result: **pass for documentation setup**.

## Executed checks

- `claude --help` and `claude --version`: installed options checked; saved as `tasks/bootstrap_20261030/claude_help.txt` and `claude_version.txt` (2.1.286).
- Python standard-library structural check: all 16 expected guides/control files exist, 48 local Markdown links resolve, files have final newlines and no trailing whitespace, and the original root `AGENTS.md` matches its pre-setup bytes. Exact counts and timestamp are in `tasks/bootstrap_20261030/validation_results.json`.
- `bash -n`: syntax-checked the Bash example extracted from `agent_handoffs/README.md` on stdin; exit 0. The example was not executed against project files.
- Python JSON/hash checks: successful launch/exit pairs 03 and 04 have matching session IDs, matching prompt hashes, recorded PID/start identity, separate raw output files, exit 0, `is_error: false`, no permission denials, and reaped children. Resume used the same exact session. Full executed CLI argument arrays are in `launch_03.json` and `launch_04.json` under the task archive.
- Submission scope: successful calls used safe mode in an empty temporary directory with no tools. Only generic template text/corrections were supplied. The detailed-outline retry 02 was rejected before launch; no `launch_02.json` exists.

## Preserved failures and limits

Attempt 01 failed with DNS/API error `EAI_AGAIN` and exited 1; this is a network failure, not a rate limit. Attempt 02 was rejected by automatic approval review for detailed-outline egress. Generic template calls 03 and 04 succeeded as a safer alternative. Raw evidence and dispositions are retained.

These checks validate documentation and the generic Claude dispatch/correction/resume path. Claude did not inspect scientific source or the local brief. No model, scientific reproduction, CPU/GPU performance, integration, or full ecological build was tested. No environment, package installation, Git initialization, commit, push, or upstream edit was performed. Source identities must be established before later implementation reliance.
