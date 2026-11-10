# Repository Guidelines

## Project Structure & Module Organization

This workspace is an initial scaffold: no application source, tests, assets, or dependency manifests are present. The `.git/` directory does not currently contain usable Git metadata.

When adding the first implementation, organize application code under `src/`, automated tests under `tests/`, and static resources under `assets/`, unless the chosen framework requires another layout. Keep related modules together and document the actual structure in `README.md`.

## Build, Test, and Development Commands

No build, test, or development commands are configured yet. When introducing tooling, add the dependency manifest, lockfile, and exact setup and execution commands to `README.md`. Update this guide with the supported commands at the same time.

- `rg --files --hidden -g '!.git'`: list workspace files, including configuration files.
- `git diff --check`: check changes for whitespace errors once Git metadata is available.

## Coding Style & Naming Conventions

Use spaces, LF line endings, and a final newline. Use two spaces for Markdown list nesting. For source code, adopt the chosen language's standard formatter and commit its configuration before establishing additional style rules.

Prefer descriptive names. Use lowercase, hyphen-separated documentation filenames, except established names such as `README.md` and `AGENTS.md`. Follow language and framework conventions for source filenames.

## Testing Guidelines

No test framework or coverage threshold is configured. Introduce an appropriate framework with the first executable code. Cover new behavior and add regression tests for bug fixes. Mirror source modules in `tests/` when practical, and use test names that describe expected behavior. Document the test command and any required fixtures.

## Commit & Pull Request Guidelines

No commit history is available to establish an existing convention. Use short, imperative commit subjects, such as `Add configuration loader`, and keep each commit focused.

Pull requests should explain the purpose and behavior changes, link relevant issues, and report verification commands and results. Include screenshots for visible interface changes. Document required configuration and keep credentials, local environment files, and generated artifacts out of commits.
