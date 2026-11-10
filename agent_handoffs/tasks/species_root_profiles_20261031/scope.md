# Species-specific root-layer profiles

User request: implement different species/PFT rooting profiles, with grasses
weighted toward surface layers, shrubs toward bottom layers, and roots in all
layers. Extend only the existing synthetic water/growth benchmark. Keep daily
uptake and weekly biomass/geometry/demand updates and conserve water.

Claude is the sole implementation author for benchmarks/daily_water_growth.py,
tests/test_daily_water_growth.py and benchmarks/daily-water-growth.md, through
generic-only no-tools code generation. Codex owns baseline snapshots, prompts,
installation after dispatch completes, review, tests, performance fixtures,
measurement wrappers, docs/species-root-profiles.md, README updates, a historical
notice in docs/daily-water-30year.md, and active handoff/context records.

Only the generic benchmark source/tests/documentation and task specification may
be submitted to Claude. Exclude private project briefs/outlines, ECOTONE source,
other local scientific code/data, credentials and upstream repositories.

Use the existing interpreter. No upstream edits, installs, environments, Git
actions, population truncation or resource-cap increases. Original AGENTS.md and
accepted task evidence/results are immutable. Current task records are local.

Acceptance: configurable normalized strictly positive profiles; explicit compact
per-plant profile IDs for additional named species; same layer weights in CPU/GPU
request and actual uptake; profile IDs/table validated before large allocation;
unchanged sparse horizontal graph; updated exact memory ledger and bounded
scratch; focused analytical water tests, invalid-input/refusal checks, weekly
calendar/split-run checks, real CPU/GPU parity, and bounded 30-year GPU runs at
representative and dense populations. Preserve a uniform-profile compatibility
control and report new performance separately from historical equal-layer runs.
