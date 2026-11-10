# STEP-style individual aboveground growth

Status: complete and independently accepted, revision3.

Claude implemented the [bounded assignment](prompt_02.md), then corrected
[review findings](codex_review_02.md) under [revision3](prompt_03.md). The user
explicitly approved candidate-source/specification transfer to Anthropic and
reconfirmed automatic orchestration. Both actual writer launches and exits,
separate raw streams, interrupted attempts and authorization are preserved.

The installed opt-in adapter uses actual daily SYRUP transpiration, species
biomass-support pressure, logistic size growth and weekly/year-end updates.
Total aboveground biomass includes an internal seasonal component removed at
year-end. Native water, species root profiles, existing spatial competition and
legacy defaults are preserved. Traits and climate modifiers are explicit and
uncalibrated; documented temporal/spatial adaptations remain.

- [Independent acceptance](codex_review.md)
- [Verification and exact outcomes](verification.md)
- [Claude report](claude_report_03.md)
- [Source baseline](baseline.json)
- [Current source lock](../../../dependencies/stepwat-growth.lock.json)
- [Installed API guide](../../../docs/stepwat-growth.md)

Final CPU90checks:83passed/7CUDA skips; CUDA90checks:89passed/1intentional skip.
Independent numerical/native-water/calendar/retry/validation oracles pass on both
backends. No upstream source changes, installs, commits, pushes or new field runs.
The new mode has not been calibrated or benchmarked for full-domain performance.
