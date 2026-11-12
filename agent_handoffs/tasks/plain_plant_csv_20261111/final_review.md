# Final independent review — accept

User requirements: plant files contain no embedded metadata; contextual metadata lives elsewhere; generated plants have independent uniform initial ages from zero to the species life expectancy; commit and push when complete. Codex scoped/reviewed/verified, Claude was the sole candidate implementation writer under normal Read/Edit/Write permissions. All four launched processes are reaped. The initial default-model usage limit was recorded; Sonnet continuation and both implementation corrections completed successfully.

## Result

Every normal plant emitter writes a plain nine-column CSV with a separate `<stem>.metadata.json` companion. Automatic readers verify byte count, SHA-256, catalog/domain/time/row count and SURFACE context. Legacy embedded plant input remains readable, never emitted. No-clobber, rollback and interrupted-pair rejection are exercised. Checkpoints now use layout 4 (CSV + companion + arrays + manifest), published atomically as a directory; layouts 1–3 require their matching earlier runtime and are not silently migrated. Metadata memory admission scales with its conservatively counted JSON size; Python encoder/control overhead follows the documented exclusions.

The generator defaults to `uniform_to_life_expectancy` when ages are omitted; shipped generation files explicitly select it. Bounds resolve from the species catalog, remain species-local, and are recorded outside the CSV. Numeric/constant/ranged controls remain for explicit fixtures. Size, cover, placement, conservation and mortality kernels are unchanged. Explicit plant-list/restart ages and age-zero recruits retain their behavior. Current trial bounds are 0–5475 days grass (15 yr) and 0–29200 days shrubs (80 yr), based on provisional existing traits. This sampling bound is not a deterministic death age.

## Independent evidence

* `cpu_suite_03.stderr.log` and `.exit.json`: 576 tests, **531 executed passes +45 expected CUDA skips**, exit 0. Actual native SYRUP restart/withdrawal, failure rollback, new plain-file/companion guards, large-provenance admission, uniform-age distributions, units, seed replay, catalog-origin records, CLI and existing newborn behaviors passed. No GPU jobs/benchmarks were needed for host-only serialization/initial sampling.
* `scope_audit.json`, `final_scope.json`, `final_source_manifest.json`: bounded tool/source/write checks, preserved science files; AST comparison confirms traits/interevent/package edits outside IO/generator are documentation only. Root AGENTS.md preserved.
* `final_case_verification.json`: all case JSON/TOML parse; actual native case validation resolves species-linked bounds without generating the full case. The saved preview has **25,589 plants at time 0**, all nine-field rows and exact header/plant-row bytes preserved. PNG, terrain, species catalog, original render provenance and original initialization report remain unchanged.
* `preview/plant_files_migration.json`: current plain CSV/companion hashes and immutable original render-provenance identity. The existing picture/ages belong to the earlier fixed-age initialization; current generation settings apply to a future requested initialization. No actual case was regenerated, rendered or advanced in this task.

Claude's initial mutation fixture incorrectly attempted to overwrite deliberately corrupted unrelated-schema metadata; the production guard was retained and the fixture corrected. Review replaced a fixed metadata-memory guess with a size-aware bound and removed an unnecessary rerender feature before acceptance. The final tests exercise these corrections.

Codex's final operator documentation cleanup updates checkpoint layout labels and companion paths in the trial case manifest/parameter/output/input notes. No additional runtime code or scientific parameters were changed after the final CPU suite; only JSON/TOML parsing and static file checks were repeated for those labels. Full annual transport coupling and ecological calibration remain separate work. Prior reviewed trial-input/preview deliverables are included in the authorized publication.

Publication is authorized and will be recorded after actual Git operations; this review records acceptance, not a claim that a push has already occurred.
