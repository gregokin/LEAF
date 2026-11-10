# Biomass-specific daily water use literature survey

User requests additional literature values of water use per kg aboveground
biomass per day, explicitly emphasizing DRY biomass. Codex owns public-source
research, arithmetic, evidence and documentation; no implementation assignment
or Claude launch is needed for this literature-only task.

Allowed writes: this new task directory; docs/biomass-specific-water-use.md;
link/clarification in README.md; root handoff current_prompt.md,
orchestration_state.md, codex_review.md and verification.md. Read LEAF guidance,
existing published study analysis and public primary literature/data. Do not
change runtime, parameter candidates, source locks or accepted task archives.
No upstream edits, installations, new environment, model runs, contact, commits
or pushes. No private payload transfer.

Acceptance: directly verify numerical observations and units in primary
sources; distinguish transpiration from ET, standing aboveground dry mass from
leaf/total/final dry mass and newly produced dry mass; mark derived values and
normalization windows. Do not call daylight-hour measurements full daily fluxes
or divide cumulative water by final mass without labeling that proxy. State
age, environment and uncertainty limitations. Preserve source identities and
all existing files outside the allowed scope.
