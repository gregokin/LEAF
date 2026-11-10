# Phase 0 — focused correction revision 3

Continue the exact approved Phase 0 session. Same 17 candidate write paths,
ordinary Read/Edit/Write only, no shell/agents/network tools/installs/Git or
execution claims. Preserve copied legacy numerical modules. Codex independently
executes checks. Make the focused changes below; do not rewrite unrelated code.
Update report.md honestly and leave everything uncommitted.

Additional READ-only review artifacts allowed for this revision (all task-local,
not candidate writes): candidate_phase0_02.stderr.log,
checkpoint_state_02.stderr.log, checkpoint_memory_replay_02.stdout.log,
cover_memory_replay_oracles.py, transfer_preflight_oracle.py,
transfer_preflight_02.stdout.log. Prior permitted review/source scope still
applies. Do not read any other outside files.

## What already passes

Your mixed-size exclusion algorithm now works well. Independent 128²/.25m
cases: allowed 3243 plants ~0.10s, disallowed 3145 plants ~0.18s, exact global
nonoverlap confirmed; 512²/.25m 51893-plant setup ~1.95s. All use max_attempts
REDUCED to 50000, independent uniform-point cover checks and original tolerance.
All 12 independent ownership/ID/bridge oracles pass. Separate raster and tiled
scratch admission also passes a tighter independent tracemalloc test (64KiB
metadata allowance). Preserve these corrections; do not raise bounds.

Final revision 2 Phase 0 suite: 43 tests in ~2.6s, 3 failures + 8 errors.
Most are ONE serialization regression, plus one numerical error conversion and
one incorrect test expectation. Fix these, plus config replay and transfer
preflight, then add focused regression coverage.

## 1. JSON metadata is valid and necessarily has quotes

state.py now rejects a double-quote ANYWHERE in the entire CSV string, both in
plant_csv_text and parse_plant_csv. The first line is JSON metadata, which ALWAYS
has quotes. Consequently every default CSV write/roundtrip, snapshot and CLI
initialization fails with `internal error: a quoted field was produced`.

Remove the whole-document quote bans. Parse metadata separately, then use the
already-added csv.reader(..., strict=True) on CSV rows. Ordinary valid CSV quoted
fields should be accepted (spreadsheet writers commonly quote them); malformed/
unterminated quotes still fail through strict parsing and value validation.
Do not strip metadata or disable its checks to make round trips work. Keep the
exact common CSV schema, FP64 roundtrip, clock/catalog/units/domain metadata and
initial-only SURFACE checks. Update the no-quotes policy text/tests/docs to
standard strict CSV. Add an actual successful metadata-bearing file roundtrip,
valid quoted numeric/status fields, and rejected unterminated quoted last field.

## 2. Guard numerical errors without rejecting valid finite results

expected_crown_area_m2(1e200,1e200) raises raw OverflowError at Python `**2`,
before its finite guard. The test correctly expects CatalogError. Catch/convert
arithmetic overflow (similarly for lognormal parameter derivation where needed)
or compute safely and raise the declared domain error BEFORE allocation.

The inverse-overflow test is mathematically WRONG: with diameter exponent b=2,
the inverse exponent is 1/2, so a finite huge diameter can return finite biomass.
Do not change correct allometry to reject that valid result. Use b=0.5 for the
inverse-overflow test (inverse exponent2), or another actually overflowing
combination. Keep the b=2 forward-overflow check and finite-output checks.

## 3. Make recorded generation configuration directly replayable

The report['generation_config'] contains derived ordering_convention and
per-species diameter_distribution/expected_crown_area/initial_count fields that
validate_generation_config rejects. An independent JSON roundtrip then calling
generate_population(catalog, report['generation_config'], surface=...) currently
fails at unknown key ordering_convention.

Store an accepted SOURCE configuration in generation_config (schema/domain/seed/
overlap/tolerance/bounds/species input diameter + age rules/time). Keep all the
derived information in report fields outside that config. Alternatively provide
an explicitly validated replay API, but a directly usable saved config is
preferred and simpler. Do not weaken unknown-key validation or discard RNG/
NumPy-version/ordering provenance. Test exact by-array regeneration from a JSON
serialized/deserialized saved report config with the same catalog/surface.

## 4. Validate device vector metadata before copying

runtime._float_array currently calls .get() before checking B/E shape. A wrong
full-domain array would be copied to host before rejection. The independent
transfer_preflight_oracle.py uses a fake device vector with shape(3,2000000)
and get() that raises if called. Both bad-B and bad-E probes fail; the valid
P-vector transfer-once case passes.

Inspect shape/ndim/dtype of device-backed arrays against known P BEFORE .get().
Refuse missing/invalid metadata clearly; still validate actual received vector
values after transfer. No imports of CuPy/native models in this module. Preserve
exactly one requested host read of each valid B/E vector, only at capture time.
Add these tests; no real GPU dependency is needed for the protocol check.

## 5. Enforce the stated fresh-driver seasonal restore contract

restore_seasonal_mass's docstring requires a FRESH STEP driver but the function
does not enforce this. Before any mutation, validate the population and the same
ID/species/live-plant association used in capture, and require driver day==0
(and coupled accepted days==0 when applicable). A midrun call must raise and
leave driver E unchanged. Retain E<=actual driver B checks and by-ID restore.
This does not add coupled restart; do not reset a driver clock to permit it.

## 6. Finish consistently

Keep the successful corrected algorithms/budgets/default-safe canopy publication
and import isolation. Update the focused tests and relevant docs/report. Codex
will rerun the complete Phase 0 suite, independent oracles and installed legacy
checks. All new methods remain Phase 0 infrastructure, no scientific growth/
mortality/damage law or upstream source change.

Packaging note: both model envs lack setuptools/wheel; the existing SYSTEM
Python has setuptools68.1.2 but no wheel. Codex will exercise the real sdist
backend in /tmp then inspect/import the unpacked package using the NumPy env.
No install/new env/fetch is allowed or needed. Do not change build backend just
for this host and do not claim wheel checks ran.
