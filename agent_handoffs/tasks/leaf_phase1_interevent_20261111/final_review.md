# Phase 1 independent final disposition

Verdict: **accept** the composition A4 + B4 + C3 + D3, frozen in review_03 and
installed byte-for-byte in LEAF. This accepts the bounded computational mechanics,
not field calibration or a full MASY storm/evaporation/restart adapter.

## Result

Species traits are co-located; roots are fixed during each growth period and may
differ by species and individual. Per-resource proportional competition never
compensates unmet demand from elsewhere. The real SYRUP operator removes accepted
water exactly once; its actual feedback drives total delivered/requested stress.
STEP-style planned dry-AGB growth receives this stress once. Actual support and
maintenance deficit have kg*year units and are not a biomass-production yield.
Dated daily/partial uptake and weekly-or-shorter growth use explicit temperature,
PET and caller-prepared soil forcing. Seasonal turnover, exact calendar boundaries,
mutually exclusive transport gates, ages and complete LEAF-native checkpoints work
without hidden catch-up growth. Both consumers retain the common canopy geometry.

## Independent verification

- review03_cpu: 365 tests, 345 pass / 20 CUDA skips, exit0.
- review03_cuda: all88 session/restart tests pass on actual selected GTX1080Ti.
  Earlier A4/B4/C3 allocator/root/native/growth CUDA qualifications remain valid:
  those final component production bytes are identical in this composition.
- review03_guards: all8 independent guard cases pass; late-clock errors, invalid
  overrides, nonfinite pending totals and false reported water closure refuse with
  the accepted boundary intact. Actual native calls and valid retries are checked.
- review03_native_smoke: actual ten-day native water run across leap day, exact
  whole-state / water equivalence after a midweek restart.
- review03_example: compiled-CPU generated-population example, ten days, CSV and
  checkpoint output, restore successful; all coefficients / forcing are fixtures.
- Final CUDA weekly-root memory probe: actual external-array lower bound29938232B
  <= declared29938488B at100000plants. Global conservative stage peak62738456B;
  capacities no longer masquerade as peaks. No unknown frees or live allocations
  remain. D2 daily and snapshot probes fit their corrected bounds; D3 preserves
  those paths and adds conservative restore/rebuild accounting.
- Source scope: all four final writers reaped, structured success; each final scope
  audit has zero violations / tool errors. Original failures and partial Fable
  outputs remain archived; no claim they passed. C3 tests/report-only change
  confirmed that the closure guard was already present in C2.

## Corrections and limits

Earlier D1 guard failures and snapshot/daily memory undercounts, and D2's measured
2399776B external-array undercount during a rebuild, are corrected. Final strict
source/native/root guards remain enabled. Global budgets count host+device array
data, not interpreter, JIT/context or pool overhead. Weekly geometry is host based
with explicit B/E readback and CSR upload; no claim it is GPU resident. The current
rate-limited Euler adaptation does not activate the older fitted295/year grass
rate. Examples are explicitly uncalibrated, not a replacement parameter fit.

Root AGENTS.md is unchanged. Legacy drivers/locks are preserved. No installation,
new environment, upstream edit, transport scenario or resource-limit increase was
performed. MAPLE HEAD advanced through other agents' work; final identities are
recorded in installed_phase1.json. Its numerical-policy source is unchanged from
the earlier9f466d2 qualification; actual SYRUP branch and dirty native hashes match.

## Publication

User explicitly authorizes whole-phase commit/push, followed by Phase2 and Phase3.
No component-only publication is represented as phase completion. Installed files
and the source/environment lock are recorded in installed_phase1.json and
../../../dependencies/phase1-native-growth.lock.json. Actual Git publication is a
separate recorded operation after this review.

Publication whitespace check: all LEAF-written source/tests/docs/evidence pass. Exact archived external STEP reference C files retain original trailing whitespace and are excluded from that formatting-only check; originals and hashes are preserved. See phase1_whitespace.json.
