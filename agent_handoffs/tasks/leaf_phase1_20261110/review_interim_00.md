# Interim independent review — uptake snapshot 00

Verdict: **correct**. Not phase acceptance; Claude revision 1 remains active on
the rest of its existing scope. Corrections will be dispatched after it is reaped.

Snapshot: `review_snapshots/uptake_00.py`, SHA256
`bcda2e910dd5e48e39856b2e9e19fcf3c13e2e61c81498a7837a2b44978ce335`.
Evidence: `uptake_interim_probe.{launch.json,stdout.log,stderr.log,exit.json}`;
independent exact reference: `scalar_oracle.py`. Seven cases pass on both NumPy
and Numba, including the user's 75% example, local proportional competition,
nonseparable roots, zero demand/population and native partial acceptance.

## Confirmed corrections

1. **High — nonfinite aggregate accepted.** Two distinct resources with finite
   demand/supply `1e308` each produce infinite totals. `accept` returns normally
   with residual NaN and `closure.closed == false` because `abs(NaN) > inf` is
   false. Reject nonfinite intermediate arrays, aggregate packets, bounds and
   closure values before returning a plan or feedback; closure must be finite
   and true. Include scatter-overflow and aggregate-overflow tests.
2. **High — feedback admission understates scratch.** On a warmed Numba path
   with P=E=200,000, R=1, estimated additional feedback bytes are 4,800,025,
   while tracemalloc measures an 8,203,504-byte operation peak. Existing map,
   plan and caller arrays are excluded from that measured peak, as they are
   from the operation estimate. The declared 14,400,057-byte full cap accepts
   despite more than 3 MB of unreserved array-scale scratch. Fuse plant result
   validation/arithmetic or account for all simultaneous P-sized temporaries.
   Audit equivalent NumPy/CUDA and map/plan validation paths too.
3. **High — admission after materialization on two public paths.**
   `with_backend(..., max_bytes=0)` first calls `host_arrays`, materializing
   8,003,956 traced bytes before refusal. Passing Python sequence map inputs
   with zero edge/byte budget similarly converts them to arrays first
   (6,403,098 traced bytes). Preflight metadata and budget before such copies,
   casts or transfers, or explicitly require already-materialized arrays and
   reject unsupported input types without conversion. Include source+destination
   coexistence in conversion scope; do not weaken bounds.
4. **Medium — mixed bool IDs lose type information.** IDs `[True, 2]` are
   converted to an integer array and accepted. Refuse bool elements before
   conversion or enforce strict array inputs. The same audit applies to offsets,
   resources and demands.
5. **High — overflowing CSR differences admit invalid offsets.** With E=0,
   offsets `[0, INT64_MAX, INT64_MIN+1, -1, 0]` are accepted: `diff` wraps to
   positive counts even though offsets are negative/out of range. Evidence:
   `offset_probe_00.json`. Bounds-check every offset in `[0,E]` and monotonicity
   by direct comparisons before subtracting or computing counts. The construction
   probe intentionally did NOT run scatter/gather: compiled kernels on this map
   would have invalid bounds. Add refusal tests without executing invalid maps.

## Static findings requiring additional verification

- RootMap does not record/check CUDA device identity. Arrays from a different
  current device can pass namespace checks, and compiled CUDA modules are
  cached globally rather than by device/context. Enforce one device throughout
  construction, proposal, feedback and conversions; never implicitly switch.
- `_take_ownership=True` can retain external aliases. Verify that internal root
  builder use cannot admit caller-writable geometry through a public path.
- Rounding-level accepted excess is clipped to the proposal. SYRUP's actual
  debit is authoritative; verify/report any difference explicitly rather than
  claiming equality to native accepted water after silently changing it.
- Root-row tolerance currently admits sums slightly above one and later clips
  satisfaction. Audit the normalized-root contract and demand/request closure;
  clipping may conceal an inconsistent row rather than mere summation roundoff.
- Daily CUDA validation/summary code currently converts many separate reductions
  to Python scalars. Batch finite/budget/closure diagnostics into small packets;
  measure the actual synchronization cost rather than assuming scalar transfers
  are free. No population readback is currently visible in the daily path.
- The root builder's NumPy counting pass materializes per-plant overlap arrays
  before post-count edge admission. Its production compiled path counts without
  those edge arrays. All supported paths need truthful pre-allocation bounds,
  including reference helpers and coexistence with geometry/overrides.
- Multi-file top-level metadata equality uses Python `==`. Values such as notes
  `1` and `1.0` compare equal but serialize differently, so retaining the first
  document's value can make the merged hash depend on path order. Compare the
  canonical representation or deterministically omit differing metadata, without
  changing single-file hash compatibility.

The probes' exit 0 means the probe program completed and recorded its findings;
it does not mean these candidate paths passed acceptance.
