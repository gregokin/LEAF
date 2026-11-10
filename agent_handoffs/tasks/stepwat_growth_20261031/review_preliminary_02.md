# Preliminary independent review — revision2

The numerical kernel matches seven independently prepared scalar cases on real
Numba and CUDA paths. Native coupled E/T/two-area accounting, year-end partial
period, failure/retry and next partial week pass on NumPy and CUDA. These probes
precede the complete candidate report and are not acceptance.

Items to resolve after the initial writer finishes:

1. `validate_temperature_factor` promises HOST scalar/[S] input but calls `.get()`
   on a device-like value before checking its shape. An invalid P-sized forcing
   can trigger a full device-to-host copy before refusal. Enforce the host-only
   API and reject device-backed values without invoking `.get()`. Check host
   shape/length and scalar entries before conversion so malformed nested input
   cannot materialize an arbitrary matrix. Add refusal-before-transfer tests.
2. Biomass summaries contain domain sums but label their units "kg ... per
   individual". Keep the individual state unit distinct from aggregate domain
   totals, all of which have units kg. Clarify report fields and documentation.

No source corrections made by Codex: Claude remains the candidate writer.
Review legacy compatibility, completed tests/docs and full memory guard behavior
once the first candidate is stable.
