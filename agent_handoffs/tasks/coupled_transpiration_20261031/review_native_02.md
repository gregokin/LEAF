# Native revision-2 review — correct

Independent reviewers agree on three material issues. Tiny immutable-snapshot
CPU reproductions in native_review_02.* confirm all three:

- Allocation before memory refusal: `_validate_packet` creates N-sized maxima
  and where arrays before `_admit`; prepare creates saturation arrays first.
- Negative extra reserves bypass zero available bytes; public helpers silently
  cast invalid scalar sizes/reserves. Use strict non-Boolean positive sizes,
  nonnegative reserves and availability readings, matching native conventions.
- Native T storage budgets include inactive cells whereas native hydrology and
  storage_m3 exclude them. Example active [True,False], water [.02,.03] m:
  true active storage .02 m3; T reports .05. Combined storage is then derived
  by cancellation of incompatible totals rather than directly from final
  authoritative active storage.

Also correct repeated validation/result host packets (two per withdrawal,
three T-related plus hydro for combined), omitted layer-parameter host bytes,
and total_bytes reserve reporting. Native per-cell uptake law, units, time and
current-device checks otherwise match the contract.

No production source has been installed. Source snapshot and raw output retained;
classification `check_failed` (review), no scope expansion or new physics needed.
