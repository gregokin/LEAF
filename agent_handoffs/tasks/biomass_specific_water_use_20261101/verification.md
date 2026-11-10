# Research verification

Codex inspected primary papers and released observations; no implementation
assignment, Claude launch, package installation or native/model scenario.

## Numerical and scientific checks

1. Public savanna data tag v1.0.0 resolves to commit
   `862abe68c7a9f1eb6397c36516bca35c9d445298`. Seven downloaded source/data files
   match the Git blob identities in the retrieved tree response. SHA256 and
   response identities are archived in the retrieval manifests and
   [source identities](source_identities.json).
2. Primary paper explicitly describes harvest of aboveground dry biomass and
   sealed-pot gravimetric measurements. Aboveground shoot dry mass is the
   denominator; leaf and root masses are not substituted. Published results
   use leaf-normalized hourly metrics, while this task independently sums
   24-hour water loss and divides by fixed harvest shoot dry mass.
3. [Derivation](derive_daily_rates.py) processed 21,311 rows. Two ambiguous IDs
   (`Trigra1`, `Trigra2`) contain 396 excess duplicate-hour records; the latter
   also has 326 inconsistent supplied shoot-normalization values. Both entire
   IDs are excluded, with raw observations retained. Three missing-time rows
   cannot be placed. The pipeline retains negative hourly losses, requires
   complete days and finite water/potential, and uses 759 complete plant-days.
4. [Independent checks](independent_checks.json) reconstruct eligibility and
   daily sums from the original CSV using pandas. All 759 days match the
   standard-library derivation within 7.11e-15 L/kg/day. All eight group/bin
   medians and IQRs match. Fourteen independent mL/g-to-L/kg conversions agree
   within 1.43e-14. This is arithmetic verification, not an ecological model test.
5. Murano publisher Tables 2/4 and methods were inspected directly. Shoot
   values are dry mass and roots are separate. Table 2 is last-day daily T
   in covered soil pots; Table 4 is a distinct hydroponic experiment.
   [Readings](public_evidence/murano_table_readings.json) preserve source means
   and marginal SEs. No covariance or ratio confidence interval is invented.
6. IQR is descriptive variability across individuals, not a confidence
   interval. Release species-code counts differ from the paper's stated
   species counts; no independently verified species inventory is claimed.
   These greenhouse young-plant values do not select adult Jornada traits.
7. [Preservation checker](verify_deliverables.py) verifies the 1,753-file baseline,
   allowed writes, unchanged instructions/runtime/locks/accepted archives,
   retrieval/data identities, strict JSON, local Markdown links and whitespace.
   Results: [checks.json](checks.json).

## Access limits and errors

- `network`: default-shell public retrieval had literal
  `<urlopen error [Errno -3] Temporary failure in name resolution>` for all
  eight requests. [First manifest](public_evidence/retrievals_01.json) preserves
  the errors. Authorized public-only network retrieval with sandbox escalation
  succeeded for seven of those sources. No automatic approval rejection.
- `network`: the older CORE PDF and direct Murano HTML returned
  `HTTP Error 403: Forbidden`; no access restriction was bypassed. CORE values
  were not adopted. Murano numerical tables were publicly readable through
  the web tool and were verified there.
- Web accesses to some GitHub/Zenodo/PDF/figure endpoints returned cache misses,
  and PMC sometimes returned a browser challenge. Bounded public retrieval
  obtained the pinned savanna data/author PDF, bunchgrass PDF, wheat figures,
  wheat HTML and poplar HTML. No investigator contact or private data upload.
- Additional MDPI access returned 429; it was not retried and no numerical
  coefficient relies on that source. A guessed 2015 Frontiers URL resolved to
  an unrelated grapevine paper and was excluded immediately.
- No PDF parser or `pdftotext` was installed. Existing interpreter/package
  availability was checked; no installation or new environment was attempted.

The main comparison explicitly highlights **aboveground dry biomass**. Other
sources using final vegetative mass, subterranean crown, dry woody mass,
leaf area or instantaneous morning rates are qualified separately and do not
silently enter the main comparison. Automatic collaboration remains enabled.
