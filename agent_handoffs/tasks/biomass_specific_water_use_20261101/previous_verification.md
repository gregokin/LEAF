# Verification

Current evidence: [study_parameterization_20261101/verification.md](tasks/study_parameterization_20261101/verification.md).

- 29 final synthetic tests pass; the public-data fit exits 0 in 70.36 s.
- Independent raw-weight/support calculations pass 18 comparisons; 177 scalar
  endpoint replays agree, including 34 original withheld dry harvests.
- Installed analysis source matches the tested snapshot. Source/data integrity,
  write/read scope, inactive JSON and local links are checked in
  [checks.json](tasks/study_parameterization_20261101/checks.json).
- The real validator rejects the study's fast seedling growth rates. This is a
  reported compatibility finding; runtime bounds and equations are preserved.
- Claude's first timeout and corrected test/accounting failures are retained;
  the exact resumed session succeeds and all launched children are reaped.

Acceptance is limited to analysis and inactive candidates. No native hydrology,
transport or GPU scenario, package install, new environment, upstream edit,
commit or push was performed. Earlier accepted verification records remain in
their task archives.
