# Research verification

Evidence: `checks.initial.json`, final `checks.json`, `catalog.json`, and the four
public-response manifests. Hashes cover raw returned bytes, not observations
described by metadata. No numerical tests or simulations were warranted or run
for this documentation-only investigation.

## Source and document checks

- Compare all 1,530 baseline files by SHA-256. Changes are limited to the allowed
  README and Codex orchestration documents. Root AGENTS.md remains
  `90a8b6217553480283efd0dcba514306c5472809b35b1d5a0e80e3009e48ef31`.
- Parse ten EML packages, compare package IDs and metadata hashes, and verify
  archived response lengths/hashes against manifests (23 responses).
- Inspect NEON product descriptions, JORN availability and released versus
  provisional months. Record the current soil-water sensor-depth warning.
- Check local Markdown links and trailing whitespace. Inspect primary dataset
  metadata/methods and paper data-availability statements for reported claims.
- Historical `previous_*.md` aliases remain byte-exact; their relative links are
  checked against their original `agent_handoffs/` base. The first close check
  incorrectly used the archive directory for those links (`check_failed`);
  preserve that result as `checks.pre_close.json` and correct the checker.
- Do not treat the yearly ANPP product as independent of seasonal biomass, or
  NPP subvolumes/quadrat coordinates as stable whole-individual measurements.

## Access failures and interpretation

Initial sandbox DNS failure and public EDI HTTP403 responses are classified
`network`. An alternative hostname did not resolve. DataONE supplied selected
public EML metadata, including older biomass/dimension versions; these are pinned
without calling them the latest EDI publication.

The first requested seasonal-biomass v4 returned an empty HTTP200 body; classify
`other`, exclude it from parsed usable metadata, and preserve the zero-byte file
and manifest. Dimension v109, experimental biomass v5 and three mirrored data/PDF
objects returned404 (`network`); requested experiment queries produced zero
results or one400 (`tool_error`). Confirmed indexed biomass v2/dimensions v108
were then retrieved and parsed. NEON product/site metadata succeeded; selected
per-month data listings returned403 (`network`). No observation tables were
retrieved. One initial malformed apply_patch was rejected before writes and
corrected (`tool_error`). None of these failures was an automatic-approval
rejection, and none authorized data access or scientific claims beyond the
successful metadata/method evidence.

The inventory clearly leaves data download, row-level QC, species matching,
sampled biomass conventions, exact overlap and holdout selection for a later
bounded validation exercise. No investigators were contacted and no local code
or brief was uploaded. Automatic collaboration remains enabled.
