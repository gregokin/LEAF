# Research notes — literature_parameters_20261111 rev2 (Claude)

Rev2 date 2026-11-12. Scope: implement items A–E from `codex_evidence_02.md`. **Source status labels.** CODEX-CHECKED = value supplied in `codex_evidence_02.md`, which says Codex
read/visually checked the primary PDF or file; I did not re-open those PDFs (my WebFetch cannot read them). REV1-CHECKED = fetched by me in rev1 (a small model
summarised the pages). INFERENCE = my reasoning/arithmetic. No code, simulation, hashing or generation was executed; all arithmetic below was done by hand and should
be re-verified by Codex.

## 0. Outcome — actual numeric changes (vs baseline `jornada_grass05_shrub10_trial_20261111`)

| Field | Baseline | Rev2 |
|---|---:|---:|
| PROGLA allometry.diameter.a | 1.6 | 1.1087951253470727 |
| PROGLA allometry.diameter.b | 0.4 | 0.3351206434316354 |
| PROGLA allometry.height.a | 0.8 | 1.1015229296155495 |
| PROGLA allometry.height.b | 0.4 | 0.2546916890080429 |
| PROGLA growth.live_fraction | 0.4 | 0.906 |
| PROGLA growth.biomass_support_kg_year_per_m3 | 0.10537407797681772 | 0.23867228661749212 |
| BOER growth.intrinsic_rate_per_year | 0.8 | 0.474 |
| BOER reproduction.clonal.initial_agb_dry_kg | 0.001 | 0.0015096 |
| BOER reproduction.clonal.min_agb_dry_kg | 0.01 | 0.015096 |
| soil.toml theta_field_capacity | 0.20 | 0.23 |
| catalog_id | jornada_grass05_shrub10_trial_20261111 | jornada_grass05_shrub10_litparam_rev2_20261111 |

Everything else numeric is unchanged (mature masses 0.1 / 10, extra growth, activity arrays, cardinal temperatures, water demand 0.0234 / 0.0104 and reference PET, roots,
life expectancies 15 / 80 yr and Weibull scales, stress/disturbance, other reproduction fields, soil θr, θs, evaporation stress, diffusivity). Text-only changes: notes/status/units/references.

## A. PROGLA diameter allometry (CODEX-CHECKED source; derivation by hand)

Source: Northup, Zitzer, Archer, McMurtry & Boutton (2005), *J. Arid Environ.* 62:23–43, <https://doi.org/10.1016/j.jaridenv.2004.09.019>;
PDF <https://cales.arizona.edu/research/archer/publications/refereed_journal_articles/Northup-et-al_2005_JAridEnv.pdf>. **Table 7 p. 34** (PDF p. 12), row *Prosopis glandulosa*:
ln(TB_g) = 6.960 + 1.492 ln(CA_m²); df 36 (n = 37), R² = 0.98, reported CF 1.006. TB = total oven-dry aboveground mass (leaves + live wood + standing dead wood; 70 °C to constant
weight, p. 26; no roots). CA = projected elliptical crown area from longest/perpendicular axes (not basal diameter). Site: La Copita Research Area near Alice, southern Texas; May–Aug 1994/1995; mean rain ≈ 680 mm/yr.
Table 1 gives means only (no valid min/max prediction range; none invented).

Derivation (stored values from Codex; spot-checked by hand):
* B_g = e^6.960 · CA^1.492 → B_kg = (e^6.960/1000) · CA^1.492 = C · CA^1.492, with ln C = 6.960 − ln 1000 = 6.960 − 6.907755 = 0.052245, **C = 1.0536335572423194**. CF (1.006) intentionally not applied: the source's CF placement notation is inconsistent with the standard retransformation; the raw log-space median is a transparent choice.
* Native circular-equivalent diameter D = √(4·CA/π) ⇒ CA = π D²/4 ⇒ B = C (π/4)^1.492 D^2.984 ⇒ D = [C (π/4)^1.492]^(−1/2.984) · B^(1/2.984).
* b_d = 1/2.984 = **0.3351206434316354**. ln(π/4) = −0.241564; 1.492 × (−0.241564) = −0.360414; ln[C (π/4)^1.492] ≈ 0.052245 − 0.360414 = −0.308169 (consistent with the stored a to four digits); a_d = exp(0.30817 × 0.335121) = exp(0.103272) ≈ 1.1088 → stored **1.1087951253470727**.
* Checks (stored): D = 1 m → B = 0.7347907779912205 kg [hand: (1/1.10880)^(1/0.335121) = exp(−0.30817) = 0.7348 ✓]; D = 2 m → 5.813493672465879 kg [hand: exp(ln(1.80375)/0.335121) = exp(1.7595) = 5.81 ✓]; B = 10 kg → D ≈ 1.1088 × 10^0.335121 = 1.1088 × 2.1632 ≈ 2.40 m, CA ≈ 4.52 m² (arithmetic only).
* Limitations (documented): algebraic inverse of a CA→B regression, not a separately fitted inverse; wetter site; target species but not Jornada.
* Previous law: D = 1.6 B^0.4 (D = 1 m ⇔ B = 0.3088 kg); revised B at D = 1 m is 2.38× larger. Old stored preview masses read through the new law: D = 1.1088 × 0.30882^0.33512 = 1.1088 × exp(−1.17500 × 0.335121) = 1.1088 × 0.6745 ≈ 0.748 m, and H = 1.10152 × 0.30882^0.25469 = 1.10152 × 0.7414 ≈ 0.817 m.

## B. PROGLA height proxy (CODEX-CHECKED source; derivation by hand)

Source: McClaran, McMurtry & Archer (2013), *J. Arid Environ.* 88:39–42, <https://doi.org/10.1016/j.jaridenv.2012.08.015>;
PDF <https://cales.arizona.edu/research/archer/publications/refereed_journal_articles/McClaran-et-al_2012_JAridEnv.pdf>. **Table 2 p. 41** (PDF p. 4): HT_m = exp(0.11)·CA_m²^0.38, CF 1.00, R² = 0.91, n = 31.
ORIGINAL species *Prosopis velutina* (related species), Santa Rita Experimental Range near Tucson, Arizona, June 2004; CA range 0.1–77 m², H 0.5–6.9 m.

Derivation: CA = (B/C)^(1/1.492) ⇒ H = e^0.11 (B/C)^(0.38/1.492) ⇒ b_h = 0.38/1.492 = **0.2546916890080429**; a_h = e^0.11 · C^(−0.254692) = 1.116278 × exp(−0.254692 × 0.052245) = 1.116278 × 0.98678 ≈ 1.1015 → stored **1.1015229296155495**.
Checks (stored): D = 1 m → H = 1.0183722297676923 m [hand: 1.10152 × exp(−0.30817 × 0.254692) = 1.10152 × 0.92451 = 1.0184 ✓]; D = 2 m → 1.7246038235919356 m. B = 10 kg → H ≈ 1.10152 × 10^0.254692 ≈ 1.98 m.
H = D/2 no longer holds. Mixed-study deterministic proxy; seedlings (e.g. 0.002 kg ⇒ D ≈ 1.1088 × 0.002^0.33512 ≈ 0.14 m) lie outside the measured adult structure. Not claimed qualified for all future sizes.

Candidate NOT active (sensitivity): McClaran Table 3, *P. velutina* TOTAL biomass B_kg = exp(−0.59)·CA^1.60 (CF 1.06 omitted, same convention) → 0.3766258 kg at D = 1 m (CA = 0.7854 m²; check: exp(−0.59) × 0.7854^1.6 = 0.55432 × 0.6795 = 0.3767 ✓). Hand-derived native form (approximate, not stored anywhere in the catalog): b = 1/3.2 = 0.3125; ln[e^−0.59 (π/4)^1.6] = −0.59 − 0.38650 = −0.97650, a = exp(0.97650 × 0.3125) = exp(0.30516) ≈ 1.357. Paper text and table use differing exponents; the table was used. Basal-diameter equations are not crown-diameter equations and were not adopted.

## C. PROGLA live fraction and support (CODEX-CHECKED source; arithmetic)

Northup 2005 **Table 1 p. 28** (PDF p. 6), *P. glandulosa*, n = 37: leaf 10.1%, live stems 80.4%, standing dead wood 9.4% of total oven-dry aboveground mass (sum 99.9%, rounding).
live_fraction = 1 − 0.094 = **0.906** (one minus the observed mean dead fraction; includes live WOOD; not leaf area, not a metabolically active or transpiring fraction; constant is a size/site approximation).
The native water demand (`biomass_specific_pet_v1`) multiplies q by TOTAL B, so q is not multiplied by live_fraction again (documented in species.json, PARAMETERS.md). q unchanged (0.0104 m³/kg/day).
k = live_fraction / (365 q) = 0.906 / (365 × 0.0104) = 0.906 / 3.796 = **0.23867228661749212** kg·year/m³ (stored value from Codex; hand check 0.238672 ✓). BOER k unchanged (0.6/(365 × 0.0234) = 0.07024938531787846). Grass live_fraction 0.6 stays AUTHOR.

## D. STEP model-input proxies (CODEX-CHECKED file; I did not open the file)

Pinned official source: STEPWAT2 revision 003cb9b4adbf13d64a39d6d62fd95280adac27fc,
<https://github.com/DrylandEcology/STEPWAT2/blob/003cb9b4adbf13d64a39d6d62fd95280adac27fc/testing.sagebrush.master/Stepwat_Inputs/Input/species.in>
(raw `https://raw.githubusercontent.com/DrylandEcology/STEPWAT2/003cb9b4adbf13d64a39d6d62fd95280adac27fc/testing.sagebrush.master/Stepwat_Inputs/Input/species.in`; Codex file SHA-256 c20a67b0a8095f4c9c920c3f73a711437ecc43039a934a1f4a351bc864bce2f3).
Row `bogr` = *Bouteloua gracilis* (blue grama; **not** *B. eriopoda*; sagebrush/shortgrass-steppe model parameters, not field estimates): irate 0.474, slow2yr, seedling minbio 1.5096 g, mature maxbio 15.096 g, maximum age 35 yr, vegindv 3, clonal y. Header credits Bradford & Lauenroth (2006) Table A1.4 (underlying table not checked).
* BOER `intrinsic_rate_per_year` = 0.474 (admitted 0..1; no clamping).
* BOER `clonal.initial_agb_dry_kg` = 1.5096 g = 0.0015096 kg (size proxy for one separately tracked daughter, not STEP's biomass-added clonal algorithm).
* BOER `clonal.min_agb_dry_kg` = 10 × 0.0015096 = 0.015096 kg, AUTHOR-DERIVED, so that max_donation_fraction 0.1 affords one daughter. Native eligibility (read from `leaf/recruitment.py` docstring): B ≥ min_agb AND max_donation_fraction × B ≥ initial_agb. At B = min_agb the second test compares 0.1 × 0.015096 with 0.0015096 (equal in exact arithmetic; floating-point ties at exactly B = min_agb are possible, not at larger B). 0.015096 kg coincides numerically with bogr maxbio 15.096 g only because maxbio = 10 × minbio in that row; not claimed as maturity size.
* Not adopted: maxbio (mature mass stays 0.1 kg: 15.096 g is a model input for blue grama and lies below many admissible initial plants), max age 35 yr (not a mean lifespan), ratep/vegprop (different semantics from probability 0.1/max_daughters 1), extra growth.
* `stress.limit_years` 2 agrees with slow2yr — labelled agreement/model proxy only.
* Consequence of the new threshold: BOER mean initial B = (0.10/0.9)^(1/0.45) = 0.0075764 kg < 0.015096 kg; threshold mass corresponds to D = 0.9 × 0.015096^0.45 = 0.9 × exp(0.45 × ln 0.015096) = 0.9 × exp(−1.8888) = 0.9 × 0.1511 ≈ 0.136–0.137 m. Most initial BOER plants are ineligible until they grow (documented).

## E. Soil field capacity (CODEX-CHECKED source)

Allen et al. (1998) FAO-56 Chapter 7, <https://www.fao.org/4/x0490e/x0490e0c.htm>, Table 19, sandy loam: θ_FC 0.18–0.28, θ_WP 0.06–0.16 m³/m³; Example 35 uses θ_FC = 0.23 (WP 0.10). `theta_field_capacity` 0.20 → 0.23 (texture-scale illustrative prior). Retained: Carsel & Parrish (1988) <https://doi.org/10.1029/WR024i005p00755> θr 0.065, θs 0.41 (independent simple sandy-loam priors; not one van Genuchten curve; not a wilting point). Not adopted: FAO evaporation thresholds/REW (different law), residual as wilting point, Ksat as diffusivity. Ordering check for SYRUP-style layers: 0.065 < 0.15 < 0.23 < 0.41 and theta_initial 0.10–0.14 < 0.23.

## F. Gaps documented, not adopted (CODEX-CHECKED)

* BOER lifespan validation gap: Wright (1972) *J. Range Manage.* 25:476–478, <https://journals.uair.arizona.edu/index.php/jrm/article/viewFile/6125/5735>, Table 1: mean black grama clump persistence 2.2 yr (SD 2.4, n = 6528), maximum 27 yr; Latin-name OCR in the table ambiguous (B. gracilis next to "black grama"); page image blocked during Codex check. Smith et al. (2004) Gramas chapter p. 993, <https://www.ars.usda.gov/arsuserfiles/30300000/Publications/AgronMono45-975-1002MRH.pdf>: Wright & Van Dyne 1976 mean ≈ 2 yr including all seedlings, ≈ 4 yr conditional on first-year survival, max ≈ 28 yr. Includes drought/seedling mortality; 15 yr (AUTHOR age component) stays and is flagged as unvalidated; PROGLA 80 yr likewise (FEIS values are maxima).
* Grass growth experiment Fernandez & Reynolds (2000) *Oecologia* 123:90–98: BOER seedlings total dry mass RGR 0.072/day (CV 0.17), nursery 18–37 °C, ×365 = 26.28/yr, whole plant incl. roots — not a native aboveground annual rate; not activated (citation as supplied by Codex; I did not verify it).
* Nafus et al. (2009): current-year mass vs basal diameter, not total standing mass vs crown — not adopted for grass geometry.
* FEIS stolon counts/seedling frequencies are not annual per-parent probabilities; no quantified burial/exposure/dislodgement response verified; abrasion stays disabled.

## G. Corrections to rev1 text

* The unverified Ludwig 1975 ">10 kg" search snippet and the Ansley 2010 / Golubov 1999 / Ludwig 1975 references were removed as numeric justification and from the catalog `references` lists. `mature_biomass_kg` 10 is now documented as a gap (arithmetic inversion of the Northup law, B = 10 kg ↔ D ≈ 2.4 m, is not evidence).
* Rev1 statements "no STEP number adopted", "Allometry is not a published equation", "live fraction 0.4 AUTHOR" were removed where they contradict the active values.
* Rev1 PROGLA activity/Hanson 1982 labelling is retained (REV1-CHECKED from abstract pages: FEIS honey mesquite western Texas phenology 16 Apr leaf-out, 10 May full leaves, Nov–Mar dormant; Hanson 1982, J. Range Manage. 35(4):455–458, <https://rangelandsgateway.org/dlio/22027>, maximum net photosynthesis 20 °C east Texas / 30 °C west Texas). The photosynthesis optimum is not a growth cardinal; labelled PROXY-CONSISTENT, shape assumed. Rev2 did not re-open Hanson's paper.

## H. Rev1 failed fetches (literal errors, unchanged)

`getaddrinfo ENOTFOUND archive.jornada.nmsu.edu`; HTTP 403 for Wiley, PeerJ, repository.arizona.edu, data.zhenzhi.com; redirect to `https://www.uidaho.edu/404.htm` (Ludwig PDF); "Claude Code is unable to fetch from web.archive.org"; journals.uair.arizona.edu 301 redirects; cales.arizona.edu and other PDFs returned as binary with no extractable text. No new fetches were made in rev2.

## I. Preview freeze and reader check

`preview/species_at_initialization.json` was not touched in rev2 (hash-verified by Codex). Public reader signature checked in `leaf/state.py`: `read_plant_csv(path, catalog, *, domain=None, surface=None, time_days=None, provenance=None, metadata=None)`; catalog loader `leaf.traits.SpeciesCatalog.from_file(path)` (canonical hash over the parsed document, so ANY text change in `species.json` — including notes and catalog_id — changes the hash). `_check_context` refuses a catalog-hash mismatch. Reader example added to `preview/README.md`.

## K. Rev3 documentation corrections (no numeric change; no new research)

Statements supplied by Codex and used as stated (not fetched by me): Vito & Sala 2026 (<https://doi.org/10.1002/ecy.70384>) is a **lithium**-tracer uptake study, not isotope tracing (β and layer fractions unchanged); Hanson 1982 abstract (<https://rangelandsgateway.org/dlio/22027>, DOI <https://doi.org/10.2307/3898604>) verifies maximum net photosynthesis 20 °C east / 30 °C west Texas, so the PROXY-CONSISTENT label for `temperature_opt_c` 30 is supported; the earlier "reduced assimilation at 35 °C" claim came only from a rev1 search snippet and was removed (`temperature_max_c` 45 is an explicit assumption); Fernández & Reynolds 2000 (<https://doi.org/10.1007/s004420050993>, Table 1 at the CiteSeerX link in PARAMETERS.md): BOER seedlings 100–300 mg total dry mass, RGR 0.072/day, CV 0.17, nursery 18 °C night / 37 °C day (candidate only); Nafus et al. 2009 DOI <https://doi.org/10.2111/08-003> (current-year mass vs basal diameter; not adopted). Wording changes: `life_expectancy` notes no longer say "metadata only" (the value bounds the uniform initial-age draw; not a deterministic death age); FEIS "long-lived" is now stated as also describing multi-decade stolon persistence that does not establish a mean ramet/clump lifespan; PROGLA seedling size is described as below the McClaran 2013 CA range (min 0.1 m²) with Northup small-plant validity unqualified (its sampled range was not reported); PROGLA `live_fraction` is LIT-DERIVED (complement 1 − 0.094; 0.906 is not printed). Numeric values are exactly as in rev2.

## J. Schema notes

Catalog allowed keys respected: species-level `name`, `notes`, `references`; section keys `law`, `traits`, `status`, `notes`, `units`; allometry keys `model`, `diameter`, `height`, `notes`, `units`; no new keys added. Trait key sets for growth/water_use/reproduction unchanged.
