# Research notes — literature_parameters_20261111 rev1 (Claude)

Date of research: 2026-11-11. Public sources only. "CHECKED" = read directly from a fetched page (a small fast model summarised each fetched page, so numbers are
as returned by that summariser and should be re-verified before any adoption). "SEARCH" = a web-search result summary only. "INFERENCE" = my reasoning.

## Outcome

No numeric value in `species.json` was changed. Reason: none of the retrievable quantities maps defensibly onto a native trait (see per-trait decisions). Only
provenance text, `references` and `catalog_id` changed (`jornada_grass05_shrub10_trial_20261111` -> `jornada_grass05_shrub10_litprov_rev1_20261111`). Because no
q, live_fraction or mean lifespan changed, no k and no Weibull scale needed recomputation.

## Direct source checks

### FEIS black grama (Simonin 2000) — <https://research.fs.usda.gov/feis/species-reviews/boueri> — CHECKED (fetch OK)
* Height "20-60 cm" stems; roots greatest in uppermost 25 cm, "in sand roots may extend 1.2 m" (already used qualitatively).
* Phenology, Chihuahuan Desert: growth begins late April-early May; flowering late Aug-early Sep; fruiting late Sep-early Oct. Jornada Range 1925-1935 monitored growth periods 0-176 days, rainfall during growth periods 1.57-11.95 in (= 39.9-303.5 mm; 25.4 mm/in).
* Germination 1917-1926 New Mexico: 0-40.7% (mostly 2-27.5%). Seedlings produced in only 7 of 53 years (Neilson 1986, Science 232:27-34). First-year seedling survivorship 35.8% (grazed) / 38.8% (ungrazed). Mean new plants 2.8 (grazed) vs 6.4 (ungrazed) per m2 over 17 years.
* Stolons: favorable year for stolon production then another for rooting (Nelson 1934, USDA Tech. Bull. 409). Valentine 1970 (NMSU Bull. 553): light/moderate/recommended grazing -> 2.8/2.6/2.1 stolons per plant, stolon length 5.4/5.0/4.6 in, rooted sets 0.31/0.28/0.19 per plant (9-year study; "per year" not stated by the summary).
* Lifespan: "long-lived" (Gould 1978), "may survive several decades through stoloniferous reproduction" (Nelson 1934). Fire recovery 2-8 yr typical.
* Decision: phenology dates are consistent with `monthly_activity` (Apr 0.1, May 0.3, Jun 0.6, Jul-Sep 1.0, Oct 0.4, Nov 0.05); kept as ASSUMED-SHAPE. Rooted sets 0.19-0.31/plant is a count of successful sets over an unknown time basis; `clonal.probability` 0.1 (daughters per eligible parent per year, max 1) is a different quantity -> kept AUTHOR. Genet "several decades" vs ramet/tussock mean lifespan 15 yr: not comparable -> kept.

### FEIS honey mesquite (Steinberg 2001) — <https://research.fs.usda.gov/feis/species-reviews/progla> — CHECKED
* Western Texas phenology table: Nov-Mar dormant; 16 Apr most beginning to leaf out; 10 May fully developed leaves and white flowers; 26 Aug pods fallen. Leaf drop generally Nov-Dec, often initiated by killing frost. Bud break can vary 6 weeks; occurs after last frost or when soil warms to 18 degC.
* Seed: first seed by ~3 yr; 131 or 97 pods/tree; ~12 seeds/pod (~5 destroyed by bruchids); optimum seed depth 0.5-1.5 cm, no emergence below 5 cm; germination laboratory 33 degC. Seedling 1-yr survival 27% (clipped) vs 9.4% (unclipped).
* Longevity: Amarillo TX, maximum age 40-110 yr; Rio Grande Plains 89-93% of plants < 100 yr, maximum 172-217 yr.
* Fire: age-dependent mortality (Wright et al.), top-kill 72-90% with whole-plant mortality 0-11% in winter fires; no fire module in LEAF -> not usable.
* Decisions: monthly_activity consistent (Apr 0.4, May 0.8, Jun-Aug 1.0, Sep 0.8, Oct 0.4, Nov 0.05, Dec 0); note Nov is "dormant" in the Texas table, 0.05 left unchanged as a negligible residual. Lifespan values are MAXIMA -> mean 80 kept (the prompt forbids converting an assumed mean to a maximum observation).

### Hanson 1982 — <https://rangelandsgateway.org/dlio/22027> — CHECKED (abstract page)
Hanson JD (1982) J. Range Manage. 35(4):455-458, "Effect of light, temperature and water stress on net photosynthesis in two populations of honey mesquite". Two Texas P. glandulosa populations; maximum net photosynthesis at 20 degC (east Texas) and 30 degC (west Texas); max PN 0.69-0.82 mg m-2 s-1 (units as returned). Search summary additionally said 35 degC treatment reduced assimilation (SEARCH only). Citation volume/pages from a search result (SEARCH).
* Decision: PROGLA `temperature_opt_c` 30 equals the west-Texas photosynthesis optimum -> labelled PROXY-CONSISTENT; not claimed as measured growth cardinal. Min/max remain assumed.

### Peters 2002 AJB — <https://doi.org/10.3732/ajb.89.10.1616> — abstract only via <https://api.crossref.org/works/10.3732/ajb.89.10.1616> (CHECKED); publisher page HTTP 403
Abstract: compares seed production and soil seed presence of Bouteloua gracilis and B. eriopoda; "short-lived stoloniferous" species more seeds/plant with viability < 60%; "long-lived bunchgrass" viability > 90% and greater soil seed presence. The fetch summary did not tie each growth form to its species. INFERENCE (unverified): B. eriopoda is the stoloniferous one; do not rely on this. Nothing adopted (BOER seed pathway null).

### STEPWAT2 pinned 003cb9b4 — <https://github.com/DrylandEcology/STEPWAT2/tree/003cb9b4adbf13d64a39d6d62fd95280adac27fc/testing.sagebrush.master/Stepwat_Inputs/Input> — CHECKED
`rgroup.in` (raw GitHub URL) lists groups sagebrush, a/p cool and warm forbs, a/p cool grass, p.warm.grass (C4), shrub, succulents (sagebrush-steppe test project). No Bouteloua eriopoda or Prosopis record, so there is no STEP-derived proxy for these species; none adopted. The per-group numeric columns (maxbio, max age, etc.) were not obtained from the summariser and were not needed.

### Mesquite biomass partitioning — SEARCH / partial
* Ansley, Mirik & Castellano 2010, GCB Bioenergy 2:26-36 (SEARCH summaries): regrowth (2-12 yr) vs undisturbed honey mesquite of the same height; leaf:twig mass near 1; stems > 3 cm = 18% of tree mass at 7 yr and 41% at 12 yr (regrowth). Does not give the mature-tree leaf+twig/total dry AGB ratio and does not define a transpiring fraction -> live_fraction 0.4 unchanged.
* Ludwig, Reynolds & Whitson 1975, Am. Midl. Nat. 94:451-461: bibliographic page CHECKED, paper PDF not obtained (uidaho link redirected to a 404, web.archive.org not fetchable, Jornada archive DNS failure). Only a SEARCH summary: Prosopis averaged "over 10 kg" aboveground. No fraction values. INFERENCE: consistent in order of magnitude with `mature_biomass_kg` 10 (note a mean of sampled plants is not a saturation cap).
* Navar et al. 2019 PeerJ 7:e6782, <https://doi.org/10.7717/peerj.6782> (PMC6487190): AGB vs basal diameter equations for 510 mesquite trees in northern Mexico (oven-dry 70-80 degC) — allometry, the Codex lane; not used here.
* Northup et al. 2005, J. Arid Environ. 62:23-43 (DOI 10.1016/j.jaridenv.2004.09.019; PDF binary and unreadable by the fetch tool) and McClaran et al. 2012 (P. velutina): allometry/leaf fractions not obtained.
* One search snippet gave 7,377 g leaves vs 219 kg wood for a single honey mesquite tree (~3.3% leaves); source unidentified, one large tree -> not used.

### Golubov et al. 1999 — <https://researchportal.plymouth.ac.uk/en/publications/demography-of-the-invasive-woody-perennial-prosopis-glandulosa-ho/> — CHECKED (abstract)
J. Ecol. 87(6):955-962. 1306 plants in 1 ha, 4 years, nine size classes; lambda 1.29 (annual 0.99-1.44); seedling recruitment elasticity very low. Population lambda is not a LEAF per-plant rate; not adopted.

## Failed or unusable sources (literal errors)
* `https://archive.jornada.nmsu.edu/...` (node/50925 and files/bibliography/JRN00364.pdf): `getaddrinfo ENOTFOUND archive.jornada.nmsu.edu`.
* `https://www.webpages.uidaho.edu/range357/extra-refs/Ludwig_et_al_1975.pdf`: 301 redirect to `https://www.uidaho.edu/404.htm` (not followed).
* `https://web.archive.org/...`: "Claude Code is unable to fetch from web.archive.org".
* `https://bsapubs.onlinelibrary.wiley.com/doi/10.3732/ajb.89.10.1616`, `https://peerj.com/articles/6782` (+ .pdf), `https://repository.arizona.edu/handle/10150/644049`, `https://data.zhenzhi.com/details/1/908115453`: "The server returned HTTP 403 Forbidden."
* `https://journals.uair.arizona.edu/...`: 301 redirect to `https://journals.librarypublishing.arizona.edu/` (not followed).
* `cales.arizona.edu` Northup 2005 / Hughes 2006 PDFs and USDA-ARS "Simulating the Productivity of Desert Woody Shrubs" PDF, mountainscholar download: fetched as binary/scanned PDF; the summariser could not extract text. (The fetch tool saved the binaries under its own tool-results directory, outside my allowed read scope; I did not read them.)

## Traits with no usable literature mapping (retained AUTHOR)
Growth rates, extra_growth, live fraction (both), BOER mature biomass, activity levels, min/max cardinals, clonal/seed probabilities and initial masses, all age-hazard/stress parameters, all burial/exposure/dislodgement thresholds (searches for black grama sand burial returned only qualitative wind-erosion/seedling results), BOER and PROGLA adult water demand (only proxies, unchanged).

## Unit conversions (checked arithmetic)
* 1.57 in x 25.4 = 39.88 mm; 11.95 in x 25.4 = 303.53 mm.
* 0.2 / 0.1 daughters comparison: 0.1 daughters/parent/year vs 0.19-0.31 rooted sets/plant (time basis unknown) — order-of-magnitude only.
* No q, live_fraction or lifespan change, so k = live_fraction/(365 q) (BOER 0.6/(365 x 0.0234) = 0.070249385; PROGLA 0.4/(365 x 0.0104) = 0.105374078) and Weibull scales 15/Gamma(1.5), 80/Gamma(1.5) are unchanged.

## Preview freeze
`preview/species_at_initialization.json` was written first from the baseline `species.json` text as read at session start (identical content intended, trailing newline included). I could not hash or diff (no execution), so byte identity must be verified by Codex against `baseline.json`.
