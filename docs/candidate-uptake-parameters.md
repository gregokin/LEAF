# Candidate Jornada uptake parameters

Recorded 2026-11-01 at the user's request. The
[parameter record](../parameters/candidates/jornada-uptake.json) is **inactive**;
current model defaults are unchanged.

## Reported observations and fits

[Vito & Sala (2026)](https://doi.org/10.1002/ecy.70384) measured relative lithium
uptake in mesquite–black grama pairs in 2018–2019. Their fitted cumulative profile
is `F(z) = 1 - beta**z`, with numerical depth in centimeters.

| Species / condition | Beta | Reported 95% CI |
| --- | ---: | --- |
| Black grama, combined | 0.977 | 0.970–0.984 |
| Black grama, drier 2018 | 0.973 | 0.959–0.986 |
| Black grama, wetter 2019 | 0.980 | 0.973–0.986 |
| Honey mesquite, combined | 0.983 | 0.980–0.987 |
| Honey mesquite, drier 2018 | 0.984 | 0.980–0.989 |
| Honey mesquite, wetter 2019 | 0.982 | 0.976–0.988 |

Reported depth-bin summaries put over 59% of grass uptake above 50 cm and over
50% of shrub uptake below 50 cm. Grass uptake below 50 cm was approximately
33% in 2018 and 50% in 2019. These summaries need not equal the fitted curves.
The JSON also preserves the paper's optional shrub-size fit and uncertainty:
`beta(V) = 0.9869 * V / (0.0016 + V)`. Here `V` is group mean canopy volume in
m³, from overlapping windows; this is not a measured individual growth law.
[Data and analysis archive](https://doi.org/10.6073/pasta/4bcdaa76d6994ac6e81a9097b8a8dcab)

## Proposed use in LEAF

Treat these as uptake-profile priors to evaluate. For physical layer boundaries
`a, b`, the fitted allocation is `F(b) - F(a)`; explicitly decide how to represent
the tail below the modeled bottom. Check source depth coding before conversion.
Do not assign these coefficients unchanged to depths expressed in meters.

Compare dry/wet grass profiles and shrub-size sensitivity with the current fixed
[species profiles](species-root-profiles.md). A future adapter would need chosen
layer depths, biomass-to-canopy-volume allometry and an explicit tail policy.
No adapter or calibration is included here.

Tracer uptake is neither root biomass density nor absolute transpiration. It
cannot establish daily water use or a biomass yield per liter. Those require
[paired growth and water observations](paired-growth-water-data.md).
