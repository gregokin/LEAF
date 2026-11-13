# Additional independent restart finding

probe_revision02_rng.py (frozen review_02 source), exit1 with assertion:
`rng_seed=10 -> 20` and `patch_size_m=20 -> 4` are both accepted at load_biology and reproduce the SAME token.
Catalog, source and geometry remain equal; neither setting is in the manifest/token binding. They affect future keyed mortality/reproduction draws and patch trials, so restart equivalence fails.
Preserve the actual RNG seed and full active patch configuration, either serialize/restore or refuse mismatched explicit caller configuration. Also bind the session token/config identity. This does not authorize new ecological laws or any case parameter change.
