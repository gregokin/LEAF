# Species catalog v2: preferential-infiltration receiving weights

`leaf.traits` accepts an explicit, opt-in catalog **schema_version 2**: version 1 plus an optional per-species `hydrology` block.

```json
"hydrology": {"preferential_infiltration_receiver_weights": [0, 1, 1], "notes": "optional free text"}
```

* The list has exactly `root_layers` entries; each is a finite real number `>= 0` (not bool, string or null). They need not sum to 1. `[0, 0, 0]` is an explicit *disabled* declaration; `[0, 1, 1]` means equal weight on the lower two of three layers. Hugely large finite values are accepted.
* These are **not** root-uptake fractions (`roots.layer_fractions`, which must sum to 1) and are never derived from them or from `functional_type`.
* LEAF returns the **raw specified** vectors. Normalising a positive vector (and handling an unrepresentable sum) belongs to SYRUP's native preferential-infiltration operator. This receiving-weight trait API defines no cover, vegetation factor `g`, infiltration, event, storm, dead or burial rule; the separate `leaf.water_vegetation` projection supplies the approved cover rules.
* The block is allowed only in version 2 (a v1 document with it is refused as an unknown key). `block: null`, missing `preferential_infiltration_receiver_weights`, extra keys, wrong length, negative, non-finite, bool or nested values are refused. A v2 species may omit the block.

Versions: `1` (and, as before, the float `1.0`) is the unchanged legacy catalog; `2` must be the exact integer (`2.0`, `true`, strings are refused). `traits.SCHEMA_VERSION` stays `1`; `traits.SUPPORTED_SCHEMA_VERSIONS == (1, 2)`; `catalog.schema_version` and `catalog.metadata()['schema_version']` report the actual version. Nothing migrates old catalogs, checkpoints or recipes; a v1 document hashes exactly as before. `SpeciesCatalog.from_files` merges only files of the same version (mixed 1/2 is refused); duplicates, root-layer mismatches and the canonical-metadata merge policy are unchanged.

Accessors (all return fresh copies; the stored table `(n_species, L)` is read-only): `has_preferential_receiver_weights(sid)`, `preferential_receiver_weights(sid)` (length-L float64) and `preferential_receiver_weight_table(species=None)` (`[n, L]`, sorted species order). `species` must be a list/tuple (or 1-D array) of actual species-id strings; ids are never coerced (an integer `1` is not the species `'1'`), a bare string, bool, number, `None`, set or generator is refused, repeated ids stay repeated and an empty sequence gives shape `(0, L)`. Unknown ids and ids without the trait are refused before the table is allocated. A missing trait raises `CatalogError`; no default vector exists. The generic `catalog.section(sid, 'hydrology')` is known only for an actual v2 catalog (a defensive copy, or `None` when the optional block is absent); for v1 it stays an unknown section.

A `PlantState` binds the catalog by hash as before, so a state written with one catalog is refused by another (changed weights change the hash).
