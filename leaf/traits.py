'''leaf.traits: versioned LEAF species catalog (JSON) with strict validation and power-law allometry.

Landscape Ecohydrology with Agent-based Flora (LEAF), Phase 0. A catalog is a JSON document

    {"schema": "leaf.species_catalog", "schema_version": 1,
     "units": {"biomass": "kg", "length": "m", "time": "days"},
     "root_layers": L,
     "species": {"<id>": {...}, ...}}

keyed by stable, case-sensitive species IDs. Per species:
  functional_type  non-empty string
  allometry        {"model": "power_law_v1", "diameter": {"a", "b"}, "height": {"a", "b"}}
                   diameter_m = a_d * B_kg**b_d, height_m = a_h * B_kg**b_h, with a > 0 (units m kg^-b) and
                   b > 0 (dimensionless), so both laws are monotone increasing in B and invertible:
                   B_kg = (diameter_m / a_d) ** (1 / b_d). Non-finite OUTPUTS (overflow) are refused.
  roots            {"lateral_radius_multiplier": > 0, "layer_fractions": [L finite nonnegative, sum 1]}
  porosity         {"value": [0, 1], "status": str}  constant optical/aerodynamic porosity handed to MAPLE;
                   LEAF ecology never reads it, native MAPLE wind DOES use it.
  growth, water_use, reproduction, mortality, disturbance
                   null or an object with a non-empty string "law" (e.g. "unimplemented"); contents are
                   validated generically (finite numbers) and preserved verbatim for later consumers.
  hydrology        (schema_version 2 ONLY; optional per species) {"preferential_infiltration_receiver_weights": [L finite
                   nonnegative reals], "notes": ...}. The weights say how this species' preferential-infiltration contribution
                   is shared over the L soil layers; they are NOT root-uptake fractions and need not sum to 1. An all-zero
                   vector DISABLES the species' pathway; a positive vector is normalised later by SYRUP (LEAF returns the RAW
                   specified values). No default is ever invented and the functional_type never enables or disables it. A v1
                   catalog (or a v2 species without the block) simply has no such trait; the accessors refuse rather than guess.
  life_expectancy  null or {"mean": > 0, "unit": "days" | "years"}  NOMINAL mean life expectancy (years = 365 days).
                   It is NOT a death age: LEAF never kills a plant because its age reaches it (mean != hard maximum;
                   mortality is the separate probabilistic law of leaf.demography). It also supplies the UPPER BOUND of
                   the species-linked initial-age rule of leaf.initialization (the default when a generation config
                   omits initial_age_days): initial ages are drawn uniformly on [0, life_expectancy_days). A species
                   with life_expectancy null cannot use that rule (validation error; explicit ages still work).

Ownership: the parsed document is private; `catalog.document` returns a deep copy, so user edits can never
change canonical content behind the cached coefficients or the hash. The species name/index table is the
SORTED tuple of species IDs, hence deterministic across JSON key reordering (which also leaves the
canonical hash unchanged). Pure stdlib + NumPy; nothing here imports numba, cupy, MAPLE or SYRUP.
'''
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re

import numpy as np

LEAF_VERSION = '0.1.0'
SCHEMA = 'leaf.species_catalog'
SCHEMA_VERSION = 1                      # the legacy/base catalog version (unchanged; v1 documents are not rewritten)
SCHEMA_VERSION_HYDROLOGY = 2           # explicit opt-in version: v1 + the optional per-species `hydrology` block (see HYDROLOGY_KEYS)
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_HYDROLOGY)
HYDROLOGY_KEYS = ('preferential_infiltration_receiver_weights', 'notes')
RECEIVER_WEIGHTS_KEY = 'preferential_infiltration_receiver_weights'
ALLOMETRY_MODEL = 'power_law_v1'
REQUIRED_UNITS = {'biomass': 'kg', 'length': 'm', 'time': 'days'}
BIOMASS_UNIT_LABEL = 'kg dry aboveground mass per individual'
TIME_UNIT_DAYS = {'days': 1.0, 'years': 365.0}
LAYER_FRACTION_TOL = 1e-9
SPECIES_ID_PATTERN = re.compile(r'[A-Za-z0-9_.\-]{1,64}')
REQUIRED_SECTIONS = ('functional_type', 'allometry', 'roots', 'porosity', 'growth', 'water_use',
                     'reproduction', 'life_expectancy', 'mortality', 'disturbance')
OPTIONAL_SPECIES_KEYS = ('name', 'notes', 'references')
HYDROLOGY_SPECIES_KEY = 'hydrology'     # allowed ONLY in schema_version 2
LAW_SECTIONS = ('growth', 'water_use', 'reproduction', 'mortality', 'disturbance')
TOP_LEVEL_KEYS = ('schema', 'schema_version', 'catalog_id', 'units', 'root_layers', 'species', 'notes')
UNIMPLEMENTED = 'unimplemented'
STEPWAT_GROWTH_LAW = 'stepwat_weekly_aboveground_v0.1'
SPECIES_TABLE_ORDER = 'sorted species ids (deterministic across JSON key order)'


class CatalogError(ValueError):
    '''Any species catalog schema / value error (raised before any state is built or mutated).'''


# ------------------------------------------------------------------ strict JSON
def _pairs_hook(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise CatalogError(f'duplicate JSON key {k!r}')
        out[k] = v
    return out


def _refuse_constant(name):
    raise CatalogError(f'non-finite JSON constant {name!r} is not allowed')


def parse_strict_json(text, *, what='JSON document'):
    '''json.loads refusing duplicate keys and NaN/Infinity constants.'''
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    try:
        return json.loads(text, object_pairs_hook=_pairs_hook, parse_constant=_refuse_constant)
    except CatalogError:
        raise
    except (ValueError, TypeError) as exc:
        raise CatalogError(f'{what}: invalid JSON: {exc}') from exc


def canonical_json(document):
    '''Compact, key-sorted, ASCII JSON text; NaN/Infinity refused.'''
    return json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def catalog_hash(document):
    '''sha256 over canonical_json(document); prefixed "sha256:". Whitespace and key order do not matter;
    numeric spelling does (1 vs 1.0 are different JSON tokens).'''
    return 'sha256:' + hashlib.sha256(canonical_json(document).encode('ascii')).hexdigest()


# ------------------------------------------------------------------ scalar validators (pure)
def is_real(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def real(v, name):
    if not is_real(v):
        raise CatalogError(f'{name} must be a real number (not bool/str/None), got {v!r}')
    v = float(v)
    if not math.isfinite(v):
        raise CatalogError(f'{name} must be finite, got {v}')
    return v


def positive(v, name):
    v = real(v, name)
    if v <= 0.0:
        raise CatalogError(f'{name} must be > 0, got {v}')
    return v


def nonnegative(v, name):
    v = real(v, name)
    if v < 0.0:
        raise CatalogError(f'{name} must be >= 0, got {v}')
    return v


def unit_interval(v, name):
    v = real(v, name)
    if v < 0.0 or v > 1.0:
        raise CatalogError(f'{name} must lie in [0, 1], got {v}')
    return v


def strict_int(v, name, lo=None):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise CatalogError(f'{name} must be an integer, got {v!r}')
    v = int(v)
    if lo is not None and v < lo:
        raise CatalogError(f'{name} must be >= {lo}, got {v}')
    return v


def real_array(v, name, *, nonneg=True):
    '''Finite real float64 array copy from numeric input; bool/complex/str/object inputs refused.'''
    if isinstance(v, (bool, np.bool_, str, bytes, complex)) or v is None:
        raise CatalogError(f'{name} must be real numeric, got {type(v).__name__}')
    a = np.asarray(v)
    if a.dtype.kind not in 'fiu':
        raise CatalogError(f'{name} must be real numeric (not bool/complex/str/object), got dtype {a.dtype}')
    a = np.array(a, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(a)):
        raise CatalogError(f'{name} must be finite')
    if nonneg and a.size and float(a.min()) < 0.0:
        raise CatalogError(f'{name} must be >= 0')
    return a


def _nonempty_str(v, name):
    if not isinstance(v, str) or not v:
        raise CatalogError(f'{name} must be a non-empty string, got {v!r}')
    return v


def _check_finite_tree(obj, name):
    '''Generic check of a preserved section: every number finite and no float NaN smuggled via Python objects.'''
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise CatalogError(f'{name}: object keys must be strings')
            _check_finite_tree(v, f'{name}.{k}')
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_finite_tree(v, f'{name}[{i}]')
    elif isinstance(obj, (bool, np.bool_)) or obj is None or isinstance(obj, str):
        return
    elif is_real(obj):
        if not math.isfinite(float(obj)):
            raise CatalogError(f'{name} must be finite')
    else:
        raise CatalogError(f'{name}: unsupported value type {type(obj).__name__}')


# ------------------------------------------------------------------ document validation
def _validate_law_section(sec, name):
    if sec is None:
        return
    if not isinstance(sec, dict):
        raise CatalogError(f'{name} must be null or an object with a string "law"')
    if 'law' not in sec:
        raise CatalogError(f'{name} must declare "law" (use "{UNIMPLEMENTED}" for an unselected law)')
    _nonempty_str(sec['law'], f'{name}.law')
    _check_finite_tree(sec, name)


def _validate_hydrology(hy, name, L):
    if not isinstance(hy, dict):
        raise CatalogError(f'{name} must be an object (null is refused: omit the block instead)')
    unknown = sorted(set(hy) - set(HYDROLOGY_KEYS))
    if unknown or RECEIVER_WEIGHTS_KEY not in hy:
        raise CatalogError(f'{name}: must hold {RECEIVER_WEIGHTS_KEY!r} and only {list(HYDROLOGY_KEYS)}; unknown keys {unknown}')
    w = hy[RECEIVER_WEIGHTS_KEY]
    if not isinstance(w, list) or len(w) != L:
        raise CatalogError(f'{name}.{RECEIVER_WEIGHTS_KEY} must be a list of exactly root_layers={L} finite nonnegative reals')
    for i, v in enumerate(w):
        nonnegative(v, f'{name}.{RECEIVER_WEIGHTS_KEY}[{i}]')
    if 'notes' in hy and hy['notes'] is not None:
        _check_finite_tree(hy['notes'], f'{name}.notes')


def _validate_species(sid, rec, L, version=SCHEMA_VERSION):
    name = f'species[{sid!r}]'
    if not isinstance(rec, dict):
        raise CatalogError(f'{name} must be an object')
    optional = OPTIONAL_SPECIES_KEYS + ((HYDROLOGY_SPECIES_KEY,) if version == SCHEMA_VERSION_HYDROLOGY else ())
    missing = [k for k in REQUIRED_SECTIONS if k not in rec]
    unknown = sorted(set(rec) - set(REQUIRED_SECTIONS) - set(optional))
    if missing or unknown:
        raise CatalogError(f'{name}: missing sections {missing}, unknown keys {unknown}')
    _nonempty_str(rec['functional_type'], f'{name}.functional_type')
    for k in OPTIONAL_SPECIES_KEYS:
        if k in rec and rec[k] is not None:
            _check_finite_tree(rec[k], f'{name}.{k}')
    if HYDROLOGY_SPECIES_KEY in rec:
        _validate_hydrology(rec[HYDROLOGY_SPECIES_KEY], f'{name}.{HYDROLOGY_SPECIES_KEY}', L)
    al = rec['allometry']
    if not isinstance(al, dict) or al.get('model') != ALLOMETRY_MODEL:
        raise CatalogError(f'{name}.allometry must be an object with model {ALLOMETRY_MODEL!r}')
    extra = sorted(set(al) - {'model', 'diameter', 'height', 'notes', 'units'})
    if extra:
        raise CatalogError(f'{name}.allometry: unknown keys {extra}')
    for law in ('diameter', 'height'):
        p = al.get(law)
        if not isinstance(p, dict) or sorted(set(p) - {'a', 'b', 'notes'}) or 'a' not in p or 'b' not in p:
            raise CatalogError(f'{name}.allometry.{law} must be an object with exactly "a" and "b"')
        positive(p['a'], f'{name}.allometry.{law}.a')
        positive(p['b'], f'{name}.allometry.{law}.b')
    if 'units' in al and al['units'] is not None:
        _check_finite_tree(al['units'], f'{name}.allometry.units')
    ro = rec['roots']
    if not isinstance(ro, dict) or 'lateral_radius_multiplier' not in ro or 'layer_fractions' not in ro:
        raise CatalogError(f'{name}.roots must hold lateral_radius_multiplier and layer_fractions')
    extra = sorted(set(ro) - {'lateral_radius_multiplier', 'layer_fractions', 'notes'})
    if extra:
        raise CatalogError(f'{name}.roots: unknown keys {extra}')
    positive(ro['lateral_radius_multiplier'], f'{name}.roots.lateral_radius_multiplier')
    fr = ro['layer_fractions']
    if not isinstance(fr, list) or len(fr) != L:
        raise CatalogError(f'{name}.roots.layer_fractions must be a list of exactly root_layers={L} fractions')
    vals = [nonnegative(v, f'{name}.roots.layer_fractions[{i}]') for i, v in enumerate(fr)]
    tot = math.fsum(vals)
    if abs(tot - 1.0) > LAYER_FRACTION_TOL:
        raise CatalogError(f'{name}.roots.layer_fractions must sum to 1 within {LAYER_FRACTION_TOL} (got {tot!r}); '
                           'not normalized silently')
    po = rec['porosity']
    if not isinstance(po, dict) or 'value' not in po or 'status' not in po:
        raise CatalogError(f'{name}.porosity must be an object with "value" in [0, 1] and a "status" label')
    unit_interval(po['value'], f'{name}.porosity.value')
    _nonempty_str(po['status'], f'{name}.porosity.status')
    _check_finite_tree(po, f'{name}.porosity')
    le = rec['life_expectancy']
    if le is not None:
        if not isinstance(le, dict) or 'mean' not in le or 'unit' not in le:
            raise CatalogError(f'{name}.life_expectancy must be null or an object with "mean" and "unit"')
        positive(le['mean'], f'{name}.life_expectancy.mean')
        if le['unit'] not in TIME_UNIT_DAYS:
            raise CatalogError(f'{name}.life_expectancy.unit must be one of {sorted(TIME_UNIT_DAYS)}')
        _check_finite_tree(le, f'{name}.life_expectancy')
    for k in LAW_SECTIONS:
        _validate_law_section(rec[k], f'{name}.{k}')


def document_version(document):
    '''The actual schema version of a document: 1 (legacy; a float 1.0 stays accepted exactly as before) or 2 (an exact
    integer only: 2.0, True and strings are refused). Anything else raises.'''
    ver = document.get('schema_version') if isinstance(document, dict) else None
    if isinstance(ver, (bool, np.bool_)):
        raise CatalogError(f'unsupported species catalog schema_version {ver!r} (supported: {list(SUPPORTED_SCHEMA_VERSIONS)})')
    if isinstance(ver, (int, float)) and ver == SCHEMA_VERSION:
        return SCHEMA_VERSION
    if type(ver) is int and ver == SCHEMA_VERSION_HYDROLOGY:
        return SCHEMA_VERSION_HYDROLOGY
    raise CatalogError(f'unsupported species catalog schema_version {ver!r} (supported: {list(SUPPORTED_SCHEMA_VERSIONS)})')


def validate_catalog_document(document):
    '''PURE validation of a parsed catalog document; returns (sorted species_ids, root_layers).'''
    if not isinstance(document, dict):
        raise CatalogError('species catalog must be a JSON object')
    unknown = sorted(set(document) - set(TOP_LEVEL_KEYS))
    if unknown:
        raise CatalogError(f'species catalog: unknown top-level keys {unknown}')
    if document.get('schema') != SCHEMA:
        raise CatalogError(f'species catalog schema must be {SCHEMA!r}, got {document.get("schema")!r}')
    version = document_version(document)
    units = document.get('units')
    if units != REQUIRED_UNITS:
        raise CatalogError(f'species catalog units must be exactly {REQUIRED_UNITS}, got {units!r}')
    if 'catalog_id' in document and document['catalog_id'] is not None:
        _nonempty_str(document['catalog_id'], 'catalog_id')
    if 'notes' in document and document['notes'] is not None:
        _check_finite_tree(document['notes'], 'notes')
    L = strict_int(document.get('root_layers'), 'root_layers', 1)
    sp = document.get('species')
    if not isinstance(sp, dict) or not sp:
        raise CatalogError('species catalog must hold a non-empty "species" object')
    for sid, rec in sp.items():
        if not isinstance(sid, str) or not SPECIES_ID_PATTERN.fullmatch(sid):
            raise CatalogError(f'species id {sid!r} must match {SPECIES_ID_PATTERN.pattern} (case-sensitive)')
        _validate_species(sid, rec, L, version)
    return tuple(sorted(sp.keys())), L


# ------------------------------------------------------------------ catalog object
class SpeciesCatalog:
    '''Validated species catalog with private content and vectorised allometry.

    species_ids: SORTED tuple (the species name/index table used by PlantState; see SPECIES_TABLE_ORDER).
    hash: canonical catalog hash accompanying every state.'''

    def __init__(self, document, *, source=None, sources=None, species_sources=None):
        self.species_ids, self.root_layers = validate_catalog_document(document)
        self.schema_version = document_version(document)       # the ACTUAL version of this catalog (1 or 2)
        self._document = copy.deepcopy(document)
        self.source = None if source is None else str(source)
        # provenance METADATA only (never hashed): the list of source files and the file each species came from
        if sources is None:
            sources = () if self.source is None else (self.source,)
        self.sources = tuple(str(s) for s in sources)
        if species_sources is None:
            species_sources = {} if self.source is None else {sid: self.source for sid in self.species_ids}
        self.species_sources = {str(k): str(v) for k, v in dict(species_sources).items()}
        self.hash = catalog_hash(self._document)
        self._index = {sid: i for i, sid in enumerate(self.species_ids)}
        sp = self._document['species']
        f64 = np.float64
        self._a_d = np.array([sp[s]['allometry']['diameter']['a'] for s in self.species_ids], f64)
        self._b_d = np.array([sp[s]['allometry']['diameter']['b'] for s in self.species_ids], f64)
        self._a_h = np.array([sp[s]['allometry']['height']['a'] for s in self.species_ids], f64)
        self._b_h = np.array([sp[s]['allometry']['height']['b'] for s in self.species_ids], f64)
        self._mult = np.array([sp[s]['roots']['lateral_radius_multiplier'] for s in self.species_ids], f64)
        self._por = np.array([sp[s]['porosity']['value'] for s in self.species_ids], f64)
        self._table = np.array([sp[s]['roots']['layer_fractions'] for s in self.species_ids], f64).reshape(
            len(self.species_ids), self.root_layers)
        # optional hydrology trait: a bounded (S, L) table + presence mask (S,) -- never a per-plant or per-cell map
        self._rw_has = np.array([isinstance(sp[s].get(HYDROLOGY_SPECIES_KEY), dict) for s in self.species_ids], dtype=bool)
        self._rw_table = np.zeros((len(self.species_ids), self.root_layers), f64)
        for i, s_ in enumerate(self.species_ids):
            if self._rw_has[i]:
                self._rw_table[i] = [float(v) for v in sp[s_][HYDROLOGY_SPECIES_KEY][RECEIVER_WEIGHTS_KEY]]
        for a in (self._a_d, self._b_d, self._a_h, self._b_h, self._mult, self._por, self._table, self._rw_has, self._rw_table):
            a.setflags(write=False)

    # ---- construction
    @classmethod
    def from_text(cls, text, *, source=None):
        return cls(parse_strict_json(text, what='species catalog'), source=source)

    @classmethod
    def from_file(cls, path):
        with open(path, 'r', encoding='utf-8') as fh:
            return cls.from_text(fh.read(), source=path)

    @classmethod
    def from_files(cls, paths):
        '''One catalog from one or more CO-LOCATED species files. Every file is a complete, independently
        valid catalog (same schema / units / root_layers) holding one or more COMPLETE species records
        (allometry, roots, growth, water_use, ... together); species are merged whole, never split or
        overridden across files, and a species ID appearing in two files is refused even when the records are
        identical. The merged species table is sorted, so the canonical hash is independent of the file order
        (and of key order inside files); with a single path the result is identical to from_file (same hash).
        With several files, top-level catalog_id / notes are kept only when every file carries a value with the
        SAME CANONICAL SERIALIZATION (1, 1.0 and true differ; otherwise omitted deterministically), so the merged
        hash never depends on the path order. Paths are provenance metadata (`sources`, `species_sources`), never
        hash input.'''
        if isinstance(paths, (str, bytes, os.PathLike)):
            raise CatalogError('from_files expects a sequence of catalog paths (use from_file for one path)')
        try:
            plist = [os.fspath(p) for p in paths]
        except TypeError as exc:
            raise CatalogError(f'from_files: every path must be str / PathLike ({exc})') from exc
        plist = [p.decode('utf-8') if isinstance(p, bytes) else p for p in plist]
        if not plist:
            raise CatalogError('from_files: at least one catalog file is required')
        if len({os.path.abspath(p) for p in plist}) != len(plist):
            raise CatalogError(f'from_files: a catalog file is listed more than once: {plist}')
        docs = []
        for p in plist:
            with open(p, 'r', encoding='utf-8') as fh:
                doc = parse_strict_json(fh.read(), what=f'species catalog {p!r}')
            ids, L = validate_catalog_document(doc)        # each file is a complete valid catalog on its own
            docs.append((p, doc, ids, L))
        if len(docs) == 1:
            p, doc, ids, _ = docs[0]
            return cls(doc, source=p, sources=(p,), species_sources={sid: p for sid in ids})
        L0 = docs[0][3]
        versions = [document_version(doc) for _, doc, _, _ in docs]
        if len(set(versions)) != 1:
            raise CatalogError('from_files: catalog files have different schema versions '
                               f'{[(p, v) for (p, _, _, _), v in zip(docs, versions)]}; versions are never mixed or migrated')
        species, species_sources = {}, {}
        for p, doc, ids, L in docs:
            if L != L0:
                raise CatalogError(f'root_layers differ across catalog files: {docs[0][0]!r} has {L0}, {p!r} has {L}')
            for sid in ids:
                if sid in species:
                    raise CatalogError(f'species {sid!r} is defined in both {species_sources[sid]!r} and {p!r}; a species '
                                       'must be complete in exactly one file (duplicates are refused even when identical)')
                species[sid] = copy.deepcopy(doc['species'][sid])
                species_sources[sid] = p
        merged = {'schema': SCHEMA, 'schema_version': versions[0], 'units': dict(REQUIRED_UNITS), 'root_layers': L0,
                  'species': {sid: species[sid] for sid in sorted(species)}}
        # top-level metadata is compared by CANONICAL SERIALIZATION (the text that is hashed), never by Python ==:
        # 1 == 1.0 == True in Python but they serialize differently, so keeping "the first equal value" would make
        # the merged hash depend on the path order. Differing (or missing) metadata is omitted deterministically.
        for key in ('catalog_id', 'notes'):
            if all(key in doc for _, doc, _, _ in docs):
                texts = [canonical_json(doc[key]) for _, doc, _, _ in docs]
                if all(t == texts[0] for t in texts):
                    merged[key] = copy.deepcopy(docs[0][1][key])
        return cls(merged, source=None, sources=tuple(plist), species_sources=species_sources)

    @property
    def document(self):
        '''Deep copy of the validated document (edits never reach the canonical content).'''
        return copy.deepcopy(self._document)

    @property
    def n_species(self):
        return len(self.species_ids)

    def canonical_text(self):
        return canonical_json(self._document)

    def __repr__(self):
        return f'SpeciesCatalog({list(self.species_ids)}, L={self.root_layers}, {self.hash[:19]}...)'

    # ---- lookup
    def has(self, sid):
        return isinstance(sid, str) and sid in self._index

    def index(self, sid):
        if isinstance(sid, str) and sid in self._index:
            return self._index[sid]
        raise CatalogError(f'unknown species {sid!r}; catalog species: {list(self.species_ids)}')

    def check_species_table(self, species_names):
        '''Refuse any consumer whose species name table differs from this catalog's sorted table.'''
        if tuple(species_names) != tuple(self.species_ids):
            raise CatalogError(f'species table {list(species_names)} differs from the catalog table {list(self.species_ids)}')

    def species_record(self, sid):
        return copy.deepcopy(self._document['species'][self.species_ids[self.index(sid)]])

    def section(self, sid, name):
        known = REQUIRED_SECTIONS + OPTIONAL_SPECIES_KEYS + ((HYDROLOGY_SPECIES_KEY,) if self.schema_version == SCHEMA_VERSION_HYDROLOGY else ())
        if name not in known:        # `hydrology` is known ONLY for an actual v2 catalog; for v1 it stays an unknown section
            raise CatalogError(f'unknown species section {name!r}')
        return copy.deepcopy(self._document['species'][self.species_ids[self.index(sid)]].get(name))

    def _indices(self, species):
        '''Species id / index / array of either -> int array (validated) or scalar int.'''
        if isinstance(species, str):
            return self.index(species)
        if isinstance(species, (bool, np.bool_)):
            raise CatalogError('species must be an id string or integer index, not bool')
        if isinstance(species, (int, np.integer)):
            i = int(species)
            if i < 0 or i >= self.n_species:
                raise CatalogError(f'species index {i} outside [0, {self.n_species})')
            return i
        arr = np.asarray(species)
        if arr.dtype.kind in 'US' or arr.dtype == object:
            return np.array([self.index(str(s)) for s in arr.ravel()], dtype=np.int64).reshape(arr.shape)
        if arr.dtype.kind not in 'iu':
            raise CatalogError('species must be ids or integer indices (not bool/float)')
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= self.n_species):
            raise CatalogError(f'species indices must lie in [0, {self.n_species})')
        return arr.astype(np.int64)

    # ---- allometry (vectorised; B and d in kg / m)
    @staticmethod
    def _finite_out(out, what):
        if not np.all(np.isfinite(out)):
            raise CatalogError(f'{what}: non-finite allometric output (overflow for the given input and exponent)')
        return float(out) if np.ndim(out) == 0 else out

    def diameter_from_biomass(self, species, agb_dry_kg):
        '''diameter_m = a_d * B**b_d (B = 0 -> 0); non-finite results refused.'''
        i = self._indices(species)
        B = real_array(agb_dry_kg, 'agb_dry_kg')
        with np.errstate(over='ignore', invalid='ignore'):
            out = self._a_d[i] * np.power(B, self._b_d[i])
        return self._finite_out(out, 'diameter_from_biomass')

    def height_from_biomass(self, species, agb_dry_kg):
        '''height_m = a_h * B**b_h (intact allometric height above the establishment ground).'''
        i = self._indices(species)
        B = real_array(agb_dry_kg, 'agb_dry_kg')
        with np.errstate(over='ignore', invalid='ignore'):
            out = self._a_h[i] * np.power(B, self._b_h[i])
        return self._finite_out(out, 'height_from_biomass')

    def biomass_from_diameter(self, species, diameter_m):
        '''Exact analytic inverse B_kg = (d / a_d) ** (1 / b_d) (d = 0 -> 0); non-finite results refused.'''
        i = self._indices(species)
        d = real_array(diameter_m, 'diameter_m')
        with np.errstate(over='ignore', invalid='ignore'):
            out = np.power(d / self._a_d[i], 1.0 / self._b_d[i])
        return self._finite_out(out, 'biomass_from_diameter')

    def allometry_arrays(self):
        return {'a_d': self._a_d.copy(), 'b_d': self._b_d.copy(), 'a_h': self._a_h.copy(), 'b_h': self._b_h.copy(),
                'model': ALLOMETRY_MODEL, 'units': {'a_d': 'm kg^-b_d', 'a_h': 'm kg^-b_h', 'b': 'dimensionless'}}

    # ---- roots / porosity / lifespan
    def root_multiplier_array(self):
        return self._mult.copy()

    def root_radius_multiplier(self, sid):
        return float(self._mult[self.index(sid)])

    def porosity_array(self):
        return self._por.copy()

    def porosity(self, sid):
        return float(self._por[self.index(sid)])

    def root_layer_table(self):
        '''FP64 [S, L] layer fractions (top layer first).'''
        return self._table.copy()

    def root_layer_fractions(self, sid):
        return [float(v) for v in self._table[self.index(sid)]]

    def life_expectancy_days(self, sid):
        '''Nominal mean life expectancy converted to days (years = 365), or None. Never triggers mortality by itself; it
        is the upper bound of the species-linked initial-age sampling of leaf.initialization.'''
        le = self._document['species'][self.species_ids[self.index(sid)]]['life_expectancy']
        if le is None:
            return None
        return float(le['mean']) * TIME_UNIT_DAYS[le['unit']]

    def law(self, sid, section):
        '''Declared law identifier of a law section (None when the section is null).'''
        if section not in LAW_SECTIONS:
            raise CatalogError(f'{section!r} is not a law section {LAW_SECTIONS}')
        sec = self._document['species'][self.species_ids[self.index(sid)]][section]
        return None if sec is None else sec['law']

    # ---- optional hydrology trait (schema_version 2): RAW specified preferential-infiltration receiving weights
    def has_preferential_receiver_weights(self, sid):
        '''True only when this species carries the explicit trait (never for a v1 catalog or a v2 species without the block).'''
        return bool(self._rw_has[self.index(sid)])

    def preferential_receiver_weights(self, sid):
        '''A fresh length-L float64 copy of the species' RAW specified weights (not normalised: a positive vector is normalised by
        SYRUP, an all-zero vector disables the species' pathway). Refuses a missing trait; no default vector is ever invented.'''
        i = self.index(sid)
        if not self._rw_has[i]:
            raise CatalogError(f'species {sid!r} has no {RECEIVER_WEIGHTS_KEY!r} trait (catalog schema_version {self.schema_version}); '
                               'it is never defaulted or inferred from functional_type or root fractions')
        return self._rw_table[i].copy()

    def preferential_receiver_weight_table(self, species=None):
        '''Fresh float64 [n, L] copy of the RAW weights of `species`: a list/tuple (or 1-D array) of actual species-id STRINGS in the
        requested order (default every species in the sorted table); repeated ids stay repeated and an empty sequence gives shape
        (0, L). A bare string, bool, number, None-element or any non-string id is refused (no coercion: id 1 is not '1'); unknown ids
        and ids without the trait are refused before the table is allocated.'''
        if species is None:
            idx = np.arange(self.n_species)
        else:
            if isinstance(species, np.ndarray):
                if species.ndim != 1:
                    raise CatalogError('species must be a one-dimensional sequence of species id strings')
                seq = species.tolist()
            elif isinstance(species, (list, tuple)):
                seq = list(species)
            else:
                raise CatalogError(f'species must be a list/tuple of species id strings (not {type(species).__name__}); a bare string is refused')
            idx = np.empty(len(seq), dtype=np.int64)
            for k, sid in enumerate(seq):
                if not isinstance(sid, str):
                    raise CatalogError(f'species[{k}] must be a species id string, got {type(sid).__name__} {sid!r} (ids are never coerced)')
                idx[k] = self.index(sid)                              # the strict existing lookup (unknown id -> CatalogError)
        missing = np.flatnonzero(~self._rw_has[idx])
        if missing.size:
            raise CatalogError(f'species {self.species_ids[int(idx[missing[0]])]!r} has no {RECEIVER_WEIGHTS_KEY!r} trait (catalog schema_version '
                               f'{self.schema_version}); it is never defaulted or inferred')
        return self._rw_table[idx].reshape(idx.size, self.root_layers)    # integer advanced indexing already returns ONE fresh float64 copy (never a view of the read-only table)

    def stepwat_traits(self, species=None):
        '''Extract {species id -> STEP trait row} for species whose growth law is STEPWAT_GROWTH_LAW with a
        "traits" object. Rows are returned as copies; leaf.stepwat.validate_traits (numba) validates them
        when the opt-in growth driver is constructed. Species with another/unimplemented growth law raise.'''
        ids = list(self.species_ids) if species is None else list(species)
        out = {}
        for sid in ids:
            sec = self._document['species'][self.species_ids[self.index(sid)]]['growth']
            if sec is None or sec.get('law') != STEPWAT_GROWTH_LAW or not isinstance(sec.get('traits'), dict):
                raise CatalogError(f'species {sid!r} has no growth.law == {STEPWAT_GROWTH_LAW!r} with a "traits" object')
            out[sid] = copy.deepcopy(sec['traits'])
        return out

    def metadata(self):
        return {'schema': SCHEMA, 'schema_version': self.schema_version, 'catalog_id': self._document.get('catalog_id'),
                'hash': self.hash, 'species_ids': list(self.species_ids), 'species_table_order': SPECIES_TABLE_ORDER,
                'root_layers': self.root_layers, 'units': dict(REQUIRED_UNITS), 'allometry_model': ALLOMETRY_MODEL,
                'leaf_version': LEAF_VERSION, 'source': self.source, 'sources': list(self.sources),
                'species_sources': dict(self.species_sources),
                'provenance_note': 'source paths are metadata only; the hash covers the canonical document'}


def load_catalog(path):
    '''One path -> SpeciesCatalog.from_file; a sequence of paths -> SpeciesCatalog.from_files.'''
    if isinstance(path, (str, bytes, os.PathLike)):
        return SpeciesCatalog.from_file(path)
    return SpeciesCatalog.from_files(path)


__all__ = ['LEAF_VERSION', 'SCHEMA', 'SCHEMA_VERSION', 'SCHEMA_VERSION_HYDROLOGY', 'SUPPORTED_SCHEMA_VERSIONS', 'HYDROLOGY_KEYS', 'RECEIVER_WEIGHTS_KEY',
           'HYDROLOGY_SPECIES_KEY', 'document_version', 'ALLOMETRY_MODEL', 'REQUIRED_UNITS', 'TIME_UNIT_DAYS',
           'LAYER_FRACTION_TOL', 'REQUIRED_SECTIONS', 'LAW_SECTIONS', 'UNIMPLEMENTED', 'STEPWAT_GROWTH_LAW',
           'SPECIES_TABLE_ORDER', 'CatalogError', 'parse_strict_json', 'canonical_json', 'catalog_hash',
           'validate_catalog_document', 'SpeciesCatalog', 'load_catalog', 'is_real', 'real', 'positive',
           'nonnegative', 'unit_interval', 'strict_int', 'real_array']
