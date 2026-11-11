'''leaf.checkpoint: complete, atomic, verified restart of an `IntereventSession` (UNQUALIFIED; no field data).

A checkpoint is a DIRECTORY published atomically (written under a hidden temporary name in the parent directory,
fsynced, then renamed; an existing target is never overwritten unless `overwrite=True`, and a half-written
checkpoint is never visible under the final name):

    plants.csv            canonical LEAF plant CSV (same format as initial / snapshot / final files) at the cursor:
                          stable IDs, species, X/Y, establishment ground, B, E, age and status of ALL plants
    restart_arrays.npz    typed arrays (np.load(allow_pickle=False)): native water [L, N] and its time, pending
                          requested / delivered water D, U [live], pending activity / temperature integrals [S],
                          canopy areas [live], live_rows, the EXACT current root CSR (ids, offsets, resources,
                          weights) and the ID-keyed layer-fraction overrides
    restart.json          schema / version, exact cursor and next growth boundary, deferred season flag, pending
                          days (exact Fraction text), sequence, native fingerprint (layers + mask + area), backend,
                          cadence, species / catalog hash, module source hashes, native pin digests, last receipt,
                          per-file sha256 and a per-array sha256 + a content hash over the manifest

Restoring a checkpoint reads every file into memory, verifies hashes, parses strictly (schema, shapes, dtypes,
finite values, time / boundary / pending consistency, ID mapping, root CSR validity through leaf.uptake) and by
default REBUILDS the roots / canopy areas from the restored population and compares them exactly with the stored
ones (`verify_roots=True`). Everything is validated BEFORE a session object exists: a corrupt or inconsistent
checkpoint returns nothing and adopts nothing. No silent migration: a different catalog, backend, native column,
root builder or module source is refused (`allow_source_change=True` only relaxes the module-source comparison).
The session is re-anchored at the saved cursor (anchor ages / time = the CSV values), so a continuous run and a
save / load split run agree to FP64 rounding of the age sum. Budgets (`max_edges`, `max_bytes`, `reserve_bytes`) are
supplied again by the caller and admitted globally (file sizes, hashing buffers, serialization buffers, host copies
of device arrays, the root map and the verification rebuild).

PHASE 2 state (accepted ecological state; same atomic / verified directory). `restart_arrays.npz` additionally holds
the live-sized per-plant stress counter (int32) and the four annual accumulators (float64: required support, actual
support, planned growth, actual growth); they are zero-length unless a stress law is active. `restart.json` gains a
`phase2` block: the RNG seed (decimal text), the annual decision counter, the disturbance counter, the year-due flag,
the start of the current decision period, the pending transport binding and the last decision record. The canonical
`plants.csv` remains the SINGLE plant-record format; the auxiliary arrays carry no plant identity of their own and follow
its live-row mapping (explicit limitation: they cannot be read without the CSV, and pending / annual decision state is
not representable in the CSV). A checkpoint written before Phase 2 (no `phase2` block / arrays) is refused: there is no
silent migration. Restoring verifies seed binding, counters, counter caps, finiteness, decision-period / binding
consistency and the active-law flags before a session object exists; a restored session continues with the same keyed
random draws as an uninterrupted run (only (seed, counter) are random state).
'''
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from fractions import Fraction

import numpy as np

from leaf import demography as _demo
from leaf import forcing as _forcing
from leaf import interevent as _ie
from leaf import state as _state
from leaf import syrup_water as _sw
from leaf import traits as _traits
from leaf import uptake as _uptake

SCHEMA = 'leaf.interevent_restart'
SCHEMA_VERSION = 2       # v2 = Phase 2 layout (five required arrays + `phase2` block); the plant CSV format stays version 1
CSV_NAME, NPZ_NAME, MANIFEST_NAME = 'plants.csv', 'restart_arrays.npz', 'restart.json'
SOURCE_MODULES = ('interevent', 'checkpoint', 'growth', 'roots', 'uptake', 'syrup_water', 'forcing', 'canopy',
                  'state', 'traits', 'demography', 'disturbance', 'events')
ARRAY_SPECS = {   # name -> (dtype, ndim)
    'native_water_m': (np.float64, 2), 'pending_requested_m3': (np.float64, 1), 'pending_delivered_m3': (np.float64, 1),
    'pending_activity_integral': (np.float64, 1), 'pending_temperature_integral': (np.float64, 1),
    'canopy_area_m2': (np.float64, 1), 'live_rows': (np.int64, 1), 'root_ids': (np.int64, 1),
    'root_offsets': (np.int64, 1), 'root_resources': (np.int64, 1), 'root_weights': (np.float64, 1),
    'override_ids': (np.int64, 1), 'override_rows': (np.float64, 2),
    # Phase 2 (zero-length unless a stress law is active): per-live-plant counter + annual accumulators
    'stress_counter': (np.int32, 1), 'annual_required_support_kg_year': (np.float64, 1),
    'annual_actual_support_kg_year': (np.float64, 1), 'annual_planned_growth_kg': (np.float64, 1),
    'annual_actual_growth_kg': (np.float64, 1)}
PHASE2_ARRAYS = ('stress_counter', 'annual_required_support_kg_year', 'annual_actual_support_kg_year',
                 'annual_planned_growth_kg', 'annual_actual_growth_kg')
PHASE2_SCHEMA = 'leaf.phase2_state'
PHASE2_KEYS = ('schema', 'demography_active', 'stress_active', 'disturbance_active', 'rng_seed', 'n_annual',
               'n_disturbance', 'n_population_rebinds', 'annual_due', 'annual_cursor', 'transport_pending',
               'last_decision')
TRANSPORT_PENDING_KEYS = ('sequence', 'kind', 'source', 'start', 'end', 'start_days', 'end_days', 'interval_id')
SERIALIZATION_NOTE = ('save: snapshot PlantState + CSV text (+ its UTF-8 bytes) + npz bytes + host copies of device '
                      'arrays are counted in the global ledger')


class CheckpointError(ValueError):
    '''A checkpoint cannot be written, is corrupt, or does not match the supplied catalog / operator / backend.'''


# ------------------------------------------------------------------ helpers
def _sha(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def _arr_hash(a):
    a = np.ascontiguousarray(a)
    return _sha(repr((str(a.dtype), a.shape)).encode() + a.tobytes())


def module_hashes():
    out = {}
    for name in SOURCE_MODULES:
        mod = sys.modules.get(f'leaf.{name}')
        path = getattr(mod, '__file__', None)
        if not path:
            out[name] = None
            continue
        with open(path, 'rb') as fh:
            out[name] = _sha(fh.read())
    return out


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_file(path, data):
    with open(path, 'wb') as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def estimate_save_bytes(session):
    '''Conservative serialization ledger of save_session (bytes): snapshot population (host, twice during
    construction), CSV text + encoded bytes + write buffer (~ 3 x 180 B per plant), host copies of device arrays,
    the npz buffer and its array copies.'''
    L, N, Pl, Pa, S = session._L, session._N, session._Pl, session._Pa, session._S
    rm = session._b.root_map
    E = 0 if rm is None else int(rm.n_edges)
    arrays = 8 * L * N + 8 * Pl * 4 + 16 * S + 8 * Pl + 8 * (Pl + 1) + 16 * E + 8 * (Pl + 1) + 36 * Pl
    snap = session._materialize_peak()            # structural materialisation peak incl. CUDA host B / E copies
    return {'snapshot_population': snap, 'csv_text_and_bytes': 3 * 180 * Pa,
            'host_array_copies': arrays, 'npz_buffer': arrays + 4096, 'manifest': 65536,
            'total_bytes': snap + 3 * 180 * Pa + 2 * arrays + 4096 + 65536}


# ------------------------------------------------------------------ save
def save_session(session, path, *, overwrite=False):
    '''Atomically publish a complete checkpoint directory (see the module docstring). Returns the path.'''
    if not isinstance(session, _ie.IntereventSession):
        raise CheckpointError('session must be a leaf.interevent.IntereventSession')
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path)) or '.'
    if not os.path.isdir(parent):
        raise FileNotFoundError(f'checkpoint parent directory {parent!r} does not exist')
    if os.path.lexists(path) and not overwrite:
        raise FileExistsError(f'{path!r} exists; pass overwrite=True to replace it')
    b = session._b
    est = estimate_save_bytes(session)
    persistent = _ie._sum_items(session._persistent(b))
    total = persistent + est['total_bytes']
    if total > session.max_bytes:
        raise _ie.IntereventAdmissionError('checkpoint_save', total, session.max_bytes, est)
    snap = session.snapshot()
    csv_text = _state.plant_csv_text(snap, provenance=dict(snap.provenance, checkpoint=SCHEMA))
    csv_bytes = csv_text.encode('utf-8')
    del csv_text
    rm = b.root_map
    arrays = {'native_water_m': np.ascontiguousarray(_ie._hostarr(b.native_state.water_m), dtype=np.float64),
              'pending_requested_m3': np.ascontiguousarray(_ie._hostarr(b.D), dtype=np.float64),
              'pending_delivered_m3': np.ascontiguousarray(_ie._hostarr(b.U), dtype=np.float64),
              'pending_activity_integral': np.ascontiguousarray(b.A, dtype=np.float64),
              'pending_temperature_integral': np.ascontiguousarray(b.T, dtype=np.float64),
              'canopy_area_m2': (np.ascontiguousarray(_ie._hostarr(b.area), dtype=np.float64) if b.area is not None
                                 else np.zeros(0)),
              'live_rows': np.ascontiguousarray(session._live_rows, dtype=np.int64)}
    if rm is not None:
        host = rm.host_arrays()
        arrays.update(root_ids=host['ids'], root_offsets=host['offsets'], root_resources=host['resources'],
                      root_weights=host['weights'])
        del host
    else:
        arrays.update(root_ids=np.zeros(0, np.int64), root_offsets=np.zeros(1, np.int64),
                      root_resources=np.zeros(0, np.int64), root_weights=np.zeros(0))
    ov = session._overrides or {}
    arrays['override_ids'] = np.array(sorted(ov), dtype=np.int64)
    arrays['override_rows'] = np.array([ov[k] for k in sorted(ov)], dtype=np.float64).reshape(len(ov), session._L)
    if session._stress_on and b.stress is not None:          # Phase 2 accepted per-plant state (explicit host copies)
        arrays['stress_counter'] = np.ascontiguousarray(_ie._hostarr(b.stress), dtype=np.int32)
        arrays['annual_required_support_kg_year'] = np.ascontiguousarray(_ie._hostarr(b.ann_req), dtype=np.float64)
        arrays['annual_actual_support_kg_year'] = np.ascontiguousarray(_ie._hostarr(b.ann_act), dtype=np.float64)
        arrays['annual_planned_growth_kg'] = np.ascontiguousarray(_ie._hostarr(b.ann_plan), dtype=np.float64)
        arrays['annual_actual_growth_kg'] = np.ascontiguousarray(_ie._hostarr(b.ann_gain), dtype=np.float64)
    else:
        arrays['stress_counter'] = np.zeros(0, dtype=np.int32)
        for k in PHASE2_ARRAYS[1:]:
            arrays[k] = np.zeros(0, dtype=np.float64)
    phase2 = {'schema': PHASE2_SCHEMA, 'demography_active': bool(session._demo_on),
              'stress_active': bool(session._stress_on), 'disturbance_active': bool(session._dist.any_active),
              'rng_seed': None if session._rng_seed is None else str(int(session._rng_seed)),
              'n_annual': int(b.n_annual or 0), 'n_disturbance': int(b.n_disturbance or 0),
              'n_population_rebinds': int(b.n_population_rebinds or 0), 'annual_due': b.annual_due,
              'annual_cursor': b.annual_cursor.isoformat(),
              'transport_pending': None if b.transport_pending is None else dict(b.transport_pending),
              'last_decision': None if b.last_decision is None else dict(b.last_decision)}
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    npz_bytes = buf.getvalue()
    del buf
    off = _forcing.offset_seconds_exact(session._offset)
    manifest = {
        'schema': SCHEMA, 'schema_version': SCHEMA_VERSION, 'leaf_interevent_version': _ie.VERSION,
        'scope': _ie.SCOPE, 'backend': session.backend,
        'config': {'growth_cadence_us': _forcing.exact_microseconds(_forcing.exact_seconds(session._cadence)),
                   'utc_offset_seconds_exact': str(off), 'root_builder': session.root_builder,
                   'n_layers': session._L, 'n_cells': session._N, 'cell_area_m2': session._area,
                   'n_plants_total': session._Pa, 'n_live': session._Pl, 'n_species': session._S,
                   'species_names': list(snap.species_names), 'catalog_hash': snap.catalog_hash,
                   'native_fingerprint': _ie.native_fingerprint(session._column),
                   'domain': snap.domain.to_dict(), 'has_root_map': rm is not None,
                   'saved_max_edges': session.max_edges, 'saved_max_bytes': session.max_bytes,
                   'budgets_note': 'informational only: budgets are supplied again at load'},
        'state': {'cursor': b.cursor.isoformat(), 'next_growth_boundary': b.next_boundary.isoformat(),
                  'season_due': bool(b.season_due), 'pending_days': str(b.pending_days), 'sequence': int(b.sequence),
                  'native_t_s': float(b.native_state.t_s), 'native_anchor_t_s': float(session._native_anchor_t),
                  'native_elapsed_offset_s': float(b.native_state.t_s) - float(session._native_anchor_t),
                  'time_days': snap.time_days, 'n_growth': int(b.n_growth), 'n_rebuilds': int(b.n_rebuilds),
                  'original_anchor': session._anchor.isoformat(), 'original_time_days': session._time0},
        'last_receipt': b.last_receipt, 'last_growth': b.last_growth, 'phase2': phase2,
        'sources': {'modules': module_hashes(), 'native_pin': _ie._clean(session._provenance.get('pin')),
                    'native_files_sha256': _ie._clean(session._provenance.get('sha256'))},
        'files': {CSV_NAME: _sha(csv_bytes), NPZ_NAME: _sha(npz_bytes)},
        'array_hashes': {k: _arr_hash(v) for k, v in arrays.items()},
        'serialization_note': SERIALIZATION_NOTE}
    manifest = _ie._clean(manifest)
    manifest['content_hash'] = _sha(_canonical(manifest))
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False) + '\n').encode('utf-8')
    tmp = tempfile.mkdtemp(prefix='.leaf-checkpoint-', suffix='.tmp', dir=parent)
    try:
        _write_file(os.path.join(tmp, CSV_NAME), csv_bytes)
        _write_file(os.path.join(tmp, NPZ_NAME), npz_bytes)
        _write_file(os.path.join(tmp, MANIFEST_NAME), manifest_bytes)
        _fsync_dir(tmp)
        os.chmod(tmp, 0o755)
        if os.path.lexists(path):
            if not overwrite:
                raise FileExistsError(f'{path!r} appeared while writing; not overwritten')
            old = tempfile.mkdtemp(prefix='.leaf-checkpoint-old-', dir=parent)
            os.rmdir(old)
            os.rename(path, old)
            try:
                os.rename(tmp, path)
            except BaseException:
                os.rename(old, path)
                raise
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.rename(tmp, path)       # atomic publication of the complete directory
        _fsync_dir(parent)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return path


# ------------------------------------------------------------------ load
def _read(path, name):
    p = os.path.join(path, name)
    try:
        with open(p, 'rb') as fh:
            return fh.read()
    except OSError as exc:
        raise CheckpointError(f'checkpoint file {p!r} cannot be read: {exc}') from exc


def _need(d, key, where):
    if not isinstance(d, dict) or key not in d:
        raise CheckpointError(f'checkpoint {where} lacks {key!r}')
    return d[key]


def load_session(path, catalog, *, operator, backend, max_edges, max_bytes, reserve_bytes=0, root_builder=None,
                 verify_roots=True, allow_source_change=False, verify_native_sources=True, rng_seed=None):
    '''Restore a complete accepted boundary (module docstring). Nothing is adopted unless every check passes. The RNG
    seed is part of the accepted state and is read from the checkpoint; a caller-supplied `rng_seed` is only a
    consistency check (a different seed refuses).'''
    # scalar budget arguments first: strict non-boolean integers, no float truncation, no negative reserve
    max_edges = _ie._real_pos_int(max_edges, 'max_edges', 0)
    max_bytes = _ie._real_pos_int(max_bytes, 'max_bytes', 1)
    reserve_bytes = _ie._real_pos_int(reserve_bytes, 'reserve_bytes', 0)
    path = os.fspath(path)
    if not os.path.isdir(path):
        raise CheckpointError(f'{path!r} is not a checkpoint directory')
    sizes = {n: os.path.getsize(os.path.join(path, n)) for n in (CSV_NAME, NPZ_NAME, MANIFEST_NAME)
             if os.path.exists(os.path.join(path, n))}
    if set(sizes) != {CSV_NAME, NPZ_NAME, MANIFEST_NAME}:
        raise CheckpointError(f'checkpoint {path!r} is incomplete: files {sorted(sizes)}')
    mb = max_bytes
    need = 4 * sizes[CSV_NAME] + 3 * sizes[NPZ_NAME] + 2 * sizes[MANIFEST_NAME] + reserve_bytes
    if need > mb:
        raise _ie.IntereventAdmissionError('checkpoint_load_read', need, mb, {'file_sizes': sizes})
    mbytes = _read(path, MANIFEST_NAME)
    try:
        man = _traits.parse_strict_json(mbytes, what='restart manifest')
    except _traits.CatalogError as exc:
        raise CheckpointError(str(exc)) from exc
    if not isinstance(man, dict):
        raise CheckpointError(f'restart manifest must be a JSON object, got {type(man).__name__}')
    ver = man.get('schema_version')
    if man.get('schema') != SCHEMA or type(ver) is not int or ver != SCHEMA_VERSION:   # a real int: not 2.0, not bool
        raise CheckpointError(
            f'unsupported restart format: schema {man.get("schema")!r} version {ver!r} ({type(ver).__name__}); this loader '
            f'reads exactly schema {SCHEMA!r} integer version {SCHEMA_VERSION}. Older (version 1) checkpoints lack the '
            'Phase 2 arrays and manifest block and are NOT migrated or defaulted (allow_source_change does not authorize '
            'a format conversion; use the pinned Phase 1 runtime that wrote them)')
    body = {k: v for k, v in man.items() if k != 'content_hash'}
    if man.get('content_hash') != _sha(_canonical(body)):
        raise CheckpointError('manifest content hash mismatch (corrupt or edited manifest)')
    csv_bytes, npz_bytes = _read(path, CSV_NAME), _read(path, NPZ_NAME)
    files = _need(man, 'files', 'manifest')
    if _sha(csv_bytes) != files.get(CSV_NAME) or _sha(npz_bytes) != files.get(NPZ_NAME):
        raise CheckpointError('file hash mismatch (corrupt plants.csv or restart_arrays.npz)')
    cfg, st = _need(man, 'config', 'manifest'), _need(man, 'state', 'manifest')
    if man.get('backend') != backend:
        raise CheckpointError(f'checkpoint backend {man.get("backend")!r} != requested {backend!r}; no silent backend migration')
    if cfg.get('catalog_hash') != catalog.hash or list(cfg.get('species_names', [])) != list(catalog.species_ids):
        raise CheckpointError('catalog hash / species table differs from the checkpoint; no silent catalog migration')
    if root_builder is not None and root_builder != cfg.get('root_builder'):
        raise CheckpointError(f'root builder {root_builder!r} != checkpoint {cfg.get("root_builder")!r}')
    builder = cfg.get('root_builder')
    column, xp, op_backend, L, N, area = _sw.check_operator(operator)
    if op_backend != backend:
        raise CheckpointError('operator backend differs from the requested backend')
    if cfg.get('native_fingerprint') != _ie.native_fingerprint(column):
        raise CheckpointError('native column fingerprint (layers / mask / area) differs from the checkpoint')
    if (cfg.get('n_layers'), cfg.get('n_cells')) != (L, N):
        raise CheckpointError('native column dimensions differ from the checkpoint')
    src = _need(man, 'sources', 'manifest')
    if not allow_source_change and src.get('modules') != _ie._clean(module_hashes()):
        raise CheckpointError('LEAF module sources differ from the checkpoint (allow_source_change=True to relax)')
    if verify_native_sources:
        prov = _sw.native_provenance(verify=True)
        if _ie._clean(prov.get('sha256')) != src.get('native_files_sha256'):
            raise CheckpointError('native SYRUP source hashes differ from the checkpoint; no silent native migration')
    # ---- arrays: strict typed load
    try:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz:
            arrays = {k: np.array(npz[k]) for k in npz.files}
    except Exception as exc:    # noqa: BLE001 - any parse failure is a corrupt checkpoint
        raise CheckpointError(f'restart_arrays.npz cannot be read without pickle: {exc}') from exc
    del npz_bytes, mbytes               # raw bytes are redundant once decoded (hashes were verified above)
    if set(arrays) != set(ARRAY_SPECS):
        raise CheckpointError(f'array set {sorted(arrays)} != {sorted(ARRAY_SPECS)}')
    ah = _need(man, 'array_hashes', 'manifest')
    for k, (dt, nd) in ARRAY_SPECS.items():
        a = arrays[k]
        if a.dtype != np.dtype(dt) or a.ndim != nd:
            raise CheckpointError(f'array {k} has dtype {a.dtype} / ndim {a.ndim}, expected {np.dtype(dt)} / {nd}')
        if a.dtype.kind == 'f' and not np.all(np.isfinite(a)):
            raise CheckpointError(f'array {k} holds non-finite values')
        if ah.get(k) != _arr_hash(a):
            raise CheckpointError(f'array {k} hash mismatch')
    # ---- plants
    try:
        plants = _state.parse_plant_csv(csv_bytes, catalog, source=os.path.join(path, CSV_NAME))
    except _state.StateError as exc:
        raise CheckpointError(f'plants.csv invalid: {exc}') from exc
    del csv_bytes                       # redundant once parsed (released before the session is initialised)
    plants.require_resolved('restoring a session')
    if plants.n != cfg.get('n_plants_total') or plants.time_days != st.get('time_days'):
        raise CheckpointError('plants.csv population size / time differs from the manifest')
    live_rows = np.flatnonzero(plants.status == _state.LIVE).astype(np.int64)
    if not np.array_equal(live_rows, arrays['live_rows']) or int(live_rows.shape[0]) != cfg.get('n_live'):
        raise CheckpointError('live-plant row mapping differs between plants.csv and the arrays')
    Pl = int(live_rows.shape[0])
    # ---- scalars / boundary consistency
    try:
        cursor = _dt.datetime.fromisoformat(st['cursor'])
        nb = _dt.datetime.fromisoformat(st['next_growth_boundary'])
        anchor0 = _dt.datetime.fromisoformat(st['original_anchor'])
        pending = Fraction(st['pending_days'])
        cadence = _dt.timedelta(microseconds=int(cfg['growth_cadence_us']))
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        raise CheckpointError(f'invalid boundary scalars: {exc}') from exc
    _forcing.check_fixed_offset_datetime(cursor, 'cursor')
    _forcing.check_fixed_offset_datetime(nb, 'next_growth_boundary')
    if str(_forcing.offset_seconds_exact(cursor.utcoffset())) != cfg.get('utc_offset_seconds_exact') \
            or nb.utcoffset() != cursor.utcoffset():
        raise CheckpointError('UTC offset inconsistent between cursor, boundary and manifest')
    if not _dt.timedelta(0) < cadence <= _ie.MAX_CADENCE:
        raise CheckpointError('growth cadence outside (0, 7 days]')
    if not cursor < nb <= cursor + cadence:
        raise CheckpointError('next growth boundary must lie in (cursor, cursor + cadence]')
    if pending < 0 or pending > Fraction(_forcing.exact_seconds(cadence), _ie.SECONDS_PER_DAY):
        raise CheckpointError('pending days outside [0, cadence]')
    if not isinstance(st.get('season_due'), bool):
        raise CheckpointError('season_due must be a boolean')
    if st['season_due'] and pending != 0:
        raise CheckpointError('a deferred season turnover cannot coexist with pending growth')
    seq = st.get('sequence')
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise CheckpointError('invalid sequence')
    ov_ids, ov_rows = arrays['override_ids'], arrays['override_rows']
    if ov_rows.shape != (ov_ids.shape[0], L):
        raise CheckpointError('override table shape mismatch')
    overrides = {int(k): tuple(float(x) for x in r) for k, r in zip(ov_ids, ov_rows)} or None
    water = arrays['native_water_m']
    if water.shape != (L, N):
        raise CheckpointError(f'native water shape {water.shape} != ({L}, {N})')
    D, U, A, T, ar = (arrays[k] for k in ('pending_requested_m3', 'pending_delivered_m3', 'pending_activity_integral',
                                          'pending_temperature_integral', 'canopy_area_m2'))
    S = catalog.n_species
    if D.shape != (Pl,) or U.shape != (Pl,) or ar.shape != ((Pl,) if Pl else (0,)) or A.shape != (S,) or T.shape != (S,):
        raise CheckpointError('pending accumulator / canopy area shapes do not match the live population')
    if Pl and (float(D.min()) < 0.0 or float(U.min()) < 0.0 or bool(np.any(U > D * (1.0 + 1e-6))) or float(ar.min()) < 0.0):
        raise CheckpointError('pending water / canopy area values are inconsistent (negative or delivered > requested)')
    if float(A.min()) < 0.0 or float(T.min()) < 0.0 or bool(np.any(T > A * (1.0 + 1e-12))) \
            or bool(np.any(A > float(pending) / 365.0 * (1.0 + 1e-9) + 1e-300)):
        raise CheckpointError('pending integrals are inconsistent with the pending days')
    if pending == 0 and (bool(np.any(A != 0.0)) or bool(np.any(T != 0.0)) or (Pl and (bool(np.any(D != 0.0))
                                                                                    or bool(np.any(U != 0.0))))):
        raise CheckpointError('pending accumulators are non-zero although no growth is pending')
    # ---- Phase 2 accepted ecological state: strict parse BEFORE any session object exists
    p2 = _need(man, 'phase2', 'manifest')
    if not isinstance(p2, dict) or set(p2) != set(PHASE2_KEYS) or p2.get('schema') != PHASE2_SCHEMA:
        raise CheckpointError('manifest phase2 block is missing or malformed (checkpoints written before Phase 2 are '
                              'refused: no silent migration)')
    for k in ('demography_active', 'stress_active', 'disturbance_active'):
        if not isinstance(p2[k], bool):
            raise CheckpointError(f'phase2.{k} must be a boolean')
    seed_txt = p2['rng_seed']
    if seed_txt is None:
        seed = None
    elif isinstance(seed_txt, str) and seed_txt.isdigit() and int(seed_txt) < (1 << 64):
        seed = int(seed_txt)
    else:
        raise CheckpointError('phase2.rng_seed must be null or a decimal integer text below 2**64')
    if rng_seed is not None:
        try:
            given = _demo.check_seed(rng_seed)
        except _demo.DemographyError as exc:
            raise CheckpointError(str(exc)) from exc
        if seed is None or given != seed:
            raise CheckpointError('the supplied rng_seed differs from the checkpoint seed (the seed is accepted state)')
    for k in ('n_annual', 'n_disturbance', 'n_population_rebinds'):
        if isinstance(p2[k], bool) or not isinstance(p2[k], int) or p2[k] < 0:
            raise CheckpointError(f'phase2.{k} must be a non-negative integer')
    if p2['annual_due'] is not None and (isinstance(p2['annual_due'], bool) or not isinstance(p2['annual_due'], int)):
        raise CheckpointError('phase2.annual_due must be null or an integer year')
    try:
        annual_cursor = _dt.datetime.fromisoformat(p2['annual_cursor'])
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f'phase2.annual_cursor invalid: {exc}') from exc
    _forcing.check_fixed_offset_datetime(annual_cursor, 'annual_cursor')
    if annual_cursor.utcoffset() != cursor.utcoffset() or annual_cursor > cursor:
        raise CheckpointError('phase2.annual_cursor must share the cursor offset and not lie after the cursor')
    tp = p2['transport_pending']
    if tp is not None:
        if not isinstance(tp, dict) or set(tp) != set(TRANSPORT_PENDING_KEYS) or tp['kind'] not in _ie.TRANSPORT_KINDS \
                or tp['source'] != _ie.TRANSPORT_SOURCE.get(tp['kind']) or tp['end'] != cursor.isoformat() \
                or not isinstance(tp['sequence'], int) or isinstance(tp['sequence'], bool) \
                or any(isinstance(tp[k], bool) or not isinstance(tp[k], (int, float)) for k in ('start_days', 'end_days')) \
                or not (tp['start_days'] < tp['end_days']) or not isinstance(tp['interval_id'], str) \
                or not isinstance(tp['start'], str):
            raise CheckpointError('phase2.transport_pending is malformed or does not end at the cursor')
        if pending != 0:
            raise CheckpointError('a pending transport binding cannot coexist with pending growth')
    ld = p2['last_decision']
    if ld is not None and (not isinstance(ld, dict) or ld.get('kind') not in ('disturbance', 'annual_decision')):
        raise CheckpointError('phase2.last_decision is malformed')
    if p2['annual_due'] is not None and pending != 0:
        raise CheckpointError('an outstanding annual decision cannot coexist with pending growth')
    # ---- the session object (nothing is returned unless everything passes)
    sess = _ie.IntereventSession.__new__(_ie.IntereventSession)
    # decoded arrays (and the parsed plants, counted by _setup as the persistent population) stay alive during the
    # whole restore: carry them as a transient reserve through EVERY allocating restore stage
    sess._transient_reserve = int(sum(int(a.nbytes) for a in arrays.values()))
    t_native = float(st['native_t_s'])
    anchor_t = float(st['native_t_s'])
    sess._setup(base=plants, catalog=catalog, operator=operator, backend=backend, cadence=cadence, max_edges=max_edges,
                max_bytes=max_bytes, reserve_bytes=reserve_bytes, root_builder=builder, overrides=overrides,
                anchor=cursor, native_anchor_t=anchor_t, verify_native=verify_native_sources, native_state=None,
                rng_seed=seed)
    sess._original_anchor = anchor0
    if (p2['demography_active'], p2['stress_active'], p2['disturbance_active']) != (
            bool(sess._demo_on), bool(sess._stress_on), bool(sess._dist.any_active)):
        raise CheckpointError('the active Phase 2 laws of the catalog differ from the checkpoint flags')
    if p2['annual_due'] is not None and not sess._demo_on:
        raise CheckpointError('an annual decision is recorded as due but no mortality law is active')
    p2_arrays = [arrays[k] for k in PHASE2_ARRAYS]
    if sess._stress_on:
        if any(a.shape != (Pl,) for a in p2_arrays):
            raise CheckpointError('stress counter / annual accumulator shapes do not match the live population')
        if Pl:
            cap = sess._demo.counter_cap()[plants.species_index[live_rows].astype(np.int64)]
            if int(p2_arrays[0].min()) < 0 or bool(np.any(p2_arrays[0] > cap)) \
                    or any(float(a.min()) < 0.0 for a in p2_arrays[1:]):
                raise CheckpointError('stress counters exceed their species cap or an annual accumulator is negative')
    elif any(a.shape != (0,) for a in p2_arrays):
        raise CheckpointError('stress / annual arrays must be empty when no stress law is active')
    ss, _ = _sw._native()
    hw = xp.asarray(water) if backend == 'cuda' else np.array(water, copy=True)
    P0 = sess._persistent(None)
    sess._check_stage('restart_native', _ie._sum_items(P0, ('root_map',)) + 3 * 8 * L * N, {})
    nstate = ss.column_state(column, hw, t_native)
    del hw
    _sw.availability_snapshot(operator, nstate, max_bytes=mb, extra_reserve_bytes=_ie._sum_items(P0, ('native_resident',)))
    b = _ie._Boundary()
    b.native_state = nstate
    b.A, b.T = np.array(A, copy=True), np.array(T, copy=True)
    b.pending_days = pending
    b.cursor, b.next_boundary = cursor, nb
    b.season_due, b.sequence = bool(st['season_due']), int(seq)
    b.last_receipt, b.last_growth = man.get('last_receipt'), man.get('last_growth')
    b.n_growth, b.n_rebuilds = int(st.get('n_growth', 0)), int(st.get('n_rebuilds', 0))
    b.n_annual, b.n_disturbance, b.n_population_rebinds = p2['n_annual'], p2['n_disturbance'], p2['n_population_rebinds']
    b.annual_due, b.annual_cursor = p2['annual_due'], annual_cursor
    b.transport_pending = None if tp is None else dict(tp)
    b.last_decision = None if ld is None else dict(ld)
    if sess._stress_on:
        if Pl:
            (b.stress, b.ann_req, b.ann_act, b.ann_plan, b.ann_gain) = (sess._to_backend(a) for a in p2_arrays)
        else:
            b.stress, b.ann_req, b.ann_act, b.ann_plan, b.ann_gain = sess._zero_accumulators(0)
    xpl = sess._xp
    if Pl:
        Bh = np.ascontiguousarray(plants.agb_dry_kg[live_rows])
        Eh = np.ascontiguousarray(plants.seasonal_agb_dry_kg[live_rows])
        b.B, b.E = sess._to_backend(Bh), sess._to_backend(Eh)
        b.D, b.U = sess._to_backend(D), sess._to_backend(U)
        # P0 already carries the decoded arrays (restore_transient_reserve), native copies, B/E/D/U on the backend,
        # the persistent population; + the host Bh / Eh gathers (16 Pl)
        live_other = _ie._sum_items(P0, ('root_map',)) + 16 * Pl
        budget = mb - live_other
        if budget <= 0:
            raise _ie.IntereventAdmissionError('restart_root_map', live_other, mb)
        try:
            rm = _uptake.RootMap.from_arrays(arrays['root_ids'], arrays['root_offsets'], arrays['root_resources'],
                                             arrays['root_weights'], n_layers=L, n_cells=N, backend=backend,
                                             max_edges=int(max_edges), max_bytes=int(budget),
                                             provenance={'restored_from': SCHEMA})
        except _uptake.UptakeError as exc:
            raise CheckpointError(f'stored root CSR is invalid: {exc}') from exc
        if not np.array_equal(arrays['root_ids'], sess._live_ids):
            raise CheckpointError('root map ids differ from the live-plant row mapping')
        br = _sw.estimate_bridge_bytes(Pl, rm.n_edges, L, N, map_backend=backend, native_backend=backend,
                                       demand_bytes=8 * Pl, map_resident_bytes=int(rm.resident_bytes),
                                       column_reserve_bytes=int(column.reserve_bytes), extra_reserve_bytes=0)
        if int(br['allocator_peak_bytes']) > int(rm.max_bytes):
            raise _ie.IntereventAdmissionError('restart_allocator_budget', br['allocator_peak_bytes'], rm.max_bytes)
        rtol = _ie.IntereventSession._overdelivery_rtol(rm)
        if bool(np.any(U > D * (1.0 + rtol))):
            raise CheckpointError('pending delivered water exceeds requested water beyond the allocator bound')
        b.root_map = rm
        b.area = sess._to_backend(np.ascontiguousarray(ar, dtype=np.float64))
        if verify_roots:
            rm2, area2 = sess._build_geometry(Bh, Eh, cursor, live_other=_ie._sum_items(P0, ('root_map',)) + 16 * Pl
                                              + int(rm.resident_bytes) + 8 * Pl, mem={}, stage='restart_verify')
            # the comparison makes host copies of BOTH maps while both maps, both areas and the decoded arrays are live
            sess._check_stage('restart_verify_compare', _ie._sum_items(P0, ('root_map',)) + 16 * Pl
                              + 2 * int(rm.resident_bytes) + 2 * int(rm2.resident_bytes) + 8 * Pl, {})
            h1, h2 = rm.host_arrays(), rm2.host_arrays()
            for k in ('ids', 'offsets', 'resources', 'weights'):
                if not np.array_equal(h1[k], h2[k]):
                    raise CheckpointError(f'stored root CSR {k} differs from the rebuild of the restored population '
                                          '(inconsistent roots / geometry / overrides)')
            if not np.array_equal(_ie._hostarr(area2), np.ascontiguousarray(ar)):
                raise CheckpointError('stored canopy areas differ from the restored geometry')
            del rm2, area2, h1, h2
    else:
        z = xpl.zeros(0, dtype=np.float64)
        b.B = b.E = b.D = b.U = z
        b.root_map, b.area = None, None
    # the decoded arrays are no longer needed: release them and the transient reserve; the caller's reserve_bytes and
    # max_bytes stay exactly as supplied (no hidden limit change)
    del arrays, water, D, U, A, T, ar, p2_arrays
    sess._transient_reserve = 0
    sess._b = b
    return sess


__all__ = ['SCHEMA', 'SCHEMA_VERSION', 'CSV_NAME', 'NPZ_NAME', 'MANIFEST_NAME', 'CheckpointError', 'module_hashes',
           'estimate_save_bytes', 'save_session', 'load_session']
