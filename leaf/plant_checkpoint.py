'''leaf.plant_checkpoint: BIOLOGICAL-ONLY save / restore of a `leaf.plant_session.PlantSession` (versioned separate codec, version 2;
UNQUALIFIED). It does NOT touch `leaf.checkpoint` (standalone layout version 4) or any historical lock.

A checkpoint is a NEW directory (never overwritten; written under a private temporary sibling and renamed once):

    plants.csv + plants.metadata.json   canonical PLAIN nine-column plant file at the schedule cursor + companion (leaf.state); ALL rows
                                        (live, dead_standing, removed, newborns)
    biology_arrays.npz                  pending D, U [P] (m3), A, T [S] (years), the Phase 2 stress counter (int32) and four annual
                                        accumulators [P] when a mortality law is active, and - only with a pending stage - `resume_D`,
                                        `resume_supply_m3`, `resume_actual_m3`, `resume_unmet_m3` ([P] / [L,N]); host arrays, no pickle
    manifest.json                       format / version, session id, LINEAGE, sequence, token, exact calendar schedule, counters, Phase 2 / 3
                                        state (`bio`: annual due / cursor, transport binding, decision / disturbance / reproduction counters,
                                        keyed-RNG counters), day binding, population + geometry content identities, catalog hash, last stage
                                        id, the LAST CONSUMED T resource (`resource_cursor`) and, separately, the EXACT external SYRUP reference
                                        (`external_reference`, e.g. the boundary closed after that T), optional pending-stage metadata, file
                                        sizes + sha256, SOURCE sha256 of the LEAF modules involved and a self digest

NO water is saved or restored. `load_biology` requires `ExternalBindings.external`: the `ResourceIdentity` of the RESTORED genuine SYRUP state
(`lifecycle_bridge.resource_identity`), which must EQUAL the saved external reference (the combined pairing), not merely be a descendant. Live
capabilities are minted again by the caller from restored genuine states; none is saved.

Byte caps. Save and load admit every copy against the session's global `max_bytes` (host + device, + `reserve_bytes`, + the current parent bytes
for load) BEFORE materializing anything: file sizes are taken from `stat` / array metadata, npz members are validated from their headers (shape,
dtype, expected size) before any array is read, hashes stream in bounded chunks, no BytesIO copy of an unbounded buffer exists.

Exactly-once resume. If SYRUP accepted a debit but the LEAF stage did not publish, `save_biology(..., pending=PendingStage(plan, proposal,
accepted))` stores the plan demand + nonce, the availability read and the ACTUAL feedback; `rebuild_pending` + `stage_day` retry the biology
WITHOUT a second withdrawal. It is refused unless the saved external reference IS the post-debit state.

Source identity. The sha256 of every LEAF module that defines the biology is saved; a difference is refused unless `SourceOverride` names exactly
the differing modules with a justification. An override never relaxes the format, the water identity, catalog, geometry or population checks.
'''
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import io
import json
import os
import shutil
import uuid
import zipfile
from fractions import Fraction

import numpy as np

from leaf import biocalendar as _bio
from leaf import lifecycle_bridge as _lb
from leaf import plant_session as _ps
from leaf import state as _state

FORMAT = 'leaf.plant_biology_checkpoint'
VERSION = 2
MANIFEST = 'manifest.json'
ARRAYS = 'biology_arrays.npz'
PLANTS = 'plants.csv'
PLANTS_META = 'plants.metadata.json'
SOURCE_MODULES = ('growth', 'uptake', 'roots', 'canopy', 'plant_pet', 'biocalendar', 'plant_session', 'plant_checkpoint', 'lifecycle_bridge',
                  'demography', 'disturbance', 'recruitment', 'state', 'traits', 'forcing')
MAX_MANIFEST_BYTES = 1 << 20
CSV_BYTES_PER_PLANT = 512          # conservative upper bound of one plain CSV row (nine fields)
META_BYTES = 1 << 16
CHUNK = 1 << 20


class PlantCheckpointError(ValueError):
    '''Any save / restore refusal (nothing is created or returned).'''


def _canon(obj):
    return json.dumps(obj, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_hashes():
    '''sha256 of the LEAF module files that define the biology (cold; read in bounded chunks).'''
    here = os.path.dirname(os.path.abspath(__file__))
    return {m: _sha_file(os.path.join(here, m + '.py')) for m in SOURCE_MODULES}


# ------------------------------------------------------------------ typed values
@dataclasses.dataclass(frozen=True)
class CheckpointLimits:
    '''Explicit caps: `max_file_bytes` bounds every file / npz member; `current_resident_bytes` is the live parent held while loading.'''
    max_file_bytes: int
    current_resident_bytes: int = 0

    def __post_init__(self):
        for n in ('max_file_bytes', 'current_resident_bytes'):
            v = getattr(self, n)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
                raise PlantCheckpointError(f'{n} must be a non-negative integer')


@dataclasses.dataclass(frozen=True)
class SourceOverride:
    '''Explicit, typed permission for the NAMED modules to differ from the saved source hashes (never relaxes format or water identity).'''
    modules: tuple
    justification: str

    def __post_init__(self):
        if not self.modules or not all(m in SOURCE_MODULES for m in self.modules) or not isinstance(self.justification, str) \
                or len(self.justification.strip()) < 10:
            raise PlantCheckpointError('SourceOverride needs known module names and a justification of at least 10 characters')


@dataclasses.dataclass(frozen=True)
class ExternalBindings:
    '''Explicit external binding at restore: `external` = identity of the RESTORED live SYRUP state (must equal the saved external reference),
    optional expected last stage id (dict), expected plant `Domain` and a typed `SourceOverride`.'''
    external: object
    expected_stage_id: object = None
    domain: object = None
    source_override: object = None


@dataclasses.dataclass(frozen=True)
class PendingStage:
    '''Live objects of a withdrawal that SYRUP accepted but LEAF has not yet staged / published (saved by `save_biology`).'''
    plan: object
    proposal: object
    accepted: object


@dataclasses.dataclass(frozen=True)
class RestoredPending:
    meta: dict
    D: object
    supply_m3: object
    actual_m3: object
    unmet_m3: object


@dataclasses.dataclass(frozen=True)
class Restored:
    session: object
    pending: object
    manifest: dict


def _schedule_from(d):
    try:
        cad = Fraction(d['cadence_seconds_exact'])
        return _bio.Schedule(_dt.datetime.fromisoformat(d['anchor']), _dt.timedelta(microseconds=int(cad * 10 ** 6)),
                             _dt.datetime.fromisoformat(d['cursor']), _dt.datetime.fromisoformat(d['next_boundary']),
                             bool(d['season_due']), Fraction(d['pending_days']))
    except (KeyError, ValueError, TypeError, _bio.BioCalendarError) as exc:
        raise PlantCheckpointError(f'invalid saved schedule: {exc}') from exc


def _host(a):
    return np.array(a.get() if hasattr(a, 'get') else a, copy=True)


def _admit(session_or_cfg, resident, need, what):
    cfg = session_or_cfg
    tot = int(cfg.reserve_bytes) + int(resident) + int(need)
    if tot > int(cfg.max_bytes):
        raise PlantCheckpointError(f'{what}: estimated {tot} bytes (reserve {cfg.reserve_bytes} + resident {resident} + {need}) exceed '
                                   f'max_bytes={cfg.max_bytes}; refused before any copy or allocation')


# ------------------------------------------------------------------ save
def save_biology(directory, session, *, external, limits, pending=None, provenance=None, syrup_source=None):
    '''Write the biological checkpoint of an ACCEPTED parent into the NEW `directory`. `external` is the `ResourceIdentity` of the exact accepted
    SYRUP reference this biology is paired with (kept separate from the last consumed T cursor). Every copy is admitted against the global
    ledger from metadata before it exists. Returns the manifest dict.'''
    if not isinstance(session, _ps.PlantSession):
        raise PlantCheckpointError('session must be an accepted leaf.plant_session.PlantSession')
    if not isinstance(external, _lb.ResourceIdentity) or not isinstance(limits, CheckpointLimits):
        raise PlantCheckpointError('external must be a ResourceIdentity and limits a CheckpointLimits')
    if session._succ:
        raise PlantCheckpointError('this session already has a published successor; save the latest accepted session')
    cur = session.resource_cursor
    if cur is not None:
        if external.lop_sha256 != cur.lop_sha256 or external.hydro_count < cur.hydro_count or external.revision < cur.revision:
            raise PlantCheckpointError('the external SYRUP reference is older than / foreign to the last consumed T resource; refused')
        if pending is None and (external.request_sequence != cur.request_sequence or external.n_requests != cur.n_requests):
            raise PlantCheckpointError(
                f'unhandled T history: the external reference has {external.request_sequence} / {external.n_requests} T requests but LEAF consumed '
                f'{cur.request_sequence} / {cur.n_requests}; a T debit without feedback cannot be paired (supply the PendingStage of that debit)')
    directory = os.fspath(directory)
    if os.path.lexists(directory):
        raise PlantCheckpointError(f'{directory!r} already exists; a checkpoint is never overwritten')
    parent = os.path.dirname(os.path.abspath(directory))
    if not os.path.isdir(parent):
        raise PlantCheckpointError(f'the parent directory {parent!r} does not exist')
    if pending is not None:
        if not isinstance(pending, PendingStage) or not isinstance(pending.accepted, _lb.AcceptedWithdrawal):
            raise PlantCheckpointError('pending must be a PendingStage of live plan / proposal / accepted objects')
        pl, pr, ac = pending.plan, pending.proposal, pending.accepted
        if pr.parent_token != session.token or pl.parent_token != session.token or pr.plan is not pl or pl._used:
            raise PlantCheckpointError('the pending stage was not made from this accepted session / plan (stale, foreign or consumed)')
        if ac.identity != pr.identity or ac.before != pr.read.identity:
            raise PlantCheckpointError('the pending accepted withdrawal does not match its proposal (debit identity / before identity)')
        a, b = ac.after, ac.before
        if (a.request_sequence != b.request_sequence + 1 or a.n_requests != b.n_requests + 1 or a.lop_sha256 != b.lop_sha256
                or a.hydro_count != b.hydro_count or a.t_s != b.t_s):
            raise PlantCheckpointError('the pending accepted `after` identity is not the single zero-time successor of `before`')
        if cur is None or b.request_sequence != cur.request_sequence or b.n_requests != cur.n_requests:
            raise PlantCheckpointError('the pending debit was not made from the LEAF-consumed T history (an extra unhandled T exists); refused')
        if external != ac.after:
            raise PlantCheckpointError('a pending stage needs the external reference to BE the post-debit SYRUP state')
    P, Pa, S = session.n_live, int(session.base.n), int(session.config.table.n_species)
    n_arr = 4 + (5 if session.pop.stress is not None else 0)
    rmap = session.pop.geometry.root_map
    LN = 8 * int(rmap.n_layers) * int(rmap.n_cells)
    est_npz = 8 * (2 * P + 2 * S + (4 * P if session.pop.stress is not None else 0)) + 4 * P + (8 * P + 3 * LN if pending is not None else 0) \
        + 512 * n_arr + 3 * 1024
    est_csv = CSV_BYTES_PER_PLANT * Pa
    for name, v in ((PLANTS, est_csv), (ARRAYS, est_npz)):
        if v > limits.max_file_bytes:
            raise PlantCheckpointError(f'{name} would need up to {v} bytes > max_file_bytes={limits.max_file_bytes}; refused before any copy')
    # host copies live together: plant state (materialize peak) + csv text + arrays + hashing chunk
    _admit(session.config, session.resident_bytes() + limits.current_resident_bytes,
           (_ps.MATERIALIZE_BYTES_PER_PLANT * Pa + 2 * est_csv + 2 * est_npz + 4 * CHUNK + 16 * P), 'save_biology')
    state = session.snapshot_state()
    pend = session.pending()
    arrays = {'D': pend['D'], 'U': pend['U'], 'A': pend['A'], 'T': pend['T']}
    if session.pop.stress is not None:
        arrays['stress'] = _host(session.pop.stress)
        for k, a in (('ann_req', session.pop.ann_req), ('ann_act', session.pop.ann_act), ('ann_plan', session.pop.ann_plan),
                     ('ann_gain', session.pop.ann_gain)):
            arrays[k] = _host(a)
    pmeta = None
    if pending is not None:
        pl, pr, ac = pending.plan, pending.proposal, pending.accepted
        for k, v in (('resume_D', pl.D), ('resume_supply_m3', pr.read.supply_m3), ('resume_actual_m3', ac.actual_m3),
                     ('resume_unmet_m3', ac.unmet_m3)):
            arrays[k] = np.array(v.get() if hasattr(v, 'get') else v, dtype=np.float64, copy=True)
        pmeta = {'day': {'start': pl.day.start.isoformat(), 'duration_us': int(pl.day.duration / _dt.timedelta(microseconds=1)),
                         'temperature_c': pl.day.temperature_c, 'transpiration_enabled': pl.day.transpiration_enabled,
                         'growth_enabled': pl.day.growth_enabled},
                 'packet': dict(pl.packet), 'pet_kind': pl.pet_kind, 'plan_digest': pl.plan_digest, 'nonce': pl.nonce,
                 'identity': pr.identity, 'read_identity': pr.read.identity.to_dict(), 'read_cell_area_m2': pr.read.cell_area_m2,
                 'read_n_layers': pr.read.n_layers, 'read_n_cells': pr.read.n_cells, 'read_backend': pr.read.backend,
                 'accepted_after': ac.after.to_dict(), 'requested_m3': ac.requested_m3, 'actual_total_m3': ac.actual_total_m3,
                 'unmet_total_m3': ac.unmet_total_m3, 'cell_area_m2': ac.cell_area_m2}
    tmp = f'{directory}.tmp-{uuid.uuid4().hex[:12]}'
    os.mkdir(tmp)
    try:
        csv_path, meta_path = _state.write_plant_files(state, os.path.join(tmp, PLANTS), initial=False,
                                                       provenance={'writer': 'leaf.plant_checkpoint', 'version': VERSION,
                                                                   **(dict(provenance) if provenance else {})})
        npz_path = os.path.join(tmp, ARRAYS)
        with open(npz_path, 'wb') as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        sizes = {PLANTS: os.path.getsize(csv_path), PLANTS_META: os.path.getsize(meta_path), ARRAYS: os.path.getsize(npz_path)}
        for name, sz in sizes.items():
            if sz > limits.max_file_bytes:
                raise PlantCheckpointError(f'{name} is {sz} bytes > max_file_bytes={limits.max_file_bytes}; nothing published')
        cfg = session.config
        man = {'format': FORMAT, 'version': VERSION, 'sources': source_hashes(), 'syrup_source_sha256': syrup_source,
               'session_id': session.session_id, 'lineage': session.lineage, 'sequence': session.sequence, 'token': session.token,
               'schedule': session.schedule.to_dict(), 'base_offset_days': str(_bio.elapsed_days(session.schedule.anchor, session.schedule.cursor)),
               'counters': session.counters(), 'bio': dict(session.bio), 'cadence_days': int(cfg.cadence_days),
               'day_binding': None if session.day_binding is None else session.day_binding.to_dict(), 'n_live': P, 'n_plants': int(state.n),
               'n_species': S, 'catalog_hash': cfg.catalog.hash, 'config': cfg.identity(), 'config_sha256': cfg.identity_sha256(),
               'geometry_generation': session.pop.generation,
               'geometry_sha256': session.geometry_sha256, 'pop_sha256': session.pop_sha256, 'stress_active': session.pop.stress is not None,
               'rng_seed_set': session.laws.seed is not None, 'laws_active': {'demography': session.laws.demo_on, 'reproduction': session.laws.repro_on},
               'resource_cursor': None if cur is None else cur.to_dict(), 'external_reference': external.to_dict(),
               'last_stage_id': None if session.last_stage_id is None else session.last_stage_id.to_dict(), 'pending': pmeta,
               'water_saved': False,
               'files': {n: {'sha256': _sha_file(os.path.join(tmp, n)), 'bytes': sz} for n, sz in sizes.items()}}
        man['manifest_sha256'] = hashlib.sha256(_canon(man)).hexdigest()
        with open(os.path.join(tmp, MANIFEST), 'wb') as fh:
            fh.write(json.dumps(man, sort_keys=True, indent=1, allow_nan=False).encode())
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp, directory)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)       # only the private temporary directory created above
        raise
    return man


# ------------------------------------------------------------------ load
def _stat(directory, name, cap):
    p = os.path.join(directory, name)
    try:
        sz = os.path.getsize(p)
    except OSError as exc:
        raise PlantCheckpointError(f'missing / unreadable checkpoint file {name!r}: {exc}') from exc
    if sz > cap:
        raise PlantCheckpointError(f'{name} is {sz} bytes > the cap {cap}; refused before reading')
    return p, sz


def read_manifest(directory):
    '''Parsed and self-verified manifest (size-capped; no state is built).'''
    p, _ = _stat(os.fspath(directory), MANIFEST, MAX_MANIFEST_BYTES)
    try:
        with open(p, 'rb') as fh:
            man = json.loads(fh.read(MAX_MANIFEST_BYTES + 1).decode())
    except ValueError as exc:
        raise PlantCheckpointError(f'manifest is not valid JSON: {exc}') from exc
    if not isinstance(man, dict) or man.get('format') != FORMAT:
        raise PlantCheckpointError(f'not a {FORMAT} manifest')
    if man.get('version') != VERSION:
        raise PlantCheckpointError(f'unsupported biological checkpoint version {man.get("version")!r} (this codec reads {VERSION}; the standalone '
                                   'leaf.checkpoint layouts are a different codec)')
    body = {k: v for k, v in man.items() if k != 'manifest_sha256'}
    if man.get('manifest_sha256') != hashlib.sha256(_canon(body)).hexdigest():
        raise PlantCheckpointError('manifest self-digest mismatch (corrupted or edited)')
    return man


def _npz_meta(path, cap):
    '''Array name -> (shape, dtype) from the npz member HEADERS only (no array is read); every member size is capped first.'''
    out = {}
    try:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.file_size > cap or info.compress_size > cap:
                    raise PlantCheckpointError(f'npz member {info.filename!r} exceeds the cap {cap}')
                with z.open(info) as fh:
                    ver = np.lib.format.read_magic(fh)
                    shape, _fo, dtype = (np.lib.format.read_array_header_1_0(fh) if ver == (1, 0) else np.lib.format.read_array_header_2_0(fh))
                    if dtype.hasobject:
                        raise PlantCheckpointError('object arrays are refused (no pickle)')
                    out[info.filename[:-4] if info.filename.endswith('.npy') else info.filename] = (tuple(shape), dtype)
    except (zipfile.BadZipFile, ValueError, OSError) as exc:
        raise PlantCheckpointError(f'biology arrays unreadable: {exc}') from exc
    return out


def load_biology(directory, config, *, bindings, limits, day_binding_check=True):
    '''Verify and restore. `config` is the caller's explicit `SessionConfig` (catalog, backend, bounds, seed, patch, overrides are the CALLER's
    choices and must match); `bindings` the explicit external pairing; `limits` the byte caps. Returns `Restored`.'''
    if not isinstance(config, _ps.SessionConfig) or not isinstance(limits, CheckpointLimits):
        raise PlantCheckpointError('config must be a SessionConfig and limits a CheckpointLimits')
    if not isinstance(bindings, ExternalBindings) or not isinstance(bindings.external, _lb.ResourceIdentity):
        raise PlantCheckpointError('bindings must be ExternalBindings holding the live ResourceIdentity of the restored SYRUP state')
    directory = os.fspath(directory)
    man = read_manifest(directory)
    total = 0
    for name, rec in man['files'].items():
        p, sz = _stat(directory, name, limits.max_file_bytes)
        if sz != rec['bytes']:
            raise PlantCheckpointError(f'file {name!r} has {sz} bytes but the manifest declares {rec["bytes"]}')
        total += sz
    Pa, P, S = int(man['n_plants']), int(man['n_live']), int(man['n_species'])
    _admit(config, limits.current_resident_bytes,
           (_ps.ROW_BYTES_PER_PLANT + _ps.MATERIALIZE_BYTES_PER_PLANT + _ps.GEOMETRY_BYTES_PER_PLANT) * Pa + 80 * P + 3 * total + 2 * CHUNK,
           'load_biology')
    for name, rec in man['files'].items():
        if _sha_file(os.path.join(directory, name)) != rec['sha256']:
            raise PlantCheckpointError(f'file {name!r} differs from the manifest (corrupted, truncated or replaced)')
    if man['catalog_hash'] != config.catalog.hash:
        raise PlantCheckpointError('wrong catalog: the saved catalog hash differs from the session catalog')
    if man['cadence_days'] != int(config.cadence_days) or man['n_species'] != int(config.table.n_species):
        raise PlantCheckpointError('growth cadence / species table differ from the saved session (explicit configuration mismatch)')
    if man['config_sha256'] != config.identity_sha256():
        mine, saved = config.identity(), man['config']
        diff = sorted(k for k in set(mine) | set(saved) if mine.get(k) != saved.get(k))
        raise PlantCheckpointError(f'the caller configuration differs from the saved biology configuration in {diff} (RNG seed, ecological patch '
                                   'size, root overrides, canopy mask, cadence, catalog); a restart never changes them silently and a '
                                   'SourceOverride does not apply here')
    now, saved = source_hashes(), man['sources']
    drift = sorted(k for k in set(now) | set(saved) if now.get(k) != saved.get(k))
    if drift:
        ov = bindings.source_override
        if not isinstance(ov, SourceOverride) or not set(drift) <= set(ov.modules):
            raise PlantCheckpointError(f'LEAF source differs from the saved source in {drift}; refused (a typed SourceOverride naming '
                                       'exactly these modules with a justification is required; it never relaxes the water identity)')
    if bindings.expected_stage_id is not None and bindings.expected_stage_id != man['last_stage_id']:
        raise PlantCheckpointError('the saved last stage differs from the expected stage identity (stale or duplicate replay)')
    saved_ext = _lb.ResourceIdentity.from_dict(man['external_reference'])
    if bindings.external != saved_ext:
        raise PlantCheckpointError('the restored SYRUP state is not the EXACT external reference saved with this biology (combined pairing '
                                   'refused: a descendant or unrelated state would lose T / growth accounting)')
    saved_cur = None if man['resource_cursor'] is None else _lb.ResourceIdentity.from_dict(man['resource_cursor'])
    pend_meta = man['pending']
    if saved_cur is not None and (saved_ext.lop_sha256 != saved_cur.lop_sha256 or saved_ext.hydro_count < saved_cur.hydro_count):
        raise PlantCheckpointError('saved external reference is inconsistent with the saved T cursor (corrupted pairing)')
    if saved_cur is not None and pend_meta is None and (saved_ext.request_sequence != saved_cur.request_sequence
                                                        or saved_ext.n_requests != saved_cur.n_requests):
        raise PlantCheckpointError('the saved external reference carries T requests that LEAF never consumed (unhandled T history); refused')
    if pend_meta is not None:
        after = _lb.ResourceIdentity.from_dict(pend_meta['accepted_after'])
        before = _lb.ResourceIdentity.from_dict(pend_meta['read_identity'])
        if saved_ext != after or saved_cur is None or (before.request_sequence, before.n_requests) != (
                saved_cur.request_sequence, saved_cur.n_requests) or (after.request_sequence, after.n_requests) != (
                before.request_sequence + 1, before.n_requests + 1) or pend_meta['identity'] != pend_meta.get('identity'):
            raise PlantCheckpointError('the saved external reference is not the post-debit state of the pending stage built from the consumed '
                                       'T history; refused')
    meta = _npz_meta(os.path.join(directory, ARRAYS), limits.max_file_bytes)
    want = {'D': ((P,), np.float64), 'U': ((P,), np.float64), 'A': ((S,), np.float64), 'T': ((S,), np.float64)}
    if man['stress_active']:
        want.update({'stress': ((P,), np.int32), **{k: ((P,), np.float64) for k in ('ann_req', 'ann_act', 'ann_plan', 'ann_gain')}})
    if pend_meta is not None:
        L, N = int(pend_meta['read_n_layers']), int(pend_meta['read_n_cells'])
        want.update({'resume_D': ((P,), np.float64), 'resume_supply_m3': ((L, N), np.float64), 'resume_actual_m3': ((L, N), np.float64),
                     'resume_unmet_m3': ((L, N), np.float64)})
    if set(meta) != set(want):
        raise PlantCheckpointError(f'biology arrays {sorted(meta)} differ from the expected {sorted(want)}')
    for k, (shp, dt) in want.items():
        if meta[k][0] != shp or meta[k][1] != np.dtype(dt):
            raise PlantCheckpointError(f'array {k!r} has {meta[k]} but {(shp, np.dtype(dt))} is required; refused before reading')
    with np.load(os.path.join(directory, ARRAYS), allow_pickle=False) as z:
        arr = {k: np.array(z[k]) for k in want}
    for k in ('D', 'U', 'A', 'T'):
        if not np.all(np.isfinite(arr[k])) or (arr[k].size and float(arr[k].min()) < 0.0):
            raise PlantCheckpointError(f'pending array {k!r} is non-finite or negative')
    state = _state.read_plant_csv(os.path.join(directory, PLANTS), config.catalog, domain=bindings.domain)
    if int(state.n) != Pa:
        raise PlantCheckpointError('plant row count differs from the manifest')
    sched = _schedule_from(man['schedule'])
    last = None if man['last_stage_id'] is None else _ps.StageId.from_dict(man['last_stage_id'])
    binding = None if man['day_binding'] is None else _ps.DayBinding(**man['day_binding'])
    stress = arr.get('stress')
    ann = None if stress is None else tuple(arr[k] for k in ('ann_req', 'ann_act', 'ann_plan', 'ann_gain'))
    try:
        sess = _ps.PlantSession.restore_accepted(
            config, state, session_id=man['session_id'], lineage=man['lineage'], sequence=man['sequence'], schedule=sched,
            base_offset_days=Fraction(man['base_offset_days']), D=arr['D'], U=arr['U'], A=arr['A'], T=arr['T'], counters=man['counters'],
            bio=man['bio'], resource_cursor=saved_cur, last_stage_id=last, day_binding=binding,
            geometry_generation=man['geometry_generation'], stress=stress, ann=ann, expected_geometry_sha256=man['geometry_sha256'],
            expected_pop_sha256=man['pop_sha256'])
    except _ps.PlantSessionError as exc:
        raise PlantCheckpointError(f'restore refused: {exc}') from exc
    if sess.token != man['token']:
        raise PlantCheckpointError('the restored session token differs from the saved one (lineage mismatch); refused')
    pending = None
    if pend_meta is not None:
        pending = RestoredPending(pend_meta, arr['resume_D'], arr['resume_supply_m3'], arr['resume_actual_m3'], arr['resume_unmet_m3'])
    return Restored(sess, pending, man)


def rebuild_pending(session, pending):
    '''Reproduce `(plan, proposal, accepted)` of a saved pending stage on the restored parent WITHOUT any SYRUP call: the saved demand and
    availability give the identical root-weighted proposal; the saved ACTUAL feedback is the accepted withdrawal. Then `session.stage_day` +
    `publish` as usual (the debit is NOT repeated). The saved read has no live capability and can never validate a NEW withdrawal.'''
    if not isinstance(pending, RestoredPending):
        raise PlantCheckpointError('pending must be the RestoredPending of load_biology')
    m = pending.meta
    xp = _ps._xp(session.config)
    d = m['day']
    day = _ps.DayContext(_dt.datetime.fromisoformat(d['start']), d['temperature_c'], d['transpiration_enabled'], d['growth_enabled'],
                         duration=_dt.timedelta(microseconds=int(d['duration_us'])))
    plan = session.rebuild_plan(day, D_host=np.ascontiguousarray(pending.D), packet=m['packet'], pet_kind=m['pet_kind'],
                                plan_digest=m['plan_digest'], nonce=m['nonce'])
    dev = (lambda a: xp.asarray(np.ascontiguousarray(a))) if session.config.backend == 'cuda' else (lambda a: np.array(a, copy=True))
    rid = _lb.ResourceIdentity.from_dict(m['read_identity'])
    read = _lb.AvailabilityRead(rid, dev(pending.supply_m3), float(m['read_cell_area_m2']), int(m['read_n_layers']), int(m['read_n_cells']),
                                m['read_backend'])
    prop = session.propose_withdrawal(plan, read)
    if prop.identity != m['identity']:
        raise PlantCheckpointError('the rebuilt debit identity differs from the saved one (lineage mismatch); refused')
    acc = _lb.AcceptedWithdrawal(m['identity'], rid, _lb.ResourceIdentity.from_dict(m['accepted_after']), dev(pending.actual_m3),
                                 dev(pending.unmet_m3), float(m['requested_m3']), float(m['actual_total_m3']), float(m['unmet_total_m3']),
                                 float(m['cell_area_m2']))
    return plan, prop, acc


__all__ = ['FORMAT', 'VERSION', 'SOURCE_MODULES', 'PlantCheckpointError', 'CheckpointLimits', 'SourceOverride', 'ExternalBindings',
           'PendingStage', 'RestoredPending', 'Restored', 'source_hashes', 'save_biology', 'read_manifest', 'load_biology',
           'rebuild_pending']
