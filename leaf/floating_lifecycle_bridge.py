"""Source-bound CPU bridge from genuine floating daily water to PlantSession.

This module reads/validates; MASY visibly calls request_floating_withdrawal and
close_floating_day. No physical water inventory, hydrology or debit is owned here.
The connection's cumulative labels are BRIDGE bookkeeping, not native counters:
FloatingDailyState.request_sequence resets each day. The nominal DayBinding drives
biology; native seconds are recorded without conversion to biological datetimes.

One complete nominal day and at most one T request per day are supported. A failed
native/LEAF operation never becomes a completed bridge day. The caller retains its
actual candidate and feedback; no rollback or combined physical restart is claimed.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import sys
import types
import uuid
import weakref

import numpy as np

from . import lifecycle_bridge as lb
from . import plant_session as ps

VERSION = '1'
PROFILE = 'leaf.floating_lifecycle_bridge.cpu.v1'
CHUNK_VALUES = 4096
VALIDATION_BYTES_PER_VALUE = 64
COLD_SOURCE_ALLOWANCE_BYTES = 4 << 20
_HEX = frozenset('0123456789abcdef')
_SOURCES = weakref.WeakKeyDictionary()
_CONNECTIONS = weakref.WeakKeyDictionary()
_LINEAGES = weakref.WeakKeyDictionary()
_READS = {}
_MODULE_NAMES = ('native_case_profile', 'floating_daily', 'floating_transpiration',
                 'floating_event_lifecycle', 'floating_event_daily_bridge',
                 'floating_soil_layers', 'sediment_bed', 'conservation', 'provenance',
                 'recharge', 'subsurface_transpiration')
_FUNCTION_NAMES = {
    'native_case_profile': ('verify_source_profile',),
    'floating_daily': ('expect_parent', '_need_state', '_check_parent',
                       'request_floating_withdrawal', 'verify_floating_withdrawal',
                       '_mint_state', 'close_floating_day'),
    'floating_transpiration': ('estimate_floating_withdrawal_memory', 'floating_layer_availability_m',
                               'floating_withdraw', 'publish_budget', '_readers', '_packet',
                               '_check_call', '_admit_call', '_quota', '_run', 'refuse_flags'),
    'floating_event_lifecycle': ('_expect', '_record', 'rebase_boundary_from_day'),
    'floating_event_daily_bridge': ('open_day', 'hand_back', '_need_bridge_state'),
    'floating_soil_layers': ('_check_state', '_check_vec'),
    'sediment_bed': ('is_windowed_bed',),
    'conservation': ('volume_roundoff_bound_m3',),
    'provenance': ('source_tree_digest',),
    'recharge': ('verify_plan', 'plan_daily_rain', '_same', '_real', '_hex64', '_count', '_digest', '_amount_ok'),
    'subsurface_transpiration': ('_fused_numpy',),
}
_CLASS_NAMES = {
    'native_case_profile': ('NativeSourceProfile',),
    'floating_daily': ('FloatingDailyState', 'ExpectedParent', 'WithdrawalOutcome', 'DailyLedger'),
    'floating_event_lifecycle': ('FloatingEventLineage', 'Boundary', 'DailyHold'),
    'floating_soil_layers': ('FloatingSoilColumn', 'FloatingSoilState'),
    'floating_transpiration': ('FloatingWithdrawalMemory',),
    'recharge': ('DailyRainPlan', 'DailyCursor', 'PendingStorm'),
    'subsurface_transpiration': ('TranspirationBudget',),
}
# Closed proof dependency roster, checked by identity (registries legitimately change
# their contents as Native mints records). No historical parent/array census is held.
_OBJECT_NAMES = {
    'native_case_profile': ('_SEALED',),
    'floating_daily': ('_MINTED', '_CAPS', '_PRODUCER', '_WITHDRAWALS'),
    'floating_event_lifecycle': ('_ROOTS', '_RECORDS', '_HOLD_BINDING'),
    'floating_transpiration': ('ROW_TEMP_DOUBLES', 'NUMPY_LN_TEMPS', 'NUMBA_STAGE_ROWS', '_F8'),
    'recharge': ('BRANCHES', 'SELECTIONS', 'DAY_S', 'CLOCK_CONTRACT'),
    'subsurface_transpiration': ('_PACKET', '_N_ROWS', '_F8', '_ROW_W_BEFORE', '_ROW_W_AFTER',
                                '_ROW_ACTUAL', '_ROW_UNMET', '_ROW_REQ', '_ROW_MIN_AFTER',
                                '_ROW_FLAGS', '_ROW_QMAX', '_FLAG_WATER', '_FLAG_REQUEST', '_FLAG_INACTIVE'),
}
_METHOD_NAMES = {
    'native_case_profile': {'NativeSourceProfile': ('recheck', '_payload'), '_Sealed': ('_payload',)},
    'recharge': {'DailyCursor': ('check_for_consume', '_check_occurrence')},
}
_MODULE_ALIASES = (('floating_daily', 'rc', 'recharge'),
                   ('floating_transpiration', 'st', 'subsurface_transpiration'),
                   ('floating_transpiration', 'fsl', 'floating_soil_layers'))


class FloatingBridgeError(lb.BridgeError):
    """Typed refusal; the bridge never mutates native water or accepted biology."""


class FloatingSourceError(FloatingBridgeError):
    """Missing, mixed, changed or unauthenticated frozen runtime Source."""


class FloatingCursorError(FloatingBridgeError):
    """Stale/foreign native branch, day, LEAF cursor or unhandled withdrawal."""


def _hex(value, name, size=64):
    if type(value) is not str or len(value) != size or not set(value) <= _HEX:
        raise FloatingSourceError(f'{name} must be a lowercase {size}-hex string')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _finite(value, name, *, nonnegative=True):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise FloatingBridgeError(f'{name} must be a finite real')
    out = float(value)
    if not math.isfinite(out) or (nonnegative and out < 0):
        raise FloatingBridgeError(f'{name} must be finite and nonnegative')
    return out


class _Sealed:
    __slots__ = ('__weakref__',)

    def __init__(self, *args, **kwargs):
        raise FloatingBridgeError(f'{type(self).__name__} is created only by its bridge factory')

    def __setattr__(self, name, value):
        raise FloatingBridgeError('bridge records are immutable')

    __delattr__ = __setattr__

    def __copy__(self):
        raise FloatingBridgeError('bridge records cannot be copied or restored')

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce_ex__(self, protocol):
        raise FloatingBridgeError('no floating bridge cursor checkpoint/restore is qualified')


class FloatingSource(_Sealed):
    """Genuine NativeSourceProfile plus explicit Root-selected committed provenance.

    Commit labels alone grant nothing: the genuine native factory/recheck and actual
    imported package bytes/origins are required. Code/physical qualification remains
    Root-owned; this record reports the Source it authenticated, not a run verdict.
    """
    __slots__ = ()

    def report(self):
        p = _source(self)
        return dict(p['report'])

    def recheck(self, *, max_bytes, reserve_bytes=0):
        """Admitted cold final check; caller prices all other live state in reserve."""
        lb._admit(COLD_SOURCE_ALLOWANCE_BYTES, max_bytes, reserve_bytes, 'floating bridge Source recheck')
        p = _source(self)
        p['profile'].recheck()
        _verify_package(p)


@dataclasses.dataclass(frozen=True)
class BridgeCursor:
    """Bridge-owned cumulative accounting, explicitly separate from native seq."""
    connection_sha256: str
    next_ordinal: int
    requests_consumed: int
    revision: int
    leaf_resource_cursor: lb.ResourceIdentity


class FloatingConnection(_Sealed):
    __slots__ = ()

    @property
    def resource_cursor(self):
        """Initial/last consumed LEAF resource identity; no native counter claim."""
        return _connection(self)['resource']

    @property
    def cursor(self):
        p = _connection(self)
        return BridgeCursor(p['identity'], p['ordinal'], p['requests'], p['revision'], p['resource'])

    @property
    def pending_feedback(self):
        p = _connection(self)
        return None if p['day'] is None else p['day']['accepted']


def _modules():
    return {n: importlib.import_module('maple_syrup.' + n) for n in _MODULE_NAMES}


def _origin(module, name, root):
    if type(module) is not types.ModuleType or type(getattr(module, '__file__', None)) is not str:
        raise FloatingSourceError(f'{name} must be a genuine file-backed module')
    path = os.path.realpath(module.__file__)
    wanted = os.path.join(root, '__init__.py' if name == 'maple_syrup' else
                          name.split('.', 1)[1].replace('.', os.sep) + '.py')
    # Package modules have __init__.py rather than the module .py spelling.
    if name != 'maple_syrup' and hasattr(module, '__path__'):
        wanted = os.path.join(root, *name.split('.')[1:], '__init__.py')
    spec = getattr(module, '__spec__', None)
    if path != os.path.realpath(wanted) or spec is None or spec.name != name or type(spec.origin) is not str or os.path.realpath(spec.origin) != path:
        raise FloatingSourceError(f'{name} has a foreign/mismatched origin')
    return path


def _verify_package(p):
    root = p['root']
    pkg = sys.modules.get('maple_syrup')
    if pkg is not p['package']:
        raise FloatingSourceError('the imported SYRUP package object changed')
    _origin(pkg, 'maple_syrup', root)
    for name, mod in tuple(sys.modules.items()):
        if name == 'maple_syrup' or name.startswith('maple_syrup.'):
            _origin(mod, name, root)
    got = p['modules']['provenance'].source_tree_digest(root).digest_sha256
    if got != p['report']['syrup_package_sha256'] or p['profile'].syrup_package_digest != got:
        raise FloatingSourceError('the actual SYRUP package differs from the frozen profile/expected digest')


def bind_source(profile, *, expected_syrup_digest, expected_syrup_commit,
                expected_maple_commit, max_bytes, reserve_bytes=0):
    """Cold Source binding; no provider, water operation or qualification verdict.

    The caller prices this Source read beside all held state. A genuine new native
    profile is indispensable; arbitrary expected hashes/commit labels are insufficient.
    The SYRUP commit label is Root's frozen provenance, while the verified bytes and
    native SourceProfile are the runtime authority (no self-referential package pin).
    """
    want = _hex(expected_syrup_digest, 'expected_syrup_digest')
    sc = _hex(expected_syrup_commit, 'expected_syrup_commit', 40)
    mc = _hex(expected_maple_commit, 'expected_maple_commit', 40)
    lb._admit(COLD_SOURCE_ALLOWANCE_BYTES, max_bytes, reserve_bytes, 'floating bridge Source binding')
    mods = _modules()
    ncp = mods['native_case_profile']
    if type(profile) is not ncp.NativeSourceProfile:
        raise FloatingSourceError('profile must be a genuinely factory-produced NativeSourceProfile')
    # Properties themselves authenticate the native registry; a forged object refuses.
    if profile.device_runtime_capability or profile.maple_commit != mc or profile.syrup_package_digest != want:
        raise FloatingSourceError('the genuine CPU profile does not name the selected committed Source pair')
    profile.recheck()
    root = os.path.realpath(profile.syrup_package_root)
    package = sys.modules.get('maple_syrup')
    bindings = []
    for short, mod in mods.items():
        name = 'maple_syrup.' + short
        path = _origin(mod, name, root)
        for fn in _FUNCTION_NAMES[short]:
            obj = getattr(mod, fn, None)
            if type(obj) is not types.FunctionType or obj.__module__ != name or obj.__globals__ is not vars(mod):
                raise FloatingSourceError(f'{name}.{fn} is not the actual owning function')
            bindings.append((mod, fn, obj, obj.__code__))
        for name in _CLASS_NAMES.get(short, ()):
            obj = getattr(mod, name, None)
            if type(obj) is not type or obj.__module__ != mod.__name__:
                raise FloatingSourceError(f'{mod.__name__}.{name} is not the actual owning class')
            bindings.append((mod, name, obj, None))
        for cls_name, names in _METHOD_NAMES.get(short, {}).items():
            cls = getattr(mod, cls_name, None)
            if type(cls) is not type or cls.__module__ != mod.__name__:
                raise FloatingSourceError(f'{mod.__name__}.{cls_name} is not the actual owning class')
            bindings.append((mod, cls_name, cls, None))
            for name in names:
                obj = getattr(cls, name, None)
                if type(obj) is not types.FunctionType or obj.__module__ != mod.__name__ or obj.__globals__ is not vars(mod):
                    raise FloatingSourceError(f'{mod.__name__}.{cls_name}.{name} is not the actual owning method')
                bindings.append((cls, name, obj, obj.__code__))
        for name in _OBJECT_NAMES.get(short, ()):
            obj = getattr(mod, name, None)
            if obj is None:
                raise FloatingSourceError(f'the native proof dependency {mod.__name__}.{name} is missing')
            bindings.append((mod, name, obj, None))
        bindings.append((mod, '__file__', mod.__file__, path))
    for short, alias, target in _MODULE_ALIASES:
        if getattr(mods[short], alias, None) is not mods[target]:
            raise FloatingSourceError(f'the native module alias {short}.{alias} is foreign')
        bindings.append((mods[short], alias, mods[target], None))
    p = {'profile': profile, 'root': root, 'package': package, 'modules': mods,
         'bindings': tuple(bindings), 'report': {
             'profile': PROFILE, 'native_profile': profile.profile, 'maple_commit': mc,
             'maple_package_sha256': profile.package_digest, 'bundle_id': profile.bundle_id,
             'syrup_commit': sc, 'syrup_package_sha256': want, 'syrup_package_root': root,
             'qualification': 'Source authentication only; Root code/physical acceptance required'}}
    _verify_package(p)
    source = object.__new__(FloatingSource)
    _SOURCES[source] = p
    return source


def _source(source):
    p = _SOURCES.get(source) if type(source) is FloatingSource else None
    if p is None:
        raise FloatingSourceError('source is not a genuine bound FloatingSource')
    if sys.modules.get('maple_syrup') is not p['package']:
        raise FloatingSourceError('SYRUP package mapping changed')
    for short, mod in p['modules'].items():
        if sys.modules.get('maple_syrup.' + short) is not mod:
            raise FloatingSourceError('a native bridge module mapping changed')
    for mod, name, obj, code_or_path in p['bindings']:
        if getattr(mod, name, None) is not obj:
            raise FloatingSourceError(f'the native binding {mod.__name__}.{name} changed')
        if name == '__file__':
            if os.path.realpath(obj) != code_or_path:
                raise FloatingSourceError('a native module origin changed')
        elif code_or_path is not None and obj.__code__ is not code_or_path:
            raise FloatingSourceError('a native function code changed')
    return p


def _connection(connection):
    p = _CONNECTIONS.get(connection) if type(connection) is FloatingConnection else None
    if p is None:
        raise FloatingCursorError('connection is not a genuine live floating bridge connection')
    _source(p['source'])
    if p['lineage']() is None:
        raise FloatingCursorError('the native lineage is gone')
    return p


def _cpu_column(column, *, shape=None):
    if column.backend != 'numpy':
        raise FloatingBridgeError('this bridge serves Dense NumPy CPU only; no backend fallback')
    L, N = int(column.n_layers), int(column.n_cells)
    area = _finite(column.cell_area_m2, 'cell_area_m2')
    if L < 1 or N < 1 or area <= 0 or (shape is not None and math.prod(shape) != N):
        raise FloatingBridgeError('native geometry/layer/area metadata is inconsistent')
    mask = column._maps['active']
    if type(mask) is not np.ndarray or mask.dtype != np.bool_ or mask.shape != (N,) or not mask.flags.c_contiguous:
        raise FloatingBridgeError('the native active mask must be bool [N] in layer-major ordering')
    return L, N, area


def connect(source, lineage, initial_boundary, *, day_binding):
    """Connect once at a genuine fresh native accepted boundary (no restore).

    Use connection.resource_cursor as PlantSession.start(initial_resource=...). The
    stable connection ID does not change with property/operator refresh after wind.
    """
    sp = _source(source)
    lc = sp['modules']['floating_event_lifecycle']
    lc._expect(lineage, initial_boundary, lc.Boundary)
    if not isinstance(day_binding, ps.DayBinding):
        raise FloatingCursorError('day_binding must be a LEAF DayBinding')
    if not initial_boundary.eligible_for_interevent or initial_boundary.pending_forcing is not None or initial_boundary.storm_match is not None:
        raise FloatingCursorError('connection needs a complete boundary with no pending storm/recession')
    c = initial_boundary.cursor
    if c is not None and (c.run_digest != day_binding.run_digest or c.ordinal != 0 or c.pending is not None):
        raise FloatingCursorError('only a new nominal daily lineage can connect; restore is unsupported')
    if initial_boundary.ledger.get('n_days', 0) != 0 or initial_boundary.ledger.get('transpiration_m3', 0.0) != 0.0:
        raise FloatingCursorError('the initial boundary has unhandled daily/T history')
    prev = _LINEAGES.get(lineage)
    if prev is not None and prev() is not None:
        raise FloatingCursorError('this native lineage already has a bridge connection')
    _cpu_column(initial_boundary.root.column, shape=initial_boundary.surface_depth_m.shape)
    if sp['modules']['sediment_bed'].is_windowed_bed(initial_boundary.bed):
        raise FloatingBridgeError('the new bridge profile serves genuine Dense beds only')
    ident = _digest({'profile': PROFILE, 'run': day_binding.to_dict(),
                     'source': sp['report'], 'anchor': initial_boundary.root.source_witness,
                     'connection': uuid.uuid4().hex})
    resource = lb.ResourceIdentity(ident, _digest(['bridge-origin', ident]), 0, 0, 0, 0,
                                   float(initial_boundary.t_s), 'boundary', None, day_binding.run_digest, False)
    obj = object.__new__(FloatingConnection)
    _CONNECTIONS[obj] = {'source': source, 'lineage': weakref.ref(lineage), 'binding': day_binding,
                         'identity': ident, 'anchor': initial_boundary.root.source_witness,
                         'config': initial_boundary.root.config.sha256, 'ordinal': 0,
                         'requests': 0, 'revision': 0, 'resource': resource,
                         'native_t_total': 0.0, 'plant_lineage': None, 'day': None}
    _LINEAGES[lineage] = weakref.ref(obj)
    return obj


def _leaf_cursor(p, cursor):
    if type(cursor) is not lb.ResourceIdentity or cursor != p['resource']:
        raise FloatingCursorError('the actual accepted LEAF resource cursor differs from this connection')


def _daily(p, state, role):
    m = _source(p['source'])['modules']
    fd, lc = m['floating_daily'], m['floating_event_lifecycle']
    fd._check_parent(state, fd.expect_parent(state), role)
    m['floating_event_daily_bridge']._need_bridge_state(state)
    ctx = state.ctx
    lc._expect(p['lineage'](), ctx.hold, lc.DailyHold)
    if ctx.run_digest != p['binding'].run_digest or state.cursor.run_digest != p['binding'].run_digest or state.cursor.ordinal != p['ordinal']:
        raise FloatingCursorError('foreign daily run, skipped nominal day or stale native cursor')
    if ctx.source_witness != p['anchor'] or ctx.config_sha256 != p['config']:
        raise FloatingCursorError('the native cold anchor/configuration changed')
    sp = _source(p['source'])['report']
    if ctx.source_pair['maple_syrup_tree_sha256'] != sp['syrup_package_sha256'] or ctx.source_pair['maple_package_sha256'] != sp['maple_package_sha256']:
        raise FloatingSourceError('the native daily context carries a different actual Source pair')
    _cpu_column(ctx.column, shape=ctx.shape)
    return m


def begin_day(connection, daily_boundary, *, leaf_resource_cursor):
    """Check the genuine opening, including after hydrology returns interevent.

    A matched full-storm route may replace a not-yet-read day opening. A read or
    accepted T candidate cannot be silently abandoned. The original boundary must
    remain live through this call so Native's own hold->parent proof can be checked.
    """
    p = _connection(connection)
    _leaf_cursor(p, leaf_resource_cursor)
    m = _daily(p, daily_boundary, 'boundary')
    ctx = daily_boundary.ctx
    bound = m['floating_event_lifecycle']._HOLD_BINDING.get(ctx.hold)
    native_parent = None if bound is None else bound[0]()
    if native_parent is None or bound[1]() is not p['lineage']():
        raise FloatingCursorError('the native hold has no live exact parent/lineage proof')
    if native_parent.root.water_state is not daily_boundary.water or native_parent.root.history is not daily_boundary.history or native_parent.root.column is not ctx.column:
        raise FloatingCursorError('the day does not start from the accepted native W/J/I/column')
    if native_parent.ledger.get('transpiration_m3', 0.0) != p['native_t_total']:
        raise FloatingCursorError('native T occurred without consumed LEAF feedback')
    if daily_boundary.request_sequence != 0 or daily_boundary.ledger.days_applied != 0 or daily_boundary.ledger.transpiration_m3 != 0.0:
        raise FloatingCursorError('the daily opening already has hydrology/T history')
    if daily_boundary.occurrence is not None or daily_boundary.origin_digest is not None:
        raise FloatingCursorError('not a genuine untouched daily opening')
    old = p['day']
    if old is not None and (old['read'] is not None or old['accepted'] is not None):
        raise FloatingCursorError('a read/accepted daily candidate is unresolved')
    p['day'] = {'initial_digest': daily_boundary.digest,
                'ctx': ctx, 'read': None, 'accepted': None, 'postdebit_state': None,
                'proposal_parent_token': None, 'proposal_plan_digest': None,
                'native_start_t': float(native_parent.ledger.get('transpiration_m3', 0.0))}


def estimate_bridge_bytes(n_layers, n_cells, *, operation):
    """Existing LEAF conversion estimate plus explicit bounded validation scratch.

    Native availability's own estimate is composed by read_availability. All other
    coexisting arrays/parents/proposals/native request outputs belong in caller reserve.
    """
    e = lb.estimate_conversion_bytes(n_layers, n_cells, operation=operation)
    scratch = VALIDATION_BYTES_PER_VALUE * min(int(n_layers) * int(n_cells), CHUNK_VALUES)
    return dict(e, validation_bytes=scratch, total_bytes=e['total_bytes'] + scratch)


def _array(a, name, shape, *, nonnegative=True):
    if type(a) is not np.ndarray or a.dtype != np.float64 or a.shape != shape or not a.flags.c_contiguous:
        raise FloatingBridgeError(f'{name} must be a C-contiguous float64 {shape} NumPy array')
    flat = a.reshape(-1)
    for i in range(0, flat.size, CHUNK_VALUES):
        row = flat[i:i + CHUNK_VALUES]
        if not np.isfinite(row).all() or (nonnegative and (row < 0).any()):
            raise FloatingBridgeError(f'{name} has nonfinite/negative values')


def _interevent(p, parent):
    m = _daily(p, parent, 'interevent')
    day = p['day']
    if day is None or parent.ctx is not day['ctx'] or parent.origin_digest != day['initial_digest']:
        raise FloatingCursorError('interevent does not descend from the explicit daily opening')
    if parent.ledger.days_applied != 1 or parent.occurrence is None:
        raise FloatingCursorError('one genuinely completed hydrology day is required')
    plan, branch = parent.occurrence
    m['floating_daily'].rc.verify_plan(plan)
    parent.cursor.check_for_consume(plan, branch)
    if parent.cursor.pending is not None and parent.ctx.hold.storm_match != parent.cursor.pending.forcing_digest:
        raise FloatingCursorError('the pending native storm lacks its genuine completion match')
    if plan.ordinal != p['ordinal'] or plan.run_digest != p['binding'].run_digest:
        raise FloatingCursorError('the native occurrence is not this nominal day')
    if parent.t_s != parent.ledger.expected_t_s:
        raise FloatingCursorError('the native physical clock differs from its accepted ledger')
    return m, day


def _read_proof(read):
    item = _READS.get(id(read))
    if type(read) is not lb.AvailabilityRead or item is None or item[0]() is not read:
        raise FloatingCursorError('read was not produced by this floating bridge')
    return item[1]


def read_availability(connection, parent, *, leaf_resource_cursor, max_bytes,
                      reserve_bytes=0, available_bytes_fn):
    """Read native availability once, return PlantSession's genuine LEAF read type.

    The native interevent cursor is the CURRENT ordinal (not ordinal-1 as in N15).
    The projected cumulative fields are explicitly bridge-owned labels.
    """
    p = _connection(connection)
    _leaf_cursor(p, leaf_resource_cursor)
    m, day = _interevent(p, parent)
    if day['read'] is not None or day['accepted'] is not None:
        raise FloatingCursorError('this nominal day already has a read/accepted withdrawal')
    if parent.request_sequence != 0 or parent.ledger.transpiration_m3 != 0.0 or parent.ledger.last_day_transpiration_m3 != 0.0:
        raise FloatingCursorError('unhandled native T before the bridge read')
    ctx = parent.ctx
    L, N, area = _cpu_column(ctx.column, shape=ctx.shape)
    e = estimate_bridge_bytes(L, N, operation='read')
    ne = m['floating_transpiration'].estimate_floating_withdrawal_memory(L, N, backend='numpy', operation='availability')
    lb._admit(max(e['total_bytes'], int(ne.required_bytes)), max_bytes, reserve_bytes, 'floating availability read')
    depth = m['floating_transpiration'].floating_layer_availability_m(ctx.column, parent.water,
                memory_budget_bytes=max_bytes, reserve_bytes=reserve_bytes, available_bytes_fn=available_bytes_fn)
    _array(depth, 'availability', (L, N))
    supply = depth * area
    del depth
    _array(supply, 'availability volume', (L, N))
    supply.flags.writeable = False
    ident = lb.ResourceIdentity(p['identity'], parent.digest, p['revision'] + 1, p['requests'],
                                 p['ordinal'] + 1, p['requests'], parent.t_s, 'interevent',
                                 p['ordinal'], p['binding'].run_digest, False)
    expected = m['floating_daily'].expect_parent(parent)
    read = lb.AvailabilityRead(ident, supply, area, L, N, 'numpy', weakref.ref(parent), expected, connection)
    key = id(read)
    proof = {'connection': weakref.ref(connection), 'parent': weakref.ref(parent), 'request': None}
    _READS[key] = (weakref.ref(read, lambda _r, key=key: _READS.pop(key, None)), proof)
    day['read'] = weakref.ref(read)
    return read


def request_args(read, connection, parent):
    """Native request keywords, bound to the exact read parent and native seq=0."""
    p = _connection(connection)
    proof = _read_proof(read)
    if proof['connection']() is not connection or proof['parent']() is not parent or read._parent_ref() is not parent:
        raise FloatingCursorError('the read names another connection/parent instance')
    _interevent(p, parent)
    if p['day']['accepted'] is not None or parent.request_sequence != 0:
        raise FloatingCursorError('one withdrawal per nominal day; duplicate/stale request refused')
    fd = _source(p['source'])['modules']['floating_daily']
    fd._check_parent(parent, read._expected, 'interevent')
    return {'expected_parent': read._expected, 'expected_sequence': parent.request_sequence}


def request_depth_m(read, proposed_m3, *, max_bytes, reserve_bytes=0):
    """One volume->depth conversion; retain that actual request for feedback proof."""
    proof = _read_proof(read)
    connection = proof['connection']()
    if connection is None:
        raise FloatingCursorError('the connection is gone')
    parent = proof['parent']()
    if parent is None:
        raise FloatingCursorError('the exact native read parent is gone')
    request_args(read, connection, parent)
    shape = (read.n_layers * read.n_cells,)
    if type(proposed_m3) is not np.ndarray or proposed_m3.shape not in (shape, (read.n_layers, read.n_cells)):
        raise FloatingBridgeError('proposal must be the layer-major aggregate volume array')
    lb._admit(estimate_bridge_bytes(read.n_layers, read.n_cells, operation='request')['total_bytes'],
              max_bytes, reserve_bytes, 'floating request conversion')
    _array(proposed_m3, 'proposal', proposed_m3.shape)
    if proof['request'] is not None:
        raise FloatingCursorError('this read already has a converted request')
    request = (proposed_m3 / read.cell_area_m2).reshape(read.n_layers, read.n_cells)
    _array(request, 'request depth', request.shape)
    request.flags.writeable = False
    proof['request'] = request
    return request


def _producer(fd, state, parent, kind):
    got = fd._PRODUCER.get(state)
    if got is None or got[0] != parent.execution_digest or state.parent_digest != parent.digest:
        raise FloatingCursorError('native successor lacks the exact parent producer proof')
    receipt = json.loads(got[1])
    if receipt.get('kind') != kind:
        raise FloatingCursorError('native successor is not the requested transaction')
    return receipt


def _sum_active(a, active):
    total = 0.0
    for row in a:
        for i in range(0, row.size, CHUNK_VALUES):
            total += float(np.sum(row[i:i + CHUNK_VALUES], where=active[i:i + CHUNK_VALUES]))
    return total


def _validate_feedback(m, parent, outcome, request, proposed, area):
    """Native producer/ledger proof plus bounded per-cell and global comparisons.

    The ONLY comparison law is Native volume_roundoff_bound_m3, using the actual
    owning withdrawal term count and genuine before/request reservoir operands.
    """
    fd = m['floating_daily']
    if type(outcome) is not fd.WithdrawalOutcome:
        raise FloatingCursorError('outcome must be the actual native WithdrawalOutcome')
    child = outcome.state
    fd._check_parent(child, fd.expect_parent(child), 'interevent')
    if child.ctx is not parent.ctx or child.cursor is not parent.cursor or child.occurrence is not parent.occurrence or child.history is not parent.history or child.surface is not parent.surface:
        raise FloatingCursorError('withdrawal changed context, cursor, occurrence, J/I or surface/faces')
    if child.origin_digest != parent.origin_digest or child.revision != parent.revision + 1 or child.request_sequence != 1 or parent.request_sequence != 0 or outcome.sequence != 0:
        raise FloatingCursorError('native withdrawal revision/one-request sequence mismatch')
    if child.t_s != parent.t_s or child.water.column_sha256 != parent.water.column_sha256 or child.water.token == parent.water.token:
        raise FloatingCursorError('withdrawal changed clock/column or reused the water token')
    receipt = _producer(fd, child, parent, 'withdrawal')
    for name in ('requested_m3', 'actual_m3', 'unmet_m3'):
        _finite(getattr(outcome, name), name)
    if receipt != {'kind': 'withdrawal', 'seq': 0, 'requested': float(outcome.requested_m3).hex(), 'actual': float(outcome.actual_m3).hex()}:
        raise FloatingCursorError('the outcome scalars differ from the native producer receipt')
    ctx = parent.ctx
    L, N = ctx.n_layers, ctx.n_cells
    shape = (L, N)
    for name, a in (('actual', outcome.actual_m), ('unmet', outcome.unmet_m), ('scale', outcome.scale),
                    ('before W', parent.water.water_m), ('after W', child.water.water_m)):
        _array(a, name, shape)
    col = ctx.column
    active = col._maps['active']
    terms = 3 * L * int(N if col.n_active is None else col.n_active) + m['floating_transpiration'].st._PACKET
    b0, b1 = parent.ledger, child.ledger
    if b1.n_terms != b0.n_terms + terms or b1.day_n_terms != b0.day_n_terms + terms:
        raise FloatingCursorError('the native component term count did not advance exactly once')
    changed = {'transpiration_m3', 'last_day_transpiration_m3', 'bound_m3', 'day_bound_m3', 'n_terms', 'day_n_terms'}
    for f in dataclasses.fields(b0):
        if f.name not in changed and getattr(b0, f.name) != getattr(b1, f.name):
            raise FloatingCursorError(f'withdrawal changed the unrelated native ledger field {f.name}')
    if b1.transpiration_m3 != b0.transpiration_m3 + outcome.actual_m3 or b1.last_day_transpiration_m3 != b0.last_day_transpiration_m3 + outcome.actual_m3:
        raise FloatingCursorError('native T accounting differs from the actual withdrawal')
    if not (math.isfinite(b1.bound_m3) and b1.bound_m3 >= b0.bound_m3 and math.isfinite(b1.day_bound_m3) and b1.day_bound_m3 >= b0.day_bound_m3):
        raise FloatingCursorError('native component bounds moved backwards or became nonfinite')
    before = _sum_active(parent.water.water_m, active) * area
    scale = before + float(outcome.requested_m3)
    comparison = m['conservation'].volume_roundoff_bound_m3(terms, scale)
    if not math.isfinite(comparison):
        raise FloatingCursorError('native comparison bound is not representable')
    for l in range(L):
        for i in range(0, N, CHUNK_VALUES):
            sl = slice(i, i + CHUNK_VALUES)
            a, u, q = outcome.actual_m[l, sl], outcome.unmet_m[l, sl], request[l, sl]
            old, new = parent.water.water_m[l, sl], child.water.water_m[l, sl]
            mask = active[sl]
            # Each resource uses the same Native global-scale bound; there is no tighter per-cell law.
            if np.any(np.abs((a + u - q) * area) > comparison) or np.any(np.abs((new + a - old) * area) > comparison):
                raise FloatingCursorError('actual/unmet or W debit does not close per resource')
            if np.any(np.abs((q * area) - proposed.reshape(shape)[l, sl]) > comparison):
                raise FloatingCursorError('the native request differs from LEAF\'s actual proposal')
            if np.any(outcome.scale[l, sl] > 1.0) or np.any(a > q) or np.any(a[~mask] != 0.0) or np.any(u[~mask] != q[~mask]) or np.any(new[~mask] != old[~mask]):
                raise FloatingCursorError('withdrawal violates scale/request or inactive-cell identity')
            if np.any(np.abs((outcome.scale[l, sl] * q - a) * area) > comparison):
                raise FloatingCursorError('the native scale does not reproduce actual withdrawal')
    act = _sum_active(outcome.actual_m, active) * area
    unm = _sum_active(outcome.unmet_m, active) * area
    req = _sum_active(request, active) * area
    after = _sum_active(child.water.water_m, active) * area
    if any(abs(x - y) > comparison for x, y in ((act, outcome.actual_m3), (unm, outcome.unmet_m3), (req, outcome.requested_m3),
                                                (act + unm, req), (after + act, before))):
        raise FloatingCursorError('native outcome/global W/T/request closure mismatch')


def accepted_withdrawal(connection, proposal, outcome, *, max_bytes, reserve_bytes=0):
    """Validate ONE real native debit; return PlantSession's accepted feedback type.

    Bookkeeping is staged, not consumed until commit_day validates the published
    PlantSession and real native close/handback. pending_feedback remains available
    if later staging/close fails; no second debit is necessary or permitted here.
    """
    p = _connection(connection)
    if type(proposal) is not ps.WithdrawalProposal:
        raise FloatingCursorError('proposal must be the actual PlantSession WithdrawalProposal')
    read = proposal.read
    proof = _read_proof(read)
    parent = proof['parent']()
    if parent is None:
        raise FloatingCursorError('the exact native read parent is gone')
    request_args(read, connection, parent)
    m, day = _interevent(p, parent)
    if proof['connection']() is not connection or proof['request'] is None:
        raise FloatingCursorError('the proposal has no converted native request for this connection')
    m['floating_daily'].verify_floating_withdrawal(parent, expected_parent=read._expected, outcome=outcome)
    if proposal.parent_token != proposal.plan.parent_token or proposal.plan._used:
        raise FloatingCursorError('stale or already-published LEAF proposal')
    lb.debit_identity(proposal.identity)
    lb._admit(estimate_bridge_bytes(read.n_layers, read.n_cells, operation='accept')['total_bytes'],
              max_bytes, reserve_bytes, 'floating accepted feedback')
    proposed = proposal.uptake_plan.proposed
    _array(proposed, 'LEAF aggregate proposal', (read.n_layers * read.n_cells,))
    rm = proposal.uptake_plan.root_map
    if (rm.n_layers, rm.n_cells, rm.backend) != (read.n_layers, read.n_cells, 'numpy'):
        raise FloatingCursorError('the actual proposal root map differs from the native resource shape/backend')
    domain = rm.provenance.get('domain')
    ctx = parent.ctx
    if not isinstance(domain, dict) or (domain.get('ny'), domain.get('nx')) != tuple(ctx.shape):
        raise FloatingCursorError('the proposal root-map domain differs from native shape/cell area')
    dx = _finite(domain.get('dx_m'), 'root-map dx_m')
    dy = _finite(domain.get('dy_m'), 'root-map dy_m')
    if dx <= 0 or dy <= 0 or dx * dy != read.cell_area_m2:
        raise FloatingCursorError('the proposal root-map domain differs from native shape/cell area')
    _validate_feedback(m, parent, outcome, proof['request'], proposed, read.cell_area_m2)
    after = lb.ResourceIdentity(p['identity'], outcome.state.digest, p['revision'] + 2, p['requests'] + 1,
                                 p['ordinal'] + 1, p['requests'] + 1, outcome.state.t_s,
                                 'interevent', p['ordinal'], p['binding'].run_digest, False)
    actual = outcome.actual_m * read.cell_area_m2
    unmet = outcome.unmet_m * read.cell_area_m2
    _array(actual, 'actual volume', actual.shape)
    _array(unmet, 'unmet volume', unmet.shape)
    actual.flags.writeable = unmet.flags.writeable = False
    accepted = lb.AcceptedWithdrawal(proposal.identity, read.identity, after, actual, unmet,
                                      outcome.requested_m3, outcome.actual_m3, outcome.unmet_m3, read.cell_area_m2)
    day['accepted'] = accepted  # two L*N float64 volume arrays, retained until commit
    day['postdebit_state'] = outcome.state  # actual W/J/I; no outcome/read/proposal retention
    day['proposal_parent_token'] = proposal.parent_token
    day['proposal_plan_digest'] = proposal.plan.plan_digest
    proof['request'] = None  # accepted day refuses another request; caller owns any retained depth
    return accepted


def commit_day(connection, interevent, closed_daily, native_boundary, *, session):
    """Consume bridge accounting LAST after native close/handback and LEAF publish.

    Pass the exact final interevent too: no-T days are validated without inventing a
    debit, and the native close's exact parent is checked without a history census.
    Nothing here closes a day, adopts Native water or publishes LEAF. A refusal keeps
    the connection's prior committed cursor and exposes its pending actual feedback.
    """
    p = _connection(connection)
    sp = _source(p['source'])
    m = sp['modules']
    fd, lc = m['floating_daily'], m['floating_event_lifecycle']
    day = p['day']
    if day is None:
        raise FloatingCursorError('there is no open bridge day to commit')
    # The native head is already the handback, so do not require the retired hold as head.
    fd._check_parent(interevent, fd.expect_parent(interevent), 'interevent')
    fd._check_parent(closed_daily, fd.expect_parent(closed_daily), 'boundary')
    lc._expect(p['lineage'](), native_boundary, lc.Boundary)
    if interevent.ctx is not day['ctx'] or interevent.origin_digest != day['initial_digest'] or interevent.ledger.days_applied != 1:
        raise FloatingCursorError('the final interevent does not belong to this bridge day')
    if closed_daily.ctx is not interevent.ctx or closed_daily.water is not interevent.water or closed_daily.history is not interevent.history or closed_daily.surface is not interevent.surface or closed_daily.ledger is not interevent.ledger:
        raise FloatingCursorError('native close replaced W/J/I/surface/ledger')
    receipt = _producer(fd, closed_daily, interevent, 'close')
    plan, branch = interevent.occurrence
    if receipt != {'kind': 'close', 'plan': plan.digest, 'branch': branch} or closed_daily.cursor.ordinal != p['ordinal'] + 1 or closed_daily.cursor.pending is not None or closed_daily.request_sequence != 0:
        raise FloatingCursorError('native close did not consume exactly this nominal occurrence')
    if native_boundary.root.water_state is not closed_daily.water or native_boundary.root.history is not closed_daily.history or native_boundary.cursor is not closed_daily.cursor or native_boundary.t_s != closed_daily.t_s:
        raise FloatingCursorError('native handback does not carry the exact closed day')
    expected_t = day['native_start_t'] + closed_daily.ledger.transpiration_m3
    if native_boundary.ledger.get('transpiration_m3', 0.0) != expected_t:
        raise FloatingCursorError('native handback T total does not match this actual day')
    if type(session) is not ps.PlantSession or session.day_binding != p['binding']:
        raise FloatingCursorError('a genuine published PlantSession of this daily run is required')
    if p['plant_lineage'] is not None and session.lineage != p['plant_lineage']:
        raise FloatingCursorError('the accepted biological lineage changed')
    wanted_date = dt.date.fromisoformat(p['binding'].origin_date) + dt.timedelta(days=p['ordinal'] + 1)
    if session.cursor.date() != wanted_date or (session.cursor.hour, session.cursor.minute, session.cursor.second, session.cursor.microsecond) != (0, 0, 0, 0):
        raise FloatingCursorError('LEAF did not publish the whole nominal day (no physical-time catchup)')
    accepted = day['accepted']
    if accepted is None:
        if day['read'] is not None or interevent.request_sequence != 0 or interevent.ledger.transpiration_m3 != 0.0:
            raise FloatingCursorError('native T/read lacks consumed LEAF feedback')
        _leaf_cursor(p, session.resource_cursor)
    else:
        if day['postdebit_state'] is not interevent or interevent.request_sequence != 1 or session.resource_cursor != accepted.after:
            raise FloatingCursorError('the actual accepted native T and published LEAF feedback differ')
        stage = session.last_stage_id
        if stage is None or stage.resource_after != accepted.after.to_dict() or stage.resource_before != accepted.before.to_dict() or stage.debit_identity != accepted.identity or stage.parent_token != day['proposal_parent_token'] or stage.plan_digest != day['proposal_plan_digest']:
            raise FloatingCursorError('published LEAF stage does not name this actual withdrawal')
    p['resource'] = session.resource_cursor
    p['requests'] += 0 if accepted is None else 1
    p['revision'] += 1 if accepted is None else 2
    p['ordinal'] += 1
    p['native_t_total'] = expected_t
    p['plant_lineage'] = session.lineage
    p['day'] = None  # release all prior native states/requests/feedback; no history list
    return connection.cursor


__all__ = ['VERSION', 'PROFILE', 'FloatingBridgeError', 'FloatingSourceError', 'FloatingCursorError',
           'FloatingSource', 'BridgeCursor', 'FloatingConnection', 'bind_source', 'connect',
           'begin_day', 'estimate_bridge_bytes', 'read_availability', 'request_args',
           'request_depth_m', 'accepted_withdrawal', 'commit_day']
