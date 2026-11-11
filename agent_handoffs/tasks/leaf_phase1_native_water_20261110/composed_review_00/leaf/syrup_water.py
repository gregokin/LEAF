'''leaf.syrup_water: PURE bridge between the LEAF sparse uptake allocator and the native SYRUP soil column.

One invocation of `withdraw_plants(root_map, demand_m3, operator, soil_state, *, max_bytes, ...)` does, in order:

  1. validation (no allocation): root-map / demand / operator / state types, shapes, namespaces, dtypes, device
     affinity, the binding of the PREPARED native operator to its own column and the state's metadata, explicit
     byte budget and the native column's reported available bytes;
  2. availability snapshot A [L, N] (m3) from the native inventory and the operator's PUBLIC residual / mask
     fields (see AVAILABILITY_PROJECTION): a read-only projection of SYRUP-owned state and floor, NOT a water
     store and NOT a configurable LEAF wilting floor; native validity (finite, >= 0, <= saturation + tolerance)
     is checked on every cell from the same pass, without a dummy native withdrawal;
  3. `leaf.uptake.propose(root_map, demand_m3, A)`: Q = sum(D w), proposed = min(A, Q), fixed normalised roots,
     no compensation, no species priority (the allocator's rule, not re-implemented here);
  4. proposed VOLUME -> native DEPTH by dividing by the column cell area EXACTLY ONCE, then ONE call of the actual
     `maple_syrup.subsurface_transpiration.withdraw_transpiration(operator, state, request_depth, ...)`. That
     native API alone creates the fresh soil inventory (time preserved) and reports actual / unmet / scale and a
     physical budget; nothing here subtracts water;
  5. actual accepted depth -> volume by multiplying by the cell area EXACTLY ONCE, then
     `leaf.uptake.accept(plan, accepted_volume)`: plants gather from ACTUAL accepted water (never proposed
     water, never the least-satisfied layer). Honest area-conversion roundoff is passed through unchanged (no
     balancing flux, no clipping away of water, no second native acceptance);
  6. conservation diagnostics: sum(delivered) must equal the native actual volume within the native FP64 bound
     plus the allocator's FP64 closure policy (an implementation failure otherwise), plus a truthful byte ledger.

Units. demand_m3 [P] m3 per timestep (the caller fixes the interval; nothing here scales a rate); root-map
resources are layer * n_cells + cell; native water_m [L, N] is FP64 depth (m); cell_area_m2 is the column's
scalar cell area; availability / proposed / accepted / delivered / unmet are m3.

Ownership and publication boundary. The result holds the FRESH native state / native result and the plant plan /
feedback as values; nothing is published, no scheduler clock advances, no interval receipt is bound. Inputs
(native state, root map, demand) are never written, also when validation, admission, the native withdrawal or
the gather fails. The later dated driver must publish native + biological state atomically and bind an
exactly-once interval receipt; this pure function is NOT a restart / idempotency / event-scheduler mechanism.

Namespaces. CPU NumPy and Numba maps/columns are compatible NumPy namespaces (either way round); CUDA requires
the exact same CuPy module and device for the map, the demand, the column, the operator and the state. No
implicit fallback, no host/device population read-back (only small scalar packets), no device switch, no dense
P x L x N arrays. Importing this module imports NumPy and leaf.uptake only; maple_syrup / numba / cupy are
imported lazily when the bridge is first used (`native_status` reports availability without raising).

Pinned native contract. SYRUP HEAD c76c603dde4e8f79ec48b18d954a386580d8bc89 (feature/subsurface-hydrology),
maple_syrup.subsurface_transpiration VERSION 0.2.0 (see NATIVE_PIN). The availability projection depends on the
withdrawal law of that version (available = max(w - residual_m[l], 0), inactive cells carry zero requests);
other versions are refused by `withdraw_plants` until re-verified. Provenance (hashes) is checked by
`native_provenance(verify=True)` at initialisation / run-evidence time, never per invocation.
'''
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from leaf import uptake as _uptake

VERSION = '0.1.0'
NATIVE_MODULES = ('maple_syrup.subsurface', 'maple_syrup.subsurface_transpiration')
SUPPORTED_NATIVE_VERSIONS = ('0.2.0',)
NATIVE_PIN = {
    'repository_head': 'c76c603dde4e8f79ec48b18d954a386580d8bc89',
    'branch': 'feature/subsurface-hydrology',
    'subsurface_transpiration_version': '0.2.0',
    'sha256': {'subsurface.py': '5e77f14df8af4e473e8a22278700b519ccb2888c722883a9f7da508abdf14ae9',
               'subsurface_transpiration.py': '3b9191872522755924271bbaba4ae2bac290e1cad4ecffebc2d2735c4a0a9fb9'},
    'withdrawal_law': 'available = max(w - residual_m[l], 0); actual = min(requested, available); inactive cells '
                      'carry zero requests; no compensation, no clipping after subtraction; t_s preserved',
}
AVAILABILITY_PROJECTION = ('A[l, c] = cell_area_m2 * max(water_m[l, c] - residual_m[l], 0) on active cells, 0 on inactive '
                           'cells (m3). residual_m is the prepared operator\'s public residual_xp [L], the mask is '
                           'column.active_u8 [N]. This is a read-only projection of the native withdrawal law of the '
                           'pinned SYRUP version, NOT a separate water store and NOT a configurable LEAF wilting floor.')
PUBLICATION_NOTE = ('withdraw_plants is pure: it returns the fresh native state and the plant feedback as values and '
                    'publishes neither; it advances no scheduler clock and binds no interval receipt. The later dated '
                    'driver publishes native + biological state atomically with an exactly-once interval receipt.')
MEMORY_SCOPE = ('conservative stage-sum ledger of every array live during ONE withdraw_plants call: resident root map, '
                'supplied demand, original native state, native operator parameters and column mask/thickness copies, '
                'availability buffer + projection scratch, allocator plan (owned + scratch), request-depth buffer, '
                'native withdrawal outputs/rows/scratch/packet, accepted-volume buffer, allocator feedback (owned + '
                'scratch), bridge reduction packets, the native column reserve and the caller reserve. Stage buffers '
                'that are freed before the next stage are NOT reclaimed in the total (labelled upper bound). Host and '
                'device bytes are listed separately; the native side (host for numpy/numba, device for cuda) plus '
                'reserves is what is compared with the column\'s available_bytes_fn and what the native withdrawal '
                'admits itself (it receives the bridge extras as extra_reserve_bytes). NOT included: NumPy/CuPy '
                'allocator pools and retained blocks, the CUDA context, Numba/NVRTC JIT caches and compile workspaces, '
                'Python object overhead, the native hydro-step arrays (not used here), other model state and the rest '
                'of the process. A reading is not a reservation; bounds are never raised and plants never discarded.')
UNITS = {'demand_m3': 'm3 per timestep [P]', 'availability_m3': 'm3 per (layer, cell) [L, N]',
         'request_depth_m': 'm per (layer, cell) [L, N] = proposed_m3 / cell_area_m2',
         'accepted_volume_m3': 'm3 per (layer, cell) [L, N] = native actual_m * cell_area_m2',
         'delivered / unmet': 'm3 per plant [P]', 'satisfaction': 'fraction [0, 1]', 'water_m': 'm depth [L, N]',
         'cell_area_m2': 'm2 (column scalar)', 't_s': 's (native, preserved)'}
FP64_EPS = _uptake.FP64_EPS
BACKENDS = ('numpy', 'numba', 'cuda')
HOST_BACKENDS = ('numpy', 'numba')
_F8 = 8
_OPERATOR_FIELDS = ('column', 'xp', 'backend', 'n_layers', 'n_cells', 'residual_xp', 'saturation_xp', 'check_device')
_NATIVE = {}
_COMPILED = {}


# ------------------------------------------------------------------ errors
class SyrupWaterError(ValueError):
    '''Any bridge validation, projection, conservation or implementation failure (inputs untouched).'''


class NativeUnavailableError(SyrupWaterError):
    '''The native SYRUP column package cannot be imported; there is no fallback water authority.'''


class NativeContractError(SyrupWaterError):
    '''The native object or module version does not match the pinned contract this bridge was verified against.'''


class BindingError(SyrupWaterError):
    '''Root map, demand, operator, column and state do not bind (shape / namespace / dtype / device / area).'''


class SyrupWaterAdmissionError(SyrupWaterError):
    '''The declared byte budget or the native column's available bytes would be exceeded; raised BEFORE allocation.'''

    def __init__(self, stage, reason, estimate):
        super().__init__(f'{stage}: {reason}')
        self.stage = stage
        self.reason = reason
        self.estimate = estimate


# ------------------------------------------------------------------ lazy native access
def _native():
    '''(subsurface, subsurface_transpiration) modules, imported on first use only. Attributes are looked up on
    the module objects at call time (so run evidence / test spies can wrap the real withdrawal).'''
    mods = _NATIVE.get('modules')
    if mods is None:
        try:
            from maple_syrup import subsurface as ss
            from maple_syrup import subsurface_transpiration as st
        except ImportError as exc:
            raise NativeUnavailableError(f'the native SYRUP column package ({NATIVE_MODULES}) is not importable: {exc!r}; '
                                         'there is no fallback water authority') from exc
        mods = (ss, st)
        _NATIVE['modules'] = mods
    return mods


def _bound_fn():
    from maple_syrup.conservation import volume_roundoff_bound_m3
    return volume_roundoff_bound_m3


def native_status():
    '''Probe the native package without raising: {'available', 'reason', 'version', 'supported', ...}.'''
    try:
        ss, st = _native()
    except NativeUnavailableError as exc:
        return {'available': False, 'reason': str(exc), 'version': None, 'supported': False,
                'pinned_versions': SUPPORTED_NATIVE_VERSIONS}
    version = getattr(st, 'VERSION', None)
    return {'available': True, 'reason': None, 'version': version, 'supported': version in SUPPORTED_NATIVE_VERSIONS,
            'pinned_versions': SUPPORTED_NATIVE_VERSIONS, 'subsurface_file': getattr(ss, '__file__', None),
            'transpiration_file': getattr(st, '__file__', None)}


def native_provenance(*, verify=False):
    '''Pinned provenance plus the imported module facts. With verify=True the sha256 of the imported module files is
    computed and compared with NATIVE_PIN (an initialisation / run-evidence operation; never done per call).'''
    ss, st = _native()
    out = {'pin': {k: (dict(v) if isinstance(v, dict) else v) for k, v in NATIVE_PIN.items()},
           'imported_version': getattr(st, 'VERSION', None),
           'version_supported': getattr(st, 'VERSION', None) in SUPPORTED_NATIVE_VERSIONS,
           'files': {'subsurface.py': getattr(ss, '__file__', None),
                     'subsurface_transpiration.py': getattr(st, '__file__', None)},
           'verified': False}
    if verify:
        digests, matches = {}, {}
        for key, mod in (('subsurface.py', ss), ('subsurface_transpiration.py', st)):
            path = getattr(mod, '__file__', None)
            if not path:
                digests[key], matches[key] = None, False
                continue
            with open(path, 'rb') as fh:
                digests[key] = hashlib.sha256(fh.read()).hexdigest()
            matches[key] = digests[key] == NATIVE_PIN['sha256'][key]
        out.update(verified=True, sha256=digests, hash_matches=matches, all_hashes_match=all(matches.values()))
    return out


# ------------------------------------------------------------------ scalar guards
def _bytes(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise SyrupWaterError(f'{name} must be a non-boolean integer >= 0, got {v!r}')
    if int(v) < 0:
        raise SyrupWaterError(f'{name} must be >= 0, got {v!r}')
    return int(v)


def _size(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 1:
        raise BindingError(f'{name} must be a non-boolean integer >= 1, got {v!r}')
    return int(v)


def _real(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise BindingError(f'{name} must be a real host number, got {type(v).__name__}')
    out = float(v)
    if not math.isfinite(out):
        raise BindingError(f'{name} must be finite, got {v!r}')
    return out


def _is_device(a):
    return hasattr(a, '__cuda_array_interface__')


def _freeze(a):
    if isinstance(a, np.ndarray):
        a.setflags(write=False)
    return a


# ------------------------------------------------------------------ native contract / binding (metadata only)
def _check_native_array(column, a, name, shape, dtype):
    '''Exact namespace type, dtype, shape, C-contiguity and (cuda) device: metadata only, no element read.'''
    xp = column.xp
    if type(a) is not xp.ndarray:
        raise BindingError(f'{name} must be exactly a {xp.__name__}.ndarray of the column namespace (no subclass, no '
                           f'implicit transfer), got {type(a).__module__}.{type(a).__name__}')
    if a.dtype != dtype:
        raise BindingError(f'{name} must be {np.dtype(dtype)}, got {a.dtype}')
    if tuple(int(s) for s in a.shape) != tuple(shape):
        raise BindingError(f'{name} must have shape {tuple(shape)}, got {tuple(a.shape)}')
    if not a.flags.c_contiguous:
        raise BindingError(f'{name} must be C-contiguous')
    if column.backend == 'cuda' and column.device_id is not None and int(a.device.id) != int(column.device_id):
        raise BindingError(f'{name} lives on CUDA device {int(a.device.id)}, the column is bound to device '
                           f'{column.device_id}; no implicit device transfer or switch')


def check_operator(operator):
    '''Pinned-contract check of a PREPARED native operator (no allocation, no native call). Returns
    (column, xp, backend, n_layers, n_cells, cell_area_m2).'''
    ss, st = _native()
    version = getattr(st, 'VERSION', None)
    if version not in SUPPORTED_NATIVE_VERSIONS:
        raise NativeContractError(f'maple_syrup.subsurface_transpiration VERSION {version!r} is not one of the pinned '
                                  f'{SUPPORTED_NATIVE_VERSIONS}; the availability projection was verified against the '
                                  'pinned withdrawal law only (re-verify before extending the pin)')
    if not isinstance(operator, st.TranspirationOperator):
        raise NativeContractError('operator must be a maple_syrup.subsurface_transpiration.TranspirationOperator from '
                                  f'prepare_transpiration (an existing prepared operator), got {type(operator).__name__}')
    for name in _OPERATOR_FIELDS:
        if not hasattr(operator, name):
            raise NativeContractError(f'prepared operator lacks the pinned public field {name!r}')
    if not callable(operator.check_device):
        raise NativeContractError('operator.check_device must be callable')
    column = operator.column
    if not isinstance(column, ss.SoilColumn):
        raise NativeContractError(f'operator.column must be a maple_syrup.subsurface.SoilColumn, got {type(column).__name__}')
    backend = operator.backend
    if backend not in BACKENDS:
        raise NativeContractError(f'operator backend {backend!r} is not one of {BACKENDS}')
    if column.backend != backend:
        raise BindingError(f'operator backend {backend!r} differs from its column backend {column.backend!r}')
    xp = operator.xp
    if xp is not column.xp:
        raise BindingError('operator.xp is not the column namespace (operator not bound to this column)')
    if backend == 'cuda':
        if xp is np or not hasattr(xp, 'cuda'):
            raise BindingError('cuda operator must carry the CuPy namespace')
        if column.device_id is None or isinstance(column.device_id, bool):
            raise BindingError('cuda column must carry an integer device_id')
    elif xp is not np:
        raise BindingError(f'host backend {backend!r} must carry the numpy namespace')
    L = _size(operator.n_layers, 'operator.n_layers')
    N = _size(operator.n_cells, 'operator.n_cells')
    if (L, N) != (int(column.n_layers), int(column.n_cells)):
        raise BindingError(f'operator dimensions ({L}, {N}) differ from its column ({column.n_layers}, {column.n_cells})')
    area = _real(column.cell_area_m2, 'column.cell_area_m2')
    if area <= 0.0:
        raise BindingError(f'column.cell_area_m2 must be > 0, got {area!r}')
    _check_native_array(column, operator.residual_xp, 'operator.residual_xp', (L,), np.float64)
    _check_native_array(column, operator.saturation_xp, 'operator.saturation_xp', (L,), np.float64)
    _check_native_array(column, column.active_u8, 'column.active_u8', (N,), np.uint8)
    reserve = column.reserve_bytes
    if isinstance(reserve, (bool, np.bool_)) or not isinstance(reserve, (int, np.integer)) or int(reserve) < 0:
        raise NativeContractError(f'column.reserve_bytes must be a non-boolean integer >= 0, got {reserve!r}')
    return column, xp, backend, L, N, area


def _check_state(column, state, L, N):
    ss, _ = _native()
    if not isinstance(state, ss.SoilColumnState):
        raise BindingError(f'soil_state must be a maple_syrup.subsurface.SoilColumnState, got {type(state).__name__}')
    _check_native_array(column, state.water_m, 'soil_state.water_m', (L, N), np.float64)
    return _real(state.t_s, 'soil_state.t_s')


def check_binding(root_map, demand_m3, operator, soil_state):
    '''Pure validation of one invocation (no allocation, no native call): native contract, operator/column/state
    binding, root-map dimensions and namespace, demand metadata, CUDA device affinity. Returns scalar metadata.'''
    if not isinstance(root_map, _uptake.RootMap):
        raise BindingError(f'root_map must be a leaf.uptake.RootMap, got {type(root_map).__name__}')
    column, xp, backend, L, N, area = check_operator(operator)
    t_s = _check_state(column, soil_state, L, N)
    if (int(root_map.n_layers), int(root_map.n_cells)) != (L, N):
        raise BindingError(f'root map resources are ({root_map.n_layers} layers, {root_map.n_cells} cells) but the native '
                           f'column is ({L}, {N}); resources must be layer * n_cells + cell of THIS column')
    P = int(root_map.n_plants)
    device_id = None
    if backend == 'cuda':
        if root_map.backend != 'cuda':
            raise BindingError(f'cuda column requires a cuda root map, got backend {root_map.backend!r} (no implicit transfer)')
        if root_map.xp is not xp:
            raise BindingError('cuda root map must use the exact same CuPy module as the native column')
        device_id = int(column.device_id)
        for name in ('offsets', 'resources', 'weights'):
            a = getattr(root_map, name)
            if int(a.device.id) != device_id:
                raise BindingError(f'root_map.{name} lives on CUDA device {int(a.device.id)}, the column on {device_id}')
        if type(demand_m3) is not xp.ndarray:
            raise BindingError('demand_m3 must be exactly a cupy.ndarray on the column device (no implicit transfer), '
                               f'got {type(demand_m3).__name__}')
        if int(demand_m3.device.id) != device_id:
            raise BindingError(f'demand_m3 lives on CUDA device {int(demand_m3.device.id)}, the column on {device_id}')
    else:
        if root_map.backend not in HOST_BACKENDS:
            raise BindingError(f'host column ({backend}) requires a numpy or numba root map, got {root_map.backend!r}')
        if root_map.xp is not np:
            raise BindingError('host root map must use the numpy namespace')
        if _is_device(demand_m3):
            raise BindingError('demand_m3 is a device array but the column is a host column; no implicit transfer')
        if not isinstance(demand_m3, np.ndarray):
            raise BindingError(f'demand_m3 must be a numpy.ndarray [P], got {type(demand_m3).__name__}')
    if demand_m3.ndim != 1 or int(demand_m3.shape[0]) != P:
        raise BindingError(f'demand_m3 must have shape ({P},) (one value per root-map row), got {tuple(demand_m3.shape)}')
    if demand_m3.dtype.kind not in 'fiu':
        raise BindingError(f'demand_m3 must be a real numeric array, got dtype {demand_m3.dtype}')
    operator.check_device()  # current CUDA device == column device, column preflight; no allocation
    return {'map_backend': root_map.backend, 'native_backend': backend, 'side': 'device' if backend == 'cuda' else 'host',
            'namespace': xp.__name__, 'device_id': device_id, 'n_plants': P, 'n_edges': int(root_map.n_edges),
            'n_layers': L, 'n_cells': N, 'n_active': int(column.n_active), 'cell_area_m2': area, 't_s': t_s,
            'native_version': getattr(_native()[1], 'VERSION', None), 'demand_dtype': str(demand_m3.dtype),
            'demand_bytes': int(demand_m3.nbytes), 'column_reserve_bytes': int(column.reserve_bytes)}


# ------------------------------------------------------------------ memory ledger (pure arithmetic)
def _projection_scratch_bytes(native_backend, L, N):
    if native_backend == 'numpy':
        return 2 * L * N                      # two bool [L, N] validity masks live at once
    if native_backend == 'numba':
        return 2 * _F8 * N                    # [2, N] per-cell flag / sum rows
    return L * N + _F8 * N + 2 * _F8          # uint8 flag [L, N] + reduction workspace + 2-double packet


def estimate_availability_bytes(n_layers, n_cells, *, native_backend, column_reserve_bytes=0, extra_reserve_bytes=0):
    '''Bytes of `availability_snapshot`: the state [L, N] (caller-held, counted), the operator parameters and
    column masks (resident), the owned A buffer, projection scratch, plus reserves.'''
    L, N = _size(n_layers, 'n_layers'), _size(n_cells, 'n_cells')
    if native_backend not in BACKENDS:
        raise SyrupWaterError(f'native_backend must be one of {BACKENDS}')
    reserve, extra = _bytes(column_reserve_bytes, 'column_reserve_bytes'), _bytes(extra_reserve_bytes, 'extra_reserve_bytes')
    LN = _F8 * L * N
    side = 'device' if native_backend == 'cuda' else 'host'
    entries = {'native_state_input': (LN, side), 'native_operator_parameters': (2 * _F8 * L, side),
               'native_column_resident': (2 * N + _F8 * L, side), 'availability_buffer': (LN, side),
               'availability_scratch': (_projection_scratch_bytes(native_backend, L, N), side),
               'native_column_reserve': (reserve, 'reserve'), 'caller_reserve': (extra, 'reserve')}
    return _totals('syrup_water.availability_snapshot', entries, side, n_layers=L, n_cells=N, native_backend=native_backend)


def _totals(stage, entries, side, **meta):
    host = sum(b for b, s in entries.values() if s == 'host')
    device = sum(b for b, s in entries.values() if s == 'device')
    reserve = sum(b for b, s in entries.values() if s == 'reserve')
    native_side = device if side == 'device' else host
    out = dict(meta)
    out.update(stage=stage, side=side, entries={k: {'bytes': int(b), 'side': s} for k, (b, s) in entries.items()},
               host_bytes=int(host), device_bytes=int(device), reserve_bytes=int(reserve),
               total_bytes=int(host + device + reserve), native_side_bytes=int(native_side),
               native_required_bytes=int(native_side + reserve), peak_policy='conservative stage sum (upper bound)',
               scope=MEMORY_SCOPE)
    return out


def estimate_bridge_bytes(n_plants, n_edges, n_layers, n_cells, *, map_backend, native_backend, demand_bytes=None,
                          map_resident_bytes=None, column_reserve_bytes=0, extra_reserve_bytes=0):
    '''Conservative stage-sum ledger of ONE withdraw_plants call (see MEMORY_SCOPE). Reuses the native
    `estimate_transpiration_memory` components (state counted once, here) and the allocator's plan / feedback
    estimates; nothing is allocated. `native_extra_reserve_bytes` is what the native withdrawal is told about the
    bridge's other simultaneously live arrays on its side, so its own admission sees the same total.'''
    ss, st = _native()
    P, E = _bytes(n_plants, 'n_plants'), _bytes(n_edges, 'n_edges')
    L, N = _size(n_layers, 'n_layers'), _size(n_cells, 'n_cells')
    if map_backend not in BACKENDS or native_backend not in BACKENDS:
        raise SyrupWaterError(f'backends must be one of {BACKENDS}, got map {map_backend!r} / native {native_backend!r}')
    side = 'device' if native_backend == 'cuda' else 'host'
    if ('device' if map_backend == 'cuda' else 'host') != side:
        raise BindingError(f'root map backend {map_backend!r} and native backend {native_backend!r} live on different sides')
    reserve = _bytes(column_reserve_bytes, 'column_reserve_bytes')
    extra = _bytes(extra_reserve_bytes, 'extra_reserve_bytes')
    R = L * N
    LN = _F8 * L * N
    tm = st.estimate_transpiration_memory(L, N, backend=native_backend, reserve_bytes=0, extra_reserve_bytes=0,
                                          include_state=False)
    plan = _uptake.estimate_plan_bytes(P, E, R, map_backend)
    feedback = _uptake.estimate_feedback_bytes(P, E, R, map_backend)
    map_res = (_uptake.estimate_map_bytes(P, E, R, map_backend)['owned_bytes'] if map_resident_bytes is None
               else _bytes(map_resident_bytes, 'map_resident_bytes'))
    demand = _F8 * P if demand_bytes is None else _bytes(demand_bytes, 'demand_bytes')
    native_own = int(tm.parameter_bytes) + LN + int(tm.input_bytes) + int(tm.output_bytes) + int(tm.row_bytes) + int(tm.scratch_bytes)
    entries = {
        'root_map_resident': (map_res, side),
        'demand_input': (demand, side),
        'native_state_input': (LN, side),
        'native_operator_parameters': (int(tm.parameter_bytes), side),
        'native_column_resident': (2 * N + _F8 * L, side),
        'native_layer_vectors_host': (int(ss.layer_vector_doubles(L)) * _F8, 'host'),
        'availability_buffer': (LN, side),
        'availability_scratch': (_projection_scratch_bytes(native_backend, L, N), side),
        'allocator_plan': (int(plan['total_bytes']), side),
        'request_depth_buffer': (int(tm.input_bytes), side),
        'native_withdrawal_outputs': (int(tm.output_bytes), side),
        'native_withdrawal_rows_scratch': (int(tm.row_bytes) + int(tm.scratch_bytes), side),
        'accepted_volume_buffer': (LN, side),
        'allocator_feedback': (int(feedback['total_bytes']), side),
        'bridge_reduction_packets': (8 * _F8, 'host'),
        'native_column_reserve': (reserve, 'reserve'),
        'caller_reserve': (extra, 'reserve'),
    }
    if native_backend == 'cuda':
        entries['native_host_packet'] = (int(tm.packet_bytes), 'host')
    out = _totals('syrup_water.withdraw_plants', entries, side, n_plants=P, n_edges=E, n_layers=L, n_cells=N,
                  n_resources=R, map_backend=map_backend, native_backend=native_backend)
    native_extra = out['native_side_bytes'] - native_own + extra
    if native_extra < 0:  # pragma: no cover - ledger invariant
        raise SyrupWaterError('implementation failure: native-side ledger smaller than the native estimate')
    out.update(native_estimate=tm.as_dict(), plan_estimate=plan, feedback_estimate=feedback,
               native_own_bytes=int(native_own), native_extra_reserve_bytes=int(native_extra),
               allocator_budget_bytes=int(map_res + int(plan['total_bytes']) + int(feedback['total_bytes'])),
               excluded='see scope: allocator pools, CUDA context, JIT caches, Python objects, hydro-step arrays')
    return out


def _admit(stage, ledger, max_bytes, column, *, root_map=None):
    '''Refuse BEFORE allocation: explicit budget, the root map's own declared budget (advisory pre-check of the
    allocator's admission so no refusal can follow the native call), and the native column's available bytes.'''
    mb = _bytes(max_bytes, 'max_bytes')
    report = dict(ledger, max_bytes=mb, available_bytes=None)
    if int(ledger['total_bytes']) > mb:
        raise SyrupWaterAdmissionError(stage, f'estimated {ledger["total_bytes"]} bytes (host {ledger["host_bytes"]} + device '
                                              f'{ledger["device_bytes"]} + reserves {ledger["reserve_bytes"]}) exceed '
                                              f'max_bytes={mb}; nothing allocated, no native call', report)
    if root_map is not None and int(ledger['allocator_budget_bytes']) > int(root_map.max_bytes):
        raise SyrupWaterAdmissionError(stage, f'root map budget max_bytes={root_map.max_bytes} is below the resident map + '
                                              f'plan + feedback estimate {ledger["allocator_budget_bytes"]}; nothing allocated',
                                       report)
    fn = getattr(column, 'available_bytes_fn', None)
    if fn is not None:
        avail = fn()
        if isinstance(avail, (bool, np.bool_)) or not isinstance(avail, (int, np.integer)) or int(avail) < 0:
            raise SyrupWaterError(f'column.available_bytes_fn must return a non-boolean integer >= 0, got {avail!r}')
        report['available_bytes'] = int(avail)
        if int(ledger['native_required_bytes']) > int(avail):
            raise SyrupWaterAdmissionError(stage, f'native side needs {ledger["native_required_bytes"]} bytes (reserves '
                                                  f'included) but the column reports {int(avail)} available; nothing '
                                                  'allocated, no native call', report)
    return report


# ------------------------------------------------------------------ availability projection (one algorithm, three paths)
def _k_project(w, residual, satb, active_u8, area, out, rows):
    '''Authoritative per-cell definition (compiled by numba; mirrored by the NumPy and CuPy paths).
    out[l, j] = area * max(w - residual[l], 0) on active cells, 0 on inactive cells;
    rows[0, j] = 1 if any layer of cell j is not native-valid (NaN / negative / above saturation bound);
    rows[1, j] = per-cell sum of out (layer-sequential).'''
    L, N = w.shape
    for j in range(N):
        rows[0, j] = 0.0
        rows[1, j] = 0.0
    for l in range(L):
        r = residual[l]
        sb = satb[l]
        for j in range(N):
            wv = w[l, j]
            if not (wv >= 0.0 and wv <= sb):
                rows[0, j] = 1.0
            a = wv - r
            if not (a > 0.0):
                a = 0.0
            if active_u8[j] == 0:
                a = 0.0
            a = a * area
            out[l, j] = a
            rows[1, j] += a


_CUPY_PROJECT_SRC = r'''
    long long l = i / n_cells;
    long long j = i - l * n_cells;
    double wv = w[i];
    flag = (wv >= 0.0 && wv <= satb[l]) ? (unsigned char)0 : (unsigned char)1;
    double a = wv - residual[l];
    if (!(a > 0.0)) a = 0.0;
    if (active[j] == 0) a = 0.0;
    out = a * area;
'''


def _numba_project():
    if 'numba' not in _COMPILED:
        from numba import njit
        _COMPILED['numba'] = njit(cache=True)(_k_project)
    return _COMPILED['numba']


def _cupy_project(xp):
    if 'cuda' not in _COMPILED:
        _COMPILED['cuda'] = xp.ElementwiseKernel(
            'raw float64 w, raw float64 residual, raw float64 satb, raw uint8 active, float64 area, int64 n_cells',
            'float64 out, uint8 flag', _CUPY_PROJECT_SRC, 'leaf_syrup_water_project')
    return _COMPILED['cuda']


def _project(operator, column, xp, water, L, N, area):
    '''Allocate the owned A buffer and fill it; validate native content from the same pass. Returns (A, total).'''
    out = xp.empty((L, N), dtype=xp.float64)
    if operator.backend == 'numpy':
        np.subtract(water, operator.residual_xp[:, None], out=out)
        mask = out > 0.0
        np.logical_not(mask, out=mask)
        np.copyto(out, 0.0, where=mask)            # max(w - residual, 0) with NaN / -0.0 -> +0.0
        del mask
        np.multiply(out, area, out=out)
        np.multiply(out, column.active_u8, out=out)  # (N,) broadcast over layers: inactive cells -> 0
        ok = water >= 0.0
        ok &= water <= operator.saturation_xp[:, None]
        valid = bool(ok.all())
        del ok
        total = float(out.sum())
    elif operator.backend == 'numba':
        rows = np.empty((2, N), dtype=np.float64)
        _numba_project()(water, operator.residual_xp, operator.saturation_xp, column.active_u8, float(area), out, rows)
        valid = float(rows[0].max()) == 0.0
        total = float(rows[1].sum())
        del rows
    else:
        flag = xp.empty((L, N), dtype=xp.uint8)
        _cupy_project(xp)(water, operator.residual_xp, operator.saturation_xp, column.active_u8, np.float64(area),
                          np.int64(N), out, flag)
        packet = xp.stack([flag.max().astype(xp.float64), out.sum()])   # ONE small host packet
        p = np.asarray(packet.get(), dtype=np.float64)
        del flag, packet
        valid = float(p[0]) == 0.0
        total = float(p[1])
    if not valid:
        raise SyrupWaterError('soil water is not native-valid on at least one cell (non-finite, negative or above the '
                              'layer saturation bound; inactive cells included); no proposal and no native withdrawal '
                              'were attempted and the state is untouched')
    if not math.isfinite(total):
        raise SyrupWaterError('implementation failure: availability total is not finite')
    return _freeze(out), total


@dataclass(frozen=True, eq=False)
class AvailabilitySnapshot:
    '''Read-only projection of the native inventory (see AVAILABILITY_PROJECTION). `volume_m3` [L, N] lives on the
    column namespace (host arrays are frozen). Not a water store: it is valid only for the state it was taken from.'''
    volume_m3: object
    total_m3: float
    n_layers: int
    n_cells: int
    n_active: int
    cell_area_m2: float
    backend: str
    native_version: str
    t_s: float
    projection: str = AVAILABILITY_PROJECTION


def _snapshot(operator, column, xp, backend, state, L, N, area, t_s):
    A, total = _project(operator, column, xp, state.water_m, L, N, area)
    return AvailabilitySnapshot(A, total, L, N, int(column.n_active), area, backend,
                                getattr(_native()[1], 'VERSION', None), t_s)


def availability_snapshot(operator, soil_state, *, max_bytes, extra_reserve_bytes=0):
    '''Construct the read-only availability snapshot A [L, N] (m3) for one native state: validation -> scalar
    admission (max_bytes, available_bytes_fn) -> ONE owned [L, N] allocation -> content validity check.'''
    column, xp, backend, L, N, area = check_operator(operator)
    t_s = _check_state(column, soil_state, L, N)
    operator.check_device()
    ledger = estimate_availability_bytes(L, N, native_backend=backend, column_reserve_bytes=int(column.reserve_bytes),
                                         extra_reserve_bytes=extra_reserve_bytes)
    _admit('syrup_water.availability_snapshot', ledger, max_bytes, column)
    return _snapshot(operator, column, xp, backend, soil_state, L, N, area, t_s)


# ------------------------------------------------------------------ result
@dataclass(frozen=True, eq=False)
class PlantWithdrawal:
    '''Result of ONE pure bridge invocation. NOT published: `native_state` is the fresh authoritative SoilColumnState
    (t_s preserved) that the later dated driver may publish together with the biological state; `native` is the
    native TranspirationResult (actual_m, unmet_m, scale, budget, memory); `plan` / `feedback` are the allocator's
    UptakePlan / UptakeFeedback (feedback.delivered / unmet / satisfaction / accepted / closure).'''
    native_state: object
    native: object
    plan: object
    feedback: object
    availability: AvailabilitySnapshot
    request_depth_m: object
    accepted_volume_m3: object
    conservation: dict
    memory: dict
    binding: dict
    publication: str = PUBLICATION_NOTE

    @property
    def delivered_m3(self):
        return self.feedback.delivered

    @property
    def unmet_m3(self):
        return self.feedback.unmet

    @property
    def satisfaction(self):
        return self.feedback.satisfaction

    @property
    def t_s(self):
        return self.native_state.t_s


def _reduce(xp, arrays):
    '''Scalar totals of several arrays in ONE small host packet (no population read-back).'''
    packet = xp.stack([xp.asarray(xp.sum(a), dtype=xp.float64) for a in arrays])
    host = packet.get() if hasattr(packet, 'get') else packet
    return [float(v) for v in np.asarray(host, dtype=np.float64)]


def _conservation(root_map, plan, feedback, native, accepted_volume, demand, xp, L, N):
    P, E, R = int(root_map.n_plants), int(root_map.n_edges), int(root_map.n_resources)
    acc_sum, proposed, delivered, unmet, demand_total = _reduce(
        xp, [accepted_volume, plan.proposed, feedback.delivered, feedback.unmet, demand])
    budget = native.budget
    native_actual = float(budget.actual_m3)
    native_requested = float(budget.requested_m3)
    magnitude = max(abs(native_actual), abs(acc_sum), abs(delivered), abs(proposed), abs(native_requested))
    bound = _bound_fn()
    tol_native = float(budget.bound_m3)                                 # native active-cell closure bound
    tol_area = float(bound(2 * L * N + 8, magnitude))                    # two orderings of the L*N area products
    tol_alloc = float(_uptake.closure_tolerance_m3(magnitude, P, E, R))  # allocator gather / scatter policy
    per_res = int(getattr(root_map, 'max_edges_per_resource', E))
    tol_clip = FP64_EPS * (per_res + 2) * abs(proposed)                  # allocator ulp-level excess handling
    tol = tol_native + tol_area + tol_alloc + tol_clip
    residual = delivered - native_actual
    closed = abs(residual) <= tol
    out = {'native_storage_before_m3': float(budget.storage_before_m3), 'native_storage_after_m3': float(budget.storage_after_m3),
           'native_requested_m3': native_requested, 'native_actual_m3': native_actual, 'native_unmet_m3': float(budget.unmet_m3),
           'native_residual_m3': float(budget.residual_m3), 'native_bound_m3': tol_native,
           'demand_m3': demand_total, 'proposed_m3': proposed, 'accepted_volume_m3': acc_sum, 'delivered_m3': delivered,
           'plant_unmet_m3': unmet, 'area_conversion_roundoff_m3': acc_sum - native_actual,
           'delivered_minus_native_actual_m3': residual, 'tolerance_m3': tol,
           'tolerance_terms_m3': {'native_bound': tol_native, 'area_products': tol_area, 'allocator_closure': tol_alloc,
                                  'allocator_excess': tol_clip},
           'closed': bool(closed), 'requested_minus_proposed_m3': native_requested - proposed,
           'requested_matches_proposal': abs(native_requested - proposed) <= tol_native + tol_area,
           'allocator_closure': dict(feedback.closure),
           'definition': 'sum(delivered) == native actual volume (ACTIVE cells, area multiplied once natively) within the '
                         'native FP64 bound + the allocator FP64 closure policy; accepted <= proposed is enforced by '
                         'leaf.uptake.accept; honest area-conversion roundoff is reported, never balanced away'}
    if not closed:
        raise SyrupWaterError(f'implementation failure: sum(delivered) - native actual = {residual!r} m3 exceeds the FP64 '
                              f'tolerance {tol!r} m3 (not a scientific result); nothing published: {out}')
    return out


# ------------------------------------------------------------------ the bridge
def withdraw_plants(root_map, demand_m3, operator, soil_state, *, max_bytes, extra_reserve_bytes=0):
    '''One pure plant withdrawal against the native soil column (module docstring). Returns PlantWithdrawal.
    Order: validation (no allocation) -> explicit scalar admission -> availability snapshot -> leaf.uptake.propose
    -> ONE native withdraw_transpiration -> leaf.uptake.accept on the ACTUAL accepted volume -> conservation check.
    Inputs are never written; a failure at any stage leaves them and the native time unchanged.'''
    ss, st = _native()
    mb = _bytes(max_bytes, 'max_bytes')
    extra = _bytes(extra_reserve_bytes, 'extra_reserve_bytes')
    binding = check_binding(root_map, demand_m3, operator, soil_state)
    column, xp, backend = operator.column, operator.xp, binding['native_backend']
    L, N, area = binding['n_layers'], binding['n_cells'], binding['cell_area_m2']
    ledger = estimate_bridge_bytes(binding['n_plants'], binding['n_edges'], L, N, map_backend=root_map.backend,
                                   native_backend=backend, demand_bytes=binding['demand_bytes'],
                                   map_resident_bytes=int(root_map.resident_bytes),
                                   column_reserve_bytes=binding['column_reserve_bytes'], extra_reserve_bytes=extra)
    memory = _admit('syrup_water.withdraw_plants', ledger, mb, column, root_map=root_map)
    # ---- allocation begins (everything below is bounded by the admitted ledger)
    snapshot = _snapshot(operator, column, xp, backend, soil_state, L, N, area, binding['t_s'])
    plan = _uptake.propose(root_map, demand_m3, snapshot.volume_m3)
    request_depth = xp.ascontiguousarray(plan.proposed_grid() / area)      # volume -> depth, exactly once
    native = st.withdraw_transpiration(operator, soil_state, request_depth,   # the ONE native withdrawal
                                       extra_reserve_bytes=int(ledger['native_extra_reserve_bytes']))
    accepted_volume = native.actual_m * area                                # ACTUAL depth -> volume, exactly once
    feedback = _uptake.accept(plan, accepted_volume)
    conservation = _conservation(root_map, plan, feedback, native, accepted_volume, demand_m3, xp, L, N)
    t_after = _real(native.state.t_s, 'native.state.t_s')
    if t_after != binding['t_s']:
        raise SyrupWaterError(f'implementation failure: native time changed from {binding["t_s"]!r} to {t_after!r}')
    memory = dict(memory, native_withdrawal_memory=native.memory.as_dict(), plan_admission=plan.summary.get('admission'),
                  feedback_admission=feedback.summary.get('admission'))
    return PlantWithdrawal(native.state, native, plan, feedback, snapshot, _freeze(request_depth), _freeze(accepted_volume),
                           conservation, memory, dict(binding, units=dict(UNITS), native_pin=dict(NATIVE_PIN)))


__all__ = ['VERSION', 'NATIVE_MODULES', 'SUPPORTED_NATIVE_VERSIONS', 'NATIVE_PIN', 'AVAILABILITY_PROJECTION',
           'PUBLICATION_NOTE', 'MEMORY_SCOPE', 'UNITS', 'BACKENDS', 'HOST_BACKENDS', 'SyrupWaterError',
           'NativeUnavailableError', 'NativeContractError', 'BindingError', 'SyrupWaterAdmissionError', 'native_status',
           'native_provenance', 'check_operator', 'check_binding', 'estimate_availability_bytes', 'estimate_bridge_bytes',
           'AvailabilitySnapshot', 'availability_snapshot', 'PlantWithdrawal', 'withdraw_plants']
