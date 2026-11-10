'''Coupled daily driver: native layered soil column (one physical water authority) + individual plants.

Order per accepted day (dt = 86400 s, DAILY-ONLY interface):
  1. request depth[L, N] = R[L, N] / cell_area (R is the resident potential-demand cache in m3/day),
     zeroed on inactive cells (horizontal weights are NOT renormalized; that demand stays unmapped);
  2. native `step_with_transpiration`: hydro step (intake -> return -> drainage -> E -> diffusion,
     advances t_s once) then explicit residual-limited transpiration withdrawal (time preserved);
  3. compiled gather of the per-plant ACTUAL uptake candidate from the native scale (same piece that was
     requested: Dp * w * species_frac * scale); candidates must be finite, non-negative and their sum must
     equal native actual depth * area within the MAPLE FP64 bound BEFORE anything is published;
  4. publish soil state / counters, accumulate plant uptake, then the existing transactional weekly
     biomass/radius/CSR/demand refresh every 7 accepted days (refusal latches; retry_weekly_update()).
No synthetic recharge, no LEAF soil store, no inferred storm/wind phases (refused explicitly).
'''
from __future__ import annotations

import math
import time

import numpy as np

from benchmarks import daily_water_growth as dwg
from benchmarks import spatial_mapping as sm

DAY_S = 86400.0
ALLOWED_PHASES = ('interevent',)
EXCLUDED_PHASES = ('wind', 'rain', 'water', 'event', 'storm')
AREA_RTOL = 4.0 * np.finfo(np.float64).eps
VERSION = '0.1.0'


class CoupledPhaseError(ValueError):
    '''Raised before any work when a non-interevent (wind/rain/water event) phase is requested.'''


def _native():
    '''Lazy native imports (vegetation code never imports maple_syrup at module import time). The opt-in
    extension must be importable as maple_syrup.subsurface_transpiration (installed alongside the column
    package or supplied by the test overlay); there is NO path fallback.'''
    from maple_syrup import subsurface as ss
    try:
        from maple_syrup import subsurface_transpiration as st
    except ImportError as exc:
        raise ImportError('maple_syrup.subsurface_transpiration (the opt-in transpiration branch of the native '
                          'column package) is not importable; install/overlay it alongside maple_syrup.subsurface'
                          ) from exc
    from maple_syrup.conservation import volume_roundoff_bound_m3
    try:
        from maple.core.backend import to_host
    except ImportError:  # pragma: no cover
        def to_host(a):
            return np.asarray(a.get() if hasattr(a, 'get') else a)
    return ss, st, volume_roundoff_bound_m3, to_host


def check_phase(phase):
    if not isinstance(phase, str):
        raise CoupledPhaseError('phase must be a string')
    p = phase.lower()
    if p in ALLOWED_PHASES:
        return p
    if any(k in p for k in EXCLUDED_PHASES):
        raise CoupledPhaseError(f'phase {phase!r} is a wind/rain/water event phase; the coupled daily driver only '
                                'integrates interevent days (storm/MASY integration is not implemented)')
    raise CoupledPhaseError(f'unknown phase {phase!r}; allowed: {ALLOWED_PHASES}')


class CoupledSimulation:
    '''Composes a water-free vegetation Simulation with a prepared native transpiration operator.

    .state / .soil_state : current native SoilColumnState (authoritative water, never copied)
    .vegetation          : benchmarks.daily_water_growth.Simulation(vegetation_only=True)
    .roots               : CSR graph + layer fractions + profile ids (rows are the native layers, top first)
    '''

    def __init__(self, case, column, state, inputs=None, *, extra_reserve_bytes=0):
        ss, st, bound_fn, to_host = self._ss, self._st, self._bound_fn, self._to_host = _native()
        ss._check_column(column)
        self.case = case = dwg.validate_case(case)
        L, N = case['layers'], case['nx'] * case['ny']
        if int(column.n_cells) != N:
            raise ValueError(f'column has {column.n_cells} cells, case grid nx*ny = {N} (row-major j*nx+i)')
        if int(column.n_layers) != L:
            raise ValueError(f'column has {column.n_layers} layers, case layers = {L}')
        area, dx2 = float(column.cell_area_m2), float(case['dx']) ** 2
        if not (math.isfinite(area) and area > 0.0) or abs(area - dx2) > AREA_RTOL * dx2:
            raise ValueError(f'column cell_area_m2 {area!r} differs from dx^2 {dx2!r}')
        if column.backend == 'cuda':
            if sm.cp is None or column.xp is not sm.cp:
                raise ValueError('CUDA column must use the same CuPy namespace as the vegetation GPU backend')
            if column.device_id is not None and int(sm.cp.cuda.Device().id) != int(column.device_id):
                raise ValueError(f'current CUDA device differs from column device {column.device_id}')
            backend = 'gpu'
        elif column.backend in ('numpy', 'numba'):
            if column.xp is not np:
                raise ValueError('CPU column must use the numpy namespace')
            backend = 'cpu'
        else:
            raise ValueError(f'unsupported column backend {column.backend!r}')
        ss._check_state(column, state)
        self.column, self.backend, self.gpu, self.xp = column, backend, backend == 'gpu', column.xp
        self.L, self.N, self.area = L, N, area
        P = int(len(inputs['x'])) if inputs is not None else case['grass_count'] + case['shrub_count']
        if P < 1:
            raise ValueError('population must be positive')
        self.extra_reserve_bytes = extra = dwg._as_int(extra_reserve_bytes, 'extra_reserve_bytes', 0)
        # --- cheap aggregate scalar lower bound BEFORE any P/N allocation: native (soil state + hydro + T
        #     arrays/parameters, column reserve) + coupling buffers + vegetation state/scratch with ZERO edges
        comb = st.estimate_combined_memory(column, extra_reserve_bytes=extra, query_available=False)
        self.native_side_bytes = side = comb.device_bytes if self.gpu else comb.host_bytes
        self.buffer_bytes = 8 * L * N + 8 * P + 8 * N  # request depth [L,N] + candidate [P] + active mask [N]
        self.operator_param_bytes = 2 * 8 * L
        self.column_reserve_bytes = dwg._as_int(int(column.reserve_bytes), 'column.reserve_bytes', 0)
        # external reserve counted by every vegetation ledger under case.max_bytes: native side + soil state
        # + coupling buffers + operator parameters + the column's own explicit reserve + caller reserve
        native_reserve = (side + 8 * L * N + self.buffer_bytes + self.operator_param_bytes
                          + self.column_reserve_bytes + extra)
        S = len(dwg.resolve_root_profile_names(case)[0])  # pure (no table materialized)
        veg_lower = (dwg.state_bytes_estimate(P, N, L, S, soil_maps=dwg._N_VEG_ONLY_SOIL_MAPS)
                     + dwg.scratch_allowance(P, N, L, S) + P + 4 * P + 8 * S * L)
        if veg_lower + native_reserve > case['max_bytes']:
            raise sm.AdmissionRejected('coupled.init_scalar_bound',
                                       f'vegetation state (zero edges) + native soil/operator reserve exceeds '
                                       f'max_bytes {case["max_bytes"]}; nothing allocated',
                                       {'vegetation_lower_bytes': veg_lower, 'native_reserve_bytes': native_reserve})
        self.last_step = None
        self.peak_required_bytes = int(comb.required_bytes) + veg_lower
        self._native_admit('coupled.init_scalar_bound', veg_lower + self.buffer_bytes + extra)
        # --- vegetation Simulation: its own init ledgers (zero-edge, full edge bound, generated) count the
        #     native reserve under max_bytes AND are passed to the strict native-availability check
        self.vegetation = veg = dwg.Simulation(case, backend, inputs=inputs, vegetation_only=True,
                                                 external_reserve_bytes=native_reserve,
                                                 admission_check=self._vegetation_admission)
        self.P = veg.P
        veg_bytes = veg.live_bytes() + veg.graph_bytes + self.buffer_bytes
        self._native_admit('coupled.init', veg_bytes + extra)  # full admission with the actual graph
        self.operator = st.prepare_transpiration(column, extra_reserve_bytes=veg_bytes + extra)
        xp = self.xp
        self.request_m = xp.zeros((L, N), dtype=xp.float64)
        self.candidate = xp.zeros(P, dtype=xp.float64)
        self.active_f = xp.ascontiguousarray(xp.asarray(column.active_u8, dtype=xp.float64))
        self._state = state
        self.t0_s = float(state.t_s)
        self.days = 0
        self.storage_initial_m3 = float(ss.storage_m3(column, state))
        self.cum = {k: 0.0 for k in ('added', 'surface_return', 'drainage', 'evaporation', 'evaporation_unmet',
                                      'transpiration', 'transpiration_unmet', 'requested', 'potential', 'mapped',
                                      'plant_actual', 'bound', 'flux_bound')}
        self.peak_required_bytes = max(self.peak_required_bytes, int(comb.required_bytes) + veg_bytes)
        self.n_active = int(column.n_active)
        self.t_native = self.t_gather = 0.0
        if self.gpu:
            sm.cp.cuda.Device().synchronize()

    # ---- properties
    @property
    def state(self):
        return self._state

    soil_state = state

    @property
    def roots(self):
        v = self.vegetation
        return {'graph': v.graph, 'root_layer_fractions': v.root_layer_fractions,
                'root_profile_ids': v.root_profile_ids, 'profile_names': list(v.profile_names),
                'layers_top_first': self.L}

    @property
    def weekly_pending(self):
        return self.vegetation.weekly_pending

    # ---- admission helpers
    def _veg_live_bytes(self):
        v = self.vegetation
        diag = 0 if self.last_step is None else self.last_step.diagnostic_bytes()
        return v.live_bytes() + v.graph_bytes + self.buffer_bytes + diag + self.extra_reserve_bytes

    def _native_admit(self, stage, extra_bytes):
        comb = self._st.estimate_combined_memory(self.column, extra_reserve_bytes=extra_bytes)
        if comb.available_bytes >= 0 and comb.required_bytes > comb.available_bytes:
            raise sm.AdmissionRejected(stage, f'native combined peak {comb.required_bytes} bytes exceeds available '
                                              f'{comb.available_bytes} bytes', comb.as_dict())
        return comb

    def _vegetation_admission(self, ledger):
        '''Strict native-availability check over a fully reserved vegetation ledger (init bounds, weekly
        rebuild bound and actual old+new graphs, retries). The ledger already contains the native external
        reserve; the native combined requirement is re-derived with the vegetation part (state + old/new
        graphs + scratch + mirrors) plus buffers, retained diagnostics and the caller reserve, and compared
        with a fresh availability reading. Raising here refuses BEFORE any graph allocation (weekly work
        latches). Also feeds the overall admitted-peak estimate.'''
        self._check_device('coupled.vegetation_admission')
        # The ledger is FULLY reserved: vegetation state + old/new graphs + scratch + mirrors + the external
        # reserve (native soil/hydro/T arrays and parameters, soil state, buffers, column reserve, retained
        # diagnostics, caller reserve). That single simultaneous-live peak is compared with BOTH bounds.
        peak = int(ledger['total_bytes'])
        if int(ledger.get('external_reserve_bytes', 0)) < self._native_reserve_bytes():
            raise sm.AdmissionRejected('coupled.vegetation_admission',
                                       f"{ledger.get('stage')}: ledger external reserve is stale (retained native "
                                       'diagnostics not counted); nothing allocated', dict(ledger))
        comb = self._st.estimate_combined_memory(self.column, extra_reserve_bytes=0)  # strict availability reading
        est = {'stage': ledger.get('stage'), 'peak_bytes': peak, 'ledger': dict(ledger), 'native': comb.as_dict(),
               'max_bytes': self.case['max_bytes']}
        if peak > self.case['max_bytes']:
            raise sm.AdmissionRejected('coupled.vegetation_admission',
                                       f"{ledger.get('stage')}: full simultaneous-live peak {peak} bytes exceeds "
                                       f"case max_bytes {self.case['max_bytes']}; nothing allocated", est)
        if comb.available_bytes >= 0 and peak > comb.available_bytes:
            raise sm.AdmissionRejected('coupled.vegetation_admission',
                                       f"{ledger.get('stage')}: full simultaneous-live peak {peak} bytes exceeds "
                                       f'native available {comb.available_bytes} bytes; nothing allocated, prior '
                                       'state preserved', est)
        self.peak_required_bytes = max(self.peak_required_bytes, peak)  # published only on admission

    # ---- daily stepping
    def step_day(self, added_water_m, potential_evaporation_m_s, *, phase='interevent'):
        '''One accepted physical day. Any native/gather/admission failure leaves the prior soil state,
        plant accumulators and counters unchanged (no partial day). A refused WEEKLY rebuild after an
        accepted day raises AdmissionRejected/RuntimeError with the day already accepted and latched.'''
        check_phase(phase)
        veg, xp, st = self.vegetation, self.xp, self._st
        veg._refuse_if_latched()
        self._check_device('step_day')  # before any request-buffer operation
        if added_water_m is None or potential_evaporation_m_s is None:
            raise ValueError('explicit daily forcing (added_water_m[L,N], potential_evaporation_m_s[N]) is required; '
                             'no synthetic recharge is inferred')
        t0 = time.perf_counter()
        # 1) requested depth from the resident demand cache (no state change yet)
        xp.multiply(veg.R, 1.0 / self.area, out=self.request_m)
        xp.multiply(self.request_m, self.active_f, out=self.request_m)
        # 2) release obsolete diagnostics, admit, run native hydro + withdrawal (pure; nothing published on failure)
        self.last_step = None
        extra = self._veg_live_bytes()
        res = st.step_with_transpiration(self.operator, self._state, DAY_S, added_water_m=added_water_m,
                                         potential_evaporation_m_s=potential_evaporation_m_s,
                                         requested_transpiration_m=self.request_m, extra_reserve_bytes=extra)
        self.t_native += time.perf_counter() - t0
        t1 = time.perf_counter()
        # 3) compiled gather of the per-plant actual candidate; validate BEFORE publishing anything
        cand = veg.gather_uptake_candidate(res.scale, self.candidate)
        packet = np.asarray(self._to_host(xp.stack([cand.sum(), cand.min(), cand.max(), veg.R.sum(), veg.Dp.sum(),
                                                     self.request_m.sum()]).astype(xp.float64)), dtype=np.float64)
        csum, cmin, cmax, rsum, dsum, qsum = (float(v) for v in packet)
        if not (math.isfinite(cmin) and math.isfinite(cmax) and math.isfinite(csum)) or cmin < 0.0:
            raise RuntimeError(f'day {self.days}: per-plant uptake candidate not finite/non-negative; day rejected')
        tb = res.transpiration_budget
        n_edges = int(veg.graph['cells'].shape[0])
        fbound = float(self._bound_fn(self.L * self.N + n_edges * self.L + self.P + 2, max(tb.actual_m3, dsum, 0.0)))
        if abs(csum - tb.actual_m3) > fbound:
            raise RuntimeError(f'day {self.days}: plant actual {csum} m3 differs from native transpiration '
                               f'{tb.actual_m3} m3 beyond bound {fbound}; day rejected')
        self.t_gather += time.perf_counter() - t1
        # 4) publish: soil state, counters, cumulative scalars, then plant accumulation / weekly refresh
        b = res.budget
        self._state = res.state
        self.last_step = res
        self.days += 1
        c = self.cum
        c['added'] += b.added_m3
        c['surface_return'] += b.surface_return_m3
        c['drainage'] += b.drainage_m3
        c['evaporation'] += b.evaporation_m3
        c['evaporation_unmet'] += b.evaporation_unmet_m3
        c['transpiration'] += b.transpiration_m3
        c['transpiration_unmet'] += b.transpiration_unmet_m3
        c['requested'] += b.transpiration_requested_m3
        c['potential'] += dsum
        c['mapped'] += qsum * self.area
        c['plant_actual'] += csum
        c['bound'] += b.bound_m3
        c['flux_bound'] += fbound
        self.peak_required_bytes = max(self.peak_required_bytes, int(res.memory.required_bytes))
        # retained diagnostics of THIS day are part of the external reserve BEFORE any weekly work (pure update)
        veg.set_external_reserve_bytes(self._native_reserve_bytes())
        veg.commit_daily_uptake(cand)  # may latch a refused weekly rebuild (day already accepted)
        return res

    def _native_reserve_bytes(self):
        '''Pure (no availability query): native side fixed at construction + soil state + buffers + operator
        parameters + column reserve + caller reserve + retained diagnostics (same terms as at construction).'''
        diag = 0 if self.last_step is None else self.last_step.diagnostic_bytes()
        return (self.native_side_bytes + 8 * self.L * self.N + self.buffer_bytes + self.operator_param_bytes
                + self.column_reserve_bytes + diag + self.extra_reserve_bytes)

    def _check_device(self, stage):
        '''Binding contract check (no switching): on CUDA the current device must be the column's device,
        the vegetation's bound construction device and (once prepared) pass the operator device/preflight.'''
        if not self.gpu:
            return
        cur = int(sm.cp.cuda.Device().id)
        if self.column.device_id is not None and cur != int(self.column.device_id):
            raise ValueError(f'{stage}: current CUDA device {cur} differs from column device {self.column.device_id}')
        veg = getattr(self, 'vegetation', None)
        if veg is not None and int(veg.device_id) != cur:
            raise ValueError(f'{stage}: current CUDA device {cur} differs from the vegetation bound device {veg.device_id}')
        op = getattr(self, 'operator', None)
        if op is not None:
            op.check_device()

    def step_days(self, records, *, phase='interevent'):
        '''records: iterable of (added_water_m, potential_evaporation_m_s) pairs or objects with those
        attributes (optional dt_s must be 86400, optional phase must be interevent). Returns days accepted.'''
        check_phase(phase)
        n = 0
        for rec in records:
            if isinstance(rec, (tuple, list)):
                if len(rec) != 2:
                    raise ValueError('forcing pair must be (added_water_m, potential_evaporation_m_s)')
                added, pet, p = rec[0], rec[1], phase
            else:
                added = getattr(rec, 'added_water_m')
                pet = getattr(rec, 'potential_evaporation_m_s')
                dt = getattr(rec, 'dt_s', DAY_S)
                if float(dt) != DAY_S:
                    raise ValueError(f'coupled driver is daily-only: record dt_s {dt} != {DAY_S}')
                p = getattr(rec, 'phase', phase)
            self.step_day(added, pet, phase=p)
            n += 1
        return n

    def retry_weekly_update(self):
        '''Retry the latched weekly boundary (same accepted day/uptake, existing geometry kept until commit).
        The retry re-runs the full rebuild ledgers through the strict native-availability check; a refusal
        latches again BEFORE any graph allocation and never re-applies the day's uptake.'''
        self._check_device('retry_weekly_update')  # bound/current device + operator preflight before any work
        self.vegetation.set_external_reserve_bytes(self._native_reserve_bytes())
        self.vegetation.retry_weekly_update()

    # ---- reporting (streamed scalars; no trajectories)
    def summary(self):
        veg, c = self.vegetation, self.cum
        ps = veg.plant_summary()
        current = float(self._ss.storage_m3(self.column, self._state))
        expected = self.storage_initial_m3 + c['added'] - c['surface_return'] - c['drainage'] - c['evaporation'] - c['transpiration']
        water_res = current - expected
        flux_res = c['plant_actual'] - c['transpiration']
        t_s = float(self._state.t_s)
        t_expected = self.t0_s + self.days * DAY_S
        checks = dict(ps['checks'])
        checks.update({'water_budget': abs(water_res) <= c['bound'],
                       'plant_grid_flux': abs(flux_res) <= c['flux_bound'],
                       'calendar_time': t_s == t_expected,
                       'request_accounting': abs(c['requested'] - c['mapped']) <= c['bound'] + c['flux_bound']})
        return {'day': self.days, 'ok': all(checks.values()), 'checks': checks,
                'water': {'unit': 'm3', 'initial_storage': self.storage_initial_m3, 'current_storage': current,
                          'added': c['added'], 'surface_return': c['surface_return'], 'drainage': c['drainage'],
                          'evaporation': c['evaporation'], 'evaporation_unmet': c['evaporation_unmet'],
                          'transpiration': c['transpiration'], 'transpiration_unmet': c['transpiration_unmet'],
                          'residual': water_res, 'bound': c['bound']},
                'demand': {'unit': 'm3', 'potential': c['potential'], 'mapped_requested': c['mapped'],
                           'unmapped_inactive': c['potential'] - c['mapped'], 'native_actual': c['transpiration'],
                           'native_unmet': c['transpiration_unmet'], 'plant_actual': c['plant_actual'],
                           'flux_residual': flux_res, 'flux_bound': c['flux_bound'],
                           'note': 'inactive-cell root demand is never renormalized onto active cells'},
                'biomass': ps['biomass'], 'uptake': ps['uptake'], 'graph': ps['graph'],
                'root_profiles': ps['root_profiles'],
                'calendar': {'accepted_days': self.days, 'native_t_s': t_s, 'native_t0_s': self.t0_s,
                             'expected_t_s': t_expected, 'dt_s': DAY_S, 'vegetation': veg.counters()},
                'mask': {'n_cells': self.N, 'n_active': self.n_active, 'layers': self.L, 'cell_area_m2': self.area,
                         'backend': self.column.backend, 'device_id': self.column.device_id},
                'memory': self.memory_report(),
                'timing': {'native_wall_s': self.t_native, 'gather_wall_s': self.t_gather,
                           'vegetation_block_wall_s': veg.t_block, 'weekly_wall_s': veg.t_weekly}}

    def memory_report(self):
        veg, op = self.vegetation, self.operator
        where = 'device' if self.gpu else 'host'
        diag = 0 if self.last_step is None else self.last_step.diagnostic_bytes()
        last = None if self.last_step is None else self.last_step.memory.as_dict()
        return {'scope_label': 'ESTIMATED simultaneous-live peak (conservative): vegetation ledger (state, old+new '
                               'graphs, scratch, mirrors, native external reserve) and native combined admission '
                               '(soil state, forcing, hydro outputs, T parameters/inputs/outputs/rows/scratch, '
                               'column reserve, vegetation reserve, retained diagnostics); not an allocator peak',
                'native_side_bytes': self.native_side_bytes, 'operator_param_bytes': self.operator_param_bytes,
                'vegetation_ledger': veg.last_ledger, 'vegetation_live_bytes': veg.live_bytes(),
                'vegetation_graph_bytes': veg.graph_bytes, 'vegetation_rebuild_peak_bytes': veg.rebuild_peak_bytes,
                'coupling_buffers_bytes': self.buffer_bytes, 'retained_diagnostics_bytes': diag,
                'soil_state_bytes': int(self._state.water_m.nbytes),
                'operator_prepare_bytes': op.prepare_bytes, 'operator_memory': op.memory.as_dict(),
                'last_native_combined_memory': last, 'peak_required_bytes': self.peak_required_bytes,
                'peak_required_label': 'ADMITTED-ESTIMATE peak (not observed): max over construction (zero-edge '
                                       'and full edge-bound ledgers), every daily native combined admission and '
                                       'every weekly rebuild ledger incl. old+new CSR peaks and retries, each '
                                       'with the native soil/operator requirement and column reserve',
                'vegetation_rebuild_peak_label': 'vegetation-side admitted ledger peak only (subset of peak_required_bytes)',
                'column_reserve_bytes': self.column_reserve_bytes, 'extra_reserve_bytes': self.extra_reserve_bytes,
                'vegetation_external_reserve_bytes': self.vegetation.external_reserve_bytes,
                'max_bytes': self.case['max_bytes'], 'where': where}


__all__ = ['CoupledSimulation', 'CoupledPhaseError', 'check_phase', 'DAY_S', 'VERSION']
