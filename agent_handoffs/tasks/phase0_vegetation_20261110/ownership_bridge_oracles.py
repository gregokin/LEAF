"""Independent callback ownership and legacy-handoff guards."""
import copy
import json
import math
from types import SimpleNamespace
import numpy as np
from review_probes_01 import catalog, state
from leaf.traits import SpeciesCatalog
from leaf.state import Domain, PlantState
from leaf.events import SurfaceEventDiagnostics, apply_event_diagnostics
from leaf.runtime import legacy_inputs_from_state, capture_snapshot, read_driver_arrays


def driver(ids=(17,), *, profile=None, biomass=None):
    ids = np.asarray(ids)
    return SimpleNamespace(host_inputs={'ids': ids},
        B=np.full(len(ids), 0.02) if biomass is None else np.asarray(biomass),
        host_profile_ids=np.zeros(len(ids), np.int32) if profile is None else np.asarray(profile),
        profile_names=['BOER'], stepwat=None, day=0)


def legacy_catalog(*, exponent=0.5):
    doc = catalog().document
    doc['species']['BOER']['allometry']['diameter'] = {'a': 2/math.sqrt(math.pi), 'b': exponent}
    return SpeciesCatalog(doc)


def raises_value_error(fn):
    try:
        fn()
        return False
    except (ValueError, TypeError, AttributeError, OverflowError):
        return True


def main():
    out = {}
    c = catalog(); s = state(c); s.advance_time(1.0)
    event = SurfaceEventDiagnostics([17], 0.0, 1.0, 'maple_wind', erosion_depth_m=[0.1])
    domain_before = copy.deepcopy(s.domain.to_dict()); B_before = s.agb_dry_kg.copy()
    def bad_handler(view, ev, indices):
        view.domain.nx = 999
        raise RuntimeError('deliberate failed response')
    rejected = raises_value_error(lambda: apply_event_diagnostics(s, event, handler=bad_handler))
    out['failed_handler_domain_rollback'] = {'pass': rejected and s.domain.to_dict() == domain_before
                                                  and np.array_equal(s.agb_dry_kg, B_before)}

    # A caller-owned diagnostic vector must not remain a mutable alias.
    raw = np.array([0.1])
    ev = SurfaceEventDiagnostics([17], 0.0, 1.0, 'maple_wind', erosion_depth_m=raw)
    writable = bool(raw.flags.writeable)
    if writable:
        raw[0] = -5.0
    out['diagnostic_owns_input'] = {'pass': writable and np.array_equal(ev.erosion_depth_m, [0.1])}

    c = catalog(); s = state(c)
    # Duplicate the one plant using the canonical constructor, with a distinct ID/location.
    recs = s.to_records(); second = dict(recs[0], plant_id=18, x_m=1.5); recs.append(second)
    s = PlantState.from_records(c, s.domain, recs); s.advance_time(1.0)
    ev = SurfaceEventDiagnostics([17], 0.0, 1.0, 'maple_wind', erosion_depth_m=[0.1])
    before = s.agb_dry_kg.copy()
    def redirect(view, event, indices):
        event.plant_id[0] = 18
        return {'agb_dry_kg': [0.005], 'seasonal_agb_dry_kg': [0.0]}
    try:
        apply_event_diagnostics(s, ev, handler=redirect)
        okay = (ev.plant_id.tolist() == [17]
                and s.agb_dry_kg[s.indices_of([18])[0]] == before[1]
                and s.agb_dry_kg[s.indices_of([17])[0]] == 0.005)
    except (ValueError, TypeError, AttributeError):
        okay = np.array_equal(s.agb_dry_kg, before) and ev.plant_id.tolist() == [17]
    out['handler_cannot_redirect_identity'] = {'pass': bool(okay)}

    c = legacy_catalog(); s = state(c); s.update_plants([17], seasonal_agb_dry_kg=[0.0])
    valid = legacy_inputs_from_state(s, c)
    out['matching_legacy_law'] = {'pass': valid['inputs']['ids'].tolist() == [17]}
    s.update_plants([17], seasonal_agb_dry_kg=[0.002])
    out['refuse_lost_seasonal_input'] = {'pass': raises_value_error(lambda: legacy_inputs_from_state(s, c))}
    out['refuse_lost_seasonal_capture'] = {'pass': raises_value_error(lambda: capture_snapshot(s, driver()))}

    c = legacy_catalog(exponent=0.25); s = state(c)
    s.update_plants([17], agb_dry_kg=[1.0], seasonal_agb_dry_kg=[0.0])
    out['refuse_pointwise_only_allometry_match'] = {
        'pass': raises_value_error(lambda: legacy_inputs_from_state(s, c))}
    c = catalog(); s = state(c); s.update_plants([17], seasonal_agb_dry_kg=[0.0])
    out['refuse_nan_tolerance'] = {'pass': raises_value_error(lambda: legacy_inputs_from_state(s, c, rtol=float('nan')))}
    out['refuse_unknown_profile'] = {'pass': raises_value_error(lambda: legacy_inputs_from_state(
        s, c, profile_names=['missing', 'BOER'], biomass_policy='accept_driver_recompute'))}
    out['refuse_unsigned_driver_id_overflow'] = {'pass': raises_value_error(lambda: read_driver_arrays(
        driver(ids=np.array([2**64-1], np.uint64))))}
    out['refuse_fractional_driver_profile'] = {'pass': raises_value_error(lambda: read_driver_arrays(
        driver(profile=np.array([0.5]))))}

    doc = legacy_catalog().document
    doc['species']['SHRB'] = copy.deepcopy(doc['species']['BOER'])
    first = SpeciesCatalog(doc); pop = state(first); pop.update_plants([17], seasonal_agb_dry_kg=[0.0])
    reverse = copy.deepcopy(doc); reverse['species'] = dict(reversed(list(reverse['species'].items())))
    second = SpeciesCatalog(reverse)
    try:
        inputs = legacy_inputs_from_state(pop, second)
        names = inputs['profile_names']; indices = inputs['inputs']['root_profile_id']
        okay = [names[int(i)] for i in indices] == pop.species_of().tolist()
    except ValueError:
        okay = True  # explicit rejection is safe; silently reinterpreting codes is not
    out['catalog_order_cannot_change_species'] = {'pass': bool(okay)}
    print(json.dumps(out, indent=2))
    return 0 if all(v['pass'] for v in out.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
