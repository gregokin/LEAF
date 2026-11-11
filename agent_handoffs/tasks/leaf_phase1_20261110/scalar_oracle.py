"""Independent exact-arithmetic oracle, deliberately scalar and outside production.

The cases are tiny physical-volume examples in m3. This verifier uses named
resource dictionaries rather than implementation CSR arrays or numerical kernels.
It does not evolve a soil inventory or stand in for SYRUP.
"""
from fractions import Fraction as F
import json


def reference(plants, available, accepted=None):
    requests = {r: F(0) for r in available}
    for plant in plants:
        demand = F(plant['demand'])
        roots = {r: F(w) for r, w in plant['roots'].items()}
        assert demand >= 0 and all(w >= 0 for w in roots.values())
        assert not roots or sum(roots.values()) == 1
        assert roots or demand == 0
        for resource, weight in roots.items():
            requests[resource] += demand * weight
    proposals = {r: min(F(available[r]), requests[r]) for r in available}
    if accepted is None:
        accepted = proposals
    else:
        accepted = {r: F(accepted[r]) for r in available}
    assert all(0 <= accepted[r] <= proposals[r] for r in available)
    actual = []
    for plant in plants:
        demand = F(plant['demand'])
        delivery = sum((demand * F(w) * accepted[r] / requests[r]
                        for r, w in plant['roots'].items() if requests[r] > 0), F(0))
        actual.append({'id': plant['id'], 'delivered': delivery,
                       'unmet': demand - delivery,
                       'satisfaction': delivery / demand if demand else F(0)})
    assert sum(row['delivered'] for row in actual) == sum(accepted.values())
    return {'combined_request': requests, 'proposed': proposals,
            'accepted': accepted, 'plants': actual}


CASES = {
    'user_two_layers': {
        'plants': [{'id': 7, 'demand': 100, 'roots': {'top': '1/2', 'bottom': '1/2'}}],
        'available': {'top': 25, 'bottom': 200},
    },
    'competition_different_roots': {
        'plants': [{'id': 7, 'demand': 100, 'roots': {'top': '1/2', 'bottom': '1/2'}},
                   {'id': 99, 'demand': 50, 'roots': {'top': 1}}],
        'available': {'top': 25, 'bottom': 200},
    },
    'native_accepts_less_than_proposal': {
        'plants': [{'id': 7, 'demand': 100, 'roots': {'top': '1/2', 'bottom': '1/2'}}],
        'available': {'top': 25, 'bottom': 200},
        'accepted': {'top': '25/2', 'bottom': 50},
    },
    'nonseparable_three_layers_two_cells': {
        'plants': [{'id': 5, 'demand': 10, 'roots': {'L0C0': '4/5', 'L1C0': '1/5'}},
                   {'id': 6, 'demand': 10, 'roots': {'L0C0': '1/10', 'L1C1': '1/5', 'L2C1': '7/10'}}],
        'available': {'L0C0': 1, 'L0C1': 50, 'L1C0': 3,
                      'L1C1': 2, 'L2C0': 100, 'L2C1': 4},
    },
    'unavailable_resource_does_not_redistribute': {
        'plants': [{'id': 5, 'demand': 10, 'roots': {'masked': '4/5', 'wet': '1/5'}}],
        'available': {'masked': 0, 'wet': 100},
    },
    'zero_demand_and_empty_roots': {
        'plants': [{'id': 5, 'demand': 0, 'roots': {}},
                   {'id': 6, 'demand': 0, 'roots': {'wet': 1}}],
        'available': {'wet': 100},
    },
    'empty_population': {'plants': [], 'available': {'wet': 100}},
}


def main():
    results = {name: reference(**case) for name, case in CASES.items()}
    assert results['user_two_layers']['plants'][0]['delivered'] == 75
    assert results['user_two_layers']['plants'][0]['satisfaction'] == F(3, 4)
    competition = results['competition_different_roots']['plants']
    assert [p['delivered'] for p in competition] == [F(125, 2), F(25, 2)]
    assert [p['satisfaction'] for p in competition] == [F(5, 8), F(1, 4)]
    assert results['native_accepts_less_than_proposal']['plants'][0]['delivered'] == F(125, 2)
    assert results['unavailable_resource_does_not_redistribute']['plants'][0]['delivered'] == 2
    swapped = dict(CASES['competition_different_roots'])
    swapped['plants'] = list(reversed(swapped['plants']))
    reverse = reference(**swapped)
    assert reverse['combined_request'] == results['competition_different_roots']['combined_request']
    assert list(reversed(reverse['plants'])) == competition
    print(json.dumps({'cases': results, 'case_count': len(results),
                      'exact_conservation': True, 'reordering_checked': True}, default=str, indent=2))


if __name__ == '__main__':
    main()
