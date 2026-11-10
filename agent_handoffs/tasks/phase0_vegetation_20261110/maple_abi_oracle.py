"""Validate the LEAF object view with the actual read-only MAPLE container ABI."""
import json
import sys
import numpy as np
from generation_oracles import make_catalog
from leaf.state import Domain, PlantState, constant_surface
from leaf.canopy import maple_vegetation_view


def main():
    sys.path.insert(0, '/home/okin/MAPLE/src')
    from maple.core.types.vegetation import VegetationObjectSet, validate_vegetation_object_set
    catalog = make_catalog()
    domain = Domain(4, 4, 0.25, origin_x_m=12.0, origin_y_m=-4.0)
    surface = constant_surface(domain, 100.0)
    records = [dict(plant_id=17, species='BOER', x_m=12.125, y_m=-3.875,
                    establishment_ground_m='SURFACE', agb_dry_kg=0.01,
                    seasonal_agb_dry_kg=0.0, age_days=100.0, status='live')]
    population = PlantState.from_records(catalog, domain, records, surface=surface)
    view = maple_vegetation_view(population, catalog, surface)
    objects = VegetationObjectSet(**view['fields'])
    validate_vegetation_object_set(objects)
    assert view['meta']['native_maple_valid']
    np.testing.assert_array_equal(objects.center_x_m, [0.125])
    np.testing.assert_array_equal(objects.center_y_m, [0.125])
    np.testing.assert_array_equal(view['sidecar']['plant_id'], [17])
    for field in view['fields'].values():
        assert field.dtype == np.float64 and field.flags.c_contiguous
    print(json.dumps({'pass': True, 'native_validator': True, 'plants': 1,
                      'nonzero_origin': True, 'id_sidecar_preserved': True,
                      'wind_or_raster_run': False, 'upstream_writes': False}, indent=2))


if __name__ == '__main__':
    main()
