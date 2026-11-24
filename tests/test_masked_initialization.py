"""Independent midpoint oracle and real initialization on a physical footprint."""
import json
from pathlib import Path

import numpy as np
import pytest

from leaf import canopy, initialization, state, traits


@pytest.mark.parametrize("periodic", [(False, False), (True, True)])
def test_cover_matches_independent_midpoint_count(periodic):
    d = state.Domain(4, 3, 1.0, periodic=periodic)
    mask = np.array([[True, False, True, True], [True, True, True, False], [False, True, True, True]])
    x, y, radius = np.array([0.2, 2.5]), np.array([0.2, 2.3]), np.array([0.9, 0.8])
    sub = 16
    count = 0
    for row in range(3 * sub):
        for col in range(4 * sub):
            if not mask[row // sub, col // sub]:
                continue
            px, py = (col + 0.5) / sub, (row + 0.5) / sub
            for cx, cy, r in zip(x, y, radius):
                dx, dy = abs(px-cx), abs(py-cy)
                if periodic[0]:
                    dx = min(dx, 4-dx)
                if periodic[1]:
                    dy = min(dy, 3-dy)
                if dx*dx + dy*dy < r*r:
                    count += 1
                    break
    result = canopy.measure_cover(d, x, y, radius, subsamples=sub, active_mask=mask)
    assert result["groups"][0]["covered_cells"] == count
    assert result["total_union_fraction"] == count / (mask.sum() * sub * sub)
    assert result["groups"][0]["union_area_m2"] == count / (sub * sub)


def test_inactive_centres_and_wrong_mask_are_refused():
    d = state.Domain(4, 3, 1.0)
    mask = np.ones(d.shape, dtype=bool)
    mask[1, 1] = False
    with pytest.raises(state.StateError, match="inactive"):
        canopy.measure_cover(d, [1.5], [1.5], [0.5], active_mask=mask)
    for wrong in (mask.astype(float), mask.T, np.zeros(d.shape, dtype=bool)):
        with pytest.raises(state.StateError):
            canopy.measure_cover(d, [0.5], [0.5], [0.5], active_mask=wrong)


@pytest.mark.parametrize("overlap", ["allowed", "disallowed"])
def test_generation_uses_active_area_and_real_anchors(overlap):
    root = Path(__file__).resolve().parents[1]
    doc = json.loads((root / "examples/phase1/species.json").read_text())
    catalog = traits.SpeciesCatalog(doc)
    sid = catalog.species_ids[0]
    d = state.Domain(12, 10, 1.0, periodic=(True, True))
    mask = np.ones(d.shape, dtype=bool)
    mask[2:8, 3:9] = False
    z = np.where(mask, 1500 + np.arange(d.ncells).reshape(d.shape) / 100.0, np.nan)
    config = {"domain": d.to_dict(), "seed": 57, "overlap": overlap, "cover_tolerance": 0.02,
        "max_iterations": 24, "cover_subsamples": 8,
        "species": {sid: {"target_cover": 0.10, "diameter_mean_m": 0.5,
                          "diameter_std_m": 0.05, "initial_age_days": 30}}}
    plants, report = initialization.generate_population(catalog, config, surface=z, active_mask=mask)
    row, col = d.cell_indices(plants.x_m, plants.y_m)
    assert np.all(mask[row, col])
    assert np.array_equal(plants.establishment_ground_m, z[row, col])
    assert np.isfinite(plants.establishment_ground_m).all()
    assert report["active_mask"]["n_active"] == int(mask.sum())
    geometry = canopy.derive_geometry(plants, catalog)
    measured = canopy.measure_cover(d, plants.x_m, plants.y_m, 0.5*geometry["diameter_m"],
        subsamples=8, active_mask=mask)
    assert abs(measured["groups"][0]["union_fraction"] - 0.10) <= 0.02
    assert np.any((plants.x_m < 1) | (plants.y_m < 1))


def test_new_mask_storage_is_admitted_before_index_creation():
    d = state.Domain(20, 20, 1.0)
    mask = np.ones(d.shape, dtype=bool)
    with pytest.raises(canopy.ResourceBoundError):
        canopy.cover_admission(d, 1, max_cover_bytes=2*d.ncells, active_mask=mask)
