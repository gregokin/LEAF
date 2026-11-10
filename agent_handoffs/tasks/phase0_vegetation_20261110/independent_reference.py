"""Independent geometry/statistics oracles, written before candidate review.

No LEAF implementation imports. Used only by Codex's bounded verification.
"""
import math
import numpy as np
from scipy.spatial import cKDTree


def poisson_count_prior(area_m2, cover_fraction, diameter_mean_m, diameter_sd_m):
    mean_crown_area = math.pi * (diameter_mean_m**2 + diameter_sd_m**2) / 4
    return -area_m2 * math.log1p(-cover_fraction) / mean_crown_area


def union_cover_random_points(x, y, radii, bounds, *, seed=18671, samples=250_000):
    """Independent uniform-point area estimate; no raster or LEAF geometry code.

    The tree indexes probes, not plants; only one union boolean per probe is
    retained. Query each circle against that independent tree. Edge clipping
    is implicit because all probes lie inside the rectangular domain.
    """
    x0, y0, width, height = bounds
    rng = np.random.Generator(np.random.PCG64(seed))
    probes = rng.random((samples, 2))
    probes[:, 0] = x0 + probes[:, 0] * width
    probes[:, 1] = y0 + probes[:, 1] * height
    tree = cKDTree(probes)
    hits = np.zeros(samples, dtype=bool)
    for cx, cy, radius in zip(x, y, radii, strict=True):
        hits[tree.query_ball_point((cx, cy), radius)] = True
    estimate = float(hits.mean())
    se = math.sqrt(estimate * (1 - estimate) / samples)
    return {'fraction': estimate, 'standard_error': se, 'samples': samples,
            'seed': seed, 'method': 'independent uniform probes and SciPy cKDTree'}


def assert_nonoverlap_small(x, y, radii):
    """Exact O(P^2) oracle restricted to <=5000 plants; production must not use it."""
    x, y, radii = map(np.asarray, (x, y, radii))
    if len(x) > 5000:
        raise ValueError('This deliberately simple oracle is bounded to small populations')
    clearance = math.inf
    for i in range(len(x)):
        distances = np.hypot(x[i + 1:] - x[i], y[i + 1:] - y[i])
        delta = distances - (radii[i + 1:] + radii[i])
        if len(delta):
            clearance = min(clearance, float(delta.min()))
        if np.any(delta < -1e-12):
            raise AssertionError(f'Canopies overlap at plant row {i}')
    return clearance
