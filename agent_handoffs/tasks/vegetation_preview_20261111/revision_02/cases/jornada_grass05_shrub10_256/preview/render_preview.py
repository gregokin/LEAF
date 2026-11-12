#!/usr/bin/env python3
"""Initial-vegetation plan-view preview for the jornada_grass05_shrub10_256 LEAF trial case.

INITIALIZATION ONLY: builds the initial surface (actual MAPLE random_field generator + reference level) and the
initial population (actual leaf.initialization.generate_population), derives circle geometry with
leaf.canopy.derive_geometry and draws it. No growth, hydrology, wind/water transport, ecology or evolution is run.
Plants are never sampled, moved or resized here; every plotted diameter/centre comes from the LEAF objects.

Usage (absolute paths required; python from /home/okin/MAPLE/.venv-cupy):
  python render_preview.py --repo /abs/leaf/repo --case /abs/leaf/repo/cases/jornada_grass05_shrub10_256 \
      --output /abs/existing/or/new/output_dir
Files are created exclusively (never overwritten); all destinations are checked before any work.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform
import sys
import tempfile
import time

MAPLE_RANDOM_FIELD = '/home/okin/MAPLE/src/maple/case_tools/generators/random_field.py'
OUTPUTS = ('vegetation_initial.png', 'initial_surface.npy', 'initial_plants.csv',
           'initialization_report.json', 'preview_provenance.json')
ZOOM = (28.0, 36.0, 28.0, 36.0)   # x0, x1, y0, y1 (m)
STYLE = {'BOER': ('BOER grass', '#3fd34a', '#2f9f38'), 'PROGLA': ('PROGLA shrub', '#0e6b66', '#073b3a')}
FIG_W_IN, FIG_H_IN, DPI = 15.0, 9.0, 200
CAPTION = 'Initial placement only; no growth or transport simulated'


def sha256_bytes(b):
    return 'sha256:' + hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return 'sha256:' + h.hexdigest()


def abs_path(text):
    if not os.path.isabs(text):
        raise argparse.ArgumentTypeError(f'{text!r} must be an absolute path')
    return os.path.normpath(text)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--repo', required=True, type=abs_path, help='absolute path of the LEAF repository (contains leaf/)')
    p.add_argument('--case', required=True, type=abs_path, help='absolute path of the case directory')
    p.add_argument('--output', required=True, type=abs_path, help='absolute output directory (files created exclusively)')
    return p.parse_args(argv)


def under(path, parent):
    path, parent = os.path.realpath(path), os.path.realpath(parent)
    return path == parent or path.startswith(parent + os.sep)


def preflight(args):
    for label, d in (('--repo', args.repo), ('--case', args.case)):
        if not os.path.isdir(d):
            raise SystemExit(f'{label} {d!r} is not an existing directory')
    if not os.path.isdir(os.path.join(args.repo, 'leaf')):
        raise SystemExit(f'{args.repo!r} has no leaf/ package')
    for name in ('generation.json', 'species.json', 'landscape.toml'):
        if not os.path.isfile(os.path.join(args.case, name)):
            raise SystemExit(f'missing case file {os.path.join(args.case, name)!r}')
    if not os.path.isfile(MAPLE_RANDOM_FIELD):
        raise SystemExit(f'missing MAPLE generator {MAPLE_RANDOM_FIELD!r}')
    if under(args.output, os.path.join(args.case, 'inputs')) or under(args.output, os.path.join(args.repo, 'leaf')):
        raise SystemExit('output directory must not be inside case inputs/ or the leaf package')
    if os.path.lexists(args.output) and not os.path.isdir(args.output):
        raise SystemExit(f'--output {args.output!r} exists and is not a directory')
    dests = {n: os.path.join(args.output, n) for n in OUTPUTS}
    clash = [p for p in dests.values() if os.path.lexists(p)]
    if clash:
        raise SystemExit(f'refusing to overwrite existing outputs: {clash}')
    return dests


def load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:   # Python < 3.11
        import tomli as tomllib
    with open(path, 'rb') as fh:
        return tomllib.load(fh)


def load_maple_generator():
    spec = importlib.util.spec_from_file_location('maple_random_field_original', MAPLE_RANDOM_FIELD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_spectral_gaussian_field


def jsonable(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not JSON serialisable: {type(o).__name__}')


def json_text(obj):
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=jsonable) + '\n'


def exclusive_write(path, data):
    mode = 'xb' if isinstance(data, bytes) else 'x'
    with open(path, mode) as fh:
        fh.write(data)


def circles_intersecting(x, y, r, box):
    """True where the circle (centre x,y radius r) intersects the closed rectangle box=(x0,x1,y0,y1)."""
    import numpy as np
    x0, x1, y0, y1 = box
    nx, ny = np.clip(x, x0, x1), np.clip(y, y0, y1)
    return (nx - x) ** 2 + (ny - y) ** 2 <= r * r


def make_collection(x, y, d, face, edge, lw, ax):
    import numpy as np
    from matplotlib.collections import EllipseCollection
    kw = dict(units='xy', angles=np.zeros(len(d)), facecolors=face, edgecolors=edge, linewidths=lw)
    offsets = np.column_stack([x, y])
    try:
        ec = EllipseCollection(d, d, offsets=offsets, offset_transform=ax.transData, **kw)
    except TypeError:   # matplotlib < 3.6
        ec = EllipseCollection(d, d, offsets=offsets, transOffset=ax.transData, **kw)
    return ec


def render(surface, geo, report, ref_level, domain):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch, Rectangle, ConnectionPatch
    from matplotlib.ticker import MultipleLocator

    x, y, d = geo['x_m'], geo['y_m'], geo['diameter_m']
    sp = np.asarray(geo['species']).astype(str)
    W, H = domain.width_m, domain.height_m
    ext = (domain.x_min_m, domain.x_max_m, domain.y_min_m, domain.y_max_m)
    cmap = LinearSegmentedColormap.from_list('sand', ['#d9cdb0', '#e8dfc8', '#c9c4b8'])
    rep = {s['species']: s for s in report['species']}
    zbox = ZOOM
    zoom_counts = {}

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor='white')
    ax = fig.add_axes([0.055, 0.115, 0.49, 0.77])
    az = fig.add_axes([0.625, 0.44, 0.33, 0.445])
    at = fig.add_axes([0.585, 0.075, 0.395, 0.28])
    at.axis('off')

    for a in (ax, az):
        a.imshow(surface, extent=ext, origin='lower', cmap=cmap, vmin=float(surface.min()), vmax=float(surface.max()),
                 interpolation='nearest', zorder=0, aspect='equal')

    order = sorted(set(sp), key=lambda s: (s != 'BOER', s))   # grass first, shrubs on top
    for sid in order:
        label, face, edge = STYLE.get(sid, (sid, '#888888', '#444444'))
        m = sp == sid
        ax.add_collection(make_collection(x[m], y[m], d[m], face, edge, 0.0 if sid == 'BOER' else 0.35, ax))
        zm = m & circles_intersecting(x, y, 0.5 * d, zbox)
        zoom_counts[sid] = int(zm.sum())
        az.add_collection(make_collection(x[zm], y[zm], d[zm], face, edge, 0.4, az))

    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    az.set_xlim(zbox[0], zbox[1]); az.set_ylim(zbox[2], zbox[3])
    for a in (ax, az):
        a.set_aspect('equal', adjustable='box')
        a.set_xlabel('x (m)'); a.set_ylabel('y (m)')
        a.tick_params(labelsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(8)); ax.yaxis.set_major_locator(MultipleLocator(8))
    ax.grid(True, which='major', color='#6b6b6b', alpha=0.25, linewidth=0.5)
    az.xaxis.set_major_locator(MultipleLocator(1)); az.yaxis.set_major_locator(MultipleLocator(1))
    az.xaxis.set_minor_locator(MultipleLocator(0.25)); az.yaxis.set_minor_locator(MultipleLocator(0.25))
    az.grid(True, which='major', color='#6b6b6b', alpha=0.30, linewidth=0.5)
    az.grid(True, which='minor', color='#6b6b6b', alpha=0.15, linewidth=0.3)
    ax.set_title(f'Full domain ({W:g} m x {H:g} m): all {len(d)} plants', fontsize=11)
    az.set_title('Close-up: 8 m x 8 m, x/y = 28 to 36 m (same plants/geometry)', fontsize=10)
    for spine in az.spines.values():
        spine.set_edgecolor('#b00020'); spine.set_linewidth(1.5)

    ax.add_patch(Rectangle((zbox[0], zbox[2]), zbox[1] - zbox[0], zbox[3] - zbox[2], fill=False, edgecolor='#b00020',
                           linewidth=1.5, zorder=5))
    for pa, pb in (((zbox[1], zbox[3]), (0.0, 1.0)), ((zbox[1], zbox[2]), (0.0, 0.0))):
        fig.add_artist(ConnectionPatch(xyA=pa, coordsA='data', axesA=ax, xyB=pb, coordsB='axes fraction', axesB=az,
                                       color='#b00020', linewidth=0.8, alpha=0.8, clip_on=False))

    fig.suptitle('LEAF trial case: initial vegetation', fontsize=17, fontweight='bold', y=0.972)
    fig.text(0.5, 0.925, f'{domain.nx} x {domain.ny} cells, dx = {domain.dx_m:g} m, dy = {domain.dy_m:g} m; '
             f'domain {W:g} m x {H:g} m ({domain.area_m2:g} m$^2$); LEAF seed {report["seed"]}; simulation time = 0 d',
             ha='center', fontsize=10.5, color='#333333')

    handles = []
    lines = []
    for sid in order:
        label, face, edge = STYLE.get(sid, (sid, '#888888', '#444444'))
        r = rep[sid]
        ds, dr = r['diameter_sample_m'], r['diameter_requested']
        handles.append(Patch(facecolor=face, edgecolor=edge, label=(
            f'{label}: N = {r["count"]:,}; cover {100.0 * r["measured_union_cover"]:.2f}% '
            f'(target {100.0 * r["target_cover"]:.2f}%)')))
        lines.append(f'{label}: sampled diameter {ds["mean"]:.4f} +/- {ds["std"]:.4f} m (mean +/- SD)\n'
                     f'      range {ds["min"]:.3f}-{ds["max"]:.3f} m; configured target {dr["mean_m"]:g} +/- {dr["std_m"]:g} m')
    at.legend(handles=handles, loc='upper left', fontsize=9.5, frameon=True, title='Species (accepted at initialization)',
                    title_fontsize=10)
    at.text(0.0, 0.62, '\n'.join(lines) +
            '\nCircle diameters are exact canopy diameters (data units, m).'
            f'\nClose-up circles: {", ".join(f"{k} {v:,}" for k, v in zoom_counts.items())}.'
            f'\nTerrain: {float(np.ptp(surface)):.4f} m peak-to-peak relief'
            f' (reference {ref_level:g} m);'
            '\nsubtle shading only, no mounds implied.',
            transform=at.transAxes, fontsize=8.6, va='top', ha='left', color='#222222')
    fig.text(0.5, 0.035, CAPTION, ha='center', fontsize=13, fontweight='bold', color='#7a1f1f')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor='white')
    plt.close(fig)
    return buf.getvalue(), zoom_counts


def main(argv=None):
    t0 = time.time()
    args = parse_args(argv)
    dests = preflight(args)
    os.makedirs(args.output, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'leaf_preview_mplconfig'))
    sys.dont_write_bytecode = True
    sys.path.insert(0, args.repo)

    import numpy as np
    import matplotlib
    from leaf import canopy, initialization, state, traits
    import leaf
    leaf_dir = os.path.realpath(os.path.dirname(leaf.__file__))
    if not under(leaf_dir, args.repo):
        raise SystemExit(f'imported leaf package {leaf_dir!r} is not under --repo {args.repo!r}')

    gen_path = os.path.join(args.case, 'generation.json')
    spc_path = os.path.join(args.case, 'species.json')
    lsc_path = os.path.join(args.case, 'landscape.toml')
    with open(gen_path, 'r', encoding='utf-8') as fh:
        gen_cfg = json.load(fh)
    land = load_toml(lsc_path)
    surf_cfg = land['surface']
    call_args = dict(surf_cfg['call_arguments'])
    ref_level = float(surf_cfg['reference_level_m'])

    catalog = traits.load_catalog(spc_path)
    field_fn = load_maple_generator()
    perturb = np.asarray(field_fn(**call_args), dtype=np.float64)
    surface = perturb + ref_level
    domain = state.Domain.from_dict(gen_cfg['domain'])
    surface = domain.check_field(surface, 'initial surface')
    relief = float(np.ptp(perturb))
    if abs(relief - float(call_args['relief_m'])) > 1e-9:
        print(f'WARNING: perturbation peak-to-peak {relief!r} differs from requested {call_args["relief_m"]!r}')

    plants, report = initialization.generate_population(catalog, gen_cfg, surface=surface)
    geo = canopy.derive_geometry(plants, catalog, current_surface=surface)

    # ---- consistency diagnostics (measured, not assumed)
    anchors = plants.establishment_ground_m
    sampled = domain.sample(surface, plants.x_m, plants.y_m)
    anchor_max_diff = float(np.max(np.abs(anchors - sampled))) if plants.n else 0.0
    sp_geo = np.asarray(geo['species']).astype(str)
    geo_stats, diam_diff = {}, {}
    for s in report['species']:
        m = sp_geo == s['species']
        dd = geo['diameter_m'][m]
        geo_stats[s['species']] = {'n': int(m.sum()), 'mean_m': float(dd.mean()) if dd.size else None,
                                   'std_m': float(dd.std()) if dd.size else None}
        diam_diff[s['species']] = (abs(geo_stats[s['species']]['mean_m'] - s['diameter_sample_m']['mean'])
                                   if dd.size else None)
    checks = {
        'time_days': float(plants.time_days), 'time_days_is_zero': plants.time_days == 0.0,
        'anchors_resolved': bool(plants.establishment_resolved),
        'anchor_vs_initial_surface_max_abs_diff_m': anchor_max_diff,
        'surface_delta_all_zero': bool(np.all(geo['surface_delta_m'] == 0.0)),
        'geometry_diameter_stats_m': geo_stats, 'geometry_vs_report_mean_diameter_abs_diff_m': diam_diff,
        'all_species_accepted': all(s['accepted'] for s in report['species']),
        'status_counts': plants.counts(),
        'age_days_by_species': {s: sorted({float(a) for a in plants.age_days[sp_geo == s]}) for s in sorted(set(sp_geo))},
        'perturbation_peak_to_peak_m': relief,
        'surface_min_m': float(surface.min()), 'surface_max_m': float(surface.max()),
    }

    png, zoom_counts = render(surface, geo, report, ref_level, domain)

    # ---- exclusive writes (all destinations were preflighted)
    npy_buf = io.BytesIO()
    np.save(npy_buf, surface, allow_pickle=False)
    npy_bytes = npy_buf.getvalue()
    report_text = json_text(report)
    exclusive_write(dests['vegetation_initial.png'], png)
    exclusive_write(dests['initial_surface.npy'], npy_bytes)
    state.write_plant_csv(plants, dests['initial_plants.csv'], initial=True)
    exclusive_write(dests['initialization_report.json'], report_text)

    src_files = {
        'generation.json': gen_path, 'species.json': spc_path, 'landscape.toml': lsc_path,
        'render_preview.py': os.path.abspath(__file__), 'maple_random_field.py': MAPLE_RANDOM_FIELD,
        'leaf/initialization.py': initialization.__file__, 'leaf/canopy.py': canopy.__file__,
        'leaf/state.py': state.__file__, 'leaf/traits.py': traits.__file__, 'leaf/__init__.py': leaf.__file__,
    }
    prov = {
        'schema': 'leaf.vegetation_preview_provenance', 'schema_version': 1,
        'scope': 'INITIALIZATION_ONLY',
        'statement': 'Initial placement/surface only. No ecological or transport simulation (no growth, hydrology, '
                     'wind/water events, mortality, recruitment, evolution) was run; simulation time is 0 days.',
        'preview_population_note': 'separate preview population; case inputs/plants.csv and inputs/surface.npy were not created',
        'repo': args.repo, 'case': args.case, 'output_dir': args.output,
        'sources': {k: {'path': os.path.realpath(v), 'sha256': sha256_file(v)} for k, v in src_files.items()},
        'catalog_hash': catalog.hash, 'leaf_version': traits.LEAF_VERSION,
        'generation_config': report['generation_config'],
        'surface': {'generator': MAPLE_RANDOM_FIELD + '::generate_spectral_gaussian_field', 'call_arguments': call_args,
                    'reference_level_m': ref_level, 'shape': list(surface.shape), 'dtype': str(surface.dtype)},
        'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                        'matplotlib': matplotlib.__version__, 'matplotlib_backend': 'Agg',
                        'MPLCONFIGDIR': os.environ.get('MPLCONFIGDIR')},
        'image': {'file': 'vegetation_initial.png', 'figsize_in': [FIG_W_IN, FIG_H_IN], 'dpi': DPI,
                  'pixels': [int(FIG_W_IN * DPI), int(FIG_H_IN * DPI)], 'zoom_window_m': list(ZOOM),
                  'circle_rendering': 'matplotlib EllipseCollection, units=xy, widths=heights=derive_geometry diameter_m '
                                      '(exact data units), offsets=derive_geometry centres; no fixed-size markers',
                  'zoom_circles_intersecting_window': zoom_counts, 'plants_plotted_main': int(plants.n),
                  'derive_geometry_call': 'derive_geometry(plants, catalog, current_surface=surface)'},
        'checks_measured_at_run': checks,
        'outputs': {'vegetation_initial.png': sha256_bytes(png), 'initial_surface.npy': sha256_bytes(npy_bytes),
                    'initial_plants.csv': sha256_file(dests['initial_plants.csv']),
                    'initialization_report.json': sha256_bytes(report_text.encode('utf-8'))},
        'elapsed_s_before_provenance': round(time.time() - t0, 3),
    }
    exclusive_write(dests['preview_provenance.json'], json_text(prov))

    cov = ', '.join(f'{s["species"]}: n={s["count"]} cover={s["measured_union_cover"]:.4f} (target {s["target_cover"]:g})'
                    for s in report['species'])
    print(f'done: {plants.n} plants; {cov}; grid {domain.nx}x{domain.ny} @ {domain.dx_m:g} m; time_days={plants.time_days:g}; '
          f'anchor max|diff|={anchor_max_diff:.3g} m; zoom circles {zoom_counts}; output {args.output}; '
          f'elapsed {time.time() - t0:.1f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
