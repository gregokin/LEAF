'''python -m leaf: LEAF Phase 0 command line (NumPy + stdlib only; never runs hydrology or growth).

  python -m leaf validate-species --species species.json
  python -m leaf init     --species S --plants in.csv [domain] [surface] --out out.csv
  python -m leaf generate --species S --config generation.json [surface] --out out.csv [--report r.json]
  python -m leaf inspect  --species S --plants state.csv [domain] [surface] [--cover-subsamples N]
  python -m leaf export-canopy --species S --plants state.csv [domain] [surface] --out view.json|view.npz

domain : --domain domain.json  (object with nx, ny, dx_m[, dy_m, origin_x_m, origin_y_m])
         or --nx NX --ny NY --dx-m DX [--dy-m DY --origin-x-m X0 --origin-y-m Y0]
         (not needed when the plant CSV carries a metadata line)
surface: --surface surface.npy (np.load allow_pickle=False, shape (ny, nx)) or --surface-constant Z
Outputs are written atomically; existing files are refused unless --overwrite. Errors exit with status 1.
'''
from __future__ import annotations

import argparse
import io
import json
import sys

import numpy as np

from leaf import canopy as _canopy
from leaf import initialization as _init
from leaf import state as _state
from leaf import traits as _traits


def _add_domain_args(p):
    g = p.add_argument_group('domain (optional when the CSV has a metadata line)')
    g.add_argument('--domain', metavar='JSON', help='domain JSON file {nx, ny, dx_m[, dy_m, origin_x_m, origin_y_m]}')
    g.add_argument('--nx', type=int)
    g.add_argument('--ny', type=int)
    g.add_argument('--dx-m', type=float, dest='dx_m')
    g.add_argument('--dy-m', type=float, dest='dy_m')
    g.add_argument('--origin-x-m', type=float, dest='origin_x_m', default=0.0)
    g.add_argument('--origin-y-m', type=float, dest='origin_y_m', default=0.0)


def _add_surface_args(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument('--surface', metavar='NPY', help='(ny, nx) surface elevation .npy (allow_pickle=False)')
    g.add_argument('--surface-constant', type=float, dest='surface_constant', metavar='Z', help='constant surface elevation m')


def _domain_from_args(args):
    if getattr(args, 'domain', None):
        with open(args.domain, 'r', encoding='utf-8') as fh:
            return _state.Domain.from_dict(_traits.parse_strict_json(fh.read(), what='domain file'))
    if args.nx is not None or args.ny is not None or args.dx_m is not None:
        if args.nx is None or args.ny is None or args.dx_m is None:
            raise _state.StateError('--nx, --ny and --dx-m must be given together')
        return _state.Domain(args.nx, args.ny, args.dx_m, args.dy_m, args.origin_x_m, args.origin_y_m)
    return None


def _surface_from_args(args, domain):
    if getattr(args, 'surface', None):
        return _state.load_surface(args.surface, domain)
    if getattr(args, 'surface_constant', None) is not None:
        if domain is None:
            raise _state.StateError('--surface-constant needs a domain')
        return _state.constant_surface(domain, args.surface_constant)
    return None


def _load_state(args, catalog):
    domain = _domain_from_args(args)
    state = _state.read_plant_csv(args.plants, catalog, domain=domain, time_days=getattr(args, 'time_days', None))
    surface = _surface_from_args(args, state.domain)
    if surface is not None:
        state.resolve_establishment(surface)
    return state, surface


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, _state.Domain):
        return o.to_dict()
    return o


def _dump(obj, stream=None):
    (stream or sys.stdout).write(json.dumps(_jsonable(obj), indent=1, allow_nan=False) + '\n')


# ------------------------------------------------------------------ commands
def cmd_validate_species(args):
    cat = _traits.load_catalog(args.species)
    _dump(cat.metadata())
    return 0


def cmd_init(args):
    cat = _traits.load_catalog(args.species)
    state, _ = _load_state(args, cat)
    initial = not state.establishment_resolved
    if initial:
        print(f'note: {int(state.establishment_unresolved.sum())} plant(s) keep the SURFACE token (no surface given); '
              'the output is an INITIAL input file, not a publishable snapshot', file=sys.stderr)
    path = _state.write_plant_csv(state, args.out, overwrite=args.overwrite, initial=initial,
                                  provenance={'command': 'leaf init', 'source_csv': args.plants, 'species_file': args.species})
    _dump({'written': path, 'summary': state.summary()})
    return 0


def cmd_generate(args):
    cat = _traits.load_catalog(args.species)
    with open(args.config, 'r', encoding='utf-8') as fh:
        config = _traits.parse_strict_json(fh.read(), what='generation config')
    cfg = _init.validate_generation_config(config, cat)
    surface = _surface_from_args(args, cfg['domain'])
    state, report = _init.generate_population(cat, config, surface=surface)
    initial = not state.establishment_resolved
    path = _state.write_plant_csv(state, args.out, overwrite=args.overwrite, initial=initial,
                                  provenance=dict(state.provenance, command='leaf generate', config_file=args.config))
    report['written'] = path
    if args.report:
        _state.atomic_write_text(args.report, json.dumps(_jsonable(report), indent=1, allow_nan=False) + '\n',
                                 overwrite=args.overwrite)
        report['report_written'] = args.report
    brief = {'written': path, 'n_plants': report['n_plants'], 'establishment': report['establishment'],
             'species': [{k: p[k] for k in ('species', 'target_cover', 'measured_union_cover', 'count', 'accepted')}
                         for p in report['species']],
             'total_union_cover_all_species': report['total_union_cover_all_species']}
    _dump(brief)
    return 0


def cmd_inspect(args):
    cat = _traits.load_catalog(args.species)
    state, surface = _load_state(args, cat)
    out = {'summary': state.summary(), 'catalog': cat.metadata()}
    geo = _canopy.derive_geometry(state, cat, current_surface=surface)
    standing = state.standing_mask()
    def st(a):
        a = np.asarray(a)[standing]
        return {'min': float(a.min()), 'mean': float(a.mean()), 'max': float(a.max())} if a.size else None
    out['geometry_standing'] = {'n': int(standing.sum()), 'diameter_m': st(geo['diameter_m']), 'height_m': st(geo['height_m']),
                                'root_radius_m': st(geo['root_radius_m']), 'height_definition': geo['height_definition']}
    if surface is not None:
        out['burial_exposure'] = {'burial_depth_m': st(geo['burial_depth_m']), 'exposure_depth_m': st(geo['exposure_depth_m']),
                                  'fully_buried': int(np.count_nonzero(geo['fully_buried'] & standing)),
                                  'note': geo['diagnostics_note']}
    if args.cover_subsamples:
        sel = standing & (geo['radius_m'] > 0.0)
        m = _canopy.measure_cover(state.domain, geo['x_m'][sel], geo['y_m'][sel], geo['radius_m'][sel],
                                  geo['species_index'][sel], len(state.species_names), subsamples=args.cover_subsamples)
        out['cover'] = {'method': m['method'], 'resolution_m': m['resolution_m'],
                        'per_species': {state.species_names[g['group']]: g for g in m['groups']},
                        'total_union_fraction': m['total_union_fraction'], 'error_basis': m['error_basis']}
    _dump(out)
    return 0


def cmd_export_canopy(args):
    cat = _traits.load_catalog(args.species)
    state, surface = _load_state(args, cat)
    statuses = tuple(s.strip() for s in args.statuses.split(',') if s.strip())
    view = _canopy.maple_vegetation_view(state, cat, current_surface=surface, statuses=statuses,
                                         height_mode=args.height_mode, unsupported_policy=args.unsupported)
    if args.out.endswith('.npz'):
        arrays = dict(view['fields'])
        for k, v in view['sidecar'].items():
            a = np.asarray(v)
            arrays['sidecar_' + k] = a.astype(str) if a.dtype == object else a
        buf = io.BytesIO()
        np.savez(buf, meta=json.dumps(_jsonable(view['meta'])), **arrays)
        _state.atomic_write_bytes(args.out, buf.getvalue(), overwrite=args.overwrite)
    else:
        _state.atomic_write_text(args.out, json.dumps(_jsonable(view), indent=1, allow_nan=False) + '\n', overwrite=args.overwrite)
    _dump({'written': args.out, 'n_objects': view['meta']['n_objects'], 'height_mode': args.height_mode,
           'n_unsupported': view['meta']['n_unsupported'], 'excluded_by_status': view['meta']['excluded_by_status']})
    return 0


# ------------------------------------------------------------------ parser
def build_parser():
    ap = argparse.ArgumentParser(prog='python -m leaf', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command')

    p = sub.add_parser('validate-species', help='validate a species catalog and print its metadata/hash')
    p.add_argument('--species', required=True)
    p.set_defaults(func=cmd_validate_species)

    p = sub.add_parser('init', help='read a plant list CSV, resolve SURFACE anchors, write a validated state CSV')
    p.add_argument('--species', required=True)
    p.add_argument('--plants', required=True, help='input plant CSV (handwritten lists may omit the metadata line)')
    p.add_argument('--time-days', type=float, dest='time_days', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--overwrite', action='store_true')
    _add_domain_args(p)
    _add_surface_args(p)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser('generate', help='generate a multispecies population from target covers')
    p.add_argument('--species', required=True)
    p.add_argument('--config', required=True, help='generation JSON (domain, seed, overlap, limits, species targets)')
    p.add_argument('--out', required=True)
    p.add_argument('--report', default=None, help='optional JSON generation report path')
    p.add_argument('--overwrite', action='store_true')
    _add_surface_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser('inspect', help='summarise a state CSV (geometry, burial diagnostics, optional cover)')
    p.add_argument('--species', required=True)
    p.add_argument('--plants', required=True)
    p.add_argument('--time-days', type=float, dest='time_days', default=None)
    p.add_argument('--cover-subsamples', type=int, dest='cover_subsamples', default=0, help='measure cover at dx/N (0 = skip)')
    _add_domain_args(p)
    _add_surface_args(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser('export-canopy', help='export the MAPLE-compatible canopy view (+ID sidecars) as JSON or NPZ')
    p.add_argument('--species', required=True)
    p.add_argument('--plants', required=True)
    p.add_argument('--time-days', type=float, dest='time_days', default=None)
    p.add_argument('--out', required=True, help='.json or .npz')
    p.add_argument('--height-mode', dest='height_mode', choices=_canopy.HEIGHT_MODES, default='intact')
    p.add_argument('--unsupported', choices=_canopy.UNSUPPORTED_POLICIES, default='error',
                   help='eroded (elevated-base) / fully buried plants: error (default), exclude or flag')
    p.add_argument('--statuses', default='live,dead_standing')
    p.add_argument('--overwrite', action='store_true')
    _add_domain_args(p)
    _add_surface_args(p)
    p.set_defaults(func=cmd_export_canopy)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, 'command', None):
        ap.print_help(sys.stderr)
        return 2
    try:
        return int(args.func(args))
    except (_traits.CatalogError, _state.StateError, _init.GenerationError, _canopy.ResourceBoundError, ValueError,
            OSError) as exc:
        print(f'error: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
