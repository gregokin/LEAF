'''python -m leaf: LEAF Phase 0 command line (NumPy + stdlib only; never runs hydrology or growth).

  python -m leaf validate-species --species species.json
  python -m leaf init     --species S --plants in.csv [domain] [surface] --out out.csv
  python -m leaf generate --species S --config generation.json [surface] --out out.csv [--report r.json]
  python -m leaf inspect  --species S --plants state.csv [domain] [surface] [--cover-subsamples N]
  python -m leaf export-canopy --species S --plants state.csv [domain] SURFACE --out view.json|view.npz

domain : --domain domain.json  (object with nx, ny, dx_m[, dy_m, origin_x_m, origin_y_m])
         or --nx NX --ny NY --dx-m DX [--dy-m DY --origin-x-m X0 --origin-y-m Y0]
         (not needed when a companion <stem>.metadata.json sits beside the plant CSV)
surface: --surface surface.npy (np.load allow_pickle=False, shape (ny, nx)) or --surface-constant Z
         init/generate: the INITIAL surface resolving SURFACE anchors (optional: without it the output is an
         initial file keeping the token). inspect/export-canopy: the CURRENT surface; export-canopy REQUIRES
         it because safe MAPLE publication validates every canopy against its establishment anchor.
Plant files: plant CSVs are PLAIN nine-column CSV (no embedded metadata, comments or JSON). init/generate write the
         CSV and a companion JSON "<stem>.metadata.json" beside it (out.csv -> out.metadata.json: time, domain, catalog
         identity, row count, units, initial / SURFACE declaration, provenance and the CSV SHA-256). Every command that
         reads a plant CSV loads and verifies the companion automatically; a CSV that does not match it is refused.
         A handwritten plain list has no companion: give the domain (--domain or --nx/--ny/--dx-m).
Outputs are written atomically (both plant files are validated and serialised first, both destinations are
preflighted, the companion is published before the CSV and removed again if the CSV cannot be written); existing
files, including a companion or a --report that would collide with either plant file, are refused unless
--overwrite. Errors exit with status 1.
'''
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np

from leaf import canopy as _canopy
from leaf import initialization as _init
from leaf import state as _state
from leaf import traits as _traits


def _add_domain_args(p):
    g = p.add_argument_group('domain (optional when a <stem>.metadata.json companion sits beside the CSV)')
    g.add_argument('--domain', metavar='JSON', help='domain JSON file {nx, ny, dx_m[, dy_m, origin_x_m, origin_y_m]}')
    g.add_argument('--nx', type=int)
    g.add_argument('--ny', type=int)
    g.add_argument('--dx-m', type=float, dest='dx_m')
    g.add_argument('--dy-m', type=float, dest='dy_m')
    g.add_argument('--origin-x-m', type=float, dest='origin_x_m', default=0.0)
    g.add_argument('--origin-y-m', type=float, dest='origin_y_m', default=0.0)


def _add_surface_args(p, what):
    g = p.add_mutually_exclusive_group()
    g.add_argument('--surface', metavar='NPY', help=f'{what} (ny, nx) surface elevation .npy (allow_pickle=False)')
    g.add_argument('--surface-constant', type=float, dest='surface_constant', metavar='Z', help=f'constant {what} surface elevation m')


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


def _load_state(args, catalog, *, resolve):
    '''Read the plant CSV; with resolve=True a given surface is the INITIAL surface used to resolve SURFACE
    anchors, otherwise it is returned as the CURRENT surface and the state must already be resolved.'''
    domain = _domain_from_args(args)
    state = _state.read_plant_csv(args.plants, catalog, domain=domain, time_days=getattr(args, 'time_days', None))
    surface = _surface_from_args(args, state.domain)
    if surface is not None and resolve:
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


def _same_path(a, b):
    if a is None or b is None:
        return False
    if os.path.realpath(os.path.abspath(a)) == os.path.realpath(os.path.abspath(b)):
        return True
    try:
        return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)
    except OSError:
        return False


# ------------------------------------------------------------------ commands
def cmd_validate_species(args):
    cat = _traits.load_catalog(args.species)
    _dump(cat.metadata())
    return 0


def cmd_init(args):
    cat = _traits.load_catalog(args.species)
    _state.preflight_plant_files(args.out, overwrite=args.overwrite)       # CSV and companion destinations
    state, _ = _load_state(args, cat, resolve=True)
    initial = not state.establishment_resolved
    if initial:
        print(f'note: {int(state.establishment_unresolved.sum())} plant(s) keep the SURFACE token (no initial surface '
              'given); the output is an INITIAL input file (companion metadata "initial": true), not a publishable '
              'snapshot', file=sys.stderr)
    path, meta_path = _state.write_plant_files(
        state, args.out, overwrite=args.overwrite, initial=initial,
        provenance={'command': 'leaf init', 'source_csv': args.plants, 'species_file': args.species})
    _dump({'written': path, 'metadata_written': meta_path, 'initial': initial, 'summary': state.summary()})
    return 0


def cmd_generate(args):
    cat = _traits.load_catalog(args.species)
    meta_out = _state.metadata_path(args.out)
    if args.report is not None and _same_path(args.out, args.report):
        raise ValueError('--report must not be the same file as --out (the JSON report would replace the plant CSV)')
    if args.report is not None and _same_path(meta_out, args.report):
        raise ValueError(f'--report must not be the plant metadata companion {meta_out!r} of --out (the JSON report '
                         'would replace the companion metadata of the plant CSV)')
    _state.preflight_plant_files(args.out, overwrite=args.overwrite)       # CSV and companion destinations
    if args.report is not None:
        _state.preflight_destination(args.report, overwrite=args.overwrite)
    with open(args.config, 'r', encoding='utf-8') as fh:
        config = _traits.parse_strict_json(fh.read(), what='generation config')
    cfg = _init.validate_generation_config(config, cat)
    surface = _surface_from_args(args, cfg['domain'])
    state, report = _init.generate_population(cat, config, surface=surface)
    initial = not state.establishment_resolved
    # serialise ALL outputs (plain CSV, companion metadata, report) before touching any file
    csv_text, meta_text = _state.plant_files_text(
        state, initial=initial, provenance=dict(state.provenance, command='leaf generate', config_file=args.config),
        csv_file=os.path.basename(args.out))
    report['written'] = os.path.abspath(args.out)
    report['metadata_written'] = os.path.abspath(meta_out)
    report_text = json.dumps(_jsonable(report), indent=1, allow_nan=False) + '\n' if args.report is not None else None
    report_created = False
    if report_text is not None:
        report_existed = os.path.lexists(args.report)
        _state.atomic_write_text(args.report, report_text, overwrite=args.overwrite)
        report_created = not report_existed
    try:
        path, meta_path = _state.publish_plant_files(args.out, csv_text.encode('utf-8'), meta_text.encode('utf-8'),
                                                     overwrite=args.overwrite)
    except BaseException:
        if report_created:
            try:
                os.unlink(args.report)   # do not leave a report describing a population that was never written
            except OSError:
                pass
        raise
    brief = {'written': path, 'metadata_written': meta_path, 'report_written': args.report, 'n_plants': report['n_plants'],
             'establishment': report['establishment'], 'initial': initial,
             'species': [{k: p[k] for k in ('species', 'target_cover', 'measured_union_cover', 'count', 'accepted')}
                         for p in report['species']],
             'total_union_cover_all_species': report['total_union_cover_all_species'],
             'attempts': report['attempts'], 'placement_passes': report['placement_passes']}
    _dump(brief)
    return 0


def cmd_inspect(args):
    cat = _traits.load_catalog(args.species)
    state, surface = _load_state(args, cat, resolve=False)
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
                                  'exposed_height_m': st(geo['exposed_height_m']),
                                  'fully_buried': int(np.count_nonzero(geo['fully_buried'] & standing)),
                                  'elevated_base': int(np.count_nonzero((geo['exposure_depth_m'] > 0.0) & standing)),
                                  'note': geo['diagnostics_note']}
    if args.cover_subsamples:
        sel = standing & (geo['radius_m'] > 0.0)
        m = _canopy.measure_cover(state.domain, geo['x_m'][sel], geo['y_m'][sel], geo['radius_m'][sel],
                                  geo['species_index'][sel], len(state.species_names), subsamples=args.cover_subsamples)
        out['cover'] = {'method': m['method'], 'resolution_m': m['resolution_m'], 'admission': m['admission'],
                        'per_species': {state.species_names[g['group']]: g for g in m['groups']},
                        'total_union_fraction': m['total_union_fraction'], 'error_basis': m['error_basis'],
                        'boundary_note': m['boundary_note']}
    _dump(out)
    return 0


def cmd_export_canopy(args):
    cat = _traits.load_catalog(args.species)
    _state.preflight_destination(args.out, overwrite=args.overwrite)
    state, surface = _load_state(args, cat, resolve=False)
    if surface is None:
        raise ValueError('export-canopy requires the CURRENT surface (--surface or --surface-constant): safe MAPLE '
                         'publication validates every canopy against its establishment anchor')
    statuses = tuple(s.strip() for s in args.statuses.split(',') if s.strip())
    view = _canopy.maple_vegetation_view(state, cat, surface, statuses=statuses, height_mode=args.height_mode,
                                         unsupported_policy=args.unsupported)
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
    if not view['meta']['native_maple_valid']:
        print('warning: the exported view is NOT native-MAPLE-valid (' + view['meta']['validity_note'] + ')', file=sys.stderr)
    _dump({'written': args.out, 'n_objects': view['meta']['n_objects'], 'height_mode': args.height_mode,
           'native_maple_valid': view['meta']['native_maple_valid'], 'n_unsupported': view['meta']['n_unsupported'],
           'n_unsupported_excluded': view['meta']['n_unsupported_excluded'],
           'n_unsupported_flagged_kept': view['meta']['n_unsupported_flagged_kept'],
           'excluded_by_status': view['meta']['excluded_by_status']})
    return 0


# ------------------------------------------------------------------ parser
def build_parser():
    ap = argparse.ArgumentParser(prog='python -m leaf', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command')

    p = sub.add_parser('validate-species', help='validate a species catalog and print its metadata/hash')
    p.add_argument('--species', required=True)
    p.set_defaults(func=cmd_validate_species)

    p = sub.add_parser('init', help='read a plant list CSV, resolve SURFACE anchors from the INITIAL surface, write a state CSV')
    p.add_argument('--species', required=True)
    p.add_argument('--plants', required=True,
                   help='input plain plant CSV (a handwritten list has no <stem>.metadata.json companion: give the domain)')
    p.add_argument('--time-days', type=float, dest='time_days', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--overwrite', action='store_true')
    _add_domain_args(p)
    _add_surface_args(p, 'INITIAL')
    p.set_defaults(func=cmd_init)

    p = sub.add_parser('generate', help='generate a multispecies population from target covers')
    p.add_argument('--species', required=True)
    p.add_argument('--config', required=True, help='generation JSON (domain, seed, overlap, limits, species targets)')
    p.add_argument('--out', required=True)
    p.add_argument('--report', default=None,
                   help='optional JSON generation report path (must differ from --out and from its <stem>.metadata.json companion)')
    p.add_argument('--overwrite', action='store_true')
    _add_surface_args(p, 'INITIAL')
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser('inspect', help='summarise a state CSV (geometry, burial diagnostics vs the CURRENT surface, optional cover)')
    p.add_argument('--species', required=True)
    p.add_argument('--plants', required=True)
    p.add_argument('--time-days', type=float, dest='time_days', default=None)
    p.add_argument('--cover-subsamples', type=int, dest='cover_subsamples', default=0, help='measure cover at dx/N (0 = skip)')
    _add_domain_args(p)
    _add_surface_args(p, 'CURRENT')
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser('export-canopy', help='export the MAPLE-compatible canopy view (+ID sidecars) as JSON or NPZ; '
                                             'requires the CURRENT surface')
    p.add_argument('--species', required=True)
    p.add_argument('--plants', required=True)
    p.add_argument('--time-days', type=float, dest='time_days', default=None)
    p.add_argument('--out', required=True, help='.json or .npz')
    p.add_argument('--height-mode', dest='height_mode', choices=_canopy.HEIGHT_MODES, default='exposed',
                   help='exposed (safe default: intact height minus burial) or intact_diagnostic (labelled, never native-valid)')
    p.add_argument('--unsupported', choices=_canopy.UNSUPPORTED_POLICIES, default='error',
                   help='eroded (elevated-base) / fully buried plants: error (default), exclude or flag')
    p.add_argument('--statuses', default='live,dead_standing')
    p.add_argument('--overwrite', action='store_true')
    _add_domain_args(p)
    _add_surface_args(p, 'CURRENT')
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
