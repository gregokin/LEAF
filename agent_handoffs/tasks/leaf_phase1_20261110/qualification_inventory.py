"""Report source guards and actual interpreter/module origins for independent checks."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib
import json
import platform
import sys

TASK = Path(__file__).resolve().parent


def identity(path):
    raw = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def main():
    pinned = json.loads((TASK / 'upstream_sources.json').read_text())
    sources = {path: identity(path) for path in pinned}
    changed = [path for path in pinned if pinned[path] != sources[path]]
    modules = {}
    for name in ('numpy', 'numba', 'cupy', 'leaf',
                 'maple_syrup.subsurface', 'maple_syrup.subsurface_transpiration'):
        module = importlib.import_module(name)
        modules[name] = {'file': str(Path(module.__file__).resolve()),
                         'version': getattr(module, '__version__', getattr(module, 'VERSION', None))}
    print(json.dumps({'at_utc': datetime.now(timezone.utc).isoformat(),
                      'interpreter': sys.executable, 'python': sys.version,
                      'platform': platform.platform(), 'modules': modules,
                      'upstream_sources': sources, 'changed_from_task_baseline': changed,
                      'scope': 'source-only inventory; no physical model or GPU allocation'}, indent=2))
    if changed:
        raise SystemExit('Upstream source changed; inspect and rebind before relying on it')


if __name__ == '__main__':
    main()
