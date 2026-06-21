# -*- coding: utf-8 -*-
"""Load model classes from official GitHub repos (runtime import, no reimplementation)."""
import importlib.util
import os
import sys

from benchmark.external_paths import REPOS, REPO_URLS, ensure_repos


def _exec_module_prefix(py_path, stop_marker):
    """Execute a repo script up to its training block."""
    with open(py_path, 'r', encoding='utf-8') as f:
        src = f.read()
    if stop_marker not in src:
        raise RuntimeError(f'Stop marker not found in {py_path}')
    src = src.split(stop_marker)[0]
    ns = {'__file__': py_path}
    repo_root = os.path.dirname(os.path.dirname(py_path))
    old_path = sys.path[:]
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.dirname(py_path))
    try:
        exec(compile(src, py_path, 'exec'), ns)
    finally:
        sys.path[:] = old_path
    return ns


def load_geofno_classes():
    ensure_repos()
    path = os.path.join(REPOS['geofno'], 'elasticity', 'elas_geofno.py')
    ns = _exec_module_prefix(path, '################################################################\n# load data')
    return ns['FNO2d'], ns['IPHI']


def load_pino_fno2d():
    ensure_repos()
    root = REPOS['physics_informed']
    if root not in sys.path:
        sys.path.insert(0, root)
    from models.fourier2d import FNO2d
    return FNO2d


def load_gino():
    ensure_repos()
    root = REPOS['neuraloperator']
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from neuralop.models.gino import GINO
    except (ImportError, ModuleNotFoundError, TypeError) as exc:
        raise ImportError(
            'GINO requires neuraloperator deps (tensorly, tltorch, opt_einsum) '
            f'and Python>=3.9 recommended. See {REPO_URLS["GINO"]}. '
            f'Original error: {exc}'
        ) from exc
    return GINO
