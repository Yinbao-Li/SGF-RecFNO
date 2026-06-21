# -*- coding: utf-8 -*-
"""Central path resolution and dataset file locations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def repo_root() -> Path:
    """RecFNO repository root (directory containing pyproject.toml)."""
    env = os.environ.get('RECFNO_REPO_ROOT')
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def _first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.is_dir():
            return path
    return None


def data_root() -> Path:
    """Root directory for datasets (heat/, cylinder/, darcy/, noaa/)."""
    env = os.environ.get('RECFNO_DATA_ROOT')
    if env:
        return Path(env).expanduser().resolve()
    root = repo_root()
    candidates = [root / 'data', root.parent / 'data']
    for path in candidates:
        for name in ('temperature6000.h5', 'temperature.h5'):
            if (path / 'heat' / name).is_file():
                return path
    found = _first_existing(candidates)
    return found or (root / 'data')


def external_root() -> Path:
    """Root directory for cloned third-party baseline repos."""
    env = os.environ.get('RECFNO_EXTERNAL_ROOT')
    if env:
        return Path(env).expanduser().resolve()
    root = repo_root()
    candidates = [root / 'external', root.parent / 'external']
    for path in candidates:
        if (path / 'neuraloperator').is_dir() or (path / 'physics_informed').is_dir():
            return path
    found = _first_existing(candidates)
    return found or (root / 'external')


def heat2d_log_root() -> Path:
    """Default log/checkpoint directory for heat2D experiments."""
    env = os.environ.get('RECFNO_LOG_ROOT')
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / 'heat2D' / 'logs'


# --- Dataset file paths (used by dataset.py and training scripts) ---

DATA_ROOT = str(data_root())

CYLINDER_PICKLE = os.path.join(DATA_ROOT, 'cylinder', 'Cy_Taira.pickle')
DARCY_H5 = os.path.join(DATA_ROOT, 'darcy', 'darcy12000.h5')
NOAA_MAT = os.path.join(DATA_ROOT, 'noaa', 'sst_weekly.mat')


def _resolve_heat_h5() -> str:
    """Official SharePoint bundle uses temperature6000.h5; older docs say temperature.h5."""
    for name in ('temperature6000.h5', 'temperature.h5'):
        path = os.path.join(DATA_ROOT, 'heat', name)
        if os.path.isfile(path):
            return path
    return os.path.join(DATA_ROOT, 'heat', 'temperature6000.h5')


HEAT_H5 = _resolve_heat_h5()


def heat_field_key(h5_path: Optional[str] = None) -> str:
    """Official temperature6000.h5 uses 'sol'; older docs use 'u'."""
    import h5py

    path = h5_path or HEAT_H5
    with h5py.File(path, 'r') as f:
        if 'u' in f:
            return 'u'
        if 'sol' in f:
            return 'sol'
    raise KeyError(f"no 'u' or 'sol' dataset in {path}")
