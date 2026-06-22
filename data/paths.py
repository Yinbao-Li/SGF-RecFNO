# -*- coding: utf-8 -*-
"""Central path resolution and dataset file locations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Union


def repo_root() -> Path:
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
    env = os.environ.get('RECFNO_DATA_ROOT')
    if env:
        return Path(env).expanduser().resolve()
    root = repo_root()
    candidates = [root / 'data', root.parent / 'data']
    for path in candidates:
        heat = path / 'heat'
        if (heat / 'train.h5').is_file() or (heat / 'temperature6000.h5').is_file():
            return path
    found = _first_existing(candidates)
    return found or (root / 'data')


def external_root() -> Path:
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


def checkpoint_root() -> Path:
    env = os.environ.get('RECFNO_CKPT_ROOT')
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / 'checkpoints'


def heat2d_log_root() -> Path:
    env = os.environ.get('RECFNO_LOG_ROOT')
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / 'heat2D' / 'logs'


DATA_ROOT = str(data_root())
CHECKPOINT_ROOT = str(checkpoint_root())
HEAT_DIR = os.path.join(DATA_ROOT, 'heat')

HEAT_TRAIN_H5 = os.path.join(HEAT_DIR, 'train.h5')
HEAT_VAL_H5 = os.path.join(HEAT_DIR, 'val.h5')
HEAT_TEST_H5 = os.path.join(HEAT_DIR, 'test.h5')
HEAT_SPLITS_JSON = os.path.join(HEAT_DIR, 'splits.json')


def _resolve_heat_h5() -> str:
    for name in ('temperature6000.h5', 'temperature.h5'):
        path = os.path.join(HEAT_DIR, name)
        if os.path.isfile(path):
            return path
    return os.path.join(HEAT_DIR, 'temperature6000.h5')


HEAT_H5 = _resolve_heat_h5()


def heat_split_paths() -> Dict[str, Union[str, None]]:
    """
    Return layout for loading heat data.
    mode 'split': train/val/test.h5 in repo
    mode 'monolithic': single temperature6000.h5
    """
    train, val, test = HEAT_TRAIN_H5, HEAT_VAL_H5, HEAT_TEST_H5
    if all(os.path.isfile(p) for p in (train, val, test)):
        return {'mode': 'split', 'train': train, 'val': val, 'test': test, 'monolithic': None}
    mono = HEAT_H5
    if os.path.isfile(mono):
        return {'mode': 'monolithic', 'monolithic': mono, 'train': None, 'val': None, 'test': None}
    return {'mode': 'monolithic', 'monolithic': mono, 'train': train, 'val': val, 'test': test}


def heat_field_key(h5_path: Optional[str] = None) -> str:
    import h5py

    if h5_path:
        path = h5_path
    else:
        layout = heat_split_paths()
        if layout['mode'] == 'split':
            path = layout['train']
        else:
            path = layout['monolithic']

    with h5py.File(path, 'r') as f:
        if 'u' in f:
            return 'u'
        if 'sol' in f:
            return 'sol'
    raise KeyError(f"no 'u' or 'sol' dataset in {path}")
