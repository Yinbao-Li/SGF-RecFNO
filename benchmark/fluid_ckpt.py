# -*- coding: utf-8 -*-
"""Checkpoint resolution for fluid benchmark models."""
from __future__ import annotations

from benchmark.registry import _find_ckpt

# Models retrained from scratch when task geometry supervision changes.
FRESH_GEOMETRY_MODELS = frozenset({'IsoRecFNO', 'SGF-RecFNO (K=8)'})

_MODEL_SLUG = {
    'SGF-RecFNO (K=8)': 'sgfrecfnok8',
    'SGF-RecFNO': 'sgfrecfno',
    'IsoRecFNO': 'isorecfno',
    'RecFNO': 'recfno',
    'Geo-FNO': 'geofno',
    'GINO': 'gino',
    'PINO': 'pino',
}


def fluid_exp_name(task: str, model: str) -> str:
    """Canonical experiment directory name."""
    slug = _MODEL_SLUG.get(model, model.lower().replace('-', '').replace(' ', ''))
    return f'fluid_{task}_{slug}_300'


def find_fluid_ckpt(task: str, model: str) -> tuple[str | None, str | None]:
    """Return (checkpoint_path, exp_name)."""
    exp = fluid_exp_name(task, model)
    ckpt = _find_ckpt(exp)
    return (ckpt, exp) if ckpt else (None, exp)
