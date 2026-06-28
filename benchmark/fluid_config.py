# -*- coding: utf-8 -*-
"""Fluid benchmark task definitions (RecFNO paper splits).

SGF-RecFNO geometry presets (see ``utils/field_geometry.py``):

* **cylinder** — wake mode R1: ``|ω|`` + ``|∇ω|`` SDF (no ω=0), wake column loss ×3
* **darcy** — dual mode: pressure ``p`` + ``log(1+|∇p|)`` SDF channels
"""
from data.fluid_dataset import (
    CYLINDER_TEST,
    CYLINDER_TRAIN,
    CYLINDER_VAL,
    DARCY_TEST,
    DARCY_TRAIN,
    DARCY_VAL,
)
from utils.field_geometry import GEOMETRY_PRESETS

FLUID_MODELS = [
    'SGF-RecFNO', 'SGF-RecFNO (K=8)',
    'IsoRecFNO', 'RecFNO', 'PINO', 'Geo-FNO', 'GINO',
]

CYLINDER_TASK = {
    'name': 'cylinder',
    'train_index': CYLINDER_TRAIN,
    'val_index': CYLINDER_VAL,
    'test_index': CYLINDER_TEST,
    'mean': 0.0,
    'std': 1.0,
    'field_std': 1.0,
    'sensor_num': 4,
    'fc_size': (7, 12),
    'out_size': (112, 192),
    'modes': 24,
    'width': 32,
    'batch_size': 16,
    'epochs': 500,
    'log_dir': 'logs/fluid_cylinder',
    'geometry': GEOMETRY_PRESETS['cylinder'],
}

DARCY_TASK = {
    'name': 'darcy',
    'train_index': DARCY_TRAIN,
    'val_index': DARCY_VAL,
    'test_index': DARCY_TEST,
    'mean': 0.0,
    'std': 0.01,
    'field_std': 0.01,
    'sensor_num': 25,
    'fc_size': (16, 16),
    'out_size': (128, 128),
    'modes': 18,
    'width': 32,
    'batch_size': 16,
    'epochs': 500,
    'log_dir': 'logs/fluid_darcy',
    'geometry': GEOMETRY_PRESETS['darcy'],
}

TASKS = {'cylinder': CYLINDER_TASK, 'darcy': DARCY_TASK}
