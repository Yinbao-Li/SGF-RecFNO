# -*- coding: utf-8 -*-
"""Cylinder wake & Darcy flow datasets (RecFNO paper settings)."""
from __future__ import annotations

import json
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import _nearest_sensor_index_map
from data.paths import CYLINDER_PICKLE, CYLINDER_SPLITS_JSON, DARCY_H5, DARCY_SPLITS_JSON


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


CYLINDER_META = _load_json(CYLINDER_SPLITS_JSON)
DARCY_META = _load_json(DARCY_SPLITS_JSON)

CYLINDER_SENSOR_4 = np.array(CYLINDER_META['sensor_positions_4'], dtype=np.int64)
CYLINDER_SENSOR_8 = np.array(CYLINDER_META['sensor_positions_8'], dtype=np.int64)
DARCY_SENSOR_POS = np.array(DARCY_META['sensor_positions'], dtype=np.int64)

CYLINDER_TRAIN = list(range(*CYLINDER_META['train']))
CYLINDER_VAL = list(range(*CYLINDER_META['val']))
CYLINDER_TEST = list(range(*CYLINDER_META['test']))
DARCY_TRAIN = list(range(*DARCY_META['train']))
DARCY_VAL = list(range(*DARCY_META['val']))
DARCY_TEST = list(range(*DARCY_META['test']))


def _sensor_values(fields: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """fields: (N,H,W) or (N,1,H,W) -> observe (N, n_sensors)."""
    if fields.ndim == 4:
        fields = fields[:, 0]
    chunks = []
    for r, c in positions:
        chunks.append(fields[:, r, c].reshape(-1, 1))
    return np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)


def _build_grid_tensors(fields: np.ndarray, positions: np.ndarray, precompute: bool = True):
    """Nearest-neighbor Voronoi grid input (observe, mask, coor) for PINO."""
    if fields.ndim == 3:
        fields = fields[:, np.newaxis, :, :]
    n, _, h, w = fields.shape

    x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
    x_coor, y_coor = np.meshgrid(x_coor, y_coor)

    sparse_locations_ex = np.zeros((len(positions), 2), dtype=np.float32)
    for i, (r, c) in enumerate(positions):
        sparse_locations_ex[i, 0] = x_coor[r, c]
        sparse_locations_ex[i, 1] = y_coor[r, c]

    sparse_vals = _sensor_values(fields, positions)
    nn_index = _nearest_sensor_index_map(sparse_locations_ex, x_coor, y_coor)

    observe = None
    if precompute:
        flat = nn_index.ravel()
        grid = sparse_vals[:, flat].reshape(n, h, w)
        observe = torch.from_numpy(grid).float().unsqueeze(1)

    mask = np.zeros((h, w), dtype=np.float32)
    for r, c in positions:
        mask[r, c] = 1.0

    coor = torch.cat([
        torch.from_numpy(x_coor).unsqueeze(0) / w,
        torch.from_numpy(y_coor).unsqueeze(0) / h,
    ], dim=0).float()

    return {
        'fields': torch.from_numpy(fields).float(),
        'nn_index': nn_index,
        'sparse_vals': sparse_vals,
        'observe': observe,
        'mask': torch.from_numpy(mask).float().unsqueeze(0),
        'coor': coor,
    }


class CylinderDataset(Dataset):
    """4 sensors → 112×192 vorticity field."""

    def __init__(self, index, mean=0.0, std=1.0):
        with open(CYLINDER_PICKLE, 'rb') as f:
            raw = pickle.load(f)
        raw = torch.from_numpy(np.asarray(raw)).float().permute(0, 3, 1, 2)[index]
        self.data = (raw - mean) / std
        self.observe = torch.from_numpy(
            _sensor_values(self.data.numpy(), CYLINDER_SENSOR_4)
        ).float()

    def __getitem__(self, index):
        return self.observe[index], self.data[index]

    def __len__(self):
        return self.data.shape[0]


class CylinderInterpolDataset(Dataset):
    """8 sensors, Voronoi grid input for PINO (112×192)."""

    def __init__(self, index, mean=0.0, std=1.0, precompute=True):
        with open(CYLINDER_PICKLE, 'rb') as f:
            raw = pickle.load(f)[index, :, :, :].transpose(0, 3, 1, 2)
        raw = (raw - mean) / std
        parts = _build_grid_tensors(raw, CYLINDER_SENSOR_8, precompute=precompute)
        self.data = parts['fields']
        self.nn_index = parts['nn_index']
        self.sparse_vals = parts['sparse_vals']
        self.observe = parts['observe']
        self.mask = parts['mask']
        self.coor = parts['coor']

    def __getitem__(self, index):
        if self.observe is None:
            interp = self.sparse_vals[index][self.nn_index]
            observe = torch.from_numpy(interp).float().unsqueeze(0)
        else:
            observe = self.observe[index]
        return torch.cat([observe, self.mask, self.coor], dim=0), self.data[index]

    def __len__(self):
        return self.data.shape[0]


class DarcyDataset(Dataset):
    """25 sensors → 128×128 pressure field."""

    def __init__(self, index, mean=0.0, std=0.01):
        with h5py.File(DARCY_H5, 'r') as f:
            raw = np.asarray(f['sol'][index, ::2, ::2], dtype=np.float32)
        raw = (raw - mean) / std
        self.data = torch.from_numpy(raw).float().unsqueeze(1)
        self.observe = torch.from_numpy(
            _sensor_values(raw, DARCY_SENSOR_POS)
        ).float()

    def __getitem__(self, index):
        return self.observe[index], self.data[index]

    def __len__(self):
        return self.data.shape[0]


class DarcyInterpolDataset(Dataset):
    """25 sensors, Voronoi grid input for PINO (128×128)."""

    def __init__(self, index, mean=0.0, std=0.01, precompute=True):
        with h5py.File(DARCY_H5, 'r') as f:
            raw = np.asarray(f['sol'][index, ::2, ::2], dtype=np.float32)
        raw = (raw - mean) / std
        parts = _build_grid_tensors(raw, DARCY_SENSOR_POS, precompute=precompute)
        self.data = parts['fields']
        self.nn_index = parts['nn_index']
        self.sparse_vals = parts['sparse_vals']
        self.observe = parts['observe']
        self.mask = parts['mask']
        self.coor = parts['coor']

    def __getitem__(self, index):
        if self.observe is None:
            interp = self.sparse_vals[index][self.nn_index]
            observe = torch.from_numpy(interp).float().unsqueeze(0)
        else:
            observe = self.observe[index]
        return torch.cat([observe, self.mask, self.coor], dim=0), self.data[index]

    def __len__(self):
        return self.data.shape[0]
