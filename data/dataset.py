# -*- coding: utf-8 -*-
"""Heat conduction datasets for sparse field reconstruction."""
from __future__ import annotations

import os
from collections import defaultdict

import h5py
import numpy as np
import torch
from scipy.interpolate import griddata
from torch.utils.data import Dataset
from tqdm import tqdm

from data.paths import heat_split_paths, heat_field_key

# Fixed 25-sensor grid (200×200 field)
HEAT_SENSOR_POSITIONS = np.array(
    [[33, 33], [33, 66], [33, 99], [33, 132], [33, 165],
     [66, 33], [66, 66], [66, 99], [66, 132], [66, 165],
     [99, 33], [99, 66], [99, 99], [99, 132], [99, 165],
     [132, 33], [132, 66], [132, 99], [132, 132], [132, 165],
     [165, 33], [165, 66], [165, 99], [165, 132], [165, 165]],
    dtype=np.int64,
)

# Global index ranges (6000 samples)
_SPLIT_RANGES = (
    (0, 4000, 'train'),
    (4000, 5000, 'val'),
    (5000, 6000, 'test'),
)


def _global_to_local(global_idx: int) -> tuple[str, int]:
    for lo, hi, name in _SPLIT_RANGES:
        if lo <= global_idx < hi:
            return name, global_idx - lo
    raise IndexError(f'global index {global_idx} out of range 0..5999')


def _load_fields(global_indices: list[int]) -> np.ndarray:
    """Load samples by global index from monolithic or split H5 files."""
    paths = heat_split_paths()
    index_list = list(global_indices)

    if paths['mode'] == 'monolithic':
        path = paths['monolithic']
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'Heat data not found: {path}. Run git lfs pull or see data/heat/README.md'
            )
        with h5py.File(path, 'r') as f:
            k = heat_field_key(path)
            return np.asarray(f[k][index_list])

    # split files
    for name in ('train', 'val', 'test'):
        if not os.path.isfile(paths[name]):
            raise FileNotFoundError(
                f'Missing {paths[name]}. Run: git lfs pull'
            )

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for gi in index_list:
        split, local = _global_to_local(gi)
        grouped[split].append((gi, local))

    out: np.ndarray | None = None
    pos = {gi: i for i, gi in enumerate(index_list)}

    for split, pairs in grouped.items():
        path = paths[split]
        locals_ = [p[1] for p in pairs]
        globals_ = [p[0] for p in pairs]
        with h5py.File(path, 'r') as f:
            k = heat_field_key(path)
            block = np.asarray(f[k][locals_])
        if out is None:
            out = np.empty((len(index_list),) + block.shape[1:], dtype=block.dtype)
        for g, row in zip(globals_, block):
            out[pos[g]] = row

    assert out is not None
    return out


class HeatDataset(Dataset):
    """25 sensor values → full 200×200 temperature field."""

    def __init__(self, index, mean=308, std=50):
        super().__init__()
        self.mean, self.std = mean, std
        index_list = list(index)
        arr = _load_fields(index_list)
        self.data = torch.from_numpy(arr).float()
        if self.data.dim() == 3:
            self.data = self.data.unsqueeze(1)
        self.data = (self.data - mean) / std

        sparse_data = []
        for i in range(HEAT_SENSOR_POSITIONS.shape[0]):
            r, c = HEAT_SENSOR_POSITIONS[i]
            sparse_data.append(self.data[:, 0, r, :][:, c].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class HeatInterpolDataset(Dataset):
    """4-channel grid input (interp + mask + coords) for PINO-style models."""

    def __init__(self, index, mean=308, std=50):
        super().__init__()
        self.mean, self.std = mean, std
        index_list = list(index)
        raw = _load_fields(index_list)
        if raw.ndim == 3:
            raw = raw[:, np.newaxis, :, :]
        raw = (raw - mean) / std

        _, _, h, w = raw.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)

        sparse_locations_ex = np.zeros_like(HEAT_SENSOR_POSITIONS)
        for i in range(HEAT_SENSOR_POSITIONS.shape[0]):
            r, c = HEAT_SENSOR_POSITIONS[i]
            sparse_locations_ex[i, 0] = x_coor[r, c]
            sparse_locations_ex[i, 1] = y_coor[r, c]

        sparse_vals = []
        for i in range(HEAT_SENSOR_POSITIONS.shape[0]):
            r, c = HEAT_SENSOR_POSITIONS[i]
            sparse_vals.append(raw[:, 0, r, :][:, c].reshape(-1, 1))
        sparse_vals = np.concatenate(sparse_vals, axis=-1)

        sparse_datas = []
        print('Building interpolated grid inputs (one-time)...', flush=True)
        for i in tqdm(range(sparse_vals.shape[0])):
            interp = griddata(sparse_locations_ex, sparse_vals[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(interp, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)

        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(HEAT_SENSOR_POSITIONS.shape[0]):
            r, c = HEAT_SENSOR_POSITIONS[i]
            mask[r, c] = 1

        self.data = torch.from_numpy(raw).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        self.coor = torch.cat([
            torch.from_numpy(x_coor).unsqueeze(dim=0) / w,
            torch.from_numpy(y_coor).unsqueeze(dim=0) / h,
        ], dim=0).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index], self.mask, self.coor], dim=0), self.data[index]

    def __len__(self):
        return self.data.shape[0]
