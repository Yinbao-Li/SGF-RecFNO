# -*- coding: utf-8 -*-
"""Heat conduction datasets for sparse field reconstruction."""
import h5py
import numpy as np
import torch
from scipy.interpolate import griddata
from torch.utils.data import Dataset
from tqdm import tqdm

from data.paths import HEAT_H5, heat_field_key

# Fixed 25-sensor grid (200×200 field)
HEAT_SENSOR_POSITIONS = np.array(
    [[33, 33], [33, 66], [33, 99], [33, 132], [33, 165],
     [66, 33], [66, 66], [66, 99], [66, 132], [66, 165],
     [99, 33], [99, 66], [99, 99], [99, 132], [99, 165],
     [132, 33], [132, 66], [132, 99], [132, 132], [132, 165],
     [165, 33], [165, 66], [165, 99], [165, 132], [165, 165]],
    dtype=np.int64,
)


class HeatDataset(Dataset):
    """25 sensor values → full 200×200 temperature field."""

    def __init__(self, index, mean=308, std=50):
        super().__init__()
        self.mean, self.std = mean, std
        with h5py.File(HEAT_H5, 'r') as f:
            key = heat_field_key()
            self.data = torch.from_numpy(f[key][index, :, :, :]).float()
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
        with h5py.File(HEAT_H5, 'r') as f:
            key = heat_field_key()
            raw = f[key][index, :, :, :]
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
