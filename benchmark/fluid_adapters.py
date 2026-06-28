# -*- coding: utf-8 -*-
"""Adapters for external models on fluid tasks (variable sensor count & resolution)."""
import numpy as np
import torch
import torch.nn.functional as F

from benchmark.external_loaders import load_geofno_classes, load_gino, load_pino_fno2d
from benchmark.heat_adapters import grid_input_to_hwc


def sensor_coords_from_positions(positions, h: int, w: int, device='cuda'):
    pos = np.asarray(positions, dtype=np.float64)
    xs = pos[:, 1].astype(np.float32) / max(w - 1, 1)
    ys = 1.0 - pos[:, 0].astype(np.float32) / max(h - 1, 1)
    return torch.from_numpy(np.stack([xs, ys], axis=1)).to(device)


def grid_coords_normalized(h: int, w: int, device='cuda'):
    xs = np.linspace(0, 1, w, dtype=np.float32)
    ys = np.linspace(1, 0, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return torch.from_numpy(np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)).to(device)


def sensor_batch_to_gino(sensor_in, sensor_coords, out_h, out_w, latent=64):
    """Pack sensor batch for GINO (same layout as heat benchmark)."""
    device = sensor_in.device
    b = sensor_in.shape[0]
    coords = sensor_coords.to(device)
    latent_coords = grid_coords_normalized(latent, latent, device).reshape(1, latent, latent, 2)
    output_coords = grid_coords_normalized(latent, latent, device).unsqueeze(0)
    return {
        'input_geom': coords.unsqueeze(0),
        'latent_queries': latent_coords,
        'output_queries': output_coords,
        'x': sensor_in.unsqueeze(-1),
        'batch': b,
        'latent': latent,
    }


class PINOFluidRecon(torch.nn.Module):
    def __init__(self, modes=16, width=32):
        super().__init__()
        FNO2d = load_pino_fno2d()
        n = 4
        self.model = FNO2d(
            modes1=[modes] * n, modes2=[modes] * n,
            width=width, layers=[width] * (n + 1),
            in_dim=4, out_dim=1,
        )

    def forward(self, grid_in):
        out = self.model(grid_input_to_hwc(grid_in))
        return out.permute(0, 3, 1, 2)


class GeoFNOFluidRecon(torch.nn.Module):
    """Geo-FNO with flattened mesh coords (same layout as heat benchmark)."""

    def __init__(self, sensor_positions, out_size, modes=12, width=32, latent=64):
        super().__init__()
        FNO2d, IPHI = load_geofno_classes()
        self.out_h, self.out_w = out_size
        self.latent = latent
        self.iphi = IPHI(width=width)
        self.model = FNO2d(
            modes, modes, width,
            in_channels=1, out_channels=1,
            is_mesh=True, s1=latent, s2=latent,
        )
        coords = sensor_coords_from_positions(sensor_positions, self.out_h, self.out_w, device='cpu')
        self.register_buffer('_x_in', coords, persistent=False)
        self.register_buffer('_x_out', grid_coords_normalized(latent, latent, device='cpu'), persistent=False)

    def forward(self, sensor_in):
        b = sensor_in.shape[0]
        u = sensor_in.unsqueeze(-1)
        x_in = self._x_in.unsqueeze(0).expand(b, -1, -1)
        x_out = self._x_out.unsqueeze(0).expand(b, -1, -1)
        y = self.model(u, x_in=x_in, x_out=x_out, iphi=self.iphi)
        y = y.permute(0, 2, 1).reshape(b, 1, self.latent, self.latent)
        return F.interpolate(y, size=(self.out_h, self.out_w), mode='bilinear', align_corners=False)


class GINOFluidRecon(torch.nn.Module):
    def __init__(self, sensor_positions, out_size, width=32, latent=64):
        super().__init__()
        GINO = load_gino()
        self.out_h, self.out_w = out_size
        self.latent = latent
        self.model = GINO(
            in_channels=1, out_channels=1,
            gno_coord_dim=2,
            fno_n_modes=(16, 16),
            fno_hidden_channels=width,
            fno_n_layers=2,
            in_gno_radius=0.2,
            out_gno_radius=0.05,
        )
        self.register_buffer(
            '_sensor_coords',
            sensor_coords_from_positions(sensor_positions, self.out_h, self.out_w, device='cpu'),
            persistent=False,
        )

    def forward(self, sensor_in):
        packs = sensor_batch_to_gino(
            sensor_in, self._sensor_coords, self.out_h, self.out_w, self.latent,
        )
        out = self.model(
            packs['input_geom'],
            packs['latent_queries'],
            packs['output_queries'],
            x=packs['x'],
        )
        if isinstance(out, dict):
            out = next(iter(out.values()))
        field = out.reshape(packs['batch'], 1, self.latent, self.latent)
        return F.interpolate(field, size=(self.out_h, self.out_w), mode='bilinear', align_corners=False)
