# -*- coding: utf-8 -*-
"""Wrapper modules that call official repo implementations on heat data."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.external_loaders import (
    load_geofno_classes,
    load_gino,
    load_pino_fno2d,
)
from benchmark.heat_adapters import (
    grid_coords_normalized,
    grid_input_to_hwc,
    sensor_batch_to_gino,
    sensor_coords_normalized,
)


class PINOHeatRecon(nn.Module):
    """PINO operator-learning FNO2d from neuraloperator/physics_informed."""

    def __init__(self, modes=16, width=32):
        super().__init__()
        FNO2d = load_pino_fno2d()
        n = 4
        self.model = FNO2d(
            modes1=[modes] * n,
            modes2=[modes] * n,
            width=width,
            layers=[width] * (n + 1),
            in_dim=4,
            out_dim=1,
        )

    def forward(self, grid_in):
        out = self.model(grid_input_to_hwc(grid_in))
        return out.permute(0, 3, 1, 2)


class GeoFNOHeatRecon(nn.Module):
    """Geo-FNO (neuraloperator/Geo-FNO elasticity/elas_geofno.py) with 25 sensor mesh input."""

    OUT_SIZE = 64

    def __init__(self, modes=12, width=32):
        super().__init__()
        FNO2d, IPHI = load_geofno_classes()
        self.iphi = IPHI(width=width)
        self.model = FNO2d(
            modes, modes, width,
            in_channels=1, out_channels=1,
            is_mesh=True, s1=self.OUT_SIZE, s2=self.OUT_SIZE,
        )
        self.register_buffer('_x_in', sensor_coords_normalized(), persistent=False)
        self.register_buffer('_x_out', grid_coords_normalized(self.OUT_SIZE, self.OUT_SIZE), persistent=False)

    def forward(self, sensor_in):
        b = sensor_in.shape[0]
        u = sensor_in.unsqueeze(-1)
        x_in = self._x_in.unsqueeze(0).expand(b, -1, -1)
        x_out = self._x_out.unsqueeze(0).expand(b, -1, -1)
        y = self.model(u, x_in=x_in, x_out=x_out, iphi=self.iphi)
        y = y.permute(0, 2, 1).reshape(b, 1, self.OUT_SIZE, self.OUT_SIZE)
        y = F.interpolate(y, size=(200, 200), mode='bilinear', align_corners=False)
        return y


class GINOHeatRecon(nn.Module):
    """GINO from neuraloperator/neuraloperator."""

    LATENT = 64

    def __init__(self, width=32):
        super().__init__()
        GINO = load_gino()
        self.model = GINO(
            in_channels=1,
            out_channels=1,
            gno_coord_dim=2,
            fno_n_modes=(16, 16),
            fno_hidden_channels=width,
            fno_n_layers=2,
            in_gno_radius=0.2,
            out_gno_radius=0.05,
        )
        coords = grid_coords_normalized(self.LATENT, self.LATENT).reshape(1, self.LATENT, self.LATENT, 2)
        self.register_buffer('_latent_queries', coords, persistent=False)

    def forward(self, sensor_in):
        packs = sensor_batch_to_gino(sensor_in, self.LATENT, self.LATENT)
        out = self.model(
            packs['input_geom'],
            packs['latent_queries'],
            packs['output_queries'],
            x=packs['x'],
        )
        if isinstance(out, dict):
            out = next(iter(out.values()))
        field = out.reshape(packs['batch'], 1, self.LATENT, self.LATENT)
        return F.interpolate(field, size=(200, 200), mode='bilinear', align_corners=False)
