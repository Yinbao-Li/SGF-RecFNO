# -*- coding: utf-8 -*-
"""Self-Geometry Feedback RecFNO (SGF-RecFNO)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.fno import FNORecon, SpectralConv2d
from utils.iso_geometry import (
    DEFAULT_QUANTILES,
    soft_isotherm_sdf,
    temperature_quantile_levels,
)


def extract_self_geometry(field, quantiles=DEFAULT_QUANTILES, sdf_scale=5.0):
    """Build multi-level SDF maps from a predicted coarse field only."""
    levels = temperature_quantile_levels(field, quantiles)
    sdf = soft_isotherm_sdf(field, levels, scale=sdf_scale)
    return sdf, levels


class RefinementBlock(nn.Module):
    """Lightweight Fourier refinement on [coarse, SDF] features."""

    def __init__(self, in_channels, width=32, modes1=24, modes2=24):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.smooth = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(width, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.lift(x)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        x = self.smooth(x)
        return self.out(x)


class SGFRecFNO(nn.Module):
    """Self-Geometry Feedback RecFNO.

    Pipeline:
        coarse = RecFNO(sparse sensors)
        sdf_self = SDF(coarse, quantile levels from coarse)
        delta = RefinementBlock([coarse, sdf_self])
        field = coarse + delta

    GT SDF is never used as input; it only appears in the training loss.
    """

    def __init__(
        self,
        sensor_num,
        fc_size,
        out_size,
        modes1=50,
        modes2=50,
        width=32,
        num_sdf=4,
        quantiles=DEFAULT_QUANTILES,
        sdf_scale=5.0,
        refine_modes1=24,
        refine_modes2=24,
        refine_width=32,
    ):
        super().__init__()
        self.quantiles = tuple(quantiles)
        self.sdf_scale = sdf_scale

        self.coarse_net = FNORecon(
            sensor_num=sensor_num,
            fc_size=fc_size,
            out_size=out_size,
            modes1=modes1,
            modes2=modes2,
            width=width,
        )
        self.refine_net = RefinementBlock(
            in_channels=1 + num_sdf,
            width=refine_width,
            modes1=refine_modes1,
            modes2=refine_modes2,
        )

    def forward(self, x, return_aux=False):
        coarse = self.coarse_net(x)
        sdf_self, levels = extract_self_geometry(
            coarse, self.quantiles, self.sdf_scale,
        )
        geom_feat = torch.cat([coarse, sdf_self], dim=1)
        delta = self.refine_net(geom_feat)
        field = coarse + delta

        if return_aux:
            return {
                'field': field,
                'coarse': coarse,
                'sdf_self': sdf_self,
                'delta': delta,
                'levels': levels,
            }
        return field


if __name__ == '__main__':
    net = SGFRecFNO(25, (12, 12), (200, 200))
    out = net(torch.randn(2, 25), return_aux=True)
    print({k: (v.shape if torch.is_tensor(v) else v) for k, v in out.items()})
