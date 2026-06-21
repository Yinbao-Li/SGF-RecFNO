# -*- coding: utf-8 -*-
"""Geometry-aware RecFNO (IsoRecFNO) for sparse field reconstruction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.fno import FNORecon


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class GeometryBranch(nn.Module):
    """Predict multi-level isotherm SDF maps from a coarse temperature field."""

    def __init__(self, num_levels=4, width=32):
        super().__init__()
        self.num_levels = num_levels
        self.stem = _ConvBlock(2, width)
        self.down = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, width * 2), width * 2),
            nn.GELU(),
            _ConvBlock(width * 2, width * 2),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1),
            nn.GELU(),
            _ConvBlock(width, width),
        )
        self.head = nn.Conv2d(width, num_levels, 1)

    def forward(self, coarse):
        gx, gy = _central_grad(coarse)
        x = torch.cat([coarse, torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)], dim=1)
        x = self.up(self.down(self.stem(x)))
        return torch.tanh(self.head(x))


class ResidualDecoder(nn.Module):
    """Fuse coarse field and SDF features into a high-frequency residual."""

    def __init__(self, num_levels=4, width=32):
        super().__init__()
        in_ch = 1 + num_levels
        self.enc1 = _ConvBlock(in_ch, width)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(width, width * 2)
        self.up = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
        self.dec = _ConvBlock(width * 2, width)
        self.out = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, coarse, sdf_pred):
        e1 = self.enc1(torch.cat([coarse, sdf_pred], dim=1))
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        d1 = self.dec(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def _central_grad(field):
    left = F.pad(field, (1, 0, 0, 0))[:, :, :, :-1]
    right = F.pad(field, (0, 1, 0, 0))[:, :, :, 1:]
    gx = 0.5 * (right - left)
    top = F.pad(field, (0, 0, 1, 0))[:, :, :-1, :]
    bottom = F.pad(field, (0, 0, 0, 1))[:, :, 1:, :]
    gy = 0.5 * (bottom - top)
    return gx, gy


class IsoRecFNO(nn.Module):
    """RecFNO coarse recovery + geometry SDF branch + residual refinement.

    Architecture:
        coarse = FNORecon(sparse sensors)
        sdf_pred = GeometryBranch(coarse)
        delta = ResidualDecoder(coarse, sdf_pred)
        field = coarse + delta
    """

    def __init__(
        self,
        sensor_num,
        fc_size,
        out_size,
        modes1=50,
        modes2=50,
        width=32,
        num_iso_levels=4,
        quantiles=(0.2, 0.4, 0.6, 0.8),
    ):
        super().__init__()
        self.quantiles = tuple(quantiles)
        self.num_iso_levels = num_iso_levels

        self.coarse_net = FNORecon(
            sensor_num=sensor_num,
            fc_size=fc_size,
            out_size=out_size,
            modes1=modes1,
            modes2=modes2,
            width=width,
        )
        self.geometry_branch = GeometryBranch(num_levels=num_iso_levels, width=width)
        self.residual_decoder = ResidualDecoder(num_levels=num_iso_levels, width=width)

    def forward(self, x, return_aux=False):
        coarse = self.coarse_net(x)
        sdf_pred = self.geometry_branch(coarse)
        delta = self.residual_decoder(coarse, sdf_pred)
        field = coarse + delta

        if return_aux:
            return {
                'field': field,
                'coarse': coarse,
                'sdf_pred': sdf_pred,
                'delta': delta,
            }
        return field

    @classmethod
    def from_fno_recfno(cls, fno_recon, num_iso_levels=4, quantiles=(0.2, 0.4, 0.6, 0.8)):
        """Build IsoRecFNO around an existing FNORecon instance."""
        model = cls(
            sensor_num=fno_recon.fc[0].in_features,
            fc_size=fno_recon.fc_size,
            out_size=fno_recon.out_size,
            modes1=fno_recon.modes1,
            modes2=fno_recon.modes2,
            width=fno_recon.width,
            num_iso_levels=num_iso_levels,
            quantiles=quantiles,
        )
        model.coarse_net.load_state_dict(fno_recon.state_dict())
        return model


if __name__ == '__main__':
    net = IsoRecFNO(sensor_num=25, fc_size=(12, 12), out_size=(200, 200))
    x = torch.randn(2, 25)
    out = net(x, return_aux=True)
    print({k: v.shape for k, v in out.items()})
