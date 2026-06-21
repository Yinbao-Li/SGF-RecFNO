# -*- coding: utf-8 -*-
"""Convert RecFNO heat datasets into formats expected by external repos."""
import numpy as np
import torch

# Fixed 25-sensor layout (same as HeatDataset)
HEAT_SENSOR_POSITIONS = np.array(
    [[33, 33], [33, 66], [33, 99], [33, 132], [33, 165],
     [66, 33], [66, 66], [66, 99], [66, 132], [66, 165],
     [99, 33], [99, 66], [99, 99], [99, 132], [99, 165],
     [132, 33], [132, 66], [132, 99], [132, 132], [132, 165],
     [165, 33], [165, 66], [165, 99], [165, 132], [165, 165]],
    dtype=np.int64,
)


def sensor_coords_normalized(h=200, w=200, device='cuda'):
    """Return (25, 2) sensor coordinates in [0, 1]^2."""
    xs = HEAT_SENSOR_POSITIONS[:, 1].astype(np.float32) / (w - 1)
    ys = 1.0 - HEAT_SENSOR_POSITIONS[:, 0].astype(np.float32) / (h - 1)
    return torch.from_numpy(np.stack([xs, ys], axis=1)).to(device)


def grid_coords_normalized(h=200, w=200, device='cuda'):
    """Return (H*W, 2) grid coordinates in [0, 1]^2."""
    xs = np.linspace(0, 1, w, dtype=np.float32)
    ys = np.linspace(1, 0, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    return torch.from_numpy(coords).to(device)


def grid_input_to_hwc(grid_in):
    """HeatInterpolDataset tensor (B,4,H,W) -> (B,H,W,4) for PINO grid FNO."""
    return grid_in.permute(0, 2, 3, 1)


def sensor_batch_to_gino(sensor_in, h=200, w=200):
    """
    HeatDataset vector (B,25) -> GINO inputs (shared geometry across batch).
    """
    device = sensor_in.device
    b = sensor_in.shape[0]
    coords = sensor_coords_normalized(h, w, device)
    out_coords = grid_coords_normalized(h, w, device)
    latent = out_coords.reshape(1, h, w, 2)

    return {
        'input_geom': coords.unsqueeze(0),
        'latent_queries': latent,
        'output_queries': out_coords.unsqueeze(0),
        'x': sensor_in.unsqueeze(-1),
        'batch': b,
    }
