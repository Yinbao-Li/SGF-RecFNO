# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 22:16
# @Author  : zhaoxiaoyu
# @File    : plot_results.py
import h5py
import numpy as np
import sys
import os

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)

from utils.visualization import plot_locations
from data.paths import NOAA_MAT

data = h5py.File(NOAA_MAT)
sst = data['sst'][:]
mask = np.isnan(sst[0, :]).reshape(360, 180).transpose()
mask = np.flip(mask, axis=0).copy()
sst[np.isnan(sst)] = 0
data = sst.reshape(sst.shape[0], 1, 360, 180)[0, 0, :, :].transpose(1, 0)
data = np.flip(data, axis=[0])

positions = np.array(
    [[43, 49], [125, 302], [64, 119], [22, 196], [101, 278], [146, 144], [167, 174], [0, 228]]
)
plot_locations(positions, data)
