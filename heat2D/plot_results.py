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
from data.paths import HEAT_H5, heat_field_key

f = h5py.File(HEAT_H5, 'r')
data = f[heat_field_key()][0, 0, :, :]
f.close()

positions = np.array([[199, 90], [31, 178], [0, 97], [0, 16]])
print(positions.tolist())
plot_locations(positions, data)
