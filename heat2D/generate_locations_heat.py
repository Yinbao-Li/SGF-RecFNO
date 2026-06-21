# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 10:57
# @Author  : zhaoxiaoyu
# @File    : generate_locations_heat.py
import h5py
import sys
import os

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)

from utils.utils import generate_locations
from data.paths import HEAT_H5, heat_field_key

f = h5py.File(HEAT_H5, 'r')
data = f[heat_field_key()][:, 0, :, :]
f.close()
locations = generate_locations(data, observe_num=5000, interval=2)

print(locations)
