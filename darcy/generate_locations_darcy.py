# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 10:57
# @Author  : zhaoxiaoyu
# @File    : generate_locations_cylinder.py
import h5py

from utils.utils import generate_locations

import sys
import os

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)

from data.paths import DARCY_H5

index = [i for i in range(8000)]
f = h5py.File(DARCY_H5, 'r')
data = f['sol'][index, :, :]
f.close()
locations = generate_locations(data, observe_num=36, interval=20)
print(locations)
