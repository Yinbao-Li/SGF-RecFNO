# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 10:57
# @Author  : zhaoxiaoyu
# @File    : generate_locations_cylinder.py
import pickle
import numpy as np
import sys
import os

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)

from utils.utils import generate_locations
from data.paths import CYLINDER_PICKLE

df = open(CYLINDER_PICKLE, 'rb')
data = np.squeeze(pickle.load(df), axis=-1)[[i for i in range(4250)], :, :]
locations = generate_locations(data, observe_num=100, interval=2)

print(locations)
