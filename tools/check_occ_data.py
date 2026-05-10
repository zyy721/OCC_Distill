'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-03-29 15:48:49
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
import numpy as np


openocc_fp = "/mnt/data2/zhanghm/Code/Github/OccNet/data/nuscenes/openocc_v2/scene-0001/1e19d0a5189b46f4b62aa47508f2983e/labels.npz"
occ3d_fp = "data/nuscenes/gts/scene-0001/1e19d0a5189b46f4b62aa47508f2983e/labels.npz"

openocc_data = np.load(openocc_fp)
occ3d_data = np.load(occ3d_fp)

print(openocc_data.files)
print(occ3d_data.files)

openocc_gt = openocc_data['semantics']
occ3d_gt = occ3d_data['semantics']

print(openocc_gt.shape, occ3d_gt.shape)
print(np.unique(openocc_gt))
print(np.unique(occ3d_gt))