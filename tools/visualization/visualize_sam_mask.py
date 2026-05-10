'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-29 22:47:06
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
import numpy as np
import cv2
import pycocotools.mask as maskUtils
import mmcv
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((900, 1600, 4))
    img[:,:,3] = 0
    for ann in tqdm(sorted_anns):
        m = maskUtils.decode(ann['segmentation'])
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    cv2.imwrite('test.png', img.astype(np.uint8))


file_path = 'data/nuscenes/sam_mask_json/CAM_FRONT_LEFT/n008-2018-05-21-11-06-59-0400__CAM_FRONT_LEFT__1526915267904917.json'
masks = mmcv.load(file_path)
print(masks[0].keys(), type(masks[0]['segmentation']))

show_anns(masks)