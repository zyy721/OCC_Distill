import argparse
from os import path as osp
from pathlib import Path

import mmengine
from nuscenes import NuScenes

import copy
from copy import deepcopy
import numpy as np

# from dataset.utils import get_img2global, get_lidar2global, get_lidar2cam
from nuscenes.nuscenes import NuScenes
import os

# def create_infos_w_plan(info_sparsepad_path, info_visionpad_path, info_out_path):
#     info_visionpad = mmengine.load(info_visionpad_path)
#     info_visionpad = info_visionpad['infos']

#     info_token_visionpad = {}
#     for cur_info in info_visionpad:
#         token = cur_info['token']
#         info_token_visionpad[token] = cur_info
        
#     info = mmengine.load(info_sparsepad_path)
#     for idx, cur_sample in enumerate(info['infos']):
#         token = cur_sample['token']
#         info['infos'][idx]['occ_gt_path'] = info_token_visionpad[token]['occ_gt_path']

#     mmengine.dump(info, info_out_path, 'pkl')



# def main():
#     info_train_bevdet_path = 'data/nuscenes/bevdetv2-nuscenes_infos_train.pkl'
#     info_val_bevdet_path = 'data/nuscenes/bevdetv2-nuscenes_infos_val.pkl'

#     info_train_occ_path = 'data/nuscenes/occ_infos_temporal_train.pkl'
#     info_val_occ_path = 'data/nuscenes/occ_infos_temporal_val.pkl'

#     info_train_out_path = 'data/nuscenes/bevdetv2-nuscenes_infos_train_occ.pkl'
#     info_val_out_path = 'data/nuscenes/bevdetv2-nuscenes_infos_val_occ.pkl'

#     create_infos_w_plan(info_train_bevdet_path, info_train_occ_path, info_train_out_path)
#     create_infos_w_plan(info_val_bevdet_path, info_val_occ_path, info_val_out_path)



# def create_infos_w_plan(info_sparsepad_path, info_visionpad_path, info_out_path):
#     info_visionpad = mmengine.load(info_visionpad_path)
#     info_visionpad = info_visionpad['infos']

#     info_token_visionpad = {}
#     for cur_info in info_visionpad:
#         token = cur_info['token']
#         info_token_visionpad[token] = cur_info
        
#     info = mmengine.load(info_sparsepad_path)
#     for idx, cur_sample in enumerate(info['infos']):
#         token = cur_sample['token']
#         # info['infos'][idx]['occ_gt_path'] = info_token_visionpad[token]['occ_gt_path']
#         info['infos'][idx]['ann_infos'] = info_token_visionpad[token]['ann_infos']

#     mmengine.dump(info, info_out_path, 'pkl')



# def main():
#     info_train_bevdet_path = 'data/nuscenes/nuscenes_unified_infos_train_v4.pkl'
#     info_val_bevdet_path = 'data/nuscenes/nuscenes_unified_infos_val_v4.pkl'

#     info_train_occ_path = 'data/nuscenes/bevdetv2-nuscenes_infos_train.pkl'
#     info_val_occ_path = 'data/nuscenes/bevdetv2-nuscenes_infos_val.pkl'

#     info_train_out_path = 'data/nuscenes/nuscenes_unified_infos_train_v4_ann_infos.pkl'
#     info_val_out_path = 'data/nuscenes/nuscenes_unified_infos_val_v4_ann_infos.pkl'

#     create_infos_w_plan(info_train_bevdet_path, info_train_occ_path, info_train_out_path)
#     create_infos_w_plan(info_val_bevdet_path, info_val_occ_path, info_val_out_path)


from pyquaternion import Quaternion

def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat

def create_infos_w_plan(info_out_path, info_out_path_ego):        
    info = mmengine.load(info_out_path)
    for idx, cur_sample in enumerate(info['infos']):
        lidar_to_ego = rt2mat(cur_sample['lidar2ego_translation'],
                              cur_sample['lidar2ego_rotation'])
        ego_to_lidar = np.linalg.inv(lidar_to_ego)

        ego2img_list = []
        for lidar2img in cur_sample['lidar2img']:
            ego2img = lidar2img @ ego_to_lidar 
            ego2img_list.append(ego2img)

        ego2cam_list = []
        for lidar2cam in cur_sample['lidar2cam']:
            ego2cam = lidar2cam @ ego_to_lidar 
            ego2cam_list.append(ego2cam)

        info['infos'][idx]['lidar2img'] = ego2img_list
        info['infos'][idx]['lidar2cam'] = ego2cam_list


    mmengine.dump(info, info_out_path_ego, 'pkl')



def main():
    info_train_out_path = 'data/nuscenes/nuscenes_unified_infos_train_v4_ann_infos.pkl'
    info_val_out_path = 'data/nuscenes/nuscenes_unified_infos_val_v4_ann_infos.pkl'

    info_train_out_path_ego = 'data/nuscenes/nuscenes_unified_infos_train_v4_ann_infos_ego.pkl'
    info_val_out_path_ego = 'data/nuscenes/nuscenes_unified_infos_val_v4_ann_infos_ego.pkl'

    create_infos_w_plan(info_train_out_path, info_train_out_path_ego)
    create_infos_w_plan(info_val_out_path, info_val_out_path_ego)


if __name__ == '__main__':
    main()