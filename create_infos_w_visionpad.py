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


# This order must match ``data_config['cams']`` used by BEVDet.  Do not rely
# on the insertion order of ``info['cams']``: the nuScenes converter stores
# the cameras in a different order.
CAMERA_ORDER = (
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
)

# BEVDet uses ego2globals[:, 0, 0] as the key ego.  With CAMERA_ORDER above,
# the BEV/voxel feature is therefore expressed in CAM_FRONT_LEFT-time ego.
KEY_EGO_CAMERA = CAMERA_ORDER[0]


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


def _pad_intrinsic(cam_intrinsic):
    """Convert a 3x3 camera intrinsic matrix to homogeneous 4x4 form."""
    cam_intrinsic = np.asarray(cam_intrinsic)
    viewpad = np.eye(4, dtype=cam_intrinsic.dtype)
    viewpad[:cam_intrinsic.shape[0], :cam_intrinsic.shape[1]] = cam_intrinsic
    return viewpad


def build_keyego_camera_fields(cur_sample,
                               camera_order=CAMERA_ORDER,
                               key_ego_camera=KEY_EGO_CAMERA):
    """Build ordered camera matrices in the BEV key-ego coordinate frame.

    For camera ``i``, the stored extrinsic is

        T_cam_i_from_keyego =
            inv(T_global_from_camego_i @ T_camego_i_from_cam_i)
            @ T_global_from_keyego

    This accounts for the different nuScenes sample-data timestamps.  The
    legacy field names ``lidar2cam`` and ``lidar2img`` are retained because
    downstream VisionPAD code consumes them, but their source frame is the
    CAM_FRONT_LEFT-time key ego, not LiDAR.
    """
    cams = cur_sample['cams']
    missing_cams = [cam_name for cam_name in camera_order
                    if cam_name not in cams]
    if missing_cams:
        raise KeyError(
            f"Sample {cur_sample.get('token')} misses cameras: {missing_cams}")
    if key_ego_camera not in camera_order:
        raise ValueError(
            f'key ego camera {key_ego_camera} is absent from camera_order')

    key_cam_info = cams[key_ego_camera]
    keyego2global = rt2mat(
        key_cam_info['ego2global_translation'],
        key_cam_info['ego2global_rotation'])

    image_paths = []
    keyego2img_list = []
    keyego2cam_list = []
    cam_intrinsic_list = []
    cam2camego_list = []
    camego2global_list = []

    for cam_name in camera_order:
        cam_info = cams[cam_name]
        cam2camego = rt2mat(
            cam_info['sensor2ego_translation'],
            cam_info['sensor2ego_rotation'])
        camego2global = rt2mat(
            cam_info['ego2global_translation'],
            cam_info['ego2global_rotation'])
        cam2global = camego2global @ cam2camego
        keyego2cam = np.linalg.solve(cam2global, keyego2global)
        cam_intrinsic = _pad_intrinsic(cam_info['cam_intrinsic'])

        image_paths.append(cam_info['data_path'])
        keyego2img_list.append(cam_intrinsic @ keyego2cam)
        keyego2cam_list.append(keyego2cam)
        cam_intrinsic_list.append(cam_intrinsic)
        cam2camego_list.append(cam2camego)
        camego2global_list.append(camego2global)

    # For the key camera both ego poses are the same, so this identity is a
    # useful guard against matrix-direction or reference-frame mistakes.
    key_cam_idx = camera_order.index(key_ego_camera)
    expected_keyego2cam = np.linalg.inv(cam2camego_list[key_cam_idx])
    if not np.allclose(keyego2cam_list[key_cam_idx],
                       expected_keyego2cam,
                       rtol=1e-6,
                       atol=1e-6):
        raise AssertionError(
            f'Invalid key-ego transform for sample {cur_sample.get("token")}')

    return dict(
        cam_names=list(camera_order),
        img_filename=image_paths,
        lidar2img=keyego2img_list,
        lidar2cam=keyego2cam_list,
        cam_intrinsic=cam_intrinsic_list,
        cam2camego=cam2camego_list,
        camego2global=camego2global_list,
    )


def create_infos_w_plan(info_out_path, info_out_path_ego):
    info = mmengine.load(info_out_path)
    metadata = info.setdefault('metadata', {})
    metadata['visionpad_camera_order'] = list(CAMERA_ORDER)
    metadata['visionpad_ego_reference'] = KEY_EGO_CAMERA

    for cur_sample in info['infos']:
        cur_sample.update(build_keyego_camera_fields(cur_sample))

    mmengine.dump(info, info_out_path_ego, 'pkl')



def main():
    info_train_out_path = 'data/nuscenes/nuscenes_unified_infos_train_v4_ann_infos.pkl'
    info_val_out_path = 'data/nuscenes/nuscenes_unified_infos_val_v4_ann_infos.pkl'

    info_train_out_path_ego = (
        'data/nuscenes/'
        'nuscenes_unified_infos_train_v4_ann_infos_fl_keyego.pkl')
    info_val_out_path_ego = (
        'data/nuscenes/'
        'nuscenes_unified_infos_val_v4_ann_infos_fl_keyego.pkl')

    create_infos_w_plan(info_train_out_path, info_train_out_path_ego)
    create_infos_w_plan(info_val_out_path, info_val_out_path_ego)


if __name__ == '__main__':
    main()
