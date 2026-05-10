'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-26 21:06:33
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
import mmcv


def load_pickle():
    """Load the nuscenes dataset pickle file, and explore the
    infos contents.

    Args:
        pickle_file_path (_type_): _description_
    """
    pickle_file_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    # pickle_file_path = "data/nuscenes/bevdetv2-nuscenes_infos_train_openocc.pkl"

    with open(pickle_file_path, "rb") as fp:
        dataset = pickle.load(fp)
    
    print(type(dataset))
    print(dataset.keys())
    
    data_infos = dataset['infos']
    print(len(data_infos))

    idx = 100
    info = data_infos[idx]
    print(info.keys())

    ## check the camera keys
    print(info['cams'].keys())


def get_scene_sequence_data(data_infos):
    scene_name_list = []
    total_scene_seq = []
    curr_seq = []
    for idx, data in enumerate(data_infos):
        scene_token = data['scene_token']
        next_idx = min(idx + 1, len(data_infos) - 1)
        next_scene_token = data_infos[next_idx]['scene_token']

        curr_seq.append(data)

        if next_scene_token != scene_token:
            total_scene_seq.append(curr_seq)
            scene_name_list.append(scene_token)
            curr_seq = []

    total_scene_seq.append(curr_seq)
    scene_name_list.append(scene_token)
    return scene_name_list, total_scene_seq


def create_part_pickle(pickle_path, ratio=0.25):
    """Create part of the pickle file to accelerate the training.
    Here we keep the partial data for each scene.

    Args:
        pickle_path (_type_): _description_
        ratio (float, optional): _description_. Defaults to 0.25.
    """
    with open(pickle_path, "rb") as fp:
        nuscenes_infos = pickle.load(fp)

    metadata = nuscenes_infos['metadata']
    infos = nuscenes_infos['infos']

    infos = list(sorted(infos, key=lambda e: e['timestamp']))

    scene_name_list, total_scene_seq = get_scene_sequence_data(infos)

    new_infos = []
    for idx, (scene_name, scene_seq) in tqdm(enumerate(zip(scene_name_list, total_scene_seq)),
                                             total=len(scene_name_list)):
        keep_length = int(len(scene_seq) * ratio)
        new_infos.extend(scene_seq[:keep_length])

    print(len(new_infos))
    data = dict(infos=new_infos, metadata=metadata)

    ## start saving the pickle file
    filename, ext = osp.splitext(osp.basename(pickle_path))
    root_path = "./data/nuscenes"
    new_filename = f"{filename}-quarter{ext}"
    info_path = osp.join(root_path, new_filename)
    print(f"The results would be saved into {info_path}")
    mmcv.dump(data, info_path)


def create_instance_pickle():
    version = 'v1.0-trainval'
    dataroot = 'data/nuscenes/'
    nuscenes = NuScenes(version, dataroot)

    for set in ['train', 'val']:
        dataset = pickle.load(
            open('data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_%s.pkl' % set, 'rb'))
        
        infos = []
        for info in tqdm(dataset['infos']):
            token = info['token']
            curr_sample = nuscenes.get('sample', token)
            lidar_data = nuscenes.get('sample_data', curr_sample['data']['LIDAR_TOP'])
            lidar_path, boxes, _ = nuscenes.get_sample_data(lidar_data['token'])
            boxes_token = [box.token for box in boxes]
            instance_tokens = [nuscenes.get(
                'sample_annotation', box_token)['instance_token'] for box_token in boxes_token]
            info['instance_tokens'] = instance_tokens
            infos.append(info)
        
        dataset['infos'] = infos
        
        print(len(dataset['infos']))
        with open('data/nuscenes/bevdetv3-inst-nuscenes_infos_%s.pkl' % set, 'wb') as fid:
            pickle.dump(dataset, fid)


def create_instance_pickle_with_corners():
    version = 'v1.0-trainval'
    dataroot = 'data/nuscenes/'
    nuscenes = NuScenes(version, dataroot)

    for set in ['train', 'val']:
        dataset = pickle.load(
            open('data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_%s.pkl' % set, 'rb'))
        
        infos = []
        for info in tqdm(dataset['infos']):
            token = info['token']
            curr_sample = nuscenes.get('sample', token)
            lidar_data = nuscenes.get('sample_data', curr_sample['data']['LIDAR_TOP'])
            lidar_path, boxes, _ = nuscenes.get_sample_data(lidar_data['token'])
            boxes_token = [box.token for box in boxes]
            instance_tokens = [nuscenes.get(
                'sample_annotation', box_token)['instance_token'] for box_token in boxes_token]
            info['instance_tokens'] = instance_tokens

            # obtain the bboxes in camera from the nerfstudio
            for camera in info['cams']:
                camera_token = curr_sample['data'][camera]

                _, boxes, _ = nuscenes.get_sample_data(
                    camera_token, box_vis_level=BoxVisibility.ANY)

                all_box_corners = [box.corners() for box in boxes]
                if len(all_box_corners) > 0:
                    all_box_corners = np.stack(all_box_corners)
                info['cams'][camera]['box_corners'] = all_box_corners

            infos.append(info)
        
        dataset['infos'] = infos
        
        print(len(dataset['infos']))
        with open('data/nuscenes/bevdetv3-inst-nuscenes_infos_%s.pkl' % set, 'wb') as fid:
            pickle.dump(dataset, fid)


def update_infos(split=['train', 'val'],
                 add_occ_path=True,
                 add_scene_token=True,
                 add_lidarseg_path=False,
                 add_transforms=False):
    """Add different extra needed information in the nuScenes dataset to the pickle file.

    Args:
        split (list, optional): _description_. Defaults to ['train', 'val'].
        add_occ_path: whether to add the occ path to the pickle file.
        add_scene_token: whether to add the scene token to the pickle file.
        add_lidarseg_path: whether to add the lidarseg path to the pickle file.
    """
    import mmengine
    
    if not isinstance(split, list):
        split = [split]
    
    nuscenes_version = 'v1.0-trainval'
    dataroot = './data/nuscenes/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    
    for s in split:
        assert s in ['train', 'val']
        pickle_file = f'data/nuscenes/nuscenes_unified_infos_{s}_v3.pkl'
        print(f'Start process {pickle_file}...')

        data = mmengine.load(pickle_file)
        data_infos = data['infos']

        for info in mmcv.track_iter_progress(data_infos):
            sample = nuscenes.get('sample', info['token'])
            scene = nuscenes.get('scene', sample['scene_token'])
            
            if add_occ_path:
                info['occ_path'] = \
                    './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
            if add_scene_token:
                info['scene_token'] = sample['scene_token']

            if add_lidarseg_path:
                lidar_token = sample['data']['LIDAR_TOP']
                lidarseg_label = os.path.join(nuscenes.dataroot, 
                                              nuscenes.get('lidarseg', lidar_token)['filename'])
                info['lidarseg'] = lidarseg_label

            if add_transforms:
                image_paths = []
                lidar2img_rts = []
                # add lidar2img matrix
                lidar2cam_rts = []
                cam_intrinsics = []
                cam2camego_list = []
                camego2global_list = []
                for cam_type, cam_info in info["cams"].items():
                    image_paths.append(cam_info["data_path"])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                    lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info["cam_intrinsic"]
                    viewpad = np.eye(4)
                    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = viewpad @ lidar2cam_rt.T
                    lidar2img_rts.append(lidar2img_rt)
                    lidar2cam_rts.append(lidar2cam_rt.T)
                    cam_intrinsics.append(viewpad)

                    # obtain the camera to ego transformation matrix
                    cam2camego = np.eye(4, dtype=np.float32)
                    cam2camego[:3, :3] = Quaternion(
                        cam_info['sensor2ego_rotation']).rotation_matrix
                    cam2camego[:3, 3] = cam_info['sensor2ego_translation']
                    cam2camego_list.append(cam2camego)

                    # obtain the ego to global transformation matrix
                    ego2global = np.eye(4, dtype=np.float32)
                    ego2global[:3, :3] = Quaternion(
                        cam_info['ego2global_rotation']).rotation_matrix
                    ego2global[:3, 3] = cam_info['ego2global_translation']
                    camego2global_list.append(ego2global)

                info.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                        lidar2cam=lidar2cam_rts,
                        cam_intrinsic=cam_intrinsics,
                        cam2camego=cam2camego_list,
                        camego2global=camego2global_list,
                    )
                )
        
        ## start saving the pickle file
        filename, ext = osp.splitext(osp.basename(pickle_file))
        root_path = "./data/nuscenes"
        # new_filename = f"{filename}_v2{ext}"
        new_filename = filename.replace('_v3', '_v4') + ext
        info_path = osp.join(root_path, new_filename)
        print(f"The results will be saved into {info_path}")
        mmcv.dump(data, info_path)


def merge_infos(split=['train', 'val']):
    """Merge the infos from the UniPAD"""
    import mmengine
    
    if not isinstance(split, list):
        split = [split]
    
    for s in split:
        assert s in ['train', 'val']
        bevdet_pickle_file = f'./data/nuscenes/bevdetv2-nuscenes_infos_{s}.pkl'
        print(f'Start process {bevdet_pickle_file}...')

        data = mmengine.load(bevdet_pickle_file)
        bevdet_data_infos = data['infos']

        unipad_pickle_file = f'/mnt/data2/zhanghm/Code/Occupancy/Uni3DGS/data/nuscenes/nuscenes_unified_infos_{s}_v4_ego.pkl'
        print(f'Start process {unipad_pickle_file}...')
        unipad_data = mmengine.load(unipad_pickle_file)
        unipad_data_infos = unipad_data['infos']

        assert len(bevdet_data_infos) == len(unipad_data_infos)
        
        bevdet_camera_idx = [2, 0, 1, 4, 3, 5]
        for bevdet_info, unipad_info in tqdm(zip(bevdet_data_infos, unipad_data_infos)):
            if bevdet_info['token'] != unipad_info['token']:
                print(f"Error: {bevdet_info['token']} != {unipad_info['token']}")
                continue
            bevdet_info['lidar2img'] = [unipad_info['lidar2img'][i] for i in bevdet_camera_idx]
            bevdet_info['lidar2cam'] = [unipad_info['lidar2cam'][i] for i in bevdet_camera_idx]
            bevdet_info['cam_intrinsic'] = [unipad_info['cam_intrinsic'][i] for i in bevdet_camera_idx]
            bevdet_info['cam2camego'] = [unipad_info['cam2camego'][i] for i in bevdet_camera_idx]
            bevdet_info['camego2global'] = [unipad_info['camego2global'][i] for i in bevdet_camera_idx]
            bevdet_info['prev'] = unipad_info['prev']
            bevdet_info['next'] = unipad_info['next']
        
        ## start saving the pickle file
        bevdet_pickle_file = f'./data/nuscenes/bevdetv2-nuscenes_infos_{s}.pkl'
        filename, ext = osp.splitext(osp.basename(bevdet_pickle_file))
        root_path = "./data/nuscenes"
        # new_filename = f"{filename}_v2{ext}"
        new_filename = filename + "_visionpad" + ext
        info_path = osp.join(root_path, new_filename)
        print(f"The results will be saved into {info_path}")
        mmcv.dump(data, info_path)


if __name__ == '__main__':
    load_pickle()
    exit()
    merge_infos(['train', 'val'])
    exit()
    create_part_pickle("data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_train.pkl")
    exit()
    
    exit()
    create_instance_pickle_with_corners()
