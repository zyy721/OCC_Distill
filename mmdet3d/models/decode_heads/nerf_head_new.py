import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import argparse
import mmcv
import pycocotools.mask as maskUtils
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

NUSCENSE_LIDARSEG_PALETTE = [
    (0, 0, 0),  # noise
    (112, 128, 144),  # barrier
    (220, 20, 60),  # bicycle
    (255, 127, 80),  # bus
    (255, 158, 0),  # car
    (233, 150, 70),  # construction_vehicle
    (255, 61, 99),  # motorcycle
    (0, 0, 230),  # pedestrian
    (47, 79, 79),  # traffic_cone
    (255, 140, 0),  # trailer
    (255, 99, 71),  # Tomato
    (0, 207, 191),  # nuTonomy green
    (175, 0, 75),
    (75, 0, 75),
    (112, 180, 60),
    (222, 184, 135),  # Burlywood
    (0, 175, 0),
    (255, 255, 255),
]

learning_map = {
    1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 19: 0, 20: 0, 0: 0, 29: 0, 31: 0, 9: 1, 14: 2, 15: 3, 16: 3,
    17: 4, 18: 5, 21: 6, 2: 7, 3: 7, 4: 7, 6: 7, 12: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    30: 16
}

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--data_dir', help='specify the root path of images and masks')
    parser.add_argument('--save_img', default=False, action='store_true', help='whether to save annotated images')
    parser.add_argument('--visualize', default=False, action='store_true', help='whether to save annotated images')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--sam', default=False, action='store_true',
                        help='use SAM but not given annotation json, default is False')
    parser.add_argument('--camera', default='CAM_FRONT',
                        choices=['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                                 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'], help='which camera should be selected')
    parser.add_argument('--ckpt_path', default='ckp/sam_vit_h_4b8939.pth',
                        help='specify the root path of SAM checkpoint')
    parser.add_argument('--light_mode', default=False, action='store_true', help='use light mode')
    args = parser.parse_args()
    return args


def generate_sparse_depth_with_semantic(nus, args, sample, vis=False):
    token = sample['token']
    rec = nus.get('sample', token)

    # load lidarseg data
    lidarseg_path = os.path.join(args.data_dir,
                                 nus.get('lidarseg', nus.get('sample', token)['data']['LIDAR_TOP'])['filename'])
    lidarseg = np.fromfile(lidarseg_path, dtype=np.uint8)

    lidarseg = np.vectorize(learning_map.__getitem__)(lidarseg)
    lidarseg = torch.Tensor(lidarseg).long()

    lidar_sample = nus.get(
        'sample_data', rec['data']['LIDAR_TOP'])
    lidar_pose = nus.get(
        'ego_pose', lidar_sample['ego_pose_token'])

    lidar_rotation = Quaternion(lidar_pose['rotation'])
    lidar_translation = np.array(lidar_pose['translation'])[:, None]
    lidar_to_world = np.vstack([
        np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
        np.array([0, 0, 0, 1])
    ])

    # get lidar points
    lidar_file = os.path.join(
        args.data_dir, lidar_sample['filename'])
    lidar_points = np.fromfile(lidar_file, dtype=np.float32)
    # lidar data is stored as (x, y, z, intensity, ring index).
    lidar_points = lidar_points.reshape(-1, 5)[:, :4]

    # lidar points ==> ego frame
    sensor_sample = nus.get(
        'calibrated_sensor', lidar_sample['calibrated_sensor_token'])
    lidar_to_ego_lidar_rot = Quaternion(
        sensor_sample['rotation']).rotation_matrix
    lidar_to_ego_lidar_trans = np.array(
        sensor_sample['translation']).reshape(1, 3)

    ego_lidar_points = np.dot(
        lidar_points[:, :3], lidar_to_ego_lidar_rot.T)
    ego_lidar_points += lidar_to_ego_lidar_trans

    homo_ego_lidar_points = np.concatenate(
        (ego_lidar_points, np.ones((ego_lidar_points.shape[0], 1))), axis=1)

    homo_ego_lidar_points = torch.from_numpy(
        homo_ego_lidar_points).float()

    ###### Camera ######
    # for cam in camera_names:
    camera_sample = nus.get(
        'sample_data', rec['data'][args.camera])

    car_egopose = nus.get('ego_pose', camera_sample['ego_pose_token'])
    egopose_rotation = Quaternion(car_egopose['rotation']).inverse
    egopose_translation = - np.array(car_egopose['translation'])[:, None]
    world_to_car_egopose = np.vstack([
        np.hstack((egopose_rotation.rotation_matrix,
                   egopose_rotation.rotation_matrix @ egopose_translation)),
        np.array([0, 0, 0, 1])
    ])

    # From egopose to sensor
    sensor_sample = nus.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
    intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
    sensor_rotation = Quaternion(sensor_sample['rotation'])
    sensor_translation = np.array(sensor_sample['translation'])[:, None]
    car_egopose_to_sensor = np.vstack([
        np.hstack(
            (sensor_rotation.rotation_matrix, sensor_translation)),
        np.array([0, 0, 0, 1])
    ])
    car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

    # Combine all the transformation.
    # From sensor to lidar.
    lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
    lidar_to_sensor = torch.from_numpy(lidar_to_sensor).float()

    # load image for debugging
    image_filename = os.path.join(
        args.data_dir, camera_sample['filename'])
    img = Image.open(image_filename)
    img = np.array(img)

    ###### Transformation ######
    height, width = img.shape[:2]
    sparse_depth = torch.zeros((height, width))
    sparse_semantic = torch.zeros((height, width)).long()

    # Ego(lidar) ==> Camera
    camera_points = torch.mm(homo_ego_lidar_points, lidar_to_sensor.t())
    # depth > 0
    depth_mask = camera_points[:, 2] > 0
    camera_points = camera_points[depth_mask]
    camera_sem = lidarseg[depth_mask]

    # Camera ==> Pixel
    viewpad = torch.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    pixel_points = torch.mm(camera_points, viewpad.t())[:, :3]
    pixel_points[:, :2] = pixel_points[:, :2] / \
                          pixel_points[:, 2:3]

    pixel_uv = pixel_points[:, :2].round().long()
    valid_mask = (pixel_uv[:, 0] >= 0) & (pixel_uv[:, 0] <= width - 1) & (pixel_uv[:, 1] >= 0) & (
            pixel_uv[:, 1] <= height - 1)

    valid_pixel_uv = pixel_uv[valid_mask]
    valid_depth = camera_points[..., 2][valid_mask]
    valid_sem = camera_sem[valid_mask]

    sparse_depth[valid_pixel_uv[:, 1], valid_pixel_uv[:, 0]] = valid_depth
    sparse_semantic[valid_pixel_uv[:, 1], valid_pixel_uv[:, 0]] = valid_sem
    sparse_depth = sparse_depth.numpy()
    sparse_semantic = sparse_semantic.numpy()

    if vis:
        plt.imshow(img)
        plt.scatter(
            valid_pixel_uv[:, 0],
            valid_pixel_uv[:, 1],
            c=torch.Tensor(NUSCENSE_LIDARSEG_PALETTE)[valid_sem] / 255,
            alpha=0.5, s=0.5)
        plt.axis('off')
        plt.show()

    return img, sparse_depth, sparse_semantic

def main(rank, args, nus):
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    sam = sam_model_registry["vit_h"](checkpoint=args.ckpt_path).to(rank)
    if args.light_mode:
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0,  # 1 by default
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
            output_mode='coco_rle',
        )
    else:
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0,  # 1 by default
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
            output_mode='coco_rle',
        )

    samples = nus.sample

    if rank == 0:
        print('Total number of files: ', len(samples))
    samples = samples[
                      (len(samples) // args.world_size + 1) * rank: (len(samples) // args.world_size + 1) * (rank + 1)]
    total_samples = len(samples)

    for idx, sample in enumerate(samples):
        img, sparse_depth, sparse_semantic = generate_sparse_depth_with_semantic(nus, args, sample, vis=args.visualize)
        token = sample['token']
        rec = nus.get('sample', token)
        camera_sample = nus.get(
            'sample_data', rec['data'][args.camera])
        image_filename = os.path.join(
            args.data_dir, camera_sample['filename'])
        file_name = image_filename.split('/')[-1].replace('.jpg', '.json')


        mask_path = os.path.join(args.data_dir, 'sam_masks', args.camera)
        semantic_path = os.path.join(args.data_dir, 'sam_semantics', args.camera)
        os.makedirs(mask_path, exist_ok=True)
        os.makedirs(semantic_path, exist_ok=True)

        if os.path.exists(os.path.join(semantic_path, file_name.replace('.json', '.png'))) and args.save_img:
            print('{}/{} Already existed {} {}'.format(idx, total_samples, args.camera, file_name))
            continue
        else:
            print('{}/{} Processing {} {}'.format(idx, total_samples, args.camera, file_name))

        with torch.no_grad():
            if mask_generator is None:
                anns = mmcv.load(os.path.join(mask_path, file_name.replace('.jpg', '.json')))
            else:
                anns = mask_generator.generate(img)
                mmcv.dump(anns, os.path.join(mask_path, file_name.replace('.jpg', '.json')))

        semantic_image = np.zeros_like(sparse_semantic)
        for ann in anns:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            mask_label = sparse_semantic[valid_mask]
            mask_label = mask_label[mask_label > 0]
            if len(mask_label) == 0:
                continue
            top_label = np.argmax(np.bincount(mask_label))
            semantic_image[valid_mask] = top_label

        color_sem = np.array(NUSCENSE_LIDARSEG_PALETTE)[semantic_image]
        if args.visualize:
            plt.imshow(color_sem)
            plt.axis('off')
            plt.show()

        if args.save_img:
            output_file = os.path.join(semantic_path, file_name.replace('.json', '_visual.png'))
            image = Image.fromarray(color_sem.astype(np.uint8))
            image.save(output_file)

            output_file = os.path.join(semantic_path, file_name.replace('.json', '.png'))
            image = Image.fromarray(semantic_image.astype(np.uint8))
            image.save(output_file)

if __name__ == '__main__':
    args = parse_args()

    nuscenes_version = 'v1.0-trainval'
    nus = NuScenes(nuscenes_version, args.data_dir, verbose=True)

    if args.world_size > 1:
        mp.spawn(main, args=(args, nus,), nprocs=args.world_size, join=True)
    else:
        main(0, args, nus)
