'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-12-05 15:24:08
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2
import datetime
import torch
from typing import Literal
import torchvision
from torchvision import transforms as T


OCC3D_PALETTE = torch.Tensor([
    [0, 0, 0],
    [255, 120, 50],  # barrier              orangey
    [255, 192, 203],  # bicycle              pink
    [255, 255, 0],  # bus                  yellow
    [0, 150, 245],  # car                  blue
    [0, 255, 255],  # construction_vehicle cyan
    [200, 180, 0],  # motorcycle           dark orange
    [255, 0, 0],  # pedestrian           red
    [255, 240, 150],  # traffic_cone         light yellow
    [135, 60, 0],  # trailer              brown
    [160, 32, 240],  # truck                purple
    [255, 0, 255],  # driveable_surface    dark pink
    [139, 137, 137], # other_flat           dark grey
    [75, 0, 75],  # sidewalk             dard purple
    [150, 240, 80],  # terrain              light green
    [230, 230, 250],  # manmade              white
    [0, 175, 0],  # vegetation           green
    [0, 255, 127],  # ego car              dark cyan
    [255, 99, 71],
    [0, 191, 255],
    [125, 125, 125]
])


class VisElement(object):
    def __init__(self, 
                 data=None, 
                 type: Literal['rgb', 'semantic', 'depth']='rgb',
                 need_denormalize=True,
                 is_sparse=False,
                 min_value=0.5,
                 max_value=55.0):
        assert type in ['rgb', 'semantic', 'depth'], \
            "The type should be in ['rgb', 'semantic', 'depth']"
        
        self.data = data
        self.type = type
        self.need_denormalize = need_denormalize
        self.is_sparse = is_sparse
        self.min_value = min_value
        self.max_value = max_value


def visualize_depth(depth, 
                    mask=None, 
                    depth_min=None, 
                    depth_max=None, 
                    direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color


def visualize_elements(inputs: VisElement,
                       target_size=[90, 160], # (H, W)
                       cam_order=[2,0,1,4,3,5],
                       save_dir=None,
                       prefix=None):
    """A more powerful and flexible visualization function for multiple types of visual elements.

    Args:
        inputs (VisElement): List of VisElement objects.
        target_size (list, optional): the target image size, HxW. Defaults to [90, 160].
        cam_order: the camera order. Defaults to [2,0,1,4,3,5].
        save_dir (str, optional): the save directory. Defaults to None.
    """

    if not isinstance(inputs, list):
        inputs = [inputs]
    
    all_vis_imgs = []
    for vis_elem in inputs:
        assert isinstance(vis_elem, VisElement), \
            "The input should be a VisElement object"

        if vis_elem.type == 'rgb':  # (num_cam, 3, H, W)
            images = vis_elem.data
            if not torch.is_tensor(images):
                images = torch.from_numpy(images)
            images = images.detach().cpu()
            
            # reorder the camera order
            images = images[cam_order]

            ## 2) Denormalize the image
            assert images.shape[1] == 3, "The image should be in RGB format"
            if vis_elem.need_denormalize:
                img_mean = torch.Tensor([123.675, 116.28, 103.53])[None, :, None, None]
                img_std = torch.Tensor([58.395, 57.12, 57.375])[None, :, None, None]
                visual_imgs = images * img_std + img_mean
                visual_imgs = visual_imgs[:, [2, 1, 0], ...]  # BGR to RGB
            else:
                visual_imgs = images * 255.0
                visual_imgs = visual_imgs[:, [2, 1, 0], ...]  # BGR to RGB

            visual_imgs = T.functional.resize(visual_imgs, target_size)

            all_vis_imgs.append(visual_imgs)
        elif vis_elem.type == 'semantic':
            semantic = vis_elem.data  # (N, H, W, 3)

            if not torch.is_tensor(semantic):
                semantic = torch.from_numpy(semantic)
            else:
                semantic = semantic.detach().cpu()
            
            semantic = semantic[cam_order]

            if vis_elem.is_sparse:
                # visualize sparse semantic map, for example, projected from the LiDAR segmentation
                concated_semantic_maps = []
                for _sem in semantic:  # loop over each camera
                    semantic_rgb = np.ones((_sem.shape[0], _sem.shape[1], 3)) * 255

                    valid_sem = _sem != 255
                    sem_color_val = OCC3D_PALETTE[_sem[valid_sem].reshape(-1)].numpy().astype(np.uint8)

                    coord = np.where(valid_sem)
                    for idx, color in enumerate(sem_color_val):
                        cv2.circle(semantic_rgb, 
                                (coord[1][idx], coord[0][idx]), 
                                1, color.tolist(), -1)

                    concated_semantic_maps.append(semantic_rgb)
                semantic_img = np.stack(concated_semantic_maps, axis=0)
                semantic_img = torch.from_numpy(semantic_img).permute(0, 3, 1, 2)
            else:
                if semantic.shape[1] != 3:
                    semantic = semantic.permute(0, 3, 1, 2)
                semantic_img = T.functional.resize(semantic, target_size)

            all_vis_imgs.append(semantic_img)
        elif vis_elem.type == 'depth':
            from mmdet3d.utils.turbo_cmap import (turbo_colormap_data, 
                                                  normalize_depth, 
                                                  interpolate_or_clip)

            depth = vis_elem.data  # (N, H, W)

            if torch.is_tensor(depth):
                depth = depth.detach().cpu().numpy()
            
            depth = depth[cam_order]

            if vis_elem.is_sparse:
                # specify the min and max depth value for normalization
                min_depth = vis_elem.min_value
                max_depth = vis_elem.max_value

                normalized_depth = normalize_depth(
                    depth, d_min=min_depth, d_max=max_depth)  # (N, H, W)
                
                concated_depth_maps = []
                for _depth in normalized_depth:
                    depth_rgb = np.ones((_depth.shape[0], _depth.shape[1], 3)) * 255

                    valid_depth = _depth > 0.0  # the valid depth mask
                    coord = np.where(valid_depth)
                    for idx, depth_val in enumerate(_depth[valid_depth]):
                        depth_color = interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val)
                        depth_color = (np.array(depth_color) * 255).astype(np.uint8)
                        cv2.circle(depth_rgb, 
                                (coord[1][idx], coord[0][idx]), 
                                1, depth_color.tolist(), -1)

                    concated_depth_maps.append(depth_rgb)
                
                depth_img = np.stack(concated_depth_maps, axis=0)
                depth_img = torch.from_numpy(depth_img).permute(0, 3, 1, 2)
            else:
                concated_depth_maps = []
                for b in range(len(depth)):
                    pred_depth_color = visualize_depth(depth[b], direct=False)
                    concated_depth_maps.append(
                            cv2.resize(pred_depth_color, (target_size[1], target_size[0])))
                depth = np.stack(concated_depth_maps, axis=0)[..., ::-1].copy()
                depth_img = torch.from_numpy(depth).permute(0, 3, 1, 2)  # to (b, 3, h, w)
            
            all_vis_imgs.append(depth_img)
        else:
            raise ValueError(f"Unknown type {vis_elem.type}")
        
    ## Start saving
    final_imgs = torch.cat(all_vis_imgs, dim=0)

    os.makedirs(save_dir, exist_ok=True)
    file_name = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if prefix is not None:
        file_name = f"{prefix}_{file_name}"
    full_img_path = osp.join(save_dir, f'{file_name}.png')
    # need transform the image from [0, 255] into [0, 1] with RGB format
    torchvision.utils.save_image(final_imgs / 255.0, full_img_path, 
                                 nrow=3, padding=3, pad_value=1.0)