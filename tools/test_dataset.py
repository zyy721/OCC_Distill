'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-12-05 14:31:05
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the dataset pipeline.
'''
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import pickle
import torch
import argparse
import warnings
from os import path as osp
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config, DictAction, mkdir_or_exist
from einops import rearrange

from mmdet3d.datasets import build_dataset
from mmdet3d.models.utils.vis_utils import VisElement, visualize_elements


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['Normalize'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--work-dir',
        default=None,
        type=str,
        help='The work directory')
    parser.add_argument(
        '--task',
        type=str,
        choices=['det', 'seg', 'multi_modality-det', 'mono-det'],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--aug',
        action='store_true',
        help='Whether to visualize augmented datasets or original dataset.')
    parser.add_argument(
        '--online',
        action='store_true',
        help='Whether to perform online visualization. Note that you often '
        'need a monitor to do so.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def unpack_imgs(imgs, num_cams=6):
    # split the inputs into each frame
    B, N, C, H, W = imgs.shape
    num_frame = N // num_cams
    imgs = imgs.view(B, num_cams, num_frame, C, H, W)
    # imgs = torch.split(imgs, 1, 2)
    # imgs = [t.squeeze(2) for t in imgs]
    imgs = rearrange(imgs, 'B N F C H W -> F B N C H W')

    return imgs


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    dataset = build_dataset(cfg.data.train)
    print(len(dataset), type(dataset))

    data = dataset[0]
    print(data.keys())

    multi_adj_frame_id_cfg = (1, 1 + 1, 1)
    num_adj=len(range(*multi_adj_frame_id_cfg))
    print(num_adj)

    # visualize the input images and processed images
    img_inputs = data['img_inputs'][0].unsqueeze(0)
    print(img_inputs.shape)

    # num_frame = 3
    # # split the inputs into each frame
    # B, N, C, H, W = img_inputs.shape
    # N = N // num_frame
    # imgs = img_inputs.view(B, N, num_frame, C, H, W)
    # imgs = torch.split(imgs, 1, 2)
    # imgs = [t.squeeze(2) for t in imgs]
    # print(len(imgs), imgs[0].shape)

    imgs = unpack_imgs(img_inputs)
    print(len(imgs), imgs[0].shape)

    curr_img_inputs = imgs[0]
    print(curr_img_inputs.min(), curr_img_inputs.max())

    render_gt_img = data['render_gt_img'].unsqueeze(0)  # add batch dimension
    print(render_gt_img.shape, 'render_gt_img')

    render_gt_img = unpack_imgs(render_gt_img)
    curr_render_gt_img = render_gt_img[0]
    print(curr_render_gt_img.shape, 'curr_render_gt_img')

    target_imgs = torch.stack(data['target_imgs']).unsqueeze(0)
    print(target_imgs.shape, 'target_imgs')  # (0, 1)

    save_dir = "./results/ICCV"
    target_size = (curr_img_inputs.shape[-2], curr_img_inputs.shape[-1])  # (H, W)
    visualize_elements(
        [
            VisElement(
                curr_img_inputs[0],
                type='rgb'
            ),
            VisElement(
                curr_render_gt_img[0],
                type='rgb'
            ),
            VisElement(
                target_imgs[0],
                need_denormalize=False,
                type='rgb'
            )
        ],
        target_size=target_size,
        save_dir=save_dir,
        cam_order=list(range(6))
    )


if __name__ == '__main__':
    main()
