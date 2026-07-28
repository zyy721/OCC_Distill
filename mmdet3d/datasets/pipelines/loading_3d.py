'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-11-13 01:17:44
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import pickle
import torch
from PIL import Image
from mmcv.image.photometric import imnormalize
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor

from ..builder import PIPELINES
from .loading import PrepareImageInputsForNeRF


@PIPELINES.register_module()
class PrepareImageInputsForVisionPAD(PrepareImageInputsForNeRF):
    """_summary_

    Args:
        input_size (_type_): the size of the image for SSL.
    """
    def __init__(self, 
                 input_size,
                 render_size,
                 load_future_img=False,
                 load_prev_img=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.input_size = input_size  # (h, w), this is the depth ssl size
        self.render_size = render_size

        self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        self.load_prev_img = load_prev_img
        self.load_future_img = load_future_img

    def get_post_matrix(self):
        H, W = self.data_config['src_size']
        img_non_aug = self.sample_noaugmentation(H=H, W=W)
        
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        _, post_rots, post_trans = \
            self.img_transform(None, post_rot,
                               post_tran,
                               resize=img_non_aug[0],
                               resize_dims=img_non_aug[1],
                               crop=img_non_aug[2],
                               flip=img_non_aug[3],
                               rotate=img_non_aug[4])
        post_mat = torch.eye(4)
        post_mat[:2, :2] = post_rots[:2, :2]
        post_mat[:2, 2] = post_trans

        ## process the intrinsic matrix
        origin_h, origin_w = self.data_config['input_size']
        h, w = self.render_size[0], self.render_size[1]
        post_mat[0] *= w / origin_w
        post_mat[1] *= h / origin_h
        return post_mat

    def _load_image(self, filename):
        img = Image.open(filename)

        img_non_aug = self.sample_noaugmentation(
            H=img.height, W=img.width)
        
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)  # 0.44 (704, 396) (0, 12, 704, 396) False 0.0
        img, post_rot2_wo_aug, post_tran2_wo_aug = \
            self.img_transform(img, post_rot,
                               post_tran,
                               resize=img_non_aug[0],
                               resize_dims=img_non_aug[1],
                               crop=img_non_aug[2],
                               flip=img_non_aug[3],
                               rotate=img_non_aug[4])
        
        return img, post_rot2_wo_aug, post_tran2_wo_aug
    
    def transform_core(self, 
                       img, 
                       img_size=None,  # (h, w)
                       to_rgb=True):
        ## we need [0, 1] images in RGB order
        # try:
        #     img = img.resize(img_size[::-1], resample=None)
        # except (TypeError, ImportError):
        #     img = img.resize(img_size[::-1])

        img = img.resize(img_size[::-1], resample=None)
        img = imnormalize(np.array(img), self.mean, self.std, to_rgb)
        return img
    
    def prepare_data(self, results):
        output = dict()
        # read the current frame as target images
        img_ori = results['img_ori']  # without normalization, without augmentation
        ## resize the image
        imgs = [
            self.transform_core(img, self.input_size)
            for img in img_ori
        ]

        if 'K' in results and results['K'] is not None:
            # first convert to the BEVDet input size
            post_rot = torch.stack(results['post_rots']) # (2, 2)
            post_trans = torch.stack(results['post_trans']) # (2, 1)
            K = torch.as_tensor(results['K'], dtype=torch.float32)
            post_mat = torch.eye(4).repeat(K.shape[0], 1, 1)
            post_mat[:, :2, :2] = post_rot[:, :2, :2]
            post_mat[:, :2, 2] = post_trans[:, :2]
            temp_K = post_mat @ K

            ## process the intrinsic matrix
            ori_shape = np.array(img_ori[0]).shape
            origin_h, origin_w = ori_shape[0], ori_shape[1]
            h, w = self.input_size[0], self.input_size[1]
            temp_K[:, 0] *= w / origin_w
            temp_K[:, 1] *= h / origin_h
            output['K'] = temp_K
            output['inv_K'] = torch.pinverse(temp_K)

        # process multiple imgs in single frame
        imgs = [img.transpose(2, 0, 1) for img in imgs]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        output['target_imgs'] = DC(to_tensor(imgs), stack=True)
        
        cam_names = results['cam_names']

        ## load the adjacent frames
        if 'adjacent' in results and results['adjacent'] is not None:
            source_imgs_list = []
            for adj_info in results['adjacent']:
                cam_infos = adj_info["cams"]
                img_adj_list = []
                post_rot_list = []
                post_tran_list = []
                for cam_type in cam_names:
                    img_fp = cam_infos[cam_type]['data_path']
                    img_adj, post_rot, post_tran = self._load_image(img_fp)
                    img_adj_list.append(img_adj)
                    post_rot_list.append(post_rot)
                    post_tran_list.append(post_tran)

                # resize and normalize
                imgs = [
                    self.transform_core(img, self.input_size)
                    for img in img_adj_list
                ]
                imgs = [img.transpose(2, 0, 1) for img in imgs]
                source_imgs_list.append(np.ascontiguousarray(np.stack(imgs, axis=0)))

            output['source_imgs'] = DC(
                to_tensor(np.stack(source_imgs_list, axis=0)), stack=True)

        return output

    def __call__(self, results):
        ## load the current frame
        curr_results = dict()
        curr_results['cam_names'] = results['cam_names']
        curr_results["img_ori"] = results['target_imgs']
        curr_results["adjacent"] = results['adjacent_visionpad']
        curr_results["K"] = results['K']
        curr_results["post_rots"] = results['target_imgs_post_rots']
        curr_results["post_trans"] = results['target_imgs_post_trans']
        curr_data = self.prepare_data(curr_results)
        results.update(curr_data)

        # update the camera intrinsic matrix of current frame
        post_mat = self.get_post_matrix()
        results['cam_intrinsic'] = torch.stack(
            [post_mat @ torch.as_tensor(_intri, dtype=torch.float32)
             for _intri in results['cam_intrinsic']])

        if 'cam_intrinsic_future' in results:
            results['cam_intrinsic_future'] = torch.stack(
                [post_mat @ torch.as_tensor(_intri, dtype=torch.float32)
                 for _intri in results['cam_intrinsic_future']])

        if self.load_future_img:
            # load the future anchor image
            assert 'future_info' in results, \
                'future_info should be in results'
            future_info = results['future_info']
            filenames = [
                future_info['cams'][cam_name]['data_path']
                for cam_name in results['cam_names']
            ]
            loaded = [self._load_image(filename) for filename in filenames]
            imgs, post_rots, post_trans = zip(*loaded)

            future_results = dict(
                cam_names=results['cam_names'],
                img_ori=list(imgs),
                post_rots=list(post_rots),
                post_trans=list(post_trans),
                adjacent=results.get('adjacent_future', None),
                K=results.get('K_future', None),
            )
            future_data = self.prepare_data(future_results)
            for key in ('source_imgs', 'target_imgs', 'K', 'inv_K'):
                if key in future_data:
                    results[f'{key}_future'] = future_data[key]
        
        return results
    
    def get_inputs(self, results, flip=None, scale=None):
        """Load the current frame and adjacent frames, and do not apply image augmentation.

        Args:
            results (_type_): _description_
            flip (_type_, optional): _description_. Defaults to None.
            scale (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []

        render_img_gts = []
        post_rots_ori = []
        post_trans_ori = []
        img_files = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            img_files.append(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            render_img_gt = self.img_transform_core(img, self.data_config.render_size[::-1], None, False, None)
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img_non_aug = self.sample_noaugmentation(H=img.height, W=img.width)
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)  # 0.44 (704, 396) (0, 12, 704, 396) False 0.0
            _, post_rot2_wo_aug, post_tran2_wo_aug = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=img_non_aug[0],
                                   resize_dims=img_non_aug[1],
                                   crop=img_non_aug[2],
                                   flip=img_non_aug[3],
                                   rotate=img_non_aug[4])
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            post_tran_raw = torch.zeros(3)
            post_rot_raw = torch.eye(3)
            post_tran_raw[:2] = post_tran2_wo_aug
            post_rot_raw[:2, :2] = post_rot2_wo_aug

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))
            render_img_gts.append(self.normalize_img(render_img_gt))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_files.append(filename_adj)
                    render_img_gt_adjacent = self.img_transform_core(
                        img_adjacent, self.data_config.render_size[::-1],
                        None, False, None)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
                    render_img_gts.append(self.normalize_img(render_img_gt_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            post_rots_ori.append(post_rot_raw)
            post_trans_ori.append(post_tran_raw)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                post_trans_ori.extend(post_trans_ori[:len(cam_names)])
                post_rots_ori.extend(post_rots_ori[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)
        render_img_gts = torch.stack(render_img_gts)
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots_ori = torch.stack(post_rots_ori)
        post_trans_ori = torch.stack(post_trans_ori)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas
        results['img_files'] = img_files
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans), \
                render_img_gts, post_rots_ori, post_trans_ori
