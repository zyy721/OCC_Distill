'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-29 14:15:55
Email: haimingzhang@link.cuhk.edu.cn
Description: Using the 3DGS to render the Occupancy.
'''

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
from .. import builder
from ..decode_heads.common.conversions import resize_depth_with_alpha
from ..decode_heads.common.gaussians import initialize_density_head
from .bevdet_occ import BEVStereo4DOCC
from .bevdet import BEVStereo4D
from .depth_ssl import *


NUSCENSE_LIDARSEG_PALETTE = torch.Tensor([
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
    (0, 175, 0)
])

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


def warp_voxel_features(voxel_feats,
                        voxel_flow,
                        voxel_size,
                        occ_size,
                        future_keyego_from_curr_keyego=None,
                        point_cloud_range=None):
    """Backward-warp current voxel features into a future key-ego frame.

    ``voxel_flow`` is a learned backward sampling residual in metres. It is
    indexed on the future output lattice, expressed along the current key-ego
    x/y axes, and added after rigid future-to-current ego compensation:

        p_current = T_current_from_future @ p_future + backward_flow

    PyTorch 5-D ``grid_sample`` consumes a source grid in [z, y, x] order for
    an input laid out as [B, C, X, Y, Z].

    Args:
        voxel_feats (Tensor | list[Tensor]): Current [B, C, X, Y, Z] feature.
        voxel_flow (Tensor): Backward xy residual [B, F, X, Y, Z, 2].
        voxel_size (Sequence[float]): Metric xyz voxel size.
        occ_size (Sequence[int]): Number of xyz voxels.
        future_keyego_from_curr_keyego (Tensor): [B, F, 4, 4] transform.
        point_cloud_range (Sequence[float]): xyz min followed by xyz max.

    Returns:
        list[Tensor]: Future-frame features, preserving the historical API.
    """
    if point_cloud_range is None:
        raise ValueError('point_cloud_range is required for metric ego warping')

    device = voxel_flow.device
    dtype = voxel_flow.dtype
    bs, num_pred, x_size, y_size, z_size, flow_dim = voxel_flow.shape
    if flow_dim != 2:
        raise ValueError(f'Expected 2-D xy flow, got {flow_dim} channels')

    spatial_shape = torch.tensor(
        [x_size, y_size, z_size], device=device, dtype=dtype)
    configured_shape = torch.as_tensor(
        occ_size, device=device, dtype=dtype).flatten()
    if configured_shape.numel() != 3 or not torch.equal(
            configured_shape.to(torch.long), spatial_shape.to(torch.long)):
        raise ValueError(
            f'occ_size {configured_shape.tolist()} does not match flow shape '
            f'{spatial_shape.tolist()}')

    pc_range = torch.as_tensor(
        point_cloud_range, device=device, dtype=dtype).flatten()
    if pc_range.numel() != 6:
        raise ValueError('point_cloud_range must contain six values')
    xyz_min, xyz_max = pc_range[:3], pc_range[3:]
    metric_size = xyz_max - xyz_min
    configured_voxel_size = torch.as_tensor(
        voxel_size, device=device, dtype=dtype).flatten()
    actual_voxel_size = metric_size / spatial_shape
    if configured_voxel_size.numel() != 3 or not torch.allclose(
            configured_voxel_size,
            actual_voxel_size,
            rtol=1e-5,
            atol=1e-6):
        raise ValueError(
            f'voxel_size {configured_voxel_size.tolist()} is inconsistent '
            f'with range/shape {actual_voxel_size.tolist()}')

    # Physical future voxel centres, in xyz order.
    xs = xyz_min[0] + (
        torch.arange(x_size, device=device, dtype=dtype) + 0.5
    ) * actual_voxel_size[0]
    ys = xyz_min[1] + (
        torch.arange(y_size, device=device, dtype=dtype) + 0.5
    ) * actual_voxel_size[1]
    zs = xyz_min[2] + (
        torch.arange(z_size, device=device, dtype=dtype) + 0.5
    ) * actual_voxel_size[2]
    future_xyz = torch.stack([
        xs[:, None, None].expand(x_size, y_size, z_size),
        ys[None, :, None].expand(x_size, y_size, z_size),
        zs[None, None, :].expand(x_size, y_size, z_size),
    ], dim=-1)
    future_xyz = future_xyz[None, None].expand(
        bs, num_pred, -1, -1, -1, -1)

    if future_keyego_from_curr_keyego is None:
        future_keyego_from_curr_keyego = torch.eye(
            4, device=device, dtype=dtype)[None, None].expand(
                bs, num_pred, -1, -1)
    else:
        future_keyego_from_curr_keyego = (
            future_keyego_from_curr_keyego.to(device=device, dtype=dtype))
        if future_keyego_from_curr_keyego.ndim == 3:
            future_keyego_from_curr_keyego = \
                future_keyego_from_curr_keyego[:, None]
        if future_keyego_from_curr_keyego.shape[1] == 1 and num_pred > 1:
            future_keyego_from_curr_keyego = \
                future_keyego_from_curr_keyego.expand(-1, num_pred, -1, -1)
        expected_shape = (bs, num_pred, 4, 4)
        if tuple(future_keyego_from_curr_keyego.shape) != expected_shape:
            raise ValueError(
                'future_keyego_from_curr_keyego has shape '
                f'{tuple(future_keyego_from_curr_keyego.shape)}, expected '
                f'{expected_shape}')

    curr_keyego_from_future_keyego = torch.linalg.inv(
        future_keyego_from_curr_keyego)
    rotation = curr_keyego_from_future_keyego[..., :3, :3]
    translation = curr_keyego_from_future_keyego[..., :3, 3]
    source_xyz = torch.einsum(
        'bfij,bfxyzj->bfxyzi', rotation, future_xyz)
    source_xyz = source_xyz + translation[:, :, None, None, None, :]

    flow_xyz = torch.cat(
        [voxel_flow, torch.zeros_like(voxel_flow[..., :1])], dim=-1)
    source_xyz = source_xyz + flow_xyz

    # Metric xyz -> normalized xyz. Reorder to [z, y, x] because the input
    # tensor dimensions are [X, Y, Z] while grid_sample expects [D, H, W].
    source_norm_xyz = 2.0 * (source_xyz - xyz_min) / metric_size - 1.0
    grid = source_norm_xyz[..., [2, 1, 0]]
    grid = rearrange(grid, 'b f x y z dim3 -> (b f) x y z dim3')

    if not isinstance(voxel_feats, list):
        voxel_feats = [voxel_feats]

    outputs = []
    for feature in voxel_feats:
        if feature is None:
            outputs.append(None)
            continue
        if tuple(feature.shape[2:]) != (x_size, y_size, z_size):
            raise ValueError(
                f'Feature shape {tuple(feature.shape[2:])} does not match '
                f'flow shape {(x_size, y_size, z_size)}')

        feature = feature.unsqueeze(1).expand(
            -1, num_pred, -1, -1, -1, -1)
        feature = rearrange(feature, 'b f c x y z -> (b f) c x y z')
        warped = F.grid_sample(
            feature,
            grid.to(feature.dtype),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        warped = rearrange(
            warped, '(b f) c x y z -> b f c x y z', b=bs)
        warped = warped.squeeze(1)
        outputs.append(warped)

    return outputs


@DETECTORS.register_module()
class BEVStereo4DOCCVisionPAD(BEVStereo4DOCC):
    """Using the VisionPAD to pretrain the BEVDet-Occ.

    Args:
        BEVStereo4DOCC (_type_): _description_
    """
    def __init__(self,
                 is_pretrain_det=False,
                 in_dim=32,

                 render_scale=(1, 1),
                 use_semantic=False,
                 render_head_cfg=None,
                 use_depth_consistency=False,
                 render_view_indices=list(range(6)),
                 depth_ssl_size=None,
                 depth_alpha_threshold=1e-3,
                 depth_loss_weight=1.0,
                 rgb_loss_weight=1.0,
                 use_depth_gt_loss=False,
                 use_semantic_gt_loss=False,
                 depth_gt_loss_weight=1.0,
                 opt=None,
                 pred_flow=False,
                 voxel_shape=None,
                 voxel_size=None,
                 point_cloud_range=None,
                 use_flow_ssl=False,
                 use_flow_photometric_loss=False,
                 flow_depth_loss_weight=0.15,
                 use_flow_rgb=False,
                 rgb_future_loss_weight=1.0,
                 use_flow_refine_layer=False,
                 use_sperate_render_head=False,
                 use_pseudo_depth_loss=False,
                 pseudo_depth_loss_weight=1.0,
                 density_init_prob=None,
                 density_init_std=1e-3,
                 **kwargs):
        super(BEVStereo4DOCCVisionPAD, self).__init__(**kwargs)
        
        self.is_pretrain_det = is_pretrain_det
        self.in_dim = in_dim

        self.use_semantic = use_semantic

        self.pred_flow = pred_flow
        self.use_flow_ssl = use_flow_ssl
        self.use_flow_photometric_loss = use_flow_photometric_loss # whether to use the photometric loss for the flow SSL
        self.flow_depth_loss_weight = flow_depth_loss_weight
        self.use_flow_rgb = use_flow_rgb  # use the RGB to supervise the flow
        self.rgb_future_loss_weight = rgb_future_loss_weight
        self.use_flow_refine_layer = use_flow_refine_layer
        self.use_sperate_render_head = use_sperate_render_head

        self.voxel_shape = voxel_shape
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        if self.use_flow_ssl and self.point_cloud_range is None:
            raise ValueError(
                'point_cloud_range is required when use_flow_ssl=True')

        out_dim = self.out_dim

        ## use the depth self-supervised consistency loss
        self.use_depth_consistency = use_depth_consistency
        self.render_view_indices = render_view_indices
        self.depth_ssl_size = depth_ssl_size
        if not 0 < depth_alpha_threshold <= 1:
            raise ValueError('depth_alpha_threshold must be in (0, 1]')
        self.depth_alpha_threshold = depth_alpha_threshold
        self.opt = opt  # options for the depth consistency loss
        self.depth_loss_weight = depth_loss_weight

        self.rgb_loss_weight = rgb_loss_weight
        
        self.use_depth_gt_loss = use_depth_gt_loss
        self.depth_gt_loss_weight = depth_gt_loss_weight

        self.use_semantic_gt_loss = use_semantic_gt_loss

        self.use_pseudo_depth_loss = use_pseudo_depth_loss
        self.pseudo_depth_loss_weight = pseudo_depth_loss_weight

        if density_init_prob is not None and not 0 < density_init_prob < 1:
            raise ValueError('density_init_prob must be in (0, 1) or None')
        if density_init_std < 0:
            raise ValueError('density_init_std must be non-negative')
        self.density_init_prob = density_init_prob
        self.density_init_std = density_init_std

        if self.use_depth_consistency:
            h = depth_ssl_size[0]
            w = depth_ssl_size[1]
            num_cam = len(self.render_view_indices) * 2 # we assume batch size is 4
            self.backproject_depth = BackprojectDepth(num_cam, h, w)
            self.project_3d = Project3D(num_cam, h, w)

            self.ssim = SSIM()

        if render_head_cfg is not None:
            self.render_head = builder.build_head(render_head_cfg)

        self.render_head_cfg = render_head_cfg

        self.render_scale = render_scale

        self.uni_conv = nn.Sequential(
            nn.Conv3d(
                # self.out_dim,
                self.in_dim,

                self.out_dim,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.BatchNorm3d(self.out_dim),
            nn.ReLU(inplace=True),
        )

        self.occupancy_head = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.Softplus(),
            nn.Linear(out_dim * 2, 1),
        )

        if self.pred_flow:
            self.flow_head = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Softplus(),
                nn.Linear(out_dim * 2, 2),
            )
            # Start from rigid ego-motion compensation (zero learned residual)
            # instead of a random sampling field that can leave the volume.
            nn.init.zeros_(self.flow_head[-1].weight)
            nn.init.zeros_(self.flow_head[-1].bias)
        
        if self.use_flow_refine_layer:
            self.flow_refine_layer = nn.Sequential(
                nn.Conv3d(
                    out_dim,
                    out_dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
                nn.BatchNorm3d(out_dim),
                nn.ReLU(inplace=True),
            )

        if self.use_sperate_render_head:
            # use a sperate render head the future volume feature
            self.render_head_future = builder.build_head(render_head_cfg)

            self.occupancy_head_future = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Softplus(),
                nn.Linear(out_dim * 2, 1),
            )
        
        ## remove the unnecessary components
        del self.final_conv
        if self.use_predicter:
            del self.predicter

    def init_weights(self):
        super().init_weights()
        if self.density_init_prob is None:
            return

        initialize_density_head(
            self.occupancy_head,
            self.density_init_prob,
            self.density_init_std,
        )
        if hasattr(self, 'occupancy_head_future'):
            initialize_density_head(
                self.occupancy_head_future,
                self.density_init_prob,
                self.density_init_std,
            )

    @staticmethod
    def inverse_flip_aug(feat, flip_dx, flip_dy):
        batch_size = feat.shape[0]
        feat_flip = []
        for b in range(batch_size):
            flip_flag_x = flip_dx[b]
            flip_flag_y = flip_dy[b]
            tmp = feat[b]
            if flip_flag_x:
                tmp = tmp.flip(1)
            if flip_flag_y:
                tmp = tmp.flip(2)
            feat_flip.append(tmp)
        feat_flip = torch.stack(feat_flip)
        return feat_flip
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward training function."""
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs) # torch.Size([bs, 32, 16, 200, 200])
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        if self.is_pretrain_det:
            img_feats[0] = rearrange(img_feats[0], 'B (Z C) Y X -> B C Z Y X', Z=self.voxel_shape[2], C=self.in_dim)

        uni_feats = self.uni_conv(img_feats[0])  # (bs, c, z, y, x)     torch.Size([4, 256, 128, 128])

        output = dict()
        # output['pose_spatial'] = torch.inverse(kwargs['lidar2cam'])
        output['pose_spatial'] = torch.inverse(kwargs['lidar2cam'].float())
        output['intrinsics'] = kwargs['cam_intrinsic'].float().clone()  # (bs, 6, 4, 4)

        ## 2. Prepare the features for rendering
        _uni_feats = rearrange(uni_feats, 'b c z y x -> b c x y z')

        # cancel the effect of flip augmentation
        flip_dx, flip_dy = kwargs['flip_dx'], kwargs['flip_dy']
        _uni_feats = self.inverse_flip_aug(_uni_feats, flip_dx, flip_dy)
        _uni_feats = rearrange(_uni_feats, 'b c x y z -> b x y z c')

        ## predict the 3DGS
        output['volume_feat'] = _uni_feats

        occupancy_output = self.occupancy_head(_uni_feats)
        occupancy_output = rearrange(occupancy_output, 'b x y z dim1 -> b dim1 x y z')
        output['density_prob'] = occupancy_output  # density score

        ## 3. Prepare the semantic features
        output['semantic'] = None
        if self.use_semantic:
            semantic_output = self.semantic_head(_uni_feats)
            semantic_output = rearrange(semantic_output, 'b x y z C -> b C x y z')
            output['semantic'] = semantic_output

        if self.pred_flow and not self.use_flow_ssl:
            # This compatibility branch is only meaningful for consumers that
            # explicitly use a current-grid flow prediction.  Future SSL
            # predicts its backward residual after rigid future-grid alignment
            # below, where the field is actually sampled.
            flow_output = self.flow_head(_uni_feats)
            flow_output = rearrange(flow_output, 'b x y z dim1 -> b () x y z dim1')
            output['flow'] = flow_output

        ## 2. Start rendering, including neural rendering or 3DGS
        render_results = self.render_head(output)

        if self.use_flow_ssl:
            # the Flow-based SSL
            assert self.pred_flow, "The flow prediction is required for the flow self-supervised loss!"

            volume_feature = output['volume_feat']
            volume_feature_ch_first = rearrange(
                volume_feature, 'b x y z c -> b c x y z')
            future_keyego_from_curr_keyego = kwargs[
                'future_keyego_from_curr_keyego']

            # ``warp_voxel_features`` samples a backward residual indexed on
            # the future output lattice.  Predict that field from a rigidly
            # aligned future-grid feature instead of reading a current-grid
            # pointwise prediction at the same numerical index.
            batch_size, _, x_size, y_size, z_size = \
                volume_feature_ch_first.shape
            rigid_flow = volume_feature_ch_first.new_zeros(
                (batch_size, 1, x_size, y_size, z_size, 2))
            rigid_future_feature = warp_voxel_features(
                volume_feature_ch_first,
                rigid_flow,
                voxel_size=self.voxel_size,
                occ_size=self.voxel_shape,
                future_keyego_from_curr_keyego=(
                    future_keyego_from_curr_keyego),
                point_cloud_range=self.point_cloud_range)[0]
            rigid_future_feature = rearrange(
                rigid_future_feature, 'b c x y z -> b x y z c')

            voxel_flow_pred = self.flow_head(rigid_future_feature)
            voxel_flow_pred = rearrange(
                voxel_flow_pred, 'b x y z dim2 -> b () x y z dim2')
            output['flow'] = voxel_flow_pred

            warped_results = warp_voxel_features(
                volume_feature_ch_first,
                voxel_flow_pred, 
                voxel_size=self.voxel_size,
                occ_size=self.voxel_shape,
                future_keyego_from_curr_keyego=(
                    future_keyego_from_curr_keyego),
                point_cloud_range=self.point_cloud_range)
            
            future_volume_feat = warped_results[0]  # (bs, c, x, y, z)
            if self.use_flow_refine_layer:
                future_volume_feat = self.flow_refine_layer(future_volume_feat)
            future_volume_feat = rearrange(future_volume_feat, 'b c x y z -> b x y z c')

            future_output = dict()
            future_output['volume_feat'] = future_volume_feat

            # ``keyego2cam_future`` comes from the pkl ``lidar2cam`` field
            # and is future-keyego-to-camera (world-to-camera).
            # Gaussian rendering expects camera-to-world in the future key
            # ego frame, consistently with the current-frame branch above.
            future_output['pose_spatial'] = torch.inverse(
                kwargs['keyego2cam_future'].float())
            future_output['intrinsics'] = (
                kwargs['cam_intrinsic_future'].float().clone())

            # start rendering future volume feature
            if self.use_sperate_render_head:
                occupancy_output = self.occupancy_head_future(future_volume_feat)
                future_output['density_prob'] = rearrange(
                    occupancy_output, 'b x y z dim1 -> b dim1 x y z') # density score
                # TODO: add semantic head
                future_output['semantic'] = None
                future_render_results = self.render_head_future(future_output, suffix='_future')
            else:
                occupancy_output = self.occupancy_head(future_volume_feat)
                future_output['density_prob'] = rearrange(
                    occupancy_output, 'b x y z dim1 -> b dim1 x y z') # density score

                future_output['semantic'] = None
                if self.use_semantic:
                    semantic_output = self.semantic_head(future_volume_feat)
                    future_output['semantic'] = rearrange(semantic_output, 'b x y z C -> b C x y z')

                future_render_results = self.render_head(future_output, suffix='_future')

            render_results.update(future_render_results)

        ## 3. Compute the loss
        target_dict = dict(**kwargs)
        # Detection pretraining intentionally excludes the sparse LiDAR depth
        # loss computed above; only the rendering/self-supervised objectives
        # participate in backpropagation.
        losses = self.loss(render_results, target_dict)
        return losses
    def loss(self, preds_dict, targets):
        if self.use_depth_consistency:
            ## Visualize the input data
            device = preds_dict['render_depth'].device

            loss_dict = {}

            if self.depth_loss_weight > 0.0:
                ## 1) Compute the reprojected rgb images based on the rendered depth
                self.generate_image_pred(targets, preds_dict)

                ## 2) Compute the depth consistency loss
                loss_depth_ssl = self.compute_self_supervised_losses(targets, preds_dict)
                loss_dict.update(loss_depth_ssl)

            if self.use_flow_ssl:
                if not self.use_flow_photometric_loss:
                    ## Compute the flow loss with GT
                    loss_flow_ssl = self.render_head.loss(
                        preds_dict, targets, 
                        suffix='_future', weight=self.flow_depth_loss_weight)
                    loss_dict.update(loss_flow_ssl)
                else:
                    if self.use_flow_rgb:
                        render_rgb = preds_dict['render_rgb_future']  # (bs, num_cam, 3, h, w)
                        target_img = targets['target_imgs_future']  # (bs, num_cam, 3, h, w)

                        render_rgb = rearrange(
                            render_rgb, 'b num_view dim3 h w -> (b num_view) dim3 h w')
                        render_rgb = F.interpolate(
                            render_rgb, self.depth_ssl_size, 
                            mode="bilinear", align_corners=False)
                        render_rgb = rearrange(
                            render_rgb, '(b num_view) dim3 h w -> b num_view dim3 h w', b=target_img.shape[0])
                        rgb_loss = self.rgb_future_loss_weight * F.l1_loss(render_rgb, target_img)
                        loss_rgb = {'loss_rgb_future' : rgb_loss}
                        loss_dict.update(loss_rgb)
                    else:
                        ## Compute the photometric loss for the flow SSL
                        self.generate_image_pred(targets, preds_dict, suffix='_future')

                        loss_flow_ssl = self.compute_self_supervised_losses(
                            targets, preds_dict, suffix='_future')
                        loss_dict.update(loss_flow_ssl)

            ## 3) Compute the RGB reconstruction loss
            if self.rgb_loss_weight > 0.0:
                render_curr_img = preds_dict['render_rgb']  # (bs, num_cam, 3, h, w)
                target_img = targets['target_imgs']  # (bs, num_cam, 3, h, w)

                rgb_loss = self.rgb_loss_weight * self.compute_rgb_loss(
                    render_curr_img, target_img, target_size=self.depth_ssl_size)
                loss_rgb = {'loss_rgb' : rgb_loss}
                loss_dict.update(loss_rgb)

            ## 4) Compute the depth gt loss
            if self.use_depth_gt_loss:
                render_depth = preds_dict['render_depth']
                gt_depth = targets['render_gt_depth']

                mask = gt_depth > 0.0
                loss_render_depth = F.l1_loss(render_depth[mask], gt_depth[mask])
                if torch.isnan(loss_render_depth):
                    print('NaN in render depth loss!')
                    loss_render_depth = torch.Tensor([0.0]).to(device)
                loss_dict['loss_render_depth'] = self.depth_gt_loss_weight * loss_render_depth

            if self.use_semantic_gt_loss:
                assert 'render_gt_semantic' in targets.keys()

                semantic_gt = targets['render_gt_semantic']
                semantic_pred = preds_dict['render_semantic']
                
                loss_render_sem = self.compute_semantic_loss(
                    semantic_pred, semantic_gt, ignore_index=255)
                if torch.isnan(loss_render_sem):
                    print('NaN in render semantic loss!')
                    loss_render_sem = torch.Tensor([0.0]).to(device)
                loss_dict['loss_render_sem'] = 0.1 * loss_render_sem
        else:
            loss_dict = self.render_head.loss(preds_dict, targets)

            if self.use_flow_ssl:
                # compute the loss for the future frame
                future_loss_dict = self.render_head.loss(
                    preds_dict, targets, suffix='_future', weight=0.15)
                
                loss_dict.update(future_loss_dict)

        return loss_dict
    
    def compute_rgb_loss(self, 
                         pred_img, 
                         target_img, 
                         target_size=None,
                         use_ssim=False):
        if target_size is None:
            target_size = (target_img.shape[-2], target_img.shape[-1])
        
        _pred_img = rearrange(
            pred_img, 'b num_view dim3 h w -> (b num_view) dim3 h w')
        _pred_img = F.interpolate(
            _pred_img, target_size, mode="bilinear", align_corners=False)
        _pred_img = rearrange(
            _pred_img, '(b num_view) dim3 h w -> b num_view dim3 h w', 
            b=pred_img.shape[0])
        rgb_loss = F.l1_loss(_pred_img, target_img)
        return rgb_loss
    
    def generate_image_pred(self, inputs, outputs, suffix=''):
        color_source_imgs_list = []
        for idx in range(inputs['source_imgs' + suffix].shape[1]):
            color_source = inputs['source_imgs' + suffix][:, idx]  # prev and next images
            color_source = rearrange(color_source, 'b num_view c h w -> (b num_view) c h w')
            color_source_imgs_list.append(color_source)
        inputs['color_source_imgs' + suffix] = color_source_imgs_list

        inv_K = inputs['inv_K' + suffix][:, self.render_view_indices]
        K = inputs['K' + suffix][:, self.render_view_indices]
        inv_K = rearrange(inv_K, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')
        K = rearrange(K, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')

        # rescale the rendered depth
        depth = outputs['render_depth' + suffix][:, self.render_view_indices]
        depth = rearrange(depth, 'b num_view h w -> (b num_view) () h w')
        alpha = outputs['render_alpha' + suffix][:, self.render_view_indices]
        alpha = rearrange(alpha, 'b num_view h w -> (b num_view) () h w')
        depth, alpha, depth_valid = resize_depth_with_alpha(
            depth,
            alpha,
            self.depth_ssl_size,
            alpha_threshold=self.depth_alpha_threshold,
        )
        outputs['render_depth_rescaled' + suffix] = depth
        outputs['render_alpha_rescaled' + suffix] = alpha
        outputs['render_depth_valid' + suffix] = depth_valid

        cam_T_cam = inputs["cam_T_cam" + suffix][:, :, self.render_view_indices]

        ## 1) Depth to camera points
        cam_points = self.backproject_depth(depth, inv_K)  # (M, 4, h*w)
        len_temporal = cam_T_cam.shape[1]
        color_reprojection_list = []
        for frame_id in range(len_temporal):
            T = cam_T_cam[:, frame_id]
            T = rearrange(T, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')
            ## 2) Camera points to adjacent image points
            pix_coords = self.project_3d(cam_points, K, T)  # (M, h, w, 2)

            ## 3) Reproject the adjacent image
            color_source = inputs['color_source_imgs' + suffix][frame_id]  # (M, 3, h, w)
            color_reprojection = F.grid_sample(
                color_source,
                pix_coords,
                padding_mode="border", align_corners=True)
            color_reprojection_list.append(color_reprojection)

        outputs['color_reprojection' + suffix] = color_reprojection_list
        outputs['target_imgs' + suffix] = rearrange(
            inputs['target_imgs' + suffix], 'b num_view c h w -> (b num_view) c h w')
        
    def compute_self_supervised_losses(self, inputs, outputs, suffix=''):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        loss = 0

        depth = outputs["render_depth_rescaled" + suffix]  # (M, 1, h, w)
        depth_valid = outputs["render_depth_valid" + suffix]
        disp = torch.where(
            depth_valid,
            1.0 / depth.clamp_min(1e-7),
            torch.zeros_like(depth),
        )
        color = outputs["target_imgs" + suffix]
        target = outputs["target_imgs" + suffix]

        reprojection_losses = []
        for frame_id in range(len(outputs['color_reprojection' + suffix])):
            pred = outputs['color_reprojection' + suffix][frame_id]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)  # (M, 2, h, w)

        ## automasking
        identity_reprojection_losses = []
        for frame_id in range(len(outputs['color_reprojection' + suffix])):
            pred = inputs["color_source_imgs" + suffix][frame_id]
            identity_reprojection_losses.append(
                self.compute_reprojection_loss(pred, target))

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

        if self.opt.avg_reprojection:
            identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
        else:
            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

        if self.opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not self.opt.disable_automasking:
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn_like(
                identity_reprojection_loss) * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        valid_weight = depth_valid[:, 0].to(to_optimise.dtype)
        loss += (to_optimise * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)

        valid_float = depth_valid.to(disp.dtype)
        mean_disp = disp.sum((2, 3), keepdim=True) / valid_float.sum(
            (2, 3), keepdim=True).clamp_min(1.0)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color, depth_valid)

        loss += self.opt.disparity_smoothness * smooth_loss
        
        total_loss += loss
        losses["loss_depth_ct" + suffix] = self.depth_loss_weight * total_loss  # depth consistency loss
        return losses

    def compute_reprojection_loss(self, pred, target, no_ssim=False):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


@DETECTORS.register_module()
class BEVStereo4DOCCGS(BEVStereo4D):

    def __init__(self,
                 render_head=None,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 scene_filter_index=-1,
                 return_weights=False,
                 use_render_loss=True,
                 **kwargs):
        
        super(BEVStereo4DOCCGS, self).__init__(**kwargs)

        self.out_dim = out_dim
        self.scene_filter_index = scene_filter_index
        out_channels = out_dim if use_predicter else num_classes
        self.render_head = builder.build_backbone(render_head)
        self.num_classes = num_classes
        self.return_weights = return_weights
        self.lovasz = self.render_head.lovasz

        if self.render_head.img_recon_head:
            num_classes += 3

        self.final_conv = ConvModule(
            self.img_view_transformer.out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes),
            )

        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_random_view = self.render_head.num_random_view
        
        if loss_occ is not None:
            self.loss_occ = build_loss(loss_occ)
        
        self.align_after_view_transfromation = False
        self.use_render_loss = use_render_loss

    @property
    def with_occ_loss(self):
        """bool: Whether compute the occupancy loss with Occupancy GT supervision."""
        return hasattr(self, 'loss_occ') and self.loss_occ is not None

    def loss_single(self, voxel_semantics, mask_camera, preds, semi_mask=None):
        loss_ = dict()
        if semi_mask is not None:
            mask_camera *= semi_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics, )
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        volume_feat = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)

        if self.use_predicter:
            # to (b, 200, 200, 16, c)
            occ_pred_ori = self.predicter(volume_feat)
        else:
            occ_pred_ori = volume_feat
        
        occ_pred = occ_pred_ori[..., :-3] \
            if self.render_head.img_recon_head else occ_pred_ori

        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        ## nerf
        VISUALIZE = True
        if VISUALIZE:
            # to (b, c, 200, 200, 16)
            use_gt_occ = True

            occ_pred = occ_pred_ori.permute(0, 4, 1, 2, 3)

            # semantic
            if self.render_head.semantic_head:
                # (b, 17, 200, 200, 16)
                semantic_ori = occ_pred[:, :self.render_head.semantic_dim, ...]
            else:
                semantic_ori = torch.zeros_like(occ_pred[:, :self.render_head.semantic_dim, ...])
            # density
            if use_gt_occ:
                # (b, 200, 200, 16)
                density_prob = kwargs['voxel_semantics'][0].unsqueeze(1)
                density_prob = density_prob != 17
                density_prob = density_prob.float()
                density_prob[density_prob == 0] = -10  # scaling to avoid 0 in alphas
                density_prob[density_prob == 1] = 10
            else:
                density_prob = -occ_pred[:, self.render_head.semantic_dim:self.render_head.semantic_dim+1]

            intricics = kwargs['intricics']
            pose_spatial = kwargs['pose_spatial']
            rgb_recons = occ_pred[:, -3:]

            ## firstly we need to find the semantic class for each voxel
            semantic = semantic_ori.argmax(1)
            # mapping the color, to (b, h, w, d, 3)
            if use_gt_occ:
                semantic = OCC3D_PALETTE[kwargs['voxel_semantics'][0].long()].to(occ_pred)
            else:
                semantic = OCC3D_PALETTE[semantic].to(occ_pred)
            semantic = semantic.permute(0, 4, 1, 2, 3)

            render_depth, rgb_pred, semantic_pred = self.render_head(
                density_prob, semantic, semantic_ori, 
                intricics[0], pose_spatial[0], 
                is_train=False, render_mask=None, 
                vis_semantic=True, volume_feat=volume_feat)
            
            render_img_gt = kwargs['render_gt_img'][0]
            current_frame_img = render_img_gt.view(
                6, self.num_frame, -1, 
                render_img_gt.shape[-2], 
                render_img_gt.shape[-1])[:, 0].cpu().numpy()
            self.render_head.visualize_image_semantic_depth_pair(
                current_frame_img,
                rgb_pred[0].permute(0, 2, 3, 1),
                render_depth[0],
                save_dir="results/3dgs/baseline_gt"
            )
            exit()
        return [occ_res]

    @staticmethod
    def inverse_flip_aug(feat, flip_dx, flip_dy):
        batch_size = feat.shape[0]
        feat_flip = []
        for b in range(batch_size):
            flip_flag_x = flip_dx[b]
            flip_flag_y = flip_dy[b]
            tmp = feat[b]
            if flip_flag_x:
                tmp = tmp.flip(1)
            if flip_flag_y:
                tmp = tmp.flip(2)
            feat_flip.append(tmp)
        feat_flip = torch.stack(feat_flip)
        return feat_flip

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward training function.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        intricics = kwargs['intricics']
        pose_spatial = kwargs['pose_spatial']
        flip_dx, flip_dy = kwargs['flip_dx'], kwargs['flip_dy']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        render_img_gt = kwargs['render_gt_img']
        render_gt_depth = kwargs['render_gt_depth']

        # occupancy prediction
        volume_feat = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)  # to (bs, 200, 200, 16, c)
        
        if self.use_predicter:
            occ_pred = self.predicter(volume_feat)
        else:
            occ_pred = volume_feat
        
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17

        # occupancy losses
        if self.render_head.img_recon_head:
            occ_pred = occ_pred[..., :-3]
            
        if self.with_occ_loss:
            loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
            losses.update(loss_occ)

        # NeRF loss
        debug = False
        if debug:  # DEBUG ONLY!
            voxel_semantics = kwargs['voxel_semantics'].unsqueeze(1) # (bs, 1, 200, 200, 16)

            density_prob = voxel_semantics != 17
            density_prob = density_prob.float()
            density_prob[density_prob == 0] = -10  # scaling to avoid 0 in alphas
            density_prob[density_prob == 1] = 10
            batch_size = density_prob.shape[0]
            density_prob_flip = self.inverse_flip_aug(density_prob, flip_dx, flip_dy)
            voxel_semantics_flip = self.inverse_flip_aug(voxel_semantics, flip_dx, flip_dy)
            print(density_prob_flip.shape, voxel_semantics_flip.shape)

            # nerf decoder
            render_depth, render_rgb, _ = self.render_head(
                density_prob_flip,
                density_prob_flip.tile(1, 3, 1, 1, 1),
                voxel_semantics_flip,
                intricics, 
                pose_spatial, 
                is_train=True
            )
            print('density_prob', density_prob.shape)  # [1, 1, 200, 200, 16]
            print('render_depth', render_depth.shape, render_depth.min(), render_depth.max())  # [1, 6, 224, 352]
            print('render_rgb', render_rgb.shape, render_rgb.min(), render_rgb.max())
            print('gt_depth', gt_depth.shape, gt_depth.max(), gt_depth.min())  # [1, 6, 384, 704]
            current_frame_img = render_img_gt.view(batch_size, 6, self.num_frame, -1, 
                                                   render_img_gt.shape[-2], 
                                                   render_img_gt.shape[-1])[0, :, 0].cpu().numpy()
            print('current_frame_img', current_frame_img.shape)
            ## start visualize the results
            self.NeRFDecoder.visualize_image_depth_pair(
                current_frame_img, 
                render_gt_depth[0], 
                render_depth[0])
            self.NeRFDecoder.visualize_image_semantic_depth_pair(
                current_frame_img, 
                render_rgb.permute(0, 1, 3, 4, 2)[0], 
                render_depth[0],
                save_dir="./results/3dgs")
            exit()
        else:
            occ_pred = occ_pred.permute(0, 4, 1, 2, 3)

            # semantic
            if self.render_head.semantic_head:
                semantic = occ_pred[:, :self.render_head.semantic_dim, ...]
            else:
                semantic = torch.zeros_like(occ_pred)

            # density
            density_prob = -occ_pred[:, self.render_head.semantic_dim: self.render_head.semantic_dim+1, ...]

            # image reconstruction
            if self.render_head.img_recon_head:
                rgb_recons = occ_pred[:, -4:-1, ...]
            else:
                rgb_recons = torch.zeros_like(density_prob)

            # cancel the effect of flip augmentation
            rgb_flip = self.inverse_flip_aug(rgb_recons, flip_dx, flip_dy)
            semantic_flip = self.inverse_flip_aug(semantic, flip_dx, flip_dy)
            density_prob_flip = self.inverse_flip_aug(density_prob, flip_dx, flip_dy)

            # random view selection
            if self.num_random_view != -1:
                rand_ind = torch.multinomial(torch.tensor([1/self.num_random_view]*self.num_random_view), 
                                             self.num_random_view, 
                                             replacement=False)
                intricics = intricics[:, rand_ind]
                pose_spatial = pose_spatial[:, rand_ind]
                render_gt_depth = render_gt_depth[:, rand_ind]
                render_img_gt = render_img_gt[:, rand_ind]

            # rendering
            render_depth, rgb_pred, semantic_pred = self.render_head(
                density_prob_flip, 
                rgb_flip, 
                semantic_flip, 
                intricics, 
                pose_spatial, 
                volume_feat=volume_feat)

            pred_dict = {'render_depth': render_depth,
                         'render_semantic': semantic_pred}

            gs_losses = self.render_head.loss(pred_dict, dict(**kwargs))
            losses.update(gs_losses)

        return losses
