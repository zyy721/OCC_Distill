# Copyright (c) Phigent Robotics. All rights reserved.
# align_after_view_transfromation=True
# mAP: 0.4110
# mATE: 0.5763
# mASE: 0.2845
# mAOE: 0.4682
# mAVE: 0.3027
# mAAE: 0.1950
# NDS: 0.5228
# Eval time: 131.4s
#
# Per-class results:
# Object Class	AP	ATE	ASE	AOE	AVE	AAE
# car	0.624	0.392	0.155	0.073	0.248	0.185
# truck	0.342	0.530	0.200	0.087	0.225	0.185
# bus	0.365	0.674	0.205	0.074	0.698	0.330
# trailer	0.223	0.925	0.268	0.489	0.172	0.123
# construction_vehicle	0.130	0.925	0.519	1.184	0.110	0.315
# pedestrian	0.477	0.618	0.303	0.630	0.338	0.186
# motorcycle	0.410	0.551	0.267	0.594	0.468	0.231
# bicycle	0.338	0.428	0.274	0.958	0.161	0.006
# traffic_cone	0.608	0.351	0.350	nan	nan	nan
# barrier	0.593	0.369	0.305	0.124	nan	nan


# align_after_view_transfromation=False
# mAP: 0.4149
# mATE: 0.5655
# mASE: 0.2842
# mAOE: 0.4647
# mAVE: 0.2979
# mAAE: 0.1949
# NDS: 0.5268
# Eval time: 129.4s
#
# Per-class results:
# Object Class	AP	ATE	ASE	AOE	AVE	AAE
# car	0.628	0.387	0.154	0.073	0.245	0.185
# truck	0.343	0.523	0.199	0.087	0.223	0.186
# bus	0.363	0.671	0.205	0.082	0.694	0.326
# trailer	0.225	0.914	0.266	0.478	0.167	0.122
# construction_vehicle	0.134	0.915	0.516	1.180	0.111	0.327
# pedestrian	0.483	0.611	0.303	0.629	0.336	0.185
# motorcycle	0.418	0.526	0.270	0.578	0.449	0.223
# bicycle	0.342	0.410	0.275	0.954	0.158	0.005
# traffic_cone	0.613	0.340	0.350	nan	nan	nan
# barrier	0.600	0.358	0.305	0.121	nan	nan




_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    'render_size': (360, 640),

    # Augmentation
    # 'resize': (-0.06, 0.11),
    # 'rot': (-5.4, 5.4),
    # 'flip': True,
    # 'crop_h': (0.0, 0.0),
    # 'resize_test': 0.00,

    'resize': (0.0, 0.0),      # 设置为 0，不进行随机缩放
    'rot': (0.0, 0.0),         # 设置为 0，不进行随机旋转
    'flip': False,             # 关闭随机翻转
    'crop_h': (0.0, 0.0),      # 不进行裁剪
    'resize_test': 0.00,       # 测试时不缩放

}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80

multi_adj_frame_id_cfg = (1, 8+1, 1)

_render_scale = [data_config['render_size'][0]/data_config['src_size'][0],
                data_config['render_size'][1]/data_config['src_size'][1]]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# unified_voxel_shape = [128, 128, 5]
# unified_voxel_size = [0.8, 0.8, 1.6]
unified_voxel_shape = [128, 128, 4]
unified_voxel_size = [0.8, 0.8, 2]

depth_ssl_size = (360, 640)

use_semantic = False
use_flow_photometric_loss = True
_use_depth_consistency = True

model = dict(
    # type='BEVStereo4D',
    type='BEVStereo4DOCCVisionPAD',

    is_pretrain_det=True,
    in_dim=64,

    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),

    # for 3DGS
    render_scale=_render_scale,
    depth_ssl_size=depth_ssl_size,  # the image size for image warping in depth SSL
    pred_flow=False,
    use_flow_ssl=False,
    use_flow_photometric_loss=use_flow_photometric_loss,  # whether to use photometric loss or GT depth loss for flow
    use_flow_rgb=True,
    use_flow_refine_layer=False,
    use_sperate_render_head=False,
    flow_depth_loss_weight=0.1,
    rgb_future_loss_weight=0.15,
    voxel_shape=unified_voxel_shape,
    voxel_size=unified_voxel_size,
    use_depth_consistency=_use_depth_consistency,
    depth_loss_weight=10.0,
    rgb_loss_weight=0.15,
    opt=dict(
        avg_reprojection=False,
        disable_automasking=False,
        disparity_smoothness=0.001,
    ),
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=True,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    ## For 3DGS
    render_head_cfg=dict(
        type="GaussianSplattingDecoderVisionPad",
        filter_opacities=True,
        semantic_head=use_semantic,
        render_size=data_config['render_size'],
        depth_range=[0.1, 64],
        pc_range=point_cloud_range,
        voxels_size=unified_voxel_shape,
        volume_size=unified_voxel_shape,
        learn_gs_scale_rot=True,
        offset_scale=0.25,
        gs_scale=0.3,
        gs_scale_min=0.1,
        gs_scale_max=0.5,
    ),

    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    use_mask=True,
)

# Data
# dataset_type = 'NuScenesDataset'
dataset_type = 'NuScenesDatasetOccVisionPAD'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    # rot_lim=(-22.5, 22.5),
    # scale_lim=(0.95, 1.05),
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        # type='PrepareImageInputs',
        type='PrepareImageInputsForNeRF',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    # dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='PointToMultiViewDepthForNeRF', 
        downsample=1, 
        grid_config=grid_config, 
        render_size=data_config['render_size'],
        render_scale=_render_scale),
    dict(
        type='PrepareImageInputsForVisionPAD',
        data_config=data_config,
        input_size=depth_ssl_size,
        render_size=data_config['render_size'],),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(
    #     type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
    #                             'gt_depth'])
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 
                                'cam_intrinsic', 'lidar2cam',
                                "source_imgs", "target_imgs",
                                "K", "inv_K", "cam_T_cam",
                                'flip_dx', 'flip_dy', 
                                ])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    # ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')
    ann_file=data_root + 'nuscenes_unified_infos_val_v4_ann_infos_ego.pkl')


data = dict(
    # samples_per_gpu=4,
    samples_per_gpu=2,
    workers_per_gpu=4,
    # train=dict(
    #     type='CBGSDataset',
    #     dataset=dict(
    #     data_root=data_root,
    #     ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
    #     pipeline=train_pipeline,
    #     classes=class_names,
    #     test_mode=False,
    #     use_valid_flag=True,
    #     # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    #     # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    #     box_type_3d='LiDAR')),
    train=dict(
        use_depth_consistency=_use_depth_consistency,
        use_flow_photometric_loss=use_flow_photometric_loss,
        future_frames=[1],
        data_root=data_root,
        # ann_file=data_root + 'bevdetv2-nuscenes_infos_train_visionpad.pkl',
        # ann_file=data_root + 'nuscenes_unified_infos_train_v4.pkl',
        # ann_file=data_root + 'nuscenes_unified_infos_train_v4_ann_infos.pkl',
        ann_file=data_root + 'nuscenes_unified_infos_train_v4_ann_infos_ego.pkl',

        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

# for key in ['val', 'test']:
#     data[key].update(share_data_config)
# data['train']['dataset'].update(share_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    # step=[20,])
    step=[12,])

# runner = dict(type='EpochBasedRunner', max_epochs=20)
runner = dict(type='EpochBasedRunner', max_epochs=12)


# custom_hooks = [
#     dict(
#         type='MEGVIIEMAHook',
#         init_updates=10560,
#         priority='NORMAL',
#     ),
#     dict(
#         type='SequentialControlHook',
#         temporal_start_epoch=2,
#     ),
# ]

custom_hooks = [
    # dict(
    #     type='MEGVIIEMAHook',
    #     init_updates=10560,
    #     priority='NORMAL',
    # ),
    dict(
        type='SequentialControlHook',
        temporal_start_epoch=1,
    ),
]

# fp16 = dict(loss_scale='dynamic')
