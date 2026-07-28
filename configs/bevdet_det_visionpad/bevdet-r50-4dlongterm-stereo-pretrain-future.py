"""Detection pretraining with future-frame RGB flow supervision.

The current and future volumes use their respective CAM_FRONT_LEFT sample
timestamps as key ego frames. ``voxel_flow`` is a backward sampling residual
in metres; rigid ego motion is supplied separately by the dataset.
"""

_base_ = ['./bevdet-r50-4dlongterm-stereo-pretrain.py']

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    render_size=(360, 640),
    resize=(0.0, 0.0),
    rot=(0.0, 0.0),
    flip=False,
    crop_h=(0.0, 0.0),
    resize_test=0.0,
)

grid_config = dict(
    x=[-51.2, 51.2, 0.8],
    y=[-51.2, 51.2, 0.8],
    z=[-5.0, 3.0, 8.0],
    depth=[1.0, 60.0, 1.0],
)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
depth_ssl_size = (360, 640)
render_scale = [
    data_config['render_size'][0] / data_config['src_size'][0],
    data_config['render_size'][1] / data_config['src_size'][1],
]
file_client_args = dict(backend='disk')

# Keep BEV augmentation geometrically neutral while bringing up future warp.
# Image augmentation is already disabled in data_config.
bda_aug_conf = dict(
    rot_lim=(0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0,
)

model = dict(
    pred_flow=True,
    use_flow_ssl=True,
    use_depth_consistency=True,
    use_flow_photometric_loss=True,
    use_flow_rgb=True,
    point_cloud_range=point_cloud_range,
)

train_pipeline = [
    dict(
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
    dict(
        type='PointToMultiViewDepthForNeRF',
        downsample=1,
        grid_config=grid_config,
        render_size=data_config['render_size'],
        render_scale=render_scale),
    dict(
        type='PrepareImageInputsForVisionPAD',
        data_config=data_config,
        input_size=depth_ssl_size,
        render_size=data_config['render_size'],
        load_future_img=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'gt_depth',
            'cam_intrinsic', 'lidar2cam',
            'source_imgs', 'target_imgs',
            'K', 'inv_K', 'cam_T_cam',
            'flip_dx', 'flip_dy',
            'keyego2cam_future',
            'cam_intrinsic_future',
            'future_keyego_from_curr_keyego',
            'target_imgs_future',
        ]),
]

data = dict(
    train=dict(
        ann_file=(
            'data/nuscenes/'
            'nuscenes_unified_infos_train_v4_ann_infos_fl_keyego.pkl'),
        pipeline=train_pipeline,
        future_frames=[1],
        use_depth_consistency=True,
        use_flow_photometric_loss=True,
        key_ego_camera='CAM_FRONT_LEFT',
        require_keyego_metadata=True,
        # Direct future RGB supervision does not need the future frame's own
        # prev/next image triplet, K, or cam-to-cam reprojection matrices.
        load_future_adjacent=False,
    ),
    val=dict(
        ann_file=(
            'data/nuscenes/'
            'nuscenes_unified_infos_val_v4_ann_infos_fl_keyego.pkl'),
        key_ego_camera='CAM_FRONT_LEFT',
        require_keyego_metadata=True),
    test=dict(
        ann_file=(
            'data/nuscenes/'
            'nuscenes_unified_infos_val_v4_ann_infos_fl_keyego.pkl'),
        key_ego_camera='CAM_FRONT_LEFT',
        require_keyego_metadata=True),
)
