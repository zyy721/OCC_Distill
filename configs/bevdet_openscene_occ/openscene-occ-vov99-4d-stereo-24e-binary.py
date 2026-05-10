'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-27 16:12:52
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
_base_ = ['./openscene-occ-vov99-4d-stereo-24e.py']

pred_binary_occ = True
is_private_test = False

model = dict(
    type='BEVDepth4DOCCOpenScene',
    num_classes=2,
    pred_binary_occ=pred_binary_occ,
    use_loss_norm=True,

    # loss_occ=dict(
    #     type='FocalLoss',
    #     use_sigmoid=True,
    #     gamma=2.0,
    #     alpha=0.25,
    #     loss_weight=1.0,
    #     reduction='none'),
    loss_occ=dict(
        type='CrossEntropyLoss',
        class_weight=[0.2, 0.8],
        use_sigmoid=False,
        loss_weight=1.0),
    # loss_occ=dict(
    #     type='DiceLoss',

    # )
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        pred_binary_occ=pred_binary_occ),
    val=dict(
        pred_binary_occ=pred_binary_occ),
    test=dict(
        pred_binary_occ=pred_binary_occ,
        is_private_test=is_private_test),
)