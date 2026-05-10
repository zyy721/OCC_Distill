'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-11-14 16:09:24
Email: haimingzhang@link.cuhk.edu.cn
Description: Render the depth and the semantic separatly.
'''

_base_ = ['./pretrain-dvgo-nerf-depth-pw-align-tarl-256img-12e-new.py']

model = dict(
    pretrain_head=dict(
        use_semantic_align=True, 
        use_pointwise_align=False,
        loss_semantic_align_weight=0.1
    ),
)
