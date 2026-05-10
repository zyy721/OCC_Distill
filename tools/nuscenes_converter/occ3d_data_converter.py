'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-03-10 19:38:32
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
import yaml
import numpy as np
import shutil
from tqdm import tqdm


def convert_submission_2_occ3d_gt_folder(submission_root, save_root):
    sample_token_2_scene_name_file = "data/nuscenes/sample_token_2_scene_name.yaml"
    with open(sample_token_2_scene_name_file, 'r') as stream:
        sample_token_2_scene_name = yaml.safe_load(stream)
    
    os.makedirs(save_root, exist_ok=True)

    all_files = os.listdir(submission_root)
    for file in tqdm(all_files):
        sample_token = file.split('.')[0]
        scene_name = sample_token_2_scene_name[sample_token]

        scene_save_root = osp.join(save_root, scene_name, sample_token)
        os.makedirs(scene_save_root, exist_ok=True)
        file_path = osp.join(submission_root, file)

        ## load the results
        occ_res = np.load(file_path)['arr_0']
        np.savez_compressed(osp.join(scene_save_root, "labels.npz"),  
                            semantics=occ_res)

        # shutil.copy(file_path, os.path.join(scene_save_root, "labels.npz"))


    
if __name__ == "__main__":
    submission_root = "./data/bevformer_base_occ_val"
    save_root = "results/ECCV/bevformer_base_occ_val"

    submission_root = "/mnt/data2/zhanghm/Code/RenderOcc/results/renderocc_val"
    save_root = "results/ECCV/renderocc_val_val"

    submission_root = "results/bevdet_stbase_val"
    save_root = "results/ECCV/bevdet_stbase_val"

    convert_submission_2_occ3d_gt_folder(submission_root, save_root)