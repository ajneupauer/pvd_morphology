#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 17:24:52 2025

@author: alexneupauer
"""

# Import modules
import os
os.chdir('{dir_where_repo_is_stored}/pvd_morphology/')
import sys
sys.path.append('./ml_models')

import pvd_processing as pvd
import numpy as np
from pathlib import Path
import torch
import models
import pvd_classifier_1 as pc1
import tifffile

# %%
# Define function to get branch stats on an image

# Load ML models
unet_path = Path("./ml_models/20250613-pvdseg.pth")
compute_device = torch.device("mps")
model = models.AttentionUNet(1, 1, features=[16, 32, 64, 128], use_logits=True)
unet = torch.load(unet_path, weights_only=False)
model.load_state_dict(unet["model_state_dict"])
model = model.to(compute_device)

classifier = pc1.PVDNeuriteClassifier()
classifier.load_model('./ml_models/class-3.joblib')

# Function
def img_to_branches(img_path, seg_model, seg_compute_device, classifier):
    # Manage file paths
    coord_path = img_path.replace('_squished.tif','.npy')
    
    root_path = Path(img_path).parent
    root_name = Path(img_path).stem
    
    straightened_path = str(root_path) + '/straightenedImgs/' + root_name.replace('squished', 'Straightened.tif')
    maxProj_path = str(root_path) + '/maxProj/' + root_name.replace('squished', 'maxProj.tif')
    mask_path = str(root_path) + '/segmentations/' + root_name.replace('squished', 'seg.tif')
    
    
    # Straighten
    straightened = pvd.make_straightened(img_path, coord_path)
    tifffile.imwrite(straightened_path, straightened)
    print('Image straightened successfully.\n')
    
    
    # Make max intensity projection
    maxProj = straightened.max(axis = 0)
    tifffile.imwrite(maxProj_path, maxProj)
    print('Maximum intensity projection generated successfully.\n')
    
    
    # Make mask
    mask = pvd.get_big_mask3d(straightened, 
                            model = seg_model, 
                            compute_device = seg_compute_device, 
                            threshold = 0.1)
    mask = np.uint8(mask)
    tifffile.imwrite(mask_path, mask)
    print('Mask generated successfully.\n')
    
    
    # Classify branches
    
    fragments = pvd.classify_mask(mask, maxProj, classifier)
    print('Branch fragment classification complete.\n')
    
    fragments = pvd.correct_primary(fragments)
    
    while True:
        corrected_copy = fragments.copy()
        corrected_copy = pvd.correct_primary(fragments)
        if corrected_copy is not None:
            fragments = corrected_copy
        else:
            break
    
    fragments = pvd.correct_tertiary(fragments)
    print('Fragment classification corrections complete.\n')
    
    branch_stats = pvd.reconstructed_with_stats(fragments, maxProj)
    print('Branch reconstruction complete. Image processing complete.\n')
    
    return branch_stats

# %%
# For a batch of files

folder = Path('{dir_for_your_experiment}')
for in_path in folder.glob('*squished.tif'):
    out_path = str(Path(in_path).parent) + '/branches/' + Path(in_path).stem.replace('squished.tif', 'branches.csv')
    in_path = str(in_path)
    
    branches = img_to_branches(in_path, model, compute_device, classifier)
    branches.to_csv(out_path)


# %%
# For a single file

in_path = '{2x_downsampled_image_path}'
out_path = str(Path(in_path).parent) + '/branches/' + Path(in_path).stem.replace('squished.tif', 'branches.csv')

branches = img_to_branches(in_path, model, compute_device, classifier)
branches.to_csv(out_path)

