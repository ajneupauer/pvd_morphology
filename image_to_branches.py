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
neurite_seg_path = Path("./ml_models/20250613-pvdseg.pth")
compute_device = torch.device("mps")
neurite_seg_model = models.AttentionUNet(1, 1, features=[16, 32, 64, 128], use_logits=True)
unet_dict = torch.load(neurite_seg_path, weights_only=False)
neurite_seg_model.load_state_dict(unet_dict["model_state_dict"])
neurite_seg_model = neurite_seg_model.to(compute_device)

mito_seg_path = Path("./ml_models/20260117-mitoseg.pth")
mito_seg_model = models.AttentionUNet(1, 1, features=[16, 32, 64, 128], use_logits=True)
unet_dict = torch.load(mito_seg_path, weights_only=False)
mito_seg_model.load_state_dict(unet_dict["model_state_dict"])
mito_seg_model = mito_seg_model.to(compute_device)

classifier = pc1.PVDNeuriteClassifier()
classifier.load_model('./ml_models/class-3.joblib')

# Function
def img_to_branches(img_path, neurite_seg_model, mito_seg_model, compute_device, classifier):
    # Manage file paths
    coord_path = img_path.replace('_squished.tif','.npy')
    straightened_path = img_path.replace('squished', 'Straightened')
    mip_path = img_path.replace('squished', 'mip')
    mask_path = img_path.replace('squished', 'seg')
    
    # Step 1: Straighten
    straightened = pvd.make_straightened(img_path, coord_path)
    tifffile.imwrite(straightened_path, straightened, compression = 'lzw')
    print('Step 1 completed successfully:\nStraighten neurite image.\n')
    
    
    # Step 2a: Make max intensity projection
    mip = straightened.max(axis = 0)
    tifffile.imwrite(mip_path, mip, compression = 'lzw')
    print('Step 2a completed successfully:\nGenerate neurite MIP.\n')
    
    
    # Step 2b: Make mask
    mask3d = pvd.get_big_mask3d(straightened, 
                            model = neurite_seg_model, 
                            compute_device = compute_device, 
                            threshold = 0.1)
    mask3d = np.uint8(mask3d)
    mask2d = mask3d.max(axis = 0)
    tifffile.imwrite(mask_path, mask2d, compression = 'lzw')
    print('Step 2b completed successfully:\nSegment neurites.\n')
    
    
    # Step 3: Classify branches
    fragments = pvd.classify_mask(mask2d, mip, classifier)
    print('Step 3a completed successfully:\nClassify branch fragments.\n')
    G, node_data = pvd.reconstruct_graph_from_segments(fragments) # Get node data
    betweenness = pvd.calculate_betweenness_centrality(G)
    node_data['betweenness'] = betweenness
    loops = [pvd.count_loops(G)['total_cycles']]
    loops = loops + (len(node_data) - 1) * ['NA']
    node_data['loops'] = loops
    
    print(len(fragments))
    #fragments = pvd.correct_primary(fragments)
    
    
    while True:
        corrected_copy = fragments.copy()
        corrected_copy = pvd.correct_primary(fragments)
        if corrected_copy is not None:
            fragments = corrected_copy
        else:
            break
    
    fragments = pvd.correct_tertiary(fragments)
    print('Step 3b completed successfully:\nCorrect fragment classifications.\n')
    
    branch_stats = pvd.reconstructed_with_stats(fragments, mip)
    print('Step 3c completed successfully:\nReconstruct fragments into branches.\n')
    
    
    # Step 4: Straighten mito channel
    mito_straightened = pvd.make_straightened(img_path.replace('squished', 'mito_squished'), 
                                              coord_path, scale = 8)
    tifffile.imwrite(img_path.replace('squished', 'mito_Straightened'), 
                     mito_straightened, compression = 'lzw')
    print('Step 4 completed successfully:\nStraighten mito image.\n')
    
    
    # Step 5: Process mito image w/ neurite mask
    mito_rmbg = pvd.process_mito(mito_straightened, mask3d)
    tifffile.imwrite(img_path.replace('squished', 'mito_rmbg'), 
                     mito_rmbg, compression = 'lzw')
    print('Step 5 completed successfully:\nProcess mito image for segmentation.\n')
    
    
    # Step 6: Segment mitochondria
    mito_seg = pvd.get_big_mask(mito_rmbg, 
                                model = mito_seg_model, 
                                compute_device = compute_device, 
                                threshold = 0.2)
    mito_seg = np.uint8(mito_seg)
    tifffile.imwrite(img_path.replace('squished', 'mito_seg'), 
                     mito_seg, compression = 'lzw')
    print('Step 6 completed successfully:\nSegment mitochondria.\n')
    
    
    # Step 7: Generate mito data
    mito_stats = pvd.make_mito_df(mito_seg, branch_stats)
    print('Step 7 completed successfully:\nGenerate mito data.\nImage analysis complete.\n')
    
    return branch_stats, mito_stats, node_data

# %%
# For a batch of files

folder = Path('{dir_for_your_experiment}')
for in_path in folder.glob('*squished.tif'):
    if '._' in in_path.stem:
        continue
    if 'mito_' in in_path.stem:
        continue

    in_path = str(in_path)
    out_path1 = in_path.replace('squished.tif', 'branches.csv')
    out_path2 = in_path.replace('squished.tif', 'mito.csv')
    out_path3 = in_path.replace('squished.tif', 'nodes.csv')
    
    branches, mito_data, node_data = img_to_branches(in_path, neurite_seg_model, mito_seg_model, compute_device, classifier)

    branches.to_csv(out_path1)
    mito_data.to_csv(out_path2)
    node_data.to_csv(out_path3)
    
    print(f'Processed image: {in_path}')


# %%
# For a single file

in_path = '{path to 2x downsampled z-stack}'
out_path1 = in_path.replace('squished.tif', 'branches.csv')
out_path2 = in_path.replace('squished.tif', 'mito.csv')
out_path3 = in_path.replace('squished.tif', 'nodes.csv')

branches, mito_data, node_data = img_to_branches(in_path, neurite_seg_model, mito_seg_model, compute_device, classifier)

branches.to_csv(out_path1)
mito_data.to_csv(out_path2)
node_data.to_csv(out_path3)
