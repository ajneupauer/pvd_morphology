#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 13:16:08 2025

@author: alexneupauer
"""

# Import modules
import sys
sys.path.append('starr-luxton-lab/pvd-project/scripts/modules')
sys.path.append('starr-luxton-lab/pvd-project/UNets-torch/src/unets_torch')

import numpy as np
import scipy.ndimage as ndi
import tifffile
from straightening_utils import compute_resampling_coordinates
import torch
import torch.nn.functional as F
import models
import pvd_classifier_1 as pc1
import branch_reconstructor as br
import pandas as pd



# Image straightening
def make_straightened(image_file, coords):
    # load coordinates
    data = tifffile.imread(image_file)
    Nz, Ny, Nx = data.shape
    
    zc, yc, xc = compute_resampling_coordinates(coords, Nz, override_scale = 4)
    
    resampled = ndi.map_coordinates(
        data.astype(np.float32),
        (zc.ravel(), yc.ravel(), xc.ravel()),
        order=2,
    )
    resampled = resampled.reshape(zc.shape)
    
    return np.uint16(resampled)


# Get neuron segmentations
def get_mask(image, model, compute_device, threshold = None):
    # Make sure the input array is np.float32
    image = image.astype(np.float32)
    
    # calculate image shape divisible by 2^4
    calc_valid_dim = lambda n: int(((n + 2**4 - 1) // 2**4) * 2**4)
    valid_shape = [calc_valid_dim(s) for s in image.shape]
    pad_shape = [int(v - s) for v, s in zip(valid_shape, image.shape)]
     
    # normalize image by percentile (same as training)
    ilow, ihigh = np.percentile(image, (1.0, 99.0))
    image = (image - ilow) / (ihigh - ilow)
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image.astype(np.float32))[None, None, ...].to(
        compute_device
    )
    
    # pad image
    image_tensor = F.pad(image_tensor, (0, pad_shape[1], 0, pad_shape[0], 0, 0))
    
    # Generate prediction
    with torch.no_grad():
        logits = model(image_tensor)
        # Since use_logits=True, we need to apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits)
    
    # Convert to numpy array and
    # Remove batch and channel dims; also unpad   
    prob_map = probabilities.cpu().numpy()[
        0, 0, 0 : valid_shape[0] - pad_shape[0], 0 : valid_shape[1] - pad_shape[1]
    ]
    
    if threshold is None:
        mask = prob_map
    else:
        mask = prob_map > threshold
    
    return mask

def get_mask3d(image3d, model, compute_device, threshold = None):
    
    depth, height, width = image3d.shape
    
    mask = np.zeros([height, width])
    
    for k in range(0, depth):
        curr_mask = get_mask(image3d[k], model = model, compute_device = compute_device, threshold = threshold)
        mask = np.maximum(curr_mask, mask)
    
    return mask 

def get_big_mask3d(big_img, model, compute_device, threshold = None):
    # Load image and dimensions
    height = big_img.shape[1]
    width = big_img.shape[2]
    
    merged = np.empty([0, width])
    
    for i in range(0, height // width):
        y_coords = (width * i, width * (i + 1))
        chunk = big_img[:, y_coords[0]:y_coords[1], :]
        mask = get_mask3d(chunk, model = model, compute_device = compute_device, threshold = threshold)
        merged = np.vstack((merged, mask))
        
    chunk = big_img[:, y_coords[1]:height, :]
    mask = get_mask3d(chunk, model = model, compute_device = compute_device, threshold = threshold)
    merged = np.vstack((merged, mask))
    
    return merged


# Classify branches
def branch_geom(branch, interval = 7):
    dys = []
    dxs = []
    angles = []
    
    for n in range(len(branch) // interval):
        startpt = branch[interval * n]
        endpt = branch[interval * n + interval - 1]
        dy, dx = startpt[0] - endpt[0], endpt[1] - startpt[1]
        angles.append(np.arctan2(dy, dx) * 180 / np.pi)
        dys.append(dy)
        dxs.append(dx)
    
    mean_dy = np.mean(dys)
    mean_dx = np.mean(dxs)
    
    if len(branch) >= 2 * interval:
        angle_diffs = []
        sign = None
        sign_switches = 0
        
        for i in range(len(angles) - 1):
            delta_angle = angles[i + 1] - angles[i]
            if delta_angle >= 0:
                new_sign = 'pos'
            else: 
                new_sign = 'neg'
            if sign is not None and sign != new_sign:
                sign_switches += 1
            sign = new_sign
            angle_diff = min(abs(delta_angle), 360 - abs(delta_angle))
            angle_diffs.append(angle_diff)

        curvature = np.mean(angle_diffs)
        waviness = curvature * sign_switches / len(branch)
    else:
        curvature = 0
        waviness = 0
        
    mean_orientation = np.arctan2(mean_dy, mean_dx) * 180 / np.pi
    if mean_orientation < 0:
        mean_orientation += 180
    
    return (mean_orientation, curvature, waviness)

def val_accuracy(color_map, pred_results, weighted=False):
    labels = pc1.get_labels(color_map, pred_results)
    
    n_correct = 0
    len_correct = 0
    
    for i in range(len(labels)):
        if labels[i] == pred_results.loc[i]['dendrite_type']:
            if weighted:
                len_correct += pred_results.loc[i]['length']
            else:
                n_correct += 1
    
    if weighted:
        len_tot = sum(pred_results['length'])
        accuracy = len_correct/len_tot
    else:
        accuracy = n_correct/len(labels)
    
    return accuracy

def classify_mask(mask, maxProj, model):
    results = model.predict(mask, maxProj)
    branch_data = model.branch_data
    results['length'] = branch_data['length']
    results['segment'] = branch_data['segment']
    segments = results.iloc[:, [0, 12, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]]
    #results.to_csv(mask_path.replace('_seg.tif', '_classification.csv'))
    #segments = results.iloc[:, [0, 1, 2, 3]]
    return segments

def correct_tertiary(fragments):
    corrected = fragments.copy()

    for ref_idx, ref_row in fragments.iterrows():
        if ref_row['dendrite_type'] != 3:
            continue
        
        ref_rel_x = ref_row['relative_x']
        ref_startpt_y = ref_row['segment'][0][0]
        ref_endpt_y = ref_row['segment'][-1][0]
        ref_id = ref_row['id']

        for srch_idx, srch_row in fragments.iterrows():
            if srch_row['dendrite_type'] != 3 or srch_row['id'] == ref_id:
                continue
            srch_rel_x = srch_row['relative_x']
            segment = srch_row['segment']
            y_pos = []
            for i in range(len(segment)):
                y_pos.append(segment[i][0])
            startpt_inRange = ref_startpt_y >= min(y_pos) and ref_startpt_y <= max(y_pos)
            endpt_inRange = ref_endpt_y >= min(y_pos) and ref_endpt_y <= max(y_pos)
            if startpt_inRange or endpt_inRange:
                if ref_rel_x > 0.7 and srch_rel_x > 0.6:
                    if ref_rel_x >= srch_rel_x:
                        #print(f'Classification error detected for branch {ref_id}. Correcting.')
                        corrected.loc[ref_id, 'dendrite_type'] = 4
                        break
                if ref_rel_x < 0.3 and srch_rel_x < 0.4:
                    if ref_rel_x <= srch_rel_x:
                        #print(f'Classification error detected for branch {ref_id}. Correcting.')
                        corrected.loc[ref_id, 'dendrite_type'] = 4
                        break
        
    return corrected

def correct_primary(fragments):
    corrected = fragments.copy()
    mistakes = False
    
    prim_only = fragments[fragments['dendrite_type'] == 1]
    prim_start = [segment[0] for segment in prim_only['segment']]
    prim_end = [segment[-1] for segment in prim_only['segment']]
    
    for ref_idx, ref_row in fragments.iterrows():
        if ref_row['dendrite_type'] != 3:
            continue
        
        ref_rel_x = ref_row['relative_x']
        if ref_rel_x <= 0.3 or ref_rel_x >= 0.7:
            continue
        
        touching_primary = False
        left_neighbor = False
        right_neighbor = False
        
        ref_startpt_x = ref_row['segment'][0][1]
        ref_startpt_y = ref_row['segment'][0][0] 
        ref_endpt_x = ref_row['segment'][-1][1]
        ref_endpt_y = ref_row['segment'][-1][0] 
        ref_id = ref_row['id']
        
        padding = 2
        
        for i in range(ref_startpt_x - padding, ref_startpt_x + padding + 1):
            for j in range(ref_startpt_y - padding, ref_startpt_y + padding + 1):
                if (j, i) in prim_end or (j, i) in prim_start:
                    touching_primary = True
        
        for i in range(ref_endpt_x - padding, ref_endpt_x + padding + 1):
            for j in range(ref_endpt_y - padding, ref_endpt_y + padding + 1):
                if (j, i) in prim_end or (j, i) in prim_start:
                    touching_primary = True
        
        if touching_primary == False:
            continue
        
        for srch_idx, srch_row in fragments.iterrows():
            if left_neighbor == True and right_neighbor == True:
                break
            if srch_row['dendrite_type'] != 3 and srch_row['dendrite_type'] != 1: # or srch_row['id'] == ref_id:
                continue
            srch_rel_x = srch_row['relative_x']
            segment = srch_row['segment']
            y_pos = []
            for i in range(len(segment)):
                y_pos.append(segment[i][0])
            startpt_inRange = ref_startpt_y >= min(y_pos) and ref_startpt_y <= max(y_pos)
            endpt_inRange = ref_endpt_y >= min(y_pos) and ref_endpt_y <= max(y_pos)
            if startpt_inRange or endpt_inRange:
                if 0.625 > srch_rel_x:
                    left_neighbor = True
                else:
                    right_neighbor = True
        
        if right_neighbor == True and left_neighbor == True:
            corrected.loc[ref_id, 'dendrite_type'] = 1    
            mistakes = True
    
    if mistakes == False: 
        print("Nothing to correct")
        corrected = None
        
    return corrected

def reconstructed_with_stats(fragments, maxProj):
    # Reconstruct branches
    prim_fragments = list(fragments[fragments['dendrite_type'] == 1]['segment'])
    prim_branches = br.connect_segments(prim_fragments, threshold=20.0, max_step_ratio = 10.0)
    sec_fragments = list(fragments[fragments['dendrite_type'] == 2]['segment'])
    sec_branches = br.connect_segments(sec_fragments, threshold=20.0, max_step_ratio = 10.0)
    tert_fragments = list(fragments[fragments['dendrite_type'] == 3]['segment'])
    tert_branches = br.connect_segments(tert_fragments, threshold=20.0, max_step_ratio = 10.0)
    quat_fragments = list(fragments[fragments['dendrite_type'] == 4]['segment'])
    quat_branches = br.connect_segments(quat_fragments, threshold=20.0, max_step_ratio = 10.0)
    
    # Collect stats on branches
    branch_data = []
    start = []
    end = []
    n = 0
    dendrite_type = 1
    
    for branch_set in [prim_branches, sec_branches, tert_branches, quat_branches]:
        for branch in branch_set:
            # Calculate branch length
            length = len(branch)
            
            # Get start and end points (for neighbor assignment)
            start.append(branch[0])
            end.append(branch[-1])
            
            # Calculate branch orientation, curvature, tortuosity
            orientation, curvature, tortuosity = branch_geom(branch)
            
            # Branch coordinates
            x_pos = [pt[1] for pt in branch]
            y_pos = [pt[0] for pt in branch]
            x = np.mean(x_pos)
            y = np.mean(y_pos)
            
            # Average intensity
            intensities = []
            for i in range(length):
                intensities.append(maxProj[y_pos[i], x_pos[i]])
            avg_intensity = np.mean(intensities)
            
            branch_data.append({
                'id': n,
                'branch': branch,
                'dendrite_type': dendrite_type,
                'length': length,
                'orientation': orientation,
                'mean_x': x,
                'mean_y': y,
                'curvature': curvature,
                'tortuosity': tortuosity,
                'intensity': avg_intensity
                })
            
            n += 1
        dendrite_type += 1    
    
    branch_data = pd.DataFrame(branch_data)
    
    # Add neighbor information
    branches = branch_data['branch']
    padding = 2
    allBranch_neighbors = []

    for ref_branch in branches:

        neighbors = []
        
        for pt in ref_branch:
            for i in range(pt[1] - padding, pt[1] + padding + 1):
                for j in range(pt[0] - padding, pt[0] + padding + 1):
                    if (j, i) in end:
                        neighbor = end.index((j, i))
                        if neighbor not in neighbors:
                            neighbors.append(neighbor)
                    if (j, i) in start:
                        neighbor = start.index((j, i))
                        if neighbor not in neighbors:
                            neighbors.append(neighbor)
        
        allBranch_neighbors.append(neighbors)    

    branch_data['neighbors'] = allBranch_neighbors
    
    return branch_data
