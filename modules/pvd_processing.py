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
import skimage
from scipy.spatial import cKDTree
from collections import defaultdict
import networkx as nx


# Image straightening
def make_straightened(image_file, coords, scale = 4):
    # load coordinates
    data = tifffile.imread(image_file)
    
    if len(data.shape) == 2:
        tmp = np.zeros([1, data.shape[0], data.shape[1]], dtype = np.uint16)
        tmp[0] = data
        data = tmp
    
    Nz, Ny, Nx = data.shape
    
    zc, yc, xc = compute_resampling_coordinates(coords, Nz, override_scale = scale)
    
    resampled = ndi.map_coordinates(
        data.astype(np.float32),
        (zc.ravel(), yc.ravel(), xc.ravel()),
        order=2,
    )
    resampled = resampled.reshape(zc.shape)
    
    return np.uint16(resampled)


# Get neuron segmentations
def get_seg_chunk_coords(image, offset = 5):
    if len(image.shape) == 2:
        height, width = image.shape
    else:
        depth, height, width = image.shape
    
    n_full_chunks_nonOL = height // width
    remainder_OL = height % width + offset * (2*n_full_chunks_nonOL)
    
    if remainder_OL < width:
        n_full_chunks_OL = n_full_chunks_nonOL
    else:
        n_full_chunks_OL = n_full_chunks_nonOL + 1
        
    chunk_coords = []
        
    for i in range(n_full_chunks_OL):
        lower = i * width - offset * (2*i)
        upper = lower + width - 1
        #print(f'Chunk {i} x-coords: {lower} to {upper}')
        chunk_coords.append({
            'lower_y': lower,
            'upper_y': upper
            })
    
    lower = 1 + upper - 2 * offset
    upper = height
    #print(f'Chunk {i} x-coords: {lower} to {upper}')
    chunk_coords.append({
        'lower_y': lower,
        'upper_y': upper
        })
    
    return pd.DataFrame(chunk_coords)

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
    
    mask = np.zeros([depth, height, width])
    
    for k in range(depth):
        mask[k] = get_mask(image3d[k], model = model, compute_device = compute_device, threshold = threshold)
    
    return mask  

def get_big_mask3d(big_img, model, compute_device, threshold = None):
    # Load image and dimensions
    depth, height, width = big_img.shape
    coords = get_seg_chunk_coords(big_img)
    
    # Initialize big mask with the first chunk
    chunk = big_img[:, :coords['upper_y'][0] + 1, :]
    merged = get_mask3d(chunk, model = model, compute_device = compute_device, threshold = threshold)
    merged[:, :, :5] = 0 # set left edge to zero
    merged[:, :, width - 5:] = 0 # set right edge to zero
    merged[:, :5, :] = 0 # set top edge to zero
    merged = merged[:, :width - 5, :] # trim bottom edge
    
    # Loop through remaining chunks, except the final one
    for idx, row in coords.iterrows():
        if idx == 0 or idx == len(coords) - 1:
            continue
        chunk = big_img[:, row['lower_y']:row['upper_y'] + 1, :]
        mask = get_mask3d(chunk, model = model, compute_device = compute_device, threshold = threshold)
        mask[:, :, :5] = 0 # set left edge to zero
        mask[:, :, width - 5:] = 0 # set right edge to zero
        mask = mask[:, 5:width - 5, :] # trim bottom and top edges
        merged = np.concatenate((merged, mask), axis = 1)
    
    # Get mask of final chunk
    chunk = big_img[:, coords['lower_y'][len(coords) - 1]:height, :]
    mask = get_mask3d(chunk, model = model, compute_device = compute_device, threshold = threshold)
    height = mask.shape[1]
    mask[:, :, :5] = 0 # set left edge to zero
    mask[:, :, width - 5:] = 0 # set right edge to zero
    mask[:, height - 5:, :] = 0 # set bottom edge to zero
    mask = mask[:, 5:, :] # trim top edge
    merged = np.concatenate((merged, mask), axis = 1)
    
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
    #maxProj = tifffile.imread(maxProj)
    #maxProj_norm = maxProj // maxProj.max()
    
    for branch_set in [prim_branches, sec_branches, tert_branches, quat_branches]:
        for branch in branch_set:
            # Get start and end points (for neighbor assignment)
            start.append(branch[0])
            end.append(branch[-1])
            
            # Calculate branch length, orientation, curvature, tortuosity/waviness
            length, orientation, curvature, tortuosity, waviness = pc1.branch_geom(branch)
            
            # Branch coordinates
            x_pos = [pt[1] for pt in branch]
            y_pos = [pt[0] for pt in branch]
            x = np.mean(x_pos)
            y = np.mean(y_pos)
            
            # Average intensity
            intensities = []
            for i in range(len(branch)):
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
                'waviness': waviness,
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


# Process mito image for segmentation
def process_mito(mito, mask):
    mask = mask > 0
    mask = skimage.morphology.remove_small_objects(mask, min_size = 150)

    mask = skimage.transform.rescale(mask, (1, 2, 2))

    padded_mask = ndi.binary_dilation(mask, iterations = 4)
    z, y, x = np.where(padded_mask == 0)
    outImg = np.copy(mito)
    for i in range(len(y)):
        outImg[z[i], y[i], x[i]] = 100
    
    return outImg.max(axis = 0) 


# Segment mitochondria
def get_big_mask(big_img, model, compute_device, threshold = None):
    # Load image and dimensions
    height, width = big_img.shape 
    coords = get_seg_chunk_coords(big_img)
    
    # Initialize big mask with the first chunk
    chunk = big_img[:coords['upper_y'][0] + 1, :]
    merged = get_mask(chunk, model = model, compute_device = compute_device, threshold = threshold)
    merged[:, :5] = 0 # set left edge to zero
    merged[:, width - 5:] = 0 # set right edge to zero
    merged[:5, :] = 0 # set top edge to zero
    merged = merged[:width - 5, :] # trim bottom edge
    
    # Loop through remaining chunks, except the final one
    for idx, row in coords.iterrows():
        if idx == 0 or idx == len(coords) - 1:
            continue
        chunk = big_img[row['lower_y']:row['upper_y'] + 1, :]
        mask = get_mask(chunk, model = model, compute_device = compute_device, threshold = threshold)
        mask[:, :5] = 0 # set left edge to zero
        mask[:, width - 5:] = 0 # set right edge to zero
        mask = mask[5:width - 5, :] # trim bottom and top edges
        merged = np.concatenate((merged, mask), axis = 0)
    
    # Get mask of final chunk
    chunk = big_img[coords['lower_y'][len(coords) - 1]:height, :]
    mask = get_mask(chunk, model = model, compute_device = compute_device, threshold = threshold)
    height = mask.shape[0]
    mask[:, :5] = 0 # set left edge to zero
    mask[:, width - 5:] = 0 # set right edge to zero
    mask[height - 5:, :] = 0 # set bottom edge to zero
    mask = mask[5:, :] # trim top edge
    merged = np.concatenate((merged, mask), axis = 0)
    
    return merged


# Generate mitochondria data table
def extract_foci_with_properties(mask):
    """
    Alternative using skimage.measure.regionprops for richer properties.
    """
    labeled_mask = skimage.measure.label(mask, connectivity = 2)
    regions = skimage.measure.regionprops(labeled_mask)
    
    foci_data = []
    for region in regions:
        centroid = region.centroid
        centroid = tuple(round(a // 2) for a in centroid)
        foci_data.append({
            'label': region.label,
            'size': region.area * (0.1048 ** 2),
            'centroid': centroid,
            'centroid_x': centroid[1],
            'centroid_y': centroid[0],
            'maj_axis': region.axis_major_length * 0.1048,
            'min_axis': region.axis_minor_length * 0.1048,
            'eccentricity': region.eccentricity,
            'equiv_diam_area': region.equivalent_diameter_area * 0.1048, # diameter of circle w/ same area as region
            'extent': region.extent, # region area / bbox area
            'perim': region.perimeter * 0.1048,
            'perim_area_ratio': region.perimeter / (region.area * 0.1048),
            'orientation': region.orientation * 180 / np.pi,
            'feret_diam': region.feret_diameter_max * 0.1048
            #'intensity_mean': region.intensity_mean if hasattr(region, 'intensity_mean') else None
        })
    
    return pd.DataFrame(foci_data), labeled_mask

def assign_mitochondria_to_branches(foci_data, dendrite_df, max_distance=10):
    """
    Assign each mitochondrion to the nearest dendrite branch.
    
    Parameters:
    -----------
    foci_data : list of dicts
        Mitochondria data with 'centroid' and other properties
    dendrite_df : pandas DataFrame
        Must have columns: 'branch_id', 'branch_type', 'skeleton_coords'
        where skeleton_coords is a list/array of (row, col) coordinates
    max_distance : float, optional
        Maximum distance to consider a mitochondrion as belonging to a branch.
        If None, always assigns to nearest branch regardless of distance.
    
    Returns:
    --------
    pandas DataFrame with mitochondria properties plus branch assignment
    """
    
    # Build a lookup structure for all dendrite skeleton points
    all_skeleton_points = []
    point_to_branch = []  # Maps each skeleton point to branch info
    
    for idx, row in dendrite_df.iterrows():
        skeleton_coords = np.array(row['branch'])
        #print(skeleton_coords.shape)
        all_skeleton_points.append(skeleton_coords)
        
        # Store branch info for each point
        for i in range(len(skeleton_coords)):
            point_to_branch.append({
                'branch_id': row['id'],
                'branch_type': row['dendrite_type'],
                'branch_idx': idx  # Index in original dataframe
            })
    
    # Concatenate all skeleton points
    all_skeleton_points = np.vstack(all_skeleton_points)
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(all_skeleton_points)
    
    # Process each mitochondrion
    results = []
    for idx, focus in foci_data.iterrows():
        centroid = np.array(focus['centroid']).reshape(1, -1)
        
        # Find nearest skeleton point
        distance, nearest_idx = tree.query(centroid)
        distance = distance[0]
        nearest_idx = nearest_idx[0]
        
        # Get branch information
        branch_info = point_to_branch[nearest_idx]
        
        # Check if within max_distance threshold
        if max_distance is not None and distance > max_distance:
            branch_id = None
            branch_type = 'unassigned'
            distance = None
        else:
            branch_id = branch_info['branch_id']
            branch_type = branch_info['branch_type']
        
        # Combine focus data with branch assignment
        result = {
            'mito_label': focus['label'],
            #'centroid_row': focus['centroid'][0],
            #'centroid_col': focus['centroid'][1],
            #'branch_id': branch_id,
            'branch_type': branch_type,
            #'distance_to_branch': distance,
            **{k: v for k, v in focus.items() 
               if k not in ['label', 'centroid', 'coordinates']}
        }
        results.append(result)
    
    return pd.DataFrame(results)

def make_mito_df(mask, dendrites):
    myFociData, myMaskLabeled = extract_foci_with_properties(mask)
    
    branch_assignment = assign_mitochondria_to_branches(myFociData, dendrites)
    filtered = branch_assignment[(branch_assignment['branch_type'] != 'unassigned') & (branch_assignment['size'] > 2 * (0.1048 ** 2))]
    
    size_mean = np.mean(filtered['size'])
    size_sd = np.std(filtered['size'])
    size_zscore = (filtered['size'] - size_mean)/size_sd
    filtered['size_zscore'] = size_zscore
    
    return filtered


# Clean nodes tables to remove repeats
def cleanup_nodes(nodes):

    terminal = nodes[nodes['degree'] == 1]
    intersections = nodes[nodes['degree'] != 1]
    intersections = intersections.reset_index(drop = True)
    
    final_ints = []
    skip = False
    
    for i in range(len(intersections) - 1):
        if skip:
            subset = intersections.iloc[i+1, :]
            last_entry = final_ints[-1]
            # check if the most recent pt added to final_ints is in range of the next pt
            if (subset['x_pos'] <= last_entry['x_pos'] + 2) and (subset['x_pos'] >= last_entry['x_pos'] - 2):
                # edit most recent pt in final_ints to take the x_pos of the next pt if it's smaller
                if subset['x_pos'] < last_entry['x_pos']:
                    last_entry['x_pos'] = subset['x_pos']
                # edit most recent pt in final_ints to take the degree of the next pt if it's larger
                if subset['degree'] > last_entry['degree']:
                    last_entry['degree'] = subset['degree']
                # go to the next iteration
                continue
            # if the next pt is not in range, go to the next iteration
            skip = False
            continue
        subset = intersections.iloc[i:i+2, :]
        subset = subset.reset_index(drop = True)
        # take current pt if next pt is too far away in x
        if (subset['x_pos'][0] > subset['x_pos'][1] + 2) or (subset['x_pos'][0] < subset['x_pos'][1] - 2):
            x_pos, y_pos, degree = subset.iloc[0, :]
            skip = False
        # take current pt if next pt is too far away in y
        elif (subset['y_pos'][0] > subset['y_pos'][1] + 2) or (subset['y_pos'][0] < subset['y_pos'][1] - 2):
            x_pos, y_pos, degree = subset.iloc[0, :]
            skip = False
        else:
            # compare x coords in both pts and pick the minimum
            if subset['x_pos'][0] == subset['x_pos'][1]:
                x_pos = subset['x_pos'][0]
            else:
                x_pos = min(subset['x_pos'])
            # compare y coords in both pts and pick the minimum
            if subset['y_pos'][0] == subset['y_pos'][1]:
                y_pos = subset['y_pos'][0]
            else:
                y_pos = min(subset['y_pos'])
            # compare degree in both pts and pick the maximum
            if subset['degree'][0] == subset['degree'][1]:
                degree = subset['degree'][0]
            else:
                degree = max(subset['degree'])
            # trigger a skip in the next iteration
            skip = True
        
        final_ints.append({
            'x_pos': x_pos,
            'y_pos': y_pos,
            'degree': degree
            })
    
    final_ints = pd.DataFrame(final_ints)
    
    # nodes table was originally sorted in y, so only adjacent pts with the same y were in neighboring entries
    # to remove all 'repeated' points, sort the filtered table in x, and filter again
    final_ints = final_ints.sort_values(['x_pos']).reset_index(drop = True)
    
    final_final_ints = []
    skip = False
    
    for i in range(len(final_ints) - 1):
        if skip:
            subset = final_ints.iloc[i+1, :]
            last_entry = final_final_ints[-1]
            # check if the most recent pt added to final_final_ints is in range of the next pt
            if (subset['x_pos'] <= last_entry['x_pos'] + 2) and (subset['x_pos'] >= last_entry['x_pos'] - 2):
                # edit most recent pt in final_final_ints to take the x_pos of the next pt if it's smaller
                if subset['x_pos'] < last_entry['x_pos']:
                    last_entry['x_pos'] = subset['x_pos']
                # edit most recent pt in final_final_ints to take the degree of the next pt if it's larger
                if subset['degree'] > last_entry['degree']:
                    last_entry['degree'] = subset['degree']
                # go to the next iteration
                continue
            # if the next pt is not in range, go to the next iteration
            skip = False
            continue
        subset = final_ints.iloc[i:i+2, :]
        subset = subset.reset_index(drop = True)
        # take current pt if next pt is too far away in x
        if (subset['x_pos'][0] > subset['x_pos'][1] + 2) or (subset['x_pos'][0] < subset['x_pos'][1] - 2):
            x_pos, y_pos, degree = subset.iloc[0, :]
            skip = False
        # take current pt if next pt is too far away in y
        elif (subset['y_pos'][0] > subset['y_pos'][1] + 2) or (subset['y_pos'][0] < subset['y_pos'][1] - 2):
            x_pos, y_pos, degree = subset.iloc[0, :]
            skip = False
        else:
            if subset['x_pos'][0] == subset['x_pos'][1]:
                x_pos = subset['x_pos'][0]
            else:
                x_pos = min(subset['x_pos'])
            if subset['y_pos'][0] == subset['y_pos'][1]:
                y_pos = subset['y_pos'][0]
            else:
                y_pos = min(subset['y_pos'])
            if subset['degree'][0] == subset['degree'][1]:
                degree = subset['degree'][0]
            else:
                degree = max(subset['degree'])
            # trigger a skip in the next iteration
            skip = True
        
        final_final_ints.append({
            'x_pos': x_pos,
            'y_pos': y_pos,
            'degree': degree
            })
    
    final_final_ints = pd.DataFrame(final_final_ints)
    filtered_nodes = pd.concat((final_final_ints, terminal))
    return filtered_nodes

# Network/graph operations
def reconstruct_graph_from_segments(df, proximity_threshold=2):
    """
    Reconstruct a dendritic arbor graph from branch data.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file with branch data
    proximity_threshold : float
        Distance threshold for merging nearby endpoints (in pixels).
        This handles multi-pixel branch points.
    
    Returns:
    --------
    G : networkx.Graph
        Graph where nodes are branch points/endpoints and edges are branches
    node_positions : dict
        Dictionary mapping node IDs to (x, y) coordinates
    """
    
    # Load branch data
    #df = pd.read_csv(csv_path, index_col=0)
    
    # Parse segment strings into lists of tuples
    #df['segment'] = df['segment'].apply(ast.literal_eval)
    
    # Step 1: Extract all potential nodes (branch endpoints)
    endpoints = []
    for idx, row in df.iterrows():
        segment = row['segment']
        # Start and end of each branch
        endpoints.append(segment[0])
        endpoints.append(segment[-1])
    
    # Step 2: Cluster nearby endpoints to handle multi-pixel branch points
    node_positions = {}
    endpoint_to_node = {}
    node_counter = 0
    
    for endpoint in endpoints:
        # Check if this endpoint is close to any existing node
        merged = False
        for node_id, pos in node_positions.items():
            dist = np.sqrt((endpoint[0] - pos[0])**2 + (endpoint[1] - pos[1])**2)
            if dist <= proximity_threshold:
                # Merge with existing node
                endpoint_to_node[endpoint] = node_id
                # Update node position to centroid (optional refinement)
                merged = True
                break
        
        if not merged:
            # Create new node
            node_positions[node_counter] = endpoint
            endpoint_to_node[endpoint] = node_counter
            node_counter += 1
    
    # Refine node positions to centroids of merged endpoints
    node_endpoint_lists = defaultdict(list)
    for endpoint, node_id in endpoint_to_node.items():
        node_endpoint_lists[node_id].append(endpoint)
    
    for node_id, endpoint_list in node_endpoint_lists.items():
        # Calculate centroid
        centroid = (
            int(np.mean([e[0] for e in endpoint_list])),
            int(np.mean([e[1] for e in endpoint_list]))
        )
        node_positions[node_id] = centroid
    
    # Step 3: Build the graph
    G = nx.Graph()
    
    # Add nodes with their properties
    for node_id, pos in node_positions.items():
        G.add_node(node_id, pos=pos, x=pos[1], y=pos[0])
    
    # Add edges (one per branch)
    for idx, row in df.iterrows():
        segment = row['segment']
        start_point = segment[0]
        end_point = segment[-1]
        
        start_node = endpoint_to_node[start_point]
        end_node = endpoint_to_node[end_point]
        
        # Add edge with branch properties
        G.add_edge(
            start_node,
            end_node,
            branch_id=row['id'],
            segment=segment
            #length=row['length'],
            #orientation=row['orientation'],
            #curvature=row['curvature'],
            #tortuosity=row['tortuosity'],
            #waviness=row['waviness']
        )
    
    # Step 4: Classify nodes
    node_data = []
    
    for node_id in G.nodes():
        degree = G.degree(node_id)
        if degree == 1:
            G.nodes[node_id]['node_type'] = 'endpoint'
            node_type = 'endpoint'
        elif degree == 2:
            G.nodes[node_id]['node_type'] = 'continuation'
            node_type = 'continuation'  # Unusual, but possible
        else:
            G.nodes[node_id]['node_type'] = 'branch_point'
            node_type = 'branch_point'
        node_data.append({
            'id': node_id,
            'type': node_type,
            'degree': degree,
            'x_pos': G.nodes[node_id]['x'],
            'y_pos': G.nodes[node_id]['y']
            })
    
    return G, pd.DataFrame(node_data)

def calculate_betweenness_centrality(G):
    """
    Calculate betweenness centrality for all nodes in the graph.
    
    Betweenness centrality measures how often a node lies on the shortest path
    between other nodes. High centrality = critical junction point.
    
    Parameters:
    -----------
    G : networkx.Graph
        The dendritic arbor graph
    
    Returns:
    --------
    betweenness : dict
        Dictionary mapping node_id -> betweenness centrality value
    """
    
    # For disconnected graphs, calculate per component
    if not nx.is_connected(G):
        print("Graph has multiple components. Calculating centrality per component...")
        betweenness = {}
        
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(subgraph) > 2:  # Need at least 3 nodes for meaningful centrality
                component_centrality = nx.betweenness_centrality(subgraph)
                betweenness.update(component_centrality)
            else:
                # For tiny components, set centrality to 0
                for node in component:
                    betweenness[node] = 0.0
    else:
        # Single connected component
        betweenness = nx.betweenness_centrality(G)
    
    return betweenness

def count_loops(G):
    """
    Count loops (cycles) in the dendritic graph.
    
    Multiple approaches:
    1. Minimum cycle basis - fundamental cycles
    2. All simple cycles - all distinct loops
    3. Cycle count statistics
    
    Parameters:
    -----------
    G : networkx.Graph
        The dendritic arbor graph
    
    Returns:
    --------
    results : dict
        Dictionary with loop counts and cycle information
    """
    
    results = {
        'total_components': nx.number_connected_components(G),
        'cycles_per_component': [],
        'cycle_lengths': [],
        'all_cycles': [],
        'cycles_by_component': {}  # Maps component_idx -> (cycles, component_nodes)
    }
    
    #print("\nLoop Detection Analysis")
    #print("=" * 60)
    
    # Analyze each connected component
    for i, component in enumerate(nx.connected_components(G)):
        subgraph = G.subgraph(component)
        component_size = len(subgraph)
        
        # Check if component can have cycles
        # A tree with n nodes has n-1 edges. Extra edges = cycles
        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()
        n_cycles_min = n_edges - n_nodes + 1  # Cyclomatic complexity
        
        if n_cycles_min > 0:
            #print(f"\nComponent {i} (size {component_size}):")
            #print(f"  Nodes: {n_nodes}, Edges: {n_edges}")
            #print(f"  Minimum cycles (cyclomatic complexity): {n_cycles_min}")
            
            # Find minimum cycle basis
            try:
                cycle_basis = nx.minimum_cycle_basis(subgraph)
                #print(f"  Fundamental cycles found: {len(cycle_basis)}")
                
                component_cycles = []
                for j, cycle in enumerate(cycle_basis):
                    cycle_length = len(cycle)
                    results['cycle_lengths'].append(cycle_length)
                    results['all_cycles'].append(cycle)
                    component_cycles.append(cycle)
                    #print(f"    Cycle {j+1}: {cycle_length} nodes")
                
                # Store cycles with their component
                results['cycles_by_component'][i] = (component_cycles, list(component))
                results['cycles_per_component'].append(len(cycle_basis))
                
            except nx.NetworkXError as e:
                print(f"  Error finding cycles: {e}")
                results['cycles_per_component'].append(0)
        else:
            # Tree structure - no cycles
            #if component_size > 1:
                #print(f"\nComponent {i} (size {component_size}): Tree structure (no cycles)")
            results['cycles_per_component'].append(0)
    
    # Summary statistics
    total_cycles = sum(results['cycles_per_component'])
    results['total_cycles'] = total_cycles
    
    #print("\n" + "=" * 60)
    #print(f"SUMMARY:")
    #print(f"  Total cycles found: {total_cycles}")
    
    #if total_cycles > 0:
        #print(f"  Cycle length range: {min(results['cycle_lengths'])} to {max(results['cycle_lengths'])} nodes")
        #print(f"  Average cycle length: {np.mean(results['cycle_lengths']):.1f} nodes")
        #print(f"  Components with cycles: {sum(1 for x in results['cycles_per_component'] if x > 0)}")
    #else:
        #print("  Graph is a forest (collection of trees) - no cycles detected")
    
    return results