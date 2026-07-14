#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 13:16:08 2025

@author: alexneupauer
"""

# Import modules
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
import numpy as np
import scipy.ndimage as ndi
import tifffile
import torch
import torch.nn.functional as F
import pandas as pd
import skimage
from scipy.spatial import cKDTree
from collections import defaultdict
import networkx as nx

import sys
sys.path.append('./modules')
import pvd_classifier_1 as pc1
import branch_reconstructor as br


"""Part I: Neurite Segmentation"""

"""
Get the y-axis bounds of square chunks of a full PVD image.
    Square chunks have sides = image width.
    Adjacent chunks overlap by (2)*(offset) pixels.
    The final chunk may not be square.
"""
def get_seg_chunk_coords(image: np.ndarray, offset = 5) -> pd.DataFrame:
    # Get image height, width for 2D or 3D images
    if len(image.shape) == 2:
        height, width = image.shape
    else:
        depth, height, width = image.shape
    
    # Number of full square chunks without overlap
    n_full_chunks_nonOL = height // width
    # When we take that same number of square chunks but overlap them,
    # What is the discrepancy in length compared to if they were laid side-by-side?
    remainder_OL = height % width + offset * (2*n_full_chunks_nonOL)
    # Number of full square chunks with overlap
    if remainder_OL < width:
        n_full_chunks_OL = n_full_chunks_nonOL
    else: # Add one more square chunk if the discrepancy/remainder exceeds square side length
        n_full_chunks_OL = n_full_chunks_nonOL + 1
        
    chunk_coords = []

    # Each row in the DataFrame has lower and upper y-axis bounds for a chunk with respect to the original image    
    for i in range(n_full_chunks_OL):
        # Ex: for chunks of side length 100 and offset 5, lower bound = 0, 90, 180, ...
        lower = i * width - offset * (2*i)
        upper = lower + width - 1 # Upper bound guarantees y-length of side length
        chunk_coords.append({
            'lower_y': lower,
            'upper_y': upper
            })
    # Bounds of final (non-square) chunk
    lower = 1 + upper - 2 * offset
    upper = height
    
    chunk_coords.append({
        'lower_y': lower,
        'upper_y': upper
        })
    
    return pd.DataFrame(chunk_coords)

"""
Use a UNet model to produce a binary mask of a small 2D image.
"""
def get_mask(
        image: np.ndarray, 
        model: models.AttentionUNet, 
        compute_device: torch.device, 
        threshold = None
        ) -> np.ndarray:
    # Make sure the input array is of type np.float32
    image = image.astype(np.float32)
    
    # Calculate image shape divisible by 16 (2^4)
    calc_valid_dim = lambda n: int(((n + 2**4 - 1) // 2**4) * 2**4)
    valid_shape = [calc_valid_dim(s) for s in image.shape]
    pad_shape = [int(v - s) for v, s in zip(valid_shape, image.shape)]
     
    # Normalize image by percentile (same as training)
    ilow, ihigh = np.percentile(image, (1.0, 99.0))
    image = (image - ilow) / (ihigh - ilow)
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image.astype(np.float32))[None, None, ...].to(compute_device)
    
    # Pad image
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
    
    # Convert probability map to binary mask if a threshold is given
    if threshold is None:
        mask = prob_map
    else:
        mask = prob_map > threshold
    
    return mask

"""
Use a UNet model to produce a binary mask of a small 3D image.
Runs each slice of the stack through the UNet model to get a 3D mask.
"""
def get_mask3d(
        image3d: np.ndarray, 
        model: models.AttentionUNet, 
        compute_device: torch.device, 
        threshold = None
        ) -> np.ndarray:
    
    depth, height, width = image3d.shape
    
    mask = np.zeros([depth, height, width])
    
    # Segment each plane using the get_mask() function
    for k in range(depth):
        mask[k] = get_mask(image3d[k], model = model, compute_device = compute_device, threshold = threshold)
    
    return mask  

"""
Use a UNet model to produce a binary mask (neurites) of a full worm 3D image.
Get a 3D mask for each square chunk of the full image and concatenate them.
"""
def get_big_mask3d(big_img, model, compute_device, threshold = None):
    # Load image, dimensions, and chunk bounds
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
        # Extract chunk from big_img using coords
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


"""Part II: Neurite Classification"""
# Classify branches
# !!! Remove once classifier is retrained and replace with branch_geom() from pvd_classifier_1.py
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

"""
Given predicted classifications and manual labels of neurites, calculate classification accuracy.
Accuracy = proportion of classifications made that are correct.
"""
def test_accuracy(color_map: np.ndarray, pred_results: pd.DataFrame, weighted=False) -> float:
    # Obtain manual labels from a labeled color map
    labels = pc1.get_labels(color_map, pred_results)
    
    n_correct = 0
    len_correct = 0
    
    # For each branch fragment see if predicted dendrite_type matches the manual label
    for i in range(len(labels)):
        if labels[i] == pred_results.loc[i]['dendrite_type']:
            if weighted: # If accuracy is weighted, weight correct decisions by fragment length
                len_correct += pred_results.loc[i]['length']
            else: # Otherwise, accuracy is just based on the number of correct decisions
                n_correct += 1
    
    if weighted:
        len_tot = sum(pred_results['length'])
        accuracy = len_correct/len_tot
    else:
        accuracy = n_correct/len(labels)
    
    return accuracy

"""
Given a 2D binary mask of neurites, classify branch fragments and record them in a DataFrame.
"""
def classify_mask(mask: np.ndarray, maxProj: np.ndarray, model: pc1.PVDNeuriteClassifier) -> pd.DataFrame:
    results = model.predict(mask, maxProj) # Run the actual model prediction
    # Get data on fragment points and length and add to results table
    branch_data = model.branch_data
    results['length'] = branch_data['length']
    results['segment'] = branch_data['segment']
    # Reorder columns
    segments = results.iloc[:, [0, 12, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]]
    
    return segments

"""
Algorithm to correct quaternary fragments misclassified as tertiary.
If a tertiary-predicted fragment:
    1. Lies far to either side (left/right)
    2. AND is more peripheral than a parallel predicted tertiary fragment,
it is probably a quaternary fragment.
The function looks for predicted tertiary fragments meeting the above criteria
and corrects the classification of the corresponding DataFrame entry.
"""
def correct_tertiary(fragments: pd.DataFrame) -> pd.DataFrame:
    corrected = fragments.copy()

    # Look at all predicted tertiary branch fragments
    for ref_idx, ref_row in fragments.iterrows():
        # Skip if not a predicted tertiary
        if ref_row['dendrite_type'] != 3:
            continue
        # Position stats on the current fragment
        ref_rel_x = ref_row['relative_x']
        ref_startpt_y = ref_row['segment'][0][0]
        ref_endpt_y = ref_row['segment'][-1][0]
        ref_id = ref_row['id']

        # Scan over all other predicted tertiary fragments
        for srch_idx, srch_row in fragments.iterrows():
            # Skip if not a predicted tertiary or if it is the reference fragment
            if srch_row['dendrite_type'] != 3 or srch_row['id'] == ref_id:
                continue
            srch_rel_x = srch_row['relative_x']
            # Get search fragment y coordinates
            segment = srch_row['segment']
            y_pos = []
            for i in range(len(segment)):
                y_pos.append(segment[i][0])
            # Find if reference fragment endpoints lie within y bounds of search fragment
            startpt_inRange = ref_startpt_y >= min(y_pos) and ref_startpt_y <= max(y_pos)
            endpt_inRange = ref_endpt_y >= min(y_pos) and ref_endpt_y <= max(y_pos)
            if startpt_inRange or endpt_inRange: # If either reference endpoint is in range...
                # If reference and search fragments are far enough right
                # AND reference is to the right of search, reference fragment is probably quaternary
                if ref_rel_x > 0.7 and srch_rel_x > 0.6:
                    if ref_rel_x >= srch_rel_x:
                        # Correct reference fragment classification in the DataFrame to quaternary
                        corrected.loc[ref_id, 'dendrite_type'] = 4
                        # Stop searching and go to the next reference fragment
                        break
                # If reference and search fragments are far enough left
                # AND reference is to the left of search, reference fragment is probably quaternary
                if ref_rel_x < 0.3 and srch_rel_x < 0.4:
                    if ref_rel_x <= srch_rel_x:
                        # Correct reference fragment classification in the DataFrame to quaternary
                        corrected.loc[ref_id, 'dendrite_type'] = 4
                        # Stop searching and go to the next reference fragment
                        break
        
    return corrected

"""
Algorithm to correct primary fragments misclassified as tertiary.
If a predicted tertiary fragment: 
    1. Is relatively central in x position
    2. Touches a predicted primary fragment at either endpoint
    3. AND Has parallel fragments to the left and right of it
it is probably a primary fragment.
The function looks for predicted tertiary fragments meeting the above criteria
and corrects the classification of the corresponding DataFrame entry.
"""
def correct_primary(fragments: pd.DataFrame) -> pd.DataFrame | None:
    corrected = fragments.copy()
    mistakes = False
    
    # Get start and end points of predicted primary fragments
    prim_only = fragments[fragments['dendrite_type'] == 1]
    prim_start = [segment[0] for segment in prim_only['segment']]
    prim_end = [segment[-1] for segment in prim_only['segment']]
    
    # Look at all predicted tertiary branch fragments
    for ref_idx, ref_row in fragments.iterrows():
        # Skip if not predicted tertiary
        if ref_row['dendrite_type'] != 3:
            continue
        
        # Skip if not central enough in x
        ref_rel_x = ref_row['relative_x']
        if ref_rel_x <= 0.3 or ref_rel_x >= 0.7:
            continue
        
        touching_primary = False
        left_neighbor = False
        right_neighbor = False
        
        # Position stats on current reference fragment
        ref_startpt_x = ref_row['segment'][0][1]
        ref_startpt_y = ref_row['segment'][0][0] 
        ref_endpt_x = ref_row['segment'][-1][1]
        ref_endpt_y = ref_row['segment'][-1][0] 
        ref_id = ref_row['id']
        
        padding = 2
        
        # Find if the current reference fragment touches a predicted primary at either endpoint
        for i in range(ref_startpt_x - padding, ref_startpt_x + padding + 1):
            for j in range(ref_startpt_y - padding, ref_startpt_y + padding + 1):
                if (j, i) in prim_end or (j, i) in prim_start:
                    touching_primary = True
        
        for i in range(ref_endpt_x - padding, ref_endpt_x + padding + 1):
            for j in range(ref_endpt_y - padding, ref_endpt_y + padding + 1):
                if (j, i) in prim_end or (j, i) in prim_start:
                    touching_primary = True
        
        # Skip current reference fragment if it's not touching a predicted primary
        if touching_primary == False:
            continue
        
        # Scan over all predicted primary and tertiary fragments
        for srch_idx, srch_row in fragments.iterrows():
            # Stop searching if left and right neighbors of the reference were already found
            if left_neighbor == True and right_neighbor == True:
                break
            # Skip if not a predicted tertiary or primary, or if it is the reference fragment
            if srch_row['dendrite_type'] != 3 and srch_row['dendrite_type'] != 1: 
                continue
            if srch_row['id'] == ref_id:
                continue
            # Get search fragment y coordinates
            srch_rel_x = srch_row['relative_x']
            segment = srch_row['segment']
            y_pos = []
            for i in range(len(segment)):
                y_pos.append(segment[i][0])
            # Find if reference fragment endpoints lie within y bounds of search fragment
            startpt_inRange = ref_startpt_y >= min(y_pos) and ref_startpt_y <= max(y_pos)
            endpt_inRange = ref_endpt_y >= min(y_pos) and ref_endpt_y <= max(y_pos)
            if startpt_inRange or endpt_inRange: # If either reference endpoint is in range...
                # Is the search fragment to the left or right of the reference?
                if ref_rel_x > srch_rel_x: # !!!
                    left_neighbor = True
                else:
                    right_neighbor = True
        
        # If the reference fragment has right and left neighbors, it's probably a primary
        if right_neighbor == True and left_neighbor == True:
            corrected.loc[ref_id, 'dendrite_type'] = 1 # Correct classification    
            mistakes = True
    
    # If no mistakes were found/corrected, return corrected as None
    if mistakes == False: 
        print("Nothing to correct")
        corrected = None
        
    return corrected

"""
Given a DataFrame of classified branch fragments, reconstruct them into full-length branches
and collect morphological stats on them into a new DataFrame.
"""
def reconstructed_with_stats(fragments: pd.DataFrame, maxProj: np.ndarray) -> pd.DataFrame:
    # 1: Reconstruct branches from each class
    prim_fragments = list(fragments[fragments['dendrite_type'] == 1]['segment'])
    prim_branches = br.connect_segments(prim_fragments, threshold=20.0, max_step_ratio = 10.0)
    sec_fragments = list(fragments[fragments['dendrite_type'] == 2]['segment'])
    sec_branches = br.connect_segments(sec_fragments, threshold=20.0, max_step_ratio = 10.0)
    tert_fragments = list(fragments[fragments['dendrite_type'] == 3]['segment'])
    tert_branches = br.connect_segments(tert_fragments, threshold=20.0, max_step_ratio = 10.0)
    quat_fragments = list(fragments[fragments['dendrite_type'] == 4]['segment'])
    quat_branches = br.connect_segments(quat_fragments, threshold=20.0, max_step_ratio = 10.0)
    
    # 2: Collect stats on each reconstructed branch
    branch_data = []
    start = []
    end = []
    n = 0
    dendrite_type = 1
    
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
            # !!! Should consider normalizing the MIP before similar to the UNet proceedure
            intensities = []
            for i in range(len(branch)): # Get intensity at every point
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
        # Move to next dendrite class    
        dendrite_type += 1    
    
    branch_data = pd.DataFrame(branch_data)
    
    # 3: Add neighbor information
    branches = branch_data['branch']
    padding = 2
    allBranch_neighbors = [] # List where each entry is a list of a branch's neighbors

    # Look for neighbors of each branch
    for ref_branch in branches:
        neighbors = []
        # Iterate through every point in the current branch
        for pt in ref_branch:
            # Look around the current point for endpoints or startpoints of other branches
            for i in range(pt[1] - padding, pt[1] + padding + 1):
                for j in range(pt[0] - padding, pt[0] + padding + 1):
                    # If a branch endpoint is found, load the index of the neighboring branch
                    if (j, i) in end:
                        neighbor = end.index((j, i))
                        # Add this neighbor to the list of it hasn't been added
                        if neighbor not in neighbors:
                            neighbors.append(neighbor)
                    # If a branch startpoint is found, load the index of the neighboring branch
                    if (j, i) in start:
                        neighbor = start.index((j, i))
                        # Add this neighbor to the list of it hasn't been added
                        if neighbor not in neighbors:
                            neighbors.append(neighbor)
        # Add neighbors of the current branch to the overall list
        allBranch_neighbors.append(neighbors)    

    # Add neighbors to the output table
    branch_data['neighbors'] = allBranch_neighbors
    
    return branch_data


"""Part III: Mitochondria Segmentation"""

"""
Prepare a mitochondria image for segmentation by setting regions of the image 
not within neurites to the background intensity.
"""
def process_mito(mito: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = mask > 0
    # Remove specks from the neurite mask, enlarge to match the mito scale, 
    # and pad it (increase neurite thickness)
    mask = skimage.morphology.remove_small_objects(mask, min_size = 150)
    mask = skimage.transform.rescale(mask, (1, 2, 2))
    padded_mask = ndi.binary_dilation(mask, iterations = 4)
    
    # Look for neurite-free regions, where mask == 0
    z, y, x = np.where(padded_mask == 0)
    outImg = np.copy(mito)
    
    # Set all coords of neurite-free regions to background intensity in the mito image
    for i in range(len(y)):
        outImg[z[i], y[i], x[i]] = 100
    
    # Make 2D
    return outImg.max(axis = 0) 

"""
Use a UNet model to produce a binary mask (mitochondria) of a full worm 2D image.
Get a 2D mask for each square chunk of the full image and concatenate them.
"""
def get_big_mask(
        big_img: np.ndarray, 
        model: models.AttentionUNet, 
        compute_device: torch.device, 
        threshold = None
        ) -> np.ndarray:
    # Load image, dimensions, and chunk bounds
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
        # Extract chunk from big_img using coords
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


"""Part IV: Mitochondria Feature Extraction"""

"""
Use skimage.measure.regionprops to extract features of mitochondrial foci in a mask.
Record in a DataFrame and also return the labeled mask (each foci has a unique label/pixel value).
"""
def extract_foci_with_properties(mask: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    # Use skimage.measure to separate each focus and generate a data table on all foci
    labeled_mask = skimage.measure.label(mask, connectivity = 2)
    regions = skimage.measure.regionprops(labeled_mask)
    
    foci_data = []
    # Collect relevant features on foci/regions, expressed in proper units (um, degrees)
    for region in regions:
        centroid = region.centroid
        centroid = tuple(round(a // 2) for a in centroid) # !!! Scale adjusted here, must undo scale adjustment in morph_profiling.py
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
        })
    
    return pd.DataFrame(foci_data), labeled_mask

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
def assign_mitochondria_to_branches(foci_data: pd.DataFrame, dendrite_df: pd.DataFrame, max_distance=10) -> pd.DataFrame:
    # Build a lookup structure for all dendrite skeleton points
    all_skeleton_points = []
    point_to_branch = []  # Maps each skeleton point to branch info
    
    # For all branches
    for idx, row in dendrite_df.iterrows():
        # Add points in the branch to list of skeleton points 
        skeleton_coords = np.array(row['branch'])
        all_skeleton_points.append(skeleton_coords * 2)
        # Store branch info for each point
        for i in range(len(skeleton_coords)):
            point_to_branch.append({
                'branch_id': row['id'],
                'branch_type': row['dendrite_type'],
                'branch_idx': idx  # Index in original dataframe
            })
    
    # Concatenate all skeleton points
    all_skeleton_points = np.vstack(all_skeleton_points)
    # Build KD tree for fast nearest neighbor search
    tree = cKDTree(all_skeleton_points)
    
    # Process each mitochondrion
    results = []
    for idx, focus in foci_data.iterrows():
        centroid = np.array(focus['centroid']).reshape(1, -1)
        # Find nearest skeleton point
        distance, nearest_idx = tree.query(centroid)
        distance = distance[0]
        nearest_idx = nearest_idx[0]
        # Get information on corresponding branch
        branch_info = point_to_branch[nearest_idx]
        # Check if within max_distance threshold
        if max_distance is not None and distance > max_distance:
            branch_id = None
            branch_type = 'unassigned' # Not assigned to a branch if out of range
            distance = None
        else: # If unspecified max_distance, assign to nearest branch
            branch_id = branch_info['branch_id']
            branch_type = branch_info['branch_type']
        # Combine focus data with branch assignment
        result = {
            'mito_label': focus['label'],
            'branch_type': branch_type,
            # Add all data from focus data except label, centroid, and coordinates columns
            **{k: v for k, v in focus.items() 
               if k not in ['label', 'centroid', 'coordinates']}
        }
        results.append(result)
    
    return pd.DataFrame(results)

# !!! Consider merging make_mito_df(), extract_foci_with_properties(), and assign_mitochondria_to_branches() into one function
"""
Given a mask of mitochondria and a DataFrame on dendrite branches,
make a DataFrame of mitochondrial foci that includes assignment to branch classes.
"""
def make_mito_df(mask: np.ndarray, dendrites: pd.DataFrame) -> pd.DataFrame:
    # Get data on foci using skimage.measure.regionprops and add branch assignments
    fociData, myMaskLabeled = extract_foci_with_properties(mask)
    branch_assignment = assign_mitochondria_to_branches(fociData, dendrites)
    # Only keep foci assigned to branches and larger than 2 pixels
    filtered = branch_assignment[(branch_assignment['branch_type'] != 'unassigned') & (branch_assignment['size'] > 2 * (0.1048 ** 2))]
    # Add size z-score (to identify unusually large [perinuclear] foci)
    size_mean = np.mean(filtered['size'])
    size_sd = np.std(filtered['size'])
    size_zscore = (filtered['size'] - size_mean)/size_sd
    filtered['size_zscore'] = size_zscore
    
    return filtered


"""Part V: Network/Graph Operations"""

"""
Reconstruct a dendritic arbor graph from branch data.

Parameters:
-----------
df : pandas DataFrame
    Contains data on reconstructed neurite branches.
proximity_threshold : float
    Distance threshold for merging nearby endpoints (in pixels).
    This handles multi-pixel branch points.

Returns:
--------
G : networkx.Graph
    Graph where nodes are branch points/endpoints and edges are branches
node_data : pandas DataFrame
    Contains data on each node, including position, degree, and node type.
"""
def reconstruct_graph_from_segments(df: pd.DataFrame, proximity_threshold=2) -> tuple[nx.Graph, pd.DataFrame]:
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
            # If a closeby existing node is found, merge with existing node
            if dist <= proximity_threshold:
                endpoint_to_node[endpoint] = node_id
                merged = True
                break
        # Create new node if no closeby nodes are found
        if not merged:
            node_positions[node_counter] = endpoint
            endpoint_to_node[endpoint] = node_counter
            node_counter += 1
    
    # Refine node positions to centroids of merged endpoints
    node_endpoint_lists = defaultdict(list) # Get list of endpoints in each node
    for endpoint, node_id in endpoint_to_node.items():
        node_endpoint_lists[node_id].append(endpoint)

    # Calculate centroid for each node
    for node_id, endpoint_list in node_endpoint_lists.items():
        centroid = (
            int(np.mean([e[0] for e in endpoint_list])),
            int(np.mean([e[1] for e in endpoint_list]))
        )
        # Update node position with centroid
        node_positions[node_id] = centroid
    
    # Step 3: Build the graph
    G = nx.Graph()
    
    # Add nodes with their positions
    for node_id, pos in node_positions.items():
        G.add_node(node_id, pos=pos, x=pos[1], y=pos[0])
    
    # Add edges (one per branch)
    for idx, row in df.iterrows():
        segment = row['segment']
        start_point = segment[0]
        end_point = segment[-1]
        
        start_node = endpoint_to_node[start_point]
        end_node = endpoint_to_node[end_point]
        
        # Add edge with branch id and coordinates
        G.add_edge(
            start_node,
            end_node,
            branch_id=row['id'],
            segment=segment
        )
    
    # Step 4: Classify nodes and prepare node DataFrame
    node_data = []
    # Each node is an entry in the DataFrame
    for node_id in G.nodes():
        # Get node degree and assign node type:
        # Endpoints have degree 1
        # Branch points have degree >= 3
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
def calculate_betweenness_centrality(G: nx.Graph) -> dict:
    # For disconnected graphs, calculate per component
    if not nx.is_connected(G):
        betweenness = {}
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(subgraph) > 2:  # Need at least 3 nodes for meaningful centrality
                component_centrality = nx.betweenness_centrality(subgraph)
                betweenness.update(component_centrality)
            else: # For tiny components, set centrality to 0
                for node in component:
                    betweenness[node] = 0.0
    # Single connected component
    else:
        betweenness = nx.betweenness_centrality(G)
    
    return betweenness

# !!! Consider revising
# 1: Cyclomatic complexity is computed but doesn't show up in outputs. Should it be included?
# 2: Cycle length and number of cycles is important, but what component each one belongs to is not important. Waste of compute.
# 3: Desured output is number of cycles and avg cycle length.
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
def count_loops(G: nx.Graph):
    results = {
        'total_components': nx.number_connected_components(G),
        'cycles_per_component': [],
        'cycle_lengths': [],
        'all_cycles': [],
        'cycles_by_component': {}  # Maps component_idx -> (cycles, component_nodes)
    }
    
    # Analyze each connected component
    for i, component in enumerate(nx.connected_components(G)):
        subgraph = G.subgraph(component)
        # component_size = len(subgraph)
        
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

            # If cycles cannot be found for some reason, set component's # of cycles to 0    
            except nx.NetworkXError as e:
                print(f"  Error finding cycles: {e}")
                results['cycles_per_component'].append(0)
        # If tree structure has no cycles, set component's # of cycles to 0
        else:
            #if component_size > 1:
                #print(f"\nComponent {i} (size {component_size}): Tree structure (no cycles)")
            results['cycles_per_component'].append(0)
    
    # Summary statistics
    total_cycles = sum(results['cycles_per_component'])
    results['total_cycles'] = total_cycles
    
    return results


"""Part VI: Plots"""

"""
Define a sort key for different groups based on genotype and age.
Return tuple: (age_number, genotype_priority).
"""
def sort_key(genotype_age: str) -> tuple:
    # Split group description into genotype and age
    genotype, age = genotype_age.split('-')
    # Convert age to number
    age_num = int(age.replace('day', ''))
    # wt gets priority 0, mutants get priority 1
    genotype_priority = 0 if genotype == 'wt' else 1
    return (age_num, genotype_priority)

"""
Given morphological profile data, plot values of a phenotype/feature for each group with error bars.
Inputs:
    data: pandas DataFrame of morphological profiles
    trait: feature to plot, referred to by its short 2-4 character name
    size: plot size in inches (height, width); default = (10, 5)
    ylimit: bounds for the y-axis (lower, upper); default = None
    dotsize: size of the data points; default = 8
"""
def phenotype_stripchart(
        data: pd.DataFrame, 
        trait: str, 
        size = (10, 5), 
        ylimit = None, 
        dotsize = 8
        ) -> matplotlib.figure.Figure:
    # Initialize dictionary of features and corresponding plot titles and y-axis labels
    # Structure {'shortened_name':('trait_name_in_csv', 'Plot Title', 'Y-axis Label')}
    traits = {
        'ln':('length', 'Worm Length', 'Length ($\mu$m)'),
        'cb':('cellbody', 'Cell Body Position Along the Anterior-Posterior Axis', 'Position (%)'),
        
        'ct1':('prim-ct', 'Number of Primary Dendrites', 'Count/$\mu$m'),
        'ln1':('prim-length', 'Cumulative Length of 1º Dendrites', 'Normalized Length'),
        'wv1':('prim-wavy', 'Waviness of 1º Dendrites', 'Sign Changes/$\mu$m'),
        'tt1':('prim-tort', 'Tortuosity of 1º Dendrites', 'NA'),
        'cv1':('prim-curve', 'Curvature of 1º Dendrites', 'NA'),
        'it1':('prim-intensity', 'Intensity of 1º Dendrites', 'a.u.'),
        'ag1':('prim-angle', 'Mean Orientation of 1º Dendrites', 'Degrees (º)'),
        'as1':('prim-angle-sd', 'SD of Orientations of 1º Dendrites', 'Degrees (º)'),
        
        'ct2':('sec-ct', 'Number of 2º Dendrites', 'Count/$\mu$m'),
        'ln2':('sec-length', 'Cumulative Length of 2º Dendrites', 'Normalized Length'),
        'wv2':('sec-wavy', 'Waviness of 2º Dendrites', 'Sign Changes/$\mu$m'),
        'tt2':('sec-tort', 'Tortuosity of 2º Dendrites', 'NA'),
        'cv2':('sec-curve', 'Curvature of 2º Dendrites', 'NA'),
        'it2':('sec-intensity', 'Intensity of 2º Dendrites', 'a.u.'),
        'ag2':('sec-angle', 'Mean Orientation of 2º Dendrites', 'Degrees (º)'),
        'as2':('sec-angle-sd', 'SD of Orientations of 2º Dendrites', 'Degrees (º)'),
        'md2':('sec-median', 'Median of 2º Distribution', 'Percent (%)'),
        'sk2':('sec-skew', 'Skewness of 2º Distribution', 'Skewness'),
        'pt2':('post-sec', 'Number of Posterior 2º Dendrites', 'Count/$\mu$m'),
        'at2':('ant-sec', 'Number of Anterior 2º Dendrites', 'Count/$\mu$m'),
        
        'ct3':('tert-ct', 'Number of 3º Dendrites', 'Count/$\mu$m'),
        'ln3':('tert-length', 'Cumulative Length of 3º Dendrites', 'Normalized Length'),
        'wv3':('tert-wavy', 'Waviness of 3º Dendrites', 'Sign Changes/$\mu$m'),
        'tt3':('tert-tort', 'Tortuosity of 3º Dendrites', 'NA'),
        'cv3':('tert-curve', 'Curvature of 3º Dendrites', 'NA'),
        'it3':('tert-intensity', 'Intensity of 3º Dendrites', 'a.u.'),
        'ag3':('tert-angle', 'Mean Orientation of 3º Dendrites', 'Degrees (º)'),
        'as3':('quat-angle-sd', 'SD of Orientations of 3º Dendrites', 'Degrees (º)'),
        'pt3':('post-tert', 'Number of Posterior 3º Dendrites', 'Count/$\mu$m'),
        'at3':('ant-tert', 'Number of Anterior 3º Dendrites', 'Count/$\mu$m'),
        
        'ct4':('quat-ct', 'Number of 4º Dendrites', 'Count/$\mu$m'),
        'ln4':('quat-length', 'Cumulative Length of 4º Dendrites', 'Normalized Length'),
        'wv4':('quat-wavy', 'Waviness of 4º Dendrites', 'Sign Changes/$\mu$m'),
        'tt4':('quat-tort', 'Tortuosity of 4º Dendrites', 'NA'),
        'cv4':('quat-curve', 'Curvature of 4º Dendrites', 'NA'),
        'it4':('quat-intensity', 'Intensity of 4º Dendrites', 'a.u.'),
        'ag4':('quat-angle', 'Mean Orientation of 4º Dendrites', 'Degrees (º)'),
        'as4':('quat-angle-sd', 'SD of Orientations of 4º Dendrites', 'Degrees (º)'),
        'md4':('quat-median', 'Median of 4º Distribution', 'Percent (%)'),
        'sk4':('quat-skew', 'Skewness of 4º Distribution', 'Skewness'),
        'pt4':('post-quat', 'Number of Posterior 4º Dendrites', 'Count/$\mu$m'),
        'at4':('ant-quat', 'Number of Anterior 4º Dendrites', 'Count/$\mu$m'),
        
        'cn12':('12-contacts', 'Percent of 1º/2º Contacts', '% Total Contacts'),
        'cn13':('13-contacts', 'Percent of 1º/3º Contacts', '% Total Contacts'),
        'cn14':('14-contacts', 'Percent of 1º/4º Contacts', '% Total Contacts'),
        'cn23':('23-contacts', 'Percent of 2º/3º Contacts', '% Total Contacts'),
        'cn24':('24-contacts', 'Percent of 2º/4º Contacts', '% Total Contacts'),
        'cn34':('34-contacts', 'Percent of 3º/4º Contacts', '% Total Contacts'),
        'im':('iba-mean', 'Mean Interbranch Angle', 'Degrees (º)'),
        'is':('iba-sd', 'SD of Interbranch Angles', 'Degrees (º)'),
        'ik':('iba-skew', 'Skewness of Interbranch Angle Distribution', 'Skewness'),
        'ad1':('angle-dist-bin1', 'Branches with Orientation 0-10º', 'Frequency'),
        'ad2':('angle-dist-bin2', 'Branches with Orientation 10-20º', 'Frequency'),
        'ad3':('angle-dist-bin3', 'Branches with Orientation 20-30º', 'Frequency'),
        'ad4':('angle-dist-bin4', 'Branches with Orientation 30-40º', 'Frequency'),
        'ad5':('angle-dist-bin5', 'Branches with Orientation 40-50º', 'Frequency'),
        'ad6':('angle-dist-bin6', 'Branches with Orientation 50-60º', 'Frequency'),
        'ad7':('angle-dist-bin7', 'Branches with Orientation 60-70º', 'Frequency'),
        'ad8':('angle-dist-bin8', 'Branches with Orientation 70-80º', 'Frequency'),
        'ad9':('angle-dist-bin9', 'Branches with Orientation 80-90º', 'Frequency'),
        'ad10':('angle-dist-bin10', 'Branches with Orientation 90-100º', 'Frequency'),
        'ad11':('angle-dist-bin11', 'Branches with Orientation 100-110º', 'Frequency'),
        'ad12':('angle-dist-bin12', 'Branches with Orientation 110-120º', 'Frequency'),
        'ad13':('angle-dist-bin13', 'Branches with Orientation 120-130º', 'Frequency'),
        'ad14':('angle-dist-bin14', 'Branches with Orientation 130-140º', 'Frequency'),
        'ad15':('angle-dist-bin15', 'Branches with Orientation 140-150º', 'Frequency'),
        'ad16':('angle-dist-bin16', 'Branches with Orientation 150-160º', 'Frequency'),
        'ad17':('angle-dist-bin17', 'Branches with Orientation 160-170º', 'Frequency'),
        'ad18':('angle-dist-bin18', 'Branches with Orientation 170-180º', 'Frequency'),
        'lr12':('12-len-ratio', 'Ratio of 1º to 2º Dendrite Length', 'NA'),
        'lr13':('13-len-ratio', 'Ratio of 1º to 3º Dendrite Length', 'NA'),
        'lr14':('14-len-ratio', 'Ratio of 1º to 4º Dendrite Length', 'NA'),
        'lr23':('23-len-ratio', 'Ratio of 2º to 3º Dendrite Length', 'NA'),
        'lr24':('24-len-ratio', 'Ratio of 2º to 4º Dendrite Length', 'NA'),
        'lr34':('34-len-ratio', 'Ratio of 3º to 4º Dendrite Length', 'NA'),
        'cr23':('23-ct-ratio', 'Ratio of 2º to 3º Dendrites', 'NA'),
        'cr42':('42-ct-ratio', 'Ratio of 4º to 2º Dendrites', 'NA'),
        'cr43':('43-ct-ratio', 'Ratio of 4º to 3º Dendrites', 'NA'),
        
        'oo': ('ori-order', 'Orientation Order', 'NA'),
        'tn': ('num-term-nodes', 'Number of Terminal Nodes', 'Count/$\mu$m'),
        'pt': ('pct-term-nodes', 'Percent of Terminal Nodes', 'Percent (%)'),
        'id': ('int-density', 'Intersection Density', '$\mu$m^-2'),
        'ed': ('edge-density', 'Edge Density', '$\mu$m^-1'),
        'md': ('mean-degree', 'Mean Degree of Intersections', 'NA'),
        'nd4': ('num-degree-4+', 'Number of Intersections with Degree > 3', 'Count/$\mu$m'),
        'pd4': ('pct-degree-4+', 'Percent of Intersections with Degree > 3', 'Percent (%)'),
        'bt': ('mean-betweenness', 'Mean Betweenness Centrality', 'NA'),
        'lp': ('loop-ct', 'Number of Graph Loops', 'Count'),
        
        'mct': ('mito-tot-ct', 'Number of Mitochondrial Foci', 'Count/$\mu$m'),
        'mct1': ('mito-prim-ct', 'Number of 1º Mitochondrial Foci', 'Count/$\mu$m'),
        'mct2': ('mito-sec-ct', 'Number of 2º Mitochondrial Foci', 'Count/$\mu$m'),
        'mct3': ('mito-tert-ct', 'Number of 3º Mitochondrial Foci', 'Count/$\mu$m'),
        'mct4': ('mito-quat-ct', 'Number of 4º Mitochondrial Foci', 'Count/$\mu$m'),
        'mac': ('mito-ant-ct', 'Number of Anterior Mitochondrial Foci', 'Count/$\mu$m'),
        'mpc': ('mito-post-ct', 'Number of Posterior Mitochondrial Foci', 'Count/$\mu$m'),
        'mbc': ('mito-branch-pt-ct', 'Number of Mitochondria in Branch Points', 'Count/$\mu$m'),
        'mbp': ('mito-branch-pt-pct', 'Percent of Mitochondria in Branch Points', 'Percent (%)'),
        'nn': ('mean-nnd', 'Mean Mitochondrial Nearest Neighbor Distance', '$\mu$m'),
        'rn': ('rn', 'Mitochondrial Nearest Neighbor R Value', 'NA'),
        'mec': ('mito-ecc-mean', 'Mean Mitochondrial Eccentricity', 'NA'),
        'mes': ('mito-ecc-sd', 'SD of Mitochondrial Eccentricity', 'NA'),
        'mek':('mito-ecc-skew', 'Skewness of Mitochondrial Eccentricity', 'NA'),
        'mex':('mito-ext-mean', 'Mean Mitochondrial Extent', 'NA'),
        'mxs': ('mito-ext-skew', 'Skewness of Mitochondrial Extent', 'NA'),
        'mxk': ('mito-size-skew', 'Skewness of Mitochondrial Size', 'NA'),
        'mpa':('mito-perimarea-mean', 'Mean Mitochondrial Perimeter/Area', '$\mu$m^-1'),
        'mps':('mito-perimarea-sd', 'SD of Mitochondrial Perimeter/Area', '$\mu$m^-1'),
        'mpk':('mito-perimarea-skew', 'Skewness of Mitochondrial Perimeter/Area', 'NA'),
        'mos':('mito-ori-sd', 'SD of Mitochondrial Orientation', 'Degrees (º)'),
        'mdm': ('mito-dist-med', 'Median of Mitochondrial Distribution', 'Percent (%)'),
        'mds': ('mito-dist-sd', 'SD of Mitochondrial Distribution', 'Percent (%)'),
        'mdk': ('mito-dist-skew', 'Skewness of Mitochondrial Distribution', 'NA'),
        'msz': ('mito-size-mean', 'Mean Mitochondrial Size', 'Area ($\mu$m^2)'),
        'mss': ('mito-size-sd', 'SD of Mitochondrial Size', 'Area ($\mu$m^2)'),
        'msk': ('mito-size-skew', 'Skewness of Mitochondrial Size', 'NA'),
        'mja': ('mito-majaxis-mean', 'Mean Mitochondrial Major Axis Length', '$\mu$m'),
        'mjs': ('mito-majaxis-sd', 'SD of Mitochondrial Major Axis Length', '$\mu$m'),
        'mjk': ('mito-majaxis-skew', 'Skewness of Mitochondrial Major Axis Length', 'NA'),
        'mna': ('mito-minaxis-mean', 'Mean Mitochondrial Minor Axis Length', '$\mu$m'),
        'mns': ('mito-minaxis-sd', 'SD of Mitochondrial Minor Axis Length', '$\mu$m'),
        'mnk': ('mito-minaxis-skew', 'Skewness of Mitochondrial Minor Axis Length', 'NA'),
        'meq': ('mito-diam-mean', 'Mean Mitochondrial Equivalent Diameter', '$\mu$m'),
        'mqs': ('mito-diam-sd', 'SD of Mitochondrial Equivalent Diameter', '$\mu$m'),
        'mqk': ('mito-diam-skew', 'Skewness of Mitochondrial Equivalent Diameter', 'NA'),
        'mpm': ('mito-perim-mean', 'Mean Mitochondrial Perimeter', '$\mu$m'),
        'mms': ('mito-perim-sd', 'SD of Mitochondrial Perimeter', '$\mu$m'),
        'mmk': ('mito-perim-skew', 'Skewness of Mitochondrial Perimeter', 'NA'),
        'mfd': ('mito-feret-mean', "Mean Mitochondrial Feret's Diameter", '$\mu$m'),
        'mfs': ('mito-feret-sd', "SD of Mitochondrial Feret's Diameter", '$\mu$m'),
        'mfk': ('mito-feret-skew', "Skewness of Mitochondrial Feret's Diameter", 'NA')
        }

    # From the DataFrame, take the columns for genotype-age and the feature specified
    filtered = data.filter(items=['genotype-age', traits[trait][0]])
    genotypes_ages = list(filtered.iloc[:, 0].unique()) # Non-repeating list of genotypes/ages
    # Sort genotypes_ages to put wt first and go in ascending order of age for each geno
    x_labs = sorted(genotypes_ages, key=sort_key)
    
    y_avg = []
    y_err = []
    feature_data = []
    # !!! deleted: i = 0
    
    # For each genotype/age, extract data on the specified feature
    for genotype_age in x_labs:
        data_by_genotype_age = filtered[filtered['genotype-age'] == genotype_age].iloc[:, 1]
        feature_data.append(data_by_genotype_age)
        y_avg.append(np.mean(data_by_genotype_age)) # Mean value
        y_err.append(1.96 * np.std(data_by_genotype_age)/np.sqrt(len(data_by_genotype_age))) # 95% CI
        # !!! deleted: i += 1

    # If there are only 2 genotypes/ages, do a t-test
    if len(genotypes_ages) == 2:
        pval = scipy.stats.ttest_ind(feature_data[0], feature_data[1], equal_var = False).pvalue

    # Generate plot    
    fig, ax = plt.subplots(figsize = size)
    if ylimit is not None: # Apply y-axis bounds if supplied
        ax.set_ylim(ylimit[0], ylimit[1])
    # Plot values, mean, and 95% CI
    ax.errorbar(x_labs, y_avg, y_err, fmt = 'r_', markersize = 10, capsize = 5, linewidth = 2, barsabove = True)
    ax = sns.stripplot(x = 'genotype-age', y = traits[trait][0], data = filtered, 
                      jitter = 0.1, size = dotsize, color = 'k')
    # Set aesthetics
    ax.set_xticklabels(x_labs, size= 12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(None)
    plt.ylabel(traits[trait][2], size= 12) # y-axis label from features dictionary
    plt.yticks(fontsize = 16)
    if len(genotypes_ages) == 2: # Display p-value if a t-test was performed
        plt.text(0.7, 0.95, f'p = {pval:.2e}', transform = plt.gca().transAxes) 
    ax.set_title(traits[trait][1], weight = 'bold', size = 14, wrap = True) # plot title from features dictionary
    
    return fig

"""
Given a directory to an experiment, plot a histogram of branch positions over the anterior-posterior axis.
Genotype/age and dendrite type must be specified.
Data on position will be combined from all neuron images of a given genotype/age.
"""
def plot_branch_dist(infolder: str | pathlib.Path, strain: str, dendrite_type: int) -> matplotlib.figure.Figure:

    exp_id = str(infolder).split('-')[-1]
    infolder = Path(infolder)
    y_pos = []
    pattern = f'*{strain}*.csv'
    
    # Get position data of branches (anterior-posterior/y-axis)
    for file in infolder.glob(pattern = pattern):
        branches = pd.read_csv(file)
        # Get image height
        image_name = str(infolder.parent) + f'/maxProj-{exp_id}/' + file.stem.replace('branches', 'maxProj.tif')
        length = tifffile.imread(image_name).shape[0]
        # Filter for the genotype/age specified
        filtered_branches = branches[branches['dendrite_type'] == dendrite_type]
        filtered_branches = [eval(branch) for branch in filtered_branches['branch']]
        # For each branch, add its normalized y pos to the list of positions
        for n in range(len(filtered_branches)):
            # !!! Consider using y_pos directly from the DataFrame
            branch = filtered_branches[n]
            y_coords = [pt[0] for pt in branch]
            y_pos.append(np.mean(y_coords) * 100 / length)
    
    # Histogram bin cutoffs at 0, 5, 10, ..., 100
    cutoffs = [i for i in range(0, 105, 5)]
    
    # Plot positional distribution with histogram and kernel density estimate smoothed distribution
    fig, ax = plt.subplots()
    ax = sns.kdeplot(y_pos)
    ax.hist(y_pos, bins = cutoffs, density = True)
    ax.set_xlabel('Percent Along Anterior-Posterior Axis', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_title(f'Distribution of {dendrite_type}º Dendrites', fontsize=14)
    ax.text(0.05, 0.95, f'{len(y_pos)}\ndendrites', transform=ax.transAxes, fontsize=12,
            verticalalignment='top') # State the number of branches
    
    return fig