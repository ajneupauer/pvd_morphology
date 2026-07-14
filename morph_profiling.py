#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 17:24:52 2025

@author: alexneupauer
"""

# Import modules
import os
import argparse
import json
import numpy as np
from pathlib import Path
import torch
import tifffile
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as scs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
from scipy.spatial import KDTree

import sys
sys.path.append('./ml_models') # Add models dir to sys
sys.path.append('./modules') # Add module dir to sys for custom module import
import pvd_processing as pvd
# !!! deleted: import pvd_plots
import models
import pvd_classifier_1 as pc1

# Get parameters from the config file
payload = json.loads(Path('./config.json').read_text(encoding="utf-8"))
HAS_MITO = payload.get("has_mito")
HAS_MITO = True if HAS_MITO == 1 else 0
NEURITE_SEG_PATH = payload.get("neurite_seg_path")
MITO_SEG_PATH = payload.get("mito_seg_path")
CLASSIFIER_PATH = payload.get("classifier_path")

# Load ML models
neurite_seg_path = Path(NEURITE_SEG_PATH)
compute_device = torch.device("mps")
neurite_seg_model = models.AttentionUNet(1, 1, features=[16, 32, 64, 128], use_logits=True)
unet_dict = torch.load(neurite_seg_path, weights_only=False)
neurite_seg_model.load_state_dict(unet_dict["model_state_dict"])
neurite_seg_model = neurite_seg_model.to(compute_device)

if HAS_MITO:
    mito_seg_path = Path(MITO_SEG_PATH)
    mito_seg_model = models.AttentionUNet(1, 1, features=[16, 32, 64, 128], use_logits=True)
    unet_dict = torch.load(mito_seg_path, weights_only=False)
    mito_seg_model.load_state_dict(unet_dict["model_state_dict"])
    mito_seg_model = mito_seg_model.to(compute_device)
else:
    mito_seg_model = None

classifier = pc1.PVDNeuriteClassifier()
classifier.load_model(CLASSIFIER_PATH)

# %%

"""
Collect user arguments passed to the command line.
dataset_path: directory to dataset of raw images
--dry-run: add if performing a dry run (won't make/modify anything')
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and analyze morphological profiles of PVD neuron images."
    )
    parser.add_argument("dataset_path", type=str, help="Directory to dataset of raw images.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

"""Determine if a given file exists."""
def already_exists(filepath: str) -> bool:
    filepath = Path(filepath)
    if filepath.exists():
        return True
    else:
        return False

"""
Process an image into data on its branches and mitochondria.
Inputs:
    img_path = path to the image as a string
    neurite_seg_model = loaded ML model for neurite segmentation
    mito_seg_model = loaded ML model for mitochondria segmentation, set to None if no mito channel available
    compute_device = specifies device to use for the ML models
    classifier = loaded random forest model for neurite classification
Ouputs:
    branch_stats = pd.Dataframe of each segmented branch and associated meaasurements or None if the data already exists
    mito_stats = pd.Dataframe of each mitochondrial focus and associated meaasurements or None if the data already exists or no mito channel available
    node_data = pd.Dataframe of each node in a network representing the neurites and its associated meaasurements or None if the data already exists
"""
def img_to_branches(
        img_path: str, 
        neurite_seg_model: models.AttentionUNet, 
        mito_seg_model: models.AttentionUNet, 
        compute_device: torch.device, 
        classifier: pc1.PVDNeuriteClassifier
        ):
    # Manage file paths
    mip_path = img_path.replace('Straightened', 'mip')
    mask_path = img_path.replace('Straightened', 'seg')
    mask3d_path = img_path.replace('Straightened', 'seg3d')
    if mito_seg_model is not None:
        mito_straightened_path = img_path.replace('Straightened', 'mito_Straightened')
    straightened = tifffile.imread(img_path)
    
    # Step 1: Make max intensity projection
    if already_exists(mip_path):
        mip = tifffile.imread(mip_path)
        print('Step 1 already completed:\nGenerate neurite MIP.\n')
    else:
        mip = straightened.max(axis = 0)
        tifffile.imwrite(mip_path, mip, compression = 'lzw')
        print('Step 1 completed successfully:\nGenerate neurite MIP.\n')
    
    # Step 2: Make neurite mask
    if already_exists(mask_path):
        mask2d = tifffile.imread(mask_path)
        print('Step 2 already completed:\nSegment neurites.\n')
    else:    
        mask3d = pvd.get_big_mask3d(straightened, 
                                model = neurite_seg_model, 
                                compute_device = compute_device, 
                                threshold = 0.3)
        mask3d = np.uint8(mask3d)
        tifffile.imwrite(mask3d_path, mask3d, compression = 'lzw')
        mask2d = mask3d.max(axis = 0)
        tifffile.imwrite(mask_path, mask2d, compression = 'lzw')
        print('Step 2 completed successfully:\nSegment neurites.\n')
    
    # Step 3: Classify branches
    """
    Branches and nodes already made --> skip generation of both dataFrames (branches, nodes = None)
    Neither branches nor nodes made --> generate both dataFrames (branches, nodes = pd.dataFrame)
    ONLY branches made --> generate branch fragments to get nodes but don't make branches df (branches = None; nodes = pd.dataFrame)
    ONLY nodes made --> generate branch fragments for branches dataFrame (branches = pd.dataFrame; nodes = None)
    """
    branches_path = img_path.replace('Straightened.tif', 'branches.csv')
    nodes_path = img_path.replace('Straightened.tif', 'nodes.csv')
    
    # Need to extract and classify branch fragments unless BOTH branches and nodes dataFrames are saved
    if not(already_exists(branches_path) and already_exists(nodes_path)):
        fragments = pvd.classify_mask(mask2d, mip, classifier)
        print('Step 3a completed successfully:\nClassify branch fragments.\n')
    
    # Make nodes dataFrame if it hasn't yet been saved
    if already_exists(nodes_path):
        print('Nodes data already created!')
        node_data = None
    else:
        G, node_data = pvd.reconstruct_graph_from_segments(fragments) # Get nodes of network representation of neurites
        # Compute betweenness centrality and add to nodes data
        betweenness = pvd.calculate_betweenness_centrality(G)
        node_data['betweenness'] = betweenness
        # Compute number of loops in the network and add to nodes data
        loops = [pvd.count_loops(G)['total_cycles']]
        loops = loops + (len(node_data) - 1) * ['NA']
        node_data['loops'] = loops
    
    # Make branches dataFrame if it hasn't yet been saved
    if already_exists(branches_path):
        print('Steps 3b-c already completed:\nCorrect and reconstruct fragments into branches.\n')
        branch_stats = None
    else:
        # Apply primary classification correction until no more corrections are required
        while True:
            corrected_copy = fragments.copy()
            corrected_copy = pvd.correct_primary(fragments) # Returns None if no more corrections are req'd
            # If corrections were performed, update fragments dataFrame;
            # Otherwise (corrected_copy is None), stop the loop and keep previous iteration of fragments
            if corrected_copy is not None:
                fragments = corrected_copy
            else:
                break
        # Apply tertiary classification correction
        fragments = pvd.correct_tertiary(fragments)
        print('Step 3b completed successfully:\nCorrect fragment classifications.\n')
        # Reconstruct corrected fragments into full branches with corresponding stats
        branch_stats = pvd.reconstructed_with_stats(fragments, mip)
        print('Step 3c completed successfully:\nReconstruct fragments into branches.\n')
    
    # Only do steps 4-6 if a mitochondria segmentation model is provided; otherwise, set mito_stats to None
    if mito_seg_model is not None:    
        # Step 4: Process mito image w/ neurite mask
        if already_exists(img_path.replace('Straightened', 'mito_rmbg')):
            mito_rmbg = tifffile.imread(img_path.replace('Straightened', 'mito_rmbg'))
            print('Step 4 already completed:\nProcess mito image for segmentation.\n')
        else:    
            mask3d = tifffile.imread(mask3d_path)
            mito_straightened = tifffile.imread(mito_straightened_path)
            # Set mito image to background intensity in empty regions of the 3D neurite mask
            mito_rmbg = pvd.process_mito(mito_straightened, mask3d)
            tifffile.imwrite(img_path.replace('Straightened', 'mito_rmbg'), 
                             mito_rmbg, compression = 'lzw')
            print('Step 4 completed successfully:\nProcess mito image for segmentation.\n')
            
        # Step 5: Segment mitochondria
        if already_exists(img_path.replace('Straightened', 'mito_seg')):
            mito_seg = tifffile.imread(img_path.replace('Straightened', 'mito_seg'))
            print('Step 5 already completed:\nSegment mitochondria.\n')
        else:
            mito_seg = pvd.get_big_mask(mito_rmbg, 
                                        model = mito_seg_model, 
                                        compute_device = compute_device, 
                                        threshold = 0.2)
            mito_seg = np.uint8(mito_seg)
            tifffile.imwrite(img_path.replace('Straightened', 'mito_seg'), 
                             mito_seg, compression = 'lzw')
            print('Step 5 completed successfully:\nSegment mitochondria.\n')
        
        # Step 6: Generate mito data
        if already_exists(img_path.replace('Straightened.tif', 'mito.csv')):
            print('Step 6 already completed:\nGenerate mito data.\n')
            mito_stats = None
        else:
            mito_stats = pvd.make_mito_df(mito_seg, branch_stats)
            print('Step 6 completed successfully:\nGenerate mito data.\n')
    else:
        mito_stats = None

    # If any output dataFrame is already saved, tell the user    
    if branch_stats is None:
        print('Already generated branch data!')
    if node_data is None:
        print('Already generated node data!')
    if mito_stats is None:
        print('Already generated mito data or no mito channel available!')
    
    return branch_stats, mito_stats, node_data

"""Find the cell body position along the anterior-posterior axis."""
def find_cellbody(fpath: str) -> int:
    myImg = tifffile.imread(fpath)
    
    # Intensity profile along middle of the worm (A-P axis), averaged over the width (D-V axis)
    profile = []
    for n in range(myImg.shape[0]):
        lbound = round(myImg.shape[1] * 0.375) # Cut off the leftmost and rightmost 3/8 of the image
        rbound = round(myImg.shape[1] * 0.655) 
        row = myImg[n, lbound:rbound]
        avg_intensity = np.mean(row)
        profile.append(avg_intensity)
    
    # Smooth the intensity profile with a 100 px average sliding window
    smooth_profile = []
    for i in range(len(profile) - 100):
        sliding_avg = np.mean(profile[i:i+100])
        smooth_profile.append(sliding_avg)
    
    # Get a list of putative peaks by finding profile values at positions where both neighbors are lower
    # by at least 0.0001
    peaks = []
    for k in range(1, len(smooth_profile) - 1):
        if smooth_profile[k] > smooth_profile[k + 1] + 0.0001 and smooth_profile[k] > smooth_profile[k - 1] + 0.0001:
            peaks.append(smooth_profile[k])
    # Take the cell body position to be where the highest peak occurs
    breakpt = smooth_profile.index(max(peaks))
    return breakpt + 50 # Add 50 to correct for 100 px sliding window

"""Given a list of neighbors for each branch, make a list of unique pairs that are neighbors."""
def uniq_neighbor_pairs(neighbors: list[list]) -> list[tuple]:
    neighbor_pairs = []
    
    for i in range(len(neighbors)): # Loop over each branch
        for j in range(len(neighbors[i])): # Loop over neighbors of branch i
            # Branch i and its jth neighbor are a pair
            # Do not count if neighbor index matches branch index (neighbor of itself)
            # Do not count if branch index exceeds neighbor index (prevents addition of reciprocal pair e.g. (x, y) and (y, x))
            if i < neighbors[i][j]: 
                neighbor_pairs.append((i, neighbors[i][j]))
    
    return neighbor_pairs


#def sort_key(strain):
#    genotype, age = strain.split('-')
#    age_num = int(age.replace('day', ''))
    # Return tuple: (age_number, genotype_priority)
    # wt gets priority 0, anc1null gets priority 1
#    genotype_priority = 0 if genotype == 'wt' else 1
#    return (age_num, genotype_priority)

#def generate_random_hex_color():
    """Generates a random hexadecimal color code."""
#    r = random.randint(0, 255)
#    g = random.randint(0, 255)
#    b = random.randint(0, 255)
#    return f'#{r:02x}{g:02x}{b:02x}'

"""
Generate a morphological for each image in the dataset.
Inputs:
    folder = path to the image dataset
    mito = whether or not there is mitochondria data, default = True
    adj_branch_distributions = whether or not to set 0 in branch distributions to be at the cell body, default = True
Output: DataFrame where each row is a morphological profile of an image with:
    41 features describing mitochondria
    90 features describing neurites
    131 total features
"""
def write_feature_table(folder: str, mito = True, adj_branch_distributions = True) -> pd.DataFrame:
    """Collect all branches files (one per image subdirectory)."""
    folder = Path(folder)
    files = []
    for file in folder.glob('*/*branches.csv'):
        files.append(str(file))
    files.sort()
    
    """Generate a morphological profile of each file."""
    stats = []
    for file in files:
        """PART ONE: READ DATA"""
        data = pd.read_csv(file).iloc[:, 1:] # .iloc[:, 1:] removes extra first column created during DataFrame export
        data['branch'] = [eval(branch) for branch in data['branch']] # eval to convert list entries saved as str back to lists
        data['neighbors'] = [eval(neighbor) for neighbor in data['neighbors']]
        nodes = pd.read_csv(file.replace('branches.csv', 'nodes.csv')).iloc[:, 1:]
        if mito:
            mito_data = pd.read_csv(file.replace('branches.csv', 'mito.csv')).iloc[:, 1:]
        img_name = file.replace('branches.csv', 'mip.tif') # Use MIP to find cell body position
        img = tifffile.imread(img_name)
        
        # Get basic stats
        length, width = img.shape # Image dimensions in px
        cellbody = find_cellbody(img_name) # Y px position where cell body is located
        img_name = Path(file).stem.replace('_branches', '') # Image base name
        # Images are named as {date}_{description}_{genotype}_{age}_{image #}
        # Save genotype_age variable as "{genotype}-{age}""
        genotype_age = Path(file).stem.split('_')[2:4]
        genotype_age = genotype_age[0] + '-' + genotype_age[1]
        # Get neuron length and cell body pos in proper units
        cellbody_pct = 100 * cellbody / length # Cell body pos = % along A-P axis
        length_um = round(length * 0.2096) # Length in units of microns
        ant_length = round(cellbody * 0.2096) # Length anterior of cell body
        post_length = length_um - ant_length # Length posterior of cell body
        

        """PART TWO: COMPUTE STATS PER EACH BRANCH CLASS"""
        # I: Primary
        prim = data[data['dendrite_type'] == 1] # DataFrame of only primaries
        prim_ct = len(prim)
        prim_length = round(sum(prim['length']))
        prim_wavy = np.mean(prim['waviness'])
        # Tortuosity measures sometimes evaluate as infinite, only extract finite values; NA otherwise
        prim_tort_mask = np.isfinite(prim['tortuosity'])
        prim_tort = np.mean(prim['tortuosity'][prim_tort_mask])
        prim_curve = np.mean(prim['curvature'])
        prim_intensity = np.mean(prim['intensity'])
        prim_angle = np.mean(prim['orientation'])
        prim_angle_sd = np.std(prim['orientation'])
        
        # II: Secondary
        sec = data[data['dendrite_type'] == 2] # DataFrame of only secondaries
        sec_ct = len(sec)
        sec_length = round(sum(sec['length']))
        sec_wavy = np.mean(sec['waviness'])
        # Tortuosity measures sometimes evaluate as infinite, only extract finite values; NA otherwise
        sec_tort_mask = np.isfinite(sec['tortuosity'])
        sec_tort = np.mean(sec['tortuosity'][sec_tort_mask])
        sec_curve = np.mean(sec['curvature'])
        sec_intensity = np.mean(sec['intensity'])
        
        sec_angles = list(sec['orientation'])
        # Convert angles to range -90 to 90 (center at 0) by finding quadrant IV angles w/ eqivalent orientations as quad II
        for i in range(sec_ct): 
            if sec_angles[i] > 90:
                sec_angles[i] = sec_angles[i] - 180
        sec_angle = np.mean(sec_angles)
        sec_angle_sd = np.std(sec_angles)
        
        # Get 2º A-P axis distribution info
        # If we're adjusting distributions, extract branch y positions relative to cell body
        if adj_branch_distributions: # + sign means posterior of cell body
            y_pos = [y * 100 / length - cellbody_pct for y in list(sec['mean_y'])]
            # Posterior branches have positive pct along A-P axis
            post_sec = sum(1 for y in y_pos if y > 0)
            ant_sec = len(y_pos) - post_sec
        else:
            y_pos = [y * 100 / length for y in list(sec['mean_y'])]
            post_sec = None
            ant_sec = None
        sec_median = np.median(y_pos)
        sec_skew = scs.skew(y_pos)
        
        # III: Tertiary
        tert = data[data['dendrite_type'] == 3] # DataFrame of only tertiaries
        tert_ct = len(tert)
        tert_length = round(sum(tert['length']))
        tert_wavy = np.mean(tert['waviness'])
        # Tortuosity measures sometimes evaluate as infinite, only extract finite values; NA otherwise
        tert_tort_mask = np.isfinite(tert['tortuosity'])
        tert_tort = np.mean(tert['tortuosity'][tert_tort_mask])
        tert_curve = np.mean(tert['curvature'])
        tert_intensity = np.mean(tert['intensity'])
        tert_angle = np.mean(tert['orientation'])
        tert_angle_sd = np.std(tert['orientation'])
        
        # Get 3º A-P axis distribution info
        # If we're adjusting distributions, extract branch y positions relative to cell body
        if adj_branch_distributions: # + sign means posterior of cell body
            y_pos = [y * 100 / length - cellbody_pct for y in list(tert['mean_y'])]
            # Posterior branches have positive pct along A-P axis
            post_tert = sum(1 for y in y_pos if y > 0)
            ant_tert = len(y_pos) - post_tert
        else:
            y_pos = [y * 100 / length for y in list(tert['mean_y'])]
            post_tert = None
            ant_tert = None
        
        # IV: Quaternary
        quat = data[data['dendrite_type'] == 4] # DataFrame of only quaternaries
        quat_ct = len(quat)
        quat_length = round(sum(quat['length']))
        quat_wavy = np.mean(quat['waviness'])
        # Tortuosity measures sometimes evaluate as infinite, only extract finite values; NA otherwise
        quat_tort_mask = np.isfinite(quat['tortuosity'])
        quat_tort = np.mean(quat['tortuosity'][quat_tort_mask])
        quat_curve = np.mean(quat['curvature'])
        quat_intensity = np.mean(quat['intensity'])
        
        quat_angles = list(quat['orientation'])
        # Convert angles to range -90 to 90 (center at 0) by finding quadrant IV angles w/ eqivalent orientations as quad II
        for i in range(quat_ct):
            if quat_angles[i] > 90:
                quat_angles[i] = quat_angles[i] - 180
        quat_angle = np.mean(quat_angles)
        quat_angle_sd = np.std(quat_angles)
        
        # Get 4º A-P axis distribution info
        # If we're adjusting distributions, extract branch y positions relative to cell body
        if adj_branch_distributions: # + sign means posterior of cell body
            y_pos = [y * 100 / length - cellbody_pct for y in list(quat['mean_y'])]
            # Posterior branches have positive pct along A-P axis
            post_quat = sum(1 for y in y_pos if y > 0)
            ant_quat = len(y_pos) - post_quat
        else:
            y_pos = [y * 100 / length for y in list(quat['mean_y'])]
            post_quat = None
            ant_quat = None
        quat_median = np.median(y_pos)
        quat_skew = scs.skew(y_pos)
        
        
        """PART THREE: COMPUTE GLOBAL STATS ON ENTIRE NEURITE NETWORK"""
        # I: Interbranch Angles And Contacts
        neighbor_pairs = uniq_neighbor_pairs(list(data['neighbors'])) # Non-repetitive list of neighboring branches
        
        interbranch_angles = []
        contacts_12 = 0
        contacts_13 = 0
        contacts_14 = 0
        contacts_23 = 0
        contacts_24 = 0
        contacts_34 = 0

        for pair in neighbor_pairs:
            # Compute angle between a given pair of branches
            delta_angle = data.loc[pair[0]]['orientation'] - data.loc[pair[1]]['orientation']
            angle_diff = min(abs(delta_angle), 360 - abs(delta_angle)) # Ensure angle is pos and the min of two possible values
            interbranch_angles.append(angle_diff)
            # Find classes of branches in the pair and increase the count of the appropriate contact count
            pair_class = (data.loc[pair[0]]['dendrite_type'], data.loc[pair[1]]['dendrite_type'])
            if pair_class == (1, 2) or pair_class == (2, 1):
                contacts_12 += 1
            elif pair_class == (1, 3) or pair_class == (3, 1):
                contacts_13 += 1
            elif pair_class == (1, 4) or pair_class == (4, 1):
                contacts_14 += 1
            elif pair_class == (2, 3) or pair_class == (3, 2):
                contacts_23 += 1
            elif pair_class == (2, 4) or pair_class == (4, 2):
                contacts_24 += 1
            elif pair_class == (3, 4) or pair_class == (4, 3):
                contacts_34 += 1    
        
        iba_mean = np.mean(interbranch_angles)
        iba_sd = np.std(interbranch_angles)
        iba_skew = scs.skew(interbranch_angles)
        
        # II: Length And Count Ratios
        length_12 = prim_length / sec_length
        length_13 = prim_length / tert_length
        length_14 = prim_length / quat_length
        length_23 = sec_length / tert_length
        length_24 = sec_length / quat_length
        length_34 = prim_length / sec_length
        
        ct_23 = sec_ct / tert_ct
        ct_42 = quat_ct / sec_ct 
        ct_43 = quat_ct / tert_ct
        
        # III: Overall Orientation Distribution
        # Tuple of counts in each bin 0-10, 10-20, ..., 350-360
        bin_cts = np.histogram(data['orientation'], np.linspace(0, 180, 19))[0]
        total_ct = prim_ct + sec_ct + tert_ct + quat_ct
        
        # IV: Network Topology Metrics
        # Remove nodes w/ degree == 2 and get a DataFrame without terminal nodes
        nodes = nodes[nodes['type'] != 'continuation']
        nodes_noterm = nodes[nodes['type'] == 'branch_point']
        
        # Determine which mitochondrial foci fall on neurite branch points
        if mito:
            mito_in_branch_pts = []
            # Loop over all mitochondrial foci
            for ref_idx, ref_row in mito_data.iterrows():
                mito_x = ref_row['centroid_x']
                mito_y = ref_row['centroid_y']
                match_found = False
                # Loop over all intersection nodes to see if one intersects the given focus
                for query_idx, query_row in nodes_noterm.iterrows():
                    if match_found: # Stop searching if we find an overlapping intersection
                        break
                    # Allow a 5 px margin in each direction to account for imperfect overlap
                    x_inrange = (query_row['x_pos'] >= mito_x - 5) and (query_row['x_pos'] <= mito_x + 5)
                    y_inrange = (query_row['y_pos'] >= mito_y - 5) and (query_row['y_pos'] <= mito_y + 5)
                    # If a mitochondrial focus overlaps an intersecion, update match_found to True;
                    # otherwise, match_found stays False
                    if x_inrange and y_inrange:
                        match_found = True
                # Log whether or not a match was found for the given focus
                mito_in_branch_pts.append(match_found)
        
        # Need image area (microns^2) for edge and intersection density
        area = (0.2096 ** 2) * length * width
        # Number of nodes with degree >= 4
        num_high_degree = sum(nodes_noterm['degree'] >= 4)
        # Cumulative length of all branches (for edge density)
        tot_length = sum(data['length'])
        
        # Orientation order (weighted)
        # Include reciprocal orientations (e.g., 90 and 270) as per Boeing, 2019
        ori_1 = list(data['orientation'])
        ori_2 = [angle + 180 for angle in ori_1]
        ori = ori_1 + ori_2
        ar = length / width # Image aspect ratio
        
        # Need to adjust branch length to address long/thin shape of worms
        # (heavy bias to A-P axis aligned branches over D-V axis)
        length_adj = []
        for idx, row in data.iterrows():
            # Divide 1º/3º lengths by the aspect ratio, ar, since they are in the 'long' direction
            if row['dendrite_type'] == 1 or row['dendrite_type'] == 3:
                length_adj.append(row['length'] / ar)
            # 2/4 lengths are not changed
            else:
                length_adj.append(row['length'])
        
        # Weight each branch as its proportion of total adjusted length
        cumul_len = sum(length_adj)
        weights = [i / (2 * cumul_len) for i in length_adj]
        weights = 2 * weights # Copy weights for reciprocal angles (describing the same branches)
        # Pair each branch orienation with its weight
        weighted_ori = pd.DataFrame(columns = ['orientation', 'weight'])
        weighted_ori['orientation'] = ori
        weighted_ori['weight'] = weights
        # Need to sort orientations to do histogram binning analysis
        weighted_ori = weighted_ori.sort_values(by = 'orientation', ignore_index = True)
        
        # Twelve 30 degree bins, with the first centered at 0
        bins = []
        # First bin must consider angle wrap around (359 is actually close to 1)
        bin_prop = sum(weighted_ori[(weighted_ori['orientation'] < 15) | (weighted_ori['orientation'] >= 345)]['weight'])
        if bin_prop > 0:
            bins.append(bin_prop) # Only add non-zero bin frequencies/proportions
        # Keep moving in 30 degree increments
        for n in range(0, 11):
            lbound = 15 + 30 * n
            ubound = 15 + 30 * (n + 1)
            bin_prop = sum(weighted_ori[(weighted_ori['orientation'] >= lbound) & (weighted_ori['orientation'] < ubound)]['weight'])
            if bin_prop > 0:
                bins.append(bin_prop) # Only add non-zero bin frequencies/proportions
        # Formulae for weighted orientation entropy and orienation order as per Boeing, 2019
        Hw = -sum(bins * np.log(bins))
        ori_order = 1 - ((Hw - 1.386) / (2.485 - 1.386)) ** 2
        

        """PART FOUR: COMPUTE STATS ON MITOCHONDRIA"""
        if mito:
            # Counts of total mito foci as well as within each branch type
            branch_assignments = mito_data['branch_type']
            tot_mito_ct = len(mito_data)
            prim_mito_ct = sum(branch_assignments == 1)
            sec_mito_ct = sum(branch_assignments == 2)
            tert_mito_ct = sum(branch_assignments == 3)
            quat_mito_ct = sum(branch_assignments == 4)
            
            # Get A-P axis position as % total length with the cell body position as 0%
            y_pos = [y * 100 / length - cellbody_pct for y in list(mito_data['centroid_y'])]
            # Positive % for mito posterior of the cell body
            ant_mito_ct = sum(1 for y in y_pos if y < 0)
            post_mito_ct = tot_mito_ct - ant_mito_ct
            
            # Nearest neighbor analysis
            coords = mito_data[['centroid_x', 'centroid_y']]
            coords = np.array(coords)
            tree = KDTree(coords)
            distances, indices = tree.query(coords, k=2)
            nearest_neighbor_distances = distances[:, 1]  # Second column = nearest neighbor dist (NND)
            mean_nnd = np.mean(nearest_neighbor_distances)
            n = len(coords)  # Number of mito foci
            # Compute expected mean NND and Rn statistic
            expected_mean_distance = 0.5 / np.sqrt(n / (length * width))
            Rn = mean_nnd / expected_mean_distance
            
            # Filtered DataFrame w/o unusually large foci (e.g., perinuclear) which can skew size-based measurements
            mito_filtered = mito_data[mito_data['size_zscore'] <= 2].iloc[:, [2, 5, 6, 8, 10, 13]]

        """Add a row to the output DataFrame with measurements of the given image.""" 
        if mito:
            stats.append({
                'image': img_name,
                'genotype-age': genotype_age,
                'length': length_um,
                'cellbody': cellbody_pct,
                'prim-ct': prim_ct / length_um, # Normalize by neuron length
                'prim-length': prim_length / length_um, # Normalize by neuron length
                'prim-wavy': prim_wavy,
                'prim-tort': prim_tort,
                'prim-curve': prim_curve,
                'prim-intensity': prim_intensity,
                'prim-angle': prim_angle,
                'prim-angle-sd': prim_angle_sd,
                'sec-ct': sec_ct / length_um, # Normalize by neuron length
                'sec-length': sec_length / length_um, # Normalize by neuron length
                'sec-wavy': sec_wavy,
                'sec-tort': sec_tort,
                'sec-curve': sec_curve,
                'sec-intensity': sec_intensity,
                'sec-angle': sec_angle,
                'sec-angle-sd': sec_angle_sd,
                'sec-median': sec_median,
                'sec-skew': sec_skew,
                'post-sec': post_sec / post_length, # Normalize by posterior neuron length
                'ant-sec': ant_sec / ant_length, # Normalize by anterior neuron length
                'tert-ct': tert_ct / length_um, # Normalize by neuron length
                'tert-length': tert_length / length_um, # Normalize by neuron length
                'tert-wavy': tert_wavy,
                'tert-tort': tert_tort,
                'tert-curve': tert_curve,
                'tert-intensity': tert_intensity,
                'tert-angle': tert_angle,
                'tert-angle-sd': tert_angle_sd,
                'post-tert': post_tert / post_length, # Normalize by posterior neuron length
                'ant-tert': ant_tert / ant_length, # Normalize by anterior neuron length
                'quat-ct': quat_ct / length_um, # Normalize by neuron length
                'quat-length': quat_length / length_um, # Normalize by neuron length
                'quat-wavy': quat_wavy,
                'quat-tort': quat_tort,
                'quat-curve': quat_curve,
                'quat-intensity': quat_intensity,
                'quat-angle': quat_angle,
                'quat-angle-sd': quat_angle_sd,
                'quat-median': quat_median,
                'quat-skew': quat_skew,
                'post-quat': post_quat / post_length, # Normalize by posterior neuron length
                'ant-quat': ant_quat / ant_length, # Normalize by anterior neuron length
                # Ct of each contact type expressed as % of total contacts
                '12-contacts': 100 * contacts_12 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                '13-contacts': 100 * contacts_13 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                '14-contacts': 100 * contacts_14 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                '23-contacts': 100 * contacts_23 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                '24-contacts': 100 * contacts_24 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                '34-contacts': 100 * contacts_34 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                'iba-mean': iba_mean,
                'iba-sd': iba_sd,
                'iba-skew': iba_skew,
                # Ct of each branch orientation bin expressed as proportion of total
                'angle-dist-bin1': bin_cts[0] / total_ct,
                'angle-dist-bin2': bin_cts[1] / total_ct,
                'angle-dist-bin3': bin_cts[2] / total_ct,
                'angle-dist-bin4': bin_cts[3] / total_ct,
                'angle-dist-bin5': bin_cts[4] / total_ct,
                'angle-dist-bin6': bin_cts[5] / total_ct,
                'angle-dist-bin7': bin_cts[6] / total_ct,
                'angle-dist-bin8': bin_cts[7] / total_ct,
                'angle-dist-bin9': bin_cts[8] / total_ct,
                'angle-dist-bin10': bin_cts[9] / total_ct,
                'angle-dist-bin11': bin_cts[10] / total_ct,
                'angle-dist-bin12': bin_cts[11] / total_ct,
                'angle-dist-bin13': bin_cts[12] / total_ct,
                'angle-dist-bin14': bin_cts[13] / total_ct,
                'angle-dist-bin15': bin_cts[14] / total_ct,
                'angle-dist-bin16': bin_cts[15] / total_ct,
                'angle-dist-bin17': bin_cts[16] / total_ct,
                'angle-dist-bin18': bin_cts[17] / total_ct,
                '12-len-ratio': length_12,
                '13-len-ratio': length_13,
                '14-len-ratio': length_14,
                '23-len-ratio': length_23,
                '24-len-ratio': length_24,
                '34-len-ratio': length_34,
                '23-ct-ratio': ct_23,
                '42-ct-ratio': ct_42,
                '43-ct-ratio': ct_43,
                'ori-order': round(ori_order, 3),
                'num-term-nodes': (len(nodes) - len(nodes_noterm)) / length_um, # Normalize by neuron length
                'pct-term-nodes': 100 * (len(nodes) - len(nodes_noterm)) / len(nodes),
                'int-density': len(nodes_noterm) / area, # Intersection nodes per unit area
                'edge-density': tot_length / area, # Total branch ('edge') length per unit area
                'mean-degree': np.mean(nodes_noterm['degree']),
                'num-degree-4+': num_high_degree / length_um, # Normalize by neuron length
                'pct-degree-4+': 100 * num_high_degree / len(nodes_noterm), # What % of intersection nodes are degree 4+?
                'mean-betweenness': np.mean(nodes_noterm['betweenness']),
                'loop-ct': nodes['loops'][0],
                
                'mito-tot-ct': tot_mito_ct / length_um, # Normalize by neuron length
                'mito-prim-ct': prim_mito_ct / length_um, # Normalize by neuron length
                'mito-sec-ct': sec_mito_ct / length_um, # Normalize by neuron length
                'mito-tert-ct': tert_mito_ct / length_um, # Normalize by neuron length
                'mito-quat-ct': quat_mito_ct / length_um, # Normalize by neuron length
                'mito-ant-ct': ant_mito_ct / ant_length, # Normalize by anterior neuron length
                'mito-post-ct': post_mito_ct / post_length, # Normalize by posterior neuron length
                'mito-branch-pt-ct': sum(mito_in_branch_pts) / length_um, # Normalize by neuron length
                'mito-branch-pt-pct': 100 * sum(mito_in_branch_pts) / len(mito_data),
                'mean-nnd': mean_nnd * 0.2096, # Make in units of microns
                'rn': Rn,
                # Size-independent metrics are simply summary stats of measurements from mito data
                'mito-ecc-mean': np.mean(mito_data['eccentricity']) ,
                'mito-ecc-sd': np.std(mito_data['eccentricity']),
                'mito-ecc-skew': scs.skew(mito_data['eccentricity']),
                'mito-ext-mean': np.mean(mito_data['extent']) ,
                'mito-ext-sd': np.std(mito_data['extent']),
                'mito-ext-skew': scs.skew(mito_data['extent']),
                'mito-perimarea-mean': np.mean(mito_data['perim_area_ratio']) ,
                'mito-perimarea-sd': np.std(mito_data['perim_area_ratio']),
                'mito-perimarea-skew': scs.skew(mito_data['perim_area_ratio']),
                'mito-ori-sd': np.std(mito_data['orientation']),
                # Distributional metrics based on y_pos as a % adjusted for cell body position
                'mito-dist-med': np.median(y_pos),
                'mito-dist-sd': np.std(y_pos),
                'mito-dist-skew': scs.skew(y_pos),
                # Size-dependent metrics are summary stats of measurements from filtered mito data with large outliers removed
                'mito-size-mean': np.mean(mito_filtered['size']),
                'mito-size-sd': np.std(mito_filtered['size']),
                'mito-size-skew': scs.skew(mito_filtered['size']),
                'mito-majaxis-mean': np.mean(mito_filtered['maj_axis']),
                'mito-majaxis-sd': np.std(mito_filtered['maj_axis']),
                'mito-majaxis-skew': scs.skew(mito_filtered['maj_axis']),
                'mito-minaxis-mean': np.mean(mito_filtered['min_axis']),
                'mito-minaxis-sd': np.std(mito_filtered['min_axis']),
                'mito-minaxis-skew': scs.skew(mito_filtered['min_axis']),
                'mito-diam-mean': np.mean(mito_filtered['equiv_diam_area']),
                'mito-diam-sd': np.std(mito_filtered['equiv_diam_area']),
                'mito-diam-skew': scs.skew(mito_filtered['equiv_diam_area']),
                'mito-perim-mean': np.mean(mito_filtered['perim']),
                'mito-perim-sd': np.std(mito_filtered['perim']),
                'mito-perim-skew': scs.skew(mito_filtered['perim']),
                'mito-feret-mean': np.mean(mito_filtered['feret_diam']),
                'mito-feret-sd': np.std(mito_filtered['feret_diam']),
                'mito-feret-skew': scs.skew(mito_filtered['feret_diam'])
                })
        else:
            stats.append({
                'image': img_name,
                'genotype-age': genotype_age,
                'length': length_um,
                'cellbody': cellbody_pct,
                'prim-ct': prim_ct / length_um, # Normalize by neuron length
                'prim-length': prim_length / length_um, # Normalize by neuron length
                'prim-wavy': prim_wavy,
                'prim-tort': prim_tort,
                'prim-curve': prim_curve,
                'prim-intensity': prim_intensity,
                'prim-angle': prim_angle,
                'prim-angle-sd': prim_angle_sd,
                'sec-ct': sec_ct / length_um, # Normalize by neuron length
                'sec-length': sec_length / length_um, # Normalize by neuron length
                'sec-wavy': sec_wavy,
                'sec-tort': sec_tort,
                'sec-curve': sec_curve,
                'sec-intensity': sec_intensity,
                'sec-angle': sec_angle,
                'sec-angle-sd': sec_angle_sd,
                'sec-median': sec_median,
                'sec-skew': sec_skew,
                'post-sec': post_sec / post_length, # Normalize by posterior neuron length
                'ant-sec': ant_sec / ant_length, # Normalize by anterior neuron length
                'tert-ct': tert_ct / length_um, # Normalize by neuron length
                'tert-length': tert_length / length_um, # Normalize by neuron length
                'tert-wavy': tert_wavy,
                'tert-tort': tert_tort,
                'tert-curve': tert_curve,
                'tert-intensity': tert_intensity,
                'tert-angle': tert_angle,
                'tert-angle-sd': tert_angle_sd,
                'post-tert': post_tert / post_length, # Normalize by posterior neuron length
                'ant-tert': ant_tert / ant_length, # Normalize by anterior neuron length
                'quat-ct': quat_ct / length_um, # Normalize by neuron length
                'quat-length': quat_length / length_um, # Normalize by neuron length
                'quat-wavy': quat_wavy,
                'quat-tort': quat_tort,
                'quat-curve': quat_curve,
                'quat-intensity': quat_intensity,
                'quat-angle': quat_angle,
                'quat-angle-sd': quat_angle_sd,
                'quat-median': quat_median,
                'quat-skew': quat_skew,
                'post-quat': post_quat / post_length, # Normalize by posterior neuron length
                'ant-quat': ant_quat / ant_length, # Normalize by anterior neuron length
                # Ct of each contact type expressed as % of total contacts
                '12-contacts': 100 * contacts_12 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                '13-contacts': 100 * contacts_13 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                '14-contacts': 100 * contacts_14 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                '23-contacts': 100 * contacts_23 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                '24-contacts': 100 * contacts_24 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                '34-contacts': 100 * contacts_34 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
                'iba-mean': iba_mean,
                'iba-sd': iba_sd,
                'iba-skew': iba_skew,
                # Ct of each branch orientation bin expressed as proportion of total
                'angle-dist-bin1': bin_cts[0] / total_ct,
                'angle-dist-bin2': bin_cts[1] / total_ct,
                'angle-dist-bin3': bin_cts[2] / total_ct,
                'angle-dist-bin4': bin_cts[3] / total_ct,
                'angle-dist-bin5': bin_cts[4] / total_ct,
                'angle-dist-bin6': bin_cts[5] / total_ct,
                'angle-dist-bin7': bin_cts[6] / total_ct,
                'angle-dist-bin8': bin_cts[7] / total_ct,
                'angle-dist-bin9': bin_cts[8] / total_ct,
                'angle-dist-bin10': bin_cts[9] / total_ct,
                'angle-dist-bin11': bin_cts[10] / total_ct,
                'angle-dist-bin12': bin_cts[11] / total_ct,
                'angle-dist-bin13': bin_cts[12] / total_ct,
                'angle-dist-bin14': bin_cts[13] / total_ct,
                'angle-dist-bin15': bin_cts[14] / total_ct,
                'angle-dist-bin16': bin_cts[15] / total_ct,
                'angle-dist-bin17': bin_cts[16] / total_ct,
                'angle-dist-bin18': bin_cts[17] / total_ct,
                '12-len-ratio': length_12,
                '13-len-ratio': length_13,
                '14-len-ratio': length_14,
                '23-len-ratio': length_23,
                '24-len-ratio': length_24,
                '34-len-ratio': length_34,
                '23-ct-ratio': ct_23,
                '42-ct-ratio': ct_42,
                '43-ct-ratio': ct_43,
                'ori-order': round(ori_order, 3),
                'num-term-nodes': (len(nodes) - len(nodes_noterm)) / length_um, # Normalize by neuron length
                'pct-term-nodes': 100 * (len(nodes) - len(nodes_noterm)) / len(nodes),
                'int-density': len(nodes_noterm) / area, # Intersection nodes per unit area
                'edge-density': tot_length / area, # Total branch ('edge') length per unit area
                'mean-degree': np.mean(nodes_noterm['degree']),
                'num-degree-4+': num_high_degree / length_um, # Normalize by neuron length
                'pct-degree-4+': 100 * num_high_degree / len(nodes_noterm), # What % of intersection nodes are degree 4+?
                'mean-betweenness': np.mean(nodes_noterm['betweenness']),
                'loop-ct': nodes['loops'][0]
            })
        
    return pd.DataFrame(stats)

"""
Function to plot every individual morphological profile feature.
Inputs:
    data = DataFrame of morphological profiles
    folder = path to the dataset folder of images
    mito = whether or not the dataset has mitochondrial channels
"""
def plot_all_features(data: pd.DataFrame, folder: str, mito = True):
    # List of all feature short names for plotting
    short_features = [
        'ln', 'cb', 
        'ct1', 'ln1', 'wv1', 'tt1', 'cv1', 'it1', 'ag1', 'as1',
        'ct2', 'ln2', 'wv2', 'tt2', 'cv2', 'it2', 'ag2', 'as2', 'md2', 'sk2', 'pt2', 'at2',
        'ct3', 'ln3', 'wv3', 'tt3', 'cv3', 'it3', 'ag3', 'as3', 'pt3', 'at3',
        'ct4', 'ln4', 'wv4', 'tt4', 'cv4', 'it4', 'ag4', 'as4', 'md4', 'sk4', 'pt4', 'at4',
        'cn12', 'cn13', 'cn14', 'cn23', 'cn24', 'cn34', 'im', 'is', 'ik', 'ad1', 'ad2', 'ad3', 'ad4', 'ad5', 'ad6', 'ad7', 'ad8', 'ad9', 'ad10', 'ad11', 'ad12', 'ad13', 'ad14', 'ad15', 'ad16', 'ad17', 'ad18', 'lr12', 'lr13', 'lr14', 'lr23', 'lr24', 'lr34', 'cr23', 'cr42', 'cr43',
        'oo', 'tn', 'pt', 'id', 'ed', 'md', 'nd4', 'pd4', 'bt', 'lp',
        'mct', 'mct1', 'mct2', 'mct3', 'mct4', 'mac', 'mpc', 'mbc', 'mbp', 'nn', 'rn', 'mec', 'mes', 'mek', 'mex', 'mxs', 'mxk', 'mpa', 'mps', 'mpk', 'mos', 'mdm', 'mds', 'mdk', 'msz', 'mss', 'msk', 'mja', 'mjs', 'mjk', 'mna', 'mns', 'mnk', 'meq', 'mqs', 'mqk', 'mpm', 'mms', 'mmk', 'mfd', 'mfs', 'mfk'
        ]
    if not mito:
        short_features = short_features[:90]

    # Make a plots subfolder for saving plots
    Path(folder + '/plots/').mkdir(exist_ok = True)
    
    # Plot each feature and save as a png
    for feature in short_features:
        fig = pvd.phenotype_stripchart(data, feature, size = (3, 3)) # Plot function # !!! 
        # Save the plot in the plots subfolder
        fig.savefig(folder + '/plots/' + feature + '.png', bbox_inches = 'tight') 
        plt.close(fig)
        print(f'plotted {feature}')

"""
Generate distinguishable colors for genotypes and marker shapes for ages.
Parameters:
    n_genotypes : int
        Number of genotypes
    n_ages : int
        Number of age timepoints

Returns:
    tuple : (genotype_colors, age_markers)
        - genotype_colors: [color]
        - age_markers: [marker]
"""
def generate_genotype_colors_and_age_markers(n_genotypes: int, n_ages: int) -> tuple[list]:
    # Generate distinct colors for each genotype
    base_hues = np.linspace(0, 1, n_genotypes, endpoint=False) # Hues are maximally spaced
    genotype_colors = []
    
    for i in range(n_genotypes):
        saturation = 0.85
        lightness = 0.55
        # Convert HSB to hex code of RGB color
        rgb = mcolors.hsv_to_rgb([base_hues[i], saturation, lightness])
        hex_color = mcolors.to_hex(rgb)
        genotype_colors.append(hex_color)
    
    # Define marker shapes for each age
    # Using easily distinguishable markers
    marker_options = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    age_markers = []
    
    for i in range(n_ages):
        # Modulus (%) operator allows for recycling markers if n_ages > 10
        age_markers.append(marker_options[i % len(marker_options)])
    
    return genotype_colors, age_markers

# %%

def main():
    args = parse_args()
    FPATH = args.dataset_path
    DRY = args.dry_run
    
    """Extract data on each image in the dataset."""
    folder = Path(FPATH)
    for in_path in folder.glob('*/*Straightened.tif'):
        # Getting input Straightened.tif paths to put into img_to_branches()
        if '._' in in_path.stem:
            continue
        if 'mito_' in in_path.stem:
            continue
        # Manage paths of output data tables
        in_path = str(in_path)
        out_path1 = in_path.replace('Straightened.tif', 'branches.csv')
        out_path2 = in_path.replace('Straightened.tif', 'mito.csv')
        out_path3 = in_path.replace('Straightened.tif', 'nodes.csv')
        # Dry run: list files that would be processed/produced and skip rest of loop
        if DRY:
            print(f'Processing {in_path}.')
            print(f'Generating {out_path1}.')
            if HAS_MITO:
                print(f'Generating {out_path2}.')
            print(f'Generating {out_path3}.')
            continue

        branches, mito_data, node_data = img_to_branches(in_path, neurite_seg_model, mito_seg_model, compute_device, classifier)

        # Save output data tables that are not None
        if branches is not None:
            branches.to_csv(out_path1)
        if mito_data is not None:
            mito_data.to_csv(out_path2)
        if node_data is not None:
            node_data.to_csv(out_path3)
        
        print(f'Processed image: {in_path}')
    
    """Generate morphological profiles and save as stats.csv in the dataset folder."""
    if DRY:
        print(f'Generating morphological profiles at {FPATH}/stats.csv.')
    else:
        data = write_feature_table(FPATH, mito = HAS_MITO)
        data.to_csv(FPATH + '/stats.csv')
        
    """Plot features"""
    if DRY:
        print(f'Generating 131 plots at {FPATH}/plots/.')
    else:    
        plot_all_features(data, FPATH, mito = HAS_MITO)
    
    """PCA"""
    if not DRY:
        # Get normalized data
        features = list(data.columns)[3:] # [3:] excludes img name, genotype, and length from features
        x = data.loc[:, features].values
        x = StandardScaler().fit_transform(x) # Normalize data for PCA
        
        # Perform PCA with 2 principal components
        pca = PCA(n_components = 2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents,
                                columns = ['PC_1', 'PC_2'])
        # Add columns for img name and genotype
        plottingDf = pd.concat([principalDf, data[['genotype-age', 'image']]], axis = 1)
    
    # Make PCA plot
    # Make lookup structure linking genotype/age combos with colors and markers
    genotypes_ages = set(list(data.loc[:, 'genotype-age'])) # Get only unique values
    genotypes_ages = list(genotypes_ages)
    geno_age_df = [] # DataFrame will have one row for each combo, columns for geno and age
    for combo in genotypes_ages:
        genotype, age = combo.split('-')
        geno_age_df.append({
            'genotype': genotype,
            'age': age,
        })
    geno_age_df = pd.DataFrame(geno_age_df).sort_values(by = ['genotype', 'age'])
    n_genotypes = len(geno_age_df['genotype'].unique()) # num of unique genotypes
    n_ages = int(len(genotypes_ages) / n_genotypes)
    colors, markers = generate_genotype_colors_and_age_markers(n_genotypes, n_ages)
    # Add colors and markers to lookup table
    _colors = []
    for color in colors:
        for i in range(n_ages):
            _colors.append(color)
    geno_age_df['color'] = _colors # [[c1] * a, [c2] * a, ... [cg] * a]; a = n_ages, g = n_genos 
    geno_age_df['marker'] = markers * n_genotypes # [m1, m2, ..., ma] * g
    # For dry run, show the genotypes/ages that would be plotted and their markers/colors
    if DRY:
        print(f'Generating PCA plot at {FPATH}/pca.png.')
        for row in geno_age_df.iterrows():
            print(row)
    if not DRY:
        # Figure setup
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        # For each genotype/age, plot corresponding data points with distinct color/marker
        for row in geno_age_df.iterrows():
            indicesToKeep = plottingDf['genotype-age'] == f"{row['genotype']}-{row['age']}"
            ax.scatter(plottingDf.loc[indicesToKeep, 'PC_1'], 
                    plottingDf.loc[indicesToKeep, 'PC_2'], 
                    c = row['color'],
                    marker = row['marker'],
                    s = 50)
        ax.legend(genotypes_ages)
        ax.grid()
        # Save plot
        fig.savefig(FPATH + '/pca.png', bbox_inches = 'tight')
        plt.close(fig)
    
    if DRY:
        print(f'Generating principal component loadings at {FPATH}/loadings.csv.')
    else:
        # Principal component loadings
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_) # Formula for loading
        # Save as a DataFrame
        loadingsDf = pd.DataFrame(loadings, columns = ['PC_1', 'PC_2'], index = features)
        loadingsDf.to_csv(FPATH + '/loadings.csv')
    
    if DRY:
        print(f'Generating PCA report at {FPATH}/report.txt.')
    else:
        # Generate report with PCA stats
        report = open(FPATH + "/report.txt", "w", encoding = "utf-8")
        # Percent variance explained by each PC
        expl_var = pca.explained_variance_ratio_
        report.write(f"Principal component 1: {expl_var[0]:.1%} of variance explained\n")
        report.write(f"Principal component 2: {expl_var[1]:.1%} of variance explained\n")
        # Number of images
        report.write(f"\n{len(plottingDf)} images included in analysis:\n")
        # Include image names
        images = plottingDf['image']
        for image in images:
            report.write(image + "\n")
        report.close()


# %%

if __name__ == "__main__":
    main()

