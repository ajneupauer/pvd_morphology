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
import pvd_plots
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

mito_seg_path = Path(MITO_SEG_PATH)
mito_seg_model = models.AttentionUNet(1, 1, features=[16, 32, 64, 128], use_logits=True)
unet_dict = torch.load(mito_seg_path, weights_only=False)
mito_seg_model.load_state_dict(unet_dict["model_state_dict"])
mito_seg_model = mito_seg_model.to(compute_device)

classifier = pc1.PVDNeuriteClassifier()
classifier.load_model(CLASSIFIER_PATH)

# %%

"""
Collect user arguments passed to the command line.
dataset_path: directory to dataset of raw images
--dry-run: add if performing a dry run (won't make/modify anything')
--ngeno: number of genotypes in dataset
--nage: number of timepoints in dataset
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and analyze morphological profiles of PVD neuron images."
    )
    parser.add_argument("dataset_path", type=str, help="Directory to dataset of raw images.")
    parser.add_argument("--ngeno", type=int, default = 2, help ="Number of genotypes in dataset.")
    parser.add_argument("--nage", type=int, default = 1, help = "Number of timepoints in dataset.")
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
    
    # Step 2: Make mask
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
    branches_path = img_path.replace('Straightened.tif', 'branches.csv')
    nodes_path = img_path.replace('Straightened.tif', 'nodes.csv')
    if not(already_exists(branches_path) and already_exists(nodes_path)):
        fragments = pvd.classify_mask(mask2d, mip, classifier)
        print('Step 3a completed successfully:\nClassify branch fragments.\n')
    if already_exists(nodes_path):
        print('Nodes data already created!')
        node_data = None
    else:
        G, node_data = pvd.reconstruct_graph_from_segments(fragments) # Get node data
        betweenness = pvd.calculate_betweenness_centrality(G)
        node_data['betweenness'] = betweenness
        loops = [pvd.count_loops(G)['total_cycles']]
        loops = loops + (len(node_data) - 1) * ['NA']
        node_data['loops'] = loops
    
    if already_exists(branches_path):
        print('Steps 3b-c already completed:\nCorrect and reconstruct fragments into branches.\n')
        branch_stats = None
    else:
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
    
    if mito_seg_model is not None:    
        # Step 4: Process mito image w/ neurite mask
        if already_exists(img_path.replace('Straightened', 'mito_rmbg')):
            mito_rmbg = tifffile.imread(img_path.replace('Straightened', 'mito_rmbg'))
            print('Step 4 already completed:\nProcess mito image for segmentation.\n')
        else:    
            mask3d = tifffile.imread(mask3d_path)
            mito_straightened = tifffile.imread(mito_straightened_path)
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
        
    if branch_stats is None:
        print('Already generated branch data!')
    if node_data is None:
        print('Already generated node data!')
    if mito_stats is None:
        print('Already generated mito data or no mito channel available!')
    
    #print('Image analysis complete.\n')
    
    return branch_stats, mito_stats, node_data


def find_cellbody(fpath):
    myImg = tifffile.imread(fpath)
    
    profile = []
    for n in range(myImg.shape[0]):
        lbound = round(myImg.shape[1] * 0.375) 
        rbound = round(myImg.shape[1] * 0.655) 
        row = myImg[n, lbound:rbound]
        avg_intensity = np.mean(row)
        profile.append(avg_intensity)
    
    smooth_profile = []
    for i in range(len(profile) - 100):
        sliding_avg = np.mean(profile[i:i+100])
        smooth_profile.append(sliding_avg)
    
    peaks = []
    for k in range(1, len(smooth_profile) - 1):
        if smooth_profile[k] > smooth_profile[k + 1] + 0.0001 and smooth_profile[k] > smooth_profile[k - 1] + 0.0001:
            peaks.append(smooth_profile[k])
    
    breakpt = smooth_profile.index(max(peaks))
    return breakpt + 50

def uniq_neighbor_pairs(neighbors):
    neighbor_pairs = []
    
    for i in range(len(neighbors)):
        for j in range(len(neighbors[i])):
            if i < neighbors[i][j]:
                neighbor_pairs.append((i, neighbors[i][j]))
    
    return neighbor_pairs

def sort_key(strain):
    genotype, age = strain.split('-')
    age_num = int(age.replace('day', ''))
    # Return tuple: (age_number, genotype_priority)
    # wt gets priority 0, anc1null gets priority 1
    genotype_priority = 0 if genotype == 'wt' else 1
    return (age_num, genotype_priority)

def generate_random_hex_color():
    """Generates a random hexadecimal color code."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f'#{r:02x}{g:02x}{b:02x}'

def write_feature_table(folder, adj_branch_distributions = True):
    # Collect all files
    folder = Path(folder)
    files = []
    for file in folder.glob('*/*branches.csv'):
        files.append(str(file))
    files.sort()
    
    # Generate a morphological profile of each file
    stats = []
    
    for file in files:
        # PART ONE: READ DATA
        data = pd.read_csv(file).iloc[:, 1:]
        data['branch'] = [eval(branch) for branch in data['branch']]
        data['neighbors'] = [eval(neighbor) for neighbor in data['neighbors']]
        nodes = pd.read_csv(file.replace('branches.csv', 'nodes.csv')).iloc[:, 1:]
        mito_data = pd.read_csv(file.replace('branches.csv', 'mito.csv')).iloc[:, 1:]
        img_name = file.replace('branches.csv', 'mip.tif')
        img = tifffile.imread(img_name)
        length, width = img.shape # get image dimensions in px
        
        # Get basic stats
        cellbody = find_cellbody(img_name) # px
        img_name = Path(file).stem.replace('_branches', '')
        genotype = Path(file).stem.split('_')[2:4]
        genotype = genotype[0] + '-' + genotype[1]
        
        
        # PART TWO: COMPUTE STATS PER EACH BRANCH CLASS
        # I: Get Neuron Length And Cell Body Pos
        cellbody_pct = 100 * cellbody / length # %
        length_um = round(length * 0.2096) # make in units of microns
        ant_length = round(cellbody * 0.2096)
        post_length = length_um - ant_length
        
        # II: Primary
        prim = data[data['dendrite_type'] == 1]
        prim_ct = len(prim)
        prim_length = round(sum(prim['length']))
        prim_wavy = np.mean(prim['waviness'])
        prim_tort_mask = np.isfinite(prim['tortuosity'])
        prim_tort = np.mean(prim['tortuosity'][prim_tort_mask])
        prim_curve = np.mean(prim['curvature'])
        prim_intensity = np.mean(prim['intensity'])
        prim_angle = np.mean(prim['orientation'])
        prim_angle_sd = np.std(prim['orientation'])
        
        # III: Secondary
        sec = data[data['dendrite_type'] == 2]
        sec_ct = len(sec)
        sec_length = round(sum(sec['length']))
        sec_wavy = np.mean(sec['waviness'])
        sec_tort_mask = np.isfinite(sec['tortuosity'])
        sec_tort = np.mean(sec['tortuosity'][sec_tort_mask])
        sec_curve = np.mean(sec['curvature'])
        sec_intensity = np.mean(sec['intensity'])
        
        sec_angles = list(sec['orientation'])
        for i in range(sec_ct):
            if sec_angles[i] > 90:
                sec_angles[i] = sec_angles[i] - 180
        sec_angle = np.mean(sec_angles)
        sec_angle_sd = np.std(sec_angles)
        
        # Get 2º distribution info
        if adj_branch_distributions:
            y_pos = [y * 100 / length - cellbody_pct for y in list(sec['mean_y'])]
        else:
            y_pos = [y * 100 / length for y in list(sec['mean_y'])]
        
        #for n in range(sec_ct):
        #    branch = sec_branches[n]
        #    y_coords = [pt[0] for pt in branch]
        #    if adj_branch_distributions:
        #        y_pos.append(np.mean(y_coords) * 100 / length - cellbody_pct)
        #    else:
        #        y_pos.append(np.mean(y_coords) * 100 / length)
        
        sec_median = np.median(y_pos)
        sec_skew = scs.skew(y_pos)
        
        post_sec = sum(1 for y in y_pos if y > 0)
        ant_sec = len(y_pos) - post_sec
        
        # IV: Tertiary
        tert = data[data['dendrite_type'] == 3]
        tert_ct = len(tert)
        tert_length = round(sum(tert['length']))
        tert_wavy = np.mean(tert['waviness'])
        tert_tort_mask = np.isfinite(tert['tortuosity'])
        tert_tort = np.mean(tert['tortuosity'][tert_tort_mask])
        tert_curve = np.mean(tert['curvature'])
        tert_intensity = np.mean(tert['intensity'])
        tert_angle = np.mean(tert['orientation'])
        tert_angle_sd = np.std(tert['orientation'])
        
        # Get 3º distribution info
        if adj_branch_distributions:
            y_pos = [y * 100 / length - cellbody_pct for y in list(tert['mean_y'])]
        else:
            y_pos = [y * 100 / length for y in list(tert['mean_y'])]
        
        post_tert = sum(1 for y in y_pos if y > 0)
        ant_tert = len(y_pos) - post_tert
        
        # V: Quaternary
        quat = data[data['dendrite_type'] == 4]
        quat_ct = len(quat)
        quat_length = round(sum(quat['length']))
        quat_wavy = np.mean(quat['waviness'])
        quat_tort_mask = np.isfinite(quat['tortuosity'])
        quat_tort = np.mean(quat['tortuosity'][quat_tort_mask])
        quat_curve = np.mean(quat['curvature'])
        quat_intensity = np.mean(quat['intensity'])
        
        quat_angles = list(quat['orientation'])
        for i in range(quat_ct):
            if quat_angles[i] > 90:
                quat_angles[i] = quat_angles[i] - 180
        quat_angle = np.mean(quat_angles)
        quat_angle_sd = np.std(quat_angles)
        
        # Get 4º distribution info
        if adj_branch_distributions:
            y_pos = [y * 100 / length - cellbody_pct for y in list(quat['mean_y'])]
        else:
            y_pos = [y * 100 / length for y in list(quat['mean_y'])]
        
        quat_median = np.median(y_pos)
        quat_skew = scs.skew(y_pos)
        
        post_quat = sum(1 for y in y_pos if y > 0)
        ant_quat = len(y_pos) - post_quat
        
        
        # PART THREE: COMPUTE GLOBAL STATS ON ENTIRE NEURITE NETWORK
        # I: Interbranch Angles And Contacts
        neighbor_pairs = uniq_neighbor_pairs(list(data['neighbors']))
        
        interbranch_angles = []
        contacts_12 = 0
        contacts_13 = 0
        contacts_14 = 0
        contacts_23 = 0
        contacts_24 = 0
        contacts_34 = 0

        for pair in neighbor_pairs:
            delta_angle = data.loc[pair[0]]['orientation'] - data.loc[pair[1]]['orientation']
            angle_diff = min(abs(delta_angle), 360 - abs(delta_angle))
            interbranch_angles.append(angle_diff)
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
        bin_cts = np.histogram(data['orientation'], np.linspace(0, 180, 19))[0]
        total_ct = prim_ct + sec_ct + tert_ct + quat_ct
        
        # IV: 'City Streets' Metrics
        nodes = nodes[nodes['type'] != 'continuation']
        nodes_noterm = nodes[nodes['type'] == 'branch_point']
        
        mito_in_branch_pts = []
        
        for ref_idx, ref_row in mito_data.iterrows():
            mito_x = ref_row['centroid_x']
            mito_y = ref_row['centroid_y']
            match_found = False
            
            for query_idx, query_row in nodes_noterm.iterrows():
                if match_found: 
                    break
                x_inrange = (query_row['x_pos'] >= mito_x - 5) and (query_row['x_pos'] <= mito_x + 5)
                y_inrange = (query_row['y_pos'] >= mito_y - 5) and (query_row['y_pos'] <= mito_y + 5)
                if x_inrange and y_inrange:
                    match_found = True
            
            mito_in_branch_pts.append(match_found)
        
        area = (0.2096 ** 2) * length * width
        
        # Number of nodes with degree >= 4
        num_high_degree = sum(nodes_noterm['degree'] >= 4)
        
        # Cumulative length of all branches
        tot_length = sum(data['length'])
        
        # Orientation order
        ori_1 = list(data['orientation'])
        ori_2 = [angle + 180 for angle in ori_1]
        ori = ori_1 + ori_2
        ar = length / width 
        
        length_adj = []
        for idx, row in data.iterrows():
            # need to divide 1º/3º lengths by the aspect ratio, ar
            if row['dendrite_type'] == 1 or row['dendrite_type'] == 3:
                length_adj.append(row['length'] / ar)
            else:
                length_adj.append(row['length'])
        
        cumul_len = sum(length_adj)
        weights = [i / (2 * cumul_len) for i in length_adj]
        weights = 2 * weights
        
        weighted_ori = pd.DataFrame(columns = ['orientation', 'weight'])
        weighted_ori['orientation'] = ori
        weighted_ori['weight'] = weights
        weighted_ori = weighted_ori.sort_values(by = 'orientation', ignore_index = True)
        
        bins = []
        bin_prop = sum(weighted_ori[(weighted_ori['orientation'] < 15) | (weighted_ori['orientation'] >= 345)]['weight'])
        if bin_prop > 0:
            bins.append(bin_prop)
        for n in range(0, 11):
            lbound = 15 + 30 * n
            ubound = 15 + 30 * (n + 1)
            bin_prop = sum(weighted_ori[(weighted_ori['orientation'] >= lbound) & (weighted_ori['orientation'] < ubound)]['weight'])
            if bin_prop > 0:
                bins.append(bin_prop)
        
        Hw = -sum(bins * np.log(bins))
        ori_order = 1 - ((Hw - 1.386) / (2.485 - 1.386)) ** 2
        
        # PART FOUR: COMPUTE STATS ON MITOCHONDRIA!!
        branch_assignments = mito_data['branch_type']
        
        tot_mito_ct = len(mito_data)
        prim_mito_ct = sum(branch_assignments == 1)
        sec_mito_ct = sum(branch_assignments == 2)
        tert_mito_ct = sum(branch_assignments == 3)
        quat_mito_ct = sum(branch_assignments == 4)
        
        y_pos = [y * 100 / length - cellbody_pct for y in list(mito_data['centroid_y'])]
        
        ant_mito_ct = sum(1 for y in y_pos if y < 0)
        post_mito_ct = tot_mito_ct - ant_mito_ct
        
        coords = mito_data[['centroid_x', 'centroid_y']]
        coords = np.array(coords)
        tree = KDTree(coords)
        distances, indices = tree.query(coords, k=2)
        nearest_neighbor_distances = distances[:, 1]  # Second column = nearest neighbor
        mean_nnd = np.mean(nearest_neighbor_distances)
        n = len(coords)  # number of points
        area = length * width
        expected_mean_distance = 0.5 / np.sqrt(n / area)
        Rn = mean_nnd / expected_mean_distance
        
        mito_filtered = mito_data[mito_data['size_zscore'] <= 2].iloc[:, [2, 5, 6, 8, 10, 13]]

        
        stats.append({
            'image': img_name,
            'genotype': genotype,
            'length': length_um,
            'cellbody': cellbody_pct,
            'prim-ct': prim_ct / length_um,
            'prim-length': prim_length / length_um,
            'prim-wavy': prim_wavy,
            'prim-tort': prim_tort,
            'prim-curve': prim_curve,
            'prim-intensity': prim_intensity,
            'prim-angle': prim_angle,
            'prim-angle-sd': prim_angle_sd,
            'sec-ct': sec_ct / length_um,
            'sec-length': sec_length / length_um,
            'sec-wavy': sec_wavy,
            'sec-tort': sec_tort,
            'sec-curve': sec_curve,
            'sec-intensity': sec_intensity,
            'sec-angle': sec_angle,
            'sec-angle-sd': sec_angle_sd,
            'sec-median': sec_median,
            'sec-skew': sec_skew,
            'post-sec': post_sec/post_length,# / length_um,
            'ant-sec': ant_sec/ant_length,# / length_um,
            'tert-ct': tert_ct / length_um,
            'tert-length': tert_length / length_um,
            'tert-wavy': tert_wavy,
            'tert-tort': tert_tort,
            'tert-curve': tert_curve,
            'tert-intensity': tert_intensity,
            'tert-angle': tert_angle,
            'tert-angle-sd': tert_angle_sd,
            'post-tert': post_tert/post_length,# / length_um,
            'ant-tert': ant_tert/ant_length,# / length_um,
            'quat-ct': quat_ct / length_um,
            'quat-length': quat_length / length_um,
            'quat-wavy': quat_wavy,
            'quat-tort': quat_tort,
            'quat-curve': quat_curve,
            'quat-intensity': quat_intensity,
            'quat-angle': quat_angle,
            'quat-angle-sd': quat_angle_sd,
            'quat-median': quat_median,
            'quat-skew': quat_skew,
            'post-quat': post_quat/post_length,# / length_um,
            'ant-quat': ant_quat/ant_length,# / length_um,
            '12-contacts': 100 * contacts_12 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
            '13-contacts': 100 * contacts_13 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
            '14-contacts': 100 * contacts_14 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
            '23-contacts': 100 * contacts_23 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
            '24-contacts': 100 * contacts_24 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
            '34-contacts': 100 * contacts_34 / (contacts_12 + contacts_13 + contacts_14 + contacts_23 + contacts_24 + contacts_34),
            'iba-mean': iba_mean,
            'iba-sd': iba_sd,
            'iba-skew': iba_skew,
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
            #'area': area,
            'num-term-nodes': (len(nodes) - len(nodes_noterm)) / (0.2096 * length),
            'pct-term-nodes': 100 * (len(nodes) - len(nodes_noterm)) / len(nodes),
            'int-density': len(nodes_noterm) / area,
            'edge-density': tot_length / area,
            'mean-degree': np.mean(nodes_noterm['degree']),
            'num-degree-4+': num_high_degree / (0.2096 * length),
            'pct-degree-4+': 100 * num_high_degree / len(nodes_noterm),
            'mean-betweenness': np.mean(nodes_noterm['betweenness']),
            'loop-ct': nodes['loops'][0],
            
            'mito-tot-ct': tot_mito_ct / length_um, ##
            'mito-prim-ct': prim_mito_ct / length_um,
            'mito-sec-ct': sec_mito_ct / length_um,
            'mito-tert-ct': tert_mito_ct / length_um,
            'mito-quat-ct': quat_mito_ct / length_um,
            'mito-ant-ct': ant_mito_ct / ant_length,
            'mito-post-ct': post_mito_ct / post_length,
            'mito-branch-pt-ct': sum(mito_in_branch_pts) / (0.2096 * length),
            'mito-branch-pt-pct': 100 * sum(mito_in_branch_pts) / len(mito_data),
            'mean-nnd': mean_nnd * 0.2096,
            'rn': Rn,
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
            'mito-dist-med': np.median(y_pos),
            'mito-dist-sd': np.std(y_pos),
            'mito-dist-skew': scs.skew(y_pos),
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
    
    return pd.DataFrame(stats)

def plot_all_features(data, folder):
    short_features = [
        'ln', 'cb', 
        'ct1', 'ln1', 'wv1', 'tt1', 'cv1', 'it1', 'ag1', 'as1',
        'ct2', 'ln2', 'wv2', 'tt2', 'cv2', 'it2', 'ag2', 'as2', 'md2', 'sk2', 'pt2', 'at2',
        'ct3', 'ln3', 'wv3', 'tt3', 'cv3', 'it3', 'ag3', 'as3', 'pt3', 'at3',
        'ct4', 'ln4', 'wv4', 'tt4', 'cv4', 'it4', 'ag4', 'as4', 'md4', 'sk4', 'pt4', 'at4',
        'cn12', 'cn13', 'cn14', 'cn23', 'cn24', 'cn34', 'im', 'is', 'ik', 'ad1', 'ad2', 'ad3', 'ad4', 'ad5', 'ad6', 'ad7', 'ad8', 'ad9', 'ad10', 'ad11', 'ad12', 'ad13', 'ad14', 'ad15', 'ad16', 'ad17', 'ad18', 'lr12', 'lr13', 'lr14', 'lr23', 'lr24', 'lr34', 'cr23', 'cr42', 'cr43',
        'oo', 'tn', 'pt', 'id', 'ed', 'md', 'nd4', 'pd4', 'bt', 'lp',
        'mct', 'mct1', 'mct2', 'mct3', 'mct4', 'mac', 'mpc', 'mbc', 'mbp', 'nn', 'rn', 'mec', 'mes', 'mex', 'mxs', 'mxk', 'mpa', 'mps', 'mpk', 'mos', 'mdm', 'mds', 'mdk', 'msz', 'mss', 'msk', 'mja', 'mjs', 'mjk', 'mna', 'mns', 'mnk', 'meq', 'mqs', 'mqk', 'mpm', 'mms', 'mmk', 'mfd', 'mfs', 'mfk'
        ]
    
    # deleted 'mek'
    
    Path(folder + '/plots/').mkdir(exist_ok = True)
    
    for feature in short_features:
        fig = pvd_plots.phenotype_stripchart(data, feature, size = (3, 3))
        fig.savefig(folder + '/plots/' + feature + '.png', bbox_inches = 'tight')
        plt.close(fig)
        print(f'plotted {feature}')

def generate_genotype_colors_and_age_markers(n_genotypes, n_ages):
    """
    Generate distinguishable colors for genotypes and marker shapes for ages.
    
    Parameters:
    -----------
    n_genotypes : int
        Number of genotypes (default: 7)
    n_ages : int
        Number of age timepoints (default: 3)
    
    Returns:
    --------
    tuple : (color_dict, marker_dict, combined_dict)
        - color_dict: {genotype: color}
        - marker_dict: {age: marker}
        - combined_dict: {genotype: {age: (color, marker)}}
    """
    
    # Generate distinct colors for each genotype
    base_hues = np.linspace(0, 1, n_genotypes, endpoint=False)
    genotype_colors = []
    
    for i in range(n_genotypes):
        saturation = 0.85
        lightness = 0.55
        
        rgb = mcolors.hsv_to_rgb([base_hues[i], saturation, lightness])
        hex_color = mcolors.to_hex(rgb)
        genotype_colors.append(hex_color)
    
    # Define marker shapes for each age
    # Using easily distinguishable markers
    marker_options = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    age_markers = []
    
    for i in range(n_ages):
        age_markers.append(marker_options[i % len(marker_options)])
    
    return genotype_colors, age_markers

# %%

def main():
    args = parse_args()
    FPATH = args.dataset_path
    DRY = args.dry_run
    NAGE = args.ngeno
    NGENO = args.nage
    
    folder = Path(FPATH)
    for in_path in folder.glob('*/*Straightened.tif'):
        if '._' in in_path.stem:
            continue
        if 'mito_' in in_path.stem:
            continue
    
        in_path = str(in_path)
        out_path1 = in_path.replace('Straightened.tif', 'branches.csv')
        out_path2 = in_path.replace('Straightened.tif', 'mito.csv')
        out_path3 = in_path.replace('Straightened.tif', 'nodes.csv')
        
        branches, mito_data, node_data = img_to_branches(in_path, neurite_seg_model, mito_seg_model, compute_device, classifier)
    
        if branches is not None:
            branches.to_csv(out_path1)
        if mito_data is not None:
            mito_data.to_csv(out_path2)
        if node_data is not None:
            node_data.to_csv(out_path3)
        
        print(f'Processed image: {in_path}')
    
    data = write_feature_table(FPATH)
    data.to_csv(FPATH + '/stats.csv')
    
    # Plot features
    plot_all_features(data, FPATH)
    
    # PCA
    features = list(data.columns)[3:]
    x = data.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents,
                               columns = ['PC_1', 'PC_2'])
    plottingDf = pd.concat([principalDf, data[['genotype', 'image']]], axis = 1)
    
    colors, markers = generate_genotype_colors_and_age_markers(n_genotypes=NGENO, n_ages=NAGE)
    
    genotypes = set(list(data.loc[:, 'genotype']))
    genotypes = list(genotypes)
    #genotypes_sort = sorted(genotypes)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    
    for genotype, color, marker in zip(genotypes ,colors, markers):
        indicesToKeep = plottingDf['genotype'] == genotype
        ax.scatter(plottingDf.loc[indicesToKeep, 'PC_1'], 
                   plottingDf.loc[indicesToKeep, 'PC_2'], 
                   c = color,
                   marker = marker,
                   s = 50)
    
    ax.legend(genotypes)
    ax.grid()
    
    fig.savefig(FPATH + '/pca.png', bbox_inches = 'tight')
    plt.close(fig)
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_) 
    loadingsDf = pd.DataFrame(loadings, columns = ['PC_1', 'PC_2'], index = features)
    loadingsDf.to_csv(FPATH + '/loadings.csv')
    
    report = open(FPATH + "/report.txt", "w", encoding = "utf-8")
    expl_var = pca.explained_variance_ratio_
    report.write(f"Principal component 1: {expl_var[0]:.1%} of variance explained\n")
    report.write(f"Principal component 2: {expl_var[1]:.1%} of variance explained\n")
    report.write(f"\n{len(plottingDf)} images included in analysis:\n")
    
    images = plottingDf['image']
    for image in images:
        report.write(image + "\n")
    
    report.close()


# %%

if __name__ == "__main__":
    main()

