#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:50:44 2025

@author: alexneupauer
"""

# Import modules

import os
os.chdir('{dir_where_repo_is_stored}/pvd_morphology/')
import sys
sys.path.append('./modules')

import pandas as pd
import tifffile
from pathlib import Path
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pvd_plots
import random
from scipy.spatial import KDTree


# %%
# Define functions for PCA and plotting

# Find the y position of the cell body (pixels)
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

# From a list of neighbors of all branches, find unique pairs of neighboring branches
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

# Obtain morphological profiles of all images in a given directory/folder
def write_feature_table(folder, adj_branch_distributions = True):
    # Collect all files
    folder = Path(folder)
    files = []
    for file in folder.glob('*branches.csv'):
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
        genotype = file.split('_')[2:4]
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
        bin_prop = sum(weighted_ori[(weighted_ori['orientation'] < 5) | (weighted_ori['orientation'] >= 355)]['weight'])
        if bin_prop > 0:
            bins.append(bin_prop)
        for n in range(0, 35):
            lbound = 5 + 10 * n
            ubound = 5 + 10 * (n + 1)
            bin_prop = sum(weighted_ori[(weighted_ori['orientation'] >= lbound) & (weighted_ori['orientation'] < ubound)]['weight'])
            if bin_prop > 0:
                bins.append(bin_prop)
        
        Hw = -sum(bins * np.log(bins))
        ori_order = 1 - ((Hw - 1.386) / (3.584 - 1.386)) ** 2
        
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
            'area': area,
            'num-term-nodes': (len(nodes) - len(nodes_noterm)) / (0.2096 * length),
            'pct-term-nodes': 100 * (len(nodes) - len(nodes_noterm)) / len(nodes),
            'int-density': len(nodes_noterm) / area,
            'edge-density': tot_length / area,
            'mean-degree': np.mean(nodes_noterm['degree']),
            'num-degree-4+': num_high_degree / (0.2096 * length),
            'pct-degree-4+': 100 * num_high_degree / len(nodes_noterm),
            'mean-betweenness': np.mean(nodes_noterm['betweenness']),
            'loop-ct': nodes['loops'][0],
            
            'mito-tot-ct': tot_mito_ct, ##
            'mito-prim-ct': prim_mito_ct,
            'mito-sec-ct': sec_mito_ct,
            'mito-tert-ct': tert_mito_ct,
            'mito-quat-ct': quat_mito_ct,
            'mito-ant-ct': ant_mito_ct,
            'mito-post-ct': post_mito_ct,
            'mito_branch_pt_ct': sum(mito_in_branch_pts) / (0.2096 * length),
            'mito_branch_pt_pct': 100 * sum(mito_in_branch_pts) / len(mito_data),
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

# %%

# Generate morphological profiles
data = write_feature_table('{your_experiment_dir}/')
data.to_csv('{your_experiment_dir}/stats.csv')

# Perform PCA
features = list(data.columns)[2:]
x = data.loc[:, features].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['PC_1', 'PC_2'])
plottingDf = pd.concat([principalDf, data[['genotype', 'image']]], axis = 1)

#strains_to_keep = ['N2_D1', 'N2_D5', 'N2_D9', 'CH_D1', 'CH_D5', 'CH_D9']
#plottingDf = plottingDf[plottingDf['Genotype'].isin(strains_to_keep)]

colors = ['#ff7045', '#338bff', '#6ce000', '#ff0000', '#0000ff', '#00a100']
#colors = ['#ff0000', '#0000ff']
#colors, markers = generate_genotype_colors_and_age_markers(n_genotypes=2, n_ages=3)

genotypes = set(list(data.loc[:, 'genotype']))
genotypes = list(genotypes)
genotypes_sort = sorted(genotypes)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

for genotype, color in zip(genotypes ,colors):
    indicesToKeep = plottingDf['genotype'] == genotype
    ax.scatter(plottingDf.loc[indicesToKeep, 'PC_1'], 
               plottingDf.loc[indicesToKeep, 'PC_2'], 
               c = color, 
               s = 50)

ax.legend(genotypes_sort)
ax.grid()


pca.explained_variance_ratio_

loadings = pca.components_.T * np.sqrt(pca.explained_variance_) 
loadingsDf = pd.DataFrame(loadings, columns = ['PC_1', 'PC_2'], index = features)
loadingsDf.to_csv('{your_experiment_dir}/loadings.csv')


# %%
# Plot a specific morphological feature

pvd_plots.phenotype_stripchart(data, 'md4', (3.3, 3)) # 2nd argument uses 2-letter codes for each feature
# To see these codings, view the pvd_plots module file
