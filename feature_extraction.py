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

# Obtain morphological profiles of all images in a given directory/folder
def write_feature_table(folder, adj_branch_distributions = True):
    folder = Path(folder)
    
    stats = []
    
    for filepath in folder.glob('*branches.csv'):
        
        data = pd.read_csv(filepath).iloc[:, 1:]
        data['branch'] = [eval(branch) for branch in data['branch']]
        data['neighbors'] = [eval(neighbor) for neighbor in data['neighbors']]
        
        # get neuron length and cell body pos
        img_name = str(folder.parent) + '/maxProj-1/' + filepath.stem.replace('branches', 'maxProj.tif')
        img = tifffile.imread(img_name)
        length = img.shape[0]
        cellbody = find_cellbody(img_name)
        cellbody_pct = 100 * cellbody / length
        
        # Make in units of microns
        length_um = round(length * 0.1048)
        ant_length = round(cellbody * 0.1048)
        post_length = length_um - ant_length
        
        # get stats per class
        # 1. primary
        prim = data[data['dendrite_type'] == 1]
        prim_ct = len(prim)
        prim_length = round(sum(prim['length']) * 0.1048)
        prim_wavy = np.mean(prim['tortuosity'])
        prim_curve = np.mean(prim['curvature'])
        prim_intensity = np.mean(prim['intensity'])
        prim_angle = np.mean(prim['orientation'])
        prim_angle_sd = np.std(prim['orientation'])
        
        # 2. secondary
        sec = data[data['dendrite_type'] == 2]
        sec_ct = len(sec)
        sec_length = round(sum(sec['length']) * 0.1048)
        sec_wavy = np.mean(sec['tortuosity'])
        sec_curve = np.mean(sec['curvature'])
        sec_intensity = np.mean(sec['intensity'])
        
        sec_angles = list(sec['orientation'])
        for i in range(sec_ct):
            if sec_angles[i] > 90:
                sec_angles[i] = sec_angles[i] - 180
        sec_angle = np.mean(sec_angles)
        sec_angle_sd = np.std(sec_angles)
        
        # get 2º distribution info
        sec_branches = list(sec['branch'])
        y_pos = []
        for n in range(sec_ct):
            branch = sec_branches[n]
            y_coords = [pt[0] for pt in branch]
            if adj_branch_distributions:
                y_pos.append(np.mean(y_coords) * 100 / length - cellbody_pct)
            else:
                y_pos.append(np.mean(y_coords) * 100 / length)
        
        sec_median = np.median(y_pos)
        sec_skew = scs.skew(y_pos)
        
        post_sec = sum(1 for y in y_pos if y > 0)
        ant_sec = len(y_pos) - post_sec
        
        # 3. tertiary
        tert = data[data['dendrite_type'] == 3]
        tert_ct = len(tert)
        tert_length = round(sum(tert['length']) * 0.1048)
        tert_wavy = np.mean(tert['tortuosity'])
        tert_curve = np.mean(tert['curvature'])
        tert_intensity = np.mean(tert['intensity'])
        tert_angle = np.mean(tert['orientation'])
        tert_angle_sd = np.std(tert['orientation'])
        
        # get 3º distribution info
        tert_branches = list(tert['branch'])
        y_pos = []
        for n in range(tert_ct):
            branch = tert_branches[n]
            y_coords = [pt[0] for pt in branch]
            if adj_branch_distributions:
                y_pos.append(np.mean(y_coords) * 100 / length - cellbody_pct)
            else:
                y_pos.append(np.mean(y_coords) * 100 / length)
        
        post_tert = sum(1 for y in y_pos if y > 0)
        ant_tert = len(y_pos) - post_tert
        
        # 4. quaternary
        quat = data[data['dendrite_type'] == 4]
        quat_ct = len(quat)
        quat_length = round(sum(quat['length']) * 0.1048)
        quat_wavy = np.mean(quat['tortuosity'])
        quat_curve = np.mean(quat['curvature'])
        quat_intensity = np.mean(quat['intensity'])
        
        quat_angles = list(quat['orientation'])
        for i in range(quat_ct):
            if quat_angles[i] > 90:
                quat_angles[i] = quat_angles[i] - 180
        quat_angle = np.mean(quat_angles)
        quat_angle_sd = np.std(quat_angles)
        
        # get 4º distribution info
        quat_branches = list(quat['branch'])
        y_pos = []
        for n in range(quat_ct):
            branch = quat_branches[n]
            y_coords = [pt[0] for pt in branch]
            if adj_branch_distributions:
                y_pos.append(np.mean(y_coords) * 100 / length - cellbody_pct)
            else:
                y_pos.append(np.mean(y_coords) * 100 / length)
        
        quat_median = np.median(y_pos)
        quat_skew = scs.skew(y_pos)
        
        post_quat = sum(1 for y in y_pos if y > 0)
        ant_quat = len(y_pos) - post_quat
        
        # get basic stats
        img_name = filepath.stem.replace('_branches', '')
        filepath = str(filepath)
        genotype = filepath.split('_')[2:4]
        genotype = genotype[0] + '-' + genotype[1]
        
        # get global stats
        # 1. interbranch angles and contacts
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
        
        # 2. length and count ratios
        length_12 = prim_length / sec_length
        length_13 = prim_length / tert_length
        length_14 = prim_length / quat_length
        length_23 = sec_length / tert_length
        length_24 = sec_length / quat_length
        length_34 = prim_length / sec_length
        
        ct_23 = sec_ct / tert_ct
        ct_42 = quat_ct / sec_ct 
        ct_43 = quat_ct / tert_ct
        
        # 3. overall orientation distribution
        bin_cts = np.histogram(data['orientation'], np.linspace(0, 180, 19))[0]
        total_ct = prim_ct + sec_ct + tert_ct + quat_ct
        
        stats.append({
            'image': img_name,
            'genotype': genotype,
            'length': length_um,
            'cellbody': cellbody_pct,
            'prim-ct': prim_ct / length_um,
            'prim-length': prim_length / length_um,
            'prim-wavy': prim_wavy,
            'prim-curve': prim_curve,
            'prim-intensity': prim_intensity,
            'prim-angle': prim_angle,
            'prim-angle-sd': prim_angle_sd,
            'sec-ct': sec_ct,# / length_um,
            'sec-length': sec_length / length_um,
            'sec-wavy': sec_wavy,
            'sec-curve': sec_curve,
            'sec-intensity': sec_intensity,
            'sec-angle': sec_angle,
            'sec-angle-sd': sec_angle_sd,
            'sec-median': sec_median,
            'sec-skew': sec_skew,
            'post-sec': post_sec/post_length,# / length_um,
            'ant-sec': ant_sec/ant_length,# / length_um,
            'tert-ct': tert_ct,# / length_um,
            'tert-length': tert_length / length_um,
            'tert-wavy': tert_wavy,
            'tert-curve': tert_curve,
            'tert-intensity': tert_intensity,
            'tert-angle': tert_angle,
            'tert-angle-sd': tert_angle_sd,
            'post-tert': post_tert/post_length,# / length_um,
            'ant-tert': ant_tert/ant_length,# / length_um,
            'quat-ct': quat_ct,# / length_um,
            'quat-length': quat_length / length_um,
            'quat-wavy': quat_wavy,
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
            '43-ct-ratio': ct_43
            })
    return pd.DataFrame(stats)

# %%

# Generate morphological profiles
data = write_feature_table('{your_experiment_dir}/branches/')
data.to_csv('{your_experiment_dir}/featureExtraction/')

# Perform PCA
features = list(data.columns)[2:]
x = data.loc[:, features].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components = 2) # Can change if more components are desired
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['PC_1', 'PC_2'])
plottingDf = pd.concat([principalDf, data[['image', 'genotype']]], axis = 1)

plottingDf.to_csv('{file_name}.csv') # Save PCA

# Make PCA plot
colors = ['#ff7045', '#338bff', '#6ce000', '#ff0000', '#0000ff', '#00a100']

genotypes = set(list(data.loc[:, 'genotype']))
genotypes = list(genotypes)
genotypes_sort = sorted(genotypes)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

for genotype, color in zip(genotypes_sort ,colors):
    indicesToKeep = plottingDf['genotype'] == genotype
    ax.scatter(plottingDf.loc[indicesToKeep, 'PC_1'], 
               plottingDf.loc[indicesToKeep, 'PC_2'], 
               c = color, 
               s = 50)

ax.legend(genotypes_sort, bbox_to_anchor=(1, 0.5))
ax.grid()

# Get proportion of variance explained by each PC
pca.explained_variance_ratio_

# Get principal component loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_) 
loadingsDf = pd.DataFrame(loadings, columns = ['PC_1', 'PC_2'], index = features)
loadingsDf.to_csv('{file_name}.csv') # Save loadings


# %%
# Plot a specific morphological feature

pvd_plots.phenotype_stripchart(data, 'md4', (3.3, 3)) # 2nd argument uses 2-letter codes for each feature
# To see these codings, view the pvd_plots module file
