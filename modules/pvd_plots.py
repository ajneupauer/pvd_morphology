#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 20:16:13 2025

@author: alexneupauer
"""

import pandas as pd
import tifffile
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def sort_key(strain):
    genotype, age = strain.split('-')
    age_num = int(age.replace('day', ''))
    # Return tuple: (age_number, genotype_priority)
    # wt gets priority 0, anc1null gets priority 1
    genotype_priority = 0 if genotype == 'wt' else 1
    return (age_num, genotype_priority)

def phenotype_stripchart(data, trait, size = (10, 5)):
    # Initialize dictionary of traits of interest and corresponding plot titles and y-axis labels
    # 'key':('trait-name-in-csv', 'Plot Title', 'Y-axis Label')
    traits = {
        'ln':('length', 'Worm Length', 'Length ($\mu$m)'),
        'cb':('cellbody', 'Cell Body Position Along the Anterior-Posterior Axis', 'Position (%)'),
        'ct1':('prim-ct', 'Number of Primary Dendrites', 'Count'),
        'ct2':('sec-ct', 'Number of 2º Dendrites', 'Count'),
        'ct3':('tert-ct', 'Number of 3º Dendrites', 'Count'),
        'ct4':('quat-ct', 'Number of 4º Dendrites', 'Count'),
        'ol':('overlap-count', 'Number of Menorah Overlaps', 'Count'),
        'md4':('quat-median', 'Medians of 4º Distribution', 'Median (%)'),
        'sk4':('quat-skew', 'Skewness of 4º Distribution', 'Skewness'),
        'pt4':('post-quat', 'Number of Posterior 4º Dendrites', 'Count'),
        'at4':('ant-quat', 'Number of Anterior 4º Dendrites', 'Count'),
        'pt3':('post-tert', 'Number of Posterior 3º Dendrites', 'Count'),
        'at3':('ant-tert', 'Number of Anterior 3º Dendrites', 'Count'),
        'md2':('sec-median', 'Medians of 2º Distribution', 'Median (%)'),
        'sk2':('sec-skew', 'Skewness of 2º Distribution', 'Skewness'),
        'pt2':('post-sec', 'Number of Posterior 2º Dendrites', 'Count'),
        'at2':('ant-sec', 'Number of Anterior 2º Dendrites', 'Count'),
        'cv1':('prim-curve', 'Curvature of 1º Dendrites', 'Degrees (º)'),
        'cv2':('sec-curve', 'Curvature of 2º Dendrites', 'Degrees (º)'),
        'cv3':('tert-curve', 'Curvature of 3º Dendrites', 'Degrees (º)'),
        'cv4':('quat-curve', 'Curvature of 4º Dendrites', 'Degrees (º)'),
        'wv1':('prim-wavy', 'Waviness of 1º Dendrites', 'Sign Changes * º/$\mu$m'),
        'wv2':('sec-wavy', 'Waviness of 2º Dendrites', 'Sign Changes * º/$\mu$m'),
        'wv3':('tert-wavy', 'Waviness of 3º Dendrites', 'Sign Changes * º/$\mu$m'),
        'wv4':('quat-wavy', 'Waviness of 4º Dendrites', 'Sign Changes * º/$\mu$m'),
        'ag1':('prim-angle', 'Mean Orientation of 1º Dendrites', 'Degrees (º)'),
        'ag2':('sec-angle', 'Mean Orientation of 2º Dendrites', 'Degrees (º)'),
        'ag3':('tert-angle', 'Mean Orientation of 3º Dendrites', 'Degrees (º)'),
        'ag4':('quat-angle', 'Mean Orientation of 4º Dendrites', 'Degrees (º)'),
        'as1':('prim-angle-sd', 'SD of Orientations of 1º Dendrites', 'Degrees (º)'),
        'as2':('sec-angle-sd', 'SD of Orientations of 2º Dendrites', 'Degrees (º)'),
        'as3':('tert-angle-sd', 'SD of Orientations of 3º Dendrites', 'Degrees (º)'),
        'as4':('quat-angle-sd', 'SD of Orientations of 4º Dendrites', 'Degrees (º)'),
        'im':('iba-mean', 'Mean Interbranch Angle', 'Degrees (º)'),
        'is':('iba-sd', 'SD of Interbranch Angles', 'Degrees (º)'),
        'ik':('iba-skew', 'Skewness of Interbranch Angle Distribution', 'Skewness'),
        'cr23':('23-ct-ratio', 'Ratio of 2º to 3º Dendrites', 'No. 2º / No. 3º'),
        'cn34':('34-contacts', 'Number of 3º/4º Contacts', 'Count'),
        'lr14':('14-len-ratio', 'Ratio of 1º to 4º Dendrite Length', '1º Length / 4º Length')
        }
    
    filtered = data.filter(items=['genotype', traits[trait][0]])
    strains = list(filtered.iloc[:, 0].unique())
    x_labs = sorted(strains, key=sort_key)
    
    y_avg = []
    y_err = []
    for strain in x_labs:
        data_by_strain = filtered[filtered['genotype'] == strain].iloc[:, 1]
        y_avg.append(np.mean(data_by_strain))
        y_err.append(1.96 * np.std(data_by_strain)/np.sqrt(len(data_by_strain)))

    fig, ax = plt.subplots(figsize = size)
    #ax = seaborn.stripplot(x = 'genotype', y = traits[trait][0], data = filtered, 
    #                  jitter = 0.1, size = 8, color = 'k')
    ax.set_xticklabels(x_labs, size=12) #prev 12
    ax.errorbar(x_labs, y_avg, y_err, fmt = 'r_', markersize = 10, capsize = 5, linewidth = 2)
    ax = sns.stripplot(x = 'genotype', y = traits[trait][0], data = filtered, 
                      jitter = 0.1, size = 8, color = 'k')
    #for i in range(5):
    #    ax.plot([filtered[traits[trait][0]][i], 
    #             filtered[traits[trait][0]][i + 5]], 
    #             color = 'blue', linestyle='dashed')
    #ax.set_ylim(0, 170)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(None)
    plt.ylabel(traits[trait][2], size=12) #prev 12
    plt.yticks(fontsize = 10)
    ax.set_title(traits[trait][1], weight = 'bold', size = 14)
    
    return fig

def plot_branch_dist(infolder, strain, dendrite_type):

    exp_id = str(infolder).split('-')[-1]
    infolder = Path(infolder)
    y_pos = []
    pattern = f'*{strain}*.csv'
    
    for file in infolder.glob(pattern = pattern):
        branches = pd.read_csv(file)
        image_name = str(infolder.parent) + f'/maxProj-{exp_id}/' + file.stem.replace('branches', 'maxProj.tif')
        length = tifffile.imread(image_name).shape[0]
        
        quat_only = branches[branches['dendrite_type'] == dendrite_type]
        quat_branches = [eval(branch) for branch in quat_only['branch']]
        
        for n in range(len(quat_only)):
            branch = quat_branches[n]
            y_coords = [pt[0] for pt in branch]
            y_pos.append(np.mean(y_coords) * 100 / length)
    
    cutoffs = [i for i in range(0, 105, 5)]
    
    fig, ax = plt.subplots()
    ax = sns.kdeplot(y_pos)
    ax.hist(y_pos, bins = cutoffs, density = True)
    ax.set_xlabel('Percent Along Anterior-Posterior Axis', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_title(f'Distribution of {dendrite_type}º Dendrites', fontsize=14)
    ax.text(0.05, 0.95, f'{len(y_pos)}\ndendrites', transform=ax.transAxes, fontsize=12,
            verticalalignment='top')
    
    return fig

