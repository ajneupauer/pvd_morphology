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
import scipy

def sort_key(strain):
    genotype, age = strain.split('-')
    age_num = int(age.replace('day', ''))
    # Return tuple: (age_number, genotype_priority)
    # wt gets priority 0, anc1null gets priority 1
    genotype_priority = 0 if genotype == 'wt' else 1
    return (age_num, genotype_priority)

def phenotype_stripchart(data, trait, size = (10, 5), ylimit = None, dotsize = 8):
    # Initialize dictionary of traits of interest and corresponding plot titles and y-axis labels
    # 'key':('trait-name-in-csv', 'Plot Title', 'Y-axis Label')
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
        #'as3':('tert-angle-sd', 'SD of Orientations of 3º Dendrites', 'Degrees (º)'),
        #'md3':('tert-median', 'Median of 3º Distribution', 'Percent (%)'),
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
    
    
    filtered = data.filter(items=['genotype', traits[trait][0]])
    strains = list(filtered.iloc[:, 0].unique())
    x_labs = sorted(strains, key=sort_key)
    
    y_avg = []
    y_err = []
    feature_data = []
    i = 0
    
    for strain in x_labs:
        data_by_strain = filtered[filtered['genotype'] == strain].iloc[:, 1]
        feature_data.append(data_by_strain)
        y_avg.append(np.mean(data_by_strain))
        y_err.append(1.96 * np.std(data_by_strain)/np.sqrt(len(data_by_strain)))
        i += 1
    if len(strains) == 2:
        pval = scipy.stats.ttest_ind(feature_data[0], feature_data[1], equal_var = False).pvalue
        
    fig, ax = plt.subplots(figsize = size)
    if ylimit is not None:
        ax.set_ylim(ylimit[0], ylimit[1])
    ax.errorbar(x_labs, y_avg, y_err, fmt = 'r_', markersize = 10, capsize = 5, linewidth = 2, barsabove = True)
    ax = sns.stripplot(x = 'genotype', y = traits[trait][0], data = filtered, 
                      jitter = 0.1, size = dotsize, color = 'k')
    ax.set_xticklabels(x_labs, size= 12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(None)
    plt.ylabel(traits[trait][2], size= 12)
    plt.yticks(fontsize = 16)
    if len(strains) == 2:
        plt.text(0.7, 0.95, f'p = {pval:.2e}', transform = plt.gca().transAxes) 
    ax.set_title(traits[trait][1], weight = 'bold', size = 14, wrap = True)
    
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

