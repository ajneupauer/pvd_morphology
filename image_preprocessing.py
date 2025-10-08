#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:44:49 2024

@author: alexneupauer
"""

# Import Modules
import os
os.chdir('{dir_where_repo_is_stored}/pvd_morphology/')
import sys
sys.path.append('./modules')

import tifffile
from pathlib import Path
from ims import ImarisReader
import numpy as np

# %%
# Define preprocessing functions

# Make an 8x downsampled max intensity projection
def make_small(in_img):
    dsImg = in_img[:, ::8, ::8]
    maxProj = dsImg.max(axis = 0)
    return np.uint16(maxProj)

# Make a 2x downsampled z-stack
def make_squished(in_img):
    dsImg = in_img[:, ::2, ::2]
    depth, height, width = dsImg.shape
    out_img = np.empty([depth // 2, height, width])
    for i in range(depth // 2):
        out_img[i] = dsImg[2*i:2*(i+1)].max(axis = 0)
    return np.uint16(out_img)    

# %%
# For many images

fpath = Path('{dir_to_raw_images}')

for file in fpath.glob('*ims'):
    print(file)

    outpath = '{dir_to_outputs}' + file.stem.replace('Confocal - 561_Confocal - 488_', '')[11:] # ! 
    
    myImg = ImarisReader(file).get_image_data(return_array = True)
    
    small = make_small(myImg)
    squished = make_squished(myImg)
    
    tifffile.imwrite(outpath.replace('FusionStitcher', 'small.tif'), small, compression = 'lzw') # ! 
    tifffile.imwrite(outpath.replace('FusionStitcher', 'squished.tif'), squished, compression = 'lzw') # ! 

# ! = may need to adjust based on how your files are named
    

# %%
# For a single image

fpath = Path('{raw_img_path}')
outpath = '{dir_to_outputs}' + fpath.stem.replace('Confocal - 561_Confocal - 488_', '')[11:] # ! 

myImg = ImarisReader(fpath).get_image_data(return_array = True)


small = make_small(myImg)
squished = make_squished(myImg)

tifffile.imwrite(outpath.replace('FusionStitcher', '6_small.tif'), small, compression = 'lzw') # ! 
tifffile.imwrite(outpath.replace('FusionStitcher', '6_squished.tif'), squished, compression = 'lzw') # ! 


