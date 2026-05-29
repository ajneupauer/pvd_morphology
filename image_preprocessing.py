#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:44:49 2024

@author: alexneupauer
"""

# Import Modules
import shutil
import argparse
import json
import os
os.chdir('starr-luxton-lab/pvd-project/pvd_morphology/')
import sys
sys.path.append('./modules')

import tifffile
from pathlib import Path
from ims import ImarisReader
import numpy as np

payload = json.loads(Path('./config.json').read_text(encoding="utf-8"))
HAS_MITO = payload.get("has_mito") # def == True
HAS_MITO = True if HAS_MITO == 1 else 0
CHANNELS = payload.get("channels")
INPUT_IMG_FMT = payload.get("input_img_fmt")

# %%

FPATH = Path('{dir_to_raw_images}')
# DRY (False) from CLI
DRY = False

def parse_args() -> argparse.Namespace: # the -> defines which type is returned
    parser = argparse.ArgumentParser(
        description="Perform image preprocessing in preparation for image straightening."
    )
    parser.add_argument("dataset_path", type=Path, help="Directory to dataset of raw images.") # args.dataset_path
    parser.add_argument("--dry-run", action="store_true", default = False) # args.dry_run
    return parser.parse_args()

# args = parse_args()
# FPATH = args.dataset_path
# DRY = args.dry_run

# build_output_dirs(FPATH)
def build_output_dirs(folder: Path, img_ext = ".ims"):
    files = []

    for file in folder.glob('*' + img_ext):
        if '._' in file.stem:
            continue
        files.append(file)

    files.sort()

    for file in files:
        
        outdir_name = str(folder) + '/' + file.stem
        outdir_name = Path(outdir_name)
        outdir_name.mkdir(exist_ok = True)
        if not file.exists():
            continue
        else:
            shutil.move(file, outdir_name)
    
    return files

# Define preprocessing functions

# Make an 8x downsampled max intensity projection
def make_small(in_img):
    dsImg = in_img[:, ::8, ::8]
    maxProj = dsImg.max(axis = 0)
    return np.uint16(maxProj)

# Make a 2x downsampled z-stack
def make_squished(in_img, downsample = True):
    if downsample:
        in_img = in_img[:, ::2, ::2]
    depth, height, width = in_img.shape
    out_img = np.empty([depth // 2, height, width])
    for i in range(depth // 2):
        out_img[i] = in_img[2*i:2*(i+1)].max(axis = 0)
    return np.uint16(out_img)    

def filter_empty_layers(image):
    depth = image.shape[0]
    layers_to_keep = []
    
    for z in range(depth):
        if np.mean(image[z]) >= 80:
            layers_to_keep.append(z)
        else:
            continue
            #print(f'Removing layer {z}')        
    
    edited = image[layers_to_keep, :, :]
    return edited


# %%
# For many images

fpath = Path('{dir_to_raw_images}')
files = []

for file in fpath.glob('*small.tif'):
    if '._' in file.stem:
        continue
    files.append(file.stem.replace('_small', ''))

files.sort()

for file in files:    
    file = Path(file)
    outpath = '{dir_to_outputs}' + file.stem.replace('Confocal - 561_Confocal - 488_', '')[11:]
    
    myImg = ImarisReader(file).get_image_data(channel = 1, return_array = True)
    squished = make_squished(myImg, downsample = False)
    final = filter_empty_layers(squished)
    tifffile.imwrite(outpath.replace('FusionStitcher', 'mito_squished.tif'), final, compression = 'lzw')
    
    myImg = ImarisReader(file).get_image_data(channel = 0, return_array = True)
    small = make_small(myImg)
    squished = make_squished(myImg)
    final = filter_empty_layers(squished)
    tifffile.imwrite(outpath.replace('FusionStitcher', 'small.tif'), small, compression = 'lzw')
    tifffile.imwrite(outpath.replace('FusionStitcher', 'squished.tif'), final, compression = 'lzw')
    
# ! = may need to adjust based on how your files are named
    
