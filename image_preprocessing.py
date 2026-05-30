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
os.chdir('/Users/alexneupauer/starr-luxton-lab/pvd-project/pvd_morphology/')
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

#FPATH = Path('{dir_to_raw_images}')
# DRY (False) from CLI
#DRY = False

def parse_args() -> argparse.Namespace: # the -> defines which type is returned
    parser = argparse.ArgumentParser(
        description="Perform image preprocessing in preparation for image straightening."
    )
    parser.add_argument("dataset_path", type=Path, help="Directory to dataset of raw images.") # args.dataset_path
    parser.add_argument("--dry-run", action="store_true") # args.dry_run
    return parser.parse_args()

# args = parse_args()
# FPATH = args.dataset_path
# DRY = args.dry_run

# build_output_dirs(FPATH)
def build_output_dirs(folder: Path, dry: bool, img_ext = ".ims"):
    files = []
    img_paths = []

    for file in folder.glob('*' + img_ext):
        if '._' in file.stem:
            continue
        files.append(file)

    files.sort()

    for file in files:
        outdir_name = str(folder) + '/' + file.stem
        img_paths.append(outdir_name + '/' + file.name)
        if dry:
            print(f"Made dir: {outdir_name}.")
            print(f"Moved {file.name} to {outdir_name}.\n")
            continue
        else:
            outdir_name = Path(outdir_name)
            outdir_name.mkdir(exist_ok = True)
            if not file.exists():
                continue
            else:
                shutil.move(file, outdir_name)
    
    return img_paths


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

def main():
    args = parse_args()
    FPATH = args.dataset_path
    DRY = args.dry_run
    print(DRY)
    
    img_paths = build_output_dirs(FPATH, DRY, INPUT_IMG_FMT)
    
    for file in img_paths:    
        if DRY:
            # print out what would be saved
            if HAS_MITO:
                if INPUT_IMG_FMT != ".ims" or INPUT_IMG_FMT != ".tif":
                    print("Invalid image format!")
                mito_save_msg = str(file).replace(INPUT_IMG_FMT, '_mito_squished.tif')
                print(f"Saved {mito_save_msg}.")
            if INPUT_IMG_FMT != ".ims" or INPUT_IMG_FMT != ".tif":
                print("Invalid image format!")
            small_save_msg = str(file).replace(INPUT_IMG_FMT, '_small.tif')
            squished_save_msg = str(file).replace(INPUT_IMG_FMT, '_squished.tif')
            print(f"Saved {small_save_msg}.")
            print(f"Saved {squished_save_msg}.\n")
            continue
        
        if HAS_MITO:
            if INPUT_IMG_FMT == ".tif":
                img = tifffile.imread(file)[CHANNELS['mito']]
            elif INPUT_IMG_FMT == ".ims":
                file = Path(file)
                #outpath = '{dir_to_outputs}' + file.stem.replace('Confocal - 561_Confocal - 488_', '')[11:]
                img = ImarisReader(file).get_image_data(channel = CHANNELS['mito'], return_array = True)
            else:
                print("Invalid image format!")
            
            squished = make_squished(img, downsample = False)
            final = filter_empty_layers(squished)
            tifffile.imwrite(str(file).replace(INPUT_IMG_FMT, '_mito_squished.tif'), final, compression = 'lzw')
            
        if INPUT_IMG_FMT == ".tif":
            img = tifffile.imread(file)[CHANNELS['neurites']]
        elif INPUT_IMG_FMT == ".ims":
            file = Path(file)
            img = ImarisReader(file).get_image_data(channel = CHANNELS['neurites'], return_array = True)
        else:
            print("Invalid image format!")
        
        small = make_small(img)
        squished = make_squished(img)
        final = filter_empty_layers(squished)
        tifffile.imwrite(str(file).replace(INPUT_IMG_FMT, '_small.tif'), small, compression = 'lzw')
        tifffile.imwrite(str(file).replace(INPUT_IMG_FMT, '_squished.tif'), final, compression = 'lzw')

# %%

if __name__ == "__main__":
    main()

