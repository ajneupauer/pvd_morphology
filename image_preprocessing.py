#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:44:49 2024

@author: alexneupauer
"""

# Import modules
import shutil
import argparse
import json
import os
import tifffile
from pathlib import Path
import numpy as np

os.chdir('/Users/alexneupauer/starr-luxton-lab/pvd-project/pvd_morphology/') # Set working dir to repo dir
import sys
sys.path.append('./modules') # Add module dir to sys for custom module import
from ims import ImarisReader


# Get parameters from the config file
payload = json.loads(Path('./config.json').read_text(encoding="utf-8"))
HAS_MITO = payload.get("has_mito")
HAS_MITO = True if HAS_MITO == 1 else 0
CHANNELS = payload.get("channels")
INPUT_IMG_FMT = payload.get("input_img_fmt")

# %%

"""
Collect user arguments passed to the command line.
dataset_path: directory to dataset of raw images
--dry-run: add if performing a dry run (won't make/modify anything')
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform image preprocessing in preparation for image straightening."
    )
    parser.add_argument("dataset_path", type=Path, help="Directory to dataset of raw images.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

"""Based on the images in a directory with the requisite file extension, build a subdirectory for each image."""
def build_output_dirs(folder: Path, dry: bool, img_ext = ".ims") -> list[Path]:
    files = []
    img_paths = []
    
    # Look for all files with the specified extension, sort the list
    for file in folder.glob('*' + img_ext):
        if '._' in file.stem:
            continue
        files.append(file)
    files.sort()

    for file in files:
        # Make subdir name based on the image name
        outdir_name = str(folder) + '/' + file.stem
        img_paths.append(outdir_name + '/' + file.name)
        if dry: # Don't make dirs or move files, but print out what would happen if dry-run wasn't applied
            print(f"Made dir: {outdir_name}.")
            print(f"Moved {file.name} to {outdir_name}.\n")
            continue
        else: # Make the subdir and move the image into it
            outdir_name = Path(outdir_name)
            outdir_name.mkdir(exist_ok = True)
            if not file.exists(): # Skip if the file doesn't exist
                continue
            else:
                shutil.move(file, outdir_name) # Move image
    
    return img_paths


# Define preprocessing functions

"""From a 3D image array, make an 8x downsampled maximum intensity projection (MIP)."""
def make_small(in_img: np.ndarray) -> np.ndarray:
    dsImg = in_img[:, ::8, ::8] # Downsample by extracting every 8th pixel in x and y
    maxProj = dsImg.max(axis = 0) # Make a MIP
    return np.uint16(maxProj)

"""
From a 3D image array, 'squish' the stack by a factor of 2 by taking MIPs of adjacent slices z = 1,2; 3,4; 5,6; ...
Also downsample by a factor of 2 in x and y.
"""
def make_squished(in_img: np.ndarray, downsample = True) -> np.ndarray:
    if downsample: # Default is to downsample
        in_img = in_img[:, ::2, ::2] # Downsample by extracting every other pixel in x and y
    depth, height, width = in_img.shape
    out_img = np.empty([depth // 2, height, width]) # Initialize empty array for output with depth z/2
    for i in range(depth // 2):
        out_img[i] = in_img[2*i:2*(i+1)].max(axis = 0) # Each output slice is a MIP of 2 adjacent slices in the input
    return np.uint16(out_img)    

"""Find any empty image layers and remove them."""
def filter_empty_layers(image: np.ndarray) -> np.ndarray:
    depth = image.shape[0]
    layers_to_keep = []
    
    for z in range(depth):
        if np.mean(image[z]) >= 80: # Keep layer if avg intensity is over 80
            layers_to_keep.append(z) 
        else:
            continue
    
    edited = image[layers_to_keep, :, :] # Make new image with only layers to keep
    return edited


def main():
    # Extract parameters from command line
    args = parse_args()
    FPATH = args.dataset_path
    DRY = args.dry_run
    
    # Make image subdirectories and save their corresponding paths
    img_paths = build_output_dirs(FPATH, DRY, INPUT_IMG_FMT)
    
    for file in img_paths:    
        if DRY: # Print out files and folders that would be generated
            if HAS_MITO: # Process mito channel if there is one
                if INPUT_IMG_FMT != ".ims" and INPUT_IMG_FMT != ".tif":
                    print("Invalid image format! Must be .ims or .tif.")
                mito_save_msg = str(file).replace(INPUT_IMG_FMT, '_mito_squished.tif')
                print(f"Saved {mito_save_msg}.")
            if INPUT_IMG_FMT != ".ims" and INPUT_IMG_FMT != ".tif":
                print("Invalid image format! Must be .ims or .tif.")
            small_save_msg = str(file).replace(INPUT_IMG_FMT, '_small.tif')
            squished_save_msg = str(file).replace(INPUT_IMG_FMT, '_squished.tif')
            print(f"Saved {small_save_msg}.")
            print(f"Saved {squished_save_msg}.\n")
            continue # Skip rest of the for loop during a dry run
        
        if HAS_MITO: # Process mito channel if there is one
            if INPUT_IMG_FMT == ".tif": # Can read .tif with tifffile
                img = tifffile.imread(file)[CHANNELS['mito']] # Correct channel as specified in the config
            elif INPUT_IMG_FMT == ".ims": # .ims requires custom ims module
                file = Path(file)
                img = ImarisReader(file).get_image_data(channel = CHANNELS['mito'], return_array = True)
            else:
                print("Invalid image format! Must be .ims or .tif.")
            
            # Squish image, remove blank layers, and save with compression
            squished = make_squished(img, downsample = False) # Mito channel is NOT downsampled in x, y
            final = filter_empty_layers(squished)
            tifffile.imwrite(str(file).replace(INPUT_IMG_FMT, '_mito_squished.tif'), final, compression = 'lzw')
            
        if INPUT_IMG_FMT == ".tif": # Can read .tif with tifffile
            img = tifffile.imread(file)[CHANNELS['neurites']] # Correct channel as specified in the config
        elif INPUT_IMG_FMT == ".ims": # .ims requires custom ims module 
            file = Path(file)
            img = ImarisReader(file).get_image_data(channel = CHANNELS['neurites'], return_array = True)
        else:
            print("Invalid image format! Must be .ims or .tif.")
        
        # Make small 8x downsampled MIP and save with compression
        small = make_small(img)
        tifffile.imwrite(str(file).replace(INPUT_IMG_FMT, '_small.tif'), small, compression = 'lzw')
        # Squish and 2x downsample image, remove blank layers, and save with compression
        squished = make_squished(img)
        final = filter_empty_layers(squished)
        tifffile.imwrite(str(file).replace(INPUT_IMG_FMT, '_squished.tif'), final, compression = 'lzw')

# %%

if __name__ == "__main__":
    main()

