#!/usr/bin/env python3
# Import modules
import sys
import argparse
import json
import os
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import tifffile

os.chdir('/Users/alexneupauer/starr-luxton-lab/pvd-project/pvd_morphology/') # Set working dir to repo dir
import sys
sys.path.append('./modules') # Add module dir to sys for custom module import

from straightening_utils import compute_resampling_coordinates


# Get parameters from the config file
payload = json.loads(Path('./config.json').read_text(encoding="utf-8"))
HAS_MITO = payload.get("has_mito")
HAS_MITO = True if HAS_MITO == 1 else 0
CHANNELS = payload.get("channels")

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

"""Given the path to an unstraightened stack, make a straightened stack."""
def make_straightened(image_file: str, coords_file: str, scale=4) -> np.ndarray:
    data = tifffile.imread(image_file)
    Nz, Ny, Nx = data.shape
    
    # From the saved coordinates, determine coordinates when resampled at the specified scale
    zc, yc, xc = compute_resampling_coordinates(coords_file, Nz, override_scale=scale)

    # Based on resampled coordinates, make the straightened stack
    resampled = ndi.map_coordinates(
        data.astype(np.float32),
        (zc.ravel(), yc.ravel(), xc.ravel()),
        order=2,
    )
    resampled = resampled.reshape(zc.shape)

    return np.uint16(resampled)

"""Determine if an image subdir has already been processed."""
def is_already_processed(worm_folder: Path) -> bool:
    straightened_files = list(worm_folder.glob("*_Straightened.tif")) # Look for straightened images
    if HAS_MITO:
        mito_straightened_files = list(worm_folder.glob("*_mito_Straightened.tif")) # Look for straightened images
        if len(straightened_files) > 0 and len(mito_straightened_files) > 0: 
            return True
        else: # Folder has not been processed if there are no straightened images
            return False
    else: 
        if len(straightened_files) > 0:
            return True
        else: # Folder has not been processed if there are no straightened images
            False

"""Given an image subdir, get paths to input and output files."""
def find_matching_files(worm_folder: Path) -> tuple[Path]:
    npy_files = list(worm_folder.glob("*.npy"))
    if len(npy_files) == 0:
        return None

    coords_file = npy_files[0] # Coordinates for straightening
    base_name = coords_file.name.replace('.npy', '')
    squished_file = worm_folder / f"{base_name}_squished.tif" # Unstraightened neurite stack
    
    if not squished_file.exists(): # Need an unstraightened neurite stack to continue
        return None
    
    if HAS_MITO:
        mito_squished_file = worm_folder / f"{base_name}_mito_squished.tif" # Unstraightened mito stack
        mito_output_file = worm_folder / f"{base_name}_mito_Straightened.tif" # Straightened mito stack
        if not mito_squished_file.exists(): # Need an unstraightened mito stack to continue
            return None
    else: # If no mito channel, set mito file paths to None
        mito_squished_file = None
        mito_output_file = None

    output_file = worm_folder / f"{base_name}_Straightened.tif" # Straightened neurite stack
    
    return (coords_file, squished_file, mito_squished_file, output_file, mito_output_file)


def process_all_folders():
    # Extract parameters from command line
    args = parse_args()
    base_dir = args.dataset_path
    DRY = args.dry_run
    
    if DRY:
        print('Dry run selected. No files will be created.')
    
    base_dir = Path(base_dir)
    # Look for image subdirs by looking for small.tif files
    worm_folders = sorted(set(f.parent for f in base_dir.glob("*/*_small.tif")))

    if len(worm_folders) == 0: # Exit if no subdirs exist
        return

    # Track which image subdirs have been processed and which have not
    to_process = []
    already_done = []
    missing_files = []

    for folder in worm_folders:
        if is_already_processed(folder):
            already_done.append(folder)
        else: # If not yet processed, add folder to list and its associated files
            files = find_matching_files(folder)
            if files:
                to_process.append((folder, files))
            else:
                missing_files.append(folder)

    print(f"Found {len(worm_folders)} worm folders:")
    print(f"  {len(already_done)} already processed")
    print(f"  {len(to_process)} ready to process")
    print(f"  {len(missing_files)} missing files")

    if len(to_process) == 0: # Exit if no subdirs need to be procesed
        return

    # User must permit processing of unprocessed folders
    print(f"\nProcess {len(to_process)} folders? (y/n): ", end="")
    if input().strip().lower() != 'y':
        return

    for i, (folder, files) in enumerate(to_process):
        # For each image subdir, get its associated file paths
        coords_file, squished_file, mito_squished_file, output_file, mito_output_file = files
        print(f"[{i+1}/{len(to_process)}] {folder.name}")

        if DRY: # Dry run just prints what would happen and skips the rest of the loop
            print(f"Made {squished_file} from {coords_file}.")
            if HAS_MITO:
                print(f'Made {mito_squished_file} from {coords_file}.')
            continue

        try: # Make straightened images
            resampled = make_straightened(str(squished_file), str(coords_file), scale=4)
            tifffile.imwrite(str(output_file), resampled, compression='lzw')
            print(f"  Done: {resampled.shape}")
            
            if HAS_MITO:
                resampled = make_straightened(str(mito_squished_file), str(coords_file), scale=8)
                tifffile.imwrite(str(mito_output_file), resampled, compression='lzw')
                print(f"  Done: {resampled.shape}")
        except Exception as e:
            print(f"  Error: {e}")

# %%

if __name__ == "__main__":
    process_all_folders() 