
# Import modules
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import skimage as sk
import tifffile
from magicgui import magicgui
from napari.layers import Image, Labels, Shapes
from napari.types import ImageData, LabelsData, LayerDataTuple
from scipy import interpolate

os.chdir('/Users/alexneupauer/starr-luxton-lab/pvd-project/pvd_morphology/') # Set working dir to repo dir
sys.path.append('./modules') # Add module dir to sys for custom module import

import straightening_utils as u

# %%

"""
Collect user arguments passed to the command line:
    dataset_path: directory to dataset of raw images
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform image preprocessing in preparation for image straightening."
    )
    parser.add_argument("dataset_path", type=Path, help="Directory to dataset of raw images.")
    return parser.parse_args()

"""
Define the GUI panel for image masking.
The user can set options for thresholding including:
    Threshold method
    Whether or not to fill mask holes
    Blur radius
    Remove mask objects below a specified size in px
    Set open/close morphology operations
    Set a manual intensity value for thresholding
"""
@magicgui(
    call_button="generate mask",
    image={"label": "image"},
    method={"choices": ["triangle", "otsu", "yen", "li", "isodata"]},
    fill_holes={"widget_type": "CheckBox"},
    sigma={"min": 0, "max": 10, "step": 0.5},
    min_size={"min": 0, "max": 1000, "step": 50},
    morph_open={"min": 0, "max": 10, "step": 1},
    morph_close={"min": 0, "max": 10, "step": 1},
    use_manual_threshold={"widget_type": "CheckBox"},
    manual_threshold={"min": 0, "max": 65535, "step": 1}
)
# Function to make mask based on user selected options in the GUI
def threshold_image(
    image: Image,
    method: str = "otsu",
    sigma: float = 10.0,
    fill_holes: bool = True,
    min_size: int = 500,
    morph_open: int = 0,
    morph_close: int = 10,
    use_manual_threshold: bool = False,
    manual_threshold: int = 100
) -> LabelsData:
    image_data = np.array(image.data)
    
    # Make a binary mask based on user input in GUI
    if sigma > 0:
        image_data = ndi.gaussian_filter(image_data, sigma)

    if use_manual_threshold:
        thres = manual_threshold
    elif method == "triangle":
        thres = sk.filters.threshold_triangle(image_data)
    elif method == "otsu":
        thres = sk.filters.threshold_otsu(image_data)
    elif method == "yen":
        thres = sk.filters.threshold_yen(image_data)
    elif method == "li":
        thres = sk.filters.threshold_li(image_data)
    elif method == "isodata":
        thres = sk.filters.threshold_isodata(image_data)

    binary = image_data >= thres

    if fill_holes:
        binary = ndi.binary_fill_holes(binary)

    if min_size > 0:
        binary = sk.morphology.remove_small_objects(binary, min_size=min_size)

    if morph_close > 0:
        binary = sk.morphology.binary_closing(binary, sk.morphology.disk(morph_close))

    if morph_open > 0:
        binary = sk.morphology.binary_opening(binary, sk.morphology.disk(morph_open))

    # Ensure mask consists of just one contiguous region
    from skimage.measure import label, regionprops
    labeled = label(binary)
    num_components = labeled.max()

    if num_components == 0:
        mask = np.uint8(binary) 
    elif num_components == 1: 
        mask = np.uint8(binary)
    else: # If there is more than one region, keep only the largest one
        regions = regionprops(labeled)
        largest_region = max(regions, key=lambda r: r.area)
        binary = (labeled == largest_region.label)
        mask = np.uint8(binary)

    return mask

"""
Define the GUI panel for center line extraction.
The user pushes a button to trigger center line extraction.
"""
@magicgui(call_button="Extract center line")
def extract_center_line(label: LabelsData) -> LayerDataTuple:
    mask = np.array(label.data) > 0

    if np.sum(mask) == 0:
        raise ValueError("Mask is empty! Generate a mask first.")

    skeleton = sk.morphology.skeletonize(mask, method="lee") # Get center line

    if np.sum(skeleton) == 0:
        raise ValueError("Skeletonization failed - mask may be too thin or small.")

    skeleton, epts = u.trim_skeleton_to_endpoints(skeleton) # Trim center line to fit within mask

    if skeleton is None:
        raise ValueError("Skeleton trimming failed. The mask shape may be too complex.")

    if len(epts) == 0:
        raise ValueError("No skeleton endpoints found. The mask may be too blob-like.")

    sorted_yx = u.sort_edge_coords(skeleton, epts[0]) # Sort coordinates of center line to be in the correct order
    return (sorted_yx, {"name": "center line", "shape_type": "path"}, "Shapes")

"""Initialize parameters for when the program starts up."""
small_files = []
current_index = 0
viewer = None
current_image_layer = None

"""Based on the current index, return the path for the output spline."""
def get_output_path_for_current_file():
    if current_index < len(small_files): # Proceed only if index doesn't exceed # of images
        current_file = small_files[current_index]
        # Generate name for output spline path based on the small.tif image
        base_name = current_file.stem.replace('_small', '')
        return current_file.parent / f"{base_name}.npy"
    return None

"""
Define the GUI panel for generating a straightened preview.
The user can set options for straightening including:
    Width of the straightened preview
    Whether or not to 'flip' the preview about the anterior-posterior axis
The spline coordinates are adjusted based on user options and then saved upon pressing 'straighten'.
"""
@magicgui(
    call_button="straighten",
    image_layer={"label": "img"},
    path_layer={"label": "path"},
    width={"min": 10, "max": 500, "step": 5}
)
def resample_along_path(
    image_layer: Image,
    path_layer: Shapes,
    spline_output: Path = None,
    width: int = 100,
    subsample: int = 10,
    spline_smooth: float = 4,
    scale: float = 8,
    flip_worm: bool = False,
) -> LayerDataTuple:
    # Make a list of points from the center line layer
    nshapes = len(path_layer.data)
    opath = [
        path_layer.data[n]
        for n in range(nshapes)
        if path_layer.shape_type[n] == "path"
    ][0]
    
    image = np.array(image_layer.data)
    
    # Set path for saving the spline data
    if spline_output is None or str(spline_output) == '' or str(spline_output) == '.':
        spline_output = get_output_path_for_current_file()
    
    # Get length of the center line
    diffs = np.diff(opath, axis=0)
    segment_lengths = np.sum(np.sqrt(diffs**2), axis=1)
    path_length = segment_lengths.sum()
    N = round(path_length)

    # Flip the image along A-P axis by reversing the order of points in the center line
    if flip_worm:
        opath = opath[::-1]
    
    # Get interpolated spline based on the center line
    x = opath[::subsample, 1]
    y = opath[::subsample, 0]

    spl, t_orig = interpolate.splprep([x, y], s=2)

    t_new = np.linspace(0, 1, N)

    xo, yo = interpolate.splev(t_orig, spl)
    xs, ys = interpolate.splev(t_new, spl)

    path = np.stack((ys, xs), axis=1)

    # Get normal (perpendicular) lines along the spline
    r = float(width / 2)
    M = int(2 * r)
    dvals = np.linspace(-r, r, num=width)

    tangents = np.zeros_like(path)
    tangents[:-1] = path[1:] - path[:-1]
    tangents[-1] = path[-1] - path[-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, -1]
    normals[:, 1] = tangents[:, 0]

    coords = path[:, None, :] + dvals[None, :, None] * normals[:, None, :]
    coords = coords.reshape(-1, 2)

    resampled_intensities = ndi.map_coordinates(
        image, coords.T, order=2, mode="constant", prefilter=False
    )

    resampled_intensities = resampled_intensities.reshape(N, M).T

    # Save spline data, including the spline coords and other params
    path_parameters = {
        "central_spline": spl,
        "bin_factor": scale,
        "downscaled_worm_length": N,
        "downscaled_worm_width": width,
    }

    if spline_output is not None:
        np.save(spline_output, path_parameters, allow_pickle=True)
    
    # Generate layers in the viewer for the straightened preview and the spline path
    subsampled_path = np.stack((yo, xo), axis=1)

    return [
        (
            resampled_intensities,
            {
                "name": "straightened",
                "colormap": "gray",
                "gamma": 0.25,
                "contrast_limits": (100, 250),
            },
            "Image",
        ),
        (
            subsampled_path,
            {"name": "subsampled path", "shape_type": "path"},
            "Shapes",
        ),
    ]


def load_image(index):
    global viewer, current_image_layer, current_index

    if index < 0 or index >= len(small_files):
        return

    current_index = index
    path = small_files[current_index]

    try:
        image = tifffile.imread(path)

        layers_to_remove = [layer for layer in viewer.layers]
        for layer in layers_to_remove:
            viewer.layers.remove(layer)

        short_name = path.stem
        if '_small' in short_name:
            short_name = short_name.replace('_small', '')
        parts = short_name.split('_')
        if len(parts) >= 3:
            short_name = '_'.join(parts[-3:])

        current_image_layer = viewer.add_image(
            image,
            colormap="gray",
            contrast_limits=(100, 1500),
            gamma=0.25,
            name=short_name
        )

        viewer.title = f"Napari - [{current_index + 1}/{len(small_files)}] {path.parent.name}"

    except Exception as e:
        pass


@magicgui(call_button="<< Previous Image")
def previous_button():
    load_image(current_index - 1)


@magicgui(call_button="Next Image >>")
def next_button():
    load_image(current_index + 1)


if __name__ == "__main__":
    import napari
    args = parse_args()
    FPATH = args.dataset_path
    
    small_files = sorted(FPATH.glob("*/*_small.tif"))

    if len(small_files) == 0:
        exit(1)

    print(f"Found {len(small_files)} files to process")
    for i, f in enumerate(small_files):
        print(f"  {i+1}. {f.parent.name}/{f.name}")

    print(f"Enter file number (1-{len(small_files)}) or press Enter for first: ", end="")
    user_input = input().strip()

    if user_input == "":
        current_index = 0
    else:
        try:
            current_index = int(user_input) - 1
            if current_index < 0 or current_index >= len(small_files):
                current_index = 0
        except ValueError:
            current_index = 0

    viewer = napari.Viewer()

#   viewer.window.add_dock_widget(diffuse_image, name="load")
    viewer.window.add_dock_widget(threshold_image, name="make mask")
    viewer.window.add_dock_widget(extract_center_line, name="medial path")
    viewer.window.add_dock_widget(resample_along_path, name="straighten")

    viewer.window.add_dock_widget(previous_button, name="nav_prev", area="bottom")
    viewer.window.add_dock_widget(next_button, name="nav_next", area="bottom")

    load_image(current_index)

    napari.run()