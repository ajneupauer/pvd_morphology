
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

def _extract_center_line(mask: np.ndarray):
    """Skeletonize a binary mask (the user-drawn trace) and return sorted
    (y, x) centerline coordinates."""

    if np.sum(mask) == 0:
        raise ValueError("User trace is empty! Draw a line down the center of the worm first.")

    skeleton = sk.morphology.skeletonize(mask, method="lee") # Get center line

    if np.sum(skeleton) == 0:
        raise ValueError("Skeletonization failed - trace may be too thin or small.")

    skeleton, epts = u.trim_skeleton_to_endpoints(skeleton) # Trim center line to fit within mask

    if skeleton is None:
        raise ValueError("Skeleton trimming failed. The trace shape may be too complex.")

    if len(epts) == 0:
        raise ValueError("No skeleton endpoints found. The trace may be too blob-like.")

    return u.sort_edge_coords(skeleton, epts[0]) # Sort coordinates of center line to be in the correct order

"""Initialize parameters for when the program starts up."""
TRACE_LAYER_NAME = "user trace"
small_files = []
current_index = 0
viewer = None
current_image_layer = None
current_trace_layer = None

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
    trace_layer={"label": "trace"},
    width={"min": 10, "max": 500, "step": 5}
)
def resample_along_path(
    image_layer: Image,
    trace_layer: Labels,
    spline_output: Path = None,
    width: int = 100,
    subsample: int = 10,
    spline_smooth: float = 4,
    scale: float = 8,
    flip_worm: bool = False,
) -> LayerDataTuple:
    mask = np.array(trace_layer.data) > 0
    opath = _extract_center_line(mask)

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

"""Load the image of a specified index."""
def load_image(index):
    global viewer, current_image_layer, current_trace_layer, current_index

    if index < 0 or index >= len(small_files): # Index canot be negative or exceed # of images
        return

    current_index = index # Update current index to index selected
    path = small_files[current_index]

    try:
        image = tifffile.imread(path)
        
        # Remove all existing layers associated with previous image
        layers_to_remove = [layer for layer in viewer.layers]
        for layer in layers_to_remove:
            viewer.layers.remove(layer)

        # Based on the image file name, make a short name to display in the GUI
        short_name = path.stem
        if '_small' in short_name:
            short_name = short_name.replace('_small', '')
        parts = short_name.split('_')
        if len(parts) >= 3:
            short_name = '_'.join(parts[-3:])
            
        # Add image layer to the GUI and display the short name
        current_image_layer = viewer.add_image(
            image,
            colormap="gray",
            contrast_limits=(100, 1500),
            gamma=0.25,
            name=short_name
        )

        # Blank labels layer for the user to draw a line down the center
        # of the worm. The straighten widget skeletonizes this trace to
        # extract the centerline automatically.
        blank_trace = np.zeros(image.shape, dtype=np.uint8)
        current_trace_layer = viewer.add_labels(
            blank_trace,
            name=TRACE_LAYER_NAME
        )
        
        # Top banner of GUI window gives path to loaded image and progress through entire image list
        viewer.title = f"Napari - [{current_index + 1}/{len(small_files)}] {path.parent.name}"

    except Exception as e:
        pass

"""
Define GUI buttons to move to the next or previous image.
They are simply calls to load_image() of the previous or next index.
"""
@magicgui(call_button="<< Previous Image")
def previous_button():
    load_image(current_index - 1)


@magicgui(call_button="Next Image >>")
def next_button():
    load_image(current_index + 1)


if __name__ == "__main__":
    import napari
    # Get list of small.tif files within the dataset dir specified in the CLI
    args = parse_args()
    FPATH = args.dataset_path
    small_files = sorted(FPATH.glob("*/*_small.tif"))

    if len(small_files) == 0: # Exit program if there are no images
        exit(1)

    # Give the user a list of images discovered and ask which image they would like to open
    print(f"Found {len(small_files)} files to process")
    for i, f in enumerate(small_files):
        print(f"  {i+1}. {f.parent.name}/{f.name}")

    print(f"Enter file number (1-{len(small_files)}) or press Enter for first: ", end="")
    user_input = input().strip()

    # Determine current_index based on user input
    if user_input == "": # Allows user to select first image by typing nothing
        current_index = 0
    else:
        try:
            current_index = int(user_input) - 1
            if current_index < 0 or current_index >= len(small_files): # Index canot be negative or exceed # of images
                current_index = 0 # Force index = 0 if out of bounds
        except ValueError: # Force index = 0 if there's an error
            current_index = 0

    # Set up napari viewer with its widgets and buttons and load the current image
    viewer = napari.Viewer()

    viewer.window.add_dock_widget(resample_along_path, name="straighten")

    viewer.window.add_dock_widget(previous_button, name="nav_prev", area="bottom")
    viewer.window.add_dock_widget(next_button, name="nav_next", area="bottom")

    load_image(current_index)

    napari.run()