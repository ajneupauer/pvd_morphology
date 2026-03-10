# %%
# Import modules

import os
os.chdir('{dir_where_repo_is_stored}/pvd_morphology/')
import sys
from pathlib import Path

SCRIPT_DIR = Path('{dir_where_repo_is_stored}/pvd_morphology/scripts/').resolve()
BASE_DIR = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR / 'modules'))

import numpy as np
import scipy.ndimage as ndi
import skimage as sk
import tifffile
from magicgui import magicgui
from napari.layers import Image, Labels, Shapes
from napari.types import ImageData, LabelsData, LayerDataTuple
from scipy import interpolate

import straightening_utils as u


# %%
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

    from skimage.measure import label, regionprops
    labeled = label(binary)
    num_components = labeled.max()

    if num_components == 0:
        mask = np.uint8(binary)
    elif num_components == 1:
        mask = np.uint8(binary)
    else:
        regions = regionprops(labeled)
        largest_region = max(regions, key=lambda r: r.area)
        binary = (labeled == largest_region.label)
        mask = np.uint8(binary)

    return mask


@magicgui(call_button="Extract center line")
def extract_center_line(label: LabelsData) -> LayerDataTuple:
    mask = np.array(label.data) > 0

    if np.sum(mask) == 0:
        raise ValueError("Mask is empty! Generate a mask first.")

    skeleton = sk.morphology.skeletonize(mask, method="lee")

    if np.sum(skeleton) == 0:
        raise ValueError("Skeletonization failed - mask may be too thin or small.")

    skeleton, epts = u.trim_skeleton_to_endpoints(skeleton)

    if skeleton is None:
        raise ValueError("Skeleton trimming failed. The mask shape may be too complex.")

    if len(epts) == 0:
        raise ValueError("No skeleton endpoints found. The mask may be too blob-like.")

    sorted_yx = u.sort_edge_coords(skeleton, epts[0])
    return (sorted_yx, {"name": "center line", "shape_type": "path"}, "Shapes")


small_files = []
current_index = 0
viewer = None
current_image_layer = None


def get_output_path_for_current_file():
    if current_index < len(small_files):
        current_file = small_files[current_index]
        base_name = current_file.stem.replace('_small', '')
        return current_file.parent / f"{base_name}.npy"
    return None


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
    nshapes = len(path_layer.data)
    opath = [
        path_layer.data[n]
        for n in range(nshapes)
        if path_layer.shape_type[n] == "path"
    ][0]
    image = np.array(image_layer.data)

    if spline_output is None or str(spline_output) == '' or str(spline_output) == '.':
        spline_output = get_output_path_for_current_file()

    diffs = np.diff(opath, axis=0)
    segment_lengths = np.sum(np.sqrt(diffs**2), axis=1)
    path_length = segment_lengths.sum()
    N = round(path_length)

    if flip_worm:
        opath = opath[::-1]

    x = opath[::subsample, 1]
    y = opath[::subsample, 0]

    spl, t_orig = interpolate.splprep([x, y], s=2)

    t_new = np.linspace(0, 1, N)

    xo, yo = interpolate.splev(t_orig, spl)
    xs, ys = interpolate.splev(t_new, spl)

    path = np.stack((ys, xs), axis=1)

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

    path_parameters = {
        "central_spline": spl,
        "bin_factor": scale,
        "downscaled_worm_length": N,
        "downscaled_worm_width": width,
    }

    if spline_output is not None:
        np.save(spline_output, path_parameters, allow_pickle=True)

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
    
    #small_files = sorted(BASE_DIR.glob("*/*_small.tif"))
    small_files = sorted(BASE_DIR.glob("images/*/*_small.tif"))

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

#    viewer.window.add_dock_widget(diffuse_image, name="load")
    viewer.window.add_dock_widget(threshold_image, name="make mask")
    viewer.window.add_dock_widget(extract_center_line, name="medial path")
    viewer.window.add_dock_widget(resample_along_path, name="straighten")

    viewer.window.add_dock_widget(previous_button, name="nav_prev", area="bottom")
    viewer.window.add_dock_widget(next_button, name="nav_next", area="bottom")

    load_image(current_index)

    napari.run()