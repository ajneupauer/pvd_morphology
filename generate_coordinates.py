# %%
# Import modules

import os
os.chdir('{dir_where_repo_is_stored}/pvd_morphology/')
import sys
sys.path.append('./modules')

from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import skimage as sk
import tifffile
from magicgui import magicgui
from napari.layers import Image, Labels, Shapes
from napari.types import ImageData, LabelsData, LayerDataTuple
from scipy import interpolate
from scipy.optimize import minimize

import straightening_utils as u


# %%
@magicgui(image={"label": "input image"})
def diffuse_image(image: Image, sigma: float = 10.0) -> LayerDataTuple:
    data = image.data
    blurred = ndi.gaussian_filter(data, sigma)
    return (
        blurred,
        {
            "name": "blurred",
            "contrast_limits": (100, 200),
            "gamma": 0.25,
            "colormap": "plasma",
        },
        "image",
    )


@magicgui(path={"label": "input path"})
def smooth_path(
    path: Shapes, npts: int = 250, s: float = 2, width: int = 100
) -> LayerDataTuple:
    # get the first 'path'
    nshapes = len(path.data)
    input_path = [
        path.data[n] for n in range(nshapes) if path.shape_type[n] == "path"
    ][0]

    x = input_path[:, 1]
    y = input_path[:, 0]

    tck, u = interpolate.splprep([x, y], s=s)
    t_new = np.linspace(0, 1, npts)
    xs, ys = interpolate.splev(t_new, tck)

    output_path = np.stack((ys, xs), axis=1)

    return (
        output_path,
        {"shape_type": "path", "name": "smoothed", "edge_width": width},
        "Shapes",
    )


@magicgui(call_button="generate mask")
def threshold_image(image: ImageData) -> LabelsData:
    image_data = np.array(image.data)
    thres = sk.filters.threshold_triangle(image_data)
    mask = np.uint8(ndi.binary_fill_holes(image_data >= thres))
    return mask


@magicgui(call_button="Extract center line")
def extract_center_line(label: LabelsData) -> LayerDataTuple:
    mask = np.array(label.data) > 0
    skeleton = sk.morphology.skeletonize(mask, method="lee")
    skeleton, epts = u.trim_skeleton_to_endpoints(skeleton)
    sorted_yx = u.sort_edge_coords(skeleton, epts[0])
    return (sorted_yx, {"name": "center line", "shape_type": "path"}, "Shapes")


@magicgui(
    call_button="straighten",
    spline_output={"widget_type": "FileEdit", "mode": "w"},
)
def resample_along_path(
    image_layer: ImageData,
    path_layer: Shapes,
    spline_output: Path,
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
    # calculate length of path in pixels
    diffs = np.diff(opath, axis=0)
    segment_lengths = np.sum(np.sqrt(diffs**2), axis=1)
    path_length = segment_lengths.sum()
    N = round(path_length)

    if flip_worm:
        opath = opath[::-1]

    # interpolate paths using splines
    x = opath[::subsample, 1]
    y = opath[::subsample, 0]

    # generate a cubic-spline representation with smoothing = 2
    spl, t_orig = interpolate.splprep([x, y], s=2)

    # resample this to higher-res
    t_new = np.linspace(0, 1, N)

    xo, yo = interpolate.splev(t_orig, spl)  # smoothed (subsampled) spline representation

    xs, ys = interpolate.splev(t_new, spl)  # full-sampling across length

    path = np.stack((ys, xs), axis=1)

    r = float(width / 2)
    M = int(2 * r)
    dvals = np.linspace(-r, r, num=width)

    tangents = np.zeros_like(path)

    # forward finite difference to compute tangent vectors
    tangents[:-1] = path[1:] - path[:-1]
    tangents[-1] = path[-1] - path[-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    # compute normal vectors
    # normals are swapped axes and flipping sign (in 2D)
    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, -1]
    normals[:, 1] = tangents[:, 0]

    # compute 'slices' along normal vector for each central coordinate
    # use broadcasting:
    # (N x 1 x 2) + (1 x M x 1) * (N x 1 x 2)
    # 'multiplying' a normal vector to the tangent gives a slice [-r,+r]
    # across the central coordinate
    coords = path[:, None, :] + dvals[None, :, None] * normals[:, None, :]
    # coords is currently (N x M x 2)
    coords = coords.reshape(-1, 2)

    # `map_coordinates` wants input coordinates to be in row-major order
    # where the coordinates is shaped (2, N). 0 is row and 1 is columns.
    resampled_intensities = ndi.map_coordinates(
        image, coords.T, order=2, mode="constant", prefilter=False
    )

    # reshape intensities into 2D image but swap the axes to get the 'wide'
    # image rather than 'tall'. We want MxN image where normal vectors are
    # oriented along rows
    resampled_intensities = resampled_intensities.reshape(N, M).T

    path_parameters = {
        "central_spline": spl,
        "bin_factor": scale,
        "downscaled_worm_length": N,
        "downscaled_worm_width": width,
    }

    if spline_output is not None:
        np.save(spline_output, path_parameters, allow_pickle=True)

    # prepare subsampled central path to be displayed also
    # napari expects column-wise vectors, so stack along axis=1
    subsampled_path = np.stack((yo, xo), axis=1)

    return [
        (
            resampled_intensities,
            {
                "name": "straightened",
                "colormap": "plasma",
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


if __name__ == "__main__":
    import napari

    # This is where you put in the image you want to open
    path = Path("{path_to_8xds_maxProj}")
    image = tifffile.imread(path)

    viewer = napari.Viewer()
    viewer.window.add_dock_widget(diffuse_image, name="blur")
    viewer.window.add_dock_widget(threshold_image, name="make mask")
    viewer.window.add_dock_widget(extract_center_line, name="medial path")
    viewer.window.add_dock_widget(resample_along_path, name="straighten")

    imglayer = viewer.add_image(
        image, colormap="plasma", contrast_limits=(100, 1500), gamma=0.25
    )

    napari.run()
