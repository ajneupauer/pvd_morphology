from itertools import chain

import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
from scipy import interpolate

"""kernel for detecting endpoints in 2D skeletonized image"""
endpoint_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)

"""
Gives the most recently added endpoints of a dictionary of retracting branches.
In this dictionary, each value corresponds to a single branch (see trim_skeleton_to_endpoints).
    {0: [(y1, x1), (y2, x2), ..., (yi, xi)], 1: [(y1, x1), (y2, x2), ..., (yi, xi)], ...}
    *lists start from initial endpoints and move inwards
Output gives array[(yi, xi), (yi, xi), ...]
"""
def __get_last_coordinates(dict_endpoints: dict) -> np.ndarray:
    N = len(dict_endpoints.keys())
    return np.array([dict_endpoints[i][-1] for i in range(N)])

"""
Find endpoints in a skeleton via convolution.
Gives a 1D array for y and x coords where there is an endpoint:
    (array[y1, y2, ...], array[x1, x2, ...])
"""
def __find_endpoints(img: np.ndarray) -> tuple[np.ndarray]:
    # Convolve skeleton with endpoint_kernel
    endpt_response = convolve2d(
        img.astype(np.uint8), endpoint_kernel, mode="same"
    )
    # Endpoint has centerpoint (10) plus one neighbor (1)
    endpts = np.where(endpt_response == 11)
    return endpts

"""
function to 'trim' skeletonized binary mask:
Shorter 'branches' are eliminated, retaining only the longest
end-to-end skeleton. This is done by iteratively finding all
endpoints and eliminating them until two remain.
"""
def trim_skeleton_to_endpoints(skelimg: np.ndarray, n_ends=2) -> tuple[np.ndarray]:
    epts = __find_endpoints(skelimg)
    # Build dict to track retracting branches, where each value contains pts of a single branch
    # Pts listed from endpoints towards more proximal points
    # {0: [(y1, x1), (y2, x2), ...], 1: [(y1, x1), (y2, x2), ...], ...}
    # Initially, seed retracting branches with initial endpts, so we have
    # {0: [endpt], 1: [endpt], ...}
    dict_eps = {i: [pt] for i, pt in enumerate(list(zip(*epts)))}
    wrk = skelimg.copy()

    # if skeletonized image has two ends, then we are done
    if len(epts[0]) == n_ends:
        epts = tuple(zip(*epts))
        return skelimg, epts

    # otherwise, we prune to retain only the longest skeleton
    # !!! previously: elif len(epts) > n_ends:
    elif len(epts[0]) > n_ends:
        while len(epts[0]) > n_ends:
            # Erase current endpoints
            wrk[epts] = 0
            a1 = __get_last_coordinates(dict_eps)
            # Find new endpoints after erosion/retraction by 1 px
            epts = __find_endpoints(wrk)
            a2 = np.array(epts).T
            # Match new endpoints to previous endpoints
            pwdist = cdist(a1, a2)
            eid_ = pwdist.argmin(axis=0)
            # Append new endpoints to endpts dict to track retracted branches
            for id_, yx in zip(eid_, a2):
                dict_eps[id_].append((yx[0], yx[1]))

        # flatten the list of coordinates
        survived_ends = list(chain(*[dict_eps[i] for i in eid_]))
        survived_ends_id = tuple(i for i in np.array(survived_ends).T)

        # re-fill erased skeleton pixels
        wrk[survived_ends_id] = 1

        survived_epts = tuple([dict_eps[i][0] for i in eid_])
        return wrk, survived_epts

    else:
        return None, []

"""
routine to sort y,x coordinates of skeletonized edge:
The sorting starts from one endpoint and follows the single pixel neighbor
all the way to the other end.

Args:
    skeletonized_edge (2-d bool array): skeletonized edge image
    endpoint (2-tuple of y,x): endpoint coordinate

Returns:
    2-d array (N x 2), rc coordinate
"""
def sort_edge_coords(skeletonized_edge: np.ndarray, endpoint: tuple) -> np.ndarray:
    
    ydir = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    xdir = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    pos = np.array(endpoint)

    numel = skeletonized_edge.sum() # Number of 1 pts in skeleton
    wrkimg = skeletonized_edge.copy() # Save working copy of skeleton

    # preallocate output array
    sorted_edge = np.zeros((numel, 2), dtype=int)

    # Set first row in the output array to current position (supplied endpoint) 
    curpos = pos.copy()
    sorted_edge[0, :] = curpos
    
    # define pixel counter and start loop
    i = 0

    while True:
        i += 1
        wrkimg[curpos[0], curpos[1]] = 0 # Erase current pt to avoid revisiting
        # Get 3 x 3 neighborhood around current pt; stop loop if sums to 0 (other end reached)
        sbox = wrkimg[curpos[0] - 1 : curpos[0] + 2, curpos[1] - 1 : curpos[1] + 2]
        if sbox.sum() == 0:
            break
        # Take a step along the path
        curpos[0] += ydir[sbox][0] # y offset from current pt (returns values where sbox == True)
        curpos[1] += xdir[sbox][0] # x offset from current pt
        sorted_edge[i, :] = curpos # Add new point to output

    return sorted_edge

"""
Build the 3D sampling grid used to "straighten" a curved 3D volume along a precomputed centerline spline.
Spline comes from the .npy file generated by the GUI opened by generate_coordinates.py.
"""
# !!! Should redo arguments so scale is just chosen by the user, default = 4
def compute_resampling_coordinates(spline_params_file: str, Nz: int, override_scale = None) -> tuple:
    # Load data from the .npy file
    path_params = np.load(spline_params_file, allow_pickle=True).item()
    if override_scale is None:
        scale = path_params["bin_factor"]
    else:
        scale = override_scale
    spl = path_params["central_spline"]
    worm_length = path_params["downscaled_worm_length"]

    # form array of spline coordinates scaled up to the target resolution
    t = np.linspace(0, 1, num=int(worm_length * scale))

    cx, cy = interpolate.splev(t, spl)
    cx *= scale
    cy *= scale

    curve2d = np.stack((cy, cx), axis=1)

    # compute (normalized) tangent vectors
    tangents = np.zeros_like(curve2d)
    tangents[:-1] = curve2d[1:] - curve2d[:-1]
    tangents[-1] = curve2d[-1] - curve2d[-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    # compute (normalized) normal vectors
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)

    # add z-coordinates by prepending 0 columns
    curve3d = np.pad(curve2d, ((0, 0), (1, 0)))
    # define local coordinate systems (T, N, B): tangent, normal, binormal
    tangents3d = np.pad(tangents, ((0, 0), (1, 0)))
    normals3d = np.pad(normals, ((0, 0), (1, 0)))

    worm_width = int(path_params["downscaled_worm_width"] * scale)
    worm_height = Nz

    # generate axial sampling planes
    # Extend normals to half the worm width in both directions
    u = np.linspace(-worm_width / 2, worm_width / 2, worm_width)
    v = np.arange(Nz) # from first to final plane
    U, V = np.meshgrid(u, v)

    # compute rectangular mesh (y, z, x)
    Z_grid = (
        curve3d[:, None, None, 0]
        + normals3d[:, None, None, 0] * U[None, :, :]
        + V[None, :, :]
    )
    Y_grid = (curve3d[:, None, None, 1] + normals3d[:, None, None, 1] * U[None, :, :])
    X_grid = (curve3d[:, None, None, 2] + normals3d[:, None, None, 2] * U[None, :, :])

    # Reshape our output coordinates to standard image conventions (z, y, x)
    Z_grid = np.transpose(Z_grid, axes=(1, 0, 2))
    Y_grid = np.transpose(Y_grid, axes=(1, 0, 2))
    X_grid = np.transpose(X_grid, axes=(1, 0, 2))

    return Z_grid, Y_grid, X_grid
