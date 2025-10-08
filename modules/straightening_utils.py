from itertools import chain

import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
from scipy import interpolate

# kernel for detecting endpoints in 2D skeletonized image
endpoint_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)


# internal function used in trimming skeletonized image
def __get_last_coordinates(dict_endpoints):
    N = len(dict_endpoints.keys())
    return np.array([dict_endpoints[i][-1] for i in range(N)])


def __find_endpoints(img):
    endpt_response = convolve2d(
        img.astype(np.uint8), endpoint_kernel, mode="same"
    )
    endpts = np.where(endpt_response == 11)
    return endpts


def trim_skeleton_to_endpoints(skelimg, n_ends=2):
    """function to 'trim' skeletonized binary mask

    Shorter 'branches' are eliminated, retaining only the longest
    end-to-end skeleton. This is done by iteratively finding all
    endpoints and eliminating them until two remains.

    """
    epts = __find_endpoints(skelimg)
    dict_eps = {i: [pt] for i, pt in enumerate(list(zip(*epts)))}
    wrk = skelimg.copy()

    # if skeletonized image has two ends, then we are done
    if len(epts[0]) == n_ends:
        epts = tuple(zip(*epts))
        return skelimg, epts

    # otherwise, we prune to retain only the longest skeleton
    elif len(epts) > n_ends:
        while len(epts[0]) > n_ends:
            wrk[epts] = 0
            a1 = __get_last_coordinates(dict_eps)
            epts = __find_endpoints(wrk)
            a2 = np.array(epts).T
            pwdist = cdist(a1, a2)
            eid_ = pwdist.argmin(axis=0)
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


def sort_edge_coords(skeletonized_edge, endpoint):
    """routine to sort y,x coordinates of skeletonized edge

    The sorting starts from one endpoint and follows the single pixel neighbor
    all the way to the other end.

    Args:
        skeletonized_edge (2-d bool array): skeletonized edge image
        endpoint (2-tuple of y,x): endpoint coordinate

    Returns:
        2-d array (N x 2), rc coordinate

    """

    ydir = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    xdir = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    pos = np.array(endpoint)

    numel = skeletonized_edge.sum()

    wrkimg = skeletonized_edge.copy()

    # preallocate output array
    sorted_edge = np.zeros((numel, 2), dtype=int)

    curpos = pos.copy()
    sorted_edge[0, :] = curpos
    # define pixel counter and start loop
    i = 0

    while True:
        i += 1
        wrkimg[curpos[0], curpos[1]] = 0
        sbox = wrkimg[
            curpos[0] - 1 : curpos[0] + 2, curpos[1] - 1 : curpos[1] + 2
        ]
        if sbox.sum() == 0:
            break
        # move current position
        curpos[0] += ydir[sbox][0]
        curpos[1] += xdir[sbox][0]
        sorted_edge[i, :] = curpos

    return sorted_edge


def compute_resampling_coordinates(spline_params_file, Nz, override_scale = None):
    path_params = np.load(spline_params_file, allow_pickle=True).item()
    if override_scale is None:
        scale = path_params["bin_factor"]
    else:
        scale = override_scale
    spl = path_params["central_spline"]
    worm_length = path_params["downscaled_worm_length"]

    t = np.linspace(0, 1, num=int(worm_length * scale))

    cx, cy = interpolate.splev(t, spl)
    cx *= scale
    cy *= scale

    # form array of coordinate for convenience
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
    u = np.linspace(-worm_width / 2, worm_width / 2, worm_width)
    v = np.arange(Nz)
    U, V = np.meshgrid(u, v)

    # compute rectangular mesh (worm_length, Nz, worm_width)
    Z_grid = (
        curve3d[:, None, None, 0]
        + normals3d[:, None, None, 0] * U[None, :, :]
        + V[None, :, :]
    )

    Y_grid = (
        curve3d[:, None, None, 1] + normals3d[:, None, None, 1] * U[None, :, :]
    )

    X_grid = (
        curve3d[:, None, None, 2] + normals3d[:, None, None, 2] * U[None, :, :]
    )

    # for nicer reshaping, reshape to our output coordinate where we want
    # (Nz, worm_length, worm_width)
    Z_grid = np.transpose(Z_grid, axes=(1, 0, 2))
    Y_grid = np.transpose(Y_grid, axes=(1, 0, 2))
    X_grid = np.transpose(X_grid, axes=(1, 0, 2))

    return Z_grid, Y_grid, X_grid
