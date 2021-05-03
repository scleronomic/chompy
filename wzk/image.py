import zlib  # Image compression
import numpy as np
from scipy.signal import convolve2d
from skimage.io import imread, imsave  # noqa: F401 unused import

from wzk.dicts_lists_tuples import tuple_extract
from wzk.numpy2 import align_shapes, get_cropping_indices, flatten_without_last, initialize_array


def combine_n_voxels_n_dim(n_voxels, n_dim=None):
    if np.size(n_voxels) == 1:
        try:
            n_voxels = tuple(n_voxels)
        except TypeError:
            n_voxels = (n_voxels,)
        n_voxels *= n_dim
    else:
        n_voxels = tuple(n_voxels)

    return n_voxels


def image_array_shape(*, n_voxels, n_samples=None, n_dim=None, n_channels=None):
    """
    Helper to set the shape for an image array.
    n_samples=100,  n_voxels=64,          n_dim=2,    n_channels=None  ->  (100, 64, 64)
    n_samples=100,  n_voxels=64,          n_dim=3,    n_channels=2     ->  (100, 64, 64, 64, 2)
    n_samples=None, n_voxel=(10, 11, 12), n_dim=None, n_channels=None  ->  (10, 11, 12)
    """

    shape = combine_n_voxels_n_dim(n_voxels=n_voxels, n_dim=n_dim)

    if n_samples is not None:
        shape = (n_samples,) + shape
    if n_channels is not None:
        shape = shape + (n_channels,)

    return shape


def initialize_image_array(*, n_voxels, n_dim=None, n_samples=None, n_channels=None,
                           dtype=bool, initialization='zeros'):
    shape = image_array_shape(n_voxels=n_voxels, n_dim=n_dim, n_samples=n_samples, n_channels=n_channels)
    return initialize_array(shape=shape, mode=initialization, dtype=dtype)


def reshape_img(img, n_dim=2, sample_dim=True, channel_dim=True,
                n_samples=None,  # either
                n_channels=None):  # or. infer the other

    n_voxels = img.shape[1]  # Can go wrong for 1D

    if n_samples is None and n_channels is None:
        n_channels = 1
        n_samples = 1

    elif n_samples is None:
        n_samples = img.size // (n_voxels ** n_dim) // n_channels
    elif n_channels is None:
        n_channels = img.size // (n_voxels ** n_dim) // n_samples

    if n_samples == 1 and not sample_dim:
        n_samples = None

    if n_channels == 1 and not channel_dim:
        n_channels = None

    return img.reshape(image_array_shape(n_voxels=n_voxels, n_dim=n_dim, n_samples=n_samples, n_channels=n_channels))


def concatenate_images(*imgs, axis=-1):
    """
    could add parameters which allow to stack horizontal, vertical, channel_wise, and sample wise
    """
    n_dim = np.max([i.ndim for i in imgs])
    if axis == +1:
        imgs = [i if i.ndim == n_dim else i[np.newaxis, ...] for i in imgs]
    elif axis == -1:
        imgs = [i if i.ndim == n_dim else i[..., np.newaxis] for i in imgs]
    else:
        raise NotImplementedError

    return np.concatenate(imgs, axis=axis)


def block_collage(*, img_arr, inner_border=None, outer_border=None, fill_boarder=0, dtype=float):

    assert img_arr.ndim == 4
    n_rows, n_cols, n_x, n_y = img_arr.shape

    bv_i, bh_i = tuple_extract(inner_border, default=(0, 0), mode='repeat')
    bv_o, bh_o = tuple_extract(outer_border, default=(0, 0), mode='repeat')

    img = np.full(shape=(n_x * n_rows + bv_i * (n_rows - 1) + 2*bv_o,
                         n_y * n_cols + bh_i * (n_cols - 1) + 2*bh_o), fill_value=fill_boarder, dtype=dtype)

    for r in range(n_rows):
        for c in range(n_cols):
            img[bv_o + r * (n_y + bv_i):bv_o + (r + 1) * (n_y + bv_i) - bv_i,
                bh_o + c * (n_x + bh_i):bh_o + (c + 1) * (n_x + bh_i) - bh_i] = img_arr[r, c]

    return img


def reduce_n_voxels(img, n_voxels, n_dim, n_channels, kernel, pooling_type='average', n_samples=None,
                    sample_dim=False, channel_dim=False):
    # TODO use scipy method
    # https://stackoverflow.com/questions/59988649/indexing-numpy-array-with-list-of-slices
    # n_voxels_new = 3
    # for o in range(n_voxels_new):
    #     for j in range(n_voxels_new):
    #         for k in range(n_voxels_new):
    #             print(o,j,k)
    #
    # import itertools

    # for ijk in itertools.product(range(n_voxels_new), repeat=n_dim):

    # if you keep using this method rewrite the difference between 2D and 3d cleaner with map
    # assert n_voxels % kernel == 0
    if pooling_type == 'average':
        pool = np.mean
        dtype = float
    else:  # == 'max'
        pool = np.max
        dtype = bool

    n_voxels_new = n_voxels // kernel
    img_new = initialize_image_array(n_voxels=n_voxels_new, n_dim=n_dim, n_channels=n_channels, n_samples=n_samples,
                                     dtype=dtype)
    img = reshape_img(img=img, n_dim=n_dim, n_channels=n_channels, channel_dim=True, n_samples=n_samples,
                      sample_dim=True)
    img_new = reshape_img(img=img_new, n_dim=n_dim, n_channels=n_channels, channel_dim=True, n_samples=n_samples,
                          sample_dim=True)

    if n_dim == 2:
        for i in range(n_voxels_new):
            for j in range(n_voxels_new):
                img_new[:, i, j, :] = pool(img[:,
                                           kernel * i: kernel * (i + 1),
                                           kernel * j: kernel * (j + 1), :], axis=(1, 2))
    else:  # n_dim == 3
        for i in range(n_voxels_new):
            for j in range(n_voxels_new):
                for k in range(n_voxels_new):
                    img_new[:, i, j, k, :] = pool(img[:,
                                                  kernel * i: kernel * (i + 1),
                                                  kernel * j: kernel * (j + 1),
                                                  kernel * k: kernel * (k + 1), :], axis=(1, 2, 3))

    return reshape_img(img=img_new, n_dim=n_dim, n_channels=n_channels, channel_dim=channel_dim, n_samples=n_samples,
                       sample_dim=sample_dim)


def pooling(mat, kernel, method='max', pad=False):
    """
    Non-overlapping pooling on 2D or 3D Measurements.
    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel shape in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has shape
           n//f, n being <mat> shape, f being kernel shape.
           if pad, output has shape ceil(n/f).

    Return <result>: pooled matrix.
    """

    print("# TODO write general function with reshaping -> much faster")
    m, n = mat.shape[:2]
    ky, kx = kernel

    def _ceil(x, y):
        return int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = (ny * ky, nx * kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        ny = m // ky
        nx = n // kx
        mat_pad = mat[:ny * ky, :nx * kx, ...]

    new_shape = (ny, ky, nx, kx) + mat.shape[2:]

    if method == 'max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))

    return result


def get_outer_edge_2d(img):
    edge_img = convolve2d(img, np.ones((3, 3)), mode='same')
    return np.logical_xor(edge_img, img)


def tile_2d(*, pattern, v_in_row, v_to_next_row, offset=(0, 0),
            shape):
    """

    Examples:

    # Point A
    pattern = np.ones((1, 1))
    shape = (11, 11)
    offset = (0, 0)
    v_in_row = 4
    v_to_next_row = (1, 2)

    # [[1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]]

    # Point B
    pattern = np.ones((1, 1))
    shape = (11, 11)
    offset = (0, 0)
    v_in_row = 5
    v_to_next_row = (1, 2)

    # [[1 0 0 0 0 1 0 0 0 0 1]
    #  [0 0 1 0 0 0 0 1 0 0 0]
    #  [0 0 0 0 1 0 0 0 0 1 0]
    #  [0 1 0 0 0 0 1 0 0 0 0]
    #  [0 0 0 1 0 0 0 0 1 0 0]
    #  [1 0 0 0 0 1 0 0 0 0 1]
    #  [0 0 1 0 0 0 0 1 0 0 0]
    #  [0 0 0 0 1 0 0 0 0 1 0]
    #  [0 1 0 0 0 0 1 0 0 0 0]
    #  [0 0 0 1 0 0 0 0 1 0 0]
    #  [1 0 0 0 0 1 0 0 0 0 1]]

    # Dumbbell
    pattern = np.ones((2, 2))
    pattern[0, 1] = 0
    pattern[1, 0] = 0
    shape = (16, 16)
    offset = (0, 0)
    v_in_row = 7
    v_to_next_row = (1, 3)

    # [[1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0]
    #  [0 1 0 1 0 0 0 0 1 0 1 0 0 0 0 1]
    #  [0 0 0 0 1 0 1 0 0 0 0 1 0 1 0 0]
    #  [1 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0]
    #  [0 0 0 1 0 1 0 0 0 0 1 0 1 0 0 0]
    #  [0 1 0 0 0 0 1 0 1 0 0 0 0 1 0 1]
    #  [0 0 1 0 1 0 0 0 0 1 0 1 0 0 0 0]
    #  [1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0]
    #  [0 1 0 1 0 0 0 0 1 0 1 0 0 0 0 1]
    #  [0 0 0 0 1 0 1 0 0 0 0 1 0 1 0 0]
    #  [1 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0]
    #  [0 0 0 1 0 1 0 0 0 0 1 0 1 0 0 0]
    #  [0 1 0 0 0 0 1 0 1 0 0 0 0 1 0 1]
    #  [0 0 1 0 1 0 0 0 0 1 0 1 0 0 0 0]
    #  [1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0]
    #  [0 1 0 1 0 0 0 0 1 0 1 0 0 0 0 1]]

    # Triangle
    pattern = np.ones((2, 2))
    pattern[0, 1] = 0
    shape = (10, 10)
    offset = (0, 0)
    v_in_row = 8
    v_to_next_row = (1, 3)

    # [[1 0 0 0 0 1 1 0 1 0]
    #  [1 1 0 1 0 0 0 0 1 1]
    #  [0 0 0 1 1 0 1 0 0 0]
    #  [0 1 0 0 0 0 1 1 0 1]
    #  [0 1 1 0 1 0 0 0 0 1]
    #  [0 0 0 0 1 1 0 1 0 0]
    #  [1 0 1 0 0 0 0 1 1 0]
    #  [0 0 1 1 0 1 0 0 0 0]
    #  [1 0 0 0 0 1 1 0 1 0]
    #  [1 1 0 1 0 0 0 0 1 1]]

    # Cross
    pattern = np.zeros((3, 3))
    pattern[1, :] = 1
    pattern[:, 1] = 1
    shape = (13, 13)
    offset = (1, 1)
    v_in_row = 4
    v_to_next_row = (3, 2)

    # [[1 1 0 1 1 1 0 1 1 1 0 1 1]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [0 1 1 1 0 1 1 1 0 1 1 1 0]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [1 1 0 1 1 1 0 1 1 1 0 1 1]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [0 1 1 1 0 1 1 1 0 1 1 1 0]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [1 1 0 1 1 1 0 1 1 1 0 1 1]]
    """
    nodes = np.zeros((shape[0]+v_to_next_row[0], shape[1]+v_in_row))

    for ii, i in enumerate(range(0, nodes.shape[0], v_to_next_row[0])):
        nodes[i, range((ii*v_to_next_row[1]) % v_in_row, nodes.shape[1], v_in_row)] = 1

    img = convolve2d(nodes, pattern, mode='full')

    ll = (v_to_next_row[0] + offset[0],
          v_to_next_row[1] + offset[1])

    return img[ll[0]:ll[0]+shape[0],
               ll[1]:ll[1]+shape[1]]


def check_overlap(a, b, return_arr=False):
    """
    Boolean indicating if the two arrays have an overlap.
    Sum over the axis that do not match.
    """

    a, b = (a, b) if a.n_dim > b.n_dim else (b, a)

    aligned_shape = align_shapes(a=a, b=b)
    summation_axis = np.arange(a.n_dim)[aligned_shape == -1]
    a = a.sum(axis=summation_axis)

    if return_arr:
        return np.logical_and(a, b)
    else:
        return np.logical_and(a, b).any()


def safe_add_small2big(idx, small_img, big_img, mode='center'):
    """
    Insert a small picture into the complete picture at the position 'idx'
    Assumption: all dimension of the small_img are odd, and idx indicates the center of the image,
    if this is not the case, there are zeros added at the end of each dimension to make the image shape odd
    Not both 'big_img' and 'n_voxels' can be None. One is needed to calculate the other.
    """

    idx = flatten_without_last(idx)
    n_samples, n_dim = idx.shape

    ll_big, ur_big, ll_small, ur_small = get_cropping_indices(pos=idx, mode=mode,
                                                              shape_small=small_img.shape[-n_dim:],
                                                              shape_big=big_img.shape)

    if small_img.ndim > n_dim:
        for ll_b, ur_b, ll_s, ur_s, img_s in zip(ll_big, ur_big, ll_small, ur_small, small_img):
            big_img[tuple(map(slice, ll_b, ur_b))] += img_s[tuple(map(slice, ll_s, ur_s))]
    else:
        for ll_b, ur_b, ll_s, ur_s in zip(ll_big, ur_big, ll_small, ur_small):
            big_img[tuple(map(slice, ll_b, ur_b))] += small_img[tuple(map(slice, ll_s, ur_s))]


# Image Compression <-> Decompression
def img2compressed(*, img, n_dim=-1, level=9):
    """
    Compress the given image with the zlib routine to a binary string.
    Level of compression can be adjusted. A timing with respect to different compression levels for decompression showed
    no difference, so the highest level is default, this corresponds to the largest compression.
    For compression it is slightly slower but this happens just once and not during keras training, so the smaller
    needed memory was favoured.

    Alternative:
    <-> use numpy sparse for the world images, especially in 3d  -> zlib is more effective and more general
    """

    if n_dim == -1:
        return zlib.compress(img.tobytes(), level=level)
    else:
        shape = img.shape[:-n_dim]
        img_cmp = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape):
            img_cmp[idx] = zlib.compress(img[idx, ...].tobytes(), level=level)
        return img_cmp


def compressed2img(img_cmp, n_voxels, n_dim=None, n_channels=None, dtype=bool):
    """
    Decompress the binary string back to an image of given shape
    """

    shape = np.shape(img_cmp)

    if shape:
        n_samples = np.size(img_cmp)
        img_arr = initialize_image_array(n_voxels=n_voxels, n_dim=n_dim, n_samples=n_samples, n_channels=n_channels,
                                         dtype=dtype)
        for i in range(n_samples):
            img_arr[i, ...] = np.fromstring(zlib.decompress(img_cmp[i]), dtype=dtype).reshape(
                image_array_shape(n_voxels=n_voxels, n_dim=n_dim, n_channels=n_channels))
        return img_arr

    else:
        return np.fromstring(zlib.decompress(img_cmp), dtype=dtype).reshape(
            image_array_shape(n_voxels=n_voxels, n_dim=n_dim, n_channels=n_channels))


