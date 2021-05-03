import numpy as np
from scipy.sparse import csr_matrix
from itertools import product


class DummyArray:
    """Allows indexing but always returns the same 'dummy' value"""
    def __init__(self, arr, shape):
        self.arr = arr
        self.shape = shape

    def __getitem__(self, item):
        if isinstance(item, int):
            assert item in range(-self.shape[0], self.shape[0])
        else:
            assert len(item) == len(self.shape), f"Incompatible index {item} for array with shape {self.shape}"
            for i, s in zip(item, self.shape):
                assert i in range(-s, +s)

        return self.arr


def initialize_array(*, shape, mode='zeros', dtype=None, order='c'):

    if mode == 'zeros':
        return np.zeros(shape, dtype=dtype, order=order)
    elif mode == 'ones':
        return np.ones(shape, dtype=dtype, order=order)
    elif mode == 'empty':
        return np.empty(shape, dtype=dtype, order=order)
    elif mode == 'random':
        return np.random.random(shape).astype(dtype=dtype, order=order)
    else:
        raise ValueError(f"Unknown initialization method {mode}")


# Checks
def np_isinstance(o, c):
    """
    # Like isinstance if o is not np.ndarray
    np_isinstance(('this', 'that'), tuple)  # True
    np_isinstance(4.4, int)                 # False
    np_isinstance(4.4, float)               # True

    # else
    np_isinstance(np.ones(4, dtype=int), int)    # True
    np_isinstance(np.ones(4, dtype=int), float)  # False
    np_isinstance(np.full((4, 4), 'bert'), str)  # True
    """
    c2np = {bool: np.bool, int: np.integer, float: np.floating, str: np.str}

    if isinstance(o, np.ndarray):
        c = (c2np[cc] for cc in c) if isinstance(c, tuple) else c2np[c]
        return isinstance(o.flat[0], c)

    else:
        return isinstance(o, c)


# Wrapper for common numpy arguments
def axis_wrapper(axis, n_dim, sort=True, dtype='tuple'):

    if axis is None:
        axis = np.arange(n_dim)

    axis = np.atleast_1d(axis).astype(int)
    axis %= n_dim
    if sort:
        np.sort(axis)

    if dtype == 'tuple':
        return tuple(axis)
    elif dtype == 'np.ndarray':
        return axis
    else:
        raise ValueError(f"Unknown dtype {dtype}")


def shape_wrapper(shape=None):
    """
    Note the inconsistent usage of shape / shape as function arguments in numpy.
    https://stackoverflow.com/questions/44804965/numpy-size-vs-shape-in-function-arguments
    -> use shape
    """
    if shape is None:
        return ()
    elif isinstance(shape, int):
        return shape,
    elif isinstance(shape, tuple):
        return shape
    else:
        raise ValueError(f"Unknown 'shape': {shape}")


# object <-> numeric
def object2numeric_array(arr):
    return np.array([v for v in arr])


def numeric2object_array(arr):
    n = arr.shape[0]
    arr_obj = np.zeros(n, dtype=object)
    for i in range(n):
        arr_obj[i] = arr[i]

    return arr_obj


# scalar <-> matrix
def scalar2array(v, shape):
    if isinstance(shape, int):
        shape = (shape,)
    if np.shape(v) != shape:
        return np.full(shape, v)
    else:
        return v


def safe_unify(x):
    x = np.atleast_1d(x)

    x_mean = np.mean(x)
    assert np.all(x == x_mean)
    return x_mean


def safe_scalar2array(*val_or_arr, shape, squeeze=True):
    shape = shape_wrapper(shape)

    res = []
    for voa in val_or_arr:
        try:
            voa = np.asscalar(np.array(voa))
            res.append(np.full(shape=shape, fill_value=voa, dtype=type(voa)))
        except ValueError:
            assert np.shape(voa) == shape
            res.append(voa)

    if len(res) == 1 and squeeze:
        return res[0]
    else:
        return res


def flatten_without_last(x):
    return np.reshape(x, newshape=(-1, np.shape(x)[-1]))


def flatten_without_first(x):
    return np.reshape(x, newshape=(np.shape(x)[0], -1))


def args2arrays(*args):
    return [np.array(a) for a in args]


# Shapes
def get_complementary_axis(axis, n_dim):
    return tuple(set(range(n_dim)).difference(set(axis)))


def get_subshape(shape, axis):
    return tuple(np.array(shape)[np.array(axis)])


def align_shapes(a, b):
    # a = np.array((2, 3, 4, 3, 5, 1))
    # b = np.array((3, 4, 3))
    # -> array([-1, 1, 1, 1, -1, -1])

    idx = find_subarray(a=a, b=b)
    idx = np.asscalar(idx)

    aligned_shape = np.ones(len(a), dtype=int)
    aligned_shape[:idx] = -1
    aligned_shape[idx+len(b):] = -1
    return aligned_shape


def repeat2new_shape(img, new_shape):
    reps = np.ceil(np.array(new_shape) / np.array(img.shape)).astype(int)
    for i in range(img.ndim):
        img = np.repeat(img, repeats=reps[i], axis=i)

    img = img[tuple(map(slice, new_shape))]
    return img


def change_shape(arr, mode='even'):
    s = np.array(arr.shape)

    if mode == 'even':
        s_new = (s + s % 2)
    elif mode == 'odd':
        s_new = (s // 2) * 2 + 1
    else:
        raise ValueError(f"Unknown mode {mode}")

    arr_odd = np.zeros(s_new, dtype=arr.dtype)
    fill_with_air_left(arr=arr, out=arr_odd)
    return arr_odd


def fill_with_air_left(arr, out):
    assert arr.ndim == out.ndim
    out[tuple(map(slice, arr.shape))] = arr


def __argfun(a, axis, fun):

    axis = axis_wrapper(axis=axis, n_dim=a.ndim, sort=True)
    if len(axis) == 1:
        return fun(a, axis=axis)
    elif len(axis) == a.ndim:
        np.unravel_index(fun(a), shape=a.shape)
    else:
        axis2 = get_complementary_axis(axis=axis, n_dim=a.ndim)
        shape2 = get_subshape(shape=a.shape, axis=axis2)
        shape = get_subshape(shape=a.shape, axis=axis)

        a2 = np.transpose(a, axes=axis2 + axis).reshape(shape2 + (-1,))
        idx = fun(a2, axis=-1)
        idx = np.array(np.unravel_index(idx, shape=shape))

        return np.transpose(idx, axes=np.roll(np.arange(idx.ndim), -1))


def max_size(*args):
    return int(np.max([np.size(a) for a in args]))


def min_size(*args):
    return int(np.min([np.size(a) for a in args]))


def argmax(a, axis=None):
    return __argfun(a=a, axis=axis, fun=np.argmax)


def argmin(a, axis=None):
    return __argfun(a=a, axis=axis, fun=np.argmin)


def allclose(a, b, axis=None, **kwargs):
    assert a.shape == b.shape

    axis = axis_wrapper(axis=axis, n_dim=a.ndim, dtype='np.ndarray')
    shape = np.array(a.shape)[axis]
    bool_arr = np.zeros(shape, dtype=bool)

    for i in product(*(range(s) for s in shape)):
        bool_arr[i] = np.allclose(extract(a, idx=i, axis=axis),
                                  extract(b, idx=i, axis=axis),
                                  **kwargs)

    return bool_arr


def __fill_index_with(idx, axis, shape, mode='slice'):
    """
    orange <-> orth-range
    sorry but 'orange', 'slice' was just to delicious
    """
    axis = axis_wrapper(axis=axis, n_dim=len(shape))
    if mode == 'slice':
        idx_with_ = [slice(None) for _ in range(len(shape)-len(axis))]

    elif mode == 'orange':
        idx_with_ = np.ogrid[[range(s) for i, s in enumerate(shape) if i not in axis]]

    else:
        raise ValueError(f"Unknown mode {mode}")

    idx = np.array(idx)
    for i, ax in enumerate(axis):
        idx_with_.insert(ax, idx[..., i])

    return tuple(idx_with_)


def insert(a, val, idx, axis, mode='slice'):
    idx = __fill_index_with(idx=idx, axis=axis, shape=a.shape, mode=mode)
    a[idx] = val


def extract(a, idx, axis, mode='slice'):
    idx = __fill_index_with(idx=idx, axis=axis, shape=a.shape, mode=mode)
    return a[idx]


# Combine
def interleave(arrays, axis=0, out=None):
    """
    https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    """

    shape = list(np.asanyarray(arrays[0]).shape)
    if axis < 0:
        axis += len(shape)
    assert 0 <= axis < len(shape), "'axis' is out of bounds"
    if out is not None:
        out = out.reshape(shape[:axis+1] + [len(arrays)] + shape[axis+1:])
    shape[axis] = -1
    return np.stack(arrays, axis=axis+1, out=out).reshape(shape)


# Functions
def digitize_group(x, bins, right=False):
    """
    https://stackoverflow.com/a/26888164/7570817
    Similar to scipy.stats.binned_statistic but just return the indices corresponding to each bin.
    Same signature as numpy.digitize
    """
    idx_x = np.digitize(x=x, bins=bins, right=right)
    n, m = len(x), len(bins) + 1
    s = csr_matrix((np.arange(n), [idx_x, np.arange(n)]), shape=(m, n))
    return [group for group in np.split(s.data, s.indptr[1:-1])]


def sort_args(idx, *args):
    return [a[idx] for a in args]


def rolling_window(a, window):
    """https://stackoverflow.com/a/6811241/7570817"""
    a = np.array(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def find_subarray(a, b):
    """
    Find b in a. Return the index where the overlap begins.

    # a = np.array((2, 3, 4, 3, 5, 1))
    # b = np.array((3, 4, 3))
    # -> array([1])
    """
    a, b = np.atleast_1d(a, b)

    window = len(b)
    a_window = rolling_window(a=a, window=window)
    idx = np.nonzero((a_window == b).sum(axis=-1) == window)[0]
    return idx


def find_values(arr, values):
    res = np.zeros_like(arr, dtype=bool)
    for v in values:
        res[~res] = arr[~res] == v
    return res


def get_element_overlap(arr1, arr2=None, verbose=0):
    """
    arr1 is a 2D array (n, matrix)
    arr2 is a 2D array (l, k)
    along the first dimension are different samples and the second dimension are different features of one sample

    return a 2D int array (n, l) where each element o, j shows how many of the elements
    of arr1[o] are also present in arr2[j], without regard of specific position in the arrays
    """
    if arr2 is None:
        arr2 = arr1

    overlap = np.zeros((len(arr1), len(arr2)), dtype=int)
    for i, arr_i in enumerate(arr1):
        if verbose > 0:
            print(f"{i} / {len(arr1)}")
        for j, arr_j in enumerate(arr2):
            for k in arr_i:
                if k in arr_j:
                    overlap[i, j] += 1

    return overlap


def create_constant_diagonal(n, m, v, k):
    diag = np.eye(N=n, M=m, k=k) * v[0]
    for i in range(1, len(v)):
        diag += np.eye(N=n, M=m, k=k+i) * v[i]
    return diag


def banded_matrix(v_list, k0):
    m = np.diag(v_list[0], k=k0)
    for i, v in enumerate(v_list[1:], start=1):
        m += np.diag(v, k=k0+i)

    return m


def get_first_row_occurrence(bool_arr):
    """

    array([[ True,  True, False,  True,  True,  True],
           [False,  True, False,  True,  True, False],
           [False, False,  True, False, False, False],
           [False, False, False, False, False, False]])

    -> array([ 0,  1,  2, -1])
    """
    nz_i, nz_j = np.nonzero(bool_arr)
    u, idx = np.unique(nz_i, return_index=True)
    res = np.full(bool_arr.shape[0], fill_value=-1)
    res[u] = nz_j[idx]
    return res


def fill_interval_indices(interval_list, n):

    if isinstance(interval_list, np.ndarray):
        interval_list = interval_list.tolist()

    if np.size(interval_list) == 0:
        return np.array([[1, n]])

    if interval_list[0][0] != 0:
        interval_list.insert(0, [0, interval_list[0][0]])

    if interval_list[-1][1] != n:
        interval_list.insert(len(interval_list), [interval_list[-1][1], n])

    i = 1
    while i < len(interval_list):
        if interval_list[i - 1][1] != interval_list[i][0]:
            interval_list.insert(i, [interval_list[i - 1][1], interval_list[i][0]])
        i += 1

    return np.array(interval_list)


def get_interval_indices(bool_array):
    """
    Get list of start and end indices, which indicate the sections of True values in the array.

    Array is converted to bool first

    [0, 0, 0, 0]  ->  [[]]
    [0, 0, 0, 1]  ->  [[3, 4]]
    [0, 1, 1, 0]  ->  [[1, 3]]
    [1, 0, 0, 0]  ->  [[0, 1]]
    [1, 0, 0, 1]  ->  [[0, 1], [3, 4]]
    [1, 1, 0, 1]  ->  [[0, 2], [3, 4]]
    [1, 1, 1, 1]  ->  [[0, 4]]

    """

    assert bool_array.ndim == 1
    interval_list = np.where(np.diff(bool_array.astype(bool)) != 0)[0] + 1
    if bool_array[0]:
        interval_list = np.concatenate([[0], interval_list])
    if bool_array[-1]:
        interval_list = np.concatenate([interval_list, bool_array.shape])

    return interval_list.reshape(-1, 2)


def get_cropping_indices(*, pos, shape_small, shape_big, mode='lower_left'):
    """
    Adjust the boundaries to fit small array in a larger image.
    :param pos:  idx where the small image should be set in the bigger picture, option A
    :param mode:  mode how to position theta smaller array in the larger:
                  "center": pos describes the center of the small array inside the big array (shape_small must be odd)
                  "lower_left":
                  "upper_right":
    :param shape_small:  Size of the small image (=2*sm-1) in (number of pixels in each dimension)
    :param shape_big:  Size of the large image in (number of pixels in each dimension)
    :return:
    """

    shape_small, shape_big = args2arrays(shape_small, shape_big)

    if mode == 'center':
        assert np.all(np.array(shape_small) % 2 == 1)
        shape_small2 = (np.array(shape_small) - 1) // 2

        ll_big = pos - shape_small2
        ur_big = pos + shape_small2 + 1
    elif mode == 'lower_left':
        ll_big = pos
        ur_big = pos + shape_small
    elif mode == 'upper_right':
        ll_big = pos - shape_small
        ur_big = pos

    else:
        raise ValueError(f"Invalid position mode {mode}")

    ll_small = np.where(ll_big < 0,
                        -ll_big,
                        0)
    ur_small = np.where(shape_big - ur_big < 0,
                        shape_small + (shape_big - ur_big),
                        shape_small)

    ll_big = np.where(ll_big < 0, 0, ll_big)
    ur_big = np.where(shape_big - ur_big < 0, shape_big, ur_big)

    return ll_big, ur_big, ll_small, ur_small


def get_exclusion_mask(a, exclude_values):
    bool_a = np.ones_like(a, dtype=bool)
    for v in exclude_values:
        bool_a[a == v] = False

    return bool_a


def matsort(mat, order_j=None):
    """
    mat = np.where(np.random.random((500, 500)) > 0.01, 1, 0)
    ax[0].imshow(mat, cmap='gray_r')
    ax[1].imshow(matsort(mat)[0], cmap='gray_r')
    """

    def idx2interval(idx):
        idx = np.sort(idx)
        interval = np.zeros((len(idx) - 1, 2), dtype=int)
        interval[:, 0] = idx[:-1]
        interval[:, 1] = idx[1:]
        return interval

    n, m = mat.shape

    if order_j is None:
        order_j = np.argsort(np.sum(mat, axis=0))[::-1]

    order_i = np.argsort(mat[:, order_j[0]])[::-1]

    interval_idx = np.zeros(2, dtype=int)
    interval_idx[1] = n
    for i in range(0, m - 1):
        interval_idx = np.unique(np.hstack([interval_idx,
                                            get_interval_indices(mat[order_i, order_j[i]]).ravel()]))

        for j, il in enumerate(idx2interval(idx=interval_idx)):
            slice_j = slice(il[0], il[1])
            if j % 2 == 0:
                order_i[slice_j] = order_i[slice_j][np.argsort(mat[order_i[slice_j], order_j[i + 1]])[::-1]]
            else:
                order_i[slice_j] = order_i[slice_j][np.argsort(mat[order_i[slice_j], order_j[i + 1]])]

    return mat[order_i, :][:, order_j], order_i, order_j


def idx2boolmat(idx, n=100):
    """
    The last dimension of idx contains the indices. n-1 is the maximal possible index
    Returns matrix with shape np.shape(idx)[:-1] + (n,)
    """

    s = np.shape(idx)[:-1]

    mat = np.zeros(s + (n,), dtype=bool)

    for i, idx_i in enumerate(idx.reshape(-1, idx.shape[-1])):
        print(i, (np.unravel_index(i, shape=s)))
        mat[np.unravel_index(i, shape=s)][idx_i] = True
    return mat


def noise(shape, scale, mode='normal'):
    shape = shape_wrapper(shape)

    if mode == 'constant':  # could argue that this is no noise
        return np.full(shape=shape, fill_value=+scale)
    if mode == 'plusminus':
        return np.where(np.random.random(shape) < 0.5, -scale, +scale)
    if mode == 'uniform':
        return np.random.uniform(low=-scale, high=+scale, size=shape)
    elif mode == 'normal':
        return np.random.normal(loc=0, scale=scale, size=shape)
    else:
        raise ValueError(f"Unknown mode {mode}")


# Block lists
def expand_block_indices(idx_block, block_size, squeeze=True):
    """
    Expand the indices to get an index for each element

           block_size: |   1  |     2    |       3      |         4        |           5          |
    block_idx:          --------------------------------------------------------------------------|
              0        |   0  |   0,  1  |   0,  1,  2  |   0,  1,  2,  3  |   0,  1,  2,  3,  4  |
              1        |   1  |   2,  3  |   3,  4,  5  |   4,  5,  6,  7  |   5,  6,  7,  8,  9  |
              2        |   2  |   4,  5  |   6,  7,  8  |   8,  9, 10, 11  |  10, 11, 12, 13, 14  |
              3        |   3  |   6,  7  |   9, 10, 11  |  12, 13, 14, 15  |  15, 16, 17, 18, 19  |
              4        |   4  |   8,  9  |  12, 13, 14  |  16, 17, 18, 19  |  20, 21, 22, 23, 24  |
              5        |   5  |  10, 11  |  15, 16, 17  |  20, 21, 22, 23  |  25, 26, 27, 28, 29  |
              6        |   6  |  12, 13  |  18, 19, 20  |  24, 25, 26, 27  |  30, 31, 32, 33, 34  |
              7        |   7  |  14, 15  |  21, 22, 23  |  28, 29, 30, 31  |  35, 36, 37, 38, 39  |
              8        |   8  |  16, 17  |  24, 25, 26  |  32, 33, 34, 35  |  40, 41, 42, 43, 44  |
              9        |   9  |  18, 19  |  27, 28, 29  |  36, 37, 38, 39  |  45, 46, 47, 48, 49  |
              10       |  10  |  20, 21  |  30, 31, 32  |  40, 41, 42, 43  |  50, 51, 52, 53, 54  |
    """

    idx_block = np.atleast_1d(idx_block)
    if np.size(idx_block) == 1:
        return np.arange(block_size * int(idx_block), block_size * (int(idx_block) + 1))
    else:
        idx2 = np.array([expand_block_indices(i, block_size=block_size, squeeze=squeeze) for i in idx_block])
        if squeeze:
            return idx2.flatten()
        else:
            return idx2


def replace(arr, r_dict, copy=True, dtype=None):
    if copy:
        arr2 = arr.copy()
        if dtype is not None:
            arr2 = arr2.astype(dtype=dtype)
        for key in r_dict:
            arr2[arr == key] = r_dict[key]
        return arr2
    else:
        for key in r_dict:
            arr[arr == key] = r_dict[key]


def block_shuffle(arr, block_size, inside=False):
    """
    Shuffle the array along the first dimension,
    if block_size > 1, keep as many elements together and shuffle the n // block_size blocks
    """

    if isinstance(arr, int):
        n = arr
        arr = np.arange(n)
    else:
        n = arr.shape[0]

    if block_size == 1:
        np.random.shuffle(arr)
        return arr

    assert block_size > 0
    assert isinstance(block_size, int)
    assert n % block_size == 0
    n_blocks = n // block_size

    if inside:
        idx = np.arange(n)
        for i in range(0, n, block_size):
            np.random.shuffle(idx[i:i+block_size])
        return arr[idx]

    else:
        idx_block = np.arange(n_blocks)
        np.random.shuffle(idx_block)
        idx_ele = expand_block_indices(idx_block=idx_block, block_size=block_size, squeeze=True)
        return arr[idx_ele]


def replace_tail_roll(arr, arr_new):
    """
    Replace the last elements of the array with the new array and roll the new ones to the start
    So that a repeated call of this function cycles through the array
    replace_tail_roll(arr=[ 1,  2,  3,  4,  5,  6, 7, 8], arr_new=[77, 88])   -->   [77, 88,  1,  2,  3,  4,  5,  6]
    replace_tail_roll(arr=[77, 88,  1,  2,  3,  4, 5, 6], arr_new=[55, 66])   -->   [55, 66, 77, 88,  1,  2,  3,  4]
    replace_tail_roll(arr=[55, 66, 77, 88,  1,  2, 3, 4], arr_new=[33, 44])   -->   [33, 44, 55, 66, 77, 88,  1,  2]
    replace_tail_roll(arr=[33, 44, 55, 66, 77, 88, 1, 2], arr_new=[11, 22])   -->   [11, 22, 33, 44, 55, 66, 77, 88]
    """
    n_sample_new = np.shape(arr_new)[0]
    assert np.shape(arr)[0] > n_sample_new

    arr[-n_sample_new:] = arr_new
    return np.roll(arr, n_sample_new, axis=0)


def replace_tail_roll_list(arr_list, arr_new_list):
    assert len(arr_list) == len(arr_new_list)
    return (replace_tail_roll(arr=arr, arr_new=arr_new) for (arr, arr_new) in zip(arr_list, arr_new_list))


# grid
def get_points_inbetween(x, extrapolate=False):
    assert x.ndim == 1

    delta = x[1:] - x[:-1]
    x_new = np.zeros(np.size(x) + 1)
    x_new[1:-1] = x[:-1] + delta / 2
    if extrapolate:
        x_new[0] = x_new[1] - delta[0]
        x_new[-1] = x_new[-2] + delta[-1]
        return x_new
    else:
        return x_new[1:-1]


# def gen_dot_nm(x, y, z):
#     """
#     https://stackoverflow.com/questions/59347796/minimizing-overhead-due-to-the-large-number-of-numpy-dot-calls/59356461#59356461
#     """
#
#     # small kernels
#     @nb.njit(fastmath=True, parallel=True)
#     def dot_numba(A, B):
#         """
#         calculate dot product for (x,y)x(y,z)
#         """
#         assert A.shape[0] == B.shape[0]
#         assert A.shape[2] == B.shape[1]
#
#         assert A.shape[1] == x
#         assert B.shape[1] == y
#         assert B.shape[2] == z
#
#         res = np.empty((A.shape[0],A.shape[1],B.shape[2]),dtype=A.dtype)
#         for ii in nb.prange(A.shape[0]):
#             for o in range(x):
#                 for j in range(z):
#                     acc = 0.
#                     for k in range(y):
#                         acc += A[ii, o, k]*B[ii, k, j]
#                     res[ii, o, j] = acc
#         return res
#
#
#     #large kernels
#     @nb.njit(fastmath=True, parallel=True)
#     def dot_BLAS(A, B):
#         assert A.shape[0] == B.shape[0]
#         assert A.shape[2] == B.shape[1]
#
#         res = np.empty((A.shape[0], A.shape[1], B.shape[2]), dtype=A.dtype)
#         for ii in nb.prange(A.shape[0]):
#             A_ii = np.ascontiguousarray(A[ii, :, :])
#             B_ii = np.ascontiguousarray(A[ii, :, :])
#             res[ii, :, :] = np.dot(A_ii , B_ii)
#         return res
#
#     # At square matices above shape 20 calling BLAS is faster
#     if x >= 20 or y >= 20 or z >= 20:
#         return dot_BLAS
#     else:
#         return dot_numba
