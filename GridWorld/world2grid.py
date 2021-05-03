import numpy as np

from wzk import safe_unify


def limits2voxel_size(shape, limits):
    voxel_size = np.diff(limits, axis=-1)[:, 0] / np.array(shape)
    return safe_unify(x=voxel_size)


def __mode2offset(voxel_size, mode='c'):
    """Modes
        'c': center
        'b': boundary

    """
    if mode == 'c':
        return voxel_size / 2
    elif mode == 'b':
        return 0
    else:
        raise NotImplementedError(f"Unknown offset mode{mode}")


def grid_x2i(*, x, voxel_size, lower_left):
    """
    Get the indices of the grid cell at the coordinates 'x' in a grid with symmetric cells.
    Always use mode='boundary'
    """

    if x is None:
        return None

    return np.asarray((x - lower_left) / voxel_size, dtype=int)


def grid_i2x(*, i, voxel_size, lower_left, mode='c'):
    """
    Get the coordinates of the grid at the index 'o' in a grid with symmetric cells.
    borders: 0 | 2 | 4 | 6 | 8 | 10
    centers: | 1 | 3 | 5 | 7 | 9 |
    """

    if i is None:
        return None

    offset = __mode2offset(voxel_size=voxel_size, mode=mode)
    return np.asarray(lower_left + offset + i * voxel_size, dtype=float)
