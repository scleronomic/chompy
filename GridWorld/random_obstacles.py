import numpy as np

from wzk import (initialize_image_array, perlin_noise_2d, perlin_noise_3d)


def create_rectangles(*, n, size_limits=(1, 10), n_voxels=(64, 64),
                      special_dim=None):
    """
    Create a series of randomly placed and randomly sized rectangles.
    The bottom left corner and the shape in each dimension are returned.
    All values are pixel based. so there are no obstacles only occupying half a cell.
    The flag wall_dims indicates, if the obstacles should be walls, rather than general cubes. For walls one dimension
    is significant thinner than the others. The possible dimensions are defined in the tuple 'wall_dims'.
    """

    # Convert obstacle dimensions to number of occupied grid cells
    n_dim = np.size(n_voxels)

    if np.isscalar(n):
        n = int(n)
        rect_pos = np.zeros((n, n_dim), dtype=int)
        for d in range(n_dim):
            rect_pos[:, d] = np.random.randint(low=0, high=n_voxels[d] - size_limits[0] + 1, size=n)
        rect_size = np.random.randint(low=size_limits[0], high=size_limits[1] + 1, size=(n, n_dim))

        if special_dim is not None and special_dim != (None, None):
            dimensions, size = special_dim
            if isinstance(dimensions, int):
                dimensions = (dimensions,)

            rect_size[list(range(n)), np.random.choice(dimensions, size=n)] = size

        # Crop rectangle shape, if the created obstacle goes over the boundaries of the world
        diff = rect_pos + rect_size
        ex = np.where(diff > n_voxels, True, False)
        rect_size[ex] -= (diff - n_voxels)[ex]

        return rect_pos, rect_size

    else:

        rect_pos = np.empty(len(n), dtype=object)
        rect_size = np.empty(len(n), dtype=object)

        for i, nn in enumerate(n):
            rect_pos[i], rect_size[i] = create_rectangles(n_voxels=n_voxels, n=nn,
                                                          size_limits=size_limits,
                                                          special_dim=special_dim)
        return rect_pos, rect_size


def rectangles2image(*, rect_pos, rect_size, n_voxels=(64, 64), n_samples=None):
    """
    Create an image / a world out of rectangles with given position and shapes.
    The image is black/white (~True/False -> bool): False for free, True for obstacle.
    """

    if n_samples is None:
        n_obstacles, n_dim = rect_pos.shape
        obstacle_img = initialize_image_array(n_voxels=n_voxels, n_dim=n_dim)

        for i in range(n_obstacles):
            ll_i = np.array(rect_pos[i])
            ur_i = ll_i + np.array(rect_size[i])
            obstacle_img[tuple(map(slice, ll_i, ur_i))] = True

    else:
        n_dim = rect_pos[0].shape[0]
        obstacle_img = initialize_image_array(n_voxels=n_voxels, n_dim=n_dim, n_samples=n_samples)
        for i in range(n_samples):
            obstacle_img[i] = rectangles2image(n_voxels=n_voxels, rect_pos=rect_pos[i], rect_size=rect_size[i],
                                               n_samples=None)

    return obstacle_img


def create_rectangle_image(*, n, n_voxels=(64, 64),
                           size_limits=(1, 10),
                           special_dim=None, return_rectangles=False):

    if np.isscalar(n):
        n_samples = None
    else:
        n_samples = len(n)

    rect_pos, rect_size = create_rectangles(n=n, size_limits=size_limits, n_voxels=n_voxels, special_dim=special_dim)
    img = rectangles2image(rect_pos=rect_pos, rect_size=rect_size, n_voxels=n_voxels, n_samples=n_samples)

    if return_rectangles:
        return img, (rect_pos, rect_size)
    else:
        return img


def create_perlin_image(n_voxels, n=1, res=4, threshold=0.5,
                        squeeze=True):
    n_dim = np.size(n_voxels)

    if n is None:
        n = 1

    if n_dim == 2:
        noise = np.array([perlin_noise_2d(shape=n_voxels, res=(res, res)) for _ in range(n)])
    elif n_dim == 3:
        noise = np.array([perlin_noise_3d(shape=n_voxels, res=(res, res, res)) for _ in range(n)])
    else:
        raise ValueError

    noise = np.logical_or(noise < -threshold, threshold < noise)
    if squeeze and n == 1:
        noise = noise[0]
    return noise

