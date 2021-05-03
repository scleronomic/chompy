import os
import numpy as np
from wzk import atleast_list, tile_2d, repeat2new_shape, initialize_image_array

from GridWorld import  random_obstacles


def __check_ip_world(world):
    if 'empty' in world:
        return False
    elif all([isinstance(s, str) for s in world]):
        return True
    else:
        return False


def create_template(n_voxels, world='empty'):
    if world is None:
        world = 'empty'

    if isinstance(world, tuple) and len(world) == 2:
        try:
            n_obstacles, size_limits = int(world[0]), int(world[1])
            return random_obstacles.create_rectangle_image(n=n_obstacles, size_limits=size_limits, n_voxels=n_voxels,
                                                           special_dim=None)
        except ValueError:
            pass

    if os.path.isfile(world):
        return np.load(world)

    if len(n_voxels) == 2:
        return create_template_2d(n_voxels=n_voxels, world=world)
    else:
        raise ValueError(f"Only 2 dimensions are allowed but got: {n_voxels}")


def create_template_2d(n_voxels, world='empty'):
    img = initialize_image_array(n_voxels=n_voxels, n_dim=2, n_samples=None)
    s = img.shape

    world = atleast_list(world, convert=False)
    if not __check_ip_world(world=world):
        return img

    if 'perlin' in world:
        img[:] = random_obstacles.create_perlin_image(n=1, n_voxels=n_voxels)

    if 'boxes_1' in world:
        img[s[0] * 4 // 10:s[0] * 6 // 10,
            s[1] * 4 // 10:s[1] * 6 // 10] = True

    if 'boxes_1_b' in world:
        img[s[0] * 3 // 10:s[0] * 7 // 10,
            s[1] * 3 // 10:s[1] * 7 // 10] = True

    if 'boxes_1_big' in world:
        img[s[0] * 3 // 10:s[0] * 7 // 10,
            s[1] * 3 // 10:s[1] * 7 // 10] = True

    if 'boxes_1_small' in world:
        img[s[0] * 9 // 20:s[0] * 11 // 20,
            s[1] * 9 // 20:s[1] * 11 // 20] = True

    if 'boxes_1_c' in world:
        img[s[0] *  7 // 16:s[0] *  9 // 16,
            s[1] *  2 // 16:s[1] *  4 // 16] = True

    if 'boxes_2' in world:
        img[s[0] *  7 // 16:s[0] *  9 // 16,
            s[1] *  2 // 16:s[1] *  4 // 16] = True
        img[s[0] *  7 // 16:s[0] *  9 // 16,
            s[1] * 12 // 16:s[1] * 14 // 16] = True

    if 'boxes_3' in world:
        img[s[1] * 5 // 20:s[1] * 8 // 20,
            s[0] * 5 // 20:s[0] * 8 // 20] = True
        img[s[1] * 12 // 20:s[1] * 15 // 20,
            s[0] * 5 // 20:s[0] * 8 // 20] = True
        img[s[1] * 9 // 20:s[1] * 12 // 20,
            s[0] * 12 // 20:s[0] * 15 // 20] = True

    if 'boxes_4' in world:
        img[s[0] *  7 // 16:s[0] *  9 // 16,
            s[1] *  2 // 16:s[1] *  4 // 16] = True
        img[s[0] *  2 // 16:s[0] *  4 // 16,
            s[1] *  7 // 16:s[1] *  9 // 16] = True
        img[s[0] * 12 // 16:s[0] * 14 // 16,
            s[1] *  7 // 16:s[1] *  9 // 16] = True
        img[s[0] *  7 // 16:s[0] *  9 // 16,
            s[1] * 12 // 16:s[1] * 14 // 16] = True

    if 'boxes_5' in world:
        img[s[0] * 2 // 10:s[0] * 4 // 10,
            s[1] * 6 // 10:s[1] * 8 // 10] = True
        img[s[0] * 3 // 10:s[0] * 5 // 10,
            s[1] * 5 // 10:s[1] * 7 // 10] = True
        img[s[0] * 4 // 10:s[0] * 6 // 10,
            s[1] * 4 // 10:s[1] * 6 // 10] = True
        img[s[0] * 5 // 10:s[0] * 7 // 10,
            s[1] * 3 // 10:s[1] * 5 // 10] = True
        img[s[0] * 6 // 10:s[0] * 8 // 10,
            s[1] * 2 // 10:s[1] * 4 // 10] = True

    if 'boxes_8' in world:
        bs = s[0] // 16, s[1] // 16
        img[s[0] * 11 // 32: s[0] * 11 // 32 + bs[0],
            s[1] *  7 // 32: s[1] *  7 // 32 + bs[1]] = True
        img[s[0] * 19 // 32: s[0] * 19 // 32 + bs[0],
            s[1] *  7 // 32: s[1] *  7 // 32 + bs[1]] = True

        img[s[0] *  7 // 32: s[0] *  7 // 32 + bs[0],
            s[1] * 11 // 32: s[1] * 11 // 32 + bs[1]] = True
        img[s[0] * 23 // 32: s[0] * 23 // 32 + bs[0],
            s[1] * 11 // 32: s[1] * 11 // 32 + bs[1]] = True

        img[s[0] *  7 // 32: s[0] *  7 // 32 + bs[0],
            s[1] * 19 // 32: s[1] * 19 // 32 + bs[1]] = True
        img[s[0] * 23 // 32: s[0] * 23 // 32 + bs[0],
            s[1] * 19 // 32: s[1] * 19 // 32 + bs[1]] = True

        img[s[0] * 11 // 32: s[0] * 11 // 32 + bs[0],
            s[1] * 23 // 32: s[1] * 23 // 32 + bs[1]] = True
        img[s[0] * 19 // 32: s[0] * 19 // 32 + bs[0],
            s[1] * 23 // 32: s[1] * 23 // 32 + bs[1]] = True

    if 'slalom_2' in world:
        img[s[0] * 6 // 20:s[0] * 8 // 20,
            s[1] * 0 // 20:s[1] * 11 // 20] = True
        img[s[0] * 12 // 20:s[0] * 14 // 20,
            s[1] * 9 // 20:s[1] * 20 // 20] = True

    if 'slalom_3' in world:
        img[s[0] * 4 // 20:s[0] * 6 // 20,
            s[1] * 0 // 20:s[1] * 11 // 20] = True
        img[s[0] * 9 // 20:s[0] * 11 // 20,
            s[1] * 9 // 20:s[1] * 20 // 20] = True
        img[s[0] * 14 // 20:s[0] * 16 // 20,
            s[1] * 0 // 20:s[1] * 11 // 20] = True

    if 'slalom_3_b' in world:
        img[s[0] * 4 // 20:s[0] * 6 // 20,
            s[1] * 0 // 20:s[1] * 13 // 20] = True
        img[s[0] * 9 // 20:s[0] * 11 // 20,
            s[1] * 7 // 20:s[1] * 20 // 20] = True
        img[s[0] * 14 // 20:s[0] * 16 // 20,
            s[1] * 0 // 20:s[1] * 13 // 20] = True

    if 'slalom_3_c' in world:
        img[s[0] * 4 // 20:s[0] * 6 // 20,
            s[1] * 0 // 20:s[1] * 15 // 20] = True
        img[s[0] * 9 // 20:s[0] * 11 // 20,
            s[1] * 5 // 20:s[1] * 20 // 20] = True
        img[s[0] * 14 // 20:s[0] * 16 // 20,
            s[1] * 0 // 20:s[1] * 15 // 20] = True

    if 'alley' in world:
        img[s[0] * 5 // 20:s[0] * 15 // 20,
            s[1] * 0 // 20:s[1] * 20 // 20] = True
        img[s[0] * 5 // 20:s[0] * 15 // 20,
            s[1] * 9 // 20:s[1] * 11 // 20] = False

    if 'alley_narrow' in world:
        img[s[0] * 5 // 20:s[0] * 15 // 20,
            s[1] * 0 // 20:s[1] * 20 // 20] = True
        img[s[0] * 5 // 20:s[0] * 15 // 20,
            s[1] * 18 // 40:s[1] * 21 // 40] = False

    if 'alley_kink' in world:
        img[s[0] * 5 // 20:s[0] * 15 // 20,
            s[1] * 0 // 20:s[1] * 20 // 20] = True

        img[s[0] * 5 // 20:s[0] * 10 // 20,
            s[1] * 7 // 20:s[1] * 9 // 20] = False
        img[s[0] * 10 // 20:s[0] * 15 // 20,
            s[1] * 11 // 20:s[1] * 13 // 20] = False
        img[s[0] * 9 // 20:s[0] * 11 // 20,
            s[1] * 7 // 20:s[1] * 13 // 20] = False

    if 'cross_inner' in world:
        img[s[0] * 2 // 20:s[0] * 18 // 20,
            s[1] * 9 // 20:s[1] * 11 // 20] = True
        img[s[0] * 9 // 20:s[0] * 11 // 20,
            s[1] * 2 // 20:s[1] * 18 // 20] = True

    if 'cross_outer' in world:
        img[s[0] * 9 // 20:s[0] * 11 // 20, :] = True
        img[:, s[1] * 9 // 20:s[1] * 11 // 20] = True
        img[s[0] * 8 // 20:s[0] * 12 // 20,
            s[1] * 8 // 20:s[1] * 12 // 20] = False

    if 'pattern' in world:

        if 'point_a' in world:
            pattern = np.ones((1, 1))
            shape = (11, 11)
            offset = (0, 0)
            v_in_row = 4
            v_to_next_row = (1, 2)

        elif 'point_b' in world:
            pattern = np.ones((1, 1))
            shape = (11, 11)
            offset = (0, 0)
            v_in_row = 5
            v_to_next_row = (1, 2)

        elif 'dumbbell' in world:
            pattern = np.ones((2, 2))
            pattern[0, 1] = 0
            pattern[1, 0] = 0
            shape = (16, 16)
            offset = (0, 0)
            v_in_row = 7
            v_to_next_row = (1, 3)

        elif 'triangle' in world:
            pattern = np.ones((2, 2))
            pattern[0, 1] = 0
            shape = (10, 10)
            offset = (0, 0)
            v_in_row = 8
            v_to_next_row = (1, 3)

        elif 'cross' in world:
            pattern = np.zeros((3, 3))
            pattern[1, :] = 1
            pattern[:, 1] = 1
            shape = (13, 13)
            offset = (1, 1)
            v_in_row = 4
            v_to_next_row = (3, 2)

        else:
            raise ValueError(f"Unknown pattern '{world}'")

        img_pattern = tile_2d(pattern=pattern, v_in_row=v_in_row, v_to_next_row=v_to_next_row, offset=offset,
                              shape=shape)
        img[repeat2new_shape(img=img_pattern, new_shape=s).astype(bool)] = True

    if 'transposed' in world:
        img = img.T

    return img

