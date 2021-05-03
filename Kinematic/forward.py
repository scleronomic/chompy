
import numpy as np

from Kinematic import chain as kc
from Optimizer.path import get_substeps

# j[4] = j[2] + j[3]
# dj[2] =
# f[4] = f2 @ f3 @ f4
# df4 = df2 @ f2 @ f4 + f2 @ f3 @ df4


# General
def get_frames(q, robot):
    return robot.get_frames(q)


def get_frames_jac(*, q, robot):
    return robot.get_frames_jac(q=q)


def get_frames_x(*, q, robot):
    return robot.get_frames(q=q)[..., :-1, -1]


def frames2pos(f, f_idx, x_rel):
    return (f[..., f_idx, :, :] @ x_rel[..., np.newaxis])[..., :-1, 0]


def frames2pos_spheres(f, robot):
    """
    x_spheres f.shape[:-2] + (n_spheres, n_dim)
    """
    return frames2pos(f=f, f_idx=robot.spheres_frame_idx, x_rel=robot.spheres_pos)


def frames2spheres_jac(f, j, robot):
    """
    x_spheres (n_samples, n_wp, n_spheres, n_dim)
    dx_dq (n_samples, n_wp, n_dof, n_spheres, n_dim)
    """
    x_spheres = frames2pos_spheres(f=f, robot=robot)
    dx_dq = (j[..., robot.spheres_frame_idx, :, :] @ robot.spheres_pos[:, :, np.newaxis])[..., :-1, 0]
    return x_spheres, dx_dq


def get_x_spheres(q, robot,
                  return_frames2=False):
    f = robot.get_frames(q=q)
    x_spheres = frames2pos_spheres(f=f, robot=robot)
    if return_frames2:
        return f, x_spheres
    else:
        return x_spheres


def get_x_spheres_jac(*, q, robot,
                      return_frames2=False):
    f, j = robot.get_frames_jac(q=q)
    x_spheres, dx_dq = frames2spheres_jac(f=f, j=j, robot=robot)
    if return_frames2:
        return (f, j), (x_spheres, dx_dq)
    else:
        return x_spheres, dx_dq


def get_x_spheres_substeps(q, n, robot,
                           include_start=True, return_frames2=False):
    q_ss = get_substeps(x=q, n=n, infinity_joints=robot.infinity_joints, include_start=include_start)
    return get_x_spheres(q=q_ss, robot=robot, return_frames2=return_frames2)


def get_x_spheres_substeps_jac(q, n, robot,
                               include_start=True, return_frames2=False):
    q_ss = get_substeps(x=q, n=n, infinity_joints=robot.infinity_joints, include_start=include_start)
    return get_x_spheres_jac(q=q_ss, robot=robot, return_frames2=return_frames2)


def get_frames_substeps(q, n, robot,
                        include_start=True):
    q_ss = get_substeps(x=q, n=n, infinity_joints=robot.infinity_joints, include_start=include_start)
    return get_frames(q=q_ss, robot=robot)


def get_frames_substeps_jac(q, n, robot, include_start=True):
    q_ss = get_substeps(x=q, n=n, infinity_joints=robot.infinity_joints, include_start=include_start)
    return robot.get_frames_jac(q=q_ss)


# Helper
def create_frames_dict(f, nfi):
    """
    Create a dict to minimize the calculation of unnecessary transformations between the frames

    The value to the key 0 holds all transformations form the origin to the whole chain.
    Each next field holds the transformation from the current frame to all frames to come.

    The calculation happens from back to front, to save some steps
    # 0     1     2     3     4
    # F01
    # F02   F12
    # F03   F13   F23
    # F04   F14   F24   F34
    # F05   F15   F25   F35   F45

    """
    n_frames = f.shape[-3]

    d = {}
    for i in range(n_frames - 1, -1, -1):
        nfi_i = nfi[i]

        if nfi_i == -1:
            d[i] = f[..., i:i + 1, :, :]

        elif isinstance(nfi_i, (list, tuple)):
            d[i] = np.concatenate([
                f[..., i:i + 1, :, :],
                f[..., i:i + 1, :, :] @ np.concatenate([d[j] for j in nfi_i], axis=-3)],
                axis=-3)

        else:
            d[i] = np.concatenate([f[..., i:i + 1, :, :],
                                   f[..., i:i + 1, :, :] @ d[nfi_i]], axis=-3)
    return d


def create_frames_dict_b(f, nfi, ff_inf):
    *shape, n_frames, n_dim, n_dim = f.shape
    d = np.zeros(shape + [n_frames, n_frames, n_dim, n_dim])

    for i in range(n_frames - 1, -1, -1):
        nfi_i = nfi[i]
        ff_inf_i = np.nonzero(ff_inf[i])[0][1:]
        d[..., i, i, :, :] = f[..., i, :, :]

        if isinstance(nfi_i, (list, tuple)):
            nf_i_all = [nf_i2 for nf_i2 in nfi_i for _ in range(ff_inf[nf_i2].sum())]
            d[..., i, ff_inf_i, :, :] = f[..., i:i+1, :, :] @ d[..., nf_i_all, ff_inf_i, :, :]

        elif nfi_i != -1:
            d[..., i, ff_inf_i, :, :] = f[..., i:i+1, :, :] @ d[..., nfi_i, ff_inf_i, :, :]
    return d


def test_create_frames_dict():
    from wzk import tic, toc
    d, d_b = None, None
    n = 100
    from Kinematic.Robots import StaticArm

    robot = StaticArm(n_dof=20)
    robot.joint_frame_idx = robot.joint_frame_idx.tolist()
    robot.joint_frame_idx[5] = (5, 7, 9)

    tic()
    for i in range(n):
        f = np.ones((len(robot.next_frame_idx), 4, 4))
        j = np.ones((robot.n_dof, robot.n_frames, 4, 4))
        d2 = create_frames_dict(f=f, nfi=robot.next_frame_idx)

        d = np.zeros((robot.n_frames, robot.n_frames))
        for key in d2:
            d[int(key), int(key):] = d2[key][..., 0, 0]
        combine_frames_jac(j=j, d=d2, robot=robot)
    toc()

    tic()
    for i in range(n):
        f = np.ones((len(robot.next_frame_idx), 4, 4))
        j_b = np.ones((robot.n_dof, robot.n_frames, 4, 4))
        d_b = create_frames_dict_b(f=f, nfi=robot.next_frame_idx, ff_inf=robot.frame_frame_influence)
        combine_frames_jac_b(j=j_b, d=d_b, robot=robot)
    toc()

    # tic()
    # for i in range(n):
    #     f = np.ones((len(robot.next_frame_idx), 4, 4))
    #     j_c = np.ones((robot.n_dof, robot.n_frames, 4, 4))
    #     d_c = create_frames_dict_b(f=f, nfi=robot.next_frame_idx, ff_inf=robot.frame_frame_influence)
    #     combine_frames_jac_b_simple(j=j_c, d=d_c, robot=robot)
    # toc()

    print('Dict', np.allclose(d, d_b[..., 0, 0]))
    print('Jac', np.allclose(j, j_b))
    # print(np.allclose(j_b, j_c))
    b = ((j - j_b) == 0).sum(axis=(-1, -2)) == 16
    print(b.astype(int))


def combine_frames(f, prev_frame_idx):
    for i, pfi in enumerate(prev_frame_idx[1:], start=1):
        f[..., i, :, :] = f[..., pfi, :, :] @ f[..., i, :, :]


# TODO make this simpler by working just with the DH values here and combine coupled / passive joints in a separate layer
#   should make it easier to understand and also faster for the majority of 2d usecases
def combine_frames_jac(j, d, robot):
    n_dof = j.shape[-4]
    jf_all, jf_first, jf_last = kc.__get_joint_frame_indices_first_last(jfi=robot.joint_frame_idx)

    pfi_ = robot.prev_frame_idx[jf_first]
    joints_ = np.arange(n_dof)[pfi_ != -1]
    jf_first_ = jf_first[pfi_ != -1]
    pfi_ = pfi_[pfi_ != -1]

    # Previous to joint frame
    # j(b)__a_b = f__a_b * j__b
    j[..., joints_, jf_first_, :, :] = (d[0][..., pfi_, :, :] @ j[..., joints_, jf_first_, :, :])

    # After
    for i in range(n_dof):
        jf_inf_i = robot.joint_frame_influence[i, :]
        jf_inf_i[:jf_last[i] + 1] = False
        nfi_i = robot.next_frame_idx[jf_last[i]]

        # Handle joints which act on multiple frames
        if jf_first[i] != jf_last[i]:
            for kk, fj_cur in enumerate(jf_all[i][:-1]):
                jf_next = jf_all[i][kk + 1]
                jf_next1 = jf_next - 1

                if jf_next - fj_cur > 1:
                    j[..., i, fj_cur + 1:jf_next, :, :] = (j[..., i, fj_cur:fj_cur + 1, :, :] @
                                                           d[robot.next_frame_idx[fj_cur]][..., :jf_next - fj_cur - 1, :, :])

                j[..., i, jf_next, :, :] = ((j[..., i, jf_next1, :, :] @ d[robot.next_frame_idx[jf_next1]][..., 0, :, :]) +
                                            (d[0][..., jf_next1, :, :] @ j[..., i, jf_next, :, :]))

        # j(b)__a_c = j__a_b * f__b_c
        if isinstance(nfi_i, (list, tuple)):
            j[..., i, jf_inf_i, :, :] = (j[..., i, jf_last[i]:jf_last[i] + 1, :, ] @ np.concatenate([d[j] for j in nfi_i], axis=-3))
        elif nfi_i != -1:
            j[..., i, jf_inf_i, :, :] = (j[..., i, jf_last[i]:jf_last[i] + 1, :, :] @ d[nfi_i])


def combine_frames_jac_b(j, d, robot):

    jf_idx, jf_idx_first, jf_idx_last = kc.__get_joint_frame_indices_first_last(jfi=robot.joint_frame_idx)

    pfi_ = robot.prev_frame_idx[jf_idx_first]
    joints_ = np.arange(robot.n_dof)[pfi_ != -1]
    jf_first_ = jf_idx_first[pfi_ != -1]
    pfi_ = pfi_[pfi_ != -1]

    # Previous to joint frame
    # j(b)__a_b = f__a_b * j__b
    j[..., joints_, jf_first_, :, :] = (d[..., 0, pfi_, :, :] @ j[..., joints_, jf_first_, :, :])

    # After
    for i in range(robot.n_dof):
        jf_inf_i = robot.joint_frame_influence[i, :]
        jf_inf_i[:jf_idx_last[i] + 1] = False
        nfi_i = robot.next_frame_idx[jf_idx_last[i]]
        ff_inf_i = np.nonzero(robot.frame_frame_influence[i])[0]

        # Handle joints which act on multiple frames
        if jf_idx_first[i] != jf_idx_last[i]:
            for kk, fj_cur in enumerate(jf_idx[i][:-1]):
                jf_idx_next = jf_idx[i][kk + 1]
                jf_idx_next1 = jf_idx_next - 1
                nfi_j = robot.next_frame_idx[fj_cur]

                if jf_idx_next - fj_cur > 1:
                    j[..., i, fj_cur + 1:jf_idx_next, :, :] = \
                        (j[..., i, fj_cur:fj_cur + 1, :, :] @ d[..., nfi_j, nfi_j:nfi_j+jf_idx_next-fj_cur - 1, :, :])

                j[..., i, jf_idx_next, :, :] = ((j[..., i, jf_idx_next1, :, :] @ d[..., robot.next_frame_idx[jf_idx_next1], 0, :, :]) +
                                                (d[..., 0, jf_idx_next1, :, :] @ j[..., i, jf_idx_next, :, :]))

        if isinstance(nfi_i, (list, tuple)):
            nf_i_all = [nf_i2 for nf_i2 in nfi_i for _ in range(robot.frame_frame_influence[nf_i2].sum())]
            j[..., i, ff_inf_i, :, :] = (j[..., i, jf_idx_last[i]:jf_idx_last[i] + 1, :, :] @
                                         d[..., nf_i_all, jf_inf_i, :, :])
        elif nfi_i != -1:
            j[..., i, jf_inf_i, :, :] = (j[..., i, jf_idx_last[i]:jf_idx_last[i] + 1, :, :] @
                                         d[..., nfi_i, jf_inf_i, :, :])


def combine_frames_jac_b_simple(j, d, robot):

    pfi_ = robot.prev_frame_idx[robot.joint_frame_idx]
    joints_ = np.arange(robot.n_dof)[pfi_ != -1]
    jf_idx_first_ = robot.joint_frame_idx[pfi_ != -1]
    pfi_ = pfi_[pfi_ != -1]

    # Previous to joint
    j[..., joints_, jf_idx_first_, :, :] = (d[..., 0, pfi_, :, :] @ j[..., joints_, jf_idx_first_, :, :])

    # After joint
    for i in range(robot.n_dof):
        jf_inf_i = robot.joint_frame_influence[i, :]
        jf_inf_i[:robot.joint_frame_idx[i] + 1] = False
        ff_inf_i = np.nonzero(robot.frame_frame_influence[i])[0]
        nfi_i = robot.next_frame_idx[robot.joint_frame_idx[i]]
        jf_idx_i = robot.joint_frame_idx[i]

        if isinstance(nfi_i, (list, tuple)):
            nf_i_all = [nf_i2 for nf_i2 in nfi_i for _ in range(robot.frame_frame_influence[nf_i2].sum())]
            j[..., i, ff_inf_i, :, :] = (j[..., i, jf_idx_i:jf_idx_i + 1, :, :] @
                                         d[..., nf_i_all, jf_inf_i, :, :])
        elif nfi_i != -1:
            j[..., i, jf_inf_i, :, :] = (j[..., i, jf_idx_i:jf_idx_i + 1, :, :] @
                                         d[..., nfi_i, jf_inf_i, :, :])


def get_torques(f,
                torque_frame_idx, frame_frame_influence,
                mass, mass_pos, mass_frame_idx,
                gravity=None,
                mode='f'):
    # Finding the torque about a given axis does not depend on the specific location on the axis where the torque acts

    *shape, n_frames, _, _ = f.shape

    if gravity is None:
        gravity = np.array([[0., 0., -9.81]])  # matrix / s**2
    force = mass[np.newaxis, :, np.newaxis] * gravity[:, np.newaxis, :]
    x_mass = frames2pos(f=f, f_idx=mass_frame_idx, x_rel=mass_pos)

    torques_around_point = np.empty(shape + [len(torque_frame_idx), 3])
    for i, idx_frame in enumerate(torque_frame_idx):
        x_frame = f[..., idx_frame, :3, -1]
        mass_bool = frame_frame_influence[idx_frame][mass_frame_idx]
        r = x_mass[..., mass_bool, :] - x_frame[..., np.newaxis, :]
        torques_around_point[..., i, :] = np.cross(r, force[:, mass_bool, :]).sum(axis=-2)

    if mode == 'dh':
        # IDENTIFICATION OF GEOMETRIC AND NON GEOMETRIC PARAMETERS OF ROBOTS, J.L. Caenen, J.C. Angue, 1990
        # Torque_x = (0_M_j - 0_P_(i-1)) x (m_j*g) @ x_(i-1)
        # Torque_y = (0_M_j - 0_P_(i-1)) x (m_j*g) @ y_(i-1)
        # Torque_z = (0_M_j - 0_P_i)     x (m_j*g) @ z_i
        f_rot_xy = np.swapaxes(f[..., torque_frame_idx-1, :3, 0:2], -1, -2)
        f_rot_z = np.swapaxes(f[..., torque_frame_idx, :3, 2:3], -1, -2)
        f_rot = np.concatenate((f_rot_xy, f_rot_z), axis=-2)

    elif mode == 'f':
        f_rot = np.swapaxes(f[..., torque_frame_idx, :3, :3], -1, -2)

    torques_around_axes = (f_rot @ torques_around_point[..., np.newaxis])[..., 0]

    return torques_around_point, torques_around_axes


if __name__ == '__main__':
    # test_create_frames_dict()
    pass
#
# q = np.zeros(10)
#
#
# def q2q(q):
#     # q2 = np.zeros(5)
#     # for i in range(5):
#     #     q2[i] = q[2*i] + q[2*i+1]
#     q2 = np.repeat(q, 2)
#     return q2
#
#
# def q2q_jac(q):
#     j = np.zeros((len(q), len(q)*2))
#     for i in range(len(q)):
#         j[i, 2*i:2*i+2] = 1
#
#
# q = np.random.random(5)
# # q2q(np.arange(10))
#
# from Robots import StaticArm
# robot = StaticArm(n_dof=10)
# # robot.