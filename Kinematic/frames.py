import numpy as np
from scipy.spatial.transform import Rotation
from wzk import random_uniform_ndim, shape_wrapper, max_size, sample_points_on_sphere_3d, noise

# angle axis representation is like a onion, the singularity is the boarder to the next 360 shell
# 0 is 1360 degree away from the next singularity -> nice

# Nomenclature
# matrix ~ SE3 Matrix (3x3)
# frame ~ (4x4) homogen matrix, SE3 + translation
# different representations of rotations
__2d_theta = '2d_theta'
__2d_xcos_ysin = '2d_xcos_ysin'
__3d_euler = '3d_euler'
__3d_rotvec = '3d_rotvec'
__3d_quat = '3d_quat'
__euler_angle_seq = 'ZXZ'

__n_dim2rot_mode_default_dict = {1: __2d_theta,
                                 2: __2d_xcos_ysin,
                                 3: __3d_rotvec,
                                 # 3: __3d_euler,
                                 4: __3d_quat}

__rot2n_dim_dict = {1: 2,
                    2: 2,
                    3: 3,
                    4: 3}


# Util
def initialize_frames(shape, n_dim, mode='hm'):
    frames = np.zeros((shape_wrapper(shape) + (n_dim+1, n_dim+1)))
    if mode == 'zero':
        pass
    elif mode == 'eye':
        fill_frames_diag(frames=frames)
    elif mode == 'hm':
        frames[..., -1, -1] = 1
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    return frames


def invert(f):
    """
    Create the inverse of an array of hm frames
    Assume n x n are the last two dimensions of the array
    """

    n_dim = f.shape[-1] - 1
    t = f[..., :n_dim, -1]  # Translation

    # Apply the inverse rotation on the translation
    f_inv = f.copy()
    f_inv[..., :n_dim, :n_dim] = np.swapaxes(f_inv[..., :n_dim, :n_dim], axis1=-1, axis2=-2)
    f_inv[..., :n_dim, -1:] = -f_inv[..., :n_dim, :n_dim] @ t[..., np.newaxis]
    return f_inv


def apply_eye_wrapper(f, possible_eye):
    if possible_eye is None or np.allclose(possible_eye, np.eye(possible_eye.shape[0])):
        return f
    else:
        return possible_eye @ f


def from_2d_to_3d(f_2d):
    shape = np.array(f_2d.shape)
    shape[-2:] += 1
    frames_3d = np.zeros(shape)

    frames_3d[..., :-2, :-2] = f_2d[..., :-1, :-1]
    frames_3d[..., :-2, -1] = f_2d[..., :-1, -1]
    frames_3d[..., -2, -2] = 1
    frames_3d[..., -1, -1] = 1

    return frames_3d


# Switching frames representations
def angle2xcos_ysin(q):
    unit_vectors = np.empty(np.shape(q) + (2,))
    np.cos(q, out=unit_vectors[..., 0])
    np.sin(q, out=unit_vectors[..., 1])
    return unit_vectors


def xcos_ysin2angle(xcos, ysin):
    return np.arctan2(x1=ysin, x2=xcos)


# 2matrix, matrix2
def euler2matrix(euler, seq=__euler_angle_seq):
    return Rotation.from_euler(seq=seq, angles=euler.reshape((-1, 3))
                               ).as_matrix().reshape(euler.shape[:-1] + (3, 3))


def quaternions2matrix(quat):
    return Rotation.from_quat(quat=quat.reshape((-1, 4))
                              ).as_matrix().reshape(quat.shape[:-1] + (3, 3))


def rotvec2matrix(rotvec):
    return Rotation.from_rotvec(rotvec=rotvec.reshape((-1, 3))
                                ).as_matrix().reshape(rotvec.shape[:-1] + (3, 3))


def matrix2euler(matrix, seq=__euler_angle_seq):
    """
    The default sequence 'ZXZ'.
    """
    return Rotation.from_matrix(matrix=matrix.reshape((-1, 3, 3))
                                ).as_euler(seq=seq).reshape(matrix.shape[:-2] + (3,))


def matrix2quaternions(matrix):
    return Rotation.from_matrix(matrix=matrix.reshape((-1, 3, 3))
                                ).as_quat().reshape(matrix.shape[:-2] + (4,))


def matrix2rotvec(matrix):
    return Rotation.from_matrix(matrix=matrix.reshape((-1, 3, 3))
                                ).as_rotvec().reshape(matrix.shape[:-2] + (3,))


# frames2, 2frames
def frame2trans(f):
    """
    Get the xy(z) position from the 4x4 (3x3) frames / jacobi.
    Assume that the homogeneous matrices are in the last 2 dimensions of the array.
    Attention this is just a view of the Measurements!
    """
    return f[..., :-1, -1]


# frames2rotation
def frame2quat(f):
    return matrix2quaternions(matrix=f[..., :3, :3])


def frame2euler(f, seq=__euler_angle_seq):
    return matrix2euler(matrix=f[..., :3, :3], seq=seq)


def frame2rotvec(f):
    return matrix2rotvec(matrix=f[..., :3, :3])


def frame2theta_2d(f_2d):
    return np.arctan2(f_2d[..., 1, 0], f_2d[..., 0, 0])


def frame2xcosysin_2d(f_2d):
    return f_2d[..., :-1, 0]


def frame2rot(f, mode=None):
    n_dim = f.shape[-1] - 1
    if n_dim == 2:
        if mode is None:
            mode = __2d_xcos_ysin

        if mode == __2d_xcos_ysin:
            return frame2xcosysin_2d(f_2d=f)
        elif mode == __2d_theta:
            return frame2theta_2d(f_2d=f)
        else:
            raise ValueError(f"Unknown mode for n_dim:{n_dim} {mode}")

    elif n_dim == 3:
        if mode is None:
            mode = __3d_quat

        if mode == __3d_quat:
            return frame2quat(f=f)
        elif mode == __3d_euler:
            return frame2euler(f=f)
        elif mode == __3d_rotvec:
            return frame2rotvec(f=f)

        else:
            raise ValueError(f"Unknown mode for n_dim:{n_dim} {mode}")

    else:
        raise ValueError(f"Unknown shape: {f.shape}. Last two dimension must be 2D or 3D")


#
def frame2trans_rotvec(f):
    return frame2trans(f=f), frame2rotvec(f=f)


def frame2trans_quat(f):
    return frame2trans(f=f), frame2quat(f=f)


def frame2trans_rot(f, mode=None):
    return frame2trans(f=f), frame2rot(f=f, mode=mode)


def trans_quat2frame(trans=None, quat=None):
    s = quat.shape if trans is None else trans.shape

    frames = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=frames, trans=trans)
    frames[..., :-1, :-1] = quaternions2matrix(quat=quat)
    return frames


def trans_rotvec2frame(trans=None, rotvec=None):
    s = rotvec.shape if trans is None else trans.shape

    frames = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=frames, trans=trans)
    frames[..., :-1, :-1] = rotvec2matrix(rotvec=rotvec)
    return frames


def trans_euler2frame(trans=None, euler=None):
    s = euler.shape if trans is None else trans.shape

    frames = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=frames, trans=trans)
    frames[..., :-1, :-1] = euler2matrix(euler=euler)
    return frames


def trans_theta2frame(trans=None, theta=None):
    s = theta.shape if trans is None else trans.shape

    frames = initialize_frames(shape=s[:-1], n_dim=2)
    fill_frames_trans(f=frames, trans=trans)
    fill_frames_2d_theta(f=frames, theta=theta)
    return frames


def trans_rot2frame(trans=None, rot=None, mode=None):
    """
    pos:
    rot: theta, xcos_ysin, euler, quat
    """
    if trans is None:
        s = rot.shape
        n_dim = __rot2n_dim_dict[s[-1]]
    else:
        s = trans.shape
        n_dim = s[-1]

    frames = initialize_frames(shape=s[:-1], n_dim=n_dim)

    fill_frames_trans(f=frames, trans=trans)
    fill_frames_rot(f=frames, rot=rot, mode=mode)

    return frames


# Sampling matrix and quaternions
def sample_quaternions(shape=None):
    """
    Effective Sampling and Distance Metrics for 3D Rigid Body Path Planning, James J. Kuffner (2004)
    https://ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf
    """
    s = np.random.random(shape)
    sigma1 = np.sqrt(1 - s)
    sigma2 = np.sqrt(s)

    theta1 = np.random.uniform(0, 2 * np.pi, shape)
    theta2 = np.random.uniform(0, 2 * np.pi, shape)

    w = np.cos(theta2) * sigma2
    x = np.sin(theta1) * sigma1
    y = np.cos(theta1) * sigma1
    z = np.sin(theta2) * sigma2
    return np.stack([w, x, y, z], axis=-1)


def sample_matrix(shape=None):
    quat = sample_quaternions(shape=shape)
    return quaternions2matrix(quat=quat)


def sample_frames(x_low, x_high, shape=None):
    n_dim = len(x_low)
    frames = initialize_frames(shape=shape, n_dim=n_dim)

    fill_frames_trans(f=frames, trans=random_uniform_ndim(low=x_low, high=x_high, shape=shape))
    if n_dim == 2:
        fill_frames_rot(f=frames, rot=np.random.uniform(low=0, high=2 * np.pi, size=shape), mode=__2d_theta)
    elif n_dim == 3:
        fill_frames_rot(f=frames, rot=sample_quaternions(shape=shape), mode=__3d_quat)
    else:
        raise ValueError

    return frames


def sample_matrix_noise(shape, scale=0.01, mode='constant'):
    rv = sample_points_on_sphere_3d(shape)

    rv *= noise(shape=rv.shape[:-1], scale=scale, mode=mode)[..., np.newaxis]

    return rotvec2matrix(rotvec=rv)


def apply_noise(f, trans, rot, mode='normal'):
    s = tuple(np.array(np.shape(f))[:-2])

    frame2 = f.copy()
    frame2[..., :3, 3] += noise(shape=s + (3,), scale=trans, mode=mode)
    frame2[..., :3, :3] = frame2[..., :3, :3] @ sample_matrix_noise(shape=s, scale=rot, mode=mode)
    return frame2


def round_matrix(matrix, decimals=0):
    """Round matrix to degrees
    See numpy.round for more infos
    decimals=+2: 123.456 -> 123.45
    decimals=+1: 123.456 -> 123.4
    decimals= 0: 123.456 -> 123.0
    decimals=-1: 123.456 -> 120.0
    decimals=-2: 123.456 -> 100.0
    """
    euler = matrix2euler(matrix)
    euler = np.rad2deg(euler)
    euler = np.round(euler, decimals=decimals)
    euler = np.deg2rad(euler)
    return euler2matrix(euler)


# Fill frames
def fill_frames_diag(frames):
    for i in range(frames.shape[-1]):
        frames[..., i, i] = 1


def fill_frames_2d_sc(*, f, sin, cos):
    f[..., 0, 0] = cos
    f[..., 0, 1] = -sin
    f[..., 1, 0] = sin
    f[..., 1, 1] = cos


def fill_frames_2d_theta(f, theta):
    sin, cos = np.sin(theta[..., 0]), np.cos(theta[..., 0])
    fill_frames_2d_sc(f=f, sin=sin, cos=cos)


def fill_frames_jac_2d_sc(j, sin, cos):
    fill_frames_2d_sc(f=j, sin=cos, cos=-sin)


def fill_frames_jac_2d_theta(j, theta):
    if theta is not None:
        fill_frames_jac_2d_sc(j=j, sin=np.sin(theta), cos=np.cos(theta))


def fill_frames_2d_xcos_ysin(f, xcos_ysin):
    f[..., :-1, 0] = xcos_ysin
    f[..., 0, 1] = -xcos_ysin[..., 1]
    f[..., 1, 1] = xcos_ysin[..., 0]


def fill_frames_2d_xy(*, f, x=None, y=None):
    if x is not None:
        f[..., 0, -1] = x
    if y is not None:
        f[..., 1, -1] = y


def fill_frames_2d(*, f, sin, cos, x=None, y=None):
    fill_frames_2d_sc(sin=sin, cos=cos, f=f)
    fill_frames_2d_xy(x=x, y=y, f=f)


def fill_frames_trans(f, trans=None):
    if trans is not None:
        f[..., :-1, -1] = trans


def fill_frames_rot(*, f, rot, mode):

    n_dim = f.shape[-1] - 1
    if rot is None:
        fill_frames_diag(frames=f)
        return

    if mode is None:
        mode = __n_dim2rot_mode_default_dict[rot.shape[-1]]

    if mode == __3d_quat and n_dim == 3:
        f[..., :-1, :-1] = quaternions2matrix(quat=rot)

    elif mode == __3d_euler and n_dim == 3:
        f[..., :-1, :-1] = euler2matrix(euler=rot)

    elif mode == __3d_rotvec and n_dim == 3:
        f[..., :-1, :-1] = rotvec2matrix(rotvec=rot)

    elif mode == __2d_xcos_ysin and n_dim == 2:
        fill_frames_2d_xcos_ysin(f=f, xcos_ysin=rot)

    elif mode == __2d_theta and n_dim == 2:
        fill_frames_2d_theta(f=f, theta=rot)

    else:
        raise ValueError(f"Unknown rotation type {mode}")


def frames_jac_2d_theta(theta):
    frames_jac = initialize_frames(shape=np.shape(theta)[:-1], n_dim=2, mode='zero')
    fill_frames_jac_2d_theta(j=frames_jac, theta=theta[..., 0])
    return frames_jac


# DH
def frame_from_dh(q, d, theta, a, alpha):
    """Craig
    From wikipedia (https://en.wikipedia.org/wiki/Denavit–Hartenberg_parameters):
        d: offset along previous z to the common normal
        theta: angle about previous z, from old x to new x
        r: length of the common normal (aka a, but if using this notation, do not confuse with alpha)
           Assuming a revolute joint, this is the radius about previous z
        alpha: angle about common normal, from old z axis to new z axis
    """

    cos_th = np.cos(theta + q)
    sin_th = np.sin(theta + q)
    cos_al = np.cos(alpha)
    sin_al = np.sin(alpha)

    return np.array([[cos_th, -sin_th, 0., a],
                     [cos_al * sin_th, cos_al * cos_th, -sin_al, -d * sin_al],
                     [sin_al * sin_th, sin_al * cos_th, cos_al, d * cos_al],
                     [0, 0, 0, 1]])


def __frames_dh_4x4():
    pass


def frame_from_dh2(q, d, theta, a, alpha):

    cos_th = np.cos(theta + q)
    sin_th = np.sin(theta + q)
    cos_al = np.cos(alpha)
    sin_al = np.sin(alpha)

    n = max_size(q, d, theta, a, alpha)
    frames = np.zeros((n, 4, 4))

    frames[:, 0, 0] = cos_th
    frames[:, 0, 1] = -sin_th
    frames[:, 0, 3] = a
    frames[:, 1, 0] = cos_al * sin_th
    frames[:, 1, 1] = cos_al * cos_th
    frames[:, 1, 2] = -sin_al
    frames[:, 1, 3] = -d * sin_al
    frames[:, 2, 0] = sin_al * sin_th
    frames[:, 2, 1] = sin_al * cos_th
    frames[:, 2, 2] = cos_al
    frames[:, 2, 3] = d * cos_al
    frames[:, 3, 3] = 1
    return frames


def frame_from_dh_2d(q, theta, a):
    cos_th = np.cos(theta + q)
    sin_th = np.sin(theta + q)

    return np.array([[cos_th, -sin_th, a],
                     [sin_th, cos_th, 0],
                     [0, 0, 1]])
