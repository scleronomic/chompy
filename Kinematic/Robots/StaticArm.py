import numpy as np

from Kinematic.Robots import Robot
from Kinematic import forward, frames, chain

from wzk import safe_scalar2array


def get_world_limits(n_dof, limb_lengths):
    world_limits = np.empty((2, 2))
    world_limits[:, 0] = -(n_dof + 0.5) * limb_lengths
    world_limits[:, 1] = +(n_dof + 0.5) * limb_lengths
    return world_limits


def get_arm2d_spheres(n_links, spheres_per_link, limb_lengths):
    """radius * 2 * np.cos(np.arcsin(1 / 2))"""
    n_dim = 2
    n_dim1 = n_dim+1

    spheres_per_link, limb_lengths = safe_scalar2array(spheres_per_link, limb_lengths, shape=n_links)

    n_spheres = np.sum(spheres_per_link) + 1
    spheres_pos = np.zeros((n_spheres, n_dim1))
    spheres_pos[:, -1] = 1
    spheres_pos[1:, 0] = np.concatenate([np.linspace(0, ll, spl+1)[1:] for ll, spl
                                         in zip(limb_lengths, spheres_per_link)])

    spheres_frame_idx = np.zeros(n_spheres, dtype=int)
    spheres_frame_idx[1:] = np.concatenate([np.full(spl, i) for i, spl in enumerate(spheres_per_link)])

    return spheres_pos, spheres_frame_idx


class StaticArm(Robot):
    def __init__(self, n_dof, limb_lengths=0.5, radius=0.1):
        self.id = "StaticArm{:0>2}".format(n_dof)
        self.n_dim = 2
        self.n_dof = n_dof

        self.f_world_robot = None
        self.infinity_joints = np.zeros(self.n_dof, dtype=bool)
        self.limits = np.repeat(np.array([[-np.pi, np.pi]]), self.n_dof, axis=0)

        self.limb_lengths = limb_lengths
        self.spheres_pos, self.spheres_frame_idx = \
            get_arm2d_spheres(n_links=self.n_dof, spheres_per_link=3, limb_lengths=self.limb_lengths)
        self.spheres_rad = safe_scalar2array(radius, shape=len(self.spheres_frame_idx))
        _init_serial_kinematic(robot=self, n_dof=self.n_dof)
        self.n_frames = len(self.next_frame_idx)

    def get_frames(self, q):
        f = frames.initialize_frames(shape=q.shape[:-1] + (self.n_frames,), n_dim=self.n_dim, mode='hm')

        sin, cos = np.sin(q), np.cos(q)
        _fill_frames(sin=sin, cos=cos, f=f, limb_lengths=self.limb_lengths)
        forward.combine_frames(f=f, prev_frame_idx=self.prev_frame_idx)
        f = frames.apply_eye_wrapper(f=f, possible_eye=self.f_world_robot)
        return f

    def get_frames_jac(self, q):
        f = frames.initialize_frames(shape=q.shape[:-1] + (self.n_frames,), n_dim=self.n_dim, mode='hm')
        j = frames.initialize_frames(shape=q.shape[:-1] + (self.n_dof, self.n_frames), n_dim=self.n_dim, mode='zero')

        sin, cos = np.sin(q), np.cos(q)
        _fill_frames_jac(sin=sin, cos=cos, f=f, j=j,
                         joint_frame_idx=self.joint_frame_idx, limb_lengths=self.limb_lengths)

        dh_dict = forward.create_frames_dict(f=f, nfi=self.next_frame_idx)
        forward.combine_frames_jac(j=j, d=dh_dict, robot=self)
        f = dh_dict[0]
        f = frames.apply_eye_wrapper(f=f, possible_eye=self.f_world_robot)
        return f, j


class StaticTree(Robot):

    def __init__(self):
        self.next_frame_idx = [1, [2, 4], 3, -1, 5, -1]
        self.prev_frame_idx = chain.next2prev_frame_idx(nfi=self.next_frame_idx)
        self.joint_frame_idx = np.arange(len(self.next_frame_idx))
        self.joint_frame_influence = chain.influence_frames_frames2joints_frames(jfi=self.joint_frame_idx,
                                                                                 nfi=self.next_frame_idx)

        self.n_dim = 2
        self.n_dof = len(self.next_frame_idx)

        self.f_world_robot = None

        self.limb_lengths = 0.5
        self.infinity_joints = np.ones(self.n_dof, dtype=bool)
        self.limits = np.repeat(np.array([[-np.pi, np.pi]]), self.n_dof, axis=0)

        self.spheres_pos, self.spheres_frame_idx = \
            get_arm2d_spheres(n_links=self.n_dof, spheres_per_link=3, limb_lengths=self.limb_lengths)

        self.spheres_rad = safe_scalar2array(0.1, shape=len(self.spheres_frame_idx))

    def get_frames_jac(self, q):
        raise NotImplementedError

    def get_frames(self, q):
        raise NotImplementedError


def _fill_frames(f,
                 sin, cos,
                 limb_lengths):
    frames.fill_frames_2d_sc(sin=sin, cos=cos, f=f[..., :-1, :, :])
    frames.fill_frames_2d_xy(x=limb_lengths, f=f[..., 1:, :, :])
    frames.fill_frames_diag(f[..., -1, :, :])


def _fill_jacs(j,
               sin, cos,
               joint_frame_idx):
    n_dof = j.shape[-4]
    for i in range(n_dof):
        frames.fill_frames_jac_2d_sc(j=j[..., i, joint_frame_idx[i], :, :], sin=sin[..., i], cos=cos[..., i])


def _fill_frames_jac(f, j,
                     sin, cos,
                     joint_frame_idx, limb_lengths):
    _fill_frames(sin=sin, cos=cos, f=f, limb_lengths=limb_lengths)
    _fill_jacs(sin=sin, cos=cos, j=j, joint_frame_idx=joint_frame_idx)


def _init_serial_kinematic(robot, n_dof):
    """
    Include a TCP after the last frame
    """
    nfi = np.arange(1, n_dof + 2)
    nfi[-1] = -1
    jf_idx = np.arange(n_dof)
    pfi = chain.next2prev_frame_idx(nfi=nfi)
    ff_inf = chain.next_frame_idx2influence_frames_frames(nfi=nfi)
    jf_inf = chain.influence_frames_frames2joints_frames(nfi=nfi, jfi=jf_idx)

    robot.next_frame_idx = nfi
    robot.prev_frame_idx = pfi
    robot.joint_frame_idx = jf_idx
    robot.frame_frame_influence = ff_inf
    robot.joint_frame_influence = jf_inf
