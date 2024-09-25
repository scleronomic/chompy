import numpy as np
from Kinematic import forward, frames

from Kinematic.Robots import Robot
from Kinematic.Robots.SingleSphere import _fill_frames_jac__dx
from Kinematic.Robots.StaticArm import _fill_frames, _fill_frames_jac, get_arm2d_spheres, _init_serial_kinematic

from wzk import safe_scalar2array


class MovingArm(Robot):
    def __init__(self, n_dof, limb_lengths=0.5, spheres_per_link=3, radius=0.1):
        self.id = "MovingArm{:0>2}".format(n_dof)
        self.n_dim = 2
        self.n_dof_arm = n_dof
        self.n_dof = self.n_dim + self.n_dof_arm

        self.f_world_robot = None
        self.infinity_joints = np.ones(self.n_dof, dtype=bool)
        self.infinity_joints[:self.n_dim] = False

        self.limits = np.repeat(np.array([[-np.pi, np.pi]]), self.n_dof, axis=0)
        self.limits[:self.n_dim, :] = np.array([[0, 10],
                                                [0, 10]])

        self.limb_lengths = limb_lengths

        self.spheres_pos, self.spheres_frame_idx = \
            get_arm2d_spheres(n_links=self.n_dof_arm, spheres_per_link=spheres_per_link, limb_lengths=self.limb_lengths)

        self.spheres_rad = safe_scalar2array(radius, shape=len(self.spheres_frame_idx))

        _init_serial_kinematic(robot=self, n_dof=self.n_dof_arm)
        self.n_frames = len(self.next_frame_idx)

    def q2x_q(self, xq):
        x = xq[..., :self.n_dim]
        q = xq[..., self.n_dim:]
        return x, q

    @staticmethod
    def x_q2q(self, x, q):
        xq = np.concatenate((x, q), axis=-1)
        return xq

    def get_frames(self, q):
        x, q = self.q2x_q(xq=q)
        sin, cos = np.sin(q), np.cos(q)

        f = frames.initialize_frames(shape=q.shape[:-1] + (self.n_frames,), n_dim=self.n_dim, mode='hm')
        _fill_frames(f=f,
                     sin=sin, cos=cos,
                     limb_lengths=self.limb_lengths)

        forward.combine_frames(f=f, prev_frame_idx=self.prev_frame_idx)
        f[..., :-1, -1] += x[..., np.newaxis, :]
        f = frames.apply_eye_wrapper(f=f, possible_eye=self.f_world_robot)
        return f

    def get_frames_jac(self, q):
        x, q = self.q2x_q(xq=q)
        f = frames.initialize_frames(shape=q.shape[:-1] + (self.n_frames,), n_dim=self.n_dim, mode='hm')
        j = frames.initialize_frames(shape=q.shape[:-1] + (self.n_dof, self.n_frames), n_dim=self.n_dim, mode='zero')

        sin, cos = np.sin(q), np.cos(q)
        _fill_frames_jac(f=f, j=j[..., self.n_dim:, :, :, :],
                         sin=sin, cos=cos,
                         joint_frame_idx=self.joint_frame_idx, limb_lengths=self.limb_lengths)

        d = forward.create_frames_dict(f=f, nfi=self.next_frame_idx)
        forward.combine_frames_jac(j=j[..., self.n_dim:, :, :, :], d=d, robot=self)

        f = d[0]
        f[..., :-1, -1] += x[..., np.newaxis, :]
        _fill_frames_jac__dx(j=j, n_dim=self.n_dim)
        f = frames.apply_eye_wrapper(f=f, possible_eye=self.f_world_robot)
        return f, j


class MovingArm03(Robot):
    def __init__(self):
        self.id = 'MovingArm03'
        self.n_dim = 2
        self.n_dof = 3
        self.n_frames = 1

        # Configuration Space
        self.limits = np.repeat(np.array([[-np.pi, np.pi]]), self.n_dof, axis=0)
        self.limits[:self.n_dim, :] = np.array([[0, 10],
                                                [0, 10]])
        self.infinity_joints = np.array([False, False, True], dtype=bool)
        self.f_world_robot = None

        # Sphere Model
        self.spheres_pos = np.array([[0, 0, 1],
                                     [0, 0.2, 1]])
        self.spheres_rad = np.array([0.3, 0.3])
        self.spheres_frame_idx = np.array([0, 0])

        self.next_frame_idx = np.array([-1])
        self.prev_frame_idx = np.array([-1])
        self.joint_frame_idx = np.array([0])
        self.joint_frame_influence = np.ones((1, 1))

    def get_frames(self, q):
        x, q = q[..., :self.n_dim], q[..., self.n_dim:]
        f = frames.initialize_frames(shape=q.shape[:-1] + (self.n_frames,), n_dim=self.n_dim, mode='eye')

        frames.fill_frames_2d_sc(sin=np.sin(q), cos=np.cos(q), f=f)
        f[..., :-1, -1] += x[..., np.newaxis, :]
        return f

    def get_frames_jac(self, q):
        x, q = q[..., :self.n_dim], q[..., self.n_dim:]
        sin, cos = np.sin(q), np.cos(q)
        f = frames.initialize_frames(shape=q.shape[:-1] + (self.n_frames,), n_dim=self.n_dim, mode='eye')
        j = frames.initialize_frames(shape=q.shape[:-1] + (self.n_dof, self.n_frames), n_dim=self.n_dim, mode='zero')

        frames.fill_frames_2d_sc(sin=sin, cos=cos, f=f)
        f[..., :-1, -1] += x[..., np.newaxis, :]

        frames.fill_frames_jac_2d_sc(j=j[..., 2, :, :, :], sin=sin, cos=cos)
        _fill_frames_jac__dx(j=j, n_dim=self.n_dim)

        return f, j


class Blob03(MovingArm03):
    def __init__(self, n, r0=0.1, r1=0.4, seed=None):
        super().__init__()

        self.id = 'Blob03'
        np.random.seed(seed=seed)

        v = np.random.uniform(low=-2.5, high=2.5, size=n)
        r = np.random.uniform(low=r0, high=r1, size=n)
        rr = np.array((np.roll(r, +1), r))
        rr = np.sum(rr, axis=0) - np.min(rr, axis=0) * (1 - np.sqrt(3)/2)
        v = (np.array((np.cos(v), np.sin(v), np.ones(n))) * rr).T

        v[0, :] = 0
        v = np.cumsum(v, axis=0)
        v[:, -1] = 1

        self.spheres_pos = v
        self.spheres_rad = r
        self.spheres_frame_idx = np.zeros(n, dtype=int)


class MovingTree:
    pass
