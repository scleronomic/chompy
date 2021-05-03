import numpy as np
from Kinematic import frames
from Kinematic.Robots import Robot


class SingleSphere(Robot):
    def __init__(self, n_dim, radius=0.3):
        self.id = f'SingleSphere0{n_dim}'
        self.n_dim = n_dim
        self.n_dof = self.n_dim

        self.f_world_robot = None
        self.infinity_joints = np.zeros(self.n_dof, dtype=bool)
        self.limits = np.zeros((n_dim, 2))
        self.limits[:, 1] = 10

        self.spheres_pos = np.zeros((1, self.n_dim + 1))
        self.spheres_pos[:, -1] = 1
        self.spheres_rad = np.full(1, fill_value=radius)
        self.spheres_frame_idx = np.zeros(1, dtype=int)

        self.n_frames = 1
        # self.next_frame_idx = np.array([-1])
        # self.prev_frame_idx = np.array([-1])
        # self.joint_frame_idx = np.zeros((0,))
        # self.joint_frame_influence = np.ones((0, 1))

    def get_frames(self, q):
        f = frames.initialize_frames(shape=q.shape[:-1] + (self.n_frames,), n_dim=self.n_dim, mode='eye')
        f[..., :-1, -1] += q[..., np.newaxis, :]
        return f

    def get_frames_jac(self, q):
        f = self.get_frames(q)
        j = frames.initialize_frames(shape=q.shape[:-1] + (self.n_dof, self.n_frames), n_dim=self.n_dim, mode='zero')
        _fill_frames_jac__dx(j=j, n_dim=self.n_dim)
        return f, j


class SingleSphere02(SingleSphere):
    def __init__(self, radius):
        super().__init__(n_dim=2, radius=radius)


class SingleSphere03(SingleSphere):
    def __init__(self, radius):
        super().__init__(n_dim=3, radius=radius)


def _fill_frames_jac__dx(j, n_dim):
    """
    Assume that the dof xy(z) are the first 2(3)
    """
    for i in range(n_dim):
        j[:, :, i, :, i, -1] = 1
