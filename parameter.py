import copy
import numpy as np

from Kinematic.Robots import *
from wzk import safe_scalar2array, safe_unify


def str2robot(robot):
    n_dof = int(robot[-2:])
    robot = robot[:-2]
    a = robot in globals()
    print(a)
    if robot in globals():
        robot = eval("{}(n_dof={})".format(robot, n_dof))
    else:
        robot = eval("{}{:0>2}()".format(robot, n_dof))

    return robot


class CopyableObject(object):
    __slots__ = ()

    def copy(self):
        return copy.copy(self)


class GradientDescent(CopyableObject):
    __slots__ = ('n_steps',              # int                 | Number of iterations
                 'step_size',            # float                |
                 'adjust_step_size',     # bool                |
                 'staircase_iteration',  # bool                |
                 'grad_tol',             # float[n_steps]       |
                 'callback',             # fun()               |
                 'prune_limits',         # fun()               |
                 'n_processes',          # int                 |
                 'hesse_inv',            # float[n_var][n_var]  |
                 'hesse_weighting',      # float[n_steps]       |
                 'return_x_list'         # bool                |  is this a suitable parameter? not really
                 )

    def __init__(self):
        self.n_steps = 100
        self.step_size = 0.001
        self.adjust_step_size = True
        self.staircase_iteration = False
        self.grad_tol = 0.1

        self.n_processes = 1
        self.callback = None
        self.prune_limits = None
        self.hesse_inv = None
        self.hesse_weighting = 0

        self.return_x_list = False


class World(object):
    __slots__ = ('n_dim',           # int             | Number of spacial dimensions, 2D/3D
                 'limits',          # float[n_dim][2] | Lower and upper boundary of each spatial dimension [m]
                 'size',            # float[n_dim]    | Size of the world in each dimension in meters
                 'n_voxels',        # int[n_dim]      | Size of the world in each dimension in voxels
                 'voxel_size',      # float           | Side length of one pixel/voxel [m]
                                    #                 | (depends on limits and n_voxels, make sure all voxels are cubes)
                 'lower_left',      #                 | offset of the world frame, might be necesaary
                                    #                 | when switching between multiple worlds
                 )

    def __init__(self, n_dim, limits, n_voxels=64, voxel_size=None, lower_left=0,
                 robot=None):

        self.n_dim = n_dim
        self.lower_left = lower_left

        if isinstance(robot, StaticArm):
            l = robot.limb_lengths * (robot.n_dof+0.5)
            limits = np.array([[-l, +l],
                               [-l, +l]])
        if limits is None:
            self.limits = np.zeros((n_dim, 2))
            self.limits[:, 1] = 10
        else:
            self.limits = limits

        self.size = np.diff(self.limits, axis=-1)[:, 0]

        if n_voxels is None:
            self.n_voxels = (self.size / self.voxel_size).astype(int)
        else:
            self.n_voxels = safe_scalar2array(n_voxels, shape=self.n_dim)

        if voxel_size is None:
            self.voxel_size = safe_unify(self.size / n_voxels)


class ObstacleCollision(object):
    __slots__ = (
        'img',                     #
        'spheres_rad',             #
        'active_spheres',          # bool[n_spheres]  | Indicates which spheres are active for collision
        'n_substeps',              # int              | Number of substeps used in the cost function
        'n_substeps_check',        # int              | Number of substeps used in its derivative
        'dist_fun',                # fun()            |
        'dist_grad',               # fun()            |
        'edt_interp_order_cost',   # int              | Interp. order for extracting the values from the edt(0)
        'edt_interp_order_grad',   # int              | ... from the spacial derivative of the edt (1)
        'eps_dist_cost',           # float            | additional safety length for which the cost is smoothed
        #                                             | out quadratically [m] (0.05)
        'dist_threshold'           # float            |
        )

    def __init__(self,
                 edt_interp_order_cost=1, edt_interp_order_grad=1,
                 n_substeps=1, n_substeps_check=1,
                 eps_dist_cost=0.05, dist_threshold=-0.005):

        self.img = None
        self.edt_interp_order_cost = edt_interp_order_cost
        self.edt_interp_order_grad = edt_interp_order_grad
        self.n_substeps = n_substeps
        self.n_substeps_check = n_substeps_check
        self.eps_dist_cost = eps_dist_cost
        self.dist_threshold = dist_threshold


class CheckingType(object):
    __slots__ = ('obstacle_collision',  # bool |
                 'self_collision',      # bool |
                 'center_of_mass',      # bool |
                 'limits',              # bool |
                 'tcp'                  # bool |
                 )

    def __init__(self, oc=True, sc=False,
                com=False, limits=True, tcp=False):
        self.obstacle_collision = oc
        self.self_collision = sc
        self.center_of_mass = com
        self.limits = limits
        self.tcp = tcp


class PlanningType(object):
    """
    Boolean flags for turning on/off different types of constraints/objectives
    """
    __slots__ = ('obstacle_collision',  # bool |
                 'self_collision',      # bool |
                 'length',              # bool |
                 'center_of_mass',      # bool |
                 'tcp',                 # bool |
                 # 'base_rotation',       # bool |
                 # 'include_start',       # bool |  Start is always fixed, even in the context of MPC, the start of the
                 #                               | next planning horizon is given, otherwise it becomes to convoluted
                 'include_end'          # bool |
                 )

    def __init__(self, length=True, oc=True, sc=False, tcp=False, com=False, include_end=False):
        self.length = length
        self.obstacle_collision = oc
        self.self_collision = sc
        self.tcp = tcp
        self.center_of_mass = com
        self.include_end = include_end


class Weighting(CopyableObject):
    __slots__ = ('length',                        # float[gd.n_steps] |
                 'collision',                     # float[gd.n_steps] |
                 'tcp',                           # float[gd.n_steps] |
                 'tcp__rot_vs_loc',               # float[gd.n_steps] |
                 'center_of_mass',
                 'joint_motion',                  # float[shape.n_dof]|
                 'beeline_joints',                # float             |
                 'beeline_spheres',               # float             |
                 'beeline_collision_pairs'           # float             |
                 )

    def __init__(self, length=1., collision=1000.,
                 tcp=0, tcp__rot_vs_loc=0.5,
                 com=0,
                 joint_motion=1,
                 beeline_joints=1, beeline_spheres=1):
        self.length = length
        self.collision = collision
        self.tcp = tcp
        self.tcp__rot_vs_loc = tcp__rot_vs_loc
        self.center_of_mass = com
        self.beeline_joints = beeline_joints
        self.beeline_spheres = beeline_spheres
        self.joint_motion = joint_motion

    def __at_idx(self, v, i):
        x = getattr(self, v)
        if np.size(x) > 1:
            setattr(self, v, x[i])

    def __at_range(self, v, i, j):
        x = getattr(self, v)
        if np.size(x) > 1:
            setattr(self, v, x[i:j])

    def at_idx(self, i):
        new_weighting = self.copy()
        for v in ['length', 'collision']:
            self.__at_idx(v=v, i=i)

        return new_weighting

    def at_range(self, start, stop):
        new_weighting = self.copy()
        for v in ['length', 'collision']:
            self.__at_range(v=v, i=start, j =stop)

        return new_weighting


class Parameter(object):
    __slots__ = (
                 'robot',       # Robot, Kinematic + Sphere Model
                 'world',       # World, Limits + Voxels
                 'oc',          # Obstacle Collision
                 'sc',          # Self-Collision
                 'com',         # Center of Mass
                 'tcp',         # Tool Center Point
                 'pbp',         # Pass by Points
                 'planning',    # Planning options
                 'weighting',   # Weighting factors between the different parts of the cost-function
                 'check',       #
    )

    def __init__(self, robot, obstacle_img, sc_mode='spheres'):
        if isinstance(robot, str):
            self.robot = str2robot(robot)
        elif isinstance(robot, Robot):
            self.robot = robot
        else:
            raise ValueError

        self.world = World(n_dim=self.robot.n_dim, limits=None, robot=self.robot)
        self.planning = PlanningType()
        self.check = CheckingType()
        self.oc = ObstacleCollision()
        self.weighting = Weighting()

        initialize_oc(oc=self.oc, robot=self.robot, world=self.world, obstacle_img=obstacle_img)


def initialize_oc(oc, world, robot,
                  obstacle_img=None,
                  dist_img=None, dist_img_grad=None,
                  limits=None):

    from GridWorld import templates, obstacle_distance
    if limits is not None:
        world.limits = limits

    if not isinstance(obstacle_img, np.ndarray):
        obstacle_img = templates.create_template(n_voxels=world.n_voxels, world=obstacle_img)

    try:
        a = oc.active_spheres
    except AttributeError:
        oc.active_spheres = np.ones(len(robot.spheres_rad), dtype=bool)

    oc.spheres_rad = robot.spheres_rad
    oc.img = obstacle_img

    oc.dist_fun, oc.dist_grad = \
        obstacle_distance\
            .obstacle_img2funs(img=obstacle_img, add_boundary=True,
                               dist_img=dist_img, dist_img_grad=dist_img_grad,
                               voxel_size=world.voxel_size, lower_left=world.limits[:, 0],
                               interp_order_dist=oc.edt_interp_order_cost,
                               interp_order_grad=oc.edt_interp_order_grad)

    return obstacle_img



