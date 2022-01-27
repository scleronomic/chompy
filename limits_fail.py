
import numpy as np
import matplotlib.pyplot as plt
from Kinematic.Robots import SingleSphere02, StaticArm
from parameter import Parameter
from Optimizer import feasibility_check
from GridWorld import create_rectangle_image
from parameter import Parameter, initialize_oc
import plotting

np.random.seed(16)

robot = StaticArm(n_dof=3, limb_lengths=0.5, radius=0.1)
par = Parameter(robot=robot, obstacle_img=None)

# TODO if you uncomment this, it fails
# par.world.limits = np.array([[-2, 2],
#                              [-2, 2]])


# Sample random configurations
n = 20
q = robot.sample_q(n)

n_obstacles = 3
min_max_obstacle_size_voxel = [3, 15]
n_voxels = (64, 64)
img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
img[:] = 0
img[40:, :] = 1
img[:, 40:] = 1
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

q[:] = 0
q[:, 0] = np.linspace(-np.pi, +np.pi, num=n, endpoint=False)

# TODO feasibility_check expects a 3 dim array (samples x waypoints x dof)
#   if waypoints == 1, it just checks the single pose
#   if waypoints > 1, it checks the whole path between the waypoints, including substeps

status = feasibility_check(q[:, np.newaxis, :], par)
print(status)

fig, ax = plotting.new_world_fig(limits=par.world.limits)
plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)


for i in range(n):
    plotting.plot_spheres(q=q[i], robot=robot, ax=ax, color='green' if status[i] == 1 else 'red')
    plotting.plot_arm(ax=ax, q=q[i], robot=robot, zorder=10)

