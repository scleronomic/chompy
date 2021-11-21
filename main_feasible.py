import numpy as np

from Kinematic.Robots import SingleSphere02
from Optimizer import feasibility_check, path
import plotting
from parameter import Parameter

np.random.seed(2)

radius = 0.3  # Size of the robot [m]
robot = SingleSphere02(radius=radius)
par = Parameter(robot=robot, obstacle_img='perlin')

par.world.limits = np.array([[0, 10],
                             [0, 10]])
par.robot.limits = par.world.limits  # special case for Single Sphere Robot

par.check.obstacle_collision = True

# Create straight lines
n_samples = 100
n_waypoints = 20
q = robot.sample_q((n_samples, 2))
q = path.get_substeps(q, n=n_waypoints, infinity_joints=par.robot.infinity_joints)


status = feasibility_check(q=q, par=par)
print((status > 0).sum())
q = q[status > 0]

fig, ax = plotting.new_world_fig(limits=par.world.limits, title='Multi-Starts')
plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
for qq in q:
    plotting.plot_x_path(x=qq, r=par.robot.spheres_rad, ax=ax, marker='o', alpha=0.5, color='k')
