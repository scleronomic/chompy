import numpy as np

from Kinematic.Robots import SingleSphere02
from Optimizer import feasibility_check
from parameter import Parameter
import plotting

radius = 0.3  # Size of the robot [m]
robot = SingleSphere02(radius=radius)
par = Parameter(robot=robot, obstacle_img='perlin')

n = 200
q = robot.sample_q(n)
status = feasibility_check(q[:, np.newaxis, :], par)

fig, ax = plotting.new_world_fig(limits=par.world.limits, title='Multi-Starts')
plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)

plotting.plot_circles(x=q[status == +1], r=radius, ax=ax, color='blue', alpha=0.5)
plotting.plot_circles(x=q[status == -1], r=radius, ax=ax, color='red', alpha=0.5)
