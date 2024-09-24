import numpy as np
from Kinematic.Robots import StaticArm
from GridWorld import create_perlin_image
from Optimizer.chomp import chomp_cost, chomp_grad
from parameter import Parameter, initialize_oc


robot = StaticArm(n_dof=5, limb_lengths=0.4, radius=0.1)
par = Parameter(robot=robot, obstacle_img='perlin')


# Sample random configurations
q = robot.sample_q((3, 20))  # 3 random paths with 20 waypoints each
f = robot.get_frames(q)
x, dx_dq = robot.get_x_spheres_jac(q)

(length_cost, collision_cost), (length_jac, collision_jac) = chomp_grad(q=q, par=par, jac_only=False,
                                                                        return_separate=True)

print('q:            ', q.shape)
print('length_jac:   ', length_jac.shape)
print('collision_jac:', collision_jac.shape)
# Gradients for 3 paths with 18 way points each, the start and endpoints are part of the problem and not free variables
print()
print(np.round(length_jac, 3))

import plotting
fig, ax = plotting.new_world_fig(limits=par.world.limits,)
plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
plotting.plot_spheres(q=q[0, 1], robot=robot, ax=ax)

import matplotlib.pyplot as plt
plt.show()

# Create new image and get new gradients
img = create_perlin_image(n_voxels=par.world.n_voxels)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

(length_cost, collision_cost), (length_jac, collision_jac) = chomp_grad(q=q, par=par, jac_only=False,
                                                                        return_separate=True)


print(length_cost)
print(length_jac.shape)
print(collision_cost)
print(collision_jac.shape)


