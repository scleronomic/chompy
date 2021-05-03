from Kinematic.Robots import StaticArm
from GridWorld import create_perlin_image
from parameter import Parameter, initialize_oc
from Optimizer.chomp import chomp_cost, chomp_grad
robot = StaticArm(n_dof=5, limb_lengths=0.4, radius=0.1)
par = Parameter(robot=robot, obstacle_img='perlin')


# Sample random configurations
q = robot.sample_q((3, 20))  # 3 paths with 20 waypoints each
f = robot.get_frames(q)
x, dx_dq = robot.get_x_spheres_jac(q)

(length_cost, collision_cost), (length_jac, collision_jac) = chomp_grad(q=q, par=par, jac_only=False,
                                                                        return_separate=True)

print(length_jac.shape)
print(collision_jac.shape)
# Gradients for 3 paths with 18 way points each, the start and endpoints are part of the problem and not free variables

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