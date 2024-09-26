import numpy as np

from Kinematic.Robots import SingleSphere02
from Optimizer import path, choose_optimum, initial_guess, gradient_descent, chomp, obstacle_collision

import plotting
from parameter import Parameter, GradientDescent

# Sample Program for showing the Optimizer Based Motion Planning (OMP) as described by CHOMP
# Most of the parameters

np.random.seed(2)

radius = 0.3  # Size of the robot [m]
robot = SingleSphere02(radius=radius)
par = Parameter(robot=robot, obstacle_img='perlin')


world_limits = np.array([[0, 10],   # x [m]
                         [0, 10]])  # y [m]
n_voxels = (64, 64)

par.robot.limits = world_limits

n_waypoints = 20  # number of the discrete way points of the trajectory
par.oc.n_substeps = 5  # number of sub steps to check for obstacle collision

# Gradient Descent
gd = GradientDescent()
gd.n_steps = 100
gd.step_size = 0.001
gd.adjust_step_size = True
gd.grad_tol = np.full(gd.n_steps, 1)
gd.n_processes = 1

# Weighting of Objective Terms, it is possible that each gradient step as a different weighting between those terms
par.weighting.length = 1
par.weighting.collision = 100

# Number of multi starts
n_multi_start_rp = [[0, 1, 2],  # how many random points for the multi-start (0 == straight line)
                    [1, 5, 4]]  # how many trials for each variation
get_q0 = initial_guess.q0_random_wrapper(robot=par.robot, n_multi_start=n_multi_start_rp,
                                         n_waypoints=n_waypoints, order_random=True, mode='inner')


# Choose sample start and goal point of the motion problem
q_start, q_end = np.array([[[1, 1]]]), np.array([[[9, 9]]])

# Get initial guesses for given start and end / ms = multistarts
q_ms = get_q0(start=q_start, end=q_end)

#####
# Perform Gradient Descent
q_ms, objective = gradient_descent.gd_chomp(q0=q_ms, q_start=q_start, q_end=q_end, gd=gd, par=par)


# Choose the optimal trajectory from all multi starts
q_ms = path.x_inner2x(inner=q_ms, start=q_start, end=q_end)
q_opt, _ = choose_optimum.get_feasible_optimum(q=q_ms, par=par, verbose=2)


# Plot multi starts and optimal solution
fig, ax = plotting.new_world_fig(limits=par.world.limits, title='Multi-Starts')
plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
for q in q_ms:
    plotting.plot_x_path(x=q, r=par.robot.spheres_rad, ax=ax, marker='o', alpha=0.5)

plotting.plot_x_path(x=q_opt, r=par.robot.spheres_rad, ax=ax, marker='o', color='k')
plotting.plt.show()