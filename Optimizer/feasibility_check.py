import numpy as np

from Kinematic import forward
from Optimizer.obstacle_collision import oc_check


def box_constraints_limits_check(x, limits):
    n_samples, n_wp, n_dof = x.shape

    below_lower = x < limits[:, 0]
    above_upper = x > limits[:, 1]

    outside_limits = np.logical_or(below_lower, above_upper)
    outside_limits = outside_limits.reshape(n_samples, -1, n_dof)
    outside_limits = outside_limits.sum(axis=1)

    # Check the feasibility of each sample
    outside_limits = outside_limits.sum(axis=1)
    feasible = outside_limits == 0

    return feasible


def feasibility_check(q, par, verbose=0):
    n_samples = q.shape[0]

    frames, x_spheres = forward.get_x_spheres_substeps(q=q, robot=par.robot, n=par.oc.n_substeps_check,
                                                       return_frames2=True)

    # Obstacle Collision
    if par.check.obstacle_collision:
        feasible_oc = oc_check(x_spheres=x_spheres, oc=par.oc, verbose=verbose)
    else:
        feasible_oc = np.ones(n_samples, dtype=bool)

    # Joint Limits
    if par.check.limits:
        feasible_limits = box_constraints_limits_check(x=q, limits=par.robot.limits)
    else:
        feasible_limits = np.ones(n_samples, dtype=bool)

    # Override the status and return the smallest error value
    status = np.ones(n_samples, dtype=int)
    status[~feasible_limits] = -2
    status[~feasible_oc] = -1

    return status
