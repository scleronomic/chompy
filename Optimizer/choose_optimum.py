import numpy as np

from Optimizer import feasibility_check, length


def get_feasible_optimum(q, par,
                         status=None, verbose=0,
                         return_cost=False):
    if status is None:
        status = feasibility_check(q=q, par=par)

    feasible = status >= 0

    if verbose > 0:
        print("{} of the {} solutions were feasible -> choose the best".format(np.sum(feasible), np.size(feasible)))
    if np.sum(feasible) == 0:
        status_unique, status_count = np.unique(status, return_counts=True)
        most_common_error = status_unique[np.argmax(status_count)]
        if return_cost:
            return q[0:0], most_common_error, np.inf
        else:
            return q[0:0], most_common_error

    q = q[feasible]

    len_cost = length.q_cost(q=q, joint_weighting=par.weighting.joint_motion, infinity_joints=par.robot.infinity_joints)

    min_idx = np.argmin(len_cost)

    if return_cost:
        return q[min_idx:min_idx + 1, ...], 0, len_cost[min_idx]
    else:
        return q[min_idx:min_idx + 1, ...], 0
