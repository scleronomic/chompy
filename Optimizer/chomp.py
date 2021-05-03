from Kinematic import forward

from Optimizer import length as o_len
from Optimizer import obstacle_collision as o_oc


def __perform_weighting(weighting,
                        oc=0, length=0):
    return (weighting.collision * oc +
            weighting.length * length)


def chomp_cost(q, par,
               return_separate=False):
    x_spheres = forward.get_x_spheres_substeps(q=q, n=par.oc.n_substeps, robot=par.robot)

    # Obstacle Collision
    if par.planning.obstacle_collision:
        oc_cost = o_oc.oc_cost_w_length(x_spheres=x_spheres, oc=par.oc)
    else:
        oc_cost = 0

    # Length
    len_cost = o_len.q_cost(q=q, joint_weighting=par.weighting.joint_motion, infinity_joints=par.robot.infinity_joints)

    if return_separate:
        return len_cost, oc_cost
    else:

        return __perform_weighting(weighting=par.weighting, oc=oc_cost, length=len_cost)


def chomp_grad(q, par,
               jac_only=True, return_separate=False):
    x, dx_dq = forward.get_x_spheres_substeps_jac(q=q, robot=par.robot, n=par.oc.n_substeps)
    dx_dq = dx_dq[..., 1:, :, :, :]  # ignore start point, should be feasible anyway

    # Obstacle Collision
    if par.planning.obstacle_collision:
        oc_jac = o_oc.oc_cost_w_length_grad(x_spheres=x, dx_dq=dx_dq, oc=par.oc,
                                            jac_only=jac_only, include_end=par.planning.include_end)

    else:
        if jac_only:
            oc_jac = 0
        else:
            oc_jac = 0, 0

    # Path Length
    len_jac = o_len.q_cost_grad(q=q,
                                joint_weighting=par.weighting.joint_motion,
                                infinity_joints=par.robot.infinity_joints,
                                include_end=par.planning.include_end, jac_only=jac_only)

    if return_separate:
        if jac_only:
            return len_jac, oc_jac
        else:
            return (len_jac[0],oc_jac[0]), (len_jac[1], oc_jac[1])

    else:

        if jac_only:
            return __perform_weighting(weighting=par.weighting, oc=oc_jac, length=len_jac)
        else:
            return (__perform_weighting(weighting=par.weighting, oc=oc_jac[0], length=len_jac[0]),
                    __perform_weighting(weighting=par.weighting, oc=oc_jac[1], length=len_jac[1]))
