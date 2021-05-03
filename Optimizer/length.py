import numpy as np
from Optimizer import path


def q_cost(q, infinity_joints, joint_weighting):
    """
    Calculate the squared euclidean distance of each step and sum up over the path and over the joints.
    Consider infinity joints for the step shape calculation and make a weighting between the different joints
    Because the squared step shape is considered it is not necessary to handle cartesian coordinates x,y,z
    differently from the joints.
    """

    q_steps = np.diff(q, axis=-2)
    q_steps = path.inf_joint_wrapper(x=q_steps, inf_bool=infinity_joints)
    o = (q_steps ** 2).sum(axis=-2)  # Sum over path of the quadratic step length for each dof

    # Weighted sum over the joints
    o = 0.5 * (o * joint_weighting).mean(axis=-1)

    return o


def q_cost_grad(q, infinity_joints, joint_weighting,
                include_end=False, jac_only=True):
    """
    Jacobian for 'path_length(x, squared=True)' with respect to the coordinate (j) of
    a way point (o).
    length_squared = ls = 1/2 * Sum(o=1, n) |p_i - p_(o-1)|**2
    # Start- and endpoint: x 1
    # Points in the middle, with two neighbours: x 2
    """

    q_steps = path.get_steps(x=q, infinity_joints=infinity_joints)

    if include_end:
        do_dq = q_steps
        do_dq[..., :-1, :] -= q_steps[..., +1:, :]
    else:
        do_dq = q_steps[..., :-1, :] - q_steps[..., +1:, :]

    do_dq *= joint_weighting / q.shape[-1]  # Broadcasting starts from trailing dimension (numpy broadcasting)

    if jac_only:
        return do_dq
    else:
        o = (q_steps ** 2).sum(axis=-2)
        o = 0.5 * (o * joint_weighting).mean(axis=-1)
        # raise NotImplementedError
        # If end position of optimization the length norm can not be computed beforehand
        # len_cost = length_cost_a(x=x, a=a,
        #                          n_joints=n_joints, fixed_base=fixed_base, infinity_joints=infinity_joints,
        #                          length_a_penalty=length_a_penalty, length_norm=length_norm,
        #                          n_dim=n_dim, n_samples=n_samples)
        return o, do_dq


def x_cost(f):
    x = f[..., :-1, -1]
    x_steps = np.diff(x, axis=1)
    x_steps_norm2 = 0.5*(x_steps**2).sum(axis=(1, 2, 3))
    return x_steps_norm2


def x_cost_grad(f, j):
    """
    Sum_i ( |x_i - x_(i-1)|^2 ) , i =1,...n
    Sum_i ( |x_i - x_(i-1)|^2 ) , i =1,...n
    /∂xi = d_(i-(i-1)) - d_((i+1)-i)
    /∂qi = /∂xi * ∂xi/∂qi
    """
    x = f[..., :-1, -1]
    j = j[..., :-1, -1]

    x_steps = np.diff(x, axis=1)
    do_dq = x_steps[:, :, np.newaxis, :, :]
    do_dq[:, :-1, :, :] -= do_dq[:, 1:, :, :]
    do_dq = j[:, 1:, :, :, ] * do_dq
    do_dq = do_dq.sum(axis=(-2, -1))
    return do_dq


def close_to_pose_cost(q, q_close, joint_weighting, infinity_joints):
    o = path.inf_joint_wrapper(x=q_close - q, inf_bool=infinity_joints) ** 2
    o = 0.5 * (o * joint_weighting).sum(axis=(-2, -1))
    return o


def close_to_pose_grad(q, q_close, joint_weighting, infinity_joints):
    return -path.inf_joint_wrapper(x=q_close - q, inf_bool=infinity_joints) * joint_weighting
