import numpy as np
from wzk import angle2minuspi_pluspi


def x2x_inner(x):
    return x[..., 1:-1, :]


def x_inner2x(*, inner, start=None, end=None):
    n_samples = inner.shape[0]

    def repeat(x):
        if x.ndim < 3:
            x = np.atleast_2d(x)[:, np.newaxis, :]
        return x.repeat(n_samples // x.shape[0], axis=0)

    if start is not None:
        start = repeat(start)
        inner = np.concatenate((start, inner), axis=-2)

    if end is not None:
        end = repeat(end)
        inner = np.concatenate((inner, end), axis=-2)

    return inner


def get_start_end(x, squeeze=True):
    if squeeze:
        return x[..., :1, :], x[..., -1:, :]
    else:
        return x[..., 0, :], x[..., -1, :]


# PATH PROPERTIES AND VALUES
def inf_joint_wrapper(x, inf_bool=None):
    if inf_bool is not None:
        x[..., inf_bool] = angle2minuspi_pluspi(x[..., inf_bool])
    return x


def get_start_end_normalization(*, start, end, n,
                                joint_weighting=None, infinity_joints,
                                eps=0.01):  # Weighting for path, minimal distance in matrix, rad
    """
    Get minimal length cost between x_start and x_end with n_wp waypoints (linear connection)
    Divide the connection in (n_wp-1) equal steps, square each step and consider their sum
    """

    if joint_weighting is None:
        joint_weighting = 1
    else:
        joint_weighting = joint_weighting

    x_diff = (inf_joint_wrapper(x=end[..., 0, :] - start[..., 0, :], inf_bool=infinity_joints))
    x_diff = (x_diff + eps) ** 2
    # norm = 0.5 * np.sqrt((joint_weighting * x_diff).sum(axis=-1))  # Weighted sum over the joints
    norm = 0.5 * (joint_weighting * x_diff).sum(axis=-1)  # Weighted sum over the joints
    norm /= (n - 1)
    return norm


def get_steps(x, infinity_joints=None):
    return inf_joint_wrapper(np.diff(x, axis=1), inf_bool=infinity_joints)


def get_steps_norm(x, infinity_joints=None):
    return np.linalg.norm(get_steps(x, infinity_joints=infinity_joints), axis=-1)


def path_length(x, infinity_joints=None,
                squared=False):  # Options
    """
    Calculate the length of the path by summing up all individual steps  (see path.step_lengths).
    Assume linear connecting between way points.
    If boolean 'squared' is True: take the squared distance of each step -> enforces equally spaced steps.
    """

    step_lengths = get_steps_norm(x=x, infinity_joints=infinity_joints)
    if squared:
        step_lengths **= 2
    return step_lengths.sum(axis=-1)


def linear_distance(*, start=None, end=None, infinity_joints=None,
                    x=None):
    if start is None:
        start = x[..., 0, :]
    if end is None:
        end = x[..., -1, :]

    return np.linalg.norm(inf_joint_wrapper(end - start, inf_bool=infinity_joints), axis=-1)


def get_substeps(x, n,
                 infinity_joints=None,
                 include_start=True):

    n_samples, n_wp, n_dof = x.shape

    # The only fill in substeps if the number is greater 1,
    if n <= 1 or n_wp <= 1:
        if include_start:
            return x
        else:  # How often does this happen? once?: fill_linear_connection
            return x[:, 1:, :]

    steps = get_steps(x=x, infinity_joints=infinity_joints)
    delta = (np.arange(n-1, -1, -1)/n) * steps[..., np.newaxis]
    x_ss = x[..., 1:, :, np.newaxis] - delta

    x_ss = x_ss.transpose((0, 1, 3, 2)).reshape((n_samples, (n_wp-1) * n, n_dof))

    if include_start:
        x_ss = x_inner2x(inner=x_ss, start=x[:, 0, :], end=None)

    x_ss[..., infinity_joints] = angle2minuspi_pluspi(x_ss[..., infinity_joints])
    return x_ss


def fill_linear_connection(q, n, infinity_joints, weighting=None):

    _, n_points, n_dof = q.shape
    n_connections = n_points - 1
    x_rp_steps = get_steps(x=q, infinity_joints=infinity_joints)
    x_rp_steps = x_rp_steps[0]
    if weighting is not None:
        x_rp_steps *= weighting

    # Distribute the waypoints equally along the linear sequences of the initial path
    x_rp_steps_norm = np.linalg.norm(x_rp_steps, axis=-1)
    if np.sum(x_rp_steps_norm) == 0:
        x_rp_steps_relative_length = np.full(n_connections, fill_value=1 / n_connections)
    else:
        x_rp_steps_relative_length = x_rp_steps_norm / np.sum(x_rp_steps_norm)

    # Adjust the number of waypoints for each step to make the initial guess as equally spaced as possible
    n_wp_sub_exact = x_rp_steps_relative_length * (n - 1)
    n_wp_sub = np.round(n_wp_sub_exact).astype(int)
    n_wp_sub_acc = n_wp_sub_exact - n_wp_sub

    n_waypoints_diff = (n - 1) - np.sum(n_wp_sub)

    # If the waypoints do not match, change the substeps where the rounding was worst
    if n_waypoints_diff != 0:
        n_wp_sub_acc = 0.5 + np.sign(n_waypoints_diff) * n_wp_sub_acc
        steps_corrections = np.argsort(n_wp_sub_acc)[-np.abs(n_waypoints_diff):]
        n_wp_sub[steps_corrections] += np.sign(n_waypoints_diff)

    n_wp_sub_cs = n_wp_sub.cumsum()
    n_wp_sub_cs = np.hstack((0, n_wp_sub_cs)) + 1

    # Add the linear interpolation between the random waypoints step by step for each dimension
    x_path = np.zeros((1, n, n_dof))
    x_path[:, 0, :] = q[:, 0, :]
    for i_rp in range(n_connections):
        x_path[:, n_wp_sub_cs[i_rp]:n_wp_sub_cs[i_rp + 1], :] = \
            get_substeps(x=q[:, i_rp:i_rp + 2, :], n=n_wp_sub[i_rp], infinity_joints=infinity_joints,
                         include_start=False)

    x_path = inf_joint_wrapper(x=x_path, inf_bool=infinity_joints)
    return x_path


# ADDITIONAL FUNCTIONS
def order_path(x, start=None, end=None, infinity_joints=None, weights=None):
    """
    Order the points given by 'x' [2d: (n, n_dof)] according to a weighted Euclidean distance
    so that always the nearest point comes next.
    Start with the first point in the array and end with the last if 'x_start' or 'x_end' aren't given.
    """

    n_dof = None
    # Handle different input configurations
    if start is None:
        start = x[..., 0, :]
        x = np.delete(x, 0, axis=-2)
    else:
        n_dof = np.size(start)

    if x is None:
        n_waypoints = 0
    else:
        n_waypoints, n_dof = x.shape[-2:]

    if end is None:
        xi_path = np.zeros((1, n_waypoints + 1, n_dof))
    else:
        xi_path = np.zeros((1, n_waypoints + 2, n_dof))
        xi_path[0, -1, :] = end.ravel()

    xi_path[0, 0, :] = start.ravel()

    # Order the points, so that always the nearest is visited next, according to the Euclidean distance
    for i in range(n_waypoints):
        x_diff = np.linalg.norm(inf_joint_wrapper(x - start, inf_bool=infinity_joints), axis=-1)
        i_min = np.argmin(x_diff)
        xi_path[..., 1 + i, :] = x[..., i_min, :]
        start = x[..., i_min, :]
        x = np.delete(x, i_min, axis=-2)

    return xi_path


#
# DERIVATIVES
def d_steplength__dx(steps=None, step_lengths=None):

    jac = steps.copy()
    motion_idx = step_lengths != 0  # All steps where there is movement between t, t+1
    jac[motion_idx, :] = jac[motion_idx, :] / step_lengths[motion_idx][..., np.newaxis]

    return jac


def d_substeps__dx(n, order=0):
    """
    Get the dependence of substeps (') on the outer way points (x).
    The substeps are placed linear between the waypoints.
    To prevent counting points double one step includes only one of the two endpoints
    This gives a symmetric number of steps but ignores either the start or the end in
    the following calculations.

         x--'--'--'--x---'---'---'---x---'---'---'---x--'--'--'--x--'--'--'--x
    0:  {>         }{>            } {>            } {>         }{>         }
    1:     {         <} {            <} {            <}{         <}{         <}

    Ordering of the waypoints into a matrix:
    0:
    s00 s01 s02 s03 -> step 0 (with start)
    s10 s11 s12 s13
    s20 s21 s22 s23
    s30 s31 s32 s33
    s40 s41 s42 s43 -> step 4 (without end)

    1: (shifting by one, and reordering)
    s01 s02 s03 s10 -> step 0 (without start)
    s11 s12 s13 s20
    s21 s22 s23 s30
    s31 s32 s33 s40
    s41 s42 s43 s50 -> step 4 (with end)


    n          # way points
    n-1        # steps
    n_s        # intermediate points per step
    (n-1)*n_s  # substeps (total)

    n_s = 5
    jac0 = ([[1.0 , 0.8, 0.6, 0.4, 0.2],  # following step  (including way point (1.))
             [0.0 , 0.2, 0.4, 0.6, 0.8]]) # previous step  (excluding way point (1.))

    jac1 = ([[0.2, 0.4, 0.6, 0.8, 1.0 ],   # previous step  (including way point (2.))
             [0.8, 0.6, 0.4, 0.2, 0.0 ]])  # following step  (excluding way point (2.))

    """

    if order == 0:
        jac = (np.arange(n) / n)[np.newaxis, :].repeat(2, axis=0)
        jac[0, :] = 1 - jac[0, :]
    else:
        jac = (np.arange(start=n - 1, stop=-1, step=-1) / n)[np.newaxis, :].repeat(2, axis=0)
        jac[0, :] = 1 - jac[0, :]

    # jac /= n_substeps
    # Otherwise the center would always be 1, no matter how many substeps -> better this way
    # print('substeps normalization')
    return jac


def combine_d_substeps__dx(d_dxs, n,
                           n_samples, n_dof):
    # Combine the jacobians of the sub-way-points (joints) to the jacobian for the optimization variables
    if n > 1:
        d_dxs = d_dxs.reshape(n_samples, -1, n, n_dof)
        ss_jac = d_substeps__dx(n=n, order=1)[np.newaxis, np.newaxis, ..., np.newaxis]
        d_dx = np.einsum('ijkl, ijkl -> ijl', d_dxs, ss_jac[:, :, 0, :, :])
        d_dx[:, :-1, :] += np.einsum('ijkl, ijkl -> ijl', d_dxs[:, 1:, :, :], ss_jac[:, :, 1, :, :])
        return d_dx
    else:
        return d_dxs
