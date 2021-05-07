import numpy as np
from Kinematic import forward

from Optimizer.path import d_steplength__dx, combine_d_substeps__dx


def __dist(x, oc):
    return oc.dist_fun(x=x) - oc.spheres_rad


def __cost(x, oc):
    return basin_cost(distance=__dist(x=x, oc=oc), eps=oc.eps_dist_cost)


def __cost_grad(x, oc):
    return basin_cost_grad_wrapper(x=x,
                                   dist_fun=lambda x: __dist(x, oc),
                                   dist_grad=oc.dist_grad, eps=oc.eps_dist_cost)


def oc_dist(x_spheres, oc):
    return __dist(x=x_spheres[..., oc.active_spheres, :], oc=oc)


def oc_dist_jac(x_spheres, dx_dq, oc,
                jac_only=True):

    x_spheres = x_spheres[..., oc.active_spheres, :]
    dx_dq = dx_dq[..., oc.active_spheres, :]

    dd_dx = oc.dist_grad(x=x_spheres)
    dd_dq = (dd_dx[..., np.newaxis, :, :] * dx_dq).sum(axis=-1)

    if jac_only:
        return dd_dq

    else:
        d = oc.dist_fun(x_spheres)
        return d, dd_dq


def oc_cost(x_spheres, oc):
    return __cost(x=x_spheres[..., oc.active_spheres, :], oc=oc).sum(axis=(-1, -2))


def oc_cost_jac(x_spheres, dx_dq, oc,
                jac_only=True):

    x_spheres = x_spheres[..., oc.active_spheres, :]
    dx_dq = dx_dq[..., oc.active_spheres, :]
    n_samples, n_wp_ss, n_dof, n_spheres, n_dim = dx_dq.shape

    o, do_dx = __cost_grad(x=x_spheres, oc=oc)
    jac = (do_dx[..., np.newaxis, :, :] * dx_dq).sum(axis=(-2, -1))

    if n_wp_ss > 1:
        jac = combine_d_substeps__dx(d_dxs=jac, n=oc.n_substeps, n_samples=n_samples, n_dof=n_dof)

    if jac_only:
        return jac
    else:
        o = o.sum(axis=(-2, -1))
        return o, jac


def oc_cost_w_length(x_spheres, oc):
    """
    Approximate the obstacle cost of a path by summing up the values at substeps and weight them by the length of
    these steps. Approximation converges to integral for n_substeps -> inf
    """

    x_spheres = x_spheres[..., oc.active_spheres, :]
    _, n_wp_ss, _, _ = x_spheres.shape

    if n_wp_ss == 1:
        step_length = 1
    else:
        step_length = np.linalg.norm(np.diff(x_spheres, axis=-3), axis=-1)
        x_spheres = x_spheres[..., 1:, :, :]  # Ignore the start point

    # Obstacle distance cost at the substeps
    o = __cost(x=x_spheres, oc=oc)

    # Sum over the sub steps, Mean over the spheres
    o = (o * step_length).sum(axis=(-2, -1))

    return o


def oc_cost_w_length_grad(x_spheres, dx_dq, oc,
                          jac_only=True, include_end=False):
    x_spheres = x_spheres[..., oc.active_spheres, :]
    dx_dq = dx_dq[..., oc.active_spheres, :]

    n_samples, n_wp_ss, n_dof, n_spheres, n_dim = dx_dq.shape

    if n_wp_ss == 1:
        o, o_jac = __cost_grad(x=x_spheres, oc=oc)
        jac = (dx_dq * o_jac[..., np.newaxis, :, :]).sum(axis=(-2, -1)) / n_spheres
        if jac_only:
            return jac
        else:
            o = o.sum(axis=(-2, -1)) / n_spheres
            return o, jac

    # Calculate different parts for the jacobians
    # (Sub) Step length + Jac
    x_spheres_steps = np.diff(x_spheres, axis=1)
    x_spheres_step_length = np.linalg.norm(x_spheres_steps, axis=-1, keepdims=True)
    ssl_jac = d_steplength__dx(steps=x_spheres_steps, step_lengths=x_spheres_step_length[..., 0])

    # EDT cost and derivative at the substeps
    x_spheres = x_spheres[..., 1:, :, :]  # Ignore starting point, should be by definition collision free
    o, o_jac = __cost_grad(x=x_spheres, oc=oc)
    o = o[..., np.newaxis]

    # Combine the expressions to get the final jacobian
    #              v     *   l'
    jac = o * ssl_jac
    jac[..., :-1, :, :] -= jac[..., 1:, :, :]  # (Two contributions, from left and right neighbour)

    #                v'        *        l
    jac += o_jac * x_spheres_step_length
    # FIXME magnitudes smaller than v*l' ? ssl_jac is always around 1 but x_spheres_step_length depends on bee_length

    jac = (dx_dq * jac[..., np.newaxis, :, :]).sum(axis=(-2, -1))

    # Combine the jacobians of the sub-way-points (joints) to the jacobian for the optimization variables
    jac = combine_d_substeps__dx(d_dxs=jac, n=oc.n_substeps, n_samples=n_samples, n_dof=n_dof)

    if not include_end:
        jac = jac[..., :-1, :]

    if jac_only:
        return jac
    else:
        o = o[..., 0] * x_spheres_step_length[..., 0]
        o = o.sum(axis=(-2, -1))
        return o, jac


def oc_check(x_spheres, oc, verbose=1):

    d = oc_dist(x_spheres=x_spheres, oc=oc)

    d = d.min(axis=-2)  # min over all time steps
    d_idx = d.argmin(axis=-1)
    d = d.min(axis=-1)

    feasible = np.array(d > oc.dist_threshold)

    if verbose > 0:
        for i in range(np.size(feasible)):
            if not feasible[i] or verbose > 2:
                oc_info = f"Sphere {d_idx[i]}"
                print('Minimal Obstacle Distance:  ({})={:.4}m  -> Feasible: {}'.format(
                    oc_info, d[i], feasible[i]))

    return feasible


def oc_check2(q, robot, oc, verbose=1):
    x_spheres = forward.get_x_spheres_substeps(q=q, n=oc.n_substeps_check, robot=robot)
    return oc_check(x_spheres=x_spheres, oc=oc, verbose=verbose)


# Helper
def basin_cost(distance, eps, return_indices=False):
    c = distance.copy()

    # 1. Greater than epsilon
    i_eps_ = np.nonzero(c > eps)
    c[i_eps_] = 0
    # 2. Between zero and epsilon
    i_0_eps = np.nonzero(c > 0)
    c[i_0_eps] = 1 / (2 * eps) * (c[i_0_eps] - eps) ** 2
    # 3. Less than zero
    i__0 = np.nonzero(c < 0)
    c[i__0] = - c[i__0] + 1 / 2 * eps

    if return_indices:
        return c, (i_eps_, i_0_eps, i__0)
    else:
        return c


def basin_cost_grad(distance, eps):
    # A. Cost Function
    c, (i_eps_, i_0_eps, i__0) = basin_cost(distance=distance, eps=eps, return_indices=True)

    # B. Cost Derivative
    j = c.copy()
    # cost_grad[i_eps_] = 0                           # 1.
    j[i_0_eps] = 1 / eps * (distance[i_0_eps] - eps)  # 2.
    j[i__0] = -1                                      # 3.

    return c, j


def basin_cost_grad_wrapper(x, dist_fun, dist_grad, eps):
    d = dist_fun(x=x)

    j = np.zeros_like(x)
    c, j_temp = basin_cost_grad(distance=d, eps=eps)
    j[:] = j_temp[..., np.newaxis]
    mask = c != 0

    # Only calculate the gradient, if there is collision
    j[mask] *= dist_grad(x=x[mask])
    return c, j
