import numpy as np
from wzk import mp_wrapper

from Optimizer import path, chomp


# Gradient Descent
def gradient_descent_wrapper_mp(x, fun, grad, gd):
    def gd_wrapper(__x):
        return gradient_descent(x=__x, fun=fun, grad=grad, gd=gd)

    return mp_wrapper(x, n_processes=gd.n_processes, fun=gd_wrapper)


def __get_adaptive_step_size(x, x_old, df_dx, df_dx_old,
                             step_size):
    # Calculate the adaptive step size
    if x_old is None:
        ss = step_size
    else:
        diff_x = np.abs(x - x_old)
        diff_grad = np.abs(df_dx - df_dx_old)
        diff_grad_squared_sum = (diff_grad ** 2).sum(axis=(-2, -1))
        const_ss_idx = diff_grad_squared_sum == 0
        diff_grad_squared_sum[const_ss_idx] = 1

        ss = (diff_x * diff_grad).sum(axis=(-2, -1)) / diff_grad_squared_sum
        ss[const_ss_idx] = step_size
        if np.size(ss) > 1:
            ss = ss.reshape((ss.size,) + (1,) * (df_dx.ndim - 1))

    return ss


def gradient_descent(x, fun, grad, gd):

    df_dx_old = None
    x_old = None

    # If the parameters aren't given for all steps, expand them
    if np.size(gd.grad_tol) == 1:
        gd.grad_tol = np.full(gd.n_steps, fill_value=float(gd.grad_tol))

    if np.size(gd.hesse_weighting) == 1:
        gd.hesse_weighting = np.full(gd.n_steps, fill_value=float(gd.hesse_weighting))

    if gd.return_x_list:
        x_list = np.zeros(x.shape + (gd.n_steps,))
    else:
        x_list = None

    # Gradient Descent Loop
    for i in range(gd.n_steps):

        df_dx = grad(x=x, i=i)

        if gd.callback is not None:
            df_dx = gd.callback(x=x.copy(), jac=df_dx.copy())

        # Calculate the adaptive step shape
        if gd.adjust_step_size:
            ss = __get_adaptive_step_size(x=x, x_old=x_old,
                                          df_dx=df_dx, df_dx_old=df_dx_old,
                                          step_size=gd.step_size)
            x_old = x.copy()
            df_dx_old = df_dx.copy()
        else:
            ss = gd.step_size

        df_dx *= ss

        # Cut gradient that no step is larger than a tolerance value
        max_grad = np.abs(df_dx).max(axis=(1, 2))
        grad_too_large = max_grad > gd.grad_tol[i]
        df_dx[grad_too_large] /= max_grad[grad_too_large, np.newaxis, np.newaxis] / gd.grad_tol[i]

        # Apply gradient descent
        x -= df_dx

        if gd.return_x_list:
            x_list[..., i] = x

        # Clip the values to fit with the range of values
        if gd.prune_limits is not None:
            x = gd.prune_limits(x)

    objective = fun(x=x)

    if gd.return_x_list:
        return x, objective, x_list
    else:
        return x, objective


def gd_chomp(q0, q_start, q_end,
             par, gd):

    weighting = par.weighting.copy()

    def x_inner2x_mp(x):
        if par.planning.include_end:
            return path.x_inner2x(inner=x, start=q_start, end=None)
        else:
            return path.x_inner2x(inner=x, start=q_start, end=q_end)

    def fun(x):
        return chomp.chomp_cost(q=x_inner2x_mp(x=x), par=par)

    def grad(x, i):
        par.weighting = weighting.at_idx(i=i)
        return chomp.chomp_grad(q=x_inner2x_mp(x=x), par=par, jac_only=True)

    gd.prune_limits = par.robot.prune_joints2limits
    res = gradient_descent_wrapper_mp(q0, fun=fun, grad=grad, gd=gd)

    par.weighting = weighting.copy()
    return res
