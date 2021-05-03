import numpy as np
from wzk import mp_wrapper, get_n_samples_per_process, shape_wrapper


def __sample_q_feasible(*, robot, feasibility_check, shape=1,
                        max_iter=20, verbose=0):
    count = 0
    safety_factor = 1.01

    shape = shape_wrapper(shape)
    q = np.zeros(shape + (robot.n_dof,))
    feasible = None
    n_feasible = n_feasible_temp = 0
    n_samples_temp = np.prod(shape) // 2

    while n_feasible < np.prod(shape):

        if n_feasible_temp == 0:
            n_samples_temp *= 2
        else:
            n_samples_temp = (feasible.size - feasible.sum()) / feasible.mean()
        n_samples_temp *= safety_factor
        n_samples_temp = max(int(np.ceil(n_samples_temp)), 1)

        q_temp = robot.sample_q(shape=n_samples_temp)
        feasible = feasibility_check(q=q_temp[:, np.newaxis, :]) >= 0
        n_feasible_temp = np.sum(feasible)

        q.reshape(-1, robot.n_dof)[n_feasible:n_feasible + n_feasible_temp, :] = \
            q_temp[feasible, :][:np.prod(shape) - n_feasible, :]

        n_feasible += n_feasible_temp
        if verbose > 0:
            print('sample_valid_q', count, shape, n_feasible)
        if count >= max_iter:
            raise RuntimeError('Maximum number of iterations reached!')

        count += 1
    return q


def sample_q(*, robot, shape=None,
             feasibility_check=False, max_iter=20, verbose=0):
    shape = shape_wrapper(shape)
    n = np.prod(shape)
    if feasibility_check:
        bs = int(1e4)  # batch_size to compute the configurations sequentially, otherwise MemoryError

        if n > bs:
            q = np.zeros(shape + (robot.n_dof,))
            for i in range(n // bs + 1):
                q.reshape(-1, robot.n_dof)[i * bs:(i + 1) * bs, :] = \
                    __sample_q_feasible(robot=robot, shape=int(min(bs, max(0, n - i * bs))),
                                        feasibility_check=feasibility_check, max_iter=max_iter, verbose=verbose)
        else:
            q = __sample_q_feasible(robot=robot, shape=shape,
                                    feasibility_check=feasibility_check, max_iter=max_iter, verbose=verbose)

    else:
        q = robot.sample_q(shape=shape)  # without self-collision it is 250x faster

    return q


def sample_q_mp(*, robot, shape=None,
                feasibility_check=False, n_processes=1):
    """
    If valid is False is it faster for values < 1e5 to don't use multiprocessing,
    On galene there is a overhead of around 10ms for each process you start.
    """

    shape = shape_wrapper(shape)

    def fun_wrapper(n):
        return sample_q(robot=robot, shape=int(n), feasibility_check=feasibility_check)

    n_samples_per_core, _ = get_n_samples_per_process(n_samples=np.prod(shape), n_processes=n_processes)
    return mp_wrapper(n_samples_per_core, fun=fun_wrapper, n_processes=n_processes).reshape(shape)


def sample_q_frames_mp(*, robot, shape=None,
                       feasibility_check=False, n_processes=1, frames_idx=None):
    shape = shape_wrapper(shape)

    if frames_idx is None:
        def fun(n):
            q = sample_q(robot=robot, shape=int(n), feasibility_check=feasibility_check)
            frames = robot.get_frames(q=q)
            return q, frames
    else:
        def fun(n):
            q = sample_q(robot=robot, shape=int(n), feasibility_check=feasibility_check)
            frames = robot.get_frames(q=q)[:, :, frames_idx, :, :]
            return q, frames

    n_samples_per_core, _ = get_n_samples_per_process(n_samples=np.prod(shape), n_processes=n_processes)
    return mp_wrapper(n_samples_per_core, fun=fun, n_processes=n_processes).reshape(shape)
