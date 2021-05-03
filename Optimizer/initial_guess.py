import numpy as np
import Optimizer.path as path


def q0_random(*, q_start, q_end,
              n_waypoints, n_random_points=0,
              robot,
              order_random=True):
    """
    Choose 'n_random_points' random points inside the limits of the configuration space of the robot.
    Connect these points linear to get an initial path between start and end configuration.
    # But it is the most general form, what is in general better than random ?
    """

    if q_end is None:
        q_end = robot.sample_q(1)

    x_rp = robot.sample_q((1, n_random_points))
    if order_random:
        x_rp = path.order_path(x=x_rp, start=q_start, end=q_end)
    else:
        x_rp = np.concatenate((q_start, x_rp, q_end), axis=-2)

    return path.fill_linear_connection(q=x_rp, n=n_waypoints, infinity_joints=robot.infinity_joints)


def q0_random_wrapper(*, robot,
                      n_waypoints, n_multi_start,
                      order_random=True, mode='full'):
    """
    Nested wrapper function to conveniently generate initial guesses for the optimizer from random points.
    Returns as function handle with the arguments (xy_start, xy_end, obstacle_img, start_end_img) this function can be
    called for each sample and returns another function handle. This second function is called in the multi start loop t
    of the optimization and yields a random initial path for the optimizer.
    """

    def get_q0(start, end, mode=mode):
        q0 = np.zeros((sum(n_multi_start[1]), n_waypoints, start.shape[-1]))
        count = 0
        for i, n_rp in enumerate(n_multi_start[0]):
            for _ in range(n_multi_start[1][i]):  # Tries per number of random points
                q0[count, ...] = q0_random(q_start=start, q_end=end, robot=robot,
                                           n_waypoints=n_waypoints, n_random_points=n_rp,
                                           order_random=order_random)
                count += 1

        if mode == 'full':
            return q0
        elif mode == 'inner':
            return q0[..., 1:-1, :]
        elif mode == 'wo_start':
            return q0[..., 1:, :]
        elif mode == 'wo_end':
            return q0[..., :-1, :]
        else:
            raise ValueError("Unknown mode3")

    return get_q0


def get_dummy_path(*, q_start=None, q_end=None,
                   n_waypoints=20,
                   n_random_points=2,
                   robot):
    if q_start is None:
        q_start, q_end = robot.sample_q(2)

    return q0_random(q_start=q_start, q_end=q_end,
                     n_waypoints=n_waypoints, robot=robot,
                     order_random=True,
                     n_random_points=n_random_points)
