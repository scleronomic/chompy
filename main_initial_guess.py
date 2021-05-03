import numpy as np
import matplotlib.pyplot as plt

from Kinematic.Robots import SingleSphere02
from Optimizer import initial_guess

radius = 0.3  # Size of the robot [m]
robot = SingleSphere02(radius=radius)
n_waypoints = 10

# Build function to generate initial guesses, with linear connected random points
# Number of multi starts
n_multi_start_rp = [[0, 1, 2],  # how many random points for the multi-start (0 == straight line)
                    [1, 5, 4]]  # how many trials for each variation
get_q0 = initial_guess.q0_random_wrapper(robot=robot, n_multi_start=n_multi_start_rp,
                                         n_waypoints=n_waypoints, order_random=True, mode='inner')

# Get multi starts for a start-end pair
q_start, q_end = np.array([1, 1]), np.array([9, 9])
q_ms = get_q0(start=q_start, end=q_end, mode='full')

fig, ax = plt.subplots()
ax.plot(*q_ms[0].T, '-o', color='k', label='direct path')
ax.plot(*q_ms[1].T, '-o', color='r', label='one random point')
ax.plot(*q_ms[6].T, '-o', color='blue', label='two random points')

ax.legend()
plt.show()

# Do the same again for a new set of start and end configurations
q_start, q_end = np.array([2, 2]), np.array([5, 8])
q_ms = get_q0(start=q_start, end=q_end, mode='full')