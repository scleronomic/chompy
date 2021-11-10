import numpy as np

from Kinematic.Robots import SingleSphere02, StaticArm


radius = 0.3
robot = SingleSphere02(radius=0.3)  # Size of the robot [m]
# robot = StaticArm(n_dof=5, limb_lengths=0.4, radius=0.1)

# Sample random configurations
q = robot.sample_q(10)

# Forward Kinematic, get frames and position of the spheres
f = robot.get_frames(q)
x = robot.get_x_spheres(q)

# Get the derivative of the forward kinematic with respect to the joints
f, df_dq = robot.get_frames_jac(q)
x, dx_dq = robot.get_x_spheres_jac(q)

print('Frames')
print('f    ', f.shape)
print('df_dq', df_dq.shape)
print('n_frames:', df_dq.shape[-3])
print('Spheres')
print('x    ', x.shape)
print('dx_dq', dx_dq.shape)
print('n_spheres:', dx_dq.shape[-2])


# To keep things simple in the beginning:
#  * start with SingleSphere02
#  * don't use substeps, ie n_substeps:=1
#
