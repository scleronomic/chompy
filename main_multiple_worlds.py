from Kinematic.Robots import SingleSphere02
from Optimizer import chomp

from parameter import Parameter, initialize_oc
from GridWorld import create_perlin_image


world_list = create_perlin_image(n_voxels=(64, 64), n=10)
robot = SingleSphere02(radius=0.3)
par = Parameter(robot=robot, obstacle_img=None)

# Option A
for world in world_list:
    initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=world)
    q = par.robot.sample_q()
    chomp.chomp_grad(q, par=par)

# Option B:
par_list = [Parameter(robot=robot, obstacle_img=w) for w in world_list]
for par in par_list:
    q = par.robot.sample_q()
    chomp.chomp_grad(q, par=par)
    