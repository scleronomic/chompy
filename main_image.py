import numpy as np
import matplotlib.pyplot as plt

from GridWorld.random_obstacles import create_rectangle_image, create_perlin_image

n_voxels = (100, 100)
n_obstacles = 20
min_max_obstacle_size_voxel = [5, 20]
img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel,
                             n_voxels=n_voxels)


res = 4
threshold = 0.5
img = create_perlin_image(n_voxels=n_voxels, res=res, threshold=threshold)


plt.imshow(img)
plt.show()



