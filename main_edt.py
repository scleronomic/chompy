import numpy as np
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

from GridWorld.random_obstacles import create_rectangle_image
from GridWorld.obstacle_distance import obstacle_img2dist_img, obstacle_img2dist_grad
from wzk.mpl import DraggableCircleList, new_fig, imshow, imshow_update, plt, get_xaligned_axes


# A simple animation, to get to know the concept of Signed Distance Field (SDF)/Euclidean Distance Transformation (EDT)
# You can drag around the different sized spheres and change the number of random obstacles in the image
# The main idea here is that by following the gradient of the signed distance field we can get out of collision

class SDF:
    def __init__(self, n_spheres=5, radius_max=0.3,
                 n_voxels=64, voxel_size=0.1):

        self.n_voxels = (n_voxels, n_voxels)    # Size of the world in voxels
        self.voxel_size = voxel_size            # Size of one voxel [m]
        self.limits = np.zeros((2, 2))          # Size of the world in meters
        self.limits[:, 1] = np.array(n_voxels) * self.voxel_size

        self.obstacle_size_min_max = (1, 10)    # Minimal / maximal size of an obstacle in pixels

        self.fig, self.ax = new_fig()
        self.fig.subplots_adjust(bottom=0.25)

        # Obstacle Image + Signed Distance Field
        obstacle_image, distance_image = self.create_obstacle_image(n_obstacles=25)
        self.h_distance = imshow(ax=self.ax, img=distance_image, cmap='Blues', limits=self.limits)
        self.h_obstacle = imshow(ax=self.ax, img=obstacle_image, cmap='Reds', limits=self.limits,
                                 mask=obstacle_image == 0, alpha=0.3)
        plt.show()
        plt.pause(0.01)
        self.slider = Slider(get_xaligned_axes(ax=self.ax, y_distance=0.1, height=0.03),
                             'N Obstacles', valmin=1, valmax=50, valinit=25, valstep=1, valfmt='%d')

        # Spheres
        x = np.random.random((n_spheres, 2)) * n_voxels * voxel_size
        radius = np.linspace(voxel_size, radius_max, num=n_spheres)
        self.circles = DraggableCircleList(ax=self.ax, xy=x, radius=radius, color='k', alpha=0.5)

        # Gradient Descent on the SDF
        self.gd_n_steps = 15
        self.gd_step_size = 0.1
        self.anis = []

        self.slider.on_changed(self.update_obstacles)
        self.circles.add_callback(self.update_circles)

    def create_obstacle_image(self, n_obstacles):
        obstacle_image = create_rectangle_image(n=n_obstacles, size_limits=self.obstacle_size_min_max, n_voxels=self.n_voxels)
        distance_image = obstacle_img2dist_img(img=obstacle_image, voxel_size=self.voxel_size)
        self.dist_derv = obstacle_img2dist_grad(dist_img=distance_image, voxel_size=self.voxel_size, lower_left=0)
        return obstacle_image, distance_image

    def update_obstacles(self, val):
        obstacle_image, distance_image = self.create_obstacle_image(n_obstacles=int(self.slider.val))
        imshow_update(h=self.h_distance, img=distance_image, cmap='Blues')
        imshow_update(h=self.h_obstacle, img=obstacle_image, mask=obstacle_image == 0, cmap='Reds')
        self.update_circles(None)

    def update_circles(self, _):
        def animate(i):
            xy = self.circles.get_xy()
            jac = self.dist_derv(xy)
            self.circles.set_xy(xy=xy + self.gd_step_size * jac)  # Gradient Descent on the Signed Distance Field

        for ani in self.anis:
            ani.event_source.stop()
            self.anis.remove(ani)
            del ani

        self.anis.append(FuncAnimation(self.fig, animate, frames=self.gd_n_steps, repeat=False))
        self.fig.canvas.draw_idle()


sdf = SDF(n_spheres=10)
