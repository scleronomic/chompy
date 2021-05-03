import numpy as np
import scipy.ndimage as ndimage  # Image interpolation


# Finding Use voxel_size + lower_left as inputs instead of using limits and inferring those values
#  The advantage of voxel_size + lower_left as argument is that voxel_size is a scalar and lower_left can be omitted in
#  most of the cases so the standard signature is much more concise,
#  the trade off is that some functions have now two parameters instead of just one
#  Additionally limits has always 2*n_dim dimensions


# Obstacle Image to Distance Image
def obstacle_img2dist_img(img, voxel_size, add_boundary=True):
    """
    Calculate the signed distance field from an 2D/3D image of the world.
    Obstacles are 1/True, free space is 0/False.
    The distance image is of the same shape as the input image and has positive values outside objects and negative
    values inside objects see 'CHOMP - signed distance field' (10.1177/0278364913488805)
    The voxel_size is used to scale the distance field correctly (the shape of a single pixel / voxel)
    """

    n_voxels = np.array(img.shape)

    if not add_boundary:
        # Main function
        #                                         # EDT wants objects as 0, rest as 1
        dist_img = ndimage.distance_transform_edt(-img.astype(int) + 1, sampling=voxel_size)
        dist_img_complement = ndimage.distance_transform_edt(img.astype(int), sampling=voxel_size)
        dist_img[img] = - dist_img_complement[img]  # Add interior information

    else:
        # Additional branch, to include boundary filled with obstacles
        obstacle_img_wb = np.ones(n_voxels + 2, dtype=bool)
        inner_image_idx = tuple(map(slice, np.ones(img.ndim, dtype=int), (n_voxels + 1)))
        obstacle_img_wb[inner_image_idx] = img

        dist_img = obstacle_img2dist_img(img=obstacle_img_wb, voxel_size=voxel_size, add_boundary=False)
        dist_img = dist_img[inner_image_idx]

    return dist_img


def obstacle_img2dist_img_grad(*,
                               dist_img=None,  # either
                               obstacle_img=None, add_boundary=True,  # or
                               voxel_size):
    """
    Use Sobel-filter to get the derivative of the edt
    """

    if dist_img is None:
        dist_img = obstacle_img2dist_img(img=obstacle_img, voxel_size=voxel_size, add_boundary=add_boundary)
    return img2sobel(img=dist_img, voxel_size=voxel_size)


# Obstacle Image to Distance Function
def obstacle_img2dist_fun(voxel_size, lower_left, interp_order=1,
                          obstacle_img=None, add_boundary=True,  # A
                          dist_img=None,  # B
                          ):
    """
    Interpolate the distance field at each point (continuous).
    The optimizer is happier with the interpolated version, but it is hard to ensure that the interpolation is
    conservative, so the direct variant should be preferred. (actually)
    # DO NOT INTERPOLATE the EDT -> not conservative -> use order=0 / 'nearest'
    """

    if dist_img is None:
        dist_img = obstacle_img2dist_img(img=obstacle_img, voxel_size=voxel_size, add_boundary=add_boundary)

    dist_fun = img2interpolation_fun(img=dist_img, order=interp_order, voxel_size=voxel_size, lower_left=lower_left)
    return dist_fun


def obstacle_img2dist_grad(*,
                           interp_order=1, voxel_size, lower_left,
                           dist_img_grad=None,  # A
                           obstacle_img=None, add_boundary=True,  # B 1
                           dist_img=None,  # B 2
                           ):
    if dist_img_grad is None:
        dist_img_grad = obstacle_img2dist_img_grad(obstacle_img=obstacle_img, add_boundary=add_boundary,
                                                   dist_img=dist_img, voxel_size=voxel_size)
    dist_grad = img_grad2interpolation_fun(img_grad=dist_img_grad, order=interp_order,
                                           voxel_size=voxel_size, lower_left=lower_left)
    return dist_grad


def obstacle_img2funs(*, img=None, add_boundary=True,  # A
                      dist_img=None,  # B
                      dist_img_grad=None,  # C
                      voxel_size, lower_left,
                      interp_order_dist, interp_order_grad):
    if dist_img is None:
        dist_img = obstacle_img2dist_img(img=img, voxel_size=voxel_size, add_boundary=add_boundary)

    if dist_img_grad is None:
        dist_img_grad = obstacle_img2dist_img_grad(dist_img=dist_img, voxel_size=voxel_size)

    dist_fun = obstacle_img2dist_fun(dist_img=dist_img, interp_order=interp_order_dist,
                                     voxel_size=voxel_size, lower_left=lower_left)

    dist_grad = obstacle_img2dist_grad(dist_img_grad=dist_img_grad, interp_order=interp_order_grad,
                                       voxel_size=voxel_size, lower_left=lower_left)

    return dist_fun, dist_grad


# Helper
def __create_radius_temp(radius, shape):
    if np.size(radius) == 1:
        return radius
    d_spheres = np.nonzero(np.array(shape) == np.size(radius))[0][0]
    r_temp_shape = np.ones(len(shape) - 1, dtype=int)
    r_temp_shape[d_spheres] = np.size(radius)
    return radius.reshape(r_temp_shape)


def img2sobel(img, voxel_size):
    """
    Calculate the derivative of an image in each direction of the image, using the sobel filter.
    """

    sobel = np.zeros((img.ndim,) + img.shape)
    for d in range(img.ndim):  # Treat image boundary like obstacle
        sobel[d, ...] = ndimage.sobel(img, axis=d, mode='constant', cval=0)

    # Check appropriate scaling of sobel filter, should be correct
    sobel /= (8 * voxel_size)  # The shape of the voxels is already accounted for in the distance image
    return sobel


# Image interpolation
def img2interpolation_fun(*, img, order=1, mode='nearest',
                          voxel_size, lower_left):
    """
    Return a function which interpolates between the pixel values of the image (regular spaced grid) by using
    'scipy.ndimage.map_coordinates'. The resulting function takes as input argument either a np.array or a list of
    world coordinates (!= image coordinates)
    The 'order' keyword indicates which order of interpolation to use. Standard is linear interpolation (order=1).
    For order=0 no interpolation is performed and the value of the nearest grid cell is chosen. Here the values between
    the different cells jump and aren't continuous.

    """

    factor = 1 / voxel_size

    def interp_fun(x):
        x2 = x.copy()
        if x2.ndim == 1:
            x2 = x2[np.newaxis, :]

        # Map physical coordinates to image indices
        if np.any(lower_left != 0):
            x2 -= lower_left

        x2 *= factor
        x2 -= 0.5

        return ndimage.map_coordinates(input=img, coordinates=x2.T, order=order, mode=mode).T

    return interp_fun


def img_grad2interpolation_fun(*, img_grad, voxel_size, lower_left, order=1, mode='nearest'):
    """
    Interpolate images representing derivatives (ie from soble filter). For each dimension there is a derivative /
    layer in the image.
    Return the results combined as an (x, n_dim) array for the derivatives at each point for each dimension.
    """

    n_dim = img_grad.shape[0]

    fun_list = []
    for d in range(n_dim):
        fun_list.append(img2interpolation_fun(img=img_grad[d, ...], order=order, mode=mode, voxel_size=voxel_size,
                                              lower_left=lower_left))

    def fun_grad(x):
        res = np.empty_like(x)
        for _d in range(n_dim):
            res[..., _d] = fun_list[_d](x=x)

        return res

    return fun_grad
