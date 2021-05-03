from matplotlib.collections import LineCollection

from wzk.mpl import *
import GridWorld.world2grid as w2g


def new_world_fig(*, limits=10, n_dim=2,
                  ax_labels=False, title=None,
                  width=10, position=None, monitor=-1,
                  **kwargs_fig):

    fig, ax = new_fig(n_dim=n_dim, width=width, title=title, position=position, monitor=monitor, aspect=1,
                      **kwargs_fig)

    set_ax_limits(ax=ax, limits=limits, n_dim=n_dim)
    turn_ticklabels_off(ax=ax)
    set_ticks_position(ax=ax, position='both')

    if ax_labels:
        set_labels(ax=ax, labels=('x', 'y', 'z'))

    return fig, ax


def plot_x_path(x, r=None,
                ax=None, **kwargs):
    n_dim = x.shape[-1]
    if ax is None:
        ax = plt.gca()

    if r is not None:
        size_new = size_units2points(size=r, ax=ax)
        kwargs['markersize'] = size_new

    h = ax.plot(*x.reshape(-1, n_dim).T, **kwargs)[0]

    if r is not None:
        size_units2points_listener(ax=ax, h=h, size=r)

    return h


def update_x_path(x, h):
    n_dim = x.shape[-1]

    h.set_xdata(x[..., 0].ravel())
    h.set_ydata(x[..., 1].ravel())
    if n_dim == 3:
        h.set_3d_properties(x[..., 2].ravel())


def plot_img_patch(*, img, limits, ax=None, **kwargs):
    """
    Plot an image as a Collection of square Rectangle Patches.
    Draw all True / nonzero pixels.
    """

    if ax is None:
        ax = plt.gca()

    voxel_size = w2g.limits2voxel_size(shape=img.shape, limits=limits)

    ij = np.array(np.nonzero(img)).T
    xy = w2g.grid_i2x(i=ij, voxel_size=voxel_size, lower_left=limits[:, 0], mode='b')

    pc = mpl.collections.PatchCollection([patches.Rectangle((x, y), width=voxel_size, height=voxel_size,
                                                            fill=False, snap=True) for (x, y) in xy], **kwargs)
    ax.add_collection(pc)
    return pc


def plot_img_outlines(*, img, limits, ax=None, **kwargs):
    """
    Plot the image by drawing the outlines of the areas where the values are True.
    """

    if ax is None:
        ax = plt.gca()

    combined_edges = get_combined_edges(img)

    voxel_size = w2g.limits2voxel_size(shape=img.shape, limits=limits)
    combined_edges = [w2g.grid_i2x(i=ce, voxel_size=voxel_size, lower_left=limits[:, 0], mode='b')
                      for ce in combined_edges]

    lc = LineCollection(combined_edges, **kwargs)
    ax.add_collection(lc)
    return lc


def __img_none_limits(limits=None, img=None):
    if limits is None and img is not None:
        limits = np.zeros((img.ndim, 2))
        limits[:, 1] = img.shape

    return limits


def plot_img_patch_w_outlines(*, ax, img, limits=None,
                              color=None, facecolor='k', edgecolor='k',
                              hatch='xx',
                              lw=2,
                              alpha_outline=1, alpha_patch=1):
    if img is None:
        return None

    if color is not None:
        facecolor = color
        edgecolor = color

    limits = __img_none_limits(limits=limits, img=img)

    if img.ndim == 2:
        plot_img_outlines(img=img, limits=limits, ax=ax, color=edgecolor, ls='-', lw=lw, alpha=alpha_outline)
        plot_img_patch(img=img, limits=limits, ax=ax, lw=0, hatch=hatch, facecolor='None', edgecolor=facecolor,
                       alpha=alpha_patch)

    else:  # n_dim == 3
        voxel_size = w2g.limits2voxel_size(shape=img.shape, limits=limits)
        if isinstance(img, tuple):
            rect_pos, rect_size = img
            face_vtx = rectangles2face_vertices(rect_pos=rect_pos, rect_size=rect_size)
        else:
            face_vtx = get_combined_faces(img=img)

        face_vtx = w2g.grid_i2x(i=face_vtx, voxel_size=voxel_size, lower_left=limits[:, 0], mode='b')
        plot_poly_collection_3d(face_vtx=face_vtx, ax=ax, facecolor=facecolor, edgecolor=edgecolor, alpha=0.4)


def plot_spheres(q, robot,
                 ax=None, h=None,
                 **kwargs):
    x = robot.get_x_spheres(q=q)
    x = x.reshape((-1, x.shape[-1]))
    h = __plot_circles(h=h, ax=ax, x=x, r=robot.spheres_rad, **kwargs)
    return h


def __plot_circles(x, r,
                   ax=None, h=None,
                   color=None, alpha=None,
                   **kwargs):
    r = safe_scalar2array(r, shape=len(x))

    if h is None:
        h = []
        for x_i, r_i in zip(x, r):
            c = patches.Circle(xy=x_i, radius=r_i, alpha=alpha, color=color, **kwargs)
            ax.add_patch(c)
            h.append(c)
        return h

    else:
        for h_i, x_i in zip(h, x):
            h_i.set_center(x_i)

            if alpha is not None:
                h_i.set_alpha(alpha)

            if color is not None:
                h_i.set_color(color)
        return h