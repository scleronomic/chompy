import numpy as np
import matplotlib.pyplot as plt

from wzk.dicts_lists_tuples import atleast_list

def get_pip(ax, x, y, width, height, **kwargs):
    """Picture in Picture.
    Positions of the new axes as fractions from the parent axis.
    """
    # from wzk import tic, toc
    #
    # plt.draw()  # Necessary so
    # tic()
    # plt.pause(0.01)  # Necessary so
    # toc()
    p = ax.get_position()
    return plt.axes([p.x0 + x * p.fig_width_inch, p.y0 + y * p.height,
                     width * p.fig_width_inch, height * p.height], **kwargs)


# Axes
def get_xaligned_axes(ax, y_distance, height, factor=1., **kwargs):
    """

    :param ax:
    :param y_distance: negative distance in y direction to the x axis
    :param height: height of the new axes
    :param factor: factor for scaling the fig_width_inch of the new axes
    :param kwargs: kwargs passed to matplotlib.pyplot.axes()
    :return:
    """
    p = ax.get_position()
    x0, y0, width0 = p.x0, p.y0, p.width

    x0 += (1-factor) * width0 / 2
    width0 *= factor

    return plt.axes([x0, y0 - y_distance, width0, height], **kwargs)


# Limits
def limits4axes(limits, n_dim):
    """
    n_dim = 2
    limits = max_x                           -> (    0,     0), (max_x, max_x)
    limits = (max_x, max_y)                  -> (    0,     0), (max_x, max_y)
    limits = (min_x, min_y), (max_x, max_y)  -> (min_x, min_y), (max_x, max_y)

    n_dim = 3
    limits = max_x                                         -> (    0,     0,     0), (max_x, max_x, max_x)
    limits = (max_x, max_y, max_z)                         -> (    0,     0,     0), (max_x, max_y, max_z)
    limits = (min_x, min_y, min_z), (max_x, max_y, max_z)  -> (min_x, min_y, min_z), (max_x, max_y, max_z)
    """

    if np.size(limits) == 1:
        mins = [0] * n_dim
        maxs = [limits] * n_dim

    elif np.size(limits) == n_dim:
        mins = [0] * n_dim
        maxs = limits

    else:
        assert np.size(limits) == 2 * n_dim
        limits = np.array(limits)
        mins, maxs = limits[:, 0], limits[:, 1]

    return mins, maxs


def limits2extent(limits, origin, axis_order):
    mins, maxs = limits4axes(limits=limits, n_dim=2)
    # extent: (left, right, bottom, top)
    if origin == 'upper':
        if axis_order == 'ij->yx':
            extent = [mins[1], maxs[1], maxs[0], mins[0]]

        else:  # axis_order == 'ij->xy':
            extent = [mins[0], maxs[0], maxs[1], mins[1]]
    else:  # origin == 'lower'

        if axis_order == 'ij->yx':
            extent = [mins[1], maxs[1], mins[0], maxs[0]]
        else:  # axis_order == 'ij->xy':
            extent = [mins[0], maxs[0], mins[1], maxs[1]]

    return extent


def set_ax_limits(ax, limits, n_dim=2):

    mins, maxs = limits4axes(limits=limits, n_dim=n_dim)

    if n_dim == 2:
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
    elif n_dim == 3:
        ax.set_xlim3d(mins[0], maxs[0])
        ax.set_ylim3d(mins[1], maxs[1])
        ax.set_zlim3d(mins[2], maxs[2])
    else:
        raise ValueError(f"Unknown number of dimensions: {n_dim}")


def add_safety_limits(limits, factor):
    # TODO might be better of in arrays?
    limits = np.atleast_1d(limits)
    diff = np.diff(limits, axis=-1)[..., 0]
    return np.array([limits[..., 0] - factor * diff,
                     limits[..., 1] + factor * diff]).T


# Label
def set_labels(ax, labels, **kwargs):

    ax.set_xlabel(labels[0], **kwargs)
    ax.set_ylabel(labels[1], **kwargs)

    try:
        ax.set_zlabel(labels[2], **kwargs)
    except (IndexError, AttributeError):
        pass


# Sizes
def size_units2points(size, ax, reference='y'):
    """
    Convert a shape in Measurements units to shape in points.
    For linewidth of markersize

    Parameters
    ----------
    size: float
        Linewidth in Measurements units of the respective reference-axis
    ax: matplotlib axis
        The axis which is used to extract the relevant transformation
        Measurements (Measurements limits and shape must not change afterwards)
    reference: string
        The axis that is taken as a reference for the Measurements fig_width_inch.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    https://stackoverflow.com/questions/19394505/matplotlib-expand-the-line-with-specified-width-in-data-unit
    """

    fig = ax.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.fig_width_inch * ax.get_position().fig_width_inch
        value_range = np.diff(ax.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * ax.get_position().height
        value_range = np.diff(ax.get_ylim())
    else:
        raise(ValueError("Pass either 'x' or 'y' as reference"))
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return size * (length / value_range)


def size_units2points_listener(ax, h, size, reference='y', mode='ms'):

    if mode is None:
        mode = 'ms'
    elif mode == 'both':
        mode = ['ms', 'lw']

    h, mode = atleast_list(h, mode)

    def on_change(*args):
        size_new = size_units2points(ax=ax, size=size, reference=reference)
        for hh in h:
            if 'ms' in mode or 'markersize' in mode:
                hh.set_markersize(size_new)

            if 'lw' in mode or 'linewidth' in mode:
                hh.set_linewidth(size_new)

    ax.callbacks.connect('xlim_changed', on_change)
    ax.callbacks.connect('ylim_changed', on_change)
    ax.get_figure().canvas.mpl_connect('resize_event', on_change)


def get_aspect_ratio(ax=None):
    """https://stackoverflow.com/questions/41597177/get-aspect-ratio-of-axes"""
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll
    return height / width
