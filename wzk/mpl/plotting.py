import datetime
import numpy as np
from itertools import combinations
from scipy.stats import linregress
from matplotlib import collections

from wzk.mpl.figure import plt, new_fig, subplot_grid
from wzk.mpl.colors import arr2rgba
from wzk.mpl.axes import limits4axes, limits2extent, set_ax_limits, add_safety_limits
from wzk.math2 import binomial
from wzk.numpy2 import safe_scalar2array
from wzk.dicts_lists_tuples import tuple_extract, atleast_tuple


def imshow(*, ax, img, limits=None, cmap=None,
           origin='lower', axis_order='ij->yx',
           mask=None, vmin=None, vmax=None, **kwargs):
    """

    ## origin: upper
    # axis_order: ij
    (i0 ,j0), (i0 ,j1), (i0 ,j2), ...
    (i1, j0), (i1, j1), (i1, j2), ...
    (i2, j0), (i2, j1), (i2, j2), ...
    ...     , ...     ,  ...    , ...

    # axis_order: ji
     (i0 ,j0), (i1 ,j0), (i2 ,j0), ...
     (i0, j1), (i1, j1), (i2, j1), ...
     (i0, j2), (i1, j2), (i2, j2), ...
     ...     , ...     ,  ...    , ...

    ## origin: lower
    # axis_order: ij
    ...     , ...     ,  ...    , ...
    (i2, j0), (i2, j1), (i2, j2), ...
    (i1, j0), (i1, j1), (i1, j2), ...
    (i0 ,j0), (i0 ,j1), (i0 ,j2), ...

    # axis_order: ji
     ...     , ...     ,  ...    , ...
     (i0, j2), (i1, j2), (i2, j2), ...
     (i0, j1), (i1, j1), (i2, j1), ...
     (i0 ,j0), (i1 ,j0), (i2 ,j0), ...
    """

    assert img.ndim == 2
    assert origin in ('lower', 'upper')
    assert axis_order in ('ij->yx', 'ij->xy')
    if limits is None:
        limits = img.shape

    extent = limits2extent(limits=limits, origin=origin, axis_order=axis_order)

    img = arr2rgba(img=img, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask, axis_order=axis_order)

    return ax.imshow(img, extent=extent, origin=origin, **kwargs)


def imshow_update(h, img, cmap=None, axis_order='ij->yx', vmin=None, vmax=None, mask=None):
    if cmap is None:
        cmap = h.cmap

    img = arr2rgba(img=img, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask, axis_order=axis_order)
    h.set_data(img)


def plot_projections_2d(x, dim_labels=None, ax=None, limits=None, aspect='auto', **kwargs):
    n = x.shape[-1]

    n_comb = binomial(n, 2)
    if ax is None:
        ax = subplot_grid(n=n_comb, squeeze=False, aspect=aspect)
    else:
        ax = np.atleast_2d(ax)

    if dim_labels is None:
        dim_labels = [str(i) for i in range(n)]
    assert len(dim_labels) == n, f"{dim_labels} | {n}"

    comb = combinations(np.arange(n), 2)
    for i, c in enumerate(comb):
        i = np.unravel_index(i, shape=ax.shape)
        ax[i].scatter(*x[..., c].T, **kwargs)
        ax[i].set_xlabel(dim_labels[c[0]])
        ax[i].set_ylabel(dim_labels[c[1]])
        if limits is not None:
            set_ax_limits(ax=ax[i], limits=limits[c, :])


def color_plot_connected(y, color_s, x=None, connect_jumps=True, ax=None, **kwargs):
    """
    Parameter
    ---------
    ax: matplotlib.axes
    y: array_like
        y-Measurements
    color_s: array_like
        Same length as y, colors for each Measurements point.
    x:
        x-Measurements
    connect_jumps: bool
        If True, the border between two different colors is drawn 50 / 50 with both neighboring colors
    **d:
        Additional keyword arguments for plt.plot()
    """

    if ax is None:
        ax = plt.gca()

    n = len(y)
    if x is None:
        x = range(n)

    i = 0
    h = []
    while i < n:
        cur_col = color_s[i]
        j = i + 1
        while j < n and color_s[j] == cur_col:
            j += 1

        h.append(ax.plot(x[i:j], y[i:j], c=cur_col, **kwargs)[0])

        if connect_jumps:
            h.append(line_2colored(ax, x[j - 1:j + 1], y[j - 1:j + 1], colors=color_s[j - 1:j + 1], **kwargs))
        i = j

    return h


def line_2colored(x, y, colors, ax=None, **kwargs):
    """
    Plot a line with 2 colors.
    Parameter
    ---------
    ax: matplotlib.axes
    x:
        x-Measurements, 2 points
    y:
        y-Measurements, 2 points
    colors:
        2 colors. First is for the first half of the line and the second color for the second part of the line
    **d:
        Additional keyword-arguments for plt.plot()
    """

    if ax is None:
        ax = plt.gca()

    if type(x[0]) is datetime.date or type(x[0]) is datetime.datetime:
        xh = datetime.datetime(year=x[0].year, month=x[0].month, day=x[0].day)
        xh += (x[1] - x[0]) / 2
    else:
        xh = np.mean(x)
    yh = np.mean(y)

    h = [ax.plot([x[0], xh], [y[0], yh], c=colors[0], **kwargs)[0],
         ax.plot([xh, x[1]], [yh, y[1]], c=colors[1], **kwargs)[0]]
    return h


def color_plot(x, y, color_s, plot_fcn, **kwargs):
    """
    Plot a line with an individual color for each point.
    :param x: Data for x-axis
    :param y: Data for y-axis
    :param color_s: array of colors with the same length as x and y respectively. If now enough colors are given,
                    use just the first (only) one given
    :param plot_fcn: Matplotlib function, which should be used for plotting -> use ax.METHOD to ensure that the right
                     axis is used
    :param kwargs: Additional d for matplotlib.pyplot.plot()
    """

    h = []
    for i in range(len(x)):
        c = color_s[i%len(color_s)]
        h.append(plot_fcn([x[i]], [y[i]], color=c, **kwargs))

    return h


def draw_lines_between(*, x1=None, x2=None, y1, y2, ax=None, **kwargs):
    # https://stackoverflow.com/questions/59976046/connect-points-with-horizontal-lines
    ax = ax or plt.gca()
    x1 = x1 if x1 is not None else np.arange(len(y1))
    x2 = x2 if x2 is not None else x1

    cl = collections.LineCollection(np.stack((np.c_[x1, x2], np.c_[y1, y2]), axis=2), **kwargs)
    ax.add_collection(cl)
    return cl


# Grid
def grid_lines(ax, start, step, limits, **kwargs):
    mins, maxs = limits4axes(limits=limits, n_dim=2)
    start = tuple_extract(t=start, default=(0, 0), mode='repeat')
    step = tuple_extract(t=step, default=(0, 0), mode='repeat')

    ax.hlines(y=np.arange(start=start[0], stop=maxs[1], step=step[0]), xmin=mins[0], xmax=maxs[0], **kwargs)
    ax.vlines(x=np.arange(start=start[1], stop=maxs[0], step=step[1]), ymin=mins[1], ymax=maxs[1], **kwargs)


def hvlines_grid(ax, x, limits='ax', **kwargs):
    if limits == 'ax':
        limits = np.array([ax.get_xlim(), ax.get_ylim()])
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
    elif limits == 'data':
        limits = np.array([[x[:, 0].min(), x[:, 0].max()],
                           [x[:, 1].min(), x[:, 1].max()]])
    else:
        raise ValueError

    ax.vlines(x[..., 0].ravel(), ymin=limits[1, 0], ymax=limits[1, 1], **kwargs)
    ax.hlines(x[..., 1].ravel(), xmin=limits[0, 0], xmax=limits[0, 1], **kwargs)


def update_vlines(*, h, x, ymin=None, ymax=None):
    seg_old = h.get_segments()
    if ymin is None:
        ymin = seg_old[0][0, 1]
    if ymax is None:
        ymax = seg_old[0][1, 1]

    seg_new = [np.array([[xx, ymin],
                         [xx, ymax]]) for xx in x]

    h.set_segments(seg_new)


def update_hlines(*, h, y, xmin=None, xmax=None):
    seg_old = h.get_segments()
    if xmin is None:
        xmin = seg_old[0][0, 0]
    if xmax is None:
        xmax = seg_old[0][1, 0]

    seg_new = [np.array([[xmin, yy],
                         [xmax, yy]]) for yy in y]

    h.set_segments(seg_new)


def hist_vlines(x, name, bins=100,
                hl_idx=None, hl_color=None, hl_name=None,
                lower_perc=None, upper_perc=None):

    if lower_perc is not None:
        _range = (np.percentile(x, lower_perc), np.percentile(x, upper_perc))
    else:
        _range = None
    fig, ax = new_fig(title=f'Histogram: {name}', scale=2)
    hist = ax.hist(x, bins=bins, range=_range)

    perc_i = []
    if hl_idx is not None:
        hl_idx, hl_color, hl_name = safe_scalar2array(hl_idx, hl_color, hl_name, shape=np.size(hl_idx))
        for i, c, n in zip(hl_idx, hl_color, hl_name):
            perc_i.append(np.sum(x[i] > x))
            label = None if n is None else f"{n} | {perc_i[-1]} / {len(x)}"
            ax.vlines(x[i], ymin=0, ymax=len(x), lw=4, color=c, label=label)

    ax.set_ylim(0, hist[0].max() * 1.02)

    if lower_perc is not None:
        ax.set_xlim(np.percentile(x, lower_perc), np.percentile(x, upper_perc))

    if hl_name is not None:
        ax.legend()
    return ax, perc_i

