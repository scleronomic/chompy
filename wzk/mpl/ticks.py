import numpy as np

from matplotlib import transforms
from wzk.dicts_lists_tuples import atleast_list
from wzk.numpy2 import np_isinstance


def turn_ticklabels_off(ax, axes='xy'):
    """Remove the tick labels for all specified axes"""
    if 'x' in axes:
        ax.axes.xaxis.set_ticklabels([])
    if 'y' in axes:
        ax.axes.yaxis.set_ticklabels([])
    if 'z' in axes:
        try:
            ax.axes.zaxis.set_ticklabels([])
        except AttributeError:
            pass


def turn_ticks_off(ax):
    turn_ticklabels_off(ax=ax)
    set_ticks_position(ax=ax, position='none')


def position2tlbr(position='default'):
    """map keywords to booleans indicating which axis are active"""
    bottom = top = left = right = False
    if position == 'all':
        bottom = top = left = right = True

    elif position == 'default':
        bottom = left = True

    elif position == 'inverse':
        top = right = True

    elif position == 'none':
        pass

    else:
        if 'bottom' in position:
            bottom = True
        if 'top' in position:
            top = True
        if 'left' in position:
            left = True
        if 'right' in position:
            right = True

    return top, left, bottom, right


def tlbr2positions(top, left, bottom, right):

    position_x = 'none'
    if bottom and top:
        position_x = 'both'
    elif bottom:
        position_x = 'bottom'
    elif top:
        position_x = 'top'

    position_y = 'none'
    if left and right:
        position_y = 'both'
    elif left:
        position_y = 'left'
    elif right:
        position_y = 'right'

    return position_x, position_y


def set_ticks_position(ax, position):

    position_x, position_y = tlbr2positions(*position2tlbr(position=position))

    ax.axes.xaxis.set_ticks_position(position=position_x)
    ax.axes.yaxis.set_ticks_position(position=position_y)

    # try:
    #     ax.axes.zaxis.set_ticks_position(position=position[2])
    # except AttributeError:
    #     pass


def set_labels_position(ax, position):

    top, left, bottom, right = position2tlbr(position=position)
    ax.tick_params(labeltop=top, labelleft=left, labelbottom=bottom, labelright=right)


def get_ticks(ax, axis='x'):

    def ticks2arr(tt):
        return np.array([t for t in tt])

    if axis == 'x':
        return ticks2arr(ax.get_xticks())
    elif axis == 'y':
        return ticks2arr(ax.get_yticks())
    elif axis == 'xy' or 'both':
        return ticks2arr(ax.get_xticks()), ticks2arr(ax.get_yticks())


def get_ticklabels(ax, axis='x'):

    def labels2arr(ll):
        return np.array([l.get_text() for l in ll], dtype=object)

    if axis == 'x':
        return labels2arr(ax.get_xticklabels())
    elif axis == 'y':
        return labels2arr(ax.get_yticklabels())
    elif axis == 'xy' or 'both':
        return labels2arr(ax.get_xticklabels()), labels2arr(ax.get_yticklabels())


def get_ticks_index(ax, axis, v, squeeze=True):
    assert axis == 'x' or axis == 'y'
    t = get_ticks(ax=ax, axis=axis)

    v = np.atleast_1d(v).copy()

    if np_isinstance(v, float):
        for i, vv in enumerate(v):
            temp = np.nonzero(np.isclose(t, vv))[0]
            v[i] = np.nan if np.size(temp) == 0 else temp[0]

    elif np_isinstance(v, int):
        pass

    else:
        raise ValueError

    v = v.astype(int)
    if squeeze:
        v = np.squeeze(v)
    return v


def remove_ticks(ax, v, axis='x'):
    """
    Remove the ticks corresponding to values in v from the axis.
    If the values are not found they are ignored.
    """

    assert axis in ('x', 'y', 'xy', 'both')

    def __remove_ticks(_axis):
        ticks = get_ticks(ax=ax, axis=_axis)
        i = get_ticks_index(ax=ax, axis=_axis, v=v, squeeze=False)
        i = i[i != np.nan]

        ticks2 = np.delete(ticks, i)
        return ticks2

    if 'x' in axis or axis == 'both':
        ax.set_xticks(__remove_ticks(_axis='x'))
    if 'y' in axis or axis == 'both':
        ax.set_yticks(__remove_ticks(_axis='y'))


def change_tick_appearance(ax, position, v, size=None, color=None):
    """
    x: bottom, top
    y: left, right
    """

    def __apply(h_i, s=None, c=None):
        if s is not None:
            h_i.set_markersize(s)
        if c is not None:
            h_i.set_color(c)

    # Handle different positions and axis
    p_list = ['bottom', 'top', 'left', 'right']
    ip = p_list.index(position)
    axis = 'x' if ip//2 == 0 else 'y'
    h = ax.xaxis.get_majorticklines() if axis == 'x' else ax.yaxis.get_majorticklines()

    idx = get_ticks_index(ax=ax, axis=axis, v=v, squeeze=False)
    idx = np.array([np.ravel_multi_index((i, ip % 2), (len(h) // 2, 2)) for i in idx])

    for i in idx:
        __apply(h_i=h[i], s=size, c=color)


def add_ticks(*, ax, ticks, labels=None, axis='x'):

    assert axis in ('x', 'y', 'xy', 'both')
    if ticks is None or np.size(ticks) == 0:
        return

    def __add_ticks(_axis):
        ticks_old = get_ticks(ax=ax, axis=_axis)
        labels_old = get_ticklabels(ax=ax, axis=_axis)

        _ticks = np.hstack((ticks_old, atleast_list(ticks)))
        sort_idx = np.argsort(_ticks)
        _ticks = _ticks[sort_idx].tolist()

        if any(np.atleast_1d(labels)):
            # if len(labels_old) > len(ticks_old):  # two axes (bottom, top), (left, right)

            _labels = np.hstack((labels_old[:len(ticks_old)], atleast_list(labels)))
            _labels = _labels[sort_idx].tolist()
        else:
            _labels = None

        set_ticks_and_labels(ax=ax, axis=axis, ticks=_ticks, labels=_labels)

    if 'x' in axis or axis == 'both':
        __add_ticks(_axis='x')

    if 'y' in axis or axis == 'both':
        __add_ticks(_axis='y')


def set_ticks_and_labels(*, ax, ticks=None, labels=None, axis='x'):

    def __set_ticks(_set_ticks, _set_ticklabels, _ticks, _labels):

        if any(np.atleast_1d(_ticks)):
            _set_ticks(np.array(_ticks).tolist())

        if any(np.atleast_1d(_labels)):
            _set_ticklabels(np.array(_labels).tolist())

    assert axis in ('x', 'y', 'xy', 'both')

    if 'x' in axis or axis == 'both':
        __set_ticks(_set_ticks=ax.set_xticks, _set_ticklabels=ax.set_xticklabels, _ticks=ticks, _labels=labels)

    if 'y' in axis or axis == 'both':
        __set_ticks(_set_ticks=ax.set_yticks, _set_ticklabels=ax.set_yticklabels, _ticks=ticks, _labels=labels)


def transform_tick_labels(*, ax, xt=0., yt=0., rotation=0., axis, ha=None, va=None):
    offset = transforms.ScaledTranslation(xt=xt, yt=yt, scale_trans=ax.get_figure().dpi_scale_trans)

    if rotation != 0:
        ax.tick_params(axis=axis, rotation=rotation)

    def __set_alignment(lbl):
        if va is not None:
            lbl.set_va(va)
        if ha is not None:
            lbl.set_ha(ha)

    if 'x' in axis or 'both' in axis:
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
            __set_alignment(lbl=label)
    if 'y' in axis or 'both' in axis:
        for label in ax.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
            __set_alignment(lbl=label)

