import numpy as np
import matplotlib.patches as patches

from wzk.mpl.axes import get_aspect_ratio
from wzk.numpy2 import scalar2array, max_size


class DraggablePatch:
    lock = None  # only one can be animated at a time

    def __init__(self, ax, vary_xy=(True, True), limits=None, callback=None, **kwargs):
        ax.add_patch(self)

        self.vary_xy = np.array(vary_xy)
        self.callback_drag = callback  # Patches already have an attribute callback, add_callback()
        self.limits = limits

        self.press = None
        self.background = None

        # Connections
        self.cid_press = None
        self.cid_release = None
        self.cid_motion = None
        self.connect()

    def set_callback_drag(self, callback):
        self.callback_drag = callback

    def add_callback_drag(self, callback):
        if self.callback_drag is None:
            self.set_callback_drag(callback=callback)
        else:
            def cb2():
                self.callback_drag()
                callback()

            self.callback_drag = cb2

    def get_callback_drag(self):
        return self.callback_drag

    def get_xy_drag(self):
        raise NotImplementedError

    def set_xy_drag(self, xy):
        raise NotImplementedError

    def apply_limits(self, xy):
        if self.limits is not None and self.limits is not (None, None):
            return np.clip(xy, a_min=self.limits[:, 0], a_max=self.limits[:, 1])
        else:
            return xy

    def set_limits(self, limits=None):
        self.limits = limits
        self.set_xy_drag(xy=self.apply_limits(self.get_xy_drag()))

    def connect(self):
        self.cid_press = self.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def disconnect(self):
        self.figure.canvas.mpl_disconnect(self.cid_press)
        self.figure.canvas.mpl_disconnect(self.cid_release)
        self.figure.canvas.mpl_disconnect(self.cid_motion)

    def on_press(self, event):
        if event.inaxes != self.axes:
            return
        if DraggablePatch.lock is not None:
            return
        contains, attrd = self.contains(event)
        if not contains:
            return

        self.press = np.array(self.get_xy_drag()), np.array([event.xdata, event.ydata])
        DraggablePatch.lock = self

        # Draw everything but the selected rectangle and store the pixel buffer
        canvas = self.figure.canvas
        axes = self.axes
        self.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.axes.bbox)

        # Now redraw just the rectangle
        axes.draw_artist(self)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        if DraggablePatch.lock is not self:
            return
        if event.inaxes != self.axes:
            return

        xy_patch, xy_press = self.press

        dxy = np.array([event.xdata, event.ydata]) - xy_press

        new_xy_patch = xy_patch + self.vary_xy * dxy

        self.set_xy_drag(self.apply_limits(xy=new_xy_patch))

        canvas = self.figure.canvas
        axes = self.axes

        # restore the background region
        if self.background is not None:
            canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        """on release we reset the press Measurements"""
        if DraggablePatch.lock is not self:
            return

        self.press = None
        DraggablePatch.lock = None

        # turn off the rect animation property and reset the background
        self.set_animated(False)
        self.background = None

        if self.callback_drag is not None:
            self.callback_drag()

    def toggle_visibility(self, value=None):
        if value is None:
            self.set_visible(not self.get_visible())
        else:
            self.set_visible(bool(value))


class DraggableCircle(patches.Circle, DraggablePatch):
    def __init__(self,
                 ax,
                 xy, radius,
                 vary_xy=(True, True), callback=None,
                 limits=None,
                 **kwargs):
        patches.Circle.__init__(self, xy=xy, radius=radius, **kwargs)
        DraggablePatch.__init__(self, ax=ax, vary_xy=vary_xy, callback=callback, limits=limits)

    def get_xy_drag(self):
        return np.array(self.get_center()).flatten()

    def set_xy_drag(self, xy):
        self.set_center(xy=np.array(xy).flatten())


class DraggableEllipse(patches.Ellipse, DraggablePatch):
    def __init__(self,
                 ax,
                 xy, width, height, angle=0,
                 vary_xy=(True, True), callback=None, limits=None,
                 **kwargs):
        """
        If fig_width_inch or height are None,
        they are computed to form an circle for the aspect and Measurements ratio of the axis.
        """
        if width is None:
            width = get_aspect_ratio(ax) / ax.get_data_ratio() * height
        if height is None:
            height = ax.get_data_ratio() / get_aspect_ratio(ax) * width

        patches.Ellipse.__init__(self, xy=xy, width=width, height=height, angle=angle, **kwargs)
        DraggablePatch.__init__(self, ax=ax, vary_xy=vary_xy, callback=callback, limits=limits)

    def get_xy_drag(self):
        return np.array(self.get_center()).flatten()

    def set_xy_drag(self, xy):
        self.set_center(xy=np.array(xy).flatten())


class DraggableRectangle(patches.Rectangle, DraggablePatch):
    def __init__(self, *,
                 ax,
                 xy, width, height, angle=0,
                 vary_xy=(True, True), callback=None, limits=None,
                 **kwargs):
        patches.Rectangle.__init__(self, xy=xy, width=width, height=height, angle=angle, **kwargs)
        DraggablePatch.__init__(self, ax=ax, vary_xy=vary_xy, callback=callback, limits=limits)

    def get_xy_drag(self):
        return np.array(self.get_xy()).flatten()

    def set_xy_drag(self, xy):
        self.set_xy(xy=np.array(xy).flatten())


class DraggablePatchList:
    def __init__(self):
        self.dp_list = []

    def append(self, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item):
        return self.dp_list[item]

    def __n_index_wrapper(self, index, n):
        if index is None:
            index = np.arange(n)
        elif index == -1:
            index = np.arange(len(self.dp_list))
        n = int(np.max([n, np.size(index)]))

        return n, index

    @staticmethod
    def __value_wrapper(v, v_cur, n):
        if v is None:
            v = v_cur
        elif np.size(v) < n:
            v = np.full(n, v)

        if np.size(v) == 1:
            v = [v]

        return v

    def get_xy(self, index=-1):
        if index is None or (isinstance(index, int) and index == -1):
            return np.vstack([dp.get_xy_drag() for dp in self.dp_list])
        else:
            return np.vstack([dp.get_xy_drag() for i, dp in enumerate(self.dp_list) if i in index])

    def get_callback(self, index=-1):
        if index is None or (isinstance(index, int) and index == -1):
            return np.vstack([dp.get_callback_drag for dp in self.dp_list])
        else:
            return np.vstack([dp.get_callback_drag for i, dp in enumerate(self.dp_list) if i in index])

    def set_xy(self, x=None, y=None, xy=None, index=None):

        if xy is not None:
            x, y = xy.T

        n = max_size(x, y)
        n, index = self.__n_index_wrapper(index=index, n=n)

        xy_cur = self.get_xy(index=index)
        x = self.__value_wrapper(v=x, v_cur=xy_cur[:, 0], n=n)
        y = self.__value_wrapper(v=y, v_cur=xy_cur[:, 1], n=n)

        for ii, xx, yy in zip(index, x, y):
            self.dp_list[ii].set_xy_drag(xy=(xx, yy))

    def __set_or_add_callback(self, callback, index, mode):
        n = np.size(callback)
        n, index = self.__n_index_wrapper(index=index, n=n)
        callback_cur = self.get_callback(index=index)
        callback = self.__value_wrapper(v=callback, v_cur=callback_cur, n=n)

        if mode == 'set':
            for ii, cc in zip(index, callback):
                self.dp_list[ii].set_callback_drag(callback=cc)
        elif mode == 'add':
            for ii, cc in zip(index, callback):
                self.dp_list[ii].add_callback(cc)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def set_callback(self, callback, index=-1):
        self.__set_or_add_callback(callback=callback, index=index, mode='set')

    def add_callback(self, callback, index=-1):
        self.__set_or_add_callback(callback=callback, index=index, mode='add')

    def toggle_visibility(self, value=None):
        for dp in self.dp_list:
            dp.toggle_visibility(value=value)


class DraggableCircleList(DraggablePatchList):
    def __init__(self, ax, xy, radius, **kwargs):
        super().__init__()
        self.append(ax=ax, xy=xy, radius=radius, **kwargs)

    def append(self, ax, xy, radius, **kwargs):

        radius = scalar2array(radius, xy.shape[0])
        for xy_i, radius_i in zip(xy, radius):
            self.dp_list.append(DraggableCircle(ax=ax, xy=xy_i, radius=radius_i, **kwargs))


class DraggableEllipseList(DraggablePatchList):
    def __init__(self, ax,
                 xy, width, height, angle=0,
                 **kwargs):
        super().__init__()
        self.append(ax, xy=xy, width=width, height=height, angle=angle, **kwargs)

    def append(self, ax,
               xy, width, height, angle, **kwargs):
        width = scalar2array(width, xy.shape[0])
        height = scalar2array(height, xy.shape[0])
        angle = scalar2array(angle, xy.shape[0])

        for xy_i, width_i, height_i, angle_i in zip(xy, width, height, angle):
            self.dp_list.append(DraggableEllipse(ax=ax,
                                                 xy=xy_i, width=width_i, height=height_i, angle=angle_i, **kwargs))


class DraggableRectangleList(DraggablePatchList):
    def __init__(self, ax,
                 xy, width, height, angle=0,
                 **kwargs):
        super().__init__()
        self.append(ax, xy=xy, width=width, height=height, angle=angle, **kwargs)

    def append(self, ax,
               xy, width, height, angle, **kwargs):
        width = scalar2array(width, xy.shape[0])
        height = scalar2array(height, xy.shape[0])
        angle = scalar2array(angle, xy.shape[0])

        for xy_i, width_i, height_i, angle_i in zip(xy, width, height, angle):
            self.dp_list.append(DraggableRectangle(ax=ax,
                                                   xy=xy_i, width=width_i, height=height_i, angle=angle_i, **kwargs))
