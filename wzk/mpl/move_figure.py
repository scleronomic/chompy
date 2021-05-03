import numpy as np

try:
    import screeninfo
except ModuleNotFoundError:
    screeninfo = None
    monitors = [[1440, 900]]
if screeninfo is not None:
    try:
        monitors = [(m.fig_width_inch, m.height) for m in screeninfo.get_monitors()]
    except screeninfo.common.ScreenInfoError:
        # Fallback for my mac
        # monitors = [[2560, 1600]]
        monitors = [[1440, 900]]
        # from screeninfo import get_monitors, Enumerator
        # for matrix in get_monitors(Enumerator.OSX):
        #     print(str(matrix))
        # ( ModuleNotFoundError, NotImplementedError, NameError, screeninfo.common.ScreenInfoError):


def move_fig(fig, position=None, monitor=-1):
    """
    FROM
    http://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib#19943546
    With small change regarding the display resolution
    Move and resize a window to a set of standard positions on the screen.
    Possible positions are:
    top, bottom, left, right, top left, top right, bottom left, bottom right
    (n_rows, n_cols, index) (as for subplot) index starts at 1
    (offset_x, offset_y)
    """

    if position is None:
        return

    # Assumption the monitors are ordered from left to right and are aligned at the upper boundary
    offset_x_monitors = 0
    for m in monitors[:monitor]:
        offset_x_monitors += m[0]

    # Convert the position into the format (n_rows, n_cols, index)
    if isinstance(position, tuple):
        if len(position) == 2:
            offset_x, offset_y = position
            __move_fig(fig=fig, offset_x=offset_x + offset_x_monitors, offset_y=offset_y)
            return

        elif len(position) == 3:
            n_rows, n_cols, index = position

        else:
            raise NotImplementedError("Wrong input for 'position'")

    elif isinstance(position, int):
        n_rows, n_cols, index = [int(i) for i in str(position)]

    elif isinstance(position, str):
        n_rows, n_cols, index = __position_string_wrapper(position_str=position)

    else:
        raise NotImplementedError("Wrong input for 'position'")

    # Convert (n_rows, n_cols, index) into (fig_width_inch, height, offset_x, offset_y) for the figure
    screen_width, screen_height = monitors[monitor]
    fig_width = screen_width // n_cols
    fig_height = screen_height // n_rows

    offset_y, offset_x = np.unravel_index(index-1, dims=(n_rows, n_cols))
    offset_x *= fig_width
    offset_y *= fig_height

    offset_x += offset_x_monitors
    __move_fig(fig, width=fig_width, height=fig_height, offset_x=offset_x, offset_y=offset_y)


def __position_string_wrapper(position_str='top right'):
    """
    Convert the string into the format n_rows, n_cols, index
    """
    if position_str == 'full':
        return 1, 1, 1

    elif position_str == 'top':
        return 2, 1, 1
    elif position_str == 'bottom':
        return 2, 1, 2

    elif position_str == 'left':
        return 1, 2, 1
    elif position_str == 'right':
        return 2, 2, 2

    elif position_str == 'top left':
        return 2, 2, 1
    elif position_str == 'top right':
        return 2, 2, 2
    elif position_str == 'bottom left':
        return 2, 2, 3
    elif position_str == 'bottom right':
        return 2, 2, 4
    else:
        raise NotImplementedError("Unknown position '{}', "
                                  "Choose from: 'top', 'bottom', 'left', 'right', "
                                  "'top left', 'top right', 'bottom left', 'bottom right'".format(position_str))


def __move_fig(fig, width=None, height=None, offset_x=None, offset_y=None):
    """
    Move figure's upper left corner to pixel (x, y)
    Coordinate system starts in upper left corner of the monitor
    """

    # Get the current position of the figure 'fig_width_inch'x'height'+'offset_x'+'offset_y'
    current_geometry = fig.canvas.manager.window.wm_geometry()
    c_width, current_geometry = current_geometry.split('x')
    c_height, c_offset_x, c_offset_y = current_geometry.split('+')

    # Adjust the geometry if a new value is given
    if width is None:
        width = c_width
    if height is None:
        height = c_height
    if offset_x is None:
        offset_x = c_offset_x
    if offset_y is None:
        offset_y = c_offset_y

    fig.canvas.manager.window.wm_geometry(f"{width}x{height}+{offset_x}+{offset_y}")


if __name__ == '__main__':
    import matplotlib as mpl

    mpl.use('TkAgg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    move_fig(fig=fig, position='top right')

    fig, ax = plt.subplots()
    move_fig(fig=fig, position='top left')

    fig, ax = plt.subplots()
    move_fig(fig=fig, position='bottom right')

    fig, ax = plt.subplots()
    move_fig(fig=fig, position='bottom left')
    # plt.pause(0.1)
    # plt.close('all')

    for i in range(1,10):
        fig, ax = plt.subplots()
        move_fig(fig=fig, position=(3, 3, i))

    # plt.pause(0.1)
    # plt.close('all')
