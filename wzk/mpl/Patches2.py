import numpy as np
from matplotlib import patches, transforms, pyplot


class RelativeFancyArrow(patches.FancyArrow):
    def __init__(self, x, y, dx, dy, width=0.2, length_includes_head=True, head_width=0.3, head_length=0.4,
                 shape='full', overhang=0, head_starts_at_zero=False, **kwargs):
        length = np.hypot(dx, dy)
        super().__init__(x, y, dx, dy, width=width*length, length_includes_head=length_includes_head*length,
                         head_width=head_width*length, head_length=head_length*length,
                         shape=shape, overhang=overhang, head_starts_at_zero=head_starts_at_zero, **kwargs)


class FancyArrowX2(patches.FancyArrow):
    def __init__(self, xy0, xy1, offset0=0.0, offset1=0.0,
                 width=0.2, head_width=0.3, head_length=0.4, overhang=0,
                 shape='full',  length_includes_head=True, head_starts_at_zero=False, **kwargs):

        xy0, xy1 = np.array(xy0), np.array(xy1)
        dxy = xy1 - xy0
        dxy /= np.linalg.norm(dxy, keepdims=True)

        xy0 += offset0 * dxy

        dxy = xy1 - xy0
        dxy -= dxy / np.linalg.norm(dxy) * offset1

        super().__init__(*xy0, *dxy, width=width, length_includes_head=length_includes_head,
                         head_width=head_width, head_length=head_length,
                         shape=shape, overhang=overhang, head_starts_at_zero=head_starts_at_zero, **kwargs)


class FancyBbox(patches.FancyBboxPatch):
    def __init__(self, xy, width, height, boxstyle='Round', pad=0.3, corner_size=None, **kwargs):
        if boxstyle in ['Roundtooth', 'Sawtooth']:
            bs = patches.BoxStyle(boxstyle, pad=pad, tooth_size=corner_size)
        elif boxstyle in ['Round', 'Round4']:
            bs = patches.BoxStyle(boxstyle, pad=pad, rounding_size=corner_size)
        else:
            bs = patches.BoxStyle(boxstyle, pad=pad)

        super().__init__(xy=(xy[0]+pad, xy[1]+pad), width=width - 2*pad, height=height - 2*pad, boxstyle=bs, **kwargs)


# Transformations
def do_aff_trafo(patch, theta, xy=None, por=(0, 0)):
    if xy is not None:
        patch.set_xy(xy=xy)
    patch.set_transform(get_aff_trafo(theta=theta, por=por, patch=patch))


def get_aff_trafo(xy0=None, xy1=None, theta=0, por=(0, 0), ax=None, patch=None):
    """

    :param xy0: current position of the object, if not provided patch.get_xy() is used
    :param xy1: desired position of the object
    :param theta: rotation in degrees
    :param por: point of rotation relative to the objects coordinates
    :param ax:
    :param patch:
    :return:
    """

    if xy0 is None:
        if patch is None:
            xy0 = (0, 0)
        else:
            xy0 = patch.get_xy()

    if xy1 is None:
        xy1 = xy0

    if ax is None:
        if patch is None:
            ax = pyplot.gca()
        else:
            ax = patch.axes

    return (transforms.Affine2D().translate(-xy0[0]-por[0], -xy0[1]-por[1])
                                 .rotate_deg_around(0, 0, theta)
                                 .translate(xy1[0], xy1[1]) + ax.transData)

