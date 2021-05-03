import numpy as np
import mpl_toolkits.mplot3d.art3d as art3d
from itertools import combinations, product
from wzk.mpl.figure import save_fig


def save_different_views_3d(*, ax, fig_name, azim_lim=(0, 270), elev_lim=(45, 45),
                            n=None, n_azim=None, n_elev=None):
    """https://github.com/matplotlib/matplotlib/issues/1077/"""
    # TODO option to save as gif (FuncAnimation)
    n_azim = n if n is None else n_azim
    n_elev = n if n is None else n_elev

    def range_wrapper(limits, n_steps):
        return np.linspace(limits[0], limits[1], n_steps) if limits[0] != limits[1] else [limits[0]]

    azim_lim = range_wrapper(limits=azim_lim, n_steps=n_azim)
    elev_lim = range_wrapper(limits=elev_lim, n_steps=n_elev)

    for a in azim_lim:
        for e in elev_lim:
            ax.azim = a
            ax.elev = e
            save_fig(fig_name + '_a{:.4}_e{:.4}'.format(a, e), ax.get_figure())


def plot_poly_collection_3d(face_vtx, ax, **kwargs):
    n_faces = face_vtx.shape[0]
    for i in range(n_faces):
        face = art3d.Poly3DCollection([face_vtx[i, :, :]], **kwargs)
        ax.add_collection3d(face)


def plot_cube_3d(ax, lower_left, side_length, **kwargs):
    r = [[ll, ll + side_length] for ll in lower_left]
    for s, e in combinations(np.array(list(product(*r))), 2):
        if np.sum(np.abs(s - e)) == side_length:
            ax.plot3D(*zip(s, e), **kwargs)
