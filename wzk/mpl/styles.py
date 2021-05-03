import matplotlib as mpl
from wzk.mpl.figure import ieee1c, ieee2c


def __no_borders(pad=0.0):
    return {'figure.subplot.left': 0.0 + pad,
            'figure.subplot.right': 1.0 - pad,
            'figure.subplot.bottom': 0.0 + pad,
            'figure.subplot.top': 1.0 - pad}


def set_style(s=('ieee', )):
    params = {}
    if 'ieee' in s:
        params.update({
            # 'text.usetex': False,
            'font.family': 'serif',
            'font.serif':  ['Times New Roman'],  # , ['Times', 'Times New Roman']
            'font.size': 8,
            'axes.linewidth': 1,
            'axes.labelsize': 8,
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,

            # 'figure.figsize': ieee2c,
            # 'figure.dpi': 300,

            'savefig.dpi': 300,
            'savefig.bbox': 'standard',
            'savefig.pad_inches': 0.1,
            'savefig.transparent': True
        })

    if 'no_borders' in s:
        params.update(__no_borders(pad=0.005))

    mpl.rcParams.update(params)
