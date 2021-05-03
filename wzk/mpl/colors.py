import numpy as np
import matplotlib as mpl

# Use :
# import cycler
# cycler('color', x)

# https://colorhunt.co/palette/15697
pallet_campfire = ['#311d3f', '#522546', '#88304e', '#e23e57']

# https://colorhunt.co/palette/108152
pallet_sunset = ['#3a0088', '#930077', '#e61c5d', '#ffbd39']

# https://colorhunt.co/palette/1504
pallet_blues4 = ['#48466d', '#3d84a8', '#46cdcf', '#abedd8']

# https://learnui.design/tools/data-color-picker.html#single
pallet_blues10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# TUM corporate design colors - http://portal.mytum.de/corporatedesign/index_print/vorlagen/index_farben
pallet_tum = {'blue_3': '#0065BD',  # Blue ordered from light to dark
              'blue_4':  '#005293',
              'blue_5':  '#003359',
              'blue_2':  '#64A0C8',
              'blue_1':  '#98C6EA',
              'grey_80': '#333333',
              'grey_50': '#808080',
              'grey_20': '#CCCCC6',
              'beige':   '#DAD7CB',
              'green':   '#A2AD00',
              'orange':  '#E37222'}


tum_blues5 = [pallet_tum['blue_5'],
              pallet_tum['blue_4'],
              pallet_tum['blue_3'],
              pallet_tum['blue_2'],
              pallet_tum['blue_1']]

tum_mix5 = [pallet_tum['blue_3'],
            pallet_tum['orange'],
            pallet_tum['beige'],
            pallet_tum['green'],
            pallet_tum['grey_50']]

# sashamaps
# https://sashamaps.net/docs/resources/20-colors/
_20 = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
       '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9']
_16 = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#fabed4',
       '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#000075', '#a9a9a9']
_7 = ['#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000', '#000075', '#a9a9a9']


def arr2rgba(*, img, cmap, vmin=None, vmax=None, mask=None, axis_order=None):
    img = __arr2rgba(arr=img, cmap=cmap, vmin=vmin, vmax=vmax)
    if mask is not None:
        img[mask.astype(bool), 3] = 0
    if axis_order == 'ij->yx':
        img = np.swapaxes(img, axis1=0, axis2=1)
    return img


def __arr2rgba(arr, cmap, vmin=None, vmax=None):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return mpl.cm.ScalarMappable(cmap=cmap, norm=norm).to_rgba(arr, bytes=True)

