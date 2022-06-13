from typing import Optional, Union, List

import matplotlib.pyplot as plt
from matplotlib.axes._subplots import Subplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
sns.set_style('white')
from torch import Tensor
import xarray as xr

Array = Union[np.ndarray, Tensor, xr.DataArray]


## TODO: colorbar issue with float single-channel images.
def show(
    x : Union[Array, List[Array]],
    ax : Optional[Union[Subplot, List[Subplot]]] = None,
    rows : Optional[int] = None,
    columns : Optional[int] = None,
    order : str = 'R',
    title : Optional[Union[str, List[str]]] = None,
    vmax : Optional[float] = None,
    normalize : bool = True,
    fontsize : int = 15,
    colorbar : bool = False,
    axis : bool = True,
    **figure_kwargs,
) -> None:
    '''
    Does a bunch of `plt.imshow`s with commonly used arguments.
    '''

    if order not in ['C', 'R']:
        raise ValueError(f"Values expected for `order`: ['C'|'R']. Got '{order}'.")

    def tonumpy(x):
        if isinstance(x, xr.DataArray):
            return x.data
        elif isinstance(x, Tensor):
            return x.numpy()
        elif isinstance(x, np.ndarray):
            return x

    if not isinstance(x, list):
        x = [x]

    G = []
    dates = []
    for g in x:  # List of 2D / 3D / 4D arrays
        if hasattr(g, 'time'):
            time = g.time
            if time.ndim == 0:
                time = time.expand_dims(dim='time')
            time = time.to_index().astype(str)
        else:
            time = [None] * len(g)
        dates.extend(time)
        g = tonumpy(g)
        if g.ndim <= 2:
            G += [g]
        if g.ndim == 3:
            G += [g.transpose(1, 2, 0)]
        elif g.ndim == 4:
            G += [img.transpose(1, 2, 0) for img in g]

    n = len(G)

    if rows is None and columns is None:
        if order == 'R':
            rows, columns = 1, (n or 1)
        elif order == 'C':
            rows, columns = (n or 1), 1
    elif rows is None:
        rows = int(np.ceil(n / columns))
    elif columns is None:
        columns = int(np.ceil(n / rows))

    s = figure_kwargs.get('figsize', 3)
    figure_kwargs['figsize'] = (columns * s, rows * s)

    if ax is None:
        _, ax = plt.subplots(rows, columns, tight_layout=True, squeeze=False, **figure_kwargs)
        if order == 'C':
            ax = ax.T
        ax = ax.flatten()
    else:
        pass

    if isinstance(title, (xr.DataArray, np.ndarray, Tensor)):
        title = [f'{x:.4f}' for x in title.data]
    elif isinstance(title, list):
        title = [str(x) for x in title]
    elif title is None:
        title = [None] * n if n else None

    for g, ax_, t, date in zip(G, ax, title, dates):
        g = np.nan_to_num(g)
        unique_vals = np.unique(g)
        if len(unique_vals) == 0:
            ax_.imshow([[1]], cmap='gray', vmin=0, vmax=1);
            ax_.axis(False)
            return
        m = (g.max() or 1.0) if vmax is None else vmax
        cmap = plt.get_cmap().name
        if g.dtype.type is np.bool_:
            cmap = plt.cm.get_cmap('binary_r', 2)
            im = ax_.imshow(g, interpolation='nearest', cmap=cmap, vmin=0, vmax=1);
        else:
            if len(unique_vals) > 1 and normalize:
                g = g / m
                _vmin, _vmax = 0, g.max()
            else:
                _vmin, _vmax = g.min(), g.max()
            im = ax_.imshow(g, interpolation='nearest', vmin=_vmin, vmax=_vmax);
            ax_.axis(axis)
        t = t or date
        ax_.set_title(t, fontsize=fontsize);
        if colorbar and (g.shape[-1] == 1 or g.ndim == 2):
            divider = make_axes_locatable(ax_)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im.set_cmap(cmap)

            if len(unique_vals) <= 20:
                cbar = plt.colorbar(im, cax=cax, orientation='vertical', ticks=np.unique(g))
                ticklabels = [f"{x:.0f}" if x % 1 == 0 else f"{x:.3f}" for x in unique_vals]
                cbar.ax.set_yticklabels(ticklabels)
            else:
                cbar = plt.colorbar(im, cax=cax, orientation='vertical')
