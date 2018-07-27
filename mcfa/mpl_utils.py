
# A matplotlib style based on the gala package by @adrn:
# github.com/adrn/gala

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

mpl_style = {

    # Lines
    'lines.linewidth': 1.7,
    'lines.antialiased': True,
    'lines.marker': '.',
    'lines.markersize': 5.,

    # Patches
    'patch.linewidth': 1.0,
    'patch.facecolor': '#348ABD',
    'patch.edgecolor': '#CCCCCC',
    'patch.antialiased': True,

    # images
    'image.origin': 'upper',

    # colormap
    'image.cmap': 'viridis',

    # Font
    'font.size': 12.0,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.latex.preview': True,
    'axes.unicode_minus': False,

    # Axes
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'axes.titlesize': 'x-large',
    'axes.labelsize': 'large',
    'axes.labelcolor': 'k',
    'axes.axisbelow': True,

    # Ticks
    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'xtick.major.pad': 6,
    'xtick.minor.pad': 6,
    'xtick.color': '#333333',
    'xtick.direction': 'in',
    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'ytick.major.pad': 6,
    'ytick.minor.pad': 6,
    'ytick.color': '#333333',
    'ytick.direction': 'in',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',

    # Legend
    'legend.fancybox': True,
    'legend.loc': 'best',

    # Figure
    'figure.figsize': [6, 6],
    'figure.facecolor': '1.0',
    'figure.edgecolor': '0.50',
    'figure.subplot.hspace': 0.5,

    # Other
    'savefig.dpi': 300,
}

matplotlib.style.use(mpl_style)


def corner_hist(X, bins=30, label_names=None, show_ticks=False, fig=None, 
                figsize=None, **kwargs):
    """
    Make a corner plot with a 2D histogram in each axes.

    :param X:
        The data, :math:`X`, which is expected to be an array of shape
        [n_samples, n_features].

    :param bins: [optional]
        The number of bins to use in each dimension of the histogram.

    :param label_names: [optional]
        The label names to use for each feature.

    :param show_ticks: [optional]
        Show ticks on the axes.

    :param fig: [optional]
        Supply a figure (with [n_features, n_features] axes) to plot the data.

    :param figsize: [optional]
        Specify a size for the figure.  This parameter is ignored if a `fig` is
        supplied.

    :returns:
        A figure with a corner plot showing the data.
    """

    N, D = X.shape
    A = D - 1

    if figsize is None:
        figsize = (2 * A, 2 * A)
    
    if fig is None:
        if figsize is None:
            figsize = (2 * A, 2 * A)
        fig, axes = plt.subplots(A, A, figsize=figsize)
        
    else:
        axes = fig.axes

    kwds = dict(cmap="Greys", norm=LogNorm())
    kwds.update(kwargs)
    
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            if j >= i:
                ax.set_visible(False)
                continue

            H, xedges, yedges, binnumber = binned_statistic_2d(
                X.T[i], X.T[j], X.T[i], statistic="count", bins=bins)

            imshow_kwds = dict(
                aspect=np.ptp(xedges)/np.ptp(yedges), 
                extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]))
            imshow_kwds.update(kwds)
            
            image = ax.imshow(H.T, **imshow_kwds)
            ax.set_ylim(ax.get_ylim()[::-1])
        

            if ax.is_last_row() and label_names is not None:
                ax.set_xlabel(label_names[j])
                
            if ax.is_first_col() and label_names is not None:
                ax.set_ylabel(label_names[i])
                
            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
                
    fig.tight_layout()
    
    return fig


def corner_scatter(X, label_names=None, show_ticks=False, fig=None, figsize=None,
                   **kwargs):
    """
    Make a corner plot where the data are shown as scatter points in each axes.

    :param X:
        The data, :math:`X`, which is expected to be an array of shape
        [n_samples, n_features].

    :param label_names: [optional]
        The label names to use for each feature.

    :param show_ticks: [optional]
        Show ticks on the axes.

    :param fig: [optional]
        Supply a figure (with [n_features, n_features] axes) to plot the data.

    :param figsize: [optional]
        Specify a size for the figure. This parameter is ignored if a `fig` is
        supplied.

    :returns:
        A figure with a corner plot showing the data.
    """

    N, D = X.shape
    K = D 

    if fig is None:
        if figsize is None:
            figsize = (2 * K, 2 * K)
        fig, axes = plt.subplots(K, K, figsize=figsize)
        
    else:
        axes = fig.axes
    
    kwds = dict(s=1, c="tab:blue", alpha=0.5)
    kwds.update(kwargs)
    
    axes = np.atleast_2d(axes).T
    
    for j, y in enumerate(X.T):
        for i, x in enumerate(X.T):
            
            try:
                ax = axes[K - i - 1, K - j - 1]
            
            except:
                continue
            
            if j >= i: 
                ax.set_visible(False)
                continue
            
            ax.scatter(x, y, **kwds)
            
            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])

            if ax.is_last_row() and label_names is not None:
                ax.set_xlabel(label_names[i])
                
            if ax.is_first_col() and label_names is not None:
                ax.set_ylabel(label_names[j])
                
    fig.tight_layout()
    
    return fig