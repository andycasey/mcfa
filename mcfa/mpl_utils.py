
# A matplotlib style based on the gala package by @adrn:
# github.com/adrn/gala

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib.colors import (BoundaryNorm, LinearSegmentedColormap, LogNorm)
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, FuncFormatter
from matplotlib.patches import Ellipse

from .utils import find_rotation_matrix, givens_rotation_matrix

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



def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    #return base.from_list(cmap_name, color_list, N)
    return LinearSegmentedColormap.from_list(cmap_name, color_list, N)

def plot_latent_factors(model, label_names=None):

    fig, ax = plt.subplots()

    A = model.theta_[model.parameter_names.index("A")]
    D, J = A.shape
    xi = np.arange(D)

    for j in range(J):
        ax.plot(xi, A.T[j], "-")

    ax.set_xticks(xi)
    if label_names is not None:
        ax.set_xticklabels(label_names)

    ylim = np.ceil(10 * np.abs(ax.get_ylim()).max()) / 10
    ax.plot([-0.5, D - 0.5], [0, 0], ":", c="#000000", zorder=-1, lw=0.5)
    ax.set_xlim(-0.5, D - 0.5)
    
    ax.set_ylim(-ylim, +ylim)
    ax.set_yticks([-ylim, 0, ylim])

    fig.tight_layout()

    return fig



def fill_between_steps(ax, x, y1, y2=0, h_align='mid', **kwargs):
    """
    Fill between for step plots in matplotlib.

    **kwargs will be passed to the matplotlib fill_between() function.
    """

    # If no Axes opject given, grab the current one:

    # First, duplicate the x values
    xx = x.repeat(2)[1:]
    # Now: the average x binwidth
    xstep = np.repeat((x[1:] - x[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep[-1])

    # Make it possible to chenge step alignment.
    if h_align == 'mid':
        xx = xx - xstep / 2.
    elif h_align == 'right':
        xx = xx - xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = y1.repeat(2)#[:-1]
    if type(y2) == np.ndarray:
        y2 = y2.repeat(2)#[:-1]

    # now to the plotting part:
    return ax.fill_between(xx, y1, y2=y2, **kwargs)


def plot_specific_scatter(model, y=None, scales=1, 
                          xlabel=None, xticklabels=None, ylabel=None,
                          steps=False, fill=True, line_kwds=None, fill_kwds=None):

    fig, ax = plt.subplots()

    if y is None:
        y = scales * np.sqrt(model.theta_[model.parameter_names.index("psi")])
    x = np.arange(y.size)

    kwds = dict(drawstyle="steps-mid", c="#000000", lw=2)
    kwds.update(line_kwds or dict())

    if steps:
        ax.plot(np.hstack([-1, x, x.size]),
                np.hstack([y[0], y, y[-1]]),
                "-", **kwds)
    else:
        ax.plot(x, y, "-", **kwds)

    if fill:
        kwds = dict(facecolor="#000000", alpha=0.1, zorder=-1)
        kwds.update(fill_kwds or dict())

        if steps:
            fill_between_steps(ax, x, np.zeros_like(y),  y, **kwds)
        else:
            ax.fill_between(x, np.zeros_like(y), y, **kwds)

    ax.set_xlim(-0.5, x.size - 0.5)
    ylim = ax.get_ylim()[1]
    ax.set_ylim(0, ylim)
    ax.yaxis.set_major_locator(MaxNLocator(5))

    ax.set_xticks(x)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(r"$\textrm{data dimension}$")

    if ylabel is None:
        ylabel = r"$\textrm{specific scatter}$"
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig




def plot_latent_space(model, X, ellipse_kwds=None, **kwargs):

    v, v_cluster, v_mean, tau = model.factor_scores(X)

    hard_associations = np.argmax(tau, axis=1)

    fig = corner_scatter(v_mean, c=hard_associations, **kwargs)

    xi = model.theta_[model.parameter_names.index("xi")] # latent factors, number of components.
    omega = model.theta_[model.parameter_names.index("omega")]

    L = int(max(1, int(len(fig.axes)))**0.5)
    axes = np.atleast_2d(fig.axes).reshape((L, L)).T

    kwds = dict(alpha=0.3, zorder=-1, rasterized=True)
    kwds.update(ellipse_kwds or dict())

    scat = axes[0, 0].collections[0]

    J = model.n_latent_factors
    for i in range(J):
        for j in range(J):
            if j == 0 or i >= j: continue

            try:
                ax = axes[i, j - 1]

            except:
                continue

            for k in range(model.n_components):
                mu = np.array([xi[i, k], xi[j, k]])
                cov = np.array([row[[i, j]] for row in omega[:, :, k]])[[i, j]]

                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals = vals[order]
                vecs = vecs[:, order]

                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * 2 * np.sqrt(vals)

                color = scat.to_rgba(k)

                ax.add_artist(Ellipse(xy=mu, width=width, height=height,
                                      angle=angle, facecolor=color, **kwds))

    return fig


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
    assert N > D, "I don't believe that you have more dimensions than data"

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
    assert N > D, "I don't believe that you have more dimensions than data"
    K = D - 1

    if fig is None:
        if figsize is None:
            figsize = (2 * K, 2 * K)
        fig, axes = plt.subplots(K, K, figsize=figsize)
    
    axes = np.array(fig.axes).reshape((K, K)).T

    kwds = dict(s=1, c="tab:blue", alpha=0.5, rasterized=True)
    kwds.update(kwargs)
    
    for i, x in enumerate(X.T):
        for j, y in enumerate(X.T):
            if j == 0: continue

            try:
                ax = axes[i, j - 1]

            except:
                continue

            if i >= j:
                ax.set_visible(False)
                continue

            ax.scatter(x, y, **kwds)

            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])

            else:
                if not ax.is_last_row():
                    ax.set_xticklabels([])
                if not ax.is_first_col():
                    ax.set_yticklabels([])


            if ax.is_last_row() and label_names is not None:
                ax.set_xlabel(label_names[i])
                
            if ax.is_first_col() and label_names is not None:
                ax.set_ylabel(label_names[j])

    fig.tight_layout()
    
    return fig


def plot_filled_contours(J, K, Z, N=1000, colorbar_label=None, 
                         converged=None, converged_kwds=None, 
                         marker_function=None, marker_kwds=None, 
                         truth=None,
                         ax=None, **kwargs):
    power = None
        
    #if np.all(Z > 0):
    #    power = np.min(np.log10(Z).astype(int))
    #    Z = Z.copy() / (10**power)
    
    if ax is None:
        w = 0.2 + 4 + 0.1
        h = 0.5 + 4 + 0.1
        if colorbar_label is not None:
            w += 1
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig = ax.figure

    #levels = np.linspace(np.floor(np.min(Z)), np.ceil(np.max(Z)), N).astype(int)
    #norm = BoundaryNorm(levels, 256)

    cf = ax.contourf(J, K, Z, N, vmin=np.nanmin(Z), vmax=np.nanmax(Z), **kwargs)

    ax.set_xlabel(r"$\textrm{Number of latent factors } J$")
    ax.set_ylabel(r"$\textrm{Number of clusters } K$")

    if converged is not None:
        kwds = dict(marker="x", c="#000000", s=10, linewidth=1, alpha=0.3,
                    rasterized=True)
        if converged_kwds is not None:
            kwds.update(converged_kwds)

        if not np.all(converged):
            ax.scatter(J[~converged], K[~converged], **kwds)

    if marker_function is not None:
        idx = marker_function(Z)
        j_m, k_m = (J[0][idx % Z.shape[1]], K.T[0][int(idx / Z.shape[1])])
        kwds = dict(facecolor="#ffffff", edgecolor="#000000", linewidth=1.5,
                    s=50, zorder=15)
        if marker_kwds is not None:
            kwds.update(marker_kwds)

        ax.scatter(j_m, k_m, **kwds)

        if truth is not None:
            J_t, K_t = truth

            ax.scatter([J_t], [K_t], facecolor="#000000", edgecolor="#000000",
                       s=10, zorder=14)

            ax.plot([J_t, j_m], [K_t, k_m], "-", c="#000000", lw=1.5, zorder=14)

    if colorbar_label is not None:

        cbar = plt.colorbar(cf)

        if power is not None:
            cbar.set_label(colorbar_label + " $/\,\,10^{0}$".format(power))
        else:
            cbar.set_label(colorbar_label)
            
        cbar.ax.yaxis.set_major_locator(MaxNLocator(5))
        #cbarlabels = np.linspace(np.min(Z), np.max(Z), num=5, endpoint=True)
        #cbar.set_ticklabels(cbarlabels)


        # TODO; only show integers but WHAT THE ACTUAL FUCK MATPLOTLIB
        #cbar_labels = [_.get_text() for _ in cbar.ax.get_yticklabels()]
        #cbar_labels = [f"${float(ea.strip('$')):.0f}$" for ea in cbar_labels]
        #cbar.set_ticklabels(cbar_labels)
        #kcbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "${:+5.0f}$".format(x)))


    edge_percent = 0.025
    x_range = np.ptp(J)
    y_range = np.ptp(K)
    ax.set_xlim(J.min() - x_range * edge_percent,
                J.max() + x_range * edge_percent)

    ax.set_ylim(K.min() - y_range * edge_percent,
                K.max() + y_range * edge_percent)
    
    if np.unique(J.flatten()).size < 9:
        ax.set_xticks(np.sort(np.unique(J.flatten())).astype(int))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(9))

    if np.unique(K.flatten()).size < 9:
        ax.set_yticks(np.sort(np.unique(K.flatten())).astype(int))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(9))


    ax.set_xticks(J[0].astype(int))
    ax.yaxis.set_tick_params(width=0)
    ax.xaxis.set_tick_params(width=0)

    fig.tight_layout()
    for c in cf.collections:
        c.set_edgecolor("face")

    return fig


def visualize_factor_loads(L, label_names=None, colors=None, line_kwds=None,
                           absolute_only=False,
                          **kwargs):

    L = np.atleast_2d(L)

    if absolute_only:
        L = np.abs(L.copy())

    D, J = L.shape

    #fig = plt.figure()
    #gs = matplotlib.gridspec.GridSpec(J, 2, width_ratios=[2, 1], height_ratios=np.hstack([np.ones(J)/J, 1]))
    #axes = [fig.add_subplot(gs[j]) for j in range(J + 1)]

    if label_names is not None:
        latex_label_names = [r"$\textrm{{{0}}}$".format(ea) for ea in label_names]


    #fig, axes = plt.subplots(1 + J)
    fig = plt.figure()

    K = 4
    shape = (J, K)
    axes = [plt.subplot2grid(shape, (j, 0), colspan=K-1) for j in range(J)]
    axes += [plt.subplot2grid(shape, (0, K - 1), rowspan=J, colspan=1)]

    if colors is None:
        cmap = discrete_cmap(J, base_cmap="Spectral_r")
        colors = [cmap(j) for j in range(J)]


    line_kwds_ = dict(lw=5)
    line_kwds_.update(line_kwds or dict())

    x = np.arange(D)
    for j, (ax, color) in enumerate(zip(axes, colors)):

        ax.plot(x, L.T[j], "-", c=color, **line_kwds_)
        ax.axhline(0, linewidth=1, c="#666666", linestyle=":", zorder=-1)
        ax.set_xlim(0, D)

        if absolute_only:
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 1])
            ax.set_ylabel(r"$|\mathbf{{L}}_{{{0}}}|$".format(j))

        else:
            ax.set_ylim(-1.1, 1.1)
            ax.set_yticks([-1, 0, 1])
            ax.set_ylabel(r"$\mathbf{{L}}_{{{0}}}$".format(j))

        if ax.is_last_row():
            if label_names is not None:
                ax.set_xticks(x.astype(int))
                ax.set_xticklabels(latex_label_names)

        else:
            ax.set_xticks([])

        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    # On last figure, show visualisation.

    F = np.abs(L)/np.atleast_2d(np.sum(np.abs(L), axis=1)).T
    indices = np.argsort(1.0/F, axis=1)

    ax = axes[-1]
    for d, (f, idx) in enumerate(zip(F, indices)):

        left = 0
        for i in idx:
            ax.barh(D - d, f[i], left=left, facecolor=colors[i])
            left += f[i]

        ax.barh(D - d, 1, left=0, edgecolor='#000000', zorder=-1, linewidth=2)


    ax.set_yticks(np.arange(1, 1 + D))
    if label_names is not None:
        ax.set_yticklabels(latex_label_names[::-1])

    ax.set_ylim(0.5, D + 0.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_xticks([])
    ax.yaxis.set_tick_params(width=0)

    ax.set_frame_on(False)

    ax.set_xlabel(r"${|\mathbf{L}_\textrm{d}|} / {\sum_{j}|\mathbf{L}_\textrm{j,d}|}$")

    fig.tight_layout()

    return fig




def plot_factor_loads(factor_loads, scales=1, separate_axes=False,
                      target_loads=None, show_target_loads=True, 
                      load_labels=None, n_rotation_inits=25,
                      flip_loads=None, xlabel=None, xticklabels=None,
                      legend_kwds=None, colors=None, figsize=None, **kwargs):



    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    D, J = factor_loads.shape
    if flip_loads is None:
        flip_loads = np.ones(J)
    else:
        flip_loads = np.array(flip_loads)

    if load_labels is None:
        load_labels = ["_"] * J

    xi = 1 + np.arange(D)

    if target_loads is not None:

        R, p_opt, cov, *_ = find_rotation_matrix(target_loads, factor_loads, 
                                                 n_inits=n_rotation_inits,
                                                 full_output=True)
        if cov is not None:
            N_draws = 100
            rotated_factors = np.zeros((N_draws, *factor_loads.shape))

            draws = np.random.multivariate_normal(p_opt, cov, size=N_draws)

            for i, draw in enumerate(draws):
                rotated_factors[i] = factor_loads @ givens_rotation_matrix(*draw)

            rotated_percentiles = np.percentile(rotated_factors, 
                                                [16, 50, 84], axis=0)
        else:
            rotated_percentiles = np.nan * np.ones((3, D, J))
            rotated_percentiles[1] = factor_loads @ R

    else:
        rotated_percentiles = np.nan * np.ones((3, D, J))
        rotated_percentiles[1] = factor_loads        

    for i in range(rotated_percentiles.shape[0]):
        rotated_percentiles[i] *= flip_loads
        rotated_percentiles[i] = (rotated_percentiles[i].T * scales).T

    if separate_axes:
        fig, ax = plt.subplots(J, 1, figsize=figsize)
        ax = np.atleast_2d(ax).flatten()

    else:
        fig, ax = plt.subplots()
        ax = [ax]

    for j in range(J):

        axes = ax[j] if separate_axes else ax[0]
        color = colors[j % len(colors)]

        default_kwds = dict(lw=2)
        default_kwds.update(kwargs)
        axes.plot(xi, rotated_percentiles[1, :, j], "-", 
                  c=color, label=load_labels[j], **default_kwds)
        axes.fill_between(xi, 
                          rotated_percentiles[0, :, j], 
                          rotated_percentiles[2, :, j],
                          facecolor=color, alpha=0.3)

        if show_target_loads and target_loads is not None:
            axes.plot(xi, target_loads.T[j], ":", c=color)


    for i, axes in enumerate(ax):

        axes.set_xticks(xi)
        ylim = max(np.ceil(10 * np.abs(axes.get_ylim()).max()) / 10, 1)
        axes.plot([0, D + 1], [0, 0], ":", c="#000000", zorder=-1, lw=0.5)
        axes.set_xlim(0.5, D + 0.5)

        axes.set_ylim(-ylim, +ylim)
        axes.set_yticks([-ylim, 0, ylim])

        if axes.is_last_row():
            if xticklabels is not None:
                axes.set_xticklabels(xticklabels)

            if xlabel is not None:
                axes.set_xlabel(xlabel)
            else:
                axes.set_xlabel(r"$\textrm{dimension}$")

        else:
            axes.set_xticklabels([""] * D)

        if axes.is_first_col():
            if separate_axes:
                axes.set_ylabel(r"$\mathbf{{L}}_{{{0}}}$".format(i))
            else:
                axes.set_ylabel(r"$\mathbf{{L}}$")

        if load_labels is not None:
            kwds = dict(frameon=False, fontsize=12.0)
            kwds.update(legend_kwds or dict())
            axes.legend(**kwds)

        if axes.is_first_col():
            axes.set_yticklabels([rf"$-{ylim}$", r"$0$", rf"$+{ylim}$"])

    fig.tight_layout()
    return fig
