
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from numpy.ma import masked_array



import sys

sys.path.insert(0, "../../")

from mcfa import utils

matplotlib.style.use({
    'font.size': 12.0,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts,amssymb,bm}',
    'text.latex.preview': True,
    'axes.unicode_minus': False,
})
random_seed = 8




def latexify(s):

    return s

fig, ax = plt.subplots(1, 1, figsize=(7.95, 3.9))

#ax.set_xticks([])
#ax.set_yticks([])
fig.tight_layout()
#ax.set_frame_on(False)

# do left hand side table showing

J = 3
K = 4

D = 11
N = 25

X, true_theta = utils.generate_data(n_samples=N, n_features=D, n_components=K, n_latent_factors=J,
                                    omega_scale=0.1, noise_scale=0.1, random_seed=random_seed)

scores = true_theta["scores"]
tau = true_theta["R"]

def draw_matrix(ax, x, y, w, h, 
                horizontal_grid_spacing=1, vertical_grid_spacing=1,
                facecolor=None,
                grid_kwds=None, edge_kwds=None):

    patch = matplotlib.patches.Rectangle((x, y), w, h, facecolor=facecolor, edgecolor=facecolor)

    ax.add_patch(patch)

    if vertical_grid_spacing is not None:

        num_lines = int(1 + h / vertical_grid_spacing)
        for i in range(num_lines):
            is_edge = i in (0, num_lines - 1)
            if is_edge and edge_kwds is None: continue

            kwds = edge_kwds if is_edge else grid_kwds
            kwds.update(linestyle="-", marker=None, ms=0)

            y_ = y + i * vertical_grid_spacing
            ax.plot([x, x + w], [y_, y_], **kwds)


    if horizontal_grid_spacing is not None:

        num_lines = int(1 + w / horizontal_grid_spacing)
        for i in range(num_lines):
            is_edge = i in (0, num_lines - 1)
            if is_edge and edge_kwds is None: continue

            kwds = edge_kwds if is_edge else grid_kwds
            kwds.update(linestyle="-", marker=None, ms=0)

            x_ = x + i * horizontal_grid_spacing
            ax.plot([x_, x_], [y, y + h], **kwds)


def draw_matrix_values(ax, X, x, y, w, h,
                       grid_kwds=None, edge_kwds=None, cmap="Greys", **kwargs):

    # OMG MATPLOTLIB REALLY?!
    bug_offset = -0.1
    ax.imshow(X, interpolation="none", origin="lower",
              extent=(x, x+w, y, y + h + bug_offset), cmap=cmap, **kwargs)

    if grid_kwds is not None or edge_kwds is not None:

        vertical_grid_spacing = h / X.shape[0]
        horizontal_grid_spacing = w / X.shape[1]



        num_lines = int(1 + h / vertical_grid_spacing)
        for i in range(num_lines):
            is_edge = i in (0, num_lines - 1)
            if (is_edge and edge_kwds is None) \
            or (not is_edge and grid_kwds is None):
                continue

            kwds = edge_kwds if is_edge else grid_kwds
            kwds.update(linestyle="-", marker=None, ms=0)

            y_ = y + i * vertical_grid_spacing
            ax.plot([x, x + w], [y_, y_], **kwds)


        num_lines = int(1 + w / horizontal_grid_spacing)
        for i in range(num_lines):
            is_edge = i in (0, num_lines - 1)
            if (is_edge and edge_kwds is None) \
            or (not is_edge and grid_kwds is None):
                continue

            kwds = edge_kwds if is_edge else grid_kwds
            kwds.update(linestyle="-", marker=None, ms=0)

            x_ = x + i * horizontal_grid_spacing
            ax.plot([x_, x_], [y, y + h], **kwds)


def draw_matrix_values_multiple_colors(ax, X, tau, x, y, w, h, colormaps=None,
                                       grid_kwds=None, edge_kwds=None, **kwargs):


    bug_offset = -0.1
    unique_taus = np.sort(np.unique(tau))

    for i, unique_tau in enumerate(unique_taus):
        #try:
        cmap = colormaps[i]
        #except:
        #    cmap = "Greys"

        #cmap = colormaps[0]

        mask = np.repeat(tau == unique_tau, X.shape[1]).reshape(X.shape)
        
        v = masked_array(X, ~mask)

        vmin, vmax = (np.min(X[mask]), np.max(X[mask]))
        #vmin, vmax = (np.min(X), np.max(X))

        kwd = kwargs.copy()
        kwd.setdefault("vmin", vmin)
        kwd.setdefault("vmax", vmax)


        ax.imshow(v, interpolation="none", cmap=cmap, origin="lower",
                  extent=(x, x+w, y, y + h + bug_offset), **kwd)


    if grid_kwds is not None or edge_kwds is not None:

        vertical_grid_spacing = h / X.shape[0]
        horizontal_grid_spacing = w / X.shape[1]

        num_lines = int(1 + h / vertical_grid_spacing)
        for i in range(num_lines):
            is_edge = i in (0, num_lines - 1)
            if (is_edge and edge_kwds is None) \
            or (not is_edge and grid_kwds is None):
                continue

            kwds = edge_kwds if is_edge else grid_kwds
            kwds.update(linestyle="-", marker=None, ms=0)

            y_ = y + i * vertical_grid_spacing
            ax.plot([x, x + w], [y_, y_], **kwds)

        num_lines = int(1 + w / horizontal_grid_spacing)
        for i in range(num_lines):
            is_edge = i in (0, num_lines - 1)
            if (is_edge and edge_kwds is None) \
            or (not is_edge and grid_kwds is None):
                continue

            kwds = edge_kwds if is_edge else grid_kwds
            kwds.update(linestyle="-", marker=None, ms=0)

            x_ = x + i * horizontal_grid_spacing
            ax.plot([x_, x_], [y, y + h], **kwds)



# draw little points in latent space.
def draw_latent_scores(ax, scores, tau, x, y, axis_size, max_abs_lim=2.5, space=1, 
                       scatter_kwds=None, **kwargs):

    positions = [
        (x, y),
        (x + axis_size + space, y),
        (x, y + axis_size + space)
    ]

    axes = []
    for i, (x_, y_) in enumerate(positions):
        ia = inset_axes(ax, width="100%", height="100%",
                        bbox_to_anchor=(x_, y_, axis_size, axis_size),
                        bbox_transform=ax.transData, borderpad=0, loc=3)
        ia.set_xticks([])
        ia.set_yticks([])

        axes.append(ia)

    scatter_kwds = scatter_kwds or dict()
    scatter_kwds.setdefault("c", tau)
    scatter_kwds.setdefault("s", 5)

    axes[0].scatter(scores.T[0], scores.T[1], **scatter_kwds)
    axes[1].scatter(scores.T[2], scores.T[1], **scatter_kwds)
    axes[2].scatter(scores.T[0], scores.T[2], **scatter_kwds)

    label_kwds = dict(fontsize=10)
    axes[0].set_xlabel(r"$\mathbf{S}_1$", **label_kwds)
    axes[0].set_ylabel(r"$\mathbf{S}_2$", **label_kwds)

    axes[1].set_xlabel(r"$\mathbf{S}_3$", **label_kwds)
    axes[2].set_ylabel(r"$\mathbf{S}_3$", **label_kwds)

    padding = 0.20
    for ax in axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xptp = np.ptp(xlim) * (1 + padding)
        yptp = np.ptp(ylim) * (1 + padding)

        ax.set_xlim(np.mean(xlim) - 0.5 * xptp, np.mean(xlim) + 0.5 * xptp)
        ax.set_ylim(np.mean(ylim) - 0.5 * yptp, np.mean(ylim) + 0.5 * yptp)


    return axes


def draw_latent_factors(ax, A, x, y, w, h, space=1, max_abs_ylim=1, **kwargs):

    positions = [
        (x, y),
        (x, y + h + space),
        (x, y + 2 * h + 2 * space)
    ]

    #max_abs_ylim = np.max(np.abs(A))


    axes = []
    for i, (x_, y_) in enumerate(positions):

        c = matplotlib.cm.Greys(0.5 + 0.5 * i/2.)

        print(i, "latent", c)

        ia = inset_axes(ax, width="100%", height="100%",
                        bbox_to_anchor=(x_, y_, w, h),
                        bbox_transform=ax.transData, borderpad=0, loc=3)
        ia.set_xticks([])
        ia.set_yticks([])

        ia.axhline(0, c="#666666", linestyle=":", lw=0.5, ms=0)
        ia.plot(A.T[i], c=c, lw=2, ms=0)
        ia.set_ylabel(r"$\mathbf{{L}}_{0}$".format(i + 1), fontsize=10)

        ia.set_ylim(-max_abs_ylim, +max_abs_ylim)



        axes.append(ia)

    return axes




equation_y_text = N + 4
matrix_kwds = dict(facecolor="#eeeeee", 
                   horizontal_grid_spacing=1, vertical_grid_spacing=1,
                   edge_kwds=dict(lw=0.75, c="#000000", zorder=100),
                   grid_kwds=dict(lw=0.5, c="#cccccc", zorder=10))

space = 8

# draw data matrix
kw = matrix_kwds.copy()
kw.update(horizontal_grid_spacing=1.5, vertical_grid_spacing=1.5)
xs, ys = (-5, -0.25 * N)
w, h = (1.5 * D, 1.5 * N)

#xs, ys = (-5, 0)
#w, h = (D, N)
#draw_matrix(ax, xs, ys, w, h, **kw)

data_matrix_kwds = dict(edge_kwds=dict(lw=0.75, c="#000000", zorder=10),
                        grid_kwds=None)
draw_matrix_values(ax, X, -2.5, 0, D, N, **data_matrix_kwds)
   

ax.text(-2.5, -2, r"$\textrm{abundances}$", fontsize=10)
ax.text(-2.5 - 2, 0, r"$\textrm{stars}$", fontsize=10, rotation=90)

text_kwds = dict(horizontalalignment="center", verticalalignment="center")
data_text_y = ys + h + 3
ax.text(xs + w/2, data_text_y, r"$\textrm{data}$", **text_kwds)
ax.text(xs + w/2, equation_y_text, r"$\mathbf{X}^\top$", **text_kwds)


# draw lines connecting
dashes = np.hstack([np.tile([2, 8], 7), [0, 1000]])
dash_kwds = dict(dashes=dashes, c="#666666", lw=0.5, ms=0)
#ax.plot([xs + w, D + space], [ys + h, N], **dash_kwds)
#ax.plot([xs + w, D + space], [ys, 0], **dash_kwds)

ax.plot([-2.5 + D, D + space], [N, N], **dash_kwds)
ax.plot([-2.5 + D, D + space], [0, 0], **dash_kwds)

    
ax.text(-2.5 + D + space/2. + 5, N/2, r"$=$", **text_kwds)


#draw_matrix(ax, x, y, D, 1, **matrix_kwds)
mean = np.mean(X, axis=0).reshape((1, D))
edge_kwds = dict(lw=0.75, c="#000000", zorder=10)
draw_matrix_values(ax, mean, D + space, N-1, D, 1, edge_kwds=edge_kwds, vmin=-0.5, vmax=2)

ax.text(D + space + D/2, data_text_y, r"$\textrm{mean}$", **text_kwds)
ax.text(D + space + D/2, equation_y_text, r"$\boldsymbol{\mu}^\top$", **text_kwds)

#boldsymbol{\mathbf{#1}}
ax.text(D + space + D + space/2., N/2, r"$+$", **text_kwds)



# draw latent score matrix
scores = true_theta["scores"]
tau = true_theta["R"]

x, y = (D + space + D + space, 0)
#draw_matrix(ax, x, y, J, N, **matrix_kwds)

#draw_matrix_values(ax, scores, x, y, J, N, 
#                   edge_kwds=dict(lw=0.75, c="#000000", zorder=10),
#                   grid_kwds=None, vmin=-3, vmax=3)

# colors for unique taus.


scores_cmap = matplotlib.cm.Pastel1
custom_cmaps = []
custom_cmaps2 = []

foo_cmap = []

for i in range(len(np.unique(tau))):

    #color = scores_cmap(int(np.linspace(0, 255, K)[i]))
    color = scores_cmap(i)

    foo_cmap.append(color)

    print(i, color)
    Z = 256
    vals = np.ones((Z, 4))

    vals[:, 0] = color[0]
    vals[:, 1] = color[1]
    vals[:, 2] = color[2]
    vals[:, 3] = np.linspace(0.0, 1, Z)

    vals2 = np.copy(vals)
    vals2[:, 3] = np.linspace(0.5, 1, Z)

    custom_cmaps.append(matplotlib.colors.ListedColormap(vals))
    custom_cmaps2.append(matplotlib.colors.ListedColormap(vals2))


foo_cmap = matplotlib.colors.ListedColormap(foo_cmap)


draw_matrix_values_multiple_colors(ax, scores, tau, x, y, J, N,
                                   edge_kwds=dict(lw=0.75, c="#000000", zorder=10),
                                   grid_kwds=None, colormaps=custom_cmaps2)
ax.text(x + 0.5 + 0.0, y - 1, r"$1$", fontsize=6, **text_kwds)
ax.text(x + 0.5 + 1.1, y - 1, r"$2$", fontsize=6, **text_kwds)
ax.text(x + 0.5 + 2.2, y - 1, r"$3$", fontsize=6, **text_kwds)

#("viridis", "PRGn", "Greys", "RdBu", "twilight"))

#scores, x, y, D, J)
ax.text(x + D/2, data_text_y, "$\\textrm{latent}$\n$\\textrm{scores}$", **text_kwds)
ax.text(x + D/2, equation_y_text, r"$\mathbf{S}^\top$", **text_kwds)

#ax.text(x + D/2, data_text_y - 1.2, r"$\textrm{scores}$", **text_kwds)


ax.text(x + D + space/2., N/2, r"$\cdot$", **text_kwds)

ax.plot([x + J/2, x + J/2], [-2, -4], c="#666666", lw=0.5, linestyle=":")


#draw_matrix_values(ax, scores, x, 40, J, N, 
#                   edge_kwds=dict(lw=0.75, c="#000000", zorder=10),
#                   grid_kwds=None)


# draw latent factor matrix
x += (D - J) + space
y = N - J

#draw_matrix(ax, x, y, D, J, **matrix_kwds)
edge_kwds = dict(lw=0.75, c="#000000", zorder=10)
draw_matrix_values(ax, true_theta["A"].T, x, y, D, J, grid_kwds=None, edge_kwds=edge_kwds)
yspacing = 1.1
ax.text(x - 1, y + 2 * yspacing, r"$1$", fontsize=6, horizontalalignment="center", verticalalignment="center")
ax.text(x - 1, y + yspacing, r"$2$", fontsize=6, horizontalalignment="center", verticalalignment="center")
ax.text(x - 1, y, r"$3$", fontsize=6, horizontalalignment="center", verticalalignment="center")





latent_factors_y = -16
draw_latent_factors(ax, true_theta["A"], x, latent_factors_y, D, 3)
ax.text(x + D/2, data_text_y, "$\\textrm{latent}$\n$\\textrm{factors}$", **text_kwds)
ax.text(x + D/2, equation_y_text, r"$\mathbf{L^\top}$", **text_kwds)

ax.text(x + D + space/2., N/2, r"$+$", **text_kwds)

ax.plot([x + D/2, x + D/2], [y - 1, -4], c="#666666", lw=0.5, linestyle=":")



x, y = (x + D + space, N - 1)
# draw noise matrix
#draw_matrix(ax, x, y, D, 1, **matrix_kwds)
noise = true_theta["psi"].reshape((1, D))
draw_matrix_values(ax, noise, x, y, D, 1, edge_kwds=edge_kwds)


ax.text(x + D/2, equation_y_text, r"$\mathbf{\Psi}^\top$", **text_kwds)

ax.text(x + D/2, data_text_y, r"$\textrm{noise}$", **text_kwds)


draw_latent_scores(ax, scores, tau, 2 * (D + space), latent_factors_y, 5, space=1,
    scatter_kwds=dict(cmap=foo_cmap))



#ax.text(x + D + space/2., N/2, r"$=$", **text_kwds)





dash_kwds = dict(dashes=dashes, c="#666666", lw=0.5, ms=0)
ax.plot([x + D, x + D + space][::-1], [ys + h, N], **dash_kwds)
ax.plot([x + D, x + D + space][::-1], [ys, 0], **dash_kwds)


x += D + space

# Draw a data matrix that is coloured.
#draw_matrix(ax, x, ys, w, h, **kw)
#draw_matrix_values(ax, X, x, ys, w, h, cmap="RdYlGn", **data_matrix_kwds)



draw_matrix_values_multiple_colors(ax, X, tau, x, ys, w, h,
                                   edge_kwds=dict(lw=0.75, c="#000000", zorder=10),
                                   grid_kwds=None, colormaps=custom_cmaps,
                                   vmin=np.min(X), vmax=np.max(X))
#("viridis", "PRGn", "Greys", "RdBu", "twilight"))




#ax.text(x + w/2, data_text_y, r"$\textrm{data}$", **text_kwds)



# draw connecting lines.

x_lims = (-2.5, 110)
x_pad = 2
x_frac = x_pad/(2 * x_pad + np.ptp(x_lims))

y_lims = (-16, 37)
y_pad = x_frac * np.ptp(y_lims)

ax.set_xlim(x_lims[0] - x_pad, x_lims[1] + x_pad)
ax.set_ylim(y_lims[0] - y_pad, y_lims[1] + y_pad)

ax.set_aspect(1.0)
ax.set_xticks([])
ax.set_yticks([])

ax.set_frame_on(False)


#ptp = np.ptp(x_lims) + 2 * x_pad
#ax.set_ylim(-20, -20 + ptp)


"""
wh = 55 + 60
xs, ys = (-10, -20)
ax.set_xlim(xs, xs + wh)
ax.set_ylim(ys,ys +wh)
""" 
fig.tight_layout()
plt.show()


fig.savefig(__file__[:-3] + ".png", dpi=300)
fig.savefig(__file__[:-3] + ".pdf", dpi=300)

