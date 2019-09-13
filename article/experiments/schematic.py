
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid.inset_locator import inset_axes


import sys

sys.path.insert(0, "../../")

from mcfa import utils

np.random.seed(0)



def latexify(s):

    return s

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

#ax.set_xticks([])
#ax.set_yticks([])
fig.tight_layout()
#ax.set_frame_on(False)

# do left hand side table showing

J = 3
K = 5

D = 10
N = 25

X, true_theta = utils.generate_data(n_samples=N, n_features=D, n_components=K, n_latent_factors=J,
                                    omega_scale=1e-2)

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

            y_ = y + i * vertical_grid_spacing
            ax.plot([x, x + w], [y_, y_], **kwds)


    if horizontal_grid_spacing is not None:

        num_lines = int(1 + w / horizontal_grid_spacing)
        for i in range(num_lines):
            is_edge = i in (0, num_lines - 1)
            if is_edge and edge_kwds is None: continue

            kwds = edge_kwds if is_edge else grid_kwds

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

            y_ = y + i * vertical_grid_spacing
            ax.plot([x, x + w], [y_, y_], **kwds)


        num_lines = int(1 + w / horizontal_grid_spacing)
        for i in range(num_lines):
            is_edge = i in (0, num_lines - 1)
            if (is_edge and edge_kwds is None) \
            or (not is_edge and grid_kwds is None):
                continue

            kwds = edge_kwds if is_edge else grid_kwds

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

    padding = 0.10
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

    axes = []
    for i, (x_, y_) in enumerate(positions):
        ia = inset_axes(ax, width="100%", height="100%",
                        bbox_to_anchor=(x_, y_, w, h),
                        bbox_transform=ax.transData, borderpad=0, loc=3)
        ia.set_xticks([])
        ia.set_yticks([])

        ia.axhline(0, c="#666666", linestyle=":", lw=0.5)
        ia.plot(A.T[i], c="k", lw=2)
        ax.set_ylabel(r"$L_{0}$".format(i + 1), labelpad=-0.5)

        ax.set_ylim(-max_abs_ylim, +max_abs_ylim)


        axes.append(ia)

    return axes





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
draw_matrix(ax, xs, ys, w, h, **kw)

data_matrix_kwds = dict(edge_kwds=dict(lw=0.75, c="#000000", zorder=10),
                        grid_kwds=None, vmin=-10, vmax=10)
#draw_matrix_values(ax, X, xs, ys, w, h, **data_matrix_kwds)
                   


text_kwds = dict(horizontalalignment="center", verticalalignment="center")
ax.text(xs + w/2, ys + h + 1, "data", **text_kwds)


# draw lines connecting
dashes = np.hstack([np.tile([2, 8], 7), [0, 1000]])
dash_kwds = dict(dashes=dashes, c="#666666", lw=0.5)
ax.plot([xs + w, D + space], [ys + h, N], **dash_kwds)
ax.plot([xs + w, D + space], [ys, 0], **dash_kwds)
    
# draw latent score matrix
scores = true_theta["scores"]
tau = true_theta["R"]

x, y = (D + space, 0)
#draw_matrix(ax, x, y, J, N, **matrix_kwds)

draw_matrix_values(ax, scores, x, y, J, N, 
                   edge_kwds=dict(lw=0.75, c="#000000", zorder=10),
                   grid_kwds=None, vmin=-3, vmax=3)
#scores, x, y, D, J)
ax.text(x + D/2, y + N + 5, "latent scores", **text_kwds)


ax.text(x + D + space/2., N/2, ".", **text_kwds)



#draw_matrix_values(ax, scores, x, 40, J, N, 
#                   edge_kwds=dict(lw=0.75, c="#000000", zorder=10),
#                   grid_kwds=None)


# draw latent factor matrix
x += (D - J) + space
y = N - J

#draw_matrix(ax, x, y, D, J, **matrix_kwds)
edge_kwds = dict(lw=0.75, c="#000000", zorder=10)
draw_matrix_values(ax, true_theta["A"].T, x, y, D, J, grid_kwds=None, edge_kwds=edge_kwds)
draw_latent_factors(ax, true_theta["A"], x, -12, D, 3)
ax.text(x + D/2, N + 5, "latent factors", **text_kwds)


ax.text(x + D + space/2., N/2, "+", **text_kwds)


x, y = (x + D + space, N - 1)
# draw noise matrix
#draw_matrix(ax, x, y, D, 1, **matrix_kwds)
noise = true_theta["psi"].reshape((1, D))
draw_matrix_values(ax, noise, x, y, D, 1, edge_kwds=edge_kwds)

ax.text(x + D/2, N + 5, "noise", **text_kwds)


draw_latent_scores(ax, scores, tau,  D + space, -12, 5, space=1)


ax.text(x + D + space/2., N/2, "=", **text_kwds)

x += D + space

# Draw a data matrix that is coloured.
#draw_matrix(ax, x, ys, w, h, **kw)
draw_matrix_values(ax, X, x, ys, w, h, cmap="RdYlGn", **data_matrix_kwds)

ax.text(x + w/2, ys + h + 1, "data", **text_kwds)



# draw connecting lines.





wh = 55 + 50

xs, ys = (-10, -20)
ax.set_xlim(xs, xs + wh)
ax.set_ylim(ys,ys +wh)

plt.show()


