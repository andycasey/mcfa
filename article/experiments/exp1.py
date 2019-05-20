
import sys
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from time import time

sys.path.insert(0, "../")

from mcfa import (mcfa, grid_search, mpl_utils, utils)

matplotlib.style.use(mpl_utils.mpl_style)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


n_features = D = 15
n_components = K = 10
n_latent_factors = J = 5
n_samples = N = 10_000

omega_scale = 1
noise_scale = 1
random_seed = 100

data_kwds = dict(n_features=n_features,
                 n_components=n_components,
                 n_latent_factors=n_latent_factors,
                 n_samples=n_samples,
                 omega_scale=omega_scale,
                 noise_scale=noise_scale,
                 random_seed=random_seed)


def savefig(fig, suffix):
    prefix = os.path.basename(__file__)[:-3]
    here = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(here, f"{prefix}-{suffix}")
    fig.savefig(f"{filename}.pdf", dpi=300)
    fig.savefig(f"{filename}.png", dpi=150)
    print(f"Created figures {filename}.png and {filename}.pdf")


mcfa_kwds = dict(tol=1e-5, 
                 max_iter=1_000,
                 init_factors="random",
                 init_components="random",
                 random_seed=random_seed)




Y, truth = utils.generate_data(**data_kwds)
truth_packed = (truth["pi"], truth["A"], truth["xi"], truth["omega"], truth["psi"])


# Fit with true number of latent factors and components.
model = mcfa.MCFA(n_components=data_kwds["n_components"],
                  n_latent_factors=data_kwds["n_latent_factors"],
                  **mcfa_kwds)
tick = time()
model.fit(Y)
tock = time()

model.message_length(Y)

print(f"Model took {tock - tick:.1f} seconds")


# Plot the true latent factors w.r.t. the estimated ones, after rotation.
A_true = truth["A"]
A_est = model.theta_[model.parameter_names.index("A")]

#R, *_ = utils.find_rotation_matrix(A_true, A_est, n_inits=100)

# Get exact transformation.
R = utils.exact_rotation_matrix(A_true, A_est)

# Now make it a valid rotation matrix.
L = linalg.cholesky(R.T @ R)
R = R @ linalg.solve(L, np.eye(n_latent_factors))


A_est_rot = A_est @ R


D, J = A_true.shape
xi = 1 + np.arange(D)

fig_factor_loads, ax = plt.subplots()

for j in range(J):
    ax.plot(xi, A_est.T[j], ":", lw=1, c=colors[j])
    ax.plot(xi, A_est_rot.T[j], "-", lw=1, c=colors[j])
    ax.plot(xi, A_true.T[j], "-", lw=2, c=colors[j])

ax.set_xticks(xi)
ax.set_xlabel(r"$\textrm{dimension}$")
ax.set_ylabel(r"$\mathbf{L}$")

ylim = np.ceil(10 * np.abs(ax.get_ylim()).max()) / 10
ax.plot([0, D + 1], [0, 0], ":", c="#000000", zorder=-1, lw=0.5)
ax.set_xlim(0.5, D + 0.5)

ax.set_ylim(-ylim, +ylim)
ax.set_yticks([-ylim, 0, ylim])

fig_factor_loads.tight_layout()
savefig(fig_factor_loads, "factor_loads")



# Plot the latent space.
cmap = mpl_utils.discrete_cmap(n_components, "Spectral")

label_names = [f"$\\mathbf{{S}}_{{{i}}}$" for i in range(n_latent_factors)]
fig_latent = mpl_utils.plot_latent_space(model, Y, cmap=cmap,
                                         label_names=label_names)
for ax in fig_latent.axes:
    if ax.get_visible():
        if ax.is_last_row():
            ax.xaxis.set_major_locator(MaxNLocator(3))
        if ax.is_first_col():
            ax.yaxis.set_major_locator(MaxNLocator(3))

        xlim = np.max(np.abs(ax.get_xlim()))
        ylim = np.max(np.abs(ax.get_ylim()))
        ax.set_xlim(-xlim, +xlim)
        ax.set_ylim(-ylim, +ylim)
fig_latent.tight_layout()
fig_latent.subplots_adjust(hspace=0, wspace=0)
savefig(fig_latent, "latent")



ll, tau = model.expectation(Y, *model.theta_)

model.rotate(R)

ll2, tau = model.expectation(Y, *model.theta_)

print(f"Difference in log-likeihood after rotation: {ll - ll2}")


# Take model with true number of components and latent factors.

scatter_kwds = dict(s=25, rasterized=True, c="tab:blue")


# Compare factor loads to true values.
from matplotlib import gridspec

fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])

ax_residual = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])

A_est = model.theta_[model.parameter_names.index("A")]

x, y = (A_true.flatten(), A_est.flatten())
ax.scatter(x, y, **scatter_kwds)
ax_residual.scatter(x, y - x, **scatter_kwds)

lims =lims = np.max(np.abs(np.hstack([ax.get_xlim(), ax.get_ylim()])))
kwds = dict(c="#666666", linestyle=":", linewidth=0.5, zorder=-1)
ax.plot([-lims, +lims], [-lims, +lims], "-", **kwds)
ax_residual.plot([-lims, +lims], [0, 0], "-", **kwds)


ax.set_xlim(-lims, +lims)
ax.set_ylim(-lims, +lims)
ax_residual.set_xlim(-lims, +lims)
ylim = np.max(np.abs(ax_residual.get_ylim()))
ax_residual.set_ylim(-ylim, +ylim)


ax_residual.yaxis.set_major_locator(MaxNLocator(3))
ax_residual.xaxis.set_major_locator(MaxNLocator(5))
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(5))

ax_residual.set_xticks([])

ax.set_xlabel(r"$\mathbf{L}_\textrm{true}$")
ax.set_ylabel(r"$\mathbf{L}_\textrm{est}$")
ax_residual.set_ylabel(r"$\Delta\mathbf{L}$")

fig.tight_layout()

savefig(fig, "compare-loads")


# Compare factor scores to true values.
x = truth["scores"].flatten()
y = model.factor_scores(Y)[1].flatten()
 

# Compare factor loads to true values.
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])

ax_residual = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])

ax.scatter(x, y, **scatter_kwds)
ax_residual.scatter(x, y - x, **scatter_kwds)

lims = np.max(np.abs(np.hstack([ax.get_xlim(), ax.get_ylim()])))
kwds = dict(c="#666666", linestyle=":", linewidth=0.5, zorder=-1)
ax.plot([-lims, +lims], [-lims, +lims], "-", **kwds)
ax_residual.plot([-lims, +lims], [0, 0], "-", **kwds)

ax.set_xlim(-lims, +lims)
ax.set_ylim(-lims, +lims)
ax_residual.set_xlim(-lims, +lims)
ylim = np.max(np.abs(ax_residual.get_ylim()))
ax_residual.set_ylim(-ylim, +ylim)

ax_residual.yaxis.set_major_locator(MaxNLocator(3))
ax_residual.xaxis.set_major_locator(MaxNLocator(5))
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(5))
ax_residual.set_xticks([])


ax.set_xlabel(r"$\mathbf{S}_\textrm{true}$")
ax.set_ylabel(r"$\mathbf{S}_\textrm{est}$")
ax_residual.set_ylabel(r"$\Delta\mathbf{S}$")

fig.tight_layout()

savefig(fig, "compare-scores")


# Compare specific scatter values to true values.
x = truth["psi"].flatten()
y = model.theta_[-1]


fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])

ax_residual = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])


ax.scatter(x, y, **scatter_kwds)
ax_residual.scatter(x, y - x, **scatter_kwds)

lims = np.max(np.abs(np.hstack([ax.get_xlim(), ax.get_ylim()])))
kwds = dict(c="#666666", linestyle=":", linewidth=0.5, zorder=-1)
ax.plot([-lims, +lims], [-lims, +lims], "-", **kwds)
ax_residual.plot([-lims, +lims], [0, 0], "-", **kwds)

ax.set_xlim(0, +lims)
ax.set_ylim(0, +lims)
ax_residual.set_xlim(0, lims)
ylim = np.max(np.abs(ax_residual.get_ylim()))
ax_residual.set_ylim(-ylim, +ylim)

ax_residual.yaxis.set_major_locator(MaxNLocator(3))
ax_residual.xaxis.set_major_locator(MaxNLocator(5))
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(5))
ax_residual.set_xticks([])


ax.set_xlabel(r"$\mathbf{\Psi}_\textrm{true}$")
ax.set_ylabel(r"$\mathbf{\Psi}_\textrm{est}$")
ax_residual.set_ylabel(r"$\Delta\mathbf{\Psi}$")

fig.tight_layout()

savefig(fig, "compare-specific-scatter")




scatter_kwds = dict(s=1, rasterized=True, c="#000000")

fig = plt.figure(figsize=(7.5, 3.09))


gs = gridspec.GridSpec(2, 3, height_ratios=[1, 4], width_ratios=[1, 1, 1])

A_est = model.theta_[model.parameter_names.index("A")]

xs = [
    A_true.flatten(),
    truth["scores"].flatten(),
    truth["psi"].flatten()
]

ys = [
    A_est.flatten(),
    model.factor_scores(Y)[1].flatten(),
    model.theta_[-1]
]

xlabels = [
    r"$\mathbf{L}_\textrm{true}$",
    r"$\mathbf{S}_\textrm{true}$",
    r"$\mathbf{\Psi}_\textrm{true}$"
]

ylabels = [
    r"$\mathbf{L}_\textrm{est}$",
    r"$\mathbf{S}_\textrm{est}$",
    r"$\mathbf{\Psi}_\textrm{est}$"
]

delta_labels = [
    r"$\Delta\mathbf{L}$",
    r"$\Delta\mathbf{S}$",
    r"$\Delta\mathbf{\Psi}$"
]

idx = 0
for i in range(3):
    ax_residual = fig.add_subplot(gs[idx])
    ax = fig.add_subplot(gs[idx +3])

    x, y = (xs[i], ys[i])

    ax.scatter(x, y, **scatter_kwds)
    ax_residual.scatter(x, y - x, **scatter_kwds)

    lims = np.max(np.abs(np.hstack([ax.get_xlim(), ax.get_ylim()])))
    if i == 2:
        lims = (0, +lims)
    else:
        lims = (-lims, +lims)

    kwds = dict(c="#666666", linestyle=":", linewidth=0.5, zorder=-1)
    ax.plot([lims[0], +lims[1]], [lims[0], +lims[1]], "-", **kwds)
    ax_residual.plot([lims[0], +lims[1]], [0, 0], "-", **kwds)

    ax.set_xlim(lims[0], +lims[1])
    ax.set_ylim(lims[0], +lims[1])
    ax_residual.set_xlim(lims[0], +lims[1])
    ylim = np.max(np.abs(ax_residual.get_ylim()))
    ax_residual.set_ylim(-ylim, +ylim)

    ax_residual.yaxis.set_major_locator(MaxNLocator(3))
    ax_residual.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax_residual.set_xticks([])


    ax.set_xlabel(xlabels[i])
    ax.set_ylabel(ylabels[i])
    ax_residual.set_ylabel(delta_labels[i])

    #ax.set_aspect(1.0)
    #ax_residual.set_aspect(1)
    idx += 1

fig.tight_layout()
savefig(fig, "compare-all")



# Plot the log-likelihood with increasing iterations.
fig_iterations, ax = plt.subplots()

ll = np.array(model.log_likelihoods_)
iterations = 1 + np.arange(len(ll))
ax.plot(iterations, ll, "-", lw=2, drawstyle="steps-mid")
ax.set_xlabel(r"$\textrm{iteration}$")
ax.set_ylabel(r"$\log{\mathcal{L}}$")
ax.set_xlim(0, iterations[-1] + 1)
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))
xt = ax.get_xticks()
ax.set_xticks(xt + np.hstack([1, np.zeros(xt.size - 1)]))
ax.set_xlim(0, iterations[-1] + 1)
fig_iterations.tight_layout()
savefig(fig_iterations, "ll-iterations")


# Plot the data, with samples
Y_drawn = model.sample(data_kwds["n_samples"])
fig_data = mpl_utils.corner_scatter(Y, 
                                    c="#000000", s=1, alpha=0.5, figsize=(8, 8))
mpl_utils.corner_scatter(Y_drawn,
                         c="tab:blue", s=1, alpha=0.25, zorder=10, fig=fig_data)

fig_data.tight_layout()
fig_data.subplots_adjust(hspace=0, wspace=0)
savefig(fig_data, "data")





# Compare specific scatter to true values.
# TODO:



fig_data2 = mpl_utils.corner_scatter(Y, 
                                    c=truth["R"], cmap="Spectral",
                                    s=1, alpha=0.5, figsize=(8, 8),
                                    label_names=[r"$\mathbf{{Y}}_{{{0}}}$".format(i) for i in range(n_features)])
fig_data2.tight_layout()
fig_data2.subplots_adjust(hspace=0, wspace=0)
savefig(fig_data2, "data-colour")


gridsearch_max_latent_factors = 10
gridsearch_max_components = 20


# Do a grid search.

Jm = np.arange(1, 1 + gridsearch_max_latent_factors)
Km = np.arange(1, 1 + gridsearch_max_components)
J_grid, K_grid, converged, metrics = grid_search.grid_search(Jm, Km, Y,
                                                             N_inits=1,
                                                             mcfa_kwds=mcfa_kwds)
ll = metrics["ll"]
bic = metrics["bic"]
mml = metrics["message_length"]

idx = np.nanargmin(bic)
jm_b, km_b = Jm[idx % bic.shape[1]], Km[int(idx / bic.shape[1])]

idx = np.nanargmin(mml)
jm_m, km_m = Jm[idx % mml.shape[1]], Km[int(idx / mml.shape[1])]


J_true, K_true = (data_kwds["n_latent_factors"], data_kwds["n_components"])

print(f"BIC is lowest at J = {jm_b} and K = {km_b}")
print(f"MML is lowest at J = {jm_m} and K = {km_m}")

print(f"True values are  J = {J_true} and K = {K_true}")



kwds = dict(converged=converged, 
            marker_function=np.nanargmin, 
            N=1000, 
            cmap="Spectral_r",
            truth=(J_true, K_true))

fig_ll = mpl_utils.plot_filled_contours(J_grid, K_grid, ll,
                                        colorbar_label=r"$-\log\mathcal{L}(\boldsymbol{\mathbf{Y}}|\boldsymbol{\mathbf{\Psi}})$", 
                                        **kwds)
fig_ll.axes[0].set_yticks([1, 5, 10, 15, 20])

fig_bic = mpl_utils.plot_filled_contours(J_grid, K_grid, bic,
                                         colorbar_label=r"$\textrm{BIC}$", 
                                         **kwds)
fig_bic.axes[0].set_yticks([1, 5, 10, 15, 20])


fig_mml = mpl_utils.plot_filled_contours(J_grid, K_grid, mml,
                                         colorbar_label=r"$I\left(\boldsymbol{\mathbf{Y}}|\boldsymbol{\mathbf{\Psi}}\right)$", 
                                         **kwds)
fig_mml.axes[0].set_yticks([1, 5, 10, 15, 20])


savefig(fig_ll, "gridsearch-ll-contours")
savefig(fig_bic, "gridsearch-bic-contours")
savefig(fig_mml, "gridsearch-mml-contours")


# Save grid search output in case we need it in future.
