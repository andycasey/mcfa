
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from time import time

sys.path.insert(0, "../../")

from mcfa import (mcfa, mpl_utils, utils)

matplotlib.style.use(mpl_utils.mpl_style)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Should all go into exp1.yaml

data_kwds = dict(n_samples=1000,
                 n_features=15,
                 n_components=10,
                 n_latent_factors=3,
                 psi_scale=1,
                 latent_scale=1,
                 random_seed=42,
                 eta=1)

mcfa_kwds = dict(tol=1e-5, 
                 max_iter=1_000,
                 init_factors="random",
                 init_components="random",
                 random_seed=42)

gridsearch_max_latent_factors = 5
gridsearch_max_components = 15

# Done with vars

Y, truth = utils.generate_data(**data_kwds)

# Fit with true number of latent factors and components.
model = mcfa.MCFA(n_components=data_kwds["n_components"],
                  n_latent_factors=data_kwds["n_latent_factors"],
                  **mcfa_kwds)
tick = time()
model.fit(Y)
tock = time()

print(f"Model took {tock - tick:.1f} seconds")

# Plot the log-likelihood with increasing iterations.
fig_iterations, ax = plt.subplots()

ll = model.log_likelihoods_
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


# Plot the data, with samples
Y_drawn = model.sample(data_kwds["n_samples"])
fig_data = mpl_utils.corner_scatter(Y, 
                                    c="#000000", s=1, alpha=0.5, figsize=(8, 8))
mpl_utils.corner_scatter(Y_drawn,
                         c="tab:blue", s=1, alpha=0.25, zorder=10, fig=fig_data)

fig_data.tight_layout()
fig_data.subplots_adjust(hspace=0, wspace=0)



# Plot the true latent factors w.r.t. the estimated ones, after rotation.
A_true = truth["A"]
A_est = model.theta_[model.parameter_names.index("A")]

R = utils.find_rotation_matrix(A_true, A_est, n_inits=100)
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


# Do a grid search.
Js = np.arange(1, 1 + gridsearch_max_latent_factors)
Ks = np.arange(1, 1 + gridsearch_max_components)

Jm, Km = np.meshgrid(Js, Ks)

ll = np.nan * np.ones_like(Jm)
bic = np.nan * np.ones_like(Jm)
pseudo_bic = np.nan * np.ones_like(Jm)
pseudo_bic_kwds = dict(omega=1, gamma=0.1)

converged = np.zeros(Jm.shape, dtype=bool)

for j, J in enumerate(Js):
    for k, K in enumerate(Ks):

        print(f"At J = {J}, K = {K}")

        model = mcfa.MCFA(n_latent_factors=J, n_components=K, **mcfa_kwds)

        try:
            model.fit(Y)

        except:
            continue

        else:
            ll[k, j] = model.log_likelihood_
            bic[k, j] = model.bic(Y)
            pseudo_bic[k, j] = model.pseudo_bic(Y, **pseudo_bic_kwds)
            converged[k, j] = True

idx = np.nanargmin(bic)
jm_b, km_b = Js[idx % bic.shape[1]], Ks[int(idx / bic.shape[1])]

idx = np.nanargmin(pseudo_bic)
jm_pb, km_pb = Js[idx % pseudo_bic.shape[1]], Ks[int(idx / pseudo_bic.shape[1])]

J_true, K_true = (data_kwds["n_latent_factors"], data_kwds["n_components"])

print(f"BIC is lowest at J = {jm_b} and K = {km_b}")
print(f"pseudo-BIC is lowest at J = {jm_pb} and K = {km_pb}")
print(f"True values are  J = {J_true} and K = {K_true}")


def plot_contours(J, K, Z, N=100, colorbar_label=None, 
                  converged=None, converged_kwds=None, 
                  marker_function=None, marker_kwds=None, 
                  ax=None, **kwargs):
    
    power = np.min(np.log10(Z).astype(int))

    Z = Z.copy() / (10**power)

    if ax is None:
        w = 0.2 + 4 + 0.1
        h = 0.5 + 4 + 0.1
        if colorbar_label is not None:
            w += 1
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig = ax.figure

    cf = ax.contourf(J, K, Z, N, **kwargs)

    ax.set_xlabel(r"$\textrm{Number of latent factors } J$")
    ax.set_ylabel(r"$\textrm{Number of clusters } K$")

    if converged is not None:
        kwds = dict(marker="x", c="#000000", s=10, linewidth=1, alpha=0.3)
        if converged_kwds is not None:
            kwds.update(converged_kwds)

        ax.scatter(J[~converged], K[~converged], **kwds)

    if marker_function is not None:
        idx = marker_function(Z)
        j_m, k_m = (J[0][idx % Z.shape[1]], K.T[0][int(idx / Z.shape[1])])
        kwds = dict(facecolor="#ffffff", edgecolor="#000000", linewidth=1.5,
                    s=50, zorder=15)
        if marker_kwds is not None:
            kwds.update(marker_kwds)

        ax.scatter(j_m, k_m, **kwds)

    if colorbar_label is not None:
        cbar = plt.colorbar(cf)
        cbar.set_label(colorbar_label + " $/\,\,10^{0}$".format(power))
        cbar.ax.yaxis.set_major_locator(MaxNLocator(5))

    edge_percent = 0.025
    x_range = np.ptp(J)
    y_range = np.ptp(K)
    ax.set_xlim(J.min() - x_range * edge_percent,
                J.max() + x_range * edge_percent)

    ax.set_ylim(K.min() - y_range * edge_percent,
                K.max() + y_range * edge_percent)
    
    ax.xaxis.set_major_locator(MaxNLocator(9))
    ax.yaxis.set_major_locator(MaxNLocator(9))

    ax.set_xticks(Js.astype(int))
    ax.yaxis.set_tick_params(width=0)
    ax.xaxis.set_tick_params(width=0)

    fig.tight_layout()

    return fig

kwds = dict(converged=converged)
fig_ll = plot_contours(Jm, Km, -ll,
                       marker_function=lambda *_: np.nanargmin(-ll),
                       colorbar_label=r"$-\log\mathcal{L}$", 
                       **kwds)

fig_bic = plot_contours(Jm, Km, bic,
                        marker_function=lambda *_: np.nanargmin(bic),
                        colorbar_label=r"$\textrm{BIC}$", 
                        **kwds)

fig_pseudo_bic = plot_contours(Jm, Km, pseudo_bic,
                               marker_function=lambda *_: np.nanargmin(pseudo_bic),
                               colorbar_label=r"$\textrm{pseudo-BIC}$", 
                               **kwds)

