
"""
Experiment using a few gravitationally-bound clusters observed by GALAH.
"""

from __future__ import division # Just in case. Use Python 3.

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from scipy import linalg

sys.path.insert(0, "../../")

from mcfa import (mcfa, grid_search, mpl_utils, utils)

import galah

np.random.seed(100)


matplotlib.style.use(mpl_utils.mpl_style)


def savefig(fig, suffix):
    here = os.path.dirname(os.path.realpath(__file__))
    prefix = os.path.basename(__file__)[:-3]
    filename = os.path.join(here, f"{prefix}-{suffix}")
    fig.savefig(f"{filename}.png", dpi=150)
    fig.savefig(f"{filename}.pdf", dpi=300)


mcfa_kwds = dict(init_factors="svd", init_components="kmeans++", tol=1e-8,
                 max_iter=10000)


# Use a few carefully-selected clusters.

use_galah_flags = True

#cluster_names = ("M67", "Trumpler20", "NGC2204")
cluster_names = ("NGC1817", "NGC2204", "Berkeley32", "M67", "Trumpler20", "NGC6253",
                 "Ruprecht147")
cluster_names = ()

abundance_counts = galah.get_abundances_breakdown(galah.available_elements,
                                                  cluster_names,
                                                  use_galah_flags=use_galah_flags)

print(f"Abundance counts: {abundance_counts}")
for k, v in abundance_counts.items():
    print(k, v)







"""
elements = "Fe, Mg, Na, K, Sc, Ca, O, Zn, Al, Cu, Mn, Cr, Ba, Ti, Ni, Si, Y, "\
           "Co, V, La, Ce, Zr, Eu".split(", ")

elements = ["Fe", "Ni", "Mn", "Cr", "Ti", "Al", "Mg", "Na", "K", "Sc", "Ca", "O", "Zn", "Ba", "Y", "Ba", "Eu"]
"""

#elements = "Mg Al Ca Sc Ti Cr Mn Fe Co Ni Sr Y Ba".split()
elements = "Fe Mg Na K Sc Ca O Zn Al Cu Mn Cr Ba Ti Ni Si Y Eu".split()
elements = "Fe Mg Na K Sc Ca O Zn Al Cu Mn Cr Ba Ti Ni Si Y".split()


mask = galah.get_abundance_mask(elements, cluster_names,
                                use_galah_flags=use_galah_flags)


minimum_clkuster_members = 10
cluster_names = []
for cluster_name, count in Counter(galah.data["wg4_field"][mask]).items():
    if count >= minimum_clkuster_members:
      cluster_names.append(cluster_name)

mask = galah.get_abundance_mask(elements, cluster_names,
                                use_galah_flags=use_galah_flags)

print(f"Number of stars after requiring minimum cluster members of {minimum_clkuster_members}: {sum(mask)}")

X_H, label_names = galah.get_abundances_wrt_h(elements, cluster_names, 
                                              use_galah_flags=use_galah_flags)

print(f"Data shape: {X_H.shape}")

X = utils.whiten(X_H)


# Do a gridsearch.
max_n_latent_factors = 15
max_n_components = 15

Js = 1 + np.arange(max_n_latent_factors)
Ks = 1 + np.arange(max_n_components)

Jg, Kg, converged, metrics = grid_search.grid_search(Js, Ks, X, 
                                                     N_inits=5,
                                                     mcfa_kwds=mcfa_kwds)

ll = metrics["ll"]
bic = metrics["bic"]

J_best_ll, K_best_ll = grid_search.best(Js, Ks, -ll)
J_best_bic, K_best_bic = grid_search.best(Js, Ks, bic)


print(f"Best log likelihood  at J = {J_best_ll} and K = {K_best_ll}")
print(f"Best BIC value found at J = {J_best_bic} and K = {K_best_bic}")

# Plot some contours.
plot_filled_contours_kwds = dict(converged=converged, 
                                 marker_function=np.nanargmin,
                                 cmap="Spectral_r")

fig_ll = mpl_utils.plot_filled_contours(Jg, Kg, -ll,
                                        colorbar_label=r"$-\log\mathcal{L}$",
                                        **plot_filled_contours_kwds)
savefig(fig_ll, "gridsearch-ll")

fig_bic = mpl_utils.plot_filled_contours(Jg, Kg, bic,
                                         colorbar_label=r"$\textrm{BIC}$",
                                         **plot_filled_contours_kwds)
savefig(fig_bic, "gridsearch-bic")


lls = []
models = []
for i in range(10):

    model = mcfa.MCFA(n_components=K_best_bic, n_latent_factors=J_best_bic,
                      **mcfa_kwds)
    try:
        model.fit(X)

    except:
        continue

    else:
        models.append(model)
        lls.append(model.log_likelihood_)

model = models[np.argmax(lls)]


A_est = model.theta_[model.parameter_names.index("A")]

# Rotate to be lose to astrophysical.
astrophysical_grouping = [
    ["al", "k"],
    ["na", "o"],
    ["ca", "mg", "ti", "si"],
    ["ni", "co", "zn", "fe", "mn", "cr", "ti", "sc", "cu"],
    ["sr", "y", "ba"],
    ["eu"]
]

A_astrophysical = np.zeros_like(A_est)
for i, tes in enumerate(astrophysical_grouping):
    for j, t in enumerate(tes):
        te = f"{t}_h"
        #idx = label_names.index("{0}_h".format(te.lower()))
        if te not in label_names:
            print(f"ignoring {te}")
            continue

        idx = label_names.index(te)
        A_astrophysical[idx, i] = 1.0

A_astrophysical /= np.sqrt(np.sum(A_astrophysical, axis=0))
A_astrophysical[~np.isfinite(A_astrophysical)] = 0.0


R, p_opt, cov, *_ = utils.find_rotation_matrix(A_astrophysical, A_est, 
                                               full_output=True)

R_opt = utils.exact_rotation_matrix(A_astrophysical, A_est, 
                                    p0=np.random.uniform(-np.pi, np.pi, model.n_latent_factors**2))

chi1 = np.sum(np.abs(A_est @ R - A_astrophysical))
chi2 = np.sum(np.abs(A_est @ R_opt - A_astrophysical))

R = R_opt if chi2 < chi1 else R

# Now make it a valid rotation matrix.
R = model.rotate(R, X=X, ensure_valid_rotation=True)

J = model.n_latent_factors
L = model.theta_[model.parameter_names.index("A")]
cmap = mpl_utils.discrete_cmap(2 + J, base_cmap="Spectral_r")
colors = [cmap(1 + j) for j in range(J)]

latex_label_names = [ea.split("_")[0].title() for ea in label_names]
fig = mpl_utils.visualize_factor_loads(L, latex_label_names, colors=colors)
savefig(fig, "latent-factors-visualize")


raise a
fig_data = mpl_utils.corner_scatter(X, c=np.argmax(model.tau_, axis=1))
fig_data.tight_layout()
fig_data.subplots_adjust(hspace=0, wspace=0)
#savefig(fig_data, "data")


raise a



# PLot the clustering in latent space?

# Plot the clustering in data space?

# Plot the specific variances.
fig_scatter = mpl_utils.plot_specific_scatter(model, scales=np.std(X_H, axis=0),
                                              steps=True,
                                              xlabel=xlabel, 
                                              xticklabels=xticklabels,
                                              ylabel=r"$\textrm{specific scatter / dex}$")
fig_scatter.axes[0].set_yticks(np.arange(0, 0.30, 0.05))
fig_scatter.savefig("specific-scatter")



raise a








cluster_names = [
    #"Pleiades",
    #"Hyades",
    #"Blanco1",
    #"Ruprecht147",
    #"NGC6253"
]

tracer_elements = [
    #["Na", "Al", "K"],
    ["Si", "Mg", "Ca", "O"],
    ["Ni", "Fe", "Ti", "Cu"],
    ["Zn", "Ba", "Y", "Eu"],
]
"""
tracer_elements = [
    ["Mg", "O"],
    ["Fe", "Ni"],
    ["Ba", "Y"]
]
"""

elements = [el for sublist in tracer_elements for el in sublist]
finite_only = True

mask = galah.get_abundance_mask(elements, cluster_names,
                                finite_only=finite_only)
cluster_counts = Counter(galah.data["wg4_field"][mask])
abundance_counts = galah.get_abundances_breakdown(elements, cluster_names,
                                                  finite_only=finite_only)

print(f"Count: {sum(mask)}")
print(f"Cluster breakdown: {cluster_counts}")
print(f"Abundance breakdown: {abundance_counts}")

suggested_elements = galah.suggest_abundances_to_include(mask, 
                                                         elements,
                                                         finite_only=finite_only,
                                                         percentage_threshold=10)

print(f"Suggested elements to include: {suggested_elements}")

X_H, labels = galah.get_abundances_wrt_h(elements, cluster_names, finite_only)

X = utils.whiten(X_H)

# Plot the data.
fig_data = mpl_utils.corner_scatter(X, c="#000000", s=1, alpha=0.5, 
                                    figsize=(8, 8))
fig_data.tight_layout()
fig_data.subplots_adjust(hspace=0, wspace=0)


# Rotate to astrophysical factors.
n_components = len(set(galah.data["wg4_field"][mask]))
n_components = 1
n_latent_factors = len(tracer_elements)
model = mcfa.MCFA(n_components=n_components,
                  n_latent_factors=n_latent_factors,
                  **mcfa_kwds)
model.fit(X)

A_est = model.theta_[model.parameter_names.index("A")]

A_astrophysical = np.zeros_like(A_est)
for i, tes in enumerate(tracer_elements):
    for j, te in enumerate(tes):
        idx = labels.index("{0}_h".format(te.lower()))
        A_astrophysical[idx, i] = 1.0

AL = linalg.cholesky(A_astrophysical.T @ A_astrophysical)
A_astrophysical = A_astrophysical @ linalg.solve(AL, np.eye(n_latent_factors))

# Rotate latent factors to be close to astrophysical.
R, p_opt, cov, *_ = utils.find_rotation_matrix(A_astrophysical, A_est, 
                                               full_output=True)

# Vary the angles....
N_draws = 100
A_est_rotations = np.zeros((N_draws, *A_est.shape))

draws = np.random.multivariate_normal(p_opt, cov, size=N_draws)

for i, draw in enumerate(draws):
    A_est_rotations[i] = A_est @ utils.givens_rotation_matrix(*draw)

A_est_rotation_percentiles = np.percentile(A_est_rotations, [16, 50, 84], axis=0)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

D, J = A_est.shape
xi = 1 + np.arange(D)

fig_factor_loads, ax = plt.subplots()

for j in range(J):
    ax.plot(xi, A_astrophysical.T[j], ":", lw=1, c=colors[j])
    ax.plot(xi, A_est_rotation_percentiles[1, :, j], "-", lw=2, c=colors[j])
    ax.fill_between(xi, 
                    A_est_rotation_percentiles[0, :, j], 
                    A_est_rotation_percentiles[2, :, j],
                    facecolor=colors[j], alpha=0.3)

ax.set_xticks(xi)
ax.set_xticklabels([l.split("_")[0].title() for l in labels])
ax.set_xlabel(r"$\textrm{dimension}$")
ax.set_ylabel(r"$\mathbf{L}$")

ylim = np.ceil(10 * np.abs(ax.get_ylim()).max()) / 10
ax.plot([0, D + 1], [0, 0], ":", c="#000000", zorder=-1, lw=0.5)
ax.set_xlim(0.5, D + 0.5)

ax.set_ylim(-ylim, +ylim)
ax.set_yticks([-ylim, 0, ylim])

fig_factor_loads.tight_layout()


X_drawn = model.sample(sum(mask))
fig_data = mpl_utils.corner_scatter(X, 
                                    c="#000000", s=1, alpha=0.5, figsize=(8, 8))
mpl_utils.corner_scatter(X_drawn,
                         c="tab:blue", s=1, alpha=0.25, zorder=10, fig=fig_data)

fig_data.tight_layout()
fig_data.subplots_adjust(hspace=0, wspace=0)



raise a


# Run a grid search.
max_n_latent_factors = len(elements) - 1
max_n_components = 10

Js = 1 + np.arange(max_n_latent_factors)
Ks = 1 + np.arange(max_n_components)

Jg, Kg, converged, ll, bic, pseudo_bic = grid_search.grid_search(Js, Ks, X, 
                                                                 mcfa_kwds)

J_best_ll, K_best_ll = grid_search.best(Js, Ks, -ll)
J_best_bic, K_best_bic = grid_search.best(Js, Ks, bic)

print(f"Best log likelihood  at J = {J_best_ll} and K = {K_best_ll}")
print(f"Best BIC value found at J = {J_best_bic} and K = {K_best_bic}")

print(f"True values at          J = ? and K = {len(cluster_names)}")

# Plot some contours.
plot_filled_contours_kwds = dict(converged=converged, marker_function=np.nanargmin)
fig_ll = mpl_utils.plot_filled_contours(Jg, Kg, -ll,
                                        colorbar_label=r"$-\log\mathcal{L}$",
                                        **plot_filled_contours_kwds)

fig_bic = mpl_utils.plot_filled_contours(Jg, Kg, bic,
                                         colorbar_label=r"$\textrm{BIC}$",
                                         **plot_filled_contours_kwds)


