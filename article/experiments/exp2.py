
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

sys.path.insert(0, "../../")

from mcfa import (mcfa, grid_search, mpl_utils, utils)

import galah

matplotlib.style.use(mpl_utils.mpl_style)

mcfa_kwds = dict(init_factors="random", init_components="random", tol=1e-5,
                 max_iter=10000)

cluster_names = [
    #"Pleiades",
    #"Hyades",
    #"Blanco1",
    #"Ruprecht147",
    #"NGC6253"
]

tracer_elements = [
    #["Na", "Al", "K"],
    #["Mg", "Ca", "Si", "O"],
    ["Si", "Mg", "Ca"],
    ["Ni", "Fe", "Cr", "Sc", "Ti", "Cu"],
    ["Ba", "Y"],
    ["Eu"],
]
"""
tracer_elements = [
    ["Mg", "O"],
    ["Fe", "Ni"],
    ["Ba", "Y"]
]
"""

elements = [el for sublist in tracer_elements for el in sublist]
finite_only = False

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
#X = X_H
#X = utils.whiten(X_H)

# Plot the data.
fig_data = mpl_utils.corner_scatter(X, c="#000000", s=1, alpha=0.5, 
                                    figsize=(8, 8))
fig_data.tight_layout()
fig_data.subplots_adjust(hspace=0, wspace=0)


# Rotate to astrophysical factors.
model = mcfa.MCFA(n_components=max(1, len(cluster_names)), 
                  n_latent_factors=len(tracer_elements),
                  **mcfa_kwds)
model.fit(X)

A_est = model.theta_[model.parameter_names.index("A")]

A_astrophysical = np.zeros_like(A_est)
for i, tes in enumerate(tracer_elements):
    for j, te in enumerate(tes):
        idx = labels.index("{0}_h".format(te.lower()))
        A_astrophysical[idx, i] = 1.0

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


