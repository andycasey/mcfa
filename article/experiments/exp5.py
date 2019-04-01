
"""
Experiment using all APOGEE DR12 data.
"""

from __future__ import division # Just in case. Use Python 3.

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from scipy import linalg
from astropy.table import Table

sys.path.insert(0, "../../")

from mcfa import (mcfa, grid_search, mpl_utils, utils)

#import galah_dr3 as galah

matplotlib.style.use(mpl_utils.mpl_style)


def savefig(fig, suffix):
    prefix = os.path.basename(__file__)[:-3]
    filename = f"{prefix}-{suffix}.png"
    fig.savefig(filename, dpi=150)


mcfa_kwds = dict(init_factors="svd", init_components="kmeans++", tol=1e-2,
                 max_iter=10000, random_seed=42)


use_galah_flags = False
require_flag_cannon_ok = True



elements = ["Na", "Mg", "Fe", "Sc", "Ti", "Zn", "Mn", "Y", "Ca", "Ni", "Cr", "O",
            "Si", "K", "Ba", "V", "Cu", "Al", "La", "Eu"]
#            # "Ni", "Mn", "Cr", "Ti", "Al", "Mg", "Na", "K", "Sc", "Ca", "O", "Zn", "Ba", "Y", "Ba", "Eu"]


barklem_elements = ['mg_h', 'al_h', 'ca_h', 'sc_h', 'ti_h', 'cr_h', 'mn_h', 'fe_h', 'co_h', 'ni_h', 'sr_h', 'y_h', 'ba_h', 'eu_h']
missing_elements = ["Sr"]

barklem_elements = [ea.split("_")[0].title() for ea in barklem_elements]
barklem_elements = list(set(barklem_elements).difference(missing_elements))

"""
In [2]: set(elements).difference(barklem_elements)
Out[2]: {'Cu', 'K', 'La', 'Na', 'O', 'Si', 'V', 'Zn'}

In [3]: set(barklem_elements).difference(elements)
Out[3]: {'Co', 'Sr'}
"""

elements = ["AL", "CA", "C", "FE", "K", "MG", "MN", "NA", "NI", "N", "O", "SI", "S", "TI", "V"]

elements = ["AL", "CA", "FE", "K", "MG", "MN", "NA", "NI", "SI", "S", "TI", "V"]
label_names = [f"{el.lower()}_h" for el in elements]

data = Table.read("../../catalogs/tc-apogee-dr12-regularized-release.fits")


X_H = np.array([data[f"{el}_H"] for el in elements]).T

print(f"Data shape: {X_H.shape}")


X = X_H

#X = X_H

# Do a gridsearch.
max_n_latent_factors = 5
#max_n_components = 100

Js = 1 + np.arange(max_n_latent_factors)
#Ks = 1 + np.arange(max_n_components)
Ks = np.array([1, 2, 5, 10, 20, 50])

Jg, Kg, converged, metrics  = grid_search.grid_search(Js, Ks, X, N_inits=1,
                                                                 mcfa_kwds=mcfa_kwds,
                                                                 suppress_exceptions=False)

ll = metrics["ll"]
bic = metrics["bic"]
mml = metrics["message_length"]

J_best_ll, K_best_ll = grid_search.best(Js, Ks, -ll)
J_best_bic, K_best_bic = grid_search.best(Js, Ks, bic)
J_best_mml, K_best_mml = grid_search.best(Js, Ks, mml)


print(f"Best log likelihood  at J = {J_best_ll} and K = {K_best_ll}")
print(f"Best BIC value found at J = {J_best_bic} and K = {K_best_bic}")
print(f"Best MML value found at J = {J_best_mml} and K = {K_best_mml}")



# Plot some contours.
plot_filled_contours_kwds = dict(converged=converged, marker_function=np.nanargmin)
fig_ll = mpl_utils.plot_filled_contours(Jg, Kg, -ll,
                                        colorbar_label=r"$-\log\mathcal{L}$",
                                        **plot_filled_contours_kwds)
savefig(fig_ll, "gridsearch-ll")

fig_bic = mpl_utils.plot_filled_contours(Jg, Kg, bic,
                                         colorbar_label=r"$\textrm{BIC}$",
                                         **plot_filled_contours_kwds)
savefig(fig_bic, "gridsearch-bic")


fig_mml = mpl_utils.plot_filled_contours(Jg, Kg, mml,
                                         colorbar_label=r"$\textrm{MML}$",
                                         **plot_filled_contours_kwds)
savefig(fig_mml, "gridsearch-mml")

# 
raise a


model = mcfa.MCFA(n_components=K_best_bic, n_latent_factors=J_best_bic,
                  **mcfa_kwds)

model.fit(X)

A_est = model.theta_[model.parameter_names.index("A")]


# Draw as-is.
xlabel = r"$\textrm{element}$"
xticklabels = [r"$\textrm{{{0}}}$".format(ln.split("_")[0].title()) \
                        for ln in label_names]

fig_factors_unrotated = mpl_utils.plot_factor_loads(A_est,
                                                    xlabel=xlabel,
                                                    xticklabels=xticklabels)

# Set some groups that we will try to rotate to.
astrophysical_grouping = [
    ["eu", "la", "ba"],
    ["ba", "y", "zn"],
    ["ni", "co", "v", "fe", "mn", "cr", "ti", "sc"],
    ["ca", "mg", "o", "si"],
    ["al", "na", "k"],
]

load_labels = [
r"$\textrm{odd-Z?}$",
r"$\textrm{alpha-process?}$",
r"$\textrm{CCSN?}$",
r"$\textrm{s-process?}$",
r"$\textrm{r-process?}$",
]



A_astrophysical = np.zeros_like(A_est)
for i, tes in enumerate(astrophysical_grouping[:model.n_latent_factors]):
    for j, te in enumerate(tes):
        try:
            idx = label_names.index("{0}_h".format(te.lower()))

        except ValueError:
            print(f"Skipping {te}")

        else:
            A_astrophysical[idx, i] = 1.0

#AL = linalg.cholesky(A_astrophysical.T @ A_astrophysical)
#A_astrophysical = A_astrophysical @ linalg.solve(AL, np.eye(model.n_latent_factors))


np.random.seed(42)

scales = 1 # np.std(X_H, axis=0)

fig_factors_rotated = mpl_utils.plot_factor_loads(A_est, 
                                                  separate_axes=True,
                                                  target_loads=A_astrophysical,
                                                  flip_loads=None,
                                                  n_rotation_inits=100,
                                                  show_target_loads=False,
                                                  xlabel=xlabel,
                                                  #load_labels=load_labels,
                                                  xticklabels=xticklabels,
                                                  legend_kwds=dict(loc="upper left",
                                                                   fontsize=14.0),
                                                  figsize=(5.5, 7.75),
                                                  )
savefig(fig_factors_rotated, "latent-factors")

# PLot the clustering in latent space?

# Plot the clustering in data space?

# Plot the specific variances.
fig_scatter = mpl_utils.plot_specific_scatter(model, scales=scales,
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


