
"""
Experiment using all GALAH data.
"""

from __future__ import division # Just in case. Use Python 3.

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter
from scipy import linalg

sys.path.insert(0, "../../")

from mcfa import (mcfa, grid_search, mpl_utils, utils)

import galah_dr3 as galah

import pickle

matplotlib.style.use(mpl_utils.mpl_style)

here = os.path.dirname(os.path.realpath(__file__))

prefix = os.path.basename(__file__)[:-3]

def savefig(fig, suffix):
    here = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(here, f"{prefix}-{suffix}")
    fig.savefig(f"{filename}-test-mask.png", dpi=150)
    fig.savefig(f"{filename}-test-mask.pdf", dpi=300)


N_elements = 18
use_galah_flags = True


mcfa_kwds = dict(init_factors="svd", init_components="kmeans++", tol=1e-8,
                 max_iter=10000)


abundance_counts = galah.get_abundances_breakdown(galah.available_elements,
                                                  use_galah_flags=use_galah_flags)

xlabels = abundance_counts.keys()
latex_labels = [r"{1}$\textrm{{{0}}}$".format(ea, "\n" if i % 2 else "") for i, ea in enumerate(xlabels)]
x = np.arange(len(xlabels))
y_independent = np.array(list(abundance_counts.values()))
y_cumulative = np.zeros_like(y_independent)

mask = np.ones(len(galah.data), dtype=bool)

for i, xlabel in enumerate(xlabels):
    mask *= galah._abundance_mask(xlabel, use_galah_flags)
    y_cumulative[i] = np.sum(mask)


def xystack(x, y, h_align="mid"):
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
    y1 = y.repeat(2)#[:-1]

    return (xx, y1)

kwds = dict(lw=2)

xs, y_i = xystack(x, y_independent)
xs, y_c = xystack(x, y_cumulative)
fig, ax = plt.subplots()
ax.plot(xs, y_i, "-", c="tab:blue", label=r"$\textrm{Independent (single element reported)}$", **kwds)
ax.plot(xs, y_c, "-", c="#000000", label=r"$\textrm{Cumulative (all elements leftward reported)}$", **kwds)

ax.set_xticks(x)
ax.xaxis.set_tick_params(width=0, which="major")
ax.set_xticks(x - 0.5, minor=True)
ax.xaxis.set_tick_params(direction="out", which="minor")

ax.set_xticklabels(latex_labels)

plt.legend(frameon=False, loc="lower left")

ax.set_yscale("log")
ax.set_xlim(xs[0], xs[-1])
ylim = (1e3, 1e6)


ax.set_ylabel(r"$\textrm{number of measurements in second GALAH data release}$")

x_ = N_elements - 0.5
ax.plot([x_, x_], ylim, "-", c="#666666", zorder=-1, linewidth=1, linestyle=":")


ax.set_ylim(*ylim)
fig.tight_layout()

savefig(fig, "abundance-counts")




elements = list(xlabels)[:N_elements]

mask = galah.get_abundance_mask(elements, use_galah_flags=use_galah_flags)

print(f"Number of stars: {sum(mask)}")


# MAGIC HACK WARNING DON'T DO THIS
minimum_snr_requirement = 50 # TODO: restrict further to 100 and see
mask *= (galah.data["snr_c1"] >= minimum_snr_requirement) \
      * (galah.data["snr_c2"] >= minimum_snr_requirement) \
      * (galah.data["snr_c3"] >= minimum_snr_requirement) \
      * (galah.data["snr_c4"] >= minimum_snr_requirement)


print(f"Number of stars after requiring SNR > {minimum_snr_requirement} in all bands: {sum(mask)}")


X_H, label_names = galah.get_abundances_wrt_h(elements, mask=mask)

"""
for i, label_name in enumerate(label_names):

    element = label_name.split("_")[0]
    denom = "fe" if element != "fe" else "h"
    col = f"e_{element}_{denom}"

    fig, ax = plt.subplots()
    ax.hist(galah.data[col][mask])
    ax.set_xlabel(col.replace("_", " "))
    ax.set_yscale("log")
"""


print(f"Data shape: {X_H.shape}")

"""
fe_index = label_names.index("fe_h")
ok = np.ones(len(label_names), dtype=bool)
ok[fe_index] = False

X_Fe = X_H - X_H.T[fe_index].T.reshape((-1, 1))
X_Fe = X_Fe[:, ok]

label_names.remove("fe_h")
"""





#X = utils.whiten(X_Fe)
X = 10**X_H
N, D = X.shape

# Do a gridsearch.
max_n_latent_factors = D - 1
max_n_components = 20

Js = 1 + np.arange(max_n_latent_factors)
Ks = 1 + np.arange(max_n_components)

Jg, Kg, converged, metrics = grid_search.grid_search(Js, Ks, X,
                                                     N_inits=1, mcfa_kwds=mcfa_kwds)

with open(f"{prefix}-gridsearch-results.pkl", "wb") as fp:
    pickle.dump((Jg, Kg, converged, metrics, X, mcfa_kwds), fp)



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

fig_mml = mpl_utils.plot_filled_contours(Jg[:, :-1], Kg[:, :-1], mml[:, :-1],
                                         colorbar_label=r"$\textrm{MML}$",
                                         **plot_filled_contours_kwds)
savefig(fig_mml, "gridsearch-mml")


raise a


model = mcfa.MCFA(n_components=K_best_bic, n_latent_factors=J_best_bic,
                  **mcfa_kwds)

model.fit(X)

A_est = model.theta_[model.parameter_names.index("A")]


latex_label_names = [r"$\textrm{{{0}}}$".format(ea.split("_")[0].title()) for ea in label_names]

# Draw unrotated.
J = model.n_latent_factors
L = model.theta_[model.parameter_names.index("A")]
cmap = mpl_utils.discrete_cmap(2 + J, base_cmap="Spectral_r")
colors = [cmap(1 + j) for j in range(J)]

fig = mpl_utils.visualize_factor_loads(L, latex_label_names, colors=colors)
savefig(fig, "latent-factors")

# Plot clustering in data space.
for k in range(K_best_bic):

    ok = np.argmax(model.tau_, axis=1) == k

    fig, ax = plt.subplots(2)
    ax[0].scatter(galah.data["raj2000"][mask][ok],
               galah.data["dej2000"][mask][ok])
    ax[0].set_title(f"{k}: {sum(ok)}")

    ax[1].scatter(galah.data["teff"][mask][ok],
                  galah.data["logg"][mask][ok],
                  c=galah.data["fe_h"][mask][ok],
                  s=1)

    ax[1].set_xlim(ax[1].get_xlim()[::-1])
    ax[1].set_ylim(ax[1].get_ylim()[::-1])


raise a



# Set some groups that we will try to rotate to.
astrophysical_grouping = [
    ["eu", "la", "ba"],
    ["ba", "y", "zn"],
    ["ni", "co", "v", "fe", "mn", "cr", "ti", "sc"],
    ["ca", "mg", "o", "si"],
    ["al", "na", "k"],
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


