
"""
Experiment using all GALAH data.
"""

from __future__ import division # Just in case. Use Python 3.

import os
import sys
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import MaxNLocator
from collections import Counter
from scipy import linalg

sys.path.insert(0, "../")

from mcfa import (mcfa, grid_search, mpl_utils, utils)

import galah_dr2 as galah


matplotlib.style.use(mpl_utils.mpl_style)

here = os.path.dirname(os.path.realpath(__file__))

with open("config.yml") as fp:
    config = yaml.load(fp)

np.random.seed(config.get("random_seed", 0))

prefix = os.path.basename(__file__)[:-3]

def savefig(fig, suffix):
    here = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(here, f"{prefix}-{suffix}")
    fig.savefig(f"{filename}.png", dpi=150)
    fig.savefig(f"{filename}.pdf", dpi=300)


N_elements = 20
use_galah_flags = config["use_galah_flags"]


mcfa_kwds = dict(init_factors="svd", init_components="random", tol=1e-5,
                 max_iter=10000)


elements = ["Mg", "Al", "Ca", "Sc", "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Y", "Ba", "Eu"] + ["K", "Na", "Si"]


print(elements)


mask = galah.get_abundance_mask(elements, use_galah_flags=use_galah_flags)

print(f"Number of stars: {sum(mask)}")



X_H, label_names = galah.get_abundances_wrt_h(elements, mask=mask)





print(f"Data shape: {X_H.shape}")


def convert_xh_to_xy(X_H, label_names, y_label):

    index = label_names.index(y_label)
    y_h = X_H[:, index]

    offsets = np.zeros_like(X_H)
    for i, label_name in enumerate(label_names):
        if label_name == y_label: continue
        offsets[:, i] = y_h

    return X_H - offsets
 

if config["wrt_x_fe"]:
    X = convert_xh_to_xy(X_H, label_names, "fe_h")
if config["subtract_mean"]:
    X = X - np.mean(X, axis=0)

N, D = X.shape

# Do a gridsearch.
max_n_latent_factors = 10
max_n_components = 10

Js = 1 + np.arange(max_n_latent_factors)
Ks = 1 + np.arange(max_n_components)

results_path = f"{prefix}-gridsearch-results.pkl"

if os.path.exists(results_path):

    with open(results_path, "rb") as fp:
        Jg, Kg, converged, metrics, X, mcfa_kwds = pickle.load(fp)


else:

    Jg, Kg, converged, metrics = grid_search.grid_search(Js, Ks, X,
                                                         N_inits=2, mcfa_kwds=mcfa_kwds)

    with open(results_path, "wb") as fp:
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
                                 marker_function=np.nanargmin, N=100,
                                 cmap="Spectral_r")
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



model = mcfa.MCFA(n_components=K_best_mml, n_latent_factors=J_best_mml,
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




A_astrophysical = np.zeros_like(A_est)#np.random.normal(0, 0.1, size=A_est.shape)
for i, tes in enumerate(config["grouped_elements"][:model.n_latent_factors]):
    for j, te in enumerate(tes):
        try:
            idx = label_names.index("{0}_h".format(te.lower()))

        except ValueError:
            print(f"Skipping {te}")

        else:
            A_astrophysical[idx, i] = 1.0

A_astrophysical /= np.clip(np.sqrt(np.sum(A_astrophysical, axis=0)), 1, np.inf)


# Un-assigned columns
for column_index in np.where(np.all(A_astrophysical == 0, axis=0))[0]:
    A_astrophysical[:, column_index] = np.random.normal(0, 1e-2, size=D)

AL = linalg.cholesky(A_astrophysical.T @ A_astrophysical)
A_astrophysical = A_astrophysical @ linalg.solve(AL, np.eye(model.n_latent_factors))




R, p_opt, cov, *_ = utils.find_rotation_matrix(A_astrophysical, A_est, 
                                               full_output=True)

R_opt = utils.exact_rotation_matrix(A_astrophysical, A_est, 
                                    p0=np.random.uniform(-np.pi, np.pi, model.n_latent_factors**2))

chi1 = np.sum(np.abs(A_est @ R - A_astrophysical))
chi2 = np.sum(np.abs(A_est @ R_opt - A_astrophysical))

R = R_opt if chi2 < chi1 else R

# Now make it a valid rotation matrix.
model.rotate(R, X=X, ensure_valid_rotation=True)
J = model.n_latent_factors
L = model.theta_[model.parameter_names.index("A")]
cmap = mpl_utils.discrete_cmap(2 + J, base_cmap="Spectral_r")
colors = [cmap(1 + j) for j in range(J)]

elements = [ea.split("_")[0].title() for ea in label_names]

fig = mpl_utils.visualize_factor_loads(L, elements, colors=colors)
savefig(fig, "latent-factors-visualize")

fig = mpl_utils.visualize_factor_loads(L, elements, colors=colors, absolute_only=True)
savefig(fig, "latent-factors-visualize-abs")



t = galah.data[mask]
_, factor_scores, __, tau = model.factor_scores(X)
for i in range(model.n_latent_factors):
    t[f"S_{i}"] = factor_scores[:, i]


t["component"] = np.argmax(tau, axis=1)


for j in range(model.n_components):
    t[f"C_{j}"] = (t["component"] == j)


t.write(f"{prefix}-results.fits", overwrite=True)

# Plot the latent space.

fig = mpl_utils.plot_latent_space(model, X, cmap=cmap, show_ticks=True,
                                  label_names=[r"$\textbf{{S}}_{{{0}}}$".format(i) for i in range(model.n_latent_factors)])
savefig(fig, "latent-space")

# Plot the specific variances.
latex_elements = [r"$\textrm{{{0}}}$".format(le) for le in elements]

fig_scatter = mpl_utils.plot_specific_scatter(model,
                                              steps=True,
                                              xlabel=r"$\textrm{element}$", 
                                              xticklabels=latex_elements,
                                              ylabel=r"$\textrm{specific scatter / dex}$")
fig_scatter.axes[0].set_yticks(np.arange(0, 0.20, 0.05))
savefig(fig_scatter, "specific-scatter")


# Draw [Fe/H] vs [Mg/Fe] coloured by component.
from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots(figsize=(5.9, 4.5))
scat = ax.scatter(t["fe_h"], t["mg_fe"], c=tau.T[0], cmap="copper", s=10)
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))


ax.set_xlabel(r"$[\textrm{Fe}/\textrm{H}]$")
ax.set_ylabel(r"$[\textrm{Mg}/\textrm{Fe}]$")
ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

fig.tight_layout()

cbar = plt.colorbar(scat)
cbar.set_label(r"$\mathbf{\tau}_{{n,1}}$")

savefig(fig, "cluster")



raise a

