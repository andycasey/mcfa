
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
from hashlib import md5

sys.path.insert(0, "../../")

from mcfa import (mcfa, grid_search, mpl_utils, utils)

import galah_dr2 as galah


matplotlib.style.use(mpl_utils.mpl_style)

here = os.path.dirname(os.path.realpath(__file__))

with open("config.yml") as fp:
    config = yaml.load(fp)

print(f"Config: {config}")

np.random.seed(config["random_seed"])

prefix = os.path.basename(__file__)[:-3]

unique_hash = md5((f"{config}").encode("utf-8")).hexdigest()[:5]

unique_config_path = f"{unique_hash}.yml"
if os.path.exists(unique_config_path):
    print(f"Warning: this configuration already exists: {unique_config_path}")

with open(unique_config_path, "w") as fp:
    yaml.dump(config, fp)

with open(__file__, "r") as fp:
    code = fp.read()

with open(f"{unique_hash}-{__file__}", "w") as fp:
    fp.write(code)

def savefig(fig, suffix):
    here = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(here, "eval-figs", f"{prefix}-{unique_hash}-{suffix}")
    fig.savefig(f"{filename}.png", dpi=150)
    fig.savefig(f"{filename}.pdf", dpi=300)


import os
os.system("rm -f *.pkl")

N_elements = 20
use_galah_flags = config["use_galah_flags"]

mcfa_kwds = dict()
mcfa_kwds.update(config["mcfa_kwds"])

elements = config[prefix]["elements"]
if config[prefix].get("ignore_elements", None) is not None:
    elements = [el for el in elements if el not in config[prefix]["ignore_elements"]]

print(elements)


mask = galah.get_abundance_mask(elements, use_galah_flags=use_galah_flags)


galah_cuts = config[prefix]["galah_cuts"]
if galah_cuts is not None:
    print(f"Applying cuts: {galah_cuts}")
    for k, (lower, upper) in galah_cuts.items():
        mask *= (upper >= galah.data[k]) * (galah.data[k] >= lower)


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
else:
    X = X_H

if not config["log_abundance"]:
    X = 10**X

if config["subtract_mean"]:
    X = X - np.mean(X, axis=0)


N, D = X.shape



# Do a gridsearch.
gs_options = config[prefix]["gridsearch"]
max_n_latent_factors = gs_options["max_n_latent_factors"]
max_n_components = gs_options["max_n_components"]

Js = 1 + np.arange(max_n_latent_factors)
Ks = 1 + np.arange(max_n_components)
N_inits = gs_options["n_inits"]

results_path = f"{prefix}-gridsearch-results.pkl"

if os.path.exists(results_path):

    with open(results_path, "rb") as fp:
        Jg, Kg, converged, meta, X, mcfa_kwds = pickle.load(fp)

else:

    Jg, Kg, converged, meta = grid_search.grid_search(Js, Ks, X,
                                                         N_inits=N_inits, mcfa_kwds=mcfa_kwds)

    with open(results_path, "wb") as fp:
        pickle.dump((Jg, Kg, converged, meta, X, mcfa_kwds), fp)



ll = meta["ll"]
bic = meta["bic"]
mml = meta["message_length"]

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
                                         colorbar_label=r"$I\left(\boldsymbol{\mathbf{Y}}|\boldsymbol{\mathbf{D}}\right)$", 
                                         **plot_filled_contours_kwds)
savefig(fig_mml, "gridsearch-mml")


model = meta["best_models"][config["adopted_metric"]]



latex_label_names = [r"$\textrm{{{0}}}$".format(ea.split("_")[0].title()) for ea in label_names]

# Draw unrotated.
J_max = config["max_n_latent_factors_for_colormap"]
J_max = 12
cmap = mpl_utils.discrete_cmap(J_max, base_cmap="Spectral")
colors = [cmap(j) for j in range(J_max)]#[::-1]


A_est = model.theta_[model.parameter_names.index("A")]

A_astrophysical = np.zeros_like(A_est)#np.random.normal(0, 0.1, size=A_est.shape)
for i, tes in enumerate(config["grouped_elements"][:model.n_latent_factors]):
    for j, te in enumerate(tes):
        try:
            idx = label_names.index("{0}_h".format(te.lower()))

        except ValueError:
            print(f"Skipping {te}")

        else:
            count = sum([(te in foo) for foo in config["grouped_elements"][:model.n_latent_factors]])
            A_astrophysical[idx, i] = 1.0/count

A_astrophysical /= np.clip(np.sqrt(np.sum(A_astrophysical, axis=0)), 1, np.inf)

# Un-assigned columns
for column_index in np.where(np.all(A_astrophysical == 0, axis=0))[0]:
    print(f"Warning: unassigned column index: {column_index}")
    A_astrophysical[:, column_index] = np.random.normal(0, 1e-2, size=D)

if config["correct_A_astrophysical"]:
    AL = linalg.cholesky(A_astrophysical.T @ A_astrophysical)
    A_astrophysical = A_astrophysical @ linalg.solve(AL, np.eye(model.n_latent_factors))



max_n_rotations = 5

for each in range(max_n_rotations):

    A_est = model.theta_[model.parameter_names.index("A")]

    R, p_opt, cov, *_ = utils.find_rotation_matrix(A_astrophysical, A_est, 
                                                   full_output=True)

    R_opt = utils.exact_rotation_matrix(A_astrophysical, A_est, 
                                        p0=np.random.uniform(-np.pi, np.pi, model.n_latent_factors**2))

    # WTF check R_opt.
    AL = linalg.cholesky(R_opt.T @ R_opt)
    R_opt2 = R_opt @ linalg.solve(AL, np.eye(model.n_latent_factors))

    chi1 = np.sum(np.abs(A_est @ R - A_astrophysical))
    chi2 = np.sum(np.abs(A_est @ R_opt2 - A_astrophysical))

    R = R_opt2 if chi2 < chi1 else R

    # Now make it a valid rotation matrix.
    model.rotate(R, X=X, ensure_valid_rotation=True)


import pickle
with open(f"{unique_hash}-{prefix}-model.pkl", "wb") as fp:
    pickle.dump(model, fp)



fig_fac = mpl_utils.plot_factor_loads_and_contributions(model, X, 
                                                        label_names=latex_label_names, colors=colors,
                                                        target_loads=A_astrophysical)
savefig(fig_fac, "latent-factors-and-contributions-with-targets")

fig_fac = mpl_utils.plot_factor_loads_and_contributions(model, X, 
                                                        label_names=latex_label_names, colors=colors)
savefig(fig_fac, "latent-factors-and-contributions")



# Plot clustering in data space and latent space.

# For the latent space we will just use a corner plot.
component_cmap = mpl_utils.discrete_cmap(7, base_cmap="Spectral_r")

fig = mpl_utils.plot_latent_space(model, X, ellipse_kwds=dict(alpha=0), s=10, edgecolor="none", alpha=1, c=[component_cmap(_) for _ in np.argmax(model.tau_, axis=1)], show_ticks=True,
                                  label_names=[r"$\mathbf{{S}}_{{{0}}}$".format(i + 1) for i in range(model.n_latent_factors)])
for ax in fig.axes:
    if ax.is_last_row():
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, 0, 1])

fig.tight_layout()

savefig(fig, "latent-space")


# For the data space we will use N x 2 panels of [X/Fe] vs [Fe/H], coloured by their responsibility.
#X_H, label_names = galah.get_abundances_wrt_h(elements, mask=mask)

X_H, label_names = galah.get_abundances_wrt_h(elements, mask=mask)


fig, axes = plt.subplots(4, 4, figsize=(8.5, 6.4))
axes = np.atleast_1d(axes).flatten()

assert len(axes) >= (len(elements) - 1)


x = X_H.T[label_names.index("fe_h")]
c = np.argmax(model.tau_, axis=1)

K = model.n_components



y_idx = 0
for i, ax in enumerate(axes):
    if i > len(elements):
        continue

    if label_names[i] == "fe_h":
        y_idx += 1

    y = X_H.T[y_idx] - x

    ax.scatter(x, y, c=[component_cmap(_) for _ in c], s=10, edgecolor="none", rasterized=True)

    element = label_names[y_idx].split("_")[0].title()
    ax.set_ylabel(r"$[\textrm{{{0}/Fe}}]$".format(element))
    y_idx += 1



x_lims = (-1.5, 0.5)
y_lims = (-0.5, 1.0)

for ax in axes:
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    ax.set_xticks([-1.5, -0.5, 0.5])
    #ax.set_yticks([-0.5, 0.25, 1.0, 1.75])
    ax.set_yticks([-0.5, 0, 0.5, 1.0])

    if ax.is_last_row():
        ax.set_xlabel(r"$[\textrm{Fe/H}]$")
    else:
        ax.set_xticklabels([])

    ax.plot(x_lims, [0, 0], ":", c="#666666", lw=0.5, zorder=-1)
    ax.plot([0, 0], y_lims, ":", c="#666666", lw=0.5, zorder=-1)

for ax in axes[len(elements):]:
    ax.set_visible(False)

fig.tight_layout()

savefig(fig, "data-space")


latex_elements = [r"$\textrm{{{0}}}$".format(le.split("_")[0].title()) for le in label_names]

fig_scatter = mpl_utils.plot_specific_scatter(model,
                                              steps=True,
                                              xlabel="",
                                              xticklabels=latex_elements,
                                              ylabel=r"$\textrm{specific scatter / dex}$",
                                              ticker_pad=20)
fig_scatter.axes[0].set_yticks(np.arange(0, 0.20, 0.05))
savefig(fig_scatter, "specific-scatter")


here = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(here, f"{prefix}-{unique_hash}-data.fits")

subset = galah.data[mask]
subset["association"] = np.argmax(model.tau_, axis=1)

subset.write(filename, overwrite=True)


