




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

with open(f"config.yml") as fp:
    config = yaml.load(fp)

print(f"Config: {config}")

np.random.seed(config["random_seed"])

prefix = os.path.basename(__file__)[:-3]
unique_hash = "none"

def savefig(fig, suffix):
    here = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(here, "eval-figs", f"{prefix}-{unique_hash}-{suffix}")
    fig.savefig(f"{filename}.png", dpi=150)
    fig.savefig(f"{filename}.pdf", dpi=300)


# Get the same original data from the experiment hash:

import os

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


# This will be the default mask. 
possible_additions_mask = np.zeros(len(mask), dtype=bool)
# Start off by selecting everything that isn't in mask
possible_additions_mask = ~mask

galah_cuts = config[prefix]["galah_cuts"]
if galah_cuts is not None:
    print(f"Applying cuts: {galah_cuts}")
    for k, (lower, upper) in galah_cuts.items():
        possible_additions_mask *= (upper >= galah.data[k]) * (galah.data[k] >= lower)

any_measured = np.any(np.array([galah._abundance_mask(el, True) for el in elements]), axis=0)
possible_additions_mask *= any_measured
print(f"Number of stars with complete abundances: {sum(mask)}")
print(f"Number of stars meeting cuts with incomplete abundances: {sum(possible_additions_mask)}")



def convert_xh_to_xy(X_H, label_names, y_label):

    index = label_names.index(y_label)
    y_h = X_H[:, index]

    offsets = np.zeros_like(X_H)
    for i, label_name in enumerate(label_names):
        if label_name == y_label: continue
        offsets[:, i] = y_h

    return X_H - offsets


# Use the number of latent factors and components found previously.
J = 6
K = 3
gs_options = config[prefix]["gridsearch"]
N_inits = gs_options["n_inits"]


OVERWRITE = False

# Do custom things for the biggest one 
"""
if True:
    size = sum(possible_additions_mask)

    indices = np.random.choice(np.where(possible_additions_mask)[0], size=size, replace=False)

    # Generate mask.
    trial_mask = np.copy(mask)
    trial_mask[indices] = True

    X_H, label_names = galah.get_unflagged_abundances_wrt_h(elements, trial_mask)


    if config["wrt_x_fe"]:
        X = convert_xh_to_xy(X_H, label_names, "fe_h")
    else:
        X = X_H

    if not config["log_abundance"]:
        X = 10**X

    if config["subtract_mean"]:
        X = X - np.nanmean(X, axis=0)


    # OK, let's do a rough grid to figure out where we are.

    #Js = np.arange(7, 12 + 1)
    ## At J = 7, K_best = 16
    #Ks = np.arange(16 - 3, 16 + 3 + 1)

    Js = np.array([10, 11, 12])
    Ks = np.array([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])

    # Claims J = 10 and K = 17 best

    # Claims J = 10 and K = 20 best.

    # Let's assume that for now and do a mini-grid around there.

    N_inits = 5

    gJs, gKs, converged, meta = grid_search.grid_search(Js, Ks, X, N_inits=N_inits, mcfa_kwds=mcfa_kwds)
    mml = meta["message_length"]


    # Plot some contours.
    plot_filled_contours_kwds = dict(converged=converged,
                                     marker_function=np.nanargmin, N=100,
                                     cmap="Spectral_r")

    fig_mml = mpl_utils.plot_filled_contours(gJs, gKs, mml,
                                             colorbar_label=r"$\textrm{MML}$",
                                             **plot_filled_contours_kwds)
    savefig(fig_mml, f"size-{size}-gridsearch-mml")
    raise a
"""




sizes = (1000, 10000, sum(possible_additions_mask))
pm_jk = (2,    5,     5)

saved_models = dict()

for size, delta_jk in zip(sizes, pm_jk):

    Js = np.arange(J - delta_jk, J + delta_jk + 1)
    Ks = np.arange(K - delta_jk, K + delta_jk + 1)

    Js = Js[Js >= 1]
    Ks = Ks[Ks >= 1]

    output_path = f"eval-figs/{prefix}-models-size-{size}.pkl"
    if os.path.exists(output_path) and not OVERWRITE:
        print(f"Skipping size = {size} because {output_path} already exists")
        continue
    
    print(f"Doing size = {size}")

    indices = np.random.choice(np.where(possible_additions_mask)[0], size=size, replace=False)

    # Generate mask.
    trial_mask = np.copy(mask)
    trial_mask[indices] = True

    X_H, label_names = galah.get_unflagged_abundances_wrt_h(elements, trial_mask)


    print(f"SIZE: {X_H.shape}")

    if config["wrt_x_fe"]:
        X = convert_xh_to_xy(X_H, label_names, "fe_h")
    else:
        X = X_H

    if not config["log_abundance"]:
        X = 10**X

    if config["subtract_mean"]:
        X = X - np.nanmean(X, axis=0)

    gJs, gKs, converged, meta = grid_search.grid_search(Js, Ks, X, N_inits=N_inits, mcfa_kwds=mcfa_kwds)

    
    # Plot the grid space.

    ll = meta["ll"]
    bic = meta["bic"]
    mml = meta["message_length"]

    J_best_ll, K_best_ll = grid_search.best(Js, Ks, -ll)
    J_best_bic, K_best_bic = grid_search.best(Js, Ks, bic)
    J_best_mml, K_best_mml = grid_search.best(Js, Ks, mml)


    print(f"Best log likelihood  for N = {size} at J = {J_best_ll} and K = {K_best_ll}")
    print(f"Best BIC value found for N = {size} at J = {J_best_bic} and K = {K_best_bic}")
    print(f"Best MML value found for N = {size} at J = {J_best_mml} and K = {K_best_mml}")

    try:
        # Plot some contours.
        plot_filled_contours_kwds = dict(converged=converged,
                                         marker_function=np.nanargmin, N=100,
                                         cmap="Spectral_r")
        fig_ll = mpl_utils.plot_filled_contours(gJs, gKs, -ll,
                                                colorbar_label=r"$-\log\mathcal{L}$",
                                                **plot_filled_contours_kwds)
        savefig(fig_ll, f"size-{size}-gridsearch-ll")

        fig_bic = mpl_utils.plot_filled_contours(gJs, gKs, bic,
                                                 colorbar_label=r"$\textrm{BIC}$",
                                                 **plot_filled_contours_kwds)
        savefig(fig_bic, f"size-{size}-gridsearch-bic")

        fig_mml = mpl_utils.plot_filled_contours(gJs, gKs, mml,
                                                 colorbar_label=r"$\textrm{MML}$",
                                                 **plot_filled_contours_kwds)
        savefig(fig_mml, f"size-{size}-gridsearch-mml")

    except:
        None
    

    # Fucking save everything.
    with open(output_path, "wb") as fp:
        pickle.dump((Js, Ks, gJs, gKs, converged, meta, X_H, label_names), fp)


    # Compare model with previously saved model.
    model = meta["best_models"][config["adopted_metric"]]

    saved_models[size] = model

    # Plot clustering in data space.
    component_cmap = mpl_utils.discrete_cmap(7, base_cmap="Spectral_r")

    fig, axes = plt.subplots(5, 3, figsize=(7.1, 9.0))
    axes = np.atleast_1d(axes).flatten()


    x = X_H.T[label_names.index("fe_h")]
    c = np.argmax(model.tau_, axis=1)

    K = model.n_components


    y_idx = 0
    for i, ax in enumerate(axes):
        if label_names[i] == "fe_h":
            y_idx += 1

        y = X_H.T[y_idx] - x

        ax.scatter(x, y, c=[component_cmap(_) for _ in c], s=1, edgecolor="none", rasterized=True)

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

    fig.tight_layout()

    savefig(fig, f"size-{size}-data-space")




# Use the same stars as some hash for a different experiment.

# Then add in XYZ random stars.