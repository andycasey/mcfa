




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


experiment_hash = "8761b"


matplotlib.style.use(mpl_utils.mpl_style)

here = os.path.dirname(os.path.realpath(__file__))

with open(f"{experiment_hash}.yml") as fp:
    config = yaml.load(fp)

print(f"Config: {config}")

np.random.seed(config["random_seed"])

prefix = os.path.basename(__file__)[:-3]
unique_hash = "none"

def savefig(fig, suffix):
    here = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(here, f"{prefix}-{unique_hash}-{suffix}")
    fig.savefig(f"{filename}.png", dpi=150)
    fig.savefig(f"{filename}.pdf", dpi=300)


# Get the same original data from the experiment hash:

import os

N_elements = 20
use_galah_flags = config["use_galah_flags"]

mcfa_kwds = dict()
mcfa_kwds.update(config["mcfa_kwds"])

elements = config["exp3"]["elements"]
if config["exp3"]["ignore_elements"] is not None:
    elements = [el for el in elements if el not in config["exp3"]["ignore_elements"]]

print(elements)


mask = galah.get_abundance_mask(elements, use_galah_flags=use_galah_flags)


galah_cuts = config["exp3"]["galah_cuts"]
if galah_cuts is not None:
    print(f"Applying cuts: {galah_cuts}")
    for k, (lower, upper) in galah_cuts.items():
        mask *= (upper >= galah.data[k]) * (galah.data[k] >= lower)


# This will be the default mask. 
possible_additions_mask = np.zeros(len(mask), dtype=bool)
# Start off by selecting everything that isn't in mask
possible_additions_mask = ~mask

galah_cuts = config["exp3"]["galah_cuts"]
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
J = 5
K = 3
gs_options = config["exp3"]["gridsearch"]
N_inits = gs_options["n_inits"]

delta_J = 2
delta_K = 2



OVERWRITE = False
sizes = (100, 1000, 10000)# sum(possible_additions_mask))

saved_models = dict()

for size in sizes:

    if size <= 1000:
        Js = np.arange(J - delta_J, J + delta_J + 1)
        Ks = np.arange(K - delta_K, K + delta_K + 1)

    else:
        Js = np.arange(J - delta_J, J + 5 * delta_J + 1)
        Ks = np.arange(K - delta_K, K + 5 * delta_K + 1)

    output_path = f"exp10-models-size-{size}.pkl"
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
        pickle.dump((Js, Ks, gJs, gKs, converged, meta), fp)


    # Compare model with previously saved model.
    saved_models[size] = meta["best_models"][config["adopted_metric"]]




# Use the same stars as some hash for a different experiment.

# Then add in XYZ random stars.