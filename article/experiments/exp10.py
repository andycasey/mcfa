




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


sizes = (100, 1000, 10000, sum(possible_additions_mask))

saved_models = dict()

for size in sizes:
    print(f"Doing size = {size}")

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


    _, __, converged, meta = grid_search.grid_search([J], [K], X, N_inits=N_inits, mcfa_kwds=mcfa_kwds)

    model = meta["best_models"][config["adopted_metric"]]

    # Compare model with previously saved model.
    saved_models[size] = model



# Use the same stars as some hash for a different experiment.

# Then add in XYZ random stars.