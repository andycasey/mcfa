
"""
Plot results from exp10
"""

import os
import pickle
import sys
import matplotlib.pyplot as plt
import yaml
import numpy as np
from scipy import linalg

sys.path.insert(0, "../../")

from mcfa import (mcfa, grid_search, mpl_utils, utils)

np.random.seed(0)


# Get original model.
with open("89dab-exp3-model.pkl", "rb") as fp:
    original_model = pickle.load(fp)


# Get various models from exp10 for comparison.
comparison_model_paths = [
    "exp10-models-size-100.pkl",
    "exp10-models-size-1000.pkl",
    "exp10-models-size-10000.pkl",
]

# 89dab is for all intents and purposes identical to 8761b
with open("8761b.yml") as fp:
    config = yaml.load(fp)



label_names = [
    "na_h",
    "al_h",
    "si_h",
    "k_h",
    "ca_h",
    "sc_h",
    "ti_h",
    "v_h",
    "mn_h",
    "fe_h",
    "ni_h",
    "cu_h",
    "zn_h",
    "y_h",
    "ba_h",
    "la_h",
    "eu_h"
]

def _get_model(path):
    with open(path, "rb") as fp:
        contents = pickle.load(fp)

    Js, Ks, gJs, gKs, converged, meta = contents

    # Get best model.
    J_best_mml, K_best_mml = grid_search.best(Js, Ks, meta["message_length"])
    model = meta["best_models"]["mml"]

    return model


# We need to know the maximum number of latent factors that we'll need to plot.
n_latent_factors = []
for comparison_model_path in comparison_model_paths:
    if not os.path.exists(comparison_model_path): continue

    model = _get_model(comparison_model_path)
    n_latent_factors.append(model.n_latent_factors)

# TODO: Set this from config, or something crazy to keep colours consistent?
J_max = np.max(n_latent_factors)

cmap = mpl_utils.discrete_cmap(J_max, base_cmap="Spectral_r")
colors = [cmap(j) for j in range(J_max)][::-1]


# Create comparison figure.
A_original = original_model.theta_[original_model.parameter_names.index("A")]

D = 17

J_ast = len(config["grouped_elements"])
A_astrophysical = np.zeros((D, J_ast))#np.random.normal(0, 0.1, size=A_est.shape)
for i, tes in enumerate(config["grouped_elements"][:J_ast]):
    for j, te in enumerate(tes):
        try:
            idx = label_names.index("{0}_h".format(te.lower()))

        except ValueError:
            print(f"Skipping {te}")

        else:
            count = sum([(te in foo) for foo in config["grouped_elements"][:J_ast]])
            A_astrophysical[idx, i] = 1.0/count

A_astrophysical /= np.clip(np.sqrt(np.sum(A_astrophysical, axis=0)), 1, np.inf)

# Un-assigned columns
for column_index in np.where(np.all(A_astrophysical == 0, axis=0))[0]:
    print(f"Warning: unassigned column index: {column_index}")
    A_astrophysical[:, column_index] = np.random.normal(0, 1e-2, size=D)

if config["correct_A_astrophysical"]:
    AL = linalg.cholesky(A_astrophysical.T @ A_astrophysical)
    A_astrophysical = A_astrophysical @ linalg.solve(AL, np.eye(J_ast))



# Fill up.
def _fill_factors(A, J_max):
    D, J = A.shape
    return np.hstack([A, np.ones((D, J_max - J)) * np.nan])



ABS_ONLY = True
if ABS_ONLY:
    viz_load = lambda A: np.abs(A)
else:
    viz_load = lambda A: A

plot_factor_loads_kwds = dict(colors=colors, separate_axes=True, lw=1, xlabel=r"$\textrm{element}$", alpha=0.25,
                              xticklabels=[r"$\textrm{{{0}}}$".format(ea.split("_")[0].title()) for ea in label_names])

fig = mpl_utils.plot_factor_loads(viz_load(_fill_factors(A_original, J_max)), 
                                  figsize=(6.75, 9.50),
                                  **plot_factor_loads_kwds)
plot_factor_loads_kwds.update(fig=fig)

plot_kwds = {
    "exp10-models-size-100.pkl": dict(linestyle=":", lw=2, alpha=0.5),
    "exp10-models-size-1000.pkl": dict(linestyle="-.", lw=3, alpha=0.75),
    "exp10-models-size-10000.pkl": dict(linestyle="-", lw=4, alpha=1),
}


#A_target = A_astrophysical
A_target = A_original
#A_target[:-2, 0] = 0
#A_target[-2:, 0] = 1


for i, comparison_model_path in enumerate(comparison_model_paths):

    with open(comparison_model_path, "rb") as fp:
        contents = pickle.load(fp)

    Js, Ks, gJs, gKs, converged, meta = contents

    # Get best model.
    J_best_mml, K_best_mml = grid_search.best(Js, Ks, meta["message_length"])
    comparison_model = meta["best_models"]["mml"]

    # Perform rotation.
    A_est = comparison_model.theta_[comparison_model.parameter_names.index("A")]
    if A_est.shape == A_target.shape:

        R, p_opt, cov, *_ = utils.find_rotation_matrix(A_target, A_est, 
                                                       full_output=True)

        R_opt = utils.exact_rotation_matrix(A_target, A_est, 
                                            p0=np.random.uniform(-np.pi, np.pi, comparison_model.n_latent_factors**2))

        AL = linalg.cholesky(R_opt.T @ R_opt)
        R_opt2 = R_opt @ linalg.solve(AL, np.eye(comparison_model.n_latent_factors))

        chi1 = np.sum(np.abs(A_est @ R - A_target))
        chi2 = np.sum(np.abs(A_est @ R_opt2 - A_target))

        R = R_opt2 if chi2 < chi1 else R

        # Now make it a valid rotation matrix.
        comparison_model.rotate(R)

    else:
        # Now it gets Hard(tm)
        A_target_copy = _fill_factors(A_target, A_est.shape[1])
        M = np.sum(~np.isfinite(A_target_copy))

        A_target_copy[~np.isfinite(A_target_copy)] = np.random.normal(0, 1e-2, size=M)


        R, p_opt, cov, *_ = utils.find_rotation_matrix(A_target_copy, A_est, 
                                                       full_output=True)

        R_opt = utils.exact_rotation_matrix(A_target_copy, A_est, 
                                            p0=np.random.uniform(-np.pi, np.pi, comparison_model.n_latent_factors**2))

        AL = linalg.cholesky(R_opt.T @ R_opt)
        R_opt2 = R_opt @ linalg.solve(AL, np.eye(comparison_model.n_latent_factors))

        chi1 = np.sum(np.abs(A_est @ R - A_target_copy))
        chi2 = np.sum(np.abs(A_est @ R_opt2 - A_target_copy))

        R = R_opt2 if chi2 < chi1 else R

        # Now make it a valid rotation matrix.
        comparison_model.rotate(R)



    # Now plot the comparison latent factors.
    A_comparison = comparison_model.theta_[comparison_model.parameter_names.index("A")]

    kwds = plot_factor_loads_kwds.copy()
    kwds.update(plot_kwds.get(comparison_model_path, {}))

    # Fill up?
    fig = mpl_utils.plot_factor_loads(viz_load(A_comparison), **kwds)


if ABS_ONLY:
    for i, ax in enumerate(fig.axes):
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_ylabel(r"$|\mathbf{{L}}_{{{0}}}|$".format(i + 1))


#fig.set_figwidth(6.75)
#fig.set_figheight(9.50)

fig.tight_layout()

fig.savefig("exp10-comparison.pdf", dpi=300)


raise a