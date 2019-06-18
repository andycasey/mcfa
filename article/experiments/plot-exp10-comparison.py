
"""
Plot results from exp10
"""

import os
import pickle
import matplotlib.pyplot as plt

from mcfa import (mcfa, grid_search, mpl_utils, utils)

np.random.seed(0)


# Get original model.
with open("8761b-exp3-model.pkl", "rb") as fp:
    original_model = pickle.load(fp)


# Get various models from exp10 for comparison.
comparison_model_paths = [
    "exp10-models-size-100.pkl",
    "exp10-models-size-1000.pkl",
    #"exp10-models-size-10000.pkl",
]


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

# Fill up.
def _fill_factors(A, J_max):
    D, J = A.shape
    return np.hstack([A, np.ones((D, J_max - J)) * np.nan])


plot_factor_loads_kwds = dict(colors=colors, separate_axes=True)

fig = mpl_utils.plot_factor_loads(_fill_factors(A_original, J_max), 
                                  **plot_factor_loads_kwds)
plot_factor_loads_kwds.update(fig=fig)

plot_kwds = {
    "exp10-models-size-100.pkl": dict(linestyle=":"),
    "exp10-models-size-1000.pkl": dict(linestyle="-."),
    "exp10-models-size-10000.pkl": dict(linestyle="--"),
}




for i, comparison_model_path in enumerate(comparison_model_paths):

    with open(comparison_model_path, "rb") as fp:
        contents = pickle.load(fp)

    Js, Ks, gJs, gKs, converged, meta = contents

    # Get best model.
    J_best_mml, K_best_mml = grid_search.best(Js, Ks, meta["message_length"])
    comparison_model = meta["best_models"]["mml"]

    # Perform rotation.
    A_est = comparison_model.theta_[comparison_model.parameter_names.index("A")]
    if A_est.shape == A_original.shape:

        R, p_opt, cov, *_ = utils.find_rotation_matrix(A_original, A_est, 
                                                       full_output=True)

        R_opt = utils.exact_rotation_matrix(A_original, A_est, 
                                            p0=np.random.uniform(-np.pi, np.pi, comparison_model.n_latent_factors**2))

        AL = linalg.cholesky(R_opt.T @ R_opt)
        R_opt2 = R_opt @ linalg.solve(AL, np.eye(comparison_model.n_latent_factors))

        chi1 = np.sum(np.abs(A_est @ R - A_original))
        chi2 = np.sum(np.abs(A_est @ R_opt2 - A_original))

        R = R_opt2 if chi2 < chi1 else R

        # Now make it a valid rotation matrix.
        comparison_model.rotate(R)

    else:
        # Now it gets Hard(tm)

        raise a


    # Now plot the comparison latent factors.
    A_comparison = comparison_model.theta_[comparison_model.parameter_names.index("A")]

    kwds = plot_factor_loads_kwds.copy()
    kwds.update(plot_kwds.get(comparison_model_path, {}))

    # Fill up?
    fig = mpl_utils.plot_factor_loads(A_comparison, **kwds)



raise a
for ax in fig.axes:
    ax.set_ylim(0, ax.get_ylim()[1])

