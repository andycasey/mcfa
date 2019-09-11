
# Experiment with missing data when the data are not missing at random.

import sys
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
import yaml

from time import time

sys.path.insert(0, "../../")

from mcfa import (mcfa, grid_search, mpl_utils, utils)

matplotlib.style.use(mpl_utils.mpl_style)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

here = os.path.dirname(__file__)



with open("config.yml") as fp:
    config = yaml.load(fp)

prefix = os.path.basename(__file__)[:-3]
this_config = config.get(prefix, dict())
data_kwds = this_config.get("data_kwds", dict())
mcfa_kwds = this_config.get("mcfa_kwds", dict())

D = data_kwds["n_features"]
K = data_kwds["n_components"]
J = data_kwds["n_latent_factors"]
N = data_kwds["n_samples"]

print(f"Data kwds: {data_kwds}")
print(f"MCFA kwds: {mcfa_kwds}")



uniform_missing_probabilities = False
fit_kwds = dict(n_inits=20)


Y, truth = utils.generate_data(**data_kwds)
truth_packed = (truth["pi"], truth["A"], truth["xi"], truth["omega"], truth["psi"])


def savefig(fig, suffix):
    prefix = os.path.basename(__file__)[:-3]
    here = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(here, "eval-figs", f"{prefix}-{suffix}")
    fig.savefig(f"{filename}.pdf", dpi=300)
    fig.savefig(f"{filename}.png", dpi=150)
    print(f"Created figures {filename}.png and {filename}.pdf")


if uniform_missing_probabilities:
    # Make uniform probabilities
    fractional_prob_missing = np.ones(data_kwds["n_features"])
    fractional_prob_missing /= np.sum(fractional_prob_missing)
    print("Doing uniform missing probabilities per dimension")

else:

    # Sort the missing data probabilities by the psi values.
    fractional_prob_missing = np.random.uniform(0, 1, size=data_kwds["n_features"])
    fractional_prob_missing /= np.sum(fractional_prob_missing)


for missing_data_fraction in (0, 0.2, 0.4):

    # Throw away data, randomly within dimensions.
    # (There must be a better way to do this,..)
    T = int(missing_data_fraction * Y.size) # Total number of entries to be missing.

    # Do we need to down-weight the largest values?
    # This can happen if the number of entries to throw away is larger than the number of samples.
    while True:
        k = np.where((fractional_prob_missing * T).astype(int) > data_kwds["n_samples"])[0]
        if 1 > k.size:
            break

        print("Down-sampling missing data fractions...")
        fractional_prob_missing[k] *= 0.9
        fractional_prob_missing /= np.sum(fractional_prob_missing)

    is_missing = np.zeros(Y.shape, dtype=bool)

    for j, p in enumerate(fractional_prob_missing):

        M = int(p * T)
        i_indices = np.random.choice(data_kwds["n_samples"], M, replace=False)
        j_indices = np.ones(i_indices.size, dtype=int) * j

        is_missing[i_indices, j_indices] = True

    
    X = np.copy(Y)
    X[is_missing] = np.nan

    # Fit with true number of latent factors and components.
    model = mcfa.MCFA(n_components=data_kwds["n_components"],
                      n_latent_factors=data_kwds["n_latent_factors"],
                      **mcfa_kwds)
    tick = time()
    model.fit(X, **fit_kwds)
    tock = time()

    model.message_length(X)

    print(f"Model took {tock - tick:.1f} seconds")

    A_true = truth["A"]
    A_est = model.theta_[model.parameter_names.index("A")]

    # Get exact transformation.
    R = utils.exact_rotation_matrix(A_true, A_est)

    # Now make it a valid rotation matrix.
    try:
        L = linalg.cholesky(R.T @ R)
        R = R @ linalg.solve(L, np.eye(data_kwds["n_latent_factors"]))

    except:
        R = utils.find_rotation_matrix(A_true, A_est, full_output=False)

    model.rotate(R)


    scatter_kwds = dict(s=1, rasterized=True, c="#000000")

    fig = plt.figure(figsize=(7.5, 3.09))


    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 4], width_ratios=[1, 1, 1])

    A_est = model.theta_[model.parameter_names.index("A")]

    xs = [
        A_true.flatten(),
        truth["scores"].flatten(),
        truth["psi"].flatten()
    ]

    ys = [
        A_est.flatten(),
        model.factor_scores(Y)[1].flatten(),
        model.theta_[-1]
    ]

    xlabels = [
        r"$\mathbf{L}_\textrm{true}$",
        r"$\mathbf{S}_\textrm{true}$",
        r"$\mathbf{D}_\textrm{true}$"
    ]

    ylabels = [
        r"$\mathbf{L}_\textrm{est}$",
        r"$\mathbf{S}_\textrm{est}$",
        r"$\mathbf{D}_\textrm{est}$"
    ]

    delta_labels = [
        r"$\Delta\mathbf{L}$",
        r"$\Delta\mathbf{S}$",
        r"$\Delta\mathbf{D}$"
    ]

    residual_ax_y_limits = [
        0.35,
        7,
        0.5
    ]

    common_ax_limits = [
        0.75,
        21,
        4
    ]
    use_common_limits = True

    idx = 0
    for i in range(3):
        ax_residual = fig.add_subplot(gs[idx])
        ax = fig.add_subplot(gs[idx +3])

        x, y = (xs[i], ys[i])

        ax.scatter(x, y, **scatter_kwds)
        ax_residual.scatter(x, y - x, **scatter_kwds)

        if use_common_limits:
            ylims = abs(residual_ax_y_limits[i])
            xlims = [-common_ax_limits[i], +common_ax_limits[i]]
            if i == 2:
                xlims[0] = 0

        else:
            xlims = np.max(np.abs(np.hstack([ax.get_xlim(), ax.get_ylim()])))
            if i == 2:
                xlims = (0, +xlims)
            else:
                xlims = (-xlims, +xlims)

            ylims = np.max(np.abs(ax_residual.get_ylim()))
            

        kwds = dict(c="#666666", linestyle=":", linewidth=0.5, zorder=-1)
        ax.plot([xlims[0], +xlims[1]], [xlims[0], +xlims[1]], "-", **kwds)
        ax_residual.plot([xlims[0], +xlims[1]], [0, 0], "-", **kwds)

        ax.set_xlim(xlims[0], +xlims[1])
        ax.set_ylim(xlims[0], +xlims[1])
        ax_residual.set_xlim(xlims[0], +xlims[1])
        ax_residual.set_ylim(-ylims, +ylims)

        ax_residual.yaxis.set_major_locator(MaxNLocator(3))
        ax_residual.xaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax_residual.set_xticks([])


        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])
        ax_residual.set_ylabel(delta_labels[i])

        #ax.set_aspect(1.0)
        #ax_residual.set_aspect(1)
        idx += 1

    # Plot the inflated specific variances.
    #scatter_kwds.update(c="tab:red")
    #ax.scatter(x, y/(1 - missing_data_fraction), **scatter_kwds)
    #ax_residual.scatter(x, y/(1 - missing_data_fraction) - x, **scatter_kwds)

    fig.tight_layout()

    #fig.savefig(os.path.join(here, f"exp-missing-data-not-at-random-inflated-{100*missing_data_fraction:.0f}percent.png"))
    savefig(fig, f"missing-data-{100*missing_data_fraction:.0f}")
    #fig.savefig(f"exp-missing-data-not-at-random-inflated-{100*missing_data_fraction:.0f}percent.pdf", dpi=300)
