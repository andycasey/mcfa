
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy import linalg

from matplotlib.ticker import MaxNLocator

sys.path.insert(0, "../../")
from mcfa import (mcfa, grid_search, mpl_utils, utils)

matplotlib.style.use(mpl_utils.mpl_style)


def savefig(fig, suffix):
    prefix = os.path.basename(__file__)[:-3]
    filename = f"{prefix}-{suffix}.png"
    fig.savefig(filename, dpi=150)
    

with open("../../catalogs/barklem.pkl", "rb") as fp:
    X_H, label_names, mask = pickle.load(fp)

# Do not include C
ignore_elements = ["c"]
element_mask = np.array([ln.split("_")[0] not in ignore_elements for ln in label_names])

X_H = X_H[:, element_mask]
label_names = list(np.array(label_names)[element_mask])

X = utils.whiten(X_H)


grouped_elements = [
    ["sr", "y", "ba", "eu"],
    ["ni", "co", "fe", "mn", "cr", "ti", "sc"],
    ["al", "ca", "mg"]
]


grouped_elements = [
    ["eu", "sr", "y", "ba"],
    ["al", "ca", "mg", "ni", "co", "fe", "mn", "cr", "ti", "sc"],
]

mcfa_kwds = dict(init_factors="random", init_components="random", tol=1e-5,
                 max_iter=10000, random_seed=8)


model = mcfa.MCFA(n_components=1, n_latent_factors=len(grouped_elements),
                  **mcfa_kwds)

model.fit(X)

# Rotate the latent factors to be something close to astrophysical.

A_est = model.theta_[model.parameter_names.index("A")]

A_astrophysical = np.zeros_like(A_est)
for i, tes in enumerate(grouped_elements):
    for j, te in enumerate(tes):
        idx = label_names.index("{0}_h".format(te.lower()))
        A_astrophysical[idx, i] = 1.0

#AL = linalg.cholesky(A_astrophysical.T @ A_astrophysical)
#A_astrophysical = A_astrophysical @ linalg.solve(AL, np.eye(model.n_latent_factors))

R, p_opt, cov, *_ = utils.find_rotation_matrix(A_astrophysical, A_est, 
                                               full_output=True)

if cov is None:
    cov = 5**2 * np.pi/180 * np.eye(p_opt.size)

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
    ax.plot(xi, A_est.T[j], ":", lw=2, c=colors[j])
    ax.plot(xi, A_astrophysical.T[j], ":", lw=1, c=colors[j])
    ax.plot(xi, A_est_rotation_percentiles[1, :, j], "-", lw=2, c=colors[j])
    ax.fill_between(xi, 
                    A_est_rotation_percentiles[0, :, j], 
                    A_est_rotation_percentiles[2, :, j],
                    facecolor=colors[j], alpha=0.3)

ax.set_xticks(xi)
ax.set_xticklabels([l.split("_")[0].title() for l in label_names])
ax.set_xlabel(r"$\textrm{element}$")
ax.set_ylabel(r"$\mathbf{L}$")

ylim = np.ceil(10 * np.abs(ax.get_ylim()).max()) / 10
ax.plot([0, D + 1], [0, 0], ":", c="#000000", zorder=-1, lw=0.5)
ax.set_xlim(0.5, D + 0.5)

ax.set_ylim(-ylim, +ylim)
ax.set_yticks([-ylim, 0, ylim])

fig_factor_loads.tight_layout()


# Run a grid search.
max_n_latent_factors = 8
max_n_components = 5

Js = 1 + np.arange(max_n_latent_factors)
Ks = 1 + np.arange(max_n_components)

Jg, Kg, converged, ll, bic, pseudo_bic = grid_search.grid_search(Js, Ks, X, 
                                                                 mcfa_kwds)

J_best_ll, K_best_ll = grid_search.best(Js, Ks, -ll)
J_best_bic, K_best_bic = grid_search.best(Js, Ks, bic)

print(f"Best log likelihood  at J = {J_best_ll} and K = {K_best_ll}")
print(f"Best BIC value found at J = {J_best_bic} and K = {K_best_bic}")

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


# Re-run model with best J, K.

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
    ["al", "ca", "mg", "ni", "co", "fe", "mn", "cr", "ti", "sc"],
    ["eu", "ba"],
    ["sr", "y"],
]


A_astrophysical = np.zeros_like(A_est)
for i, tes in enumerate(astrophysical_grouping):
    for j, te in enumerate(tes):
        idx = label_names.index("{0}_h".format(te.lower()))
        A_astrophysical[idx, i] = 1.0

AL = linalg.cholesky(A_astrophysical.T @ A_astrophysical)
A_astrophysical = A_astrophysical @ linalg.solve(AL, np.eye(model.n_latent_factors))


np.random.seed(42)

fig_factors_rotated = mpl_utils.plot_factor_loads(A_est, 
                                                  target_loads=A_astrophysical,
                                                  flip_loads=None,
                                                  n_rotation_inits=1000,
                                                  show_target_loads=False,
                                                  load_labels=[
                                                    r"$\textrm{CCSN and others?}$",
                                                    r"$\textrm{s-process?}$",
                                                    r"$\textrm{r-process?}$"
                                                  ],
                                                  xlabel=xlabel,
                                                  xticklabels=xticklabels,
                                                  legend_kwds=dict(loc="upper left",
                                                                   fontsize=14.0))
savefig(fig_factors_rotated, "latent-factors")

# PLot the clustering in latent space?

# Plot the clustering in data space?

# Plot the specific variances.
fig_scatter = mpl_utils.plot_specific_scatter(model, scales=np.std(X_H, axis=0),
                                              steps=True,
                                              xlabel=xlabel, 
                                              xticklabels=xticklabels,
                                              ylabel=r"$\textrm{specific scatter / dex}$")
fig_scatter.axes[0].set_yticks(np.arange(0, 0.30, 0.05))
savefig(fig_scatter, "specific-scatter")
