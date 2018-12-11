
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
from time import time

sys.path.insert(0, "../../")

from mcfa import (mcfa, mpl_utils, utils)

matplotlib.style.use(mpl_utils.mpl_style)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Should all go into exp1.yaml

data_kwds = dict(n_samples=100_000,
                 n_features=15,
                 n_components=10,
                 n_latent_factors=3,
                 random_seed=42)

mcfa_kwds = dict(tol=1e-5, 
                 max_iter=1_000,
                 init_factors="svd",
                 init_components="kmeans++",
                 random_seed=42)

gridsearch_max_latent_factors = 2 * data_kwds["n_latent_factors"]
gridsearch_max_components = 2 * data_kwds["n_components"]

# Done with vars
"""
Y, truth = utils.generate_data(**data_kwds)
"""

def generate_data(n_samples=20, n_features=5, n_latent_factors=3, n_components=2,
                  omega_scale=1, noise_scale=1, random_seed=0):

    rng = np.random.RandomState(random_seed)

    #A = rng.randn(n_features, n_latent_factors)
    A = stats.ortho_group.rvs(n_features, random_state=rng)[:, :n_latent_factors]
    AL = linalg.cholesky(A.T @ A)
    A = A @ linalg.solve(AL, np.eye(n_latent_factors))

    # latent variables
    pvals = np.ones(n_components) / n_components
    R = np.argmax(rng.multinomial(1, pvals, size=n_samples), axis=1)
    pi = np.array([np.sum(R == i) for i in range(n_components)])/n_samples

    xi = rng.randn(n_latent_factors, n_components)
    omega = np.zeros((n_latent_factors, n_latent_factors, n_components))
    for i in range(n_components):
        omega[(*np.diag_indices(n_latent_factors), i)] = \
            rng.gamma(1, scale=omega_scale, size=n_latent_factors)**2

    scores = np.empty((n_samples, n_latent_factors))
    for i in range(n_components):
        match = (R == i)
        scores[match] = rng.multivariate_normal(xi.T[i], omega.T[i], 
                                                size=sum(match))

    psi = rng.gamma(1, scale=noise_scale, size=n_features)

    noise = np.sqrt(psi) * rng.randn(n_samples, n_features)

    X = scores @ A.T + noise

    truth = dict(A=A, pi=pi, xi=xi, omega=omega, psi=psi,
                 noise=noise, R=R, scores=scores)

    return (X, truth)

Y, truth = generate_data(**data_kwds)



# Fit with true number of latent factors and components.
model = mcfa.MCFA(n_components=data_kwds["n_components"],
                  n_latent_factors=data_kwds["n_latent_factors"],
                  **mcfa_kwds)
tick = time()
model.fit(Y)
tock = time()

print(f"Model took {tock - tick:.1f} seconds")

# Plot the log-likelihood with increasing iterations.
fig_iterations, ax = plt.subplots()

ll = model.log_likelihoods_
iterations = 1 + np.arange(len(ll))
ax.plot(iterations, ll, "-", lw=2, drawstyle="steps-mid")
ax.set_xlabel(r"$\textrm{iteration}$")
ax.set_ylabel(r"$\log{\mathcal{L}}$")
ax.set_xlim(0, iterations[-1] + 1)
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))
xt = ax.get_xticks()
ax.set_xticks(xt + np.hstack([1, np.zeros(xt.size - 1)]))
ax.set_xlim(0, iterations[-1] + 1)
fig_iterations.tight_layout()


# Plot the data, with samples
Y_drawn = model.sample(data_kwds["n_samples"])
fig_data = mpl_utils.corner_scatter(Y, 
                                    c="#000000", s=1, alpha=0.5, figsize=(8, 8))
mpl_utils.corner_scatter(Y_drawn,
                         c="tab:blue", s=1, alpha=0.25, zorder=10, fig=fig_data)

fig_data.tight_layout()
fig_data.subplots_adjust(hspace=0, wspace=0)



# Plot the true latent factors w.r.t. the estimated ones, after rotation.
A_true = truth["A"]
A_est = model.theta_[model.parameter_names.index("A")]

R = utils.find_rotation_matrix(A_true, A_est, n_inits=100)
A_est_rot = A_est @ R

D, J = A_true.shape
xi = 1 + np.arange(D)

fig_factor_loads, ax = plt.subplots()

for j in range(J):
    ax.plot(xi, A_est.T[j], ":", lw=1, c=colors[j])
    ax.plot(xi, A_est_rot.T[j], "-", lw=1, c=colors[j])
    ax.plot(xi, A_true.T[j], "-", lw=2, c=colors[j])

ax.set_xticks(xi)
ax.set_xlabel(r"$\textrm{dimension}$")
ax.set_ylabel(r"$\mathbf{L}$")

ylim = np.ceil(10 * np.abs(ax.get_ylim()).max()) / 10
ax.plot([0, D + 1], [0, 0], ":", c="#000000", zorder=-1, lw=0.5)
ax.set_xlim(0.5, D + 0.5)

ax.set_ylim(-ylim, +ylim)
ax.set_yticks([-ylim, 0, ylim])

fig_factor_loads.tight_layout()


# Do a grid search.
Js = np.arange(1, 1 + gridsearch_max_latent_factors)
Ks = np.arange(1, 1 + gridsearch_max_components)

Jm, Km = np.meshgrid(Js, Ks)

ll = np.nan * np.ones_like(Jm)
bic = np.nan * np.ones_like(Jm)
pseudo_bic = np.nan * np.ones_like(Jm)
pseudo_bic_kwds = dict(omega=1, gamma=0.1)

converged = np.zeros(Jm.shape, dtype=bool)

for j, J in enumerate(Js):
    for k, K in enumerate(Ks):

        print(f"At J = {J}, K = {K}")

        model = mcfa.MCFA(n_latent_factors=J, n_components=K, **mcfa_kwds)

        try:
            model.fit(Y)

        except:
            continue

        else:
            ll[k, j] = model.log_likelihood_
            bic[k, j] = model.bic(Y)
            pseudo_bic[k, j] = model.pseudo_bic(Y, **pseudo_bic_kwds)
            converged[k, j] = True

idx = np.nanargmin(bic)
jm_b, km_b = Js[idx % bic.shape[1]], Ks[int(idx / bic.shape[1])]

idx = np.nanargmin(pseudo_bic)
jm_pb, km_pb = Js[idx % pseudo_bic.shape[1]], Ks[int(idx / pseudo_bic.shape[1])]

J_true, K_true = (data_kwds["n_latent_factors"], data_kwds["n_components"])

print(f"BIC is lowest at J = {jm_b} and K = {km_b}")
print(f"pseudo-BIC is lowest at J = {jm_pb} and K = {km_pb}")
print(f"True values are  J = {J_true} and K = {K_true}")



kwds = dict(converged=converged, marker_function=np.nanargmin)
fig_ll = mpl_utils.plot_filled_contours(Jm, Km, -ll,
                                        colorbar_label=r"$-\log\mathcal{L}$", 
                                        **kwds)

fig_bic = mpl_utils.plot_filled_contours(Jm, Km, bic,
                                         colorbar_label=r"$\textrm{BIC}$", 
                                         **kwds)

fig_pseudo_bic = mpl_utils.plot_filled_contours(Jm, Km, pseudo_bic,
                                                colorbar_label=r"$\textrm{pseudo-BIC}$", 
                                                **kwds)

