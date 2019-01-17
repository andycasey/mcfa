
import sys
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from time import time

sys.path.insert(0, "../../")

from mcfa import (mcfa, mpl_utils, utils)

matplotlib.style.use(mpl_utils.mpl_style)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


n_features = 15
n_components = 20
n_latent_factors = 5
n_samples = 1000

omega_scale = 1
noise_scale = 1
random_seed = 123

data_kwds = dict(n_features=n_features,
                 n_components=n_components,
                 n_latent_factors=n_latent_factors,
                 n_samples=n_samples,
                 omega_scale=omega_scale,
                 noise_scale=noise_scale,
                 random_seed=random_seed)


def savefig(fig, suffix):
    prefix = os.path.basename(__file__)[:-3]
    here = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(here, f"{prefix}-{suffix}")
    fig.savefig(f"{filename}.pdf", dpi=300)
    fig.savefig(f"{filename}.png", dpi=150)
    print(f"Created figures {filename}.png and {filename}.pdf")


mcfa_kwds = dict(tol=1e-8, 
                 max_iter=1_000,
                 init_factors="random",
                 init_components="random",
                 random_seed=123)


def generate_data(n_samples, n_features, n_latent_factors, n_components,
                  omega_scale, noise_scale, random_seed=0):

    rng = np.random.RandomState(random_seed)

    A = rng.randn(n_features, n_latent_factors)

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


def new_generate_data(n_samples, n_features, n_latent_factors, n_components,
                      omega_scale, noise_scale, random_seed=0):

    from scipy.stats import special_ortho_group
    from scipy import linalg

    rng = np.random.RandomState(random_seed)

    A = special_ortho_group.rvs(n_features, random_state=rng)
    A = A[:, :n_latent_factors]
    AL = linalg.cholesky(A.T @ A)
    A = A @ linalg.solve(AL, np.eye(n_latent_factors))

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


Y, truth = new_generate_data(**data_kwds)
truth_packed = (truth["pi"], truth["A"], truth["xi"], truth["omega"], truth["psi"])

fig_data2 = mpl_utils.corner_scatter(Y, 
                                    c=truth["R"], cmap="Spectral",
                                    s=1, alpha=0.5, figsize=(8, 8),
                                    label_names=[r"$\mathbf{{Y}}_{{{0}}}$".format(i) for i in range(n_features)])
fig_data2.tight_layout()
fig_data2.subplots_adjust(hspace=0, wspace=0)
savefig(fig_data2, "data-colour")


gridsearch_max_latent_factors = 2 * data_kwds["n_latent_factors"]
gridsearch_max_components = 2 * data_kwds["n_components"]


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
savefig(fig_iterations, "ll-iterations")


# Plot the data, with samples
Y_drawn = model.sample(data_kwds["n_samples"])
fig_data = mpl_utils.corner_scatter(Y, 
                                    c="#000000", s=1, alpha=0.5, figsize=(8, 8))
mpl_utils.corner_scatter(Y_drawn,
                         c="tab:blue", s=1, alpha=0.25, zorder=10, fig=fig_data)

fig_data.tight_layout()
fig_data.subplots_adjust(hspace=0, wspace=0)
savefig(fig_data, "data")


# Plot the latent space.


cmap = mpl_utils.discrete_cmap(n_components, "Spectral")

label_names = [f"$\\mathbf{{S}}_{{{i}}}$" for i in range(n_latent_factors)]
fig_latent = mpl_utils.plot_latent_space(model, Y, cmap=cmap,
                                         label_names=label_names)
for ax in fig_latent.axes:
    if ax.get_visible():
        if ax.is_last_row():
            ax.xaxis.set_major_locator(MaxNLocator(3))
        if ax.is_first_col():
            ax.yaxis.set_major_locator(MaxNLocator(3))

        xlim = np.max(np.abs(ax.get_xlim()))
        ylim = np.max(np.abs(ax.get_ylim()))
        ax.set_xlim(-xlim, +xlim)
        ax.set_ylim(-ylim, +ylim)
fig_latent.tight_layout()
fig_latent.subplots_adjust(hspace=0, wspace=0)
savefig(fig_latent, "latent")



# Plot the true latent factors w.r.t. the estimated ones, after rotation.
A_true = truth["A"]
A_est = model.theta_[model.parameter_names.index("A")]

R, *_ = utils.find_rotation_matrix(A_true, A_est, n_inits=100)
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
savefig(fig_factor_loads, "factor_loads")


# Take model with true number of components and latent factors.

# Perform rotation.

# Compare factor loads to true values.

fig, ax = plt.subplots()
ax.scatter(A_true.flatten(), A_est_rot.flatten(), s=1)


# Compare factor scores to true values.
S = model.factor_scores(Y)[1]

fig, ax = plt.subplots()
ax.scatter(truth["scores"].flatten(), (S @ R).flatten(), s=1)

lims = np.hstack([ax.get_xlim(), ax.get_ylim()])
lims = (np.min(lims), np.max(lims))
ax.set_xlim(lims)
ax.set_ylim(lims)


# Compare specific scatter to true values.



# Do a grid search.
Js = np.arange(1, 1 + gridsearch_max_latent_factors)
Ks = np.arange(1, 1 + gridsearch_max_components)

Jm, Km = np.meshgrid(Js, Ks)

ll = np.nan * np.ones_like(Jm)
bic = np.nan * np.ones_like(Jm)

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
            converged[k, j] = True

            idx = np.nanargmin(bic)
            jm_b, km_b = Js[idx % bic.shape[1]], Ks[int(idx / bic.shape[1])]

            print(f"Lowest BIC so far is at J = {jm_b} and K = {km_b}")


idx = np.nanargmin(bic)
jm_b, km_b = Js[idx % bic.shape[1]], Ks[int(idx / bic.shape[1])]

J_true, K_true = (data_kwds["n_latent_factors"], data_kwds["n_components"])

print(f"BIC is lowest at J = {jm_b} and K = {km_b}")
print(f"True values are  J = {J_true} and K = {K_true}")


kwds = dict(converged=converged, 
            marker_function=np.nanargmin, 
            N=1000, 
            cmap="Spectral_r",
            truth=(J_true, K_true))

fig_ll = plot_filled_contours(Jm, Km, -ll,
                                        colorbar_label=r"$-\log\mathcal{L}$", 
                                        **kwds)


fig_bic = plot_filled_contours(Jm, Km, bic,
                                         colorbar_label=r"$\textrm{BIC}$", 
                                         **kwds)


savefig(fig_ll, "gridsearch-ll-contours")
savefig(fig_bic, "gridsearch-bic-contours")



