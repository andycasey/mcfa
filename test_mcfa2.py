
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import stan_utils as stan
from mcfa import (mcfa, mpl_utils, utils)
from scipy import stats

matplotlib.style.use(mpl_utils.mpl_style)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


seed = 42

N = 50 # number of data points
D = 10   # data dimension
J = 3    # number of latent factors
K = 1    # number of components
lkj_eta = 2.0


data_kwds = dict(n_samples=N, n_features=D, n_latent_factors=J,
                 n_components=K, lkj_eta=lkj_eta, psi_scale=1,
                 latent_scale=1, random_seed=seed)

strict_op_kwds = dict(init_alpha=1, tol_obj=1e-16, tol_rel_grad=1e-16, 
                      tol_rel_obj=1e-16, seed=seed, iter=100000)


op_kwds = dict()
op_kwds.update(strict_op_kwds)

sampling_kwds = dict(chains=2, iter=1000)


def LKJPriorSampling(n, eta, size=None):

    beta = eta - 1. + n/2.
    r12 = 2. * stats.beta.rvs(a=beta, b=beta, size=size) - 1.
    P = np.eye(n)[:, :, np.newaxis] * np.ones(size)
    P[0, 1] = r12
    P[1, 1] = np.sqrt(1. - r12**2)
    for mp1 in range(2, n):
        beta -= 0.5
        y = stats.beta.rvs(a=mp1 / 2., b=beta, size=size)
        z = stats.norm.rvs(loc=0, scale=1, size=(mp1, ) + size)
        z = z / np.sqrt(np.einsum('ij,ij->j', z, z))
        P[0:mp1, mp1] = np.sqrt(y) * z
        P[mp1, mp1] = np.sqrt(1. - y)
    C = np.einsum('ji...,jk...->...ik', P, P)
    triu_idx = np.triu_indices(n, k=1)
    return C[..., triu_idx[0], triu_idx[1]]


def LKJCorrelationMatrix(D, eta):
    M = np.eye(D)
    M[np.triu_indices(D, 1)] = LKJPriorSampling(D, eta, size=(1, ))
    return M


def generate_data(n_samples, n_features, n_latent_factors, n_components,
                  lkj_eta=2, psi_scale=1, latent_scale=1, random_seed=0):

    rng = np.random.RandomState(random_seed)

    # Generate orthonormal factors.
    sigma_L = np.abs(rng.normal(0, latent_scale))
    choose = lambda x, y: int(np.math.factorial(x) \
                        / (np.math.factorial(y) * np.math.factorial(x - y)))

    M = n_latent_factors * (n_features - n_latent_factors) \
      + choose(n_latent_factors, 2)

    beta_lower_triangular = rng.normal(0, sigma_L, size=M)
    beta_diag = np.abs(rng.normal(0, latent_scale, size=n_latent_factors))
    beta_diag = np.sort(beta_diag)

    A = np.zeros((n_features, n_latent_factors), dtype=float)
    A[np.tril_indices(n_features, -1, n_latent_factors)] = beta_lower_triangular
    A[np.diag_indices(n_latent_factors)] = beta_diag[::-1]


    # latent variables
    pvals = np.ones(n_components) / n_components
    R = np.argmax(rng.multinomial(1, pvals, size=n_samples), axis=1)
    pi = np.array([np.sum(R == i) for i in range(n_components)])/n_samples

    xi = rng.randn(n_latent_factors, n_components)
    """
    omega = np.zeros((n_latent_factors, n_latent_factors, n_components))
    for i in range(n_components):
        #omega[(*np.diag_indices(n_latent_factors), i)] = \
        #    rng.gamma(1, scale=omega_scale, size=n_latent_factors)**2
        omega[(*np.diag_indices(n_latent_factors), i)] \
            = np.abs(rng.normal(0, 1, size=n_latent_factors))
    """

    # Generate Omega from LKJCovariance matrices.
    omega = np.zeros((n_latent_factors, n_latent_factors, n_components))
    omega_corr = np.zeros_like(omega)
    omega_diag = np.zeros((n_components, n_latent_factors))
    for i in range(n_components):

        rho = LKJCorrelationMatrix(J, lkj_eta)
        omega_corr[:, :, i] = rho
        omega_diag[i] = np.sqrt(np.abs(rng.normal(0, 1, size=(1, J))))
        omega[:, :, i] = (rho @ rho.T) * (omega_diag[i].T @ omega_diag[i])


    scores = np.empty((n_samples, n_latent_factors))
    for i in range(n_components):
        match = (R == i)
        scores[match] = rng.multivariate_normal(xi.T[i], omega.T[i], 
                                                size=sum(match))

    #psi = rng.gamma(1, scale=psi_scale, size=n_features)
    psi = np.abs(rng.normal(0, psi_scale, n_features))

    noise = np.sqrt(psi) * rng.randn(n_samples, n_features)

    X = (A @ scores.T + noise.T).T

    truth = dict(A=A, pi=pi, xi=xi, omega=omega, psi=psi,
                 noise=noise, R=R, scores=scores,
                 beta_diag=beta_diag, sigma_L=sigma_L,
                 beta_lower_triangular=beta_lower_triangular,
                 omega_corr=omega_corr, omega_diag=omega_diag)

    return (X, truth)

y, truth = generate_data(**data_kwds)



model = stan.load_model("mcfa2.stan")

data_dict = dict(N=N, D=D, J=J, K=K, y=y, Omega_eta=lkj_eta)

init_dict = {
    "xi": truth["xi"].T,
    "lambda": np.atleast_1d(truth["pi"]),
    "Omega_corr": truth["omega_corr"].T,
    "Omega_diag": truth["omega_diag"],
    "beta_diag": truth["beta_diag"],
    "beta_lower_triangular": truth["beta_lower_triangular"],
    "psi": truth["psi"],
    "sigma_L": truth["sigma_L"]
}

p_opt = model.optimizing(data=data_dict, init=init_dict, **op_kwds)

p_opt["lambda"] = np.atleast_1d(p_opt["lambda"])

"""
# Compare psi first.
fig, ax = plt.subplots()
ax.plot(truth["psi"], "-", c="tab:blue", lw=2)
ax.plot(p_opt["psi"], "-", c="#000000")
ax.set_ylabel(r"$\psi$")
ax.set_xlabel(r"$D$")

fig.tight_layout()


# Compare loads

xi = np.arange(D)
L_true = truth["A"]
L_opt = p_opt["L"]


fig, ax = plt.subplots()
nums = np.nanmedian(L_true / L_opt, axis=0)
for j in range(J):
    ax.plot(xi, L_true.T[j], "-", c=colors[j], lw=2)
    ax.plot(xi, nums[j] * L_opt.T[j], ":", c=colors[j], lw=1)


print(f"Nums: {nums}")
# Compare means.
fig, ax = plt.subplots()
ax.scatter(truth["xi"].flatten(), p_opt["xi"].flatten())

"""


p_samples = model.sampling(**stan.sampling_kwds(data=data_dict, init=p_opt, 
                                                **sampling_kwds))


def plot_samples(ax, samples, key, 
                 fill_percentiles=[16, 84], line_percentile=50,
                 color="tab:blue", alpha=0.3, axis=0):

    if fill_percentiles is not None:
        y_lower, y_upper = np.percentile(samples[key], fill_percentiles, axis=axis)
        x = np.arange(y_lower.size)
        ax.fill_between(x, y_lower, y_upper, facecolor=color, alpha=alpha)

    if line_percentile is not None:
        y = np.percentile(samples[key], line_percentile, axis=axis)
        x = np.arange(y.size)
        ax.plot(x, y, "-", c=color, lw=2, zorder=1)

    return None


fig, ax = plt.subplots()
ax.plot(truth["psi"], "-", c="#000000", lw=2, zorder=10)
plot_samples(ax, p_samples, "psi")
ax.set_xlabel(r"$\textrm{Dimension } D$")
ax.set_ylabel(r"$\psi$")


fig, ax = plt.subplots()
ax.hist(p_samples["sigma_L"], bins=50, facecolor="tab:blue", alpha=0.3)
ylim = ax.get_ylim()
ax.plot([truth["sigma_L"], truth["sigma_L"]], ylim, "-",
         c="#000000", lw=2, zorder=10)
ax.set_ylim(ylim)
ax.set_xlabel(r"$L_\sigma$")


# OK, beta_diag.
fig, axes = plt.subplots(J)
for j, ax in enumerate(axes):
    ax.hist(p_samples["beta_diag"].T[j], bins=50, facecolor="tab:blue", alpha=0.3)
    ylim = ax.get_ylim()
    ax.plot([truth["beta_diag"][j], truth["beta_diag"][j]],
            ylim, "-", c="#000000", lw=2, zorder=10)

    ax.set_xlabel(r"$\\Beta_{0}$".format(j))


L_percentiles = np.percentile(p_samples["L"], [5, 50, 95], axis=0)

R = utils.rotation_matrix(L_percentiles[1], truth["A"])

A_true = truth["A"] @ R

xi = np.arange(D)
fig, ax = plt.subplots()
for j in range(J):
    ax.plot(xi, A_true.T[j], "-", c=colors[j], lw=2)
    #ax.plot(xi, A_est.T[j], "-", c=colors[j], lw=1, alpha=0.5)

    ax.plot(xi, L_percentiles[1, :, j], "-", c=colors[j], lw=1)
    ax.fill_between(xi, L_percentiles[0, :, j], 
                        L_percentiles[-1, :, j],
                    alpha=0.5, facecolor=colors[j], zorder=-1)


# Means.
fig, axes = plt.subplots(J - 1, J - 1)
axes = np.atleast_2d(axes)
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        if i < j:
            ax.set_visible(False)
            continue

        else:
            ax.set_title(f"i = {i + 1}, j = {j}")

            for k in range(K):
                ax.scatter(np.atleast_1d(truth["xi"][j, k]),
                           np.atleast_1d(truth["xi"][i + 1, k]),
                           facecolor="#000000", zorder=10, s=50)

                ax.scatter(p_samples["xi"][:, k, j],
                           p_samples["xi"][:, k, i + 1],
                           facecolor="tab:blue", zorder=-1, alpha=0.3, s=1)

                # Show covariance matrix?

# Omega_corr / Omega_diag





raise a

# plot trace of eta.
fig = stan.plots.traceplot(p_samples, ("Omega_eta", ))

L_samples = p_samples.extract(("L", ))["L"]
L_percentiles = np.percentile(L_samples, [5, 50, 95], axis=0)

# Make comparisons
A_true, A_opt = (truth["A"], p_opt["L"])

R = utils.rotation_matrix(A_true, A_opt)

A_est = A_opt @ R


fig, ax = plt.subplots()
for j in range(J):
    ax.plot(xi, A_true.T[j], "-", c=colors[j], lw=2)
    #ax.plot(xi, A_est.T[j], "-", c=colors[j], lw=1, alpha=0.5)

    ax.plot(xi, np.diag(R)[j] * L_percentiles[1, :, j], "-", c=colors[j], lw=1)
    ax.fill_between(xi, np.diag(R)[j] * L_percentiles[0, :, j], 
        np.diag(R)[j] * L_percentiles[-1, :, j],
                    alpha=0.5, facecolor=colors[j], zorder=-1)

# Plot psi wrt samples.



# Check that Omega has the Omega_diag implemented correctly (as it is in python)

# Plot latent space draws.



"""

data {
    int<lower=1> N; // number of data points
    int<lower=1> D; // dimensionality of the data 
    int<lower=1> J; // number of latent factors
    vector[D] y[N]; // the data
}

transformed data {
    vector[D] mu;   // mean of the data in each dimension
    int<lower=1> M; // number of non-zero loadings

    M = J * (D - J) + choose(J, 2); 
    mu = rep_vector(0.0, D);
}

parameters {
    vector[M] beta_lower_triangular;
    vector<lower=0>[J] beta_diag;
    vector<lower=0>[D] psi;
    real<lower=0> sigma_L;
}

"""