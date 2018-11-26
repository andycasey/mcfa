
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import stan_utils as stan
from mcfa import (mcfa, mpl_utils, utils)
from scipy import stats

matplotlib.style.use(mpl_utils.mpl_style)

seed = 42

N = 100 # number of data points
D = 15   # data dimension
J = 3    # number of latent factors
K = 1    # number of components
lkj_eta = 2.0


data_kwds = dict(n_samples=N, n_features=D, n_latent_factors=J,
                 n_components=K, lkj_eta=lkj_eta, psi_scale=1,
                 latent_scale=1, random_seed=seed)

op_kwds = dict(init_alpha=1, tol_obj=1e-16, tol_rel_grad=1e-16, 
               tol_rel_obj=1e-16, seed=seed, iter=100000)

sampling_kwds = dict(chains=2, iter=2000)


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

    A = np.zeros((n_features, n_latent_factors), dtype=float)
    A[np.tril_indices(n_features, -1, n_latent_factors)] = beta_lower_triangular
    A[np.diag_indices(n_latent_factors)] = np.sort(beta_diag)[::-1]


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
    for i in range(n_components):

        rho = LKJCorrelationMatrix(J, lkj_eta)
        omega_diag = np.sqrt(np.abs(rng.normal(0, 1, size=(1, J))))
        omega[:, :, i] = (rho @ rho.T) * (omega_diag.T @ omega_diag)


    scores = np.empty((n_samples, n_latent_factors))
    for i in range(n_components):
        match = (R == i)
        scores[match] = rng.multivariate_normal(xi.T[i], omega.T[i], 
                                                size=sum(match))

    #psi = rng.gamma(1, scale=psi_scale, size=n_features)
    psi = np.abs(rng.normal(0, psi_scale, n_features))

    noise = np.sqrt(psi) * rng.randn(n_samples, n_features)

    X = scores @ A.T + noise

    truth = dict(A=A, pi=pi, xi=xi, omega=omega, psi=psi,
                 noise=noise, R=R, scores=scores,
                 beta_diag=beta_diag, sigma_L=sigma_L,
                 beta_lower_triangular=beta_lower_triangular)

    return (X, truth)

y, truth = generate_data(**data_kwds)



model = stan.load_model("mcfa2.stan")

data_dict = dict(N=N, D=D, J=J, K=K, y=y)

p_opt = model.optimizing(data=data_dict, **op_kwds)

p_opt["lambda"] = np.atleast_1d(p_opt["lambda"])

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

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots()
nums = np.nanmedian(L_true / L_opt, axis=0)
for j in range(J):
    ax.plot(xi, L_true.T[j], "-", c=colors[j], lw=2)
    ax.plot(xi, nums[j] * L_opt.T[j], ":", c=colors[j], lw=1)


print(f"Nums: {nums}")

# Compare means.
fig, ax = plt.subplots()
ax.scatter(truth["xi"].flatten(), p_opt["xi"].flatten())

raise a

p_samples = model.sampling(**stan.sampling_kwds(data=data_dict, init=p_opt, 
                                                **sampling_kwds))

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

    ax.plot(xi, L_percentiles[1, :, j], "-", c=colors[j], lw=1)
    ax.fill_between(xi, L_percentiles[0, :, j], L_percentiles[-1, :, j],
                    alpha=0.5, facecolor=colors[j], zorder=-1)




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