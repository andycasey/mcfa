

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from time import time

import stan_utils as stan
from mcfa import (mcfa, mpl_utils, utils)
from scipy import stats



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
                  psi_scale=1, latent_scale=1, random_seed=0):

    rng = np.random.RandomState(random_seed)

    # Generate orthonormal factors.
    LSigma = np.abs(rng.normal(0, latent_scale))
    choose = lambda x, y: int(np.math.factorial(x) \
                        / (np.math.factorial(y) * np.math.factorial(x - y)))

    M = n_latent_factors * (n_features - n_latent_factors) \
      + choose(n_latent_factors, 2)

    BetaLowerTriangular = rng.normal(0, LSigma, size=M)
    BetaDiagonal = np.abs(rng.normal(0, LSigma, size=n_latent_factors))
    BetaDiagonal = np.sort(BetaDiagonal)

    A = np.zeros((n_features, n_latent_factors), dtype=float)
    A[np.tril_indices(n_features, -1, n_latent_factors)] = BetaLowerTriangular
    A[np.diag_indices(n_latent_factors)] = BetaDiagonal[::-1]

    # Make A.T @ A = I
    AL = np.linalg.cholesky(A.T @ A)
    A = A @ np.linalg.solve(AL, np.eye(n_latent_factors))

    # latent variables
    pvals = np.ones(n_components) / n_components
    R = np.argmax(rng.multinomial(1, pvals, size=n_samples), axis=1)
    pi = np.array([np.sum(R == i) for i in range(n_components)])/n_samples

    # Rank-order the weights.
    pi = np.sort(pi)

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
    OmegaCorr = np.zeros_like(omega)
    OmegaDiag = np.zeros((n_components, n_latent_factors))
    for i in range(n_components):

        rho = LKJCorrelationMatrix(J, LSigma)
        OmegaCorr[:, :, i] = rho
        OmegaDiag[i] = rng.normal(0, 1, size=(1, J))
        omega[:, :, i] = (rho @ rho.T) * (OmegaDiag[i].T @ OmegaDiag[i])


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
                 BetaDiagonal=BetaDiagonal, LSigma=LSigma,
                 BetaLowerTriangular=BetaLowerTriangular,
                 OmegaCorr=OmegaCorr, OmegaDiag=OmegaDiag)

    return (X, truth)





seed = 100

N = 10000 # number of data points
D = 10   # data dimension
J = 3    # number of latent factors
K = 1    # number of components
#lkj_eta = 2.0


data_kwds = dict(n_samples=N, n_features=D, n_latent_factors=J,
                 n_components=K, psi_scale=1,
                 latent_scale=1, random_seed=seed)

op_kwds = dict()

#op_kwds.update(strict_op_kwds)

sampling_kwds = dict(chains=2, iter=2000)




y, truth = generate_data(**data_kwds)


mcfa_model = mcfa.MCFA(n_components=K, n_latent_factors=J, init_factors="random",
                  random_seed=seed, tol=1e-10)
mcfa_model.fit(y)


print("Log-likelihood: {:.0f}".format(mcfa_model.log_likelihood_))

fig, ax = plt.subplots()
ax.plot(truth["psi"], "-", c="#000000", lw=2, zorder=10)
ax.plot(mcfa_model.theta_[-1], "-", c="tab:blue")
#plot_samples(ax, p_samples, "psi")
ax.set_xlabel(r"$\textrm{Dimension } D$")
ax.set_ylabel(r"$\psi$")



colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

A_true = truth["A"]

R = utils.rotation_matrix(mcfa_model.theta_[1], A_true)

A_est = mcfa_model.theta_[1] @ R

xi = np.arange(D)
fig, ax = plt.subplots()
for j in range(J):
    ax.plot(xi, A_true.T[j], "-", c=colors[j], lw=2)
    #ax.plot(xi, A_est.T[j], "-", c=colors[j], lw=1, alpha=0.5)

    ax.plot(xi, A_est.T[j], "-", c=colors[j], lw=1)
    #ax.fill_between(xi, L_percentiles[0, :, j], 
    #                    L_percentiles[-1, :, j],
    #                alpha=0.5, facecolor=colors[j], zorder=-1)





"""
R = utils.euclidean_2d_rotation_matrix(45)

A_rotated = A_true @ R
y_approx = mcfa_model.sample(100)

from scipy import linalg

def M(axis, theta):
    return linalg.expm(np.cross(np.eye(len(axis)), axis/linalg.norm(axis) * theta))

M0 = M(A_true, 45)

M0 @ A_rotated
"""

angles = np.random.uniform(-np.pi, np.pi, size=3)

R = utils.generalized_rotation_matrix(*angles)

A = A_true
B = A_true @ R

import scipy.optimize as op


def find_euler_angles(A, B):

    p0 = np.zeros(3)

    curve = lambda x, *p: (A @ utils.generalized_rotation_matrix(*p)).flatten()

    return op.curve_fit(curve, None, B.flatten(), p0)

