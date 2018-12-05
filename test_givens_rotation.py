

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as op
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






seed = 123

N = 100 # number of data points
D = 15   # data dimension
J = 3    # number of latent factors0
K = 10    # number of components
#lkj_eta = 2.0


data_kwds = dict(n_samples=N, n_features=D, n_latent_factors=J,
                 n_components=K, psi_scale=1,
                 latent_scale=1, random_seed=seed)

op_kwds = dict()

y, truth = generate_data(**data_kwds)


mcfa_model = mcfa.MCFA(n_components=K, n_latent_factors=J, 
                       init_factors="noise", init_components="random",
                       random_seed=seed)
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
A_est = mcfa_model.theta_[1]

'''
# Rotate the matrix.

cost = lambda B, *p: (B @ utils.generalized_rotation_matrix(*p)).flatten()



def find_rotation_matrix(A, B, init=None, n_inits=10, **kwargs):
    """
    Find the Euler angles that produce the rotation matrix such that

    .. math:

        A \approx B @ R
    """

    kwds = dict(maxfev=10000)
    kwds.update(kwargs)

    A, B = (np.atleast_2d(A), np.atleast_2d(B))

    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")

    L1 = lambda R: np.sum(np.abs(A - B @ R))

    def objective_function(angles):
        return L1(utils.generalized_rotation_matrix(*angles))

    if init is None:
        inits = np.random.uniform(0, 2 * np.pi, size=(n_inits, J))
        inits[0] = 0

    else:
        inits = np.atleast_2d(inits).reshape((-1, J))

    best_R = None
    best_cost = None

    for i, init in enumerate(inits):

        p_opt = op.minimize(objective_function, init, method="BFGS")

        R = utils.generalized_rotation_matrix(*p_opt.x)
        cost = L1(R)

        # Try flipping axes?
        flips = np.ones(J)
        for j in range(J):

            R_flip = R.copy()
            R_flip[:, j] *= -1

            if L1(R_flip) < cost:
                flips[j] = -1

        R *= flips

        if best_cost is None or cost(R) < best_cost:
            best_R = R

    return best_R



def gram_schmidt_orthogonalization(A, B):

    v1, v2 = (np.atleast_2d(B).copy(), np.atleast_2d(A).copy())

    n1 = v1 / np.linalg.norm(v1)
    v2 = v2 - (n1 @ v2.T) * n1
    n2 = v2 / np.linalg.norm(v2)

    alpha = np.pi/2.0

    R = np.eye(A.size) \
      + (np.outer(n2, n1) - np.outer(n1, n2)) * np.sin(alpha) \
      + (np.outer(n1, n1) - np.outer(n2, n2)) * (np.cos(alpha) - 1)

    scalar = np.linalg.norm(v2)/np.linalg.norm(v1)
    return R * scalar



def G(D, i, theta):

    c, s = (np.cos(theta), np.sin(theta))

    R = np.eye(D)
    R[i, i] = c
    R[i + 1, i + 1] = c
    R[i, i + 1] = +s
    R[i + 1, i] = -s

    return R
'''


def givens_rotation_matrix(*angles):

    angles = np.atleast_1d(angles)
    D = len(angles)
    R = np.ones((D, D, D))

    for i, theta in enumerate(angles):

        s = np.sin(theta)

        R[i] = np.eye(D)
        R[i, -i, -i] = R[i, -i + 1, -i + 1] = np.cos(theta)
        R[i, -i, -i + 1] = +s
        R[i, -i + 1, -i] = -s

    R = np.linalg.multi_dot(R)
    assert np.allclose(R @ R.T, np.eye(R.shape[0]))
    return R



cost = lambda B, *p: (B @ givens_rotation_matrix(*p)).flatten()


def find_rotation_matrix(A, B, init=None, n_inits=25, **kwargs):
    """
    Find the Euler angles that produce the rotation matrix such that

    .. math:

        A \approx B @ R
    """

    kwds = dict(maxfev=10000)
    kwds.update(kwargs)

    A, B = (np.atleast_2d(A), np.atleast_2d(B))

    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")

    L = lambda R: np.sum(np.abs(A - B @ R))

    def objective_function(angles):
        return L(givens_rotation_matrix(*angles))

    if init is None:
        inits = np.random.uniform(0, 2 * np.pi, size=(n_inits, J))
        inits[0] = 0

    else:
        inits = np.atleast_2d(inits).reshape((-1, J))

    best_R = None
    best_cost = None

    for i, init in enumerate(inits):

        p_opt = op.minimize(objective_function, init, method="BFGS")
        #p_opt = op.basinhopping(objective_function, init, niter=10)
        #p_opt = op.minimize(objective_function, init, method="Nelder-Mead")

        R = givens_rotation_matrix(*p_opt.x)
        cost = L(R)

        # Try flipping axes
        flips = np.ones(J)
        for j in range(J):

            R_flip = R.copy()
            R_flip[:, j] *= -1

            if L(R_flip) < cost:
                flips[j] = -1

        R *= flips
        cost = L(R)

        print(p_opt.x, cost)

        if best_cost is None or cost < best_cost:
            best_R = R
            best_cost = cost


    return best_R


"""
angles = np.random.uniform(0, 2 * np.pi, 3)


R1 = utils.generalized_rotation_matrix(*angles)
R2 = givens_rotation_matrix(angles * np.array([1, -1, 1]))


"""

"""
# input vectors
v1 = np.array([[1,1,1,1,1,1]])
v2 = np.array([[2,3,4,5,6,7]])


# Gram-Schmidt orthogonalization
n1 = v1 / np.linalg.norm(v1)
v2 = v2 - np.dot(n1,v2) * n1
n2 = v2 / np.linalg.norm(v2)

# rotation by pi/2
a = np.pi/2

I = np.identity(n1.size)

R1 = I + ( np.outer(n2,n1) - np.outer(n1,n2) ) * np.sin(a) + ( np.outer(n1,n1) + np.outer(n2,n2) ) * (np.cos(a)-1)
R1 *= np.linalg.norm(v2)/np.linalg.norm(v1)

# check result
print( np.matmul( R1, v1 ) )
print( v2 )


raise a

Qa, Rb = np.linalg.qr(A_est)
Qb, Rb = np.linalg.qr(A_true)

v2_ = A_true - Qa * A_true * Qa

n2_ = v2_ / np.linalg.norm(v2_)




#Q == -n1


R = find_rotation_matrix(A_true, A_est)

#R = utils.generalized_rotation_matrix(*angles)
#R1 = np.copy(R)
#R1[:, 0] *= -1

A_est_rotated = A_est @ R
A_est_rotated1 = A_est @ R


R2 = gram_schmidt_orthogonalization(A_true.T[0], A_est.T[0])


raise a
"""

R = find_rotation_matrix(A_true, A_est)

A_est_rotated = A_est @ R


from matplotlib.ticker import MaxNLocator

xi = np.arange(D)
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

for j in range(J):
    for ax in axes:
        ax.plot(xi, A_true.T[j], "-", c=colors[j], lw=2)
        ax.set_xlabel(r"$\textrm{dimension}$")
        ax.xaxis.set_major_locator(MaxNLocator(10))
        ax.yaxis.set_major_locator(MaxNLocator(3))


    axes[0].plot(xi, A_est.T[j], "-", c=colors[j], lw=1)
    axes[1].plot(xi, A_est_rotated.T[j], "-", c=colors[j], lw=1)
    
    axes[0].set_title(r"$\textrm{without rotation}$")
    axes[1].set_title(r"$\textrm{with rotation}$")

    axes[0].set_ylabel(r"$\mathbf{L}$")
    axes[1].set_yticks([])

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

raise a


# Try to rotate closest to a random set of load factors.

R = find_rotation_matrix(A_true, A_est)

#R = utils.generalized_rotation_matrix(*angles)
#R1 = np.copy(R)
#R1[:, 0] *= -1

A_est_rotated = A_est @ R
A_est_rotated1 = A_est @ R

A_random = np.random.uniform(-0.5, 0.5, size=A_true.shape)

AL = np.linalg.cholesky(A_random.T @ A_random)
A_random = A_random @ np.linalg.solve(AL, np.eye(J))



R1 = find_rotation_matrix(A_random, A_est)

A_est_rotated_to_random = A_est @ R1


from matplotlib.ticker import MaxNLocator

xi = np.arange(D)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for j in range(J):
    for ax in axes[:2]:
        ax.plot(xi, A_true.T[j], "-", c=colors[j], lw=2)
        ax.set_xlabel(r"$\textrm{dimension}$")
        ax.xaxis.set_major_locator(MaxNLocator(10))
        ax.yaxis.set_major_locator(MaxNLocator(3))

    axes[2].plot(xi, A_random.T[j], "-", c=colors[j], lw=2)

    axes[0].plot(xi, A_est.T[j], "-", c=colors[j], lw=1)
    axes[1].plot(xi, A_est_rotated.T[j], "-", c=colors[j], lw=1)
    axes[2].plot(xi, A_est_rotated_to_random.T[j], "-", c=colors[j], lw=1)

    axes[0].set_title(r"$\textrm{without rotation}$")
    axes[1].set_title(r"$\textrm{with rotation}$")

    axes[0].set_ylabel(r"$\mathbf{L}$")
    axes[1].set_yticks([])


fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
