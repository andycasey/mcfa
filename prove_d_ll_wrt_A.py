
import sys
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import (linalg, optimize as op)
from matplotlib.ticker import MaxNLocator
from time import time

sys.path.insert(0, "../../")

from mcfa import (mcfa, mpl_utils, utils)

matplotlib.style.use(mpl_utils.mpl_style)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

np.random.seed(0)

n_features = 10
n_components = 1
n_latent_factors = 3
n_samples = 100

omega_scale = 1
noise_scale = 1
random_seed = 1

data_kwds = dict(n_features=n_features,
                 n_components=n_components,
                 n_latent_factors=n_latent_factors,
                 n_samples=n_samples,
                 omega_scale=omega_scale,
                 noise_scale=noise_scale,
                 random_seed=random_seed)


Y, truth = utils.generate_data(**data_kwds)



model = mcfa.MCFA(n_components=n_components, n_latent_factors=n_latent_factors)

pi, A, xi, omega, psi = model._initial_parameters(Y)


N, D = Y.shape
J, K = (model.n_latent_factors, model.n_components)


def f(A_flat, Y, pi, xi, omega, psi):

    N, D = Y.shape
    ll, tau = model.expectation(Y, pi, A_flat.reshape((D, -1)), xi, omega, psi)
    return np.sum(ll)



def g(A_flat, Y, pi, xi, omega, psi):

    N, D = Y.shape
    J, _, K = omega.shape

    A = A_flat.reshape((D, J))

    I_psi = np.eye(D) * psi

    O = omega[:, :, 0]
    M = np.atleast_2d(xi.T[0]).T

    v = np.ones((N, D, J))

    T0_inv = (I_psi + A @ O @ A.T)
    T0 = np.linalg.solve(T0_inv, np.eye(D))

    T1 = A @ M
    for i in range(N):

        T2 = T0 @ (Y[[i]].T - T1)
        v[i, :] = T2 @ M.T + T2 @ (Y[[i]] + (-T1).T) @ T0 @ A @ O - T0 @ A @ O

    return np.sum(v, axis=0).flatten()





args = (Y, pi, xi, omega, psi)

At = truth["A"].flatten()

f(At, *args)
g(At, *args)




A_ = truth["A"].flatten()
approx = op.approx_fprime(A_, f, 1e-10, *args)
actual = g(A_, *args)


diff = op.check_grad(f, g, A_, *args)
print(f"Difference {diff}")



import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(approx.flatten(), np.abs((actual - approx).flatten()))
ax.semilogy()



